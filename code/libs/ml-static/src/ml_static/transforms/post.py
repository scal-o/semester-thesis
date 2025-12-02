from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform, Compose

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config
    from ml_static.data import STADataset


class TargetTransform(Compose):
    """
    Composable target transformation for Pytorch Geometric datasets.
    Combines SelectTarget and ScaleTarget transforms.
    """

    def __init__(self, selector: SelectTarget, scaler: ScaleTarget):
        """
        Args:
            selector: Target selection transform.
            scaler: Target scaling transform.
        """
        super().__init__([selector, scaler])

        self.selector = selector
        self.scaler = scaler

        # # register external dispatch methods from component transforms
        for transform in [self.selector, self.scaler]:
            if hasattr(transform, "_external_dispatch"):
                for method in transform._external_dispatch():
                    if not hasattr(self, method):
                        setattr(self, method, getattr(transform, method))
                    else:
                        raise AttributeError(
                            f"Method name conflict: '{transform}' wants to register "
                            f"method '{method}' that already exists in TargetTransform."
                        )

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create transform from configuration object.

        Args:
            config: Configuration object.

        Returns:
            TargetTransform instance.
        """
        # create selector and scaler from config
        selector = SelectTarget.from_config(config)
        scaler = ScaleTarget.from_config(config)

        return cls(selector=selector, scaler=scaler)

    def fit(self, dataset: STADataset) -> None:
        """
        Fit the scaler on the dataset.

        Args:
            dataset: The dataset to fit the scaler on.
        """
        if not hasattr(self.scaler, "fit"):
            return

        ds = copy.copy(dataset)

        # apply selector to each data object in dataset
        ds = [self.selector(ds.get(idx)).y for idx in range(len(ds))]
        self.scaler.fit(ds)


class SelectTarget(BaseTransform):
    """
    Simple target selection transform for Pytorch Geometric datasets.
    """

    def __init__(self, target: tuple):
        """
        Args:
            target: List specifying the target type and label
                (e.g. real links = ["nodes", "real", "nodes"]).
        """
        self.type, self.label = target

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create transform from configuration object.

        Args:
            config: Configuration object.

        Returns:
            SelectTarget instance.
        """
        # use Config API to get target
        target = config.get_target()

        return cls(target=target)

    ## ======================
    ## === public methods ===
    def forward(self, data: HeteroData) -> HeteroData:
        """
        Applies the transformation to the data object.

        Args:
            data: The input HeteroData object.

        Returns:
            The transformed HeteroData object with the target extracted.
        """
        if not (self.type in data.node_types or self.type in data.edge_types):
            raise KeyError(f"Data type '{self.type}' not found in data object.")

        target = data[self.type].get(self.label, None)
        if target is None:
            raise KeyError(f"Target label '{self.label}' not found in data type '{self.type}'.")

        data.y = target
        return data

    ## ========================
    ## === internal methods ===
    def _external_dispatch(self) -> list[str]:
        return []


class ScaleTarget(BaseTransform):
    """
    Target scaling transformation for Pytorch Geometric datasets.
    """

    def __init__(self, transform: str | None = None):
        """
        Args:
            transform: The transform to apply to each data object. If none, no scaling is applied.
        """

        # define valid transform types
        VALID_TRANSFORMS = {
            "log": {
                "normal": self._log_transform,
                "inverse": self._log_inverse_transform,
            },
            "norm": {
                "normal": self._norm_transform,
                "inverse": self._norm_inverse_transform,
            },
        }

        # validate transform type
        if transform is not None and transform not in VALID_TRANSFORMS:
            valid_list = ", ".join(f"'{t}'" for t in sorted(VALID_TRANSFORMS))
            raise ValueError(
                f"Invalid transform type '{transform}'. Valid options are: {valid_list}, or None."
            )

        self.transform_fn = VALID_TRANSFORMS.get(transform, {}).get("normal", self._identity)
        self.inverse_fn = VALID_TRANSFORMS.get(transform, {}).get("inverse", self._identity)

        # store scaler parameters
        self.params = {}

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create transform from configuration object.

        Args:
            config: Configuration object.

        Returns:
            VarTransform instance.
        """
        # use Config API to get target and transform
        transform_type = config.get_transform()

        return cls(transform=transform_type)

    ## ======================
    ## === public methods ===
    def forward(self, data: HeteroData) -> HeteroData:
        """
        Applies the transformation to the data object.

        Args:
            data: The input HeteroData object.

        Returns:
            The transformed HeteroData object with the target extracted.
        """
        # apply transformation
        data.y = self.transform_fn(data.y)

        return data

    def inverse_transform(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Applies the inverse transformation to a tensor or numpy array.

        Args:
            x: The data to inverse transform (e.g., predictions or targets).
               Can be either a torch.Tensor or numpy.ndarray.

        Returns:
            The inverse-transformed data in the same format as the input.
        """

        if self.transform_fn == self._identity:
            return self.transform_fn(x)

        # flag if input is numpy array
        is_numpy = isinstance(x, np.ndarray)

        # transform to tensor: no-op if already tensor
        x = torch.as_tensor(x)

        # apply inverse transformation
        x = self.inverse_fn(x)

        # transform back to numpy array if necessary
        x = x.numpy() if is_numpy else x

        return x

    def fit(self, dataset: STADataset | Dataset | list[torch.Tensor]) -> None:
        """
        Compute normalization statistics across the entire dataset.
        Only needed for "norm" transform.

        Args:
            dataset: The dataset to compute statistics from.
        """
        if self.transform_fn != self._norm_transform:
            return

        # if dataset is STADataset, convert to list of data objects
        if isinstance(dataset, Dataset):
            data_list = [dataset.get(idx).y for idx in range(len(dataset))]
        else:
            data_list = dataset

        # collect all target values across dataset
        # concatenate all targets and compute statistics
        all_targets = torch.cat(data_list)

        # set flag and store parameters
        self.params["for"] = "norm"
        self.params["mean"] = all_targets.mean().item()
        self.params["std"] = all_targets.std().item()

    ## ========================
    ## === internal methods ===
    def _external_dispatch(self) -> list[str]:
        return ["inverse_transform"]

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)

    def _log_inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    def _norm_transform(self, x: torch.Tensor) -> torch.Tensor:
        # ensure statistics are computed before transforming
        if self.params.get("for", None) != "norm":
            raise ValueError(
                "Normalization statistics not computed. Call fit() on the dataset first."
            )

        return (x - self.params["mean"]) / self.params["std"]

    def _norm_inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # ensure statistics are computed before transforming
        if self.params.get("for", None) != "norm":
            raise ValueError(
                "Normalization statistics not computed. Call fit() on the dataset first."
            )
        return x * self.params["std"] + self.params["mean"]
