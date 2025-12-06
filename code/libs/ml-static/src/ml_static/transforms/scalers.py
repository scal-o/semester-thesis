from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Self

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import ScalerTransformConfig
    from ml_static.data import STADataset


class BaseScaler(ABC):
    """
    Abstract base class for scalers.

    Subclasses must implement:
    - _get_value(data): Extract the value to scale from the data object
    - _set_value(data, value): Set the scaled value back into the data object
    """

    def __init__(self, transform_type: str | None = None, **kwargs):
        """
        Initialize scaler with transform type.

        Args:
            transform_type: Type of scaling ('log', 'norm', 'minmax', None).
            **kwargs: Additional parameters (e.g., factor=100 for minmax).

        Raises:
            ValueError: If transform_type is not valid.
        """
        VALID_TRANSFORMS = {
            "log": {
                "forward": self._log_transform,
                "inverse": self._log_inverse,
                "fit": self._empty,
            },
            "norm": {
                "forward": self._norm_transform,
                "inverse": self._norm_inverse,
                "fit": self._norm_fit,
            },
            "minmax": {
                "forward": self._minmax_transform,
                "inverse": self._minmax_inverse,
                "fit": self._minmax_fit,
            },
            None: {
                "forward": self._identity,
                "inverse": self._identity,
                "fit": self._empty,
            },
        }

        if transform_type is not None and transform_type not in VALID_TRANSFORMS:
            valid_list = ", ".join(f"'{t}'" for t in VALID_TRANSFORMS.keys() if t is not None)
            raise ValueError(
                f"Invalid transform type '{transform_type}'. Valid: {valid_list}, or None."
            )

        self.transform_type = transform_type
        self.kwargs = kwargs
        self.params: dict = {}

        cfg = VALID_TRANSFORMS[transform_type]
        self.transform_fn: Callable = cfg["forward"]
        self.inverse_fn: Callable = cfg["inverse"]
        self.fit_fn: Callable = cfg["fit"]

    def fit(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        """
        Compute statistics from data for fitting the scaler.

        Args:
            data: Tensor or list of tensors to compute statistics from.
        """
        if self.fit_fn == self._empty:
            return

        # concatenate if list
        if isinstance(data, list):
            data = [d for d in data]  # type checker fix
            tensor_data = torch.cat(data)
        else:
            tensor_data = data

        self.params = self.fit_fn(tensor_data)

    @abstractmethod
    def _get_value(self, data: HeteroData) -> torch.Tensor:
        """
        Extract value to scale from data object.

        Args:
            data: HeteroData object.

        Returns:
            Tensor to be scaled.
        """
        pass

    @abstractmethod
    def _set_value(self, data: HeteroData, value: torch.Tensor) -> None:
        """
        Set scaled value back into data object.

        Args:
            data: HeteroData object.
            value: Scaled tensor.
        """
        pass

    def fit_dataset(self, dataset: STADataset) -> None:
        """
        Fit the scaler on a dataset.

        Args:
            dataset: Dataset to fit the scaler on.
        """
        values = []
        for data in dataset:
            value = self._get_value(data)
            if value is not None:
                values.append(value)

        if values:
            self.fit(values)

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Apply scaling transformation to data.

        Args:
            data: Input HeteroData object.

        Returns:
            Transformed HeteroData with scaled values.
        """
        value = self._get_value(data)
        scaled = self.transform(value)
        self._set_value(data, scaled)
        return data

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward transformation to tensor.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        return self.transform_fn(x)

    def inverse_transform(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Apply inverse transformation to tensor or array.

        Args:
            x: Input tensor or numpy array.

        Returns:
            Inverse-transformed data in same format as input.
        """
        is_numpy = isinstance(x, np.ndarray)
        x = torch.as_tensor(x)
        x = self.inverse_fn(x)  # type: ignore[misc]
        return x.numpy() if is_numpy else x

    ## =======================
    ## === private methods ===
    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        """Identity transform (no-op)."""
        return x

    def _empty(self, x: torch.Tensor) -> dict:
        """Empty fit (no parameters)."""
        return {}

    # log transform methods
    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Log transform: log(1 + x)."""
        return torch.log1p(x)

    def _log_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse log transform: exp(x) - 1."""
        return torch.expm1(x)

    # norm transform methods
    def _norm_fit(self, x: torch.Tensor) -> dict:
        """Compute mean and std for normalization."""
        return {"for": "norm", "mean": x.mean().item(), "std": x.std().item()}

    def _norm_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normalization: (x - mean) / std."""
        if self.params.get("for", "") != "norm":
            raise ValueError("Scaler not fitted. Call fit() before transforming.")
        return (x - self.params["mean"]) / self.params["std"]

    def _norm_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse standard normalization: x * std + mean."""
        if self.params.get("for", "") != "norm":
            raise ValueError("Scaler not fitted. Call fit() before inverse transforming.")
        return x * self.params["std"] + self.params["mean"]

    # min-max transform methods
    def _minmax_fit(self, x: torch.Tensor) -> dict:
        """Compute min and max for min-max scaling."""
        return {"for": "minmax", "min": x.min().item(), "max": x.max().item()}

    def _minmax_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max scaling: (x - min) / (max - min) * factor."""
        if self.params.get("for", "") != "minmax":
            raise ValueError("Scaler not fitted. Call fit() before transforming.")
        factor = self.kwargs.get("factor", 100)
        return (x - self.params["min"]) / (self.params["max"] - self.params["min"]) * factor

    def _minmax_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse min-max scaling: x / factor * (max - min) + min."""
        if self.params.get("for", "") != "minmax":
            raise ValueError("Scaler not fitted. Call fit() before inverse transforming.")
        factor = self.kwargs.get("factor", 100)
        return x / factor * (self.params["max"] - self.params["min"]) + self.params["min"]


class ScalerTransform(BaseScaler, BaseTransform):
    """
    Unified scaler for both target values and features in heterogeneous graphs.

    Handles:
    - Target values (y): specify target="y"
    - Node features: specify target=(node_type, feature_name)
    - Edge features: specify target=((src, edge_type, dst), feature_name)

    Note: Target format is validated by ScalerTransformConfig before instantiation.
    """

    def __init__(self, target: str | tuple = "y", transform_type: str | None = None, **kwargs):
        """
        Args:
            target: Specifies what to scale:
                - "y": Target labels (data.y)
                - (type_spec, feature): Node/edge feature where type_spec is str or tuple
            transform_type: Type of scaling ('log', 'norm', 'minmax', None).
            **kwargs: Additional parameters (e.g., factor=100).
        """
        BaseScaler.__init__(self, transform_type, **kwargs)
        self.target = target
        self.is_label = target == "y"

    @classmethod
    def from_config(cls, config: ScalerTransformConfig) -> Self:
        """
        Create scaler from transform configuration.

        Args:
            config: Scaler transform configuration.

        Returns:
            Scaler instance.
        """
        return cls(
            target=config.scaler_target,
            transform_type=config.transform,
            **config.kwargs,
        )

    def _get_value(self, data: HeteroData) -> torch.Tensor:
        """Extract value from graph based on target specification."""
        if self.is_label:
            return data.y

        # feature access: target is (type_spec, feature_name)
        type_spec, feature_name = self.target
        return data[type_spec][feature_name]

    def _set_value(self, data: HeteroData, value: torch.Tensor) -> None:
        """Store scaled value back based on target specification."""
        if self.is_label:
            data.y = value
        else:
            type_spec, feature_name = self.target
            data[type_spec][feature_name] = value
