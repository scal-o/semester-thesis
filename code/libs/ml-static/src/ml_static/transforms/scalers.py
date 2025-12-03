from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import ScalerTransformConfig, TargetTransformConfig
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


class TargetScaler(BaseScaler, BaseTransform):
    """
    Scaler for target values in heterogeneous graphs.
    Selects target from graph, scales it, and stores in data.y
    """

    def __init__(self, target: tuple, transform_type: str | None = None, **kwargs):
        """
        Args:
            target: Tuple (type, label) specifying target location.
            transform_type: Type of scaling ('log', 'norm', 'minmax', None).
            **kwargs: Additional parameters (e.g., factor=100).
        """
        BaseScaler.__init__(self, transform_type, **kwargs)
        self.target_type, self.target_label = target

    @classmethod
    def from_config(cls, config: TargetTransformConfig) -> TargetScaler:
        """
        Create scaler from target transform configuration.

        Args:
            config: Target transform configuration.

        Returns:
            TargetScaler instance.
        """
        return cls(
            target=config.target,
            transform_type=config.transform,
            **config.kwargs,
        )

    def _get_value(self, data: HeteroData) -> torch.Tensor:
        """Extract target from graph."""
        if not (self.target_type in data.node_types or self.target_type in data.edge_types):
            raise KeyError(f"Data type '{self.target_type}' not found.")

        value = data[self.target_type].get(self.target_label, None)
        if value is None:
            raise KeyError(f"Target '{self.target_label}' not found in '{self.target_type}'.")
        return value

    def _set_value(self, data: HeteroData, value: torch.Tensor) -> None:
        """Store scaled target in data.y"""
        data.y = value


class FeatureScaler(BaseScaler, BaseTransform):
    """
    Scaler for feature values in heterogeneous graphs.
    Scales a specific feature in-place.
    """

    def __init__(self, feature: tuple, transform_type: str | None = None, **kwargs):
        """
        Args:
            feature: Tuple (type, label) specifying feature location.
            transform_type: Type of scaling ('log', 'norm', 'minmax', None).
            **kwargs: Additional parameters (e.g., factor=100).
        """
        BaseScaler.__init__(self, transform_type, **kwargs)
        self.feature_type, self.feature_label = feature

    @classmethod
    def from_config(cls, config: ScalerTransformConfig) -> FeatureScaler:
        """
        Create scaler from feature transform configuration.

        Args:
            config: Scaler transform configuration.

        Returns:
            FeatureScaler instance.
        """
        return cls(
            feature=(config.type_spec, config.feature),
            transform_type=config.transform,
            **config.kwargs,
        )

    def _get_value(self, data: HeteroData) -> torch.Tensor:
        """Extract feature from graph."""
        if not (self.feature_type in data.node_types or self.feature_type in data.edge_types):
            raise KeyError(f"Data type '{self.feature_type}' not found.")

        value = data[self.feature_type].get(self.feature_label, None)
        if value is None:
            raise KeyError(f"Feature '{self.feature_label}' not found in '{self.feature_type}'.")
        return value

    def _set_value(self, data: HeteroData, value: torch.Tensor) -> None:
        """Store scaled feature back in original location."""
        data[self.feature_type][self.feature_label] = value
