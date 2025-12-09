from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration for Linear Layers
# =============================================================================


@dataclass(frozen=True)
class LinearLayerConfig:
    """Configuration for a linear layer in MLP.

    Attributes:
        hidden_channels: Output dimension for this layer.
    """

    hidden_channels: int

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse LinearLayerConfig from dictionary.

        Args:
            data: Dict with key: hidden_channels.

        Returns:
            LinearLayerConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        return cls(
            hidden_channels=data["hidden_channels"],
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict.

        Args:
            data: Dictionary to validate.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"hidden_channels"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in linear layer config: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate layer configuration."""
        if self.hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {self.hidden_channels}")


# =============================================================================
# Configuration for Edge Predictor
# =============================================================================


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for general MLP.

    Attributes:
        input_channels: Input feature dimension.
        output_channels: Final output dimension.
        activation: Final layer activation function.
        layers: Tuple of hidden layer configurations.
    """

    input_channels: int
    output_channels: int
    activation: str | None
    layers: tuple[LinearLayerConfig, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse MLPConfig from dictionary.

        Args:
            data: Dict with keys: input_channels, output_channels, layers.

        Returns:
            MLPConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        # Parse layers
        layers = [LinearLayerConfig.from_dict(d) for d in data["layers"]]
        layers = tuple(layers)

        return cls(
            input_channels=data["input_channels"],
            output_channels=data["output_channels"],
            activation=data.get("activation", None),
            layers=layers,
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict.

        Args:
            data: Dictionary to validate.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"input_channels", "output_channels", "layers"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in predictor config: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate predictor configuration."""
        if self.input_channels < 1:
            raise ValueError(f"input_channels must be >= 1, got {self.input_channels}")
        if self.output_channels < 1:
            raise ValueError(f"output_channels must be >= 1, got {self.output_channels}")
        if not self.layers:
            raise ValueError("Predictor must contain at least one hidden layer")


class MLP(nn.Module):
    """
    MLP.

    Supports configurable multi-layer architecture.
    """

    VALID_ACTIVATIONS = {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        None: lambda x: x,
    }

    def __init__(self, layers: list[nn.Module], final_activation: str | None) -> None:
        """Initialize edge predictor with a list of linear layers.

        Args:
            layers: List of nn.Linear layers for the prediction stack.
            final_activation: Callable activation function for the final layer.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Invalid final activation '{final_activation}'. "
                f"Supported: {list(self.VALID_ACTIVATIONS.keys())}"
            )

        self.final_activation: Callable = self.VALID_ACTIVATIONS[final_activation]

    @classmethod
    def from_config(cls, config: MLPConfig) -> Self:
        """Build MLP from typed config.

        Args:
            config: MLPConfig instance.

        Returns:
            Configured MLP instance.
        """
        config.validate()

        # start with input channels
        input_dim = config.input_channels

        linear_layers = []

        for layer_config in config.layers:
            linear_layers.append(nn.Linear(input_dim, layer_config.hidden_channels))
            input_dim = layer_config.hidden_channels

        # Final output layer
        linear_layers.append(nn.Linear(input_dim, config.output_channels))

        final_activation = config.activation

        return cls(linear_layers, final_activation)

    def _tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic tensor input."""

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)

        x = self.final_activation(x)
        return x

    def forward(
        self,
        x: torch.Tensor | None,
        data: Any = None,
        type: Any = None,
    ) -> torch.Tensor:
        """
        Forward pass implementation for graphs.

        Subclasses should override this method.
        """
        if x is None:
            raise ValueError("Input `x` must be a tensor. This can be overridden in subclasses.")

        return self._tensor_forward(x)
