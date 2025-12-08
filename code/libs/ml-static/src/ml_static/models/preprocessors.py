from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from ml_static.models.predictors import LinearLayerConfig
from ml_static.utils import validate_node_attribute

# =============================================================================
# Configuration for Edge Predictor
# =============================================================================


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration for edge predictor.

    Attributes:
        output_channels: Final output dimension.
        node_feature_dim: Input node feature dimension.
        edge_feature_dim: Edge feature dimension.
        layers: Tuple of hidden layer configurations.
    """

    input_channels: int
    output_channels: int
    layers: tuple[LinearLayerConfig, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse PredictorConfig from dictionary.

        Args:
            data: Dict with keys: output_channels, node_feature_dim,
                  edge_feature_dim, layers.

        Returns:
            PredictorConfig instance.

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


# =============================================================================
# Node preprocessor Implementation
# =============================================================================


class NodePreprocessor(nn.Module):
    """
    MLP-based node preprocessor that transforms (encodes) input node features
    into a desired output dimension.

    Supports configurable multi-layer architecture.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        """Initialize edge predictor with a list of linear layers.

        Args:
            layers: List of nn.Linear layers for the prediction stack.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> Self:
        """Build node preprocessor from typed config.

        Args:
            config: PreprocessorConfig instance.

        Returns:
            Configured NodePreprocessor instance.
        """
        config.validate()

        # Start with concatenated features: 2*node_dim + edge_dim
        input_dim = config.input_channels

        linear_layers = []

        for layer_config in config.layers:
            linear_layers.append(nn.Linear(input_dim, layer_config.hidden_channels))
            input_dim = layer_config.hidden_channels

        # Final output layer
        linear_layers.append(nn.Linear(input_dim, config.output_channels))

        return cls(linear_layers)

    def forward(
        self,
        x: torch.Tensor,
        data: HeteroData,
        node_type: str = "nodes",
    ) -> torch.Tensor:
        """
        Forward pass to predict edge values.

        Args:
            x: Node features tensor of shape [num_nodes, input_dim].
            data: Heterogeneous graph data containing edge information.
            node_type: Node type identifier in the HeteroData object.

        Returns:
            Processed node features of shape [num_nodes, output_dim].
        """
        # extract edge index and features
        validate_node_attribute(data, node_type, "x", expected_ndim=2)

        # extract node features
        x = data[node_type].x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)

        return x
