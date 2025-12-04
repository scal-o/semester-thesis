from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from ml_static.utils.validation import validate_edge_attribute

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
class PredictorConfig:
    """Configuration for edge predictor.

    Attributes:
        output_channels: Final output dimension.
        node_feature_dim: Input node feature dimension.
        edge_feature_dim: Edge feature dimension.
        layers: Tuple of hidden layer configurations.
    """

    output_channels: int
    node_feature_dim: int
    edge_feature_dim: int
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
            output_channels=data["output_channels"],
            node_feature_dim=data["node_feature_dim"],
            edge_feature_dim=data["edge_feature_dim"],
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
        required = {"output_channels", "node_feature_dim", "edge_feature_dim", "layers"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in predictor config: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate predictor configuration."""
        if self.output_channels < 1:
            raise ValueError(f"output_channels must be >= 1, got {self.output_channels}")
        if not self.layers:
            raise ValueError("Predictor must contain at least one hidden layer")


# =============================================================================
# Edge Predictor Implementation
# =============================================================================


class EdgePredictor(nn.Module):
    """
    MLP-based edge predictor that combines origin and destination node features
    with edge features to predict a target value for each edge.

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
    def from_config(cls, config: PredictorConfig) -> "EdgePredictor":
        """Build edge predictor from typed config.

        Args:
            config: PredictorConfig instance.

        Returns:
            Configured EdgePredictor instance.
        """
        config.validate()

        # Start with concatenated features: 2*node_dim + edge_dim
        input_dim = config.node_feature_dim * 2 + config.edge_feature_dim

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
        edge_type: tuple | str = ("nodes", "real", "nodes"),
    ) -> torch.Tensor:
        """
        Forward pass to predict edge values.

        Args:
            x: Node features tensor of shape [num_nodes, input_dim].
            data: Heterogeneous graph data containing edge information.
            edge_type: Edge type identifier in the HeteroData object.

        Returns:
            Predicted edge values tensor of shape [num_edges].
        """
        # extract edge index and features
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        edge_index = data[edge_type].edge_index
        validate_edge_attribute(data, edge_type, "edge_features", expected_ndim=2)
        edge_features = data[edge_type].edge_features
        origin, destination = edge_index

        # Concatenate origin and destination node features with edge features
        z = torch.cat([x[origin], x[destination], edge_features], dim=1)

        # Pass through layer stack with activations
        for i, layer in enumerate(self.layers):
            z = layer(z)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                z = F.leaky_relu(z)

        return z.view(-1)
