from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

from ml_static.utils.validation import validate_edge_attribute

# =============================================================================
# Configuration for Attention Layers
# =============================================================================


@dataclass(frozen=True)
class AttentionLayerConfig:
    """Configuration for an attention layer.

    Attributes:
        type: Layer class name (e.g., 'RealDependentAttentionLayer').
        hidden_channels: Hidden dimension for this layer.
        num_heads: Number of attention heads.
    """

    type: str
    hidden_channels: int
    num_heads: int

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse AttentionLayerConfig from dictionary.

        Args:
            data: Dict with keys: type, hidden_channels, num_heads.

        Returns:
            AttentionLayerConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        return cls(
            type=data["type"],
            hidden_channels=data["hidden_channels"],
            num_heads=data["num_heads"],
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict.

        Args:
            data: Dictionary to validate.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"type", "hidden_channels", "num_heads"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in attention layer config: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate layer configuration."""
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )


# =============================================================================
# Attention Layer Implementations
# =============================================================================


class BaseDependentAttentionLayer(nn.Module):
    """
    Base node-feature focused AttentionLayer class.
    The attention mechanism uses key, query, and value projections computed
    on the graph node features.
    These projections are combined with edge-dependent weights to compute
    attention scores over sparse graph edges.

    The compute_edge_weights method is left unimplemented and should be
    overridden by subclasses.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            msg = "hidden_dim must be divisible by num_heads"
            raise ValueError(msg)

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads

        self.lin_q = nn.Linear(input_dim, hidden_dim)
        self.lin_k = nn.Linear(input_dim, hidden_dim)
        self.lin_v = nn.Linear(input_dim, hidden_dim)

        self.scale = self.head_dim**-0.5

        self.lin_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def compute_edge_weights(
        self, x: torch.Tensor, data: HeteroData, edge_type: tuple | str, **kwargs
    ) -> torch.Tensor:
        """Return attention modifiers for each edge and head.

        Args:
            x: Node features with shape ``[num_nodes, input_dim]``.
            edge_type: Edge type as a tuple or string.
            data: HeteroData object containing the graph data.
            **kwargs: Additional tensors required by concrete subclasses.

        Returns:
            Tensor of shape ``[num_edges, num_heads]`` containing edge weights.
        """

        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, data: HeteroData, edge_type: tuple | str, **kwargs
    ) -> torch.Tensor:
        """Apply dependent attention to sparse edges.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_type: Edge type as a tuple or string.
            data: HeteroData object containing the graph data.
            **kwargs: Extra tensors consumed by subclasses.

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        num_nodes = x.size(0)
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        edge_index = data[edge_type].edge_index
        origin, destination = edge_index

        q = self.lin_q(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.lin_k(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.lin_v(x).view(num_nodes, self.num_heads, self.head_dim)

        q_edges = q[origin]
        k_edges = k[destination]
        scores = (q_edges * k_edges).sum(dim=-1) * self.scale

        edge_weights = self.compute_edge_weights(x, data, edge_type, **kwargs)
        weighted_scores = scores * edge_weights

        attention_weights = softmax(weighted_scores, origin, num_nodes=num_nodes)

        v_edges = v[destination]
        weighted_values = attention_weights.unsqueeze(-1) * v_edges

        values = torch.zeros(
            num_nodes,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        values.index_add_(0, origin, weighted_values)

        values = values.reshape(num_nodes, self.hidden_dim)
        values = self.lin_out(values)
        values = self.layer_norm(values)

        return x + values


class RealDependentAttentionLayer(BaseDependentAttentionLayer):
    """Dependent attention variant that consumes real edge features."""

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "real", "nodes"),
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge features and broadcast across heads.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Edge connectivity ``[2, num_edges]`` (unused).
            edge_features: Edge attributes ``[num_edges, edge_dim]``.

        Returns:
            Tensor with shape ``[num_edges, num_heads]`` containing weights.
        """
        validate_edge_attribute(data, edge_type, "edge_features", expected_ndim=2)
        edge_features = data[edge_type].edge_features

        # compute weights by summing edge features and broadcasting
        weights = edge_features.sum(dim=1, keepdim=True)
        return weights.repeat(1, self.num_heads)


class VirtualAdaptiveWeights(nn.Module):
    """Learns adaptive attention weights for virtual OD edges."""

    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        self.input_dim: int = input_dim * 2
        self.num_heads: int = num_heads
        self.lin_v_edges = nn.Linear(self.input_dim, self.num_heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on origin/destination features.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Virtual edge connectivity ``[2, num_edges]``.

        Returns:
            Adaptive weights ``[num_edges, num_heads]`` for each head.
        """

        origin, destination = edge_index
        pair_features = torch.cat([x[origin], x[destination]], dim=1)
        return self.lin_v_edges(pair_features)


class VirtualDependentAttentionLayer(BaseDependentAttentionLayer):
    """Dependent attention variant that adapts weights per virtual edge."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__(input_dim, hidden_dim, num_heads)
        self.adaptive_layer = VirtualAdaptiveWeights(input_dim, num_heads)

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "virtual", "nodes"),
        **kwargs,
    ) -> torch.Tensor:
        """Use learned adaptive weights for each virtual edge.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Virtual connectivity ``[2, num_edges]``.
            **_: Ignored keyword arguments.

        Returns:
            Adaptive weights ``[num_edges, num_heads]``.
        """
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        edge_index = data[edge_type].edge_index

        return self.adaptive_layer(x, edge_index)
