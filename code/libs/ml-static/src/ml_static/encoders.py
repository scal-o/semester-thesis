"""Shared encoder modules for real and virtual edges."""

import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class BaseDependentAttentionLayer(nn.Module):
    """Common sparse attention building block for edge-aware encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            msg = "hidden_dim must be divisible by num_heads"
            raise ValueError(msg)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.lin_q = nn.Linear(input_dim, hidden_dim)
        self.lin_k = nn.Linear(input_dim, hidden_dim)
        self.lin_v = nn.Linear(input_dim, hidden_dim)

        self.scale = self.head_dim**-0.5

        self.lin_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def compute_edge_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        """Return attention modifiers for each edge and head.

        Args:
            x: Node features with shape ``[num_nodes, input_dim]``.
            edge_index: Edge connectivity in COO format ``[2, num_edges]``.
            **kwargs: Additional tensors required by concrete subclasses.

        Returns:
            Tensor of shape ``[num_edges, num_heads]`` containing edge weights.
        """

        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        """Apply dependent attention to sparse edges.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Edge connectivity ``[2, num_edges]``.
            **kwargs: Extra tensors consumed by subclasses (e.g., edge features).

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        num_nodes = x.size(0)
        origin, destination = edge_index

        q = self.lin_q(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.lin_k(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.lin_v(x).view(num_nodes, self.num_heads, self.head_dim)

        q_edges = q[origin]
        k_edges = k[destination]
        scores = (q_edges * k_edges).sum(dim=-1) * self.scale

        edge_weights = self.compute_edge_weights(x, edge_index, **kwargs)
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
        edge_index: torch.Tensor,
        *,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge features and broadcast across heads.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Edge connectivity ``[2, num_edges]`` (unused).
            edge_features: Edge attributes ``[num_edges, edge_dim]``.

        Returns:
            Tensor with shape ``[num_edges, num_heads]`` containing weights.
        """

        if edge_features.dim() != 2:
            msg = "edge_features must be rank 2"
            raise ValueError(msg)

        weights = edge_features.sum(dim=1, keepdim=True)
        return weights.repeat(1, self.num_heads)


class VirtualAdaptiveWeightLayer(nn.Module):
    """Learns adaptive attention weights for virtual OD edges."""

    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        self.input_dim = input_dim * 2
        self.num_heads = num_heads
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
        self.adaptive_layer = VirtualAdaptiveWeightLayer(input_dim, num_heads)

    def compute_edge_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor, **_: torch.Tensor
    ) -> torch.Tensor:
        """Use learned adaptive weights for each virtual edge.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Virtual connectivity ``[2, num_edges]``.
            **_: Ignored keyword arguments.

        Returns:
            Adaptive weights ``[num_edges, num_heads]``.
        """

        return self.adaptive_layer(x, edge_index)


class REncoder(nn.Module):
    """Encoder for real network edges using sparse attention."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.encoder = RealDependentAttentionLayer(input_dim, hidden_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the real-edge encoder.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Real edge connectivity ``[2, num_edges]``.
            edge_features: Real edge attributes ``[num_edges, edge_dim]``.

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        return self.encoder(x, edge_index, edge_features=edge_features)


class VEncoder(nn.Module):
    """Encoder for virtual OD edges using sparse attention."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.encoder = VirtualDependentAttentionLayer(input_dim, hidden_dim, num_heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the virtual-edge encoder.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Virtual edge connectivity ``[2, num_edges]``.

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        return self.encoder(x, edge_index)
