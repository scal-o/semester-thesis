"""
OD-based node initialization via edge embedding aggregation.

Aggregates virtual edge embeddings (created by edge processors) into node features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from ml_static.utils.validation import validate_edge_attribute, validate_node_attribute


class ODNodeInitializer(nn.Module):
    """
    Aggregates virtual edge embeddings into node features.

    Assumes edge embeddings have already been created by an EdgeEmbeddingProcessor
    and stored in data[edge_type].edge_embedding.

    Aggregation process:
    1. Extract edge_embedding from virtual edges
    2. Scatter aggregate by source/target nodes
    3. Concatenate [H_out, H_in, coords] → node features [num_nodes, 2*edge_dim + coord_dim]
    """

    def forward(
        self,
        x: torch.Tensor | None,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "virtual", "nodes"),
    ) -> HeteroData:
        """Aggregate virtual edge embeddings into node features.

        Args:
            x: MUST be None. Node features will be created from edge embeddings.
            data: Heterogeneous graph data with edge_embedding attribute.
            edge_type: Edge type for virtual edges (default: ("nodes", "virtual", "nodes")).

        Returns:
            Updated HeteroData with node features in data["nodes"].x [num_nodes, 2*edge_dim + coord_dim]
            where edge_dim is the dimension of the edge embeddings and coord_dim is the coordinate dimension.
        """
        if x is not None:
            raise ValueError(
                "Input `x` must be None. Node features are created from edge embeddings."
            )

        # validate and extract edge_index and edge_embedding
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        validate_edge_attribute(data, edge_type, "edge_embedding", expected_ndim=2)

        edge_index = data[edge_type].edge_index
        edge_emb = data[edge_type].edge_embedding

        # validate and extract node coordinates (determines total number of nodes)
        validate_node_attribute(data, "nodes", "coords", expected_ndim=2)
        coords = data["nodes"].coords
        num_nodes = coords.size(0)

        # aggregate embeddings to nodes
        # nodes not in edge_index will get zero embeddings (no OD activity)
        H_out = scatter(
            edge_emb, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum"
        )  # [num_nodes, edge_dim]
        H_in = scatter(
            edge_emb, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum"
        )  # [num_nodes, edge_dim]

        # concatenate outgoing, incoming embeddings, and coordinates
        # note: nodes without virtual connections get zero embeddings from scatter, but keep their coords
        H_node = torch.cat([H_out, H_in, coords], dim=1)  # [num_nodes, 2*edge_dim + coord_dim]

        # write node features to data
        data["nodes"].x = H_node

        return data
