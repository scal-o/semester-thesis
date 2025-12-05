"""
Feature builders for graph construction.

Each builder is a function that takes raw data and assembles it into
the final graph structure. Different models can use different builders
to create the graph layout they need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch_geometric.transforms import BaseTransform

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import BuilderTransformConfig
    from ml_static.data import STADataset


# === Node Builders ===
def nodes_add_demand(data: HeteroData) -> HeteroData:
    """
    Add demand from raw data to nodes.
    """
    data["nodes"].demand = data["_raw"].demand
    return data


def nodes_add_coords(data: HeteroData) -> HeteroData:
    """
    Add coordinates from raw data to nodes.
    """
    data["nodes"].coords = data["_raw"].node_coords
    return data


def nodes_concat_demand_coords_scaled(data: HeteroData) -> HeteroData:
    """
    Concatenate nodes.demand and nodes.coords into nodes.x.

    Requires prior extraction via nodes_add_demand and nodes_add_coords.
    Use this when you need to apply scalers before concatenation.

    Args:
        data: HeteroData with nodes.demand and nodes.coords.

    Returns:
        HeteroData with nodes.x = [demand, coords].

    Raises:
        AttributeError: If nodes.demand or nodes.coords don't exist.
    """
    if not hasattr(data["nodes"], "demand"):
        raise AttributeError(
            "nodes.demand not found. Use 'nodes_add_demand' builder first, "
            "or use 'nodes_concat_demand_coords_raw' to concat directly from _raw."
        )
    if not hasattr(data["nodes"], "coords"):
        raise AttributeError(
            "nodes.coords not found. Use 'nodes_add_coords' builder first, "
            "or use 'nodes_concat_demand_coords_raw' to concat directly from _raw."
        )

    data["nodes"].x = torch.cat(
        [data["nodes"].demand, data["nodes"].coords],
        dim=-1,
    )
    return data


def nodes_concat_demand_coords_raw(data: HeteroData) -> HeteroData:
    """
    Concatenate demand and coords directly from _raw into nodes.x.

    Use this for simple pipelines without intermediate scaling.
    For scaled features, use nodes_concat_demand_coords_scaled instead.

    Args:
        data: HeteroData with _raw.demand and _raw.node_coords.

    Returns:
        HeteroData with nodes.x = [demand, coords].
    """
    data["nodes"].x = torch.cat(
        [data["_raw"].demand, data["_raw"].node_coords],
        dim=-1,
    )
    return data


# === Real Edges Builders ===
def real_edges_add_index(data: HeteroData) -> HeteroData:
    """
    Add edge index to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_index = data["_raw"].real_index
    return data


def real_edges_add_capacity(data: HeteroData) -> HeteroData:
    """
    Add capacity to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_capacity = data["_raw"].edge_capacity
    return data


def real_edges_add_free_flow_time(data: HeteroData) -> HeteroData:
    """
    Add free_flow_time to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_free_flow_time = data["_raw"].edge_free_flow_time
    return data


def real_edges_add_vcr(data: HeteroData) -> HeteroData:
    """
    Add volume-capacity ratio to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_vcr = data["_raw"].edge_vcr
    return data


def real_edges_add_flow(data: HeteroData) -> HeteroData:
    """
    Add flow to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_flow = data["_raw"].edge_flow
    return data


def real_edges_stack_capacity_free_flow(data: HeteroData) -> HeteroData:
    """
    Concatenate real edge features (capacity and free flow time).
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_features = torch.stack(
        [data["_raw"].edge_capacity, data["_raw"].edge_free_flow_time], dim=1
    )
    return data


# === Virtual Edges Builders ===
def virtual_edges_add_index(data: HeteroData) -> HeteroData:
    """
    Add edge index to virtual edges (OD pairs).

    Args:
        data: HeteroData with _raw attribute containing virtual_index.

    Returns:
        HeteroData with virtual edge index set.
    """
    edge_type = ("nodes", "virtual", "nodes")
    data[edge_type].edge_index = data["_raw"].virtual_index
    return data


# def virtual_edges_add_od_demand(data: HeteroData) -> HeteroData:
#     """
#     Add OD demand to virtual edges.

#     Args:
#         data: HeteroData with _raw attribute containing demand matrix.

#     Returns:
#         HeteroData with virtual edges containing flattened OD demand.
#     """
#     edge_type = ("nodes", "virtual", "nodes")
#     # Flatten demand matrix to match virtual edge structure
#     data[edge_type].edge_attr = data["_raw"].demand.flatten()
#     return data


# === Cleaning Builders (for cleanup after processing) ===


def clean_raw_data(data: HeteroData) -> HeteroData:
    """
    Remove _raw node type after all builders have used it.

    Args:
        data: HeteroData with _raw node type.

    Returns:
        HeteroData without _raw node type.
    """
    if "_raw" in data.node_types:
        del data["_raw"]
    return data


def nodes_clean_demand(data: HeteroData) -> HeteroData:
    """
    Remove demand from nodes after it's been used.
    """
    if hasattr(data["nodes"], "demand"):
        del data["nodes"].demand
    return data


def nodes_clean_coords(data: HeteroData) -> HeteroData:
    """
    Remove coords from nodes after they've been used.
    """
    if hasattr(data["nodes"], "coords"):
        del data["nodes"].coords
    return data


# Registry mapping builder names to functions
BUILDERS: dict[str, Callable[[HeteroData], HeteroData]] = {
    "nodes_add_demand": nodes_add_demand,
    "nodes_add_coords": nodes_add_coords,
    "nodes_concat_demand_coords_scaled": nodes_concat_demand_coords_scaled,
    "nodes_concat_demand_coords_direct": nodes_concat_demand_coords_raw,
    # "virtual_edges_add_od_demand": virtual_edges_add_od_demand,
    "real_edges_add_capacity": real_edges_add_capacity,
    "real_edges_add_free_flow_time": real_edges_add_free_flow_time,
    "real_edges_add_vcr": real_edges_add_vcr,
    "real_edges_add_flow": real_edges_add_flow,
    "real_edges_add_index": real_edges_add_index,
    "real_edges_stack_capacity_free_flow": real_edges_stack_capacity_free_flow,
    "virtual_edges_add_index": virtual_edges_add_index,
    "nodes_clean_demand": nodes_clean_demand,
    "nodes_clean_coords": nodes_clean_coords,
    "clean_raw_data": clean_raw_data,
}


def get_builder(name: str) -> Callable[[HeteroData], HeteroData]:
    """
    Get a builder function by name.

    Args:
        name: Name of the builder.

    Returns:
        Builder function.

    Raises:
        KeyError: If builder name is not found.
    """
    if name not in BUILDERS:
        available = ", ".join(f"'{k}'" for k in BUILDERS.keys())
        raise KeyError(f"Unknown builder '{name}'. Available builders: {available}")
    return BUILDERS[name]


class BuilderTransform(BaseTransform):
    """
    Wraps a builder function as a transform.

    Builders are stateless and don't require fitting.
    """

    def __init__(self, builder_name: str):
        """
        Args:
            builder_name: Name of the builder function to use.
        """
        self.builder_name = builder_name
        self.builder = get_builder(builder_name)

    @classmethod
    def from_config(cls, config: BuilderTransformConfig) -> BuilderTransform:
        """
        Create builder transform from configuration.

        Args:
            config: Builder transform configuration.

        Returns:
            BuilderTransform instance.
        """
        return cls(builder_name=config.builder)

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Apply builder function to data.

        Args:
            data: Input HeteroData object.

        Returns:
            Transformed HeteroData.
        """
        return self.builder(data)

    def fit_dataset(self, dataset: STADataset) -> None:
        """
        Builders don't need fitting (no-op).

        Args:
            dataset: Dataset (unused).
        """
        pass
