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

from ml_static.utils import validate_edge_attribute, validate_node_attribute

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import BuilderTransformConfig
    from ml_static.data import STADataset


# === Builder Registry ===
BUILDERS_REGISTRY: dict[str, Callable[[HeteroData], HeteroData]] = {}


def register_builder(name: str):
    """
    A decorator to register a builder function in the builder registry.

    Args:
        name: Name to register the builder under.

    Returns:
        Decorator function.

    Raises:
        ValueError: If builder name is already registered.
    """

    def decorator(func):
        if name in BUILDERS_REGISTRY:
            raise ValueError(f"Builder '{name}' is already registered.")
        BUILDERS_REGISTRY[name] = func
        return func

    return decorator


# === Node Builders ===
@register_builder("nodes_add_demand")
def nodes_add_demand(data: HeteroData) -> HeteroData:
    """
    Add demand from raw data to nodes.
    """
    data["nodes"].demand = data["_raw"].demand
    return data


@register_builder("nodes_add_coords")
def nodes_add_coords(data: HeteroData) -> HeteroData:
    """
    Add coordinates from raw data to nodes.
    """
    data["nodes"].coords = data["_raw"].node_coords
    return data


@register_builder("nodes_concat_demand_coords_scaled")
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
    validate_node_attribute(data, "nodes", "demand")
    validate_node_attribute(data, "nodes", "coords")

    data["nodes"].x = torch.cat(
        [data["nodes"].demand, data["nodes"].coords],
        dim=-1,
    )
    return data


@register_builder("nodes_concat_demand_coords_raw")
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


@register_builder("nodes_add_net_demand")
def nodes_add_net_demand_raw(data: HeteroData) -> HeteroData:
    """
    Add net demand (inflow - outflow) to nodes.
    """
    demand = data["_raw"].demand

    # compute incoming and outgoing flows
    inflow = torch.sum(demand, dim=0)
    outflow = torch.sum(demand, dim=1)

    net_demand = inflow - outflow

    data["nodes"].net_demand = net_demand
    return data


# === Real Edges Builders ===
@register_builder("real_edges_add_index")
def real_edges_add_index(data: HeteroData) -> HeteroData:
    """
    Add edge index to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_index = data["_raw"].real_index
    return data


@register_builder("real_edges_add_capacity")
def real_edges_add_capacity(data: HeteroData) -> HeteroData:
    """
    Add capacity to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_capacity = data["_raw"].edge_capacity
    return data


@register_builder("real_edges_add_free_flow_time")
def real_edges_add_free_flow_time(data: HeteroData) -> HeteroData:
    """
    Add free_flow_time to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_free_flow_time = data["_raw"].edge_free_flow_time
    return data


@register_builder("real_edges_add_vcr")
def real_edges_add_vcr(data: HeteroData) -> HeteroData:
    """
    Add volume-capacity ratio to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_vcr = data["_raw"].edge_vcr
    return data


@register_builder("real_edges_add_flow")
def real_edges_add_flow(data: HeteroData) -> HeteroData:
    """
    Add flow to real edges.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_flow = data["_raw"].edge_flow
    return data


@register_builder("real_edges_stack_capacity_free_flow_raw")
def real_edges_stack_capacity_free_flow(data: HeteroData) -> HeteroData:
    """
    Concatenate real edge features (capacity and free flow time).
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_features = torch.stack(
        [data["_raw"].edge_capacity, data["_raw"].edge_free_flow_time], dim=1
    )
    return data


@register_builder("real_edges_stack_capacity_free_flow_scaled")
def real_edges_stack_capacity_free_flow_scaled(data: HeteroData) -> HeteroData:
    """
    Concatenate real edge features (capacity and free flow time).
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_features = torch.stack(
        [data[edge_type].edge_capacity, data[edge_type].edge_free_flow_time], dim=1
    )
    return data


# === Target builders ===
@register_builder("target_flow")
def target_flow(data: HeteroData) -> HeteroData:
    """
    Add flow target as y.
    """
    data.target_var = "flow"
    data.y = data["_raw"].edge_flow
    return data


@register_builder("target_vcr")
def target_vcr(data: HeteroData) -> HeteroData:
    """
    Add volume-capacity ratio target as y.
    """
    data.target_var = "vcr"
    data.y = data["_raw"].edge_vcr
    return data


# === Virtual Edges Builders ===
@register_builder("virtual_edges_add_index")
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


@register_builder("virtual_edges_add_demand")
def virtual_edges_add_demand(data: HeteroData) -> HeteroData:
    """Extract OD demand from scenario data and add as virtual edge attributes.

    Assumes:
    - data["_raw"].demand: tensor [num_nodes, num_nodes] OD matrix
    - data[("nodes", "virtual", "nodes")].edge_index exists

    Adds:
    - data[("nodes", "virtual", "nodes")].edge_attr: scalar demand [num_edges]

    Args:
        data: HeteroData with _raw attribute containing demand matrix and virtual edges.

    Returns:
        HeteroData with virtual edge demand attributes set.
    """
    virtual_edge_type = ("nodes", "virtual", "nodes")

    validate_edge_attribute(data, virtual_edge_type, "edge_index", expected_ndim=2)
    validate_node_attribute(data, "_raw", "demand", expected_ndim=2)

    edge_index = data[virtual_edge_type].edge_index

    # get demand matrix (already as tensor)
    demand_matrix = data["_raw"].demand  # [num_nodes, num_nodes]

    # extract demand values for each OD pair
    sources = edge_index[0]
    targets = edge_index[1]
    demand_values = demand_matrix[sources, targets]

    # set as edge attributes
    data[virtual_edge_type].edge_demand = demand_values

    return data


# === Cleaning Builders (for cleanup after processing) ===


@register_builder("clean_raw_data")
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


@register_builder("nodes_clean_demand")
def nodes_clean_demand(data: HeteroData) -> HeteroData:
    """
    Remove demand from nodes after it's been used.
    """
    validate_node_attribute(data, "nodes", "demand")
    del data["nodes"].demand
    return data


@register_builder("nodes_clean_coords")
def nodes_clean_coords(data: HeteroData) -> HeteroData:
    """
    Remove coords from nodes after they've been used.
    """
    validate_node_attribute(data, "nodes", "coords")
    del data["nodes"].coords
    return data


@register_builder("real_edges_clean_capacity")
def real_edges_clean_capacity(data: HeteroData) -> HeteroData:
    """
    Remove capacity from real edges after it's been used.
    """
    edge_type = ("nodes", "real", "nodes")
    validate_edge_attribute(data, edge_type, "edge_capacity")
    del data[edge_type].edge_capacity
    return data


@register_builder("real_edges_clean_free_flow_time")
def real_edges_clean_free_flow_time(data: HeteroData) -> HeteroData:
    """
    Remove free_flow_time from real edges after it's been used.
    """
    edge_type = ("nodes", "real", "nodes")
    validate_edge_attribute(data, edge_type, "edge_free_flow_time")
    del data[edge_type].edge_free_flow_time
    return data


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
    if name not in BUILDERS_REGISTRY:
        available = ", ".join(f"'{k}'" for k in BUILDERS_REGISTRY.keys())
        raise KeyError(f"Unknown builder '{name}'. Available builders: {available}")
    return BUILDERS_REGISTRY[name]


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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(builder_name="{self.builder_name}")'
