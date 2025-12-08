from __future__ import annotations

from typing import TYPE_CHECKING, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from ml_static.transforms import SequentialTransform
from ml_static.utils import validate_edge_attribute, validate_node_attribute

if TYPE_CHECKING:
    from ml_static.config import Config


class LossWrapper(nn.Module):
    """
    Wrapper for possble loss functions.
    """

    def __init__(self, loss_type: str, **kwargs) -> None:
        super().__init__()

        # define valid loss types
        VALID_LOSSES = {
            "l1": self._l1_loss,
            "l2": self._l2_loss,
            "mse": self._l2_loss,
            "custom": self._custom_loss,
        }

        # set loss type
        if loss_type not in VALID_LOSSES:
            valid_list = ", ".join(f"'{t}'" for t in sorted(VALID_LOSSES))
            raise ValueError(f"Invalid loss type '{loss_type}'. Valid options are: {valid_list}.")

        self.loss_type = loss_type

        # assign keywork parameters to internal dictionary if necessary
        if self.loss_type == "custom":
            self.params = self._init_custom_params(**kwargs)
            self._validated: bool = False

        self.loss_fn = VALID_LOSSES[self.loss_type]

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create loss function from configuration object.

        Args:
            config: Configuration object containing loss parameters.

        Returns:
            LossWrapper instance.
        """
        return cls(config.loss.type, **config.loss.kwargs)

    ## === forward call ===
    def forward(self, pred, data):
        return self.loss_fn(pred, data)

    ## === public methods ===
    def register_transform(self, transform: SequentialTransform) -> None:
        """
        Register a transform pipeline with the loss function.
        This is useful for custom loss functions that may need to
        access data scalers or builders.

        Args:
            transform: SequentialTransform pipeline to register.
        """
        self.transform: SequentialTransform = transform

    ## === internal methods for custom loss ===
    def _init_custom_params(self, **kwargs) -> dict[str, float]:
        """
        Function to validate and initialize parameters for the custom loss (physics informed
        loss).
        """
        # default values from Liu & Meidani, 2024
        defaults = {"w_vcr": 1.0, "w_flow": 0.005, "w_conservation": 0.05}

        params = {}
        params["w_vcr"] = kwargs.get("w_vcr", None)
        params["w_flow"] = kwargs.get("w_flow", None)
        params["w_conservation"] = kwargs.get("w_conservation", None)

        for k, v in params.items():
            if v is None:
                print(f"[Warning] Using default value for custom loss parameter {k}: {defaults[k]}")
                params[k] = defaults[k]

        return params

    def _validate_custom_data(self, data: HeteroData) -> None:
        """
        Validate that the data object contains the necessary attributes
        for computing the custom loss.
        """

        # skip validation if already done
        if self._validated:
            return

        # def edge type
        real_edge = ("nodes", "real", "nodes")

        # validate node attributes
        validate_node_attribute(data, "nodes", "net_demand")

        # validate edge attributes
        validate_edge_attribute(data, real_edge, "edge_capacity")
        validate_edge_attribute(data, real_edge, "edge_flow")
        validate_edge_attribute(data, real_edge, "edge_index")

        # validate target variable
        if not hasattr(data, "target_var"):
            raise ValueError("data must have attribute 'target_var' for custom loss.")
        elif data.target_var != "vcr" and all(var != "vcr" for var in data.target_var):
            raise ValueError("Custom loss can only be used when target_var is 'vcr'.")

        # if all checks pass, set the internal validate flag
        self._validated = True

    ## === loss methods ===
    def _l1_loss(self, pred, data):
        target = data.y
        return F.l1_loss(pred, target)

    def _l2_loss(self, pred, data):
        target = data.y
        return F.mse_loss(pred, target)

    def _custom_loss(self, pred, data):
        """
        Physics informed custom loss function.
        Uses:
            - VCR loss
            - Flow loss
            - Conservation loss
        Only valid when target_var is 'vcr'.
        """

        # check validation
        self._validate_custom_data(data)

        # define edge type
        real_edge = ("nodes", "real", "nodes")

        # == vcr loss
        # compute predictions for vcr
        pred_vcr = pred
        # compute loss
        vcr_loss = F.mse_loss(pred_vcr, data.y)

        # == flow loss
        # retrieve real capacity and vcr values
        real_capacity = self.transform.inverse_transform(
            data[real_edge].edge_capacity, feature="edge_capacity"
        )
        real_vcr = self.transform.inverse_transform(pred_vcr, feature="target")
        true_flow = self.transform.inverse_transform(data[real_edge].edge_flow, feature="edge_flow")

        # compute flow predictions
        pred_flow = real_vcr * real_capacity
        flow_loss = F.mse_loss(pred_flow, true_flow)

        # == conservation loss
        # compute inflow and outflow for each node
        edge_index = data[real_edge].edge_index
        # use data["nodes"].num_nodes to ensure we only consider the relevant nodes
        # data.num_nodes might include other node types (e.g. _raw) if not cleaned
        num_nodes = data["nodes"].num_nodes
        inflow = scatter(pred_flow, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        outflow = scatter(pred_flow, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")

        pred_demand = inflow - outflow
        net_demand = self.transform.inverse_transform(data["nodes"].net_demand, feature="demand")

        conservation_loss = F.l1_loss(pred_demand, net_demand)

        ## === total loss
        total_loss = (
            self.params["w_vcr"] * vcr_loss
            + self.params["w_flow"] * flow_loss
            + self.params["w_conservation"] * conservation_loss
        )

        return total_loss
