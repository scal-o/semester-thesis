from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from ml_static.transforms import SequentialTransform
from ml_static.utils import validate_edge_attribute, validate_node_attribute

if TYPE_CHECKING:
    from ml_static.config import Config

# Loss component names (order matters for learnable weights indexing)
LOSS_COMPONENT_NAMES = ("vcr", "flow", "conservation")


class LossWrapper(nn.Module):
    """
    Wrapper for possible loss functions.
    """

    def __init__(self, loss_type: str, **kwargs) -> None:
        super().__init__()

        VALID_LOSSES = {
            "l1": self._l1_loss,
            "l2": self._l2_loss,
            "mse": self._l2_loss,
            "custom": self._custom_loss,
            "learnable_weights": self._learnable_loss,
        }

        INIT_FUNCTIONS = {
            "custom": self._init_custom_params,
            "learnable_weights": self._init_learnable_params,
        }

        if loss_type not in VALID_LOSSES:
            valid_list = ", ".join(f"'{t}'" for t in sorted(VALID_LOSSES))
            raise ValueError(f"Invalid loss type '{loss_type}'. Valid options are: {valid_list}.")

        self.loss_type: str = loss_type
        self.loss_fn: Callable = VALID_LOSSES[self.loss_type]

        if loss_type in INIT_FUNCTIONS:
            INIT_FUNCTIONS[loss_type](**kwargs)

        self._validated: bool = False

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create loss function from configuration object.
        """
        return cls(config.loss.type, **config.loss.kwargs)

    def forward(self, pred, data) -> tuple[torch.Tensor, dict]:
        return self.loss_fn(pred, data)

    def register_transform(self, transform: SequentialTransform) -> None:
        """
        Register a transform pipeline with the loss function.
        """
        self.transform: SequentialTransform = transform

    def _init_custom_params(self, **kwargs) -> None:
        """
        Initializes parameters for the fixed-weight custom loss.
        """
        defaults = {"w_vcr": 1.0, "w_flow": 0.005, "w_conservation": 0.02}
        self.weight_vars: dict[str, float] = {}
        for key, default_val in defaults.items():
            val = kwargs.get(key, None)
            if val is None:
                print(f"Warning: Using default weight for {key}: {default_val}")
                val = default_val
            self.weight_vars[key] = val

    def _init_learnable_params(self, **kwargs) -> None:
        """
        Initializes parameters for the learnable_weights loss.
        """
        # Learnable parameters for uncertainty weighting, representing log(sigma^2)
        # One for each loss component: vcr, flow, conservation
        self.log_vars = nn.Parameter(torch.zeros(len(LOSS_COMPONENT_NAMES)))

    def _validate_custom_data(self, data: HeteroData) -> None:
        """
        Validate that the data object contains the necessary attributes
        for computing physics-informed losses.
        """
        if self._validated:
            return

        if not hasattr(self, "transform"):
            raise RuntimeError(
                "Physics-informed losses require a transform. "
                "Call register_transform() before using this loss."
            )

        real_edge = ("nodes", "real", "nodes")
        validate_node_attribute(data, "nodes", "net_demand")
        validate_edge_attribute(data, real_edge, "edge_capacity")
        validate_edge_attribute(data, real_edge, "edge_flow")
        validate_edge_attribute(data, real_edge, "edge_index")

        if not hasattr(data, "target_var"):
            raise ValueError("data must have attribute 'target_var' for custom loss.")
        elif data.target_var != "vcr" and all(var != "vcr" for var in data.target_var):
            raise ValueError("Custom loss can only be used when target_var is 'vcr'.")

        self._validated = True

    def _calculate_physics_losses(
        self, pred: torch.Tensor, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the individual physics-based loss components (VCR, flow, conservation).
        This is a shared helper method for physics-informed losses.
        """
        self._validate_custom_data(data)

        real_edge = ("nodes", "real", "nodes")

        # 1. VCR loss
        vcr_loss = F.l1_loss(pred, data.y)

        # 2. Flow loss
        real_capacity = self.transform.inverse_transform(
            data[real_edge].edge_capacity, feature="edge_capacity"
        )
        real_vcr = self.transform.inverse_transform(pred, feature="target")
        true_flow = self.transform.inverse_transform(data[real_edge].edge_flow, feature="edge_flow")
        pred_flow = real_vcr * real_capacity
        flow_loss = F.smooth_l1_loss(pred_flow, true_flow)

        # 3. Conservation loss
        edge_index = data[real_edge].edge_index
        num_nodes = data["nodes"].num_nodes
        inflow = scatter(pred_flow, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        outflow = scatter(pred_flow, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        pred_demand = inflow - outflow
        net_demand = data["nodes"].net_demand
        conservation_loss = F.l1_loss(pred_demand, net_demand)

        return vcr_loss, flow_loss, conservation_loss

    def _l1_loss(self, pred, data) -> tuple[torch.Tensor, dict]:
        target = data.y
        return F.l1_loss(pred, target), {}

    def _l2_loss(self, pred, data) -> tuple[torch.Tensor, dict]:
        target = data.y
        return F.mse_loss(pred, target), {}

    def _custom_loss(self, pred, data) -> tuple[torch.Tensor, dict]:
        """
        Physics-informed loss with fixed weights.
        """
        vcr_loss, flow_loss, conservation_loss = self._calculate_physics_losses(pred, data)

        # Components in order matching LOSS_COMPONENT_NAMES
        loss_components = [vcr_loss, flow_loss, conservation_loss]
        weights = [
            self.weight_vars["w_vcr"],
            self.weight_vars["w_flow"],
            self.weight_vars["w_conservation"],
        ]

        total_loss = torch.tensor(0.0, device=pred.device)
        losses_log = {}

        for name, loss, weight in zip(LOSS_COMPONENT_NAMES, loss_components, weights):
            weighted = weight * loss
            total_loss = total_loss + weighted

            losses_log[f"unweighted_{name}_loss"] = loss.item()
            losses_log[f"weighted_{name}_loss"] = weighted.item()

        return total_loss, losses_log

    def _learnable_loss(self, pred, data) -> tuple[torch.Tensor, dict]:
        """
        Physics-informed loss with learnable uncertainty-based weights.

        Uses the formulation from Kendall & Gal (2017):
        L = sum_i [0.5 * exp(-s_i) * L_i + 0.5 * s_i]
        where s_i = log(sigma_i^2) is the learned log-variance.
        """
        vcr_loss, flow_loss, conservation_loss = self._calculate_physics_losses(pred, data)

        # Components in order matching LOSS_COMPONENT_NAMES and log_vars indexing
        loss_components = [vcr_loss, flow_loss, conservation_loss]

        total_loss = torch.tensor(0.0, device=pred.device)
        losses_log = {}

        for i, (name, loss) in enumerate(zip(LOSS_COMPONENT_NAMES, loss_components)):
            # Clamp log_vars for numerical stability
            log_var = self.log_vars[i].clamp(-10, 10)

            # Numerically stable implementation of uncertainty weighting
            # precision = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
            precision = 0.5 * torch.exp(-log_var)
            weighted_loss = precision * loss

            # Regularization term: log(sigma) = 0.5 * log(sigma^2) = 0.5 * log_var
            regularization = 0.5 * log_var

            total_loss = total_loss + weighted_loss + regularization

            # Logging
            sigma = torch.exp(log_var * 0.5)
            losses_log[f"unweighted_{name}_loss"] = loss.item()
            losses_log[f"weighted_{name}_loss"] = weighted_loss.item()
            losses_log[f"regularization_{name}"] = regularization.item()
            losses_log[f"learned_sigma_{name}"] = sigma.item()

        return total_loss, losses_log
