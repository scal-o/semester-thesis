from __future__ import annotations

from typing import TYPE_CHECKING, Self

import torch.nn as nn
import torch.nn.functional as F

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

    ## === internal methods ===
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

    ## === loss methods ===
    def _l1_loss(self, pred, data):
        target = data.y
        return F.l1_loss(pred, target)

    def _l2_loss(self, pred, data):
        target = data.y
        return F.mse_loss(pred, target)

    def _custom_loss(self, pred, data):
        raise NotImplementedError("Custom loss function is not implemented yet.")
