"""Early stopping for training loop."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping after last improvement.
        min_delta: Minimum change in monitored metric to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch = 0
        self.best_checkpoint: dict | None = None
        self.early_stop = False

        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")

    def __call__(self, metric: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop and update best model.

        Args:
            metric: Current validation metric value.
            model: Current model to potentially save.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            # first epoch
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # improvement found
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint in memory.

        Args:
            model: Model to save.
        """
        self.best_checkpoint = model.extract_checkpoint()

    def load_best_model(self, model: nn.Module) -> None:
        """Load the best model state into the provided model.

        Args:
            model: Model to load best weights into.

        Raises:
            RuntimeError: If no best model state is available.
        """
        if self.best_checkpoint is None:
            raise RuntimeError("No best model checkpoint available to load")
        model.load_state_dict(self.best_checkpoint["state_dict"])

    @property
    def best_metric(self) -> float:
        """Get the best metric value.

        Returns:
            Best metric value recorded.

        Raises:
            RuntimeError: If no metric has been recorded yet.
        """
        if self.best_score is None:
            raise RuntimeError("No metric recorded yet")
        return -self.best_score if self.mode == "min" else self.best_score
