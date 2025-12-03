"""
Factory functions for creating PyTorch objects from configuration.

These functions keep domain object creation separate from configuration
data structures, improving testability and reducing coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from ml_static.config import OptimizerConfig, SchedulerConfig


def create_optimizer(config: OptimizerConfig, params: Iterator[nn.Parameter]) -> optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        config: Optimizer configuration.
        params: Model parameters to optimize.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer type is not supported.
    """
    if config.type == "adam":
        return optim.Adam(params, lr=config.learning_rate)
    elif config.type == "sgd":
        return optim.SGD(params, lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")


def create_scheduler(config: SchedulerConfig, optimizer: optim.Optimizer) -> LRScheduler:
    """
    Create learning rate scheduler from configuration.

    Args:
        config: Scheduler configuration.
        optimizer: Optimizer to schedule.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: If scheduler type is not supported.
    """
    if config.type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.type}")
