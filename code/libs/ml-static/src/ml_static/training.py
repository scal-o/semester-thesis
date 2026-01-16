from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import autocast
from torch.amp import GradScaler

if TYPE_CHECKING:
    from ml_static.losses import LossWrapper

SCALER = GradScaler()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: LossWrapper,
    graph,
    device: torch.device,
) -> tuple[float, dict]:
    """
    Perform a single training step.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer for updating model parameters.
        criterion: The loss function.
        graph: A batch of graph data.
        device: Device to run computations on.

    Returns:
        Tuple of (training loss, loss components dict).
    """
    if device.type != "cuda":
        raise RuntimeError(f"AMP training requires a CUDA device; got '{device.type}'.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, dtype=torch.float16):
        pred = model(graph)
        loss, loss_components = criterion(pred, graph)

    SCALER.scale(loss).backward()
    SCALER.step(optimizer)
    SCALER.update()

    return loss.item(), loss_components


def validate(
    model: nn.Module,
    criterion: LossWrapper,
    graph,
    device: torch.device,
) -> tuple[float, dict]:
    """
    Perform validation on a batch of graph data.

    Args:
        model: The neural network model to validate.
        criterion: The loss function.
        graph: A batch of graph data.
        device: Device to run computations on.

    Returns:
        Tuple of (validation loss, loss components dict).
    """
    if device.type != "cuda":
        raise RuntimeError(f"AMP evaluation requires a CUDA device; got '{device.type}'.")

    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=torch.float16):
            pred = model(graph)
            loss, loss_components = criterion(pred, graph)

    return loss.item(), loss_components


def run_epoch(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss: LossWrapper,
    device: torch.device,
) -> tuple[float, float, dict, dict]:
    """
    Run a full epoch of training over the data loader.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader providing batches of graph data for training.
        val_loader: DataLoader providing batches of graph data for validation.
        optimizer: The optimizer for updating model parameters.
        loss: The loss function.
        device: Device to run computations on.

    Returns:
        Tuple of (avg train loss, avg val loss, train components dict, val components dict).
    """
    e_train_loss = 0.0
    e_val_loss = 0.0
    train_batches = 0
    val_batches = 0

    # accumulators for loss components
    train_components_acc: dict[str, float] = {}
    val_components_acc: dict[str, float] = {}

    for data in train_loader:
        data = data.to(device)
        train_loss, train_components = train(model, optimizer, loss, data, device)

        e_train_loss += train_loss
        train_batches += 1

        # accumulate components
        for key, value in train_components.items():
            if key not in train_components_acc:
                train_components_acc[key] = 0.0
            train_components_acc[key] += value

    for data in val_loader:
        data = data.to(device)
        val_loss, val_components = validate(model, loss, data, device)
        e_val_loss += val_loss
        val_batches += 1

        # accumulate components
        for key, value in val_components.items():
            if key not in val_components_acc:
                val_components_acc[key] = 0.0
            val_components_acc[key] += value

    # average loss
    e_train_loss /= train_batches
    e_val_loss /= val_batches

    # average components
    for key in train_components_acc:
        train_components_acc[key] /= train_batches

    for key in val_components_acc:
        val_components_acc[key] /= val_batches

    return e_train_loss, e_val_loss, train_components_acc, val_components_acc


def run_test(
    model: nn.Module,
    test_loader,
    loss: LossWrapper,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the test dataset.

    Args:
        model: The neural network model to evaluate.
        test_loader: DataLoader providing batches of graph data for testing.
        loss: The loss function.
        device: Device to run computations on.

    Returns:
        The average test loss.
    """
    test_loss = 0.0
    test_batches = 0

    for data in test_loader:
        data = data.to(device)
        loss_out, _ = validate(model, loss, data, device)
        test_loss += loss_out
        test_batches += 1

    test_loss /= test_batches
    return test_loss
