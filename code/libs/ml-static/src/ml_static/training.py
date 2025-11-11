import torch
import torch.nn as nn
from typing import Callable


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    graph,
    get_target: Callable = None,
) -> float:
    """
    Perform a single training step.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer for updating model parameters.
        criterion: The loss function.
        graph: A batch of graph data.
        get_target: A callable that extracts the target from the graph.
                   Defaults to lambda g: g["real"].edge_labels if None.

    Returns:
        The training loss for this step.
    """
    if get_target is None:
        get_target = lambda g: g["real"].edge_labels

    model.train()
    optimizer.zero_grad()

    pred = model(graph)
    target = get_target(graph)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(
    model: nn.Module,
    criterion: nn.Module,
    graph,
    get_target: Callable = None,
) -> float:
    """
    Perform validation on a batch of graph data.

    Args:
        model: The neural network model to validate.
        criterion: The loss function.
        graph: A batch of graph data.
        get_target: A callable that extracts the target from the graph.
                   Defaults to lambda g: g["real"].edge_labels if None.

    Returns:
        The validation loss.
    """
    if get_target is None:
        get_target = lambda g: g["real"].edge_labels

    model.eval()
    with torch.no_grad():
        pred = model(graph)
        target = get_target(graph)
        loss = criterion(pred, target)

    return loss.item()
