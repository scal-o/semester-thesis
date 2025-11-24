import torch
import torch.nn as nn


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    graph,
) -> float:
    """
    Perform a single training step.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer for updating model parameters.
        criterion: The loss function.
        graph: A batch of graph data.

    Returns:
        The training loss for this step.
    """
    model.train()
    optimizer.zero_grad()

    pred = model(graph)
    target = graph.y
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(
    model: nn.Module,
    criterion: nn.Module,
    graph,
) -> float:
    """
    Perform validation on a batch of graph data.

    Args:
        model: The neural network model to validate.
        criterion: The loss function.
        graph: A batch of graph data.

    Returns:
        The validation loss.
    """
    model.eval()
    with torch.no_grad():
        pred = model(graph)
        target = graph.y
        loss = criterion(pred, target)

    return loss.item()


def run_epoch(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run a full epoch of training over the data loader.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader providing batches of graph data for training.
        val_loader: DataLoader providing batches of graph data for validation.
        optimizer: The optimizer for updating model parameters.
        criterion: The loss function.

    Returns:
        The average training loss over the epoch.
    """
    e_train_loss = 0.0
    e_val_loss = 0.0
    train_batches = 0
    val_batches = 0

    for data in train_loader:
        data = data.to(device)
        train_loss = train(model, optimizer, loss, data)

        e_train_loss += train_loss
        train_batches += 1

    for data in val_loader:
        data = data.to(device)
        val_loss = validate(model, loss, data)
        e_val_loss += val_loss
        val_batches += 1

    # average loss
    e_train_loss /= train_batches
    e_val_loss /= val_batches

    return e_train_loss, e_val_loss


def run_test(
    model: nn.Module,
    test_loader,
    loss: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the test dataset.

    Args:
        model: The neural network model to evaluate.
        test_loader: DataLoader providing batches of graph data for testing.
        criterion: The loss function.
        get_target: A callable that extracts the target from the graph.

    Returns:
        The average test loss.
    """
    test_loss = 0.0
    test_batches = 0

    for data in test_loader:
        data = data.to(device)
        loss_out = validate(model, loss, data)
        test_loss += loss_out
        test_batches += 1

    test_loss /= test_batches
    return test_loss
