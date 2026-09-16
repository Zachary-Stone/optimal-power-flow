"""Train and evaluate regression models without notebook-managed state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """
    Store mean training MSE after each completed epoch.

    Parameters
    ----------
    epoch_mean_squared_errors : tuple[float, ...]
        Sample-weighted mean squared error ordered by epoch.
    """

    epoch_mean_squared_errors: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """
    Store aggregate full-dataset regression metrics.

    Parameters
    ----------
    sample_count : int
        Number of evaluated feature-target rows.
    mean_squared_error : float
        Mean squared error across every target value in every evaluated row.
    """

    sample_count: int
    mean_squared_error: float


def resolve_device(requested_device: str) -> torch.device:
    """
    Resolve an explicit or automatic PyTorch device selection.

    Parameters
    ----------
    requested_device : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    torch.device
        Available device selected for model execution.

    Raises
    ------
    ValueError
        If the requested device is unknown or unavailable.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise ValueError(f"Requested PyTorch device is unavailable: {requested_device!r}.")


def train_mse_regression(
    training_loader: DataLoader[Batch],
    model: nn.Module,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> TrainingHistory:
    """
    Train a regression model with Adam and mean squared error.

    Parameters
    ----------
    training_loader : torch.utils.data.DataLoader
        Shuffled batches of feature-target pairs.
    model : torch.nn.Module
        Model trained in place.
    epochs : int
        Number of complete training passes.
    learning_rate : float
        Adam optimizer learning rate.
    device : torch.device
        Device where batches and model are evaluated.

    Returns
    -------
    TrainingHistory
        Sample-weighted mean training MSE for each epoch.

    Raises
    ------
    ValueError
        If epochs or learning rate are invalid, or no training values are seen.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        squared_error_sum = 0.0
        value_count = 0
        for inputs, targets in training_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = torch.mean(torch.square(predictions - targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach()) * targets.numel()
            value_count += targets.numel()
        if value_count == 0:
            raise ValueError("training_loader produced no target values.")
        epoch_losses.append(squared_error_sum / value_count)
    return TrainingHistory(epoch_mean_squared_errors=tuple(epoch_losses))


def evaluate_mse_regression(
    data_loader: DataLoader[Batch], model: nn.Module, device: torch.device
) -> RegressionMetrics:
    """
    Evaluate full-dataset mean squared error without updating model parameters.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        Ordered feature-target batches to evaluate.
    model : torch.nn.Module
        Trained regression model.
    device : torch.device
        Device where batches and model are evaluated.

    Returns
    -------
    RegressionMetrics
        Evaluated row count and full-dataset mean squared error.

    Raises
    ------
    ValueError
        If the loader produces no target values.
    """
    model.to(device)
    model.eval()
    squared_error_sum = 0.0
    value_count = 0
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            squared_error_sum += float(torch.sum(torch.square(predictions - targets)))
            value_count += targets.numel()
            sample_count += len(targets)
    if value_count == 0:
        raise ValueError("data_loader produced no target values.")
    return RegressionMetrics(
        sample_count=sample_count,
        mean_squared_error=squared_error_sum / value_count,
    )
