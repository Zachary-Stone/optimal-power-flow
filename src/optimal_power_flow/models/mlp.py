"""Define baseline multilayer-perceptron OPF surrogate models."""

import torch
from torch import nn


class BaselineMLP(nn.Module):
    """
    Map load features to an AC-OPF solution-vector approximation.

    Parameters
    ----------
    input_size : int
        Number of canonical active and reactive load features.
    output_size : int
        Number of canonical generator-power and voltage targets.
    hidden_layers : tuple[int, ...], optional
        Width of each ReLU-activated hidden layer. Default is ``(128, 64)``.

    Raises
    ------
    ValueError
        If dimensions or hidden-layer widths are not positive.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: tuple[int, ...] = (128, 64),
    ) -> None:
        """Initialize the feed-forward baseline architecture."""
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1.")
        if output_size < 1:
            raise ValueError("output_size must be at least 1.")
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive widths.")

        layers: list[nn.Module] = []
        previous_size = input_size
        for width in hidden_layers:
            layers.extend((nn.Linear(previous_size, width), nn.ReLU()))
            previous_size = width
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict one target vector for each input feature vector.

        Parameters
        ----------
        inputs : torch.Tensor
            Two-dimensional batch of canonical load features.

        Returns
        -------
        torch.Tensor
            Two-dimensional batch of predicted AC-OPF target vectors.
        """
        return self.network(inputs)
