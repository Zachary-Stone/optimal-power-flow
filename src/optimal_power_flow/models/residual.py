"""Define residual and bounded AC-OPF surrogate-model architectures."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Apply a two-layer bottleneck residual transformation at a fixed width.

    Parameters
    ----------
    width : int
        Shared input and output feature width. It must be at least 2 so the
        bottleneck projection has a positive width.

    Raises
    ------
    ValueError
        If ``width`` is less than 2.
    """

    def __init__(self, width: int) -> None:
        """Initialize the bottleneck transformation and skip connection."""
        super().__init__()
        if width < 2:
            raise ValueError("Residual block width must be at least 2.")
        bottleneck_width = width // 2
        self.network = nn.Sequential(
            nn.Linear(width, bottleneck_width),
            nn.ReLU(),
            nn.Linear(bottleneck_width, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual transformation.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched vectors with final dimension equal to the block width.

        Returns
        -------
        torch.Tensor
            Transformed vectors with the original skip connection added.
        """
        return inputs + self.network(inputs)


class ResidualMLP(nn.Module):
    """
    Predict canonical AC-OPF targets with fixed-width residual blocks.

    Parameters
    ----------
    input_size : int
        Number of canonical active and reactive load features.
    output_size : int
        Number of canonical generator-power and voltage targets.
    width : int, optional
        Hidden representation width. Default is 64.
    depth : int, optional
        Number of residual blocks. Default is 4.

    Raises
    ------
    ValueError
        If dimensions, width, or depth are invalid.
    """

    def __init__(
        self, input_size: int, output_size: int, width: int = 64, depth: int = 4
    ) -> None:
        """Initialize input projection, residual blocks, and output head."""
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1.")
        if output_size < 1:
            raise ValueError("output_size must be at least 1.")
        if width < 2:
            raise ValueError("width must be at least 2.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        self.network = nn.Sequential(
            nn.Linear(input_size, width),
            *(ResidualBlock(width) for _ in range(depth)),
            nn.Linear(width, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict canonical AC-OPF targets for a batch of load features.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched canonical active and reactive load features.

        Returns
        -------
        torch.Tensor
            Unbounded canonical AC-OPF target predictions.
        """
        return self.network(inputs)


class BoundedResidualMLP(ResidualMLP):
    """
    Bound residual-MLP outputs to case-specific operating limits.

    Parameters
    ----------
    input_size : int
        Number of canonical active and reactive load features.
    output_size : int
        Number of canonical generator-power and voltage targets.
    lower_bounds : torch.Tensor
        Lower bounds with shape ``(output_size,)`` or ``(1, output_size)``.
    upper_bounds : torch.Tensor
        Upper bounds with shape ``(output_size,)`` or ``(1, output_size)``.
    width : int, optional
        Hidden representation width. Default is 64.
    depth : int, optional
        Number of residual blocks. Default is 4.

    Raises
    ------
    ValueError
        If bounds have incompatible shape, are non-finite, or do not have
        strictly positive ranges.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        """Initialize the residual backbone and persistent output bounds."""
        super().__init__(input_size, output_size, width, depth)
        lower = torch.as_tensor(lower_bounds, dtype=torch.float32).reshape(1, -1)
        upper = torch.as_tensor(upper_bounds, dtype=torch.float32).reshape(1, -1)
        if lower.shape != (1, output_size) or upper.shape != (1, output_size):
            raise ValueError("Bounds must contain exactly one value per output.")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("Bounds must be finite.")
        if not torch.all(upper > lower):
            raise ValueError("Every upper bound must be greater than its lower bound.")
        self.register_buffer("lower_bounds", lower)
        self.register_buffer("upper_bounds", upper)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict AC-OPF targets within the registered lower and upper bounds.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched canonical active and reactive load features.

        Returns
        -------
        torch.Tensor
            Canonical target predictions strictly inside finite output bounds.
        """
        raw_outputs = super().forward(inputs)
        normalized_outputs = torch.sigmoid(raw_outputs)
        return self.lower_bounds + normalized_outputs * (
            self.upper_bounds - self.lower_bounds
        )
