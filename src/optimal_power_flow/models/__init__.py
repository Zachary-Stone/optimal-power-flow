"""PyTorch surrogate-model architectures for AC-OPF."""

from optimal_power_flow.models.mlp import BaselineMLP
from optimal_power_flow.models.residual import (
    BoundedResidualMLP,
    ResidualBlock,
    ResidualMLP,
)

__all__ = ["BaselineMLP", "BoundedResidualMLP", "ResidualBlock", "ResidualMLP"]
