"""PyTorch surrogate-model architectures for AC-OPF."""

from optimal_power_flow.models.graph import (
    BusLevelModelBase,
    GraphConvolution,
    GraphFeatures,
    TopologyGNN,
    build_graph_features,
    build_normalized_adjacency,
)
from optimal_power_flow.models.mlp import BaselineMLP
from optimal_power_flow.models.physics import (
    GeneratorBalanceMapping,
    PhysicalCompletionMLP,
    complete_balancing_generators,
    identify_balancing_generators,
)
from optimal_power_flow.models.residual import (
    BoundedResidualMLP,
    ResidualBlock,
    ResidualMLP,
)
from optimal_power_flow.models.transformer import BusLevelTransformer

__all__ = [
    "BaselineMLP",
    "BoundedResidualMLP",
    "BusLevelModelBase",
    "BusLevelTransformer",
    "GeneratorBalanceMapping",
    "GraphConvolution",
    "GraphFeatures",
    "PhysicalCompletionMLP",
    "ResidualBlock",
    "ResidualMLP",
    "TopologyGNN",
    "build_graph_features",
    "build_normalized_adjacency",
    "complete_balancing_generators",
    "identify_balancing_generators",
]
