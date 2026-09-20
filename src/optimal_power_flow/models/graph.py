"""Graph-feature construction and topology-aware AC-OPF surrogate models."""

from dataclasses import dataclass

import numpy as np
import torch
from pypower import idx_brch, idx_bus, idx_gen
from torch import nn

from optimal_power_flow.power.context import OPFContext


@dataclass(frozen=True, slots=True)
class GraphFeatures:
    """Store static bus features and normalized network connectivity.

    Parameters
    ----------
    static_node_features : torch.Tensor
        Per-bus type, voltage, and online-generator capability features.
    edge_index : torch.Tensor
        Directed branch endpoints with shape ``(2, 2 * branch_count)``.
    normalized_adjacency : torch.Tensor
        Self-looped row-normalized adjacency matrix.
    load_bus_indices : torch.Tensor
        Bus positions corresponding to the canonical load-vector entries.
    """

    static_node_features: torch.Tensor
    edge_index: torch.Tensor
    normalized_adjacency: torch.Tensor
    load_bus_indices: torch.Tensor


def build_normalized_adjacency(
    edge_index: torch.Tensor, node_count: int
) -> torch.Tensor:
    """Build a self-looped row-normalized adjacency matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed source and destination indices with shape ``(2, edge_count)``.
    node_count : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Float32 adjacency matrix whose rows each sum to one.

    Raises
    ------
    ValueError
        If the indices or node count are invalid.
    """
    if node_count < 1:
        raise ValueError("node_count must be at least 1.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edge_count).")
    indices = edge_index.to(dtype=torch.long, device="cpu")
    if indices.numel() and (torch.any(indices < 0) or torch.any(indices >= node_count)):
        raise ValueError("edge_index contains a node outside the graph.")
    adjacency = torch.eye(node_count, dtype=torch.float32)
    if indices.numel():
        adjacency[indices[0], indices[1]] = 1.0
    return adjacency / adjacency.sum(dim=1, keepdim=True)


def build_graph_features(context: OPFContext) -> GraphFeatures:
    """Create topology and static bus features from an internal OPF case.

    Generator limits are expressed in per-unit using the case base MVA, so
    they align with the canonical model inputs and outputs.

    Parameters
    ----------
    context : optimal_power_flow.power.context.OPFContext
        Case schema containing the internal bus, generator, and branch tables.

    Returns
    -------
    GraphFeatures
        Static node features, bidirectional branch endpoints, normalized
        adjacency, and canonical load-bus locations.
    """
    case = context.internal_case
    bus = case["bus"]
    generator = case["gen"]
    branch = case["branch"]
    base_mva = float(case["baseMVA"])
    node_features = np.zeros((context.bus_count, 8), dtype=np.float32)
    node_features[:, 0] = bus[:, idx_bus.BUS_TYPE] / float(idx_bus.REF)
    node_features[:, 1] = bus[:, idx_bus.VMIN]
    node_features[:, 2] = bus[:, idx_bus.VMAX]

    for row in generator:
        if row[idx_gen.GEN_STATUS] <= 0:
            continue
        bus_index = int(row[idx_gen.GEN_BUS])
        node_features[bus_index, 3] = 1.0
        node_features[bus_index, 4] += row[idx_gen.PMIN] / base_mva
        node_features[bus_index, 5] += row[idx_gen.PMAX] / base_mva
        node_features[bus_index, 6] += row[idx_gen.QMIN] / base_mva
        node_features[bus_index, 7] += row[idx_gen.QMAX] / base_mva

    active_branch_rows = branch[:, idx_brch.BR_STATUS] > 0
    active_branch = branch[active_branch_rows]
    forward = active_branch[:, [idx_brch.F_BUS, idx_brch.T_BUS]].astype(np.int64)
    reverse = forward[:, ::-1]
    edge_index = torch.as_tensor(
        np.concatenate((forward, reverse), axis=0).T, dtype=torch.long
    )
    return GraphFeatures(
        static_node_features=torch.as_tensor(node_features),
        edge_index=edge_index,
        normalized_adjacency=build_normalized_adjacency(edge_index, context.bus_count),
        load_bus_indices=torch.as_tensor(context.load_bus_indices, dtype=torch.long),
    )


class BusLevelModelBase(nn.Module):
    """Convert flat canonical load inputs into bus-level node features."""

    def __init__(self, graph_features: GraphFeatures) -> None:
        """Register immutable graph features shared by bus-level architectures."""
        super().__init__()
        if graph_features.static_node_features.ndim != 2:
            raise ValueError("static_node_features must be a two-dimensional tensor.")
        self.register_buffer(
            "static_node_features",
            graph_features.static_node_features.to(torch.float32),
        )
        self.register_buffer(
            "load_bus_indices", graph_features.load_bus_indices.to(torch.long)
        )

    @property
    def node_count(self) -> int:
        """Return the number of buses represented by the graph."""
        return self.static_node_features.shape[0]

    @property
    def static_feature_count(self) -> int:
        """Return the number of static features attached to each bus."""
        return self.static_node_features.shape[1]

    def make_node_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Attach active and reactive load inputs to their corresponding buses."""
        load_count = self.load_bus_indices.numel()
        if inputs.ndim != 2 or inputs.shape[1] != 2 * load_count:
            raise ValueError(
                "inputs must have shape (batch_size, 2 * number_of_load_buses)."
            )
        dynamic = torch.zeros(
            (inputs.shape[0], self.node_count, 2),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        dynamic[:, :, 0].index_copy_(1, self.load_bus_indices, inputs[:, :load_count])
        dynamic[:, :, 1].index_copy_(1, self.load_bus_indices, inputs[:, load_count:])
        static = self.static_node_features.expand(inputs.shape[0], -1, -1)
        return torch.cat((dynamic, static), dim=2)


class GraphConvolution(nn.Module):
    """Apply one adjacency-aggregating graph-convolution layer."""

    def __init__(self, width: int) -> None:
        """Initialize self and neighbor projections at a common feature width."""
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.neighbor_projection = nn.Linear(width, width)

    def forward(
        self, node_states: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Combine each bus state with its normalized neighbor aggregate."""
        neighbors = torch.einsum("ij,bjf->bif", adjacency, node_states)
        return torch.relu(
            self.self_projection(node_states) + self.neighbor_projection(neighbors)
        )


class TopologyGNN(BusLevelModelBase):
    """Predict canonical OPF targets from bus features and network topology."""

    def __init__(
        self,
        graph_features: GraphFeatures,
        output_size: int,
        width: int = 64,
        depth: int = 2,
    ) -> None:
        """Initialize graph feature projection, message-passing layers, and head."""
        super().__init__(graph_features)
        if output_size < 1:
            raise ValueError("output_size must be at least 1.")
        if width < 1:
            raise ValueError("width must be at least 1.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        self.register_buffer(
            "normalized_adjacency", graph_features.normalized_adjacency
        )
        self.input_projection = nn.Linear(self.static_feature_count + 2, width)
        self.layers = nn.ModuleList(GraphConvolution(width) for _ in range(depth))
        self.output_head = nn.Linear(self.node_count * width, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict a canonical target vector for every flat load-input row."""
        node_states = torch.relu(self.input_projection(self.make_node_features(inputs)))
        for layer in self.layers:
            node_states = layer(node_states, self.normalized_adjacency)
        return self.output_head(node_states.flatten(start_dim=1))
