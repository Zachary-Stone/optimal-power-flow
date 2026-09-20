"""Transformer architecture operating on static and dynamic bus features."""

import torch
from torch import nn

from optimal_power_flow.models.graph import BusLevelModelBase, GraphFeatures


class BusLevelTransformer(BusLevelModelBase):
    """Encode buses as a set of topology-labelled attention tokens."""

    def __init__(
        self,
        graph_features: GraphFeatures,
        output_size: int,
        width: int = 64,
        heads: int = 4,
        depth: int = 2,
    ) -> None:
        """Initialize bus projection, learned bus embeddings, and encoder stack."""
        super().__init__(graph_features)
        if output_size < 1:
            raise ValueError("output_size must be at least 1.")
        if width < 1 or heads < 1 or depth < 1:
            raise ValueError("width, heads, and depth must be positive.")
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        self.input_projection = nn.Linear(self.static_feature_count + 2, width)
        self.bus_embedding = nn.Parameter(torch.empty(self.node_count, width))
        nn.init.normal_(self.bus_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_head = nn.Linear(self.node_count * width, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict canonical OPF targets after attention across all buses."""
        node_states = self.input_projection(self.make_node_features(inputs))
        node_states = node_states + self.bus_embedding.unsqueeze(0)
        encoded = self.encoder(node_states)
        return self.output_head(encoded.flatten(start_dim=1))
