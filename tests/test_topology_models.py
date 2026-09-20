"""Validate topology-derived features and bus-level surrogate architectures."""

import unittest

import torch

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.models import (
    BusLevelTransformer,
    TopologyGNN,
    build_graph_features,
)
from optimal_power_flow.power import build_opf_context


class TopologyModelTests(unittest.TestCase):
    """Exercise graph construction and both bus-level model forward passes."""

    def setUp(self) -> None:
        """Create the Case 5 graph schema used by the model tests."""
        self.context = build_opf_context(case5_pjm())
        self.graph_features = build_graph_features(self.context)
        self.inputs = torch.zeros((3, len(self.context.input_columns)))

    def test_graph_features_have_expected_case5_structure(self) -> None:
        """Build eight static features, bidirectional edges, and row-normalization."""
        self.assertEqual(self.graph_features.static_node_features.shape, (5, 8))
        self.assertEqual(self.graph_features.edge_index.shape, (2, 12))
        self.assertEqual(self.graph_features.load_bus_indices.tolist(), [1, 2, 3])
        self.assertTrue(
            torch.allclose(
                self.graph_features.normalized_adjacency.sum(dim=1), torch.ones(5)
            )
        )

    def test_topology_gnn_and_transformer_predict_canonical_shapes(self) -> None:
        """Map each batch of canonical loads to one full canonical target vector."""
        output_size = len(self.context.output_columns)
        gnn = TopologyGNN(self.graph_features, output_size, width=4, depth=1)
        transformer = BusLevelTransformer(
            self.graph_features, output_size, width=4, heads=2, depth=1
        )

        self.assertEqual(gnn(self.inputs).shape, (3, output_size))
        self.assertEqual(transformer(self.inputs).shape, (3, output_size))

    def test_transformer_rejects_incompatible_attention_width(self) -> None:
        """Require an integer number of attention heads per embedding width."""
        with self.assertRaises(ValueError):
            BusLevelTransformer(
                self.graph_features,
                len(self.context.output_columns),
                width=5,
                heads=2,
            )
