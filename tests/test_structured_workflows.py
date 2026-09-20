"""Exercise configured topology and physics-completing workflows on Case 5."""

import unittest

import numpy as np
import pandas as pd
from pypower import idx_bus, idx_gen

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.dataset_io import build_data_loaders
from optimal_power_flow.models import (
    BoundedResidualMLP,
    BusLevelTransformer,
    PhysicalCompletionMLP,
    TopologyGNN,
)
from optimal_power_flow.power import OPFAwareMetric, build_opf_context, solve_ac_opf
from optimal_power_flow.structs import ExperimentConfig, ModelConfig, TrainingConfig
from optimal_power_flow.training import (
    run_augmented_lagrangian_mlp,
    run_bus_transformer,
    run_physical_completion_mlp,
    run_topology_gnn,
)


class StructuredWorkflowTests(unittest.TestCase):
    """Validate short training runs for every step-seven workflow."""

    def setUp(self) -> None:
        """Create repeated solved rows for fast deterministic train/test runs."""
        self.context = build_opf_context(case5_pjm())
        result = solve_ac_opf(self.context.case)
        case = self.context.internal_case
        base_mva = float(case["baseMVA"])
        inputs = (
            np.concatenate(
                (
                    case["bus"][self.context.load_bus_indices, idx_bus.PD],
                    case["bus"][self.context.load_bus_indices, idx_bus.QD],
                )
            )
            / base_mva
        )
        outputs = np.concatenate(
            (
                result["gen"][:, idx_gen.PG] / base_mva,
                result["gen"][:, idx_gen.QG] / base_mva,
                result["bus"][:, idx_bus.VM],
                np.deg2rad(result["bus"][:, idx_bus.VA]),
            )
        )
        data = pd.DataFrame(
            np.repeat(np.concatenate((inputs, outputs)).reshape(1, -1), 4, axis=0),
            columns=self.context.input_columns + self.context.output_columns,
        )
        self.loaders = build_data_loaders(
            data,
            self.context.input_columns,
            self.context.output_columns,
            train_fraction=0.75,
            seed=0,
            batch_size=2,
        )
        self.metric = OPFAwareMetric(self.context, device="cpu")
        self.training = TrainingConfig(device="cpu", batch_size=2, epochs=1)

    def test_step_seven_workflows_complete(self) -> None:
        """Run one epoch for graph, attention, completion, and dual models."""
        topology = run_topology_gnn(
            ExperimentConfig(
                "Topology",
                training=self.training,
                model=ModelConfig(
                    name="topology_gnn", topology_width=4, topology_depth=1
                ),
            ),
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )
        transformer = run_bus_transformer(
            ExperimentConfig(
                "Transformer",
                training=self.training,
                model=ModelConfig(
                    name="bus_transformer",
                    transformer_width=4,
                    transformer_heads=2,
                    transformer_depth=1,
                ),
            ),
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )
        completion = run_physical_completion_mlp(
            ExperimentConfig(
                "Completion",
                training=self.training,
                model=ModelConfig(
                    name="physical_completion_mlp", residual_width=4, residual_depth=1
                ),
            ),
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )
        augmented = run_augmented_lagrangian_mlp(
            ExperimentConfig(
                "Augmented",
                training=self.training,
                model=ModelConfig(
                    name="augmented_lagrangian_mlp", residual_width=4, residual_depth=1
                ),
            ),
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )

        self.assertIsInstance(topology.model, TopologyGNN)
        self.assertIsInstance(transformer.model, BusLevelTransformer)
        self.assertIsInstance(completion.model, PhysicalCompletionMLP)
        self.assertIsInstance(augmented.model, BoundedResidualMLP)
        self.assertEqual(
            topology.metadata["training_signal"], "topology_message_passing"
        )
        self.assertEqual(transformer.metadata["training_signal"], "bus_attention")
        self.assertEqual(
            completion.metadata["training_signal"], "physical_generator_completion"
        )
        self.assertEqual(
            augmented.metadata["training_signal"], "adaptive_augmented_lagrangian"
        )
