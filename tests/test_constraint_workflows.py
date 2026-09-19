"""Exercise configured bounded constraint-aware workflows on Case 5 tensors."""

import unittest

import numpy as np
import pandas as pd
from pypower import idx_bus, idx_gen

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.dataset_io import build_data_loaders
from optimal_power_flow.models import BoundedResidualMLP
from optimal_power_flow.power import OPFAwareMetric, build_opf_context, solve_ac_opf
from optimal_power_flow.structs import ExperimentConfig, ModelConfig, TrainingConfig
from optimal_power_flow.training import (
    run_bounded_penalty_mlp,
    run_bounded_self_supervised_mlp,
)


class ConstraintWorkflowTests(unittest.TestCase):
    """Validate short end-to-end runs for both named bounded workflows."""

    def setUp(self) -> None:
        """Create repeated canonical Case 5 rows for fast train/test execution."""
        self.context = build_opf_context(case5_pjm())
        result = solve_ac_opf(self.context.case)
        internal_case = self.context.internal_case
        base_mva = float(internal_case["baseMVA"])
        inputs = (
            np.concatenate(
                (
                    internal_case["bus"][self.context.load_bus_indices, idx_bus.PD],
                    internal_case["bus"][self.context.load_bus_indices, idx_bus.QD],
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
        columns = self.context.input_columns + self.context.output_columns
        data = pd.DataFrame(
            np.repeat(np.concatenate((inputs, outputs)).reshape(1, -1), 8, axis=0),
            columns=columns,
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

    def test_bounded_penalty_and_self_supervised_workflows_complete(self) -> None:
        """Run each named one-epoch workflow and retain its bounded model result."""
        common = {
            "training": TrainingConfig(device="cpu", batch_size=2, epochs=1),
            "model": ModelConfig(residual_width=4, residual_depth=1),
        }
        penalty_config = ExperimentConfig(
            "Bounded penalty",
            model=ModelConfig(
                name="bounded_penalty_mlp", residual_width=4, residual_depth=1
            ),
            training=common["training"],
        )
        self_supervised_config = ExperimentConfig(
            "Bounded self-supervised",
            model=ModelConfig(
                name="bounded_self_supervised_mlp",
                residual_width=4,
                residual_depth=1,
            ),
            training=common["training"],
        )

        penalty_result = run_bounded_penalty_mlp(
            penalty_config,
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )
        self_supervised_result = run_bounded_self_supervised_mlp(
            self_supervised_config,
            self.loaders,
            self.context.input_columns,
            self.context.output_columns,
            self.metric,
        )

        self.assertIsInstance(penalty_result.model, BoundedResidualMLP)
        self.assertIsInstance(self_supervised_result.model, BoundedResidualMLP)
        self.assertEqual(
            penalty_result.metadata["training_signal"], "mse_plus_opf_penalties"
        )
        self.assertEqual(
            self_supervised_result.metadata["training_signal"],
            "opf_objective_and_constraints",
        )
