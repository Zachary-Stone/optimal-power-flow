"""Run a short CPU-only baseline workflow on synthetic canonical OPF data."""

import unittest

import numpy as np
import pandas as pd
import torch

from optimal_power_flow.dataset_io import build_data_loaders
from optimal_power_flow.evaluation import summarize_regression_results
from optimal_power_flow.structs import ExperimentConfig, ModelConfig, TrainingConfig
from optimal_power_flow.training import run_baseline_mlp


class BaselineWorkflowSmokeTests(unittest.TestCase):
    """Exercise model construction, training, evaluation, and summary creation."""

    def test_synthetic_cpu_workflow_produces_a_metric_table(self) -> None:
        """Train a small baseline MLP and summarize its held-out test MSE."""
        generator = np.random.default_rng(0)
        input_columns = ("pd_bus_1", "qd_bus_1")
        output_columns = ("pg_gen_0", "vm_bus_0")
        inputs = generator.normal(size=(48, 2)).astype(np.float32)
        targets = np.column_stack(
            (1.5 * inputs[:, 0] - 0.5 * inputs[:, 1], inputs.sum(axis=1))
        )
        data = pd.DataFrame(
            np.column_stack((inputs, targets)),
            columns=input_columns + output_columns,
        )
        loaders = build_data_loaders(
            data,
            input_columns,
            output_columns,
            train_fraction=0.75,
            seed=0,
            batch_size=8,
        )
        torch.manual_seed(0)
        config = ExperimentConfig(
            "Synthetic baseline MLP",
            training=TrainingConfig(
                device="cpu", batch_size=8, epochs=40, learning_rate=0.02
            ),
            model=ModelConfig(hidden_layers=(16,)),
        )

        result = run_baseline_mlp(config, loaders, input_columns, output_columns)
        summary = summarize_regression_results({"baseline_mlp": result})

        self.assertEqual(summary.columns.tolist(), ["experiment", "samples", "mse"])
        self.assertEqual(int(summary.loc[0, "samples"]), 12)
        self.assertLess(float(summary.loc[0, "mse"]), 0.05)
