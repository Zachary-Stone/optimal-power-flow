"""Validate typed OPF experiment configuration and result structures."""

import unittest
from pathlib import Path

from optimal_power_flow.structs import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


class ExperimentStructTests(unittest.TestCase):
    """Validate the defaults and input validation for OPF experiment structs."""

    def test_default_experiment_matches_tutorial_configuration(self) -> None:
        """Construct the Case 5 baseline MLP experiment defaults."""
        config = ExperimentConfig("Baseline MLP")

        self.assertEqual(config.case.name, "case5_pjm")
        self.assertEqual(config.dataset.source, "opflearn_case5")
        self.assertEqual(config.dataset.cache_directory, Path("artifacts/data"))
        self.assertEqual(config.training.batch_size, 128)
        self.assertEqual(config.model.hidden_layers, (128, 64))

    def test_invalid_configuration_values_are_rejected(self) -> None:
        """Reject invalid data, training, model, and experiment values early."""
        invalid_configs = [
            lambda: DatasetConfig(random_seed=-1),
            lambda: DatasetConfig(train_fraction=1.0),
            lambda: TrainingConfig(batch_size=0),
            lambda: TrainingConfig(epochs=0),
            lambda: TrainingConfig(learning_rate=0.0),
            lambda: ModelConfig(hidden_layers=(128, 0)),
            lambda: ModelConfig(residual_width=1),
            lambda: LossConfig(objective_weight=-1.0),
            lambda: ExperimentConfig(""),
        ]

        for build_config in invalid_configs:
            with self.subTest(build_config=build_config):
                with self.assertRaises(ValueError):
                    build_config()

    def test_result_retains_metrics_and_artifact_paths(self) -> None:
        """Associate a completed result with its typed configuration."""
        result = ExperimentResult(
            config=ExperimentConfig("Baseline MLP"),
            model=None,
            metrics={"mse": 0.1},
            artifact_paths=(Path("artifacts/outputs/metrics.csv"),),
        )

        self.assertEqual(result.metrics["mse"], 0.1)
        self.assertEqual(result.artifact_paths[0].name, "metrics.csv")
