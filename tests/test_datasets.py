"""Validate deterministic OPF dataframe splitting and PyTorch loaders."""

import unittest

import numpy as np
import pandas as pd
import torch

from optimal_power_flow.dataset_io.datasets import (
    OPFDataset,
    build_data_loaders,
    split_train_test,
)


class OPFDatasetTests(unittest.TestCase):
    """Validate numerical and reproducible tabular-dataset behavior."""

    def setUp(self) -> None:
        """Create a small canonical feature-target fixture."""
        self.input_columns = ("input_a", "input_b")
        self.output_columns = ("output_a", "output_b")
        self.data = pd.DataFrame(
            {
                "input_a": range(10),
                "input_b": range(10, 20),
                "output_a": range(20, 30),
                "output_b": range(30, 40),
            }
        )

    def test_dataset_returns_float32_feature_target_pairs(self) -> None:
        """Convert one canonical dataframe row into two float32 tensors."""
        dataset = OPFDataset(self.data, self.input_columns, self.output_columns)
        inputs, outputs = dataset[0]

        self.assertEqual(len(dataset), 10)
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(outputs.dtype, torch.float32)
        self.assertTrue(torch.equal(inputs, torch.tensor([0.0, 10.0])))
        self.assertTrue(torch.equal(outputs, torch.tensor([20.0, 30.0])))

    def test_split_is_reproducible_without_global_rng_mutation(self) -> None:
        """Use a local legacy generator compatible with notebook split selection."""
        np.random.seed(13)
        expected_next_value = np.random.random()
        np.random.seed(13)

        first = split_train_test(self.data, train_fraction=0.8, seed=2026)
        second = split_train_test(self.data, train_fraction=0.8, seed=2026)

        self.assertEqual(
            first.training_data.index.tolist(), second.training_data.index.tolist()
        )
        self.assertEqual(np.random.random(), expected_next_value)
        self.assertEqual(len(first.training_data), 8)
        self.assertEqual(len(first.test_data), 2)

    def test_loaders_respect_selected_columns_and_batch_size(self) -> None:
        """Create shuffled and ordered tensors with the requested batch shape."""
        loaders = build_data_loaders(
            self.data,
            self.input_columns,
            self.output_columns,
            train_fraction=0.8,
            seed=2026,
            batch_size=3,
        )
        inputs, outputs = next(iter(loaders.training))

        self.assertEqual(inputs.shape[1], 2)
        self.assertEqual(outputs.shape[1], 2)
        self.assertLessEqual(len(inputs), 3)
