"""Validate device selection and reusable MSE training/evaluation loops."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from optimal_power_flow.models import BaselineMLP
from optimal_power_flow.training.loops import (
    evaluate_mse_regression,
    resolve_device,
    train_mse_regression,
)


class TrainingLoopTests(unittest.TestCase):
    """Validate model optimization and full-dataset MSE evaluation."""

    def test_cpu_training_reduces_linear_regression_error(self) -> None:
        """Fit a simple linear target using the shared MSE loop."""
        torch.manual_seed(0)
        inputs = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1)
        targets = 2.0 * inputs - 0.5
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=True)
        model = BaselineMLP(input_size=1, output_size=1, hidden_layers=(8,))
        device = resolve_device("cpu")
        before = evaluate_mse_regression(loader, model, device).mean_squared_error

        history = train_mse_regression(
            loader, model, epochs=80, learning_rate=0.02, device=device
        )
        after = evaluate_mse_regression(loader, model, device).mean_squared_error

        self.assertEqual(len(history.epoch_mean_squared_errors), 80)
        self.assertGreater(history.training_runtime_seconds, 0.0)
        self.assertLess(after, before)
        self.assertLess(after, 0.01)

    def test_unknown_device_is_rejected(self) -> None:
        """Reject an unrecognized device identifier before training starts."""
        with self.assertRaises(ValueError):
            resolve_device("not-a-device")
