"""Validate baseline neural-surrogate model construction."""

import unittest

import torch

from optimal_power_flow.models import BaselineMLP


class BaselineMLPTests(unittest.TestCase):
    """Validate dimensions and input validation for the baseline MLP."""

    def test_model_predicts_requested_target_shape(self) -> None:
        """Map a batch of six load features to twenty OPF targets."""
        model = BaselineMLP(input_size=6, output_size=20, hidden_layers=(16, 8))

        predictions = model(torch.zeros((4, 6)))

        self.assertEqual(predictions.shape, (4, 20))

    def test_invalid_model_dimensions_are_rejected(self) -> None:
        """Reject non-positive dimensions and hidden-layer widths."""
        invalid_arguments = [
            {"input_size": 0, "output_size": 2},
            {"input_size": 2, "output_size": 0},
            {"input_size": 2, "output_size": 2, "hidden_layers": ()},
            {"input_size": 2, "output_size": 2, "hidden_layers": (4, 0)},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    BaselineMLP(**arguments)
