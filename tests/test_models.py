"""Validate baseline neural-surrogate model construction."""

import unittest

import torch

from optimal_power_flow.models import (
    BaselineMLP,
    BoundedResidualMLP,
    ResidualBlock,
    ResidualMLP,
)


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

    def test_residual_models_preserve_shape_and_apply_bounds(self) -> None:
        """Map batches through residual blocks and enforce registered output limits."""
        inputs = torch.randn((3, 2))
        residual = ResidualMLP(input_size=2, output_size=3, width=4, depth=2)
        bounded = BoundedResidualMLP(
            input_size=2,
            output_size=3,
            lower_bounds=torch.tensor([-1.0, 0.0, 2.0]),
            upper_bounds=torch.tensor([1.0, 2.0, 3.0]),
            width=4,
            depth=2,
        )

        residual_outputs = residual(inputs)
        bounded_outputs = bounded(inputs)

        self.assertEqual(ResidualBlock(4)(torch.zeros((1, 4))).shape, (1, 4))
        self.assertEqual(residual_outputs.shape, (3, 3))
        self.assertEqual(bounded_outputs.shape, (3, 3))
        self.assertTrue(torch.all(bounded_outputs >= bounded.lower_bounds))
        self.assertTrue(torch.all(bounded_outputs <= bounded.upper_bounds))

    def test_invalid_residual_widths_and_bounds_are_rejected(self) -> None:
        """Reject unusable residual bottlenecks and incompatible bound vectors."""
        with self.assertRaises(ValueError):
            ResidualBlock(1)
        with self.assertRaises(ValueError):
            BoundedResidualMLP(
                input_size=2,
                output_size=2,
                lower_bounds=torch.tensor([1.0, 0.0]),
                upper_bounds=torch.tensor([1.0, 2.0]),
            )
