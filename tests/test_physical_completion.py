"""Validate balancing-generator completion and augmented-Lagrangian dual updates."""

import unittest

import torch

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.models import (
    PhysicalCompletionMLP,
    identify_balancing_generators,
)
from optimal_power_flow.power import OPFAwareMetric, build_opf_context
from optimal_power_flow.training import AugmentedLagrangianLoss
from optimal_power_flow.training.losses import get_opf_constraint_items


class PhysicalCompletionTests(unittest.TestCase):
    """Test Case 5 generator partitioning and voltage-based completion."""

    def setUp(self) -> None:
        """Create Case 5 tensors and a CPU physical metric."""
        torch.manual_seed(0)
        self.context = build_opf_context(case5_pjm())
        self.metric = OPFAwareMetric(self.context, device="cpu")
        self.inputs = torch.zeros((2, len(self.context.input_columns)))

    def test_completion_balances_every_generator_bus_and_pins_reference_angle(
        self,
    ) -> None:
        """Complete one generator per bus after predicting the duplicate-unit output."""
        mapping = identify_balancing_generators(self.metric)
        model = PhysicalCompletionMLP(
            self.metric, len(self.context.input_columns), width=4, depth=1
        )
        outputs = model(self.inputs)
        active, reactive = self.metric.power_balance_residual(self.inputs, outputs)
        _, _, magnitude, angle = self.metric.split_outputs(outputs)
        extra_active = outputs[:, self.metric.output_slices.generator_active][
            :, mapping.extra_generator_indices
        ]
        active_bounds = self.metric.output_minimum[
            :, self.metric.output_slices.generator_active
        ][:, mapping.extra_generator_indices]
        active_upper = self.metric.output_maximum[
            :, self.metric.output_slices.generator_active
        ][:, mapping.extra_generator_indices]

        self.assertEqual(mapping.balancing_generator_indices.tolist(), [0, 2, 3, 4])
        self.assertEqual(mapping.extra_generator_indices.tolist(), [1])
        self.assertEqual(outputs.shape, (2, len(self.context.output_columns)))
        self.assertTrue(
            torch.allclose(
                active[:, mapping.balancing_bus_indices],
                torch.zeros((2, 4)),
                atol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                reactive[:, mapping.balancing_bus_indices],
                torch.zeros((2, 4)),
                atol=1e-4,
            )
        )
        self.assertTrue(torch.all(magnitude >= 0.9))
        self.assertTrue(torch.all(magnitude <= 1.1))
        self.assertTrue(
            torch.allclose(angle[:, self.context.reference_bus_index], torch.zeros(2))
        )
        self.assertTrue(torch.all(extra_active >= active_bounds))
        self.assertTrue(torch.all(extra_active <= active_upper))


class AugmentedLagrangianLossTests(unittest.TestCase):
    """Test lazy multiplier dimensions and projected dual ascent behavior."""

    def test_dual_updates_match_constraint_layout_and_keep_inequalities_nonnegative(
        self,
    ) -> None:
        """Initialize Case 5 multipliers and update them from a violated prediction."""
        context = build_opf_context(case5_pjm())
        metric = OPFAwareMetric(context, device="cpu")
        inputs = torch.zeros((2, len(context.input_columns)))
        predictions = torch.ones((2, len(context.output_columns)), requires_grad=True)
        targets = torch.ones_like(predictions)
        loss_function = AugmentedLagrangianLoss(metric)
        loss = loss_function(inputs, predictions, targets)
        items = get_opf_constraint_items(metric, inputs, predictions)
        loss.backward()
        before = loss_function.equality_multipliers.clone()
        loss_function.update_duals(inputs, predictions.detach(), learning_rate=0.1)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            loss_function.equality_multipliers.shape, (items.equality.shape[1],)
        )
        self.assertEqual(
            loss_function.inequality_multipliers.shape, (items.inequality.shape[1],)
        )
        self.assertFalse(torch.equal(before, loss_function.equality_multipliers))
        self.assertTrue(torch.all(loss_function.inequality_multipliers >= 0.0))
