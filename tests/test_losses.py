"""Validate constraint-aware training objectives on a solved OPF scenario."""

import unittest

import numpy as np
import torch
from pypower import idx_bus, idx_gen

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.power import OPFAwareMetric, build_opf_context, solve_ac_opf
from optimal_power_flow.training.losses import (
    OPFPenaltyLoss,
    SelfSupervisedOPFLoss,
    get_opf_constraint_items,
    quadratic_constraint_penalties,
)


class ConstraintLossTests(unittest.TestCase):
    """Validate OPF penalty components and their differentiable losses."""

    def setUp(self) -> None:
        """Construct canonical input and target tensors from a Case 5 solve."""
        context = build_opf_context(case5_pjm())
        result = solve_ac_opf(context.case)
        case = context.internal_case
        base_mva = float(case["baseMVA"])
        inputs = (
            np.concatenate(
                (
                    case["bus"][context.load_bus_indices, idx_bus.PD],
                    case["bus"][context.load_bus_indices, idx_bus.QD],
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
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        self.targets = torch.tensor(outputs, dtype=torch.float32).unsqueeze(0)
        self.metric = OPFAwareMetric(context)

    def test_constraint_items_match_case_dimensions(self) -> None:
        """Build signed equalities and non-negative inequalities from predictions."""
        items = get_opf_constraint_items(self.metric, self.inputs, self.targets)
        equality_penalty, inequality_penalty = quadratic_constraint_penalties(items)

        self.assertEqual(items.equality.shape, (1, 11))
        self.assertEqual(items.inequality.shape, (1, 32))
        self.assertLess(equality_penalty.item(), 1e-8)
        self.assertEqual(inequality_penalty.item(), 0.0)

    def test_constraint_losses_are_finite_and_differentiable(self) -> None:
        """Backpropagate both supervised-penalty and label-free OPF objectives."""
        predictions = (self.targets + 0.01).detach().requires_grad_()
        penalty_loss = OPFPenaltyLoss(self.metric)(
            self.inputs, predictions, self.targets
        )
        penalty_loss.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(penalty_loss))
        self.assertIsNotNone(predictions.grad)

        predictions.grad.zero_()
        self_supervised_loss = SelfSupervisedOPFLoss(self.metric)(
            self.inputs, predictions
        )
        self_supervised_loss.backward()
        self.assertTrue(torch.isfinite(self_supervised_loss))
        self.assertIsNotNone(predictions.grad)
