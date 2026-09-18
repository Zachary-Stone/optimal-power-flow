"""Validate differentiable OPF-aware metrics against a five-bus AC-OPF solve."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from pypower import idx_bus, idx_gen

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.evaluation import (
    evaluate_opf_predictions,
    plot_opf_metric_comparison,
    plot_training_mse,
    summarize_opf_results,
)
from optimal_power_flow.models import BaselineMLP
from optimal_power_flow.power import OPFAwareMetric, build_opf_context, solve_ac_opf
from optimal_power_flow.structs import ExperimentConfig, ExperimentResult
from optimal_power_flow.training import TrainingHistory


class OPFAwareMetricTests(unittest.TestCase):
    """Validate physics metrics against the tutorial's solver-generated solution."""

    def setUp(self) -> None:
        """Solve the Case 5 base scenario and encode its canonical tensors."""
        self.context = build_opf_context(case5_pjm(), "PGLib/PJM Case 5")
        result = solve_ac_opf(self.context.case)
        internal_case = self.context.internal_case
        base_mva = float(internal_case["baseMVA"])
        bus = internal_case["bus"]
        inputs = (
            np.concatenate(
                (
                    bus[self.context.load_bus_indices, idx_bus.PD],
                    bus[self.context.load_bus_indices, idx_bus.QD],
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
        self.metric = OPFAwareMetric(self.context)
        self.solver_cost = float(result["f"])

    def test_solver_solution_has_small_physical_residuals(self) -> None:
        """Recover near-zero feasibility violations from the AC-OPF solution."""
        active, reactive = self.metric.power_balance_violation(
            self.inputs, self.targets
        )
        flow, angle = self.metric.branch_flow_violations(self.targets)

        self.assertAlmostEqual(
            self.metric.generator_cost(self.targets).item(),
            self.solver_cost,
            delta=0.01,
        )
        self.assertLess(torch.cat((active, reactive), dim=1).mean().item(), 1e-4)
        self.assertEqual(
            self.metric.reference_angle_violation(self.targets).item(), 0.0
        )
        self.assertEqual(self.metric.bound_violation(self.targets).sum().item(), 0.0)
        self.assertEqual(flow.sum().item(), 0.0)
        self.assertEqual(angle.sum().item(), 0.0)

    def test_infeasible_prediction_has_a_bound_violation(self) -> None:
        """Report positive violations when a generator exceeds its maximum output."""
        infeasible = self.targets.clone()
        infeasible[:, 0] = 10.0

        violation = self.metric.bound_violation(infeasible)

        self.assertGreater(violation[:, 0].item(), 0.0)

    def test_evaluation_and_summary_preserve_opf_metrics(self) -> None:
        """Summarize an exact solver target into a complete OPF comparison row."""
        metrics = evaluate_opf_predictions(
            self.metric,
            self.inputs,
            self.targets,
            self.targets,
            inference_runtime_seconds=0.01,
        )
        result = ExperimentResult(
            config=ExperimentConfig("Solver target"),
            model=BaselineMLP(input_size=6, output_size=20, hidden_layers=(4,)),
            metrics=metrics.as_dict(),
        )

        summary = summarize_opf_results({"solver_target": result})
        figure = plot_opf_metric_comparison(summary)
        history_figure = plot_training_mse(TrainingHistory((1.0, 0.5)), "MSE")

        self.assertEqual(summary.loc[0, "samples"], 1.0)
        self.assertEqual(summary.loc[0, "mse"], 0.0)
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(len(history_figure.axes), 1)
        plt.close(figure)
        plt.close(history_figure)
