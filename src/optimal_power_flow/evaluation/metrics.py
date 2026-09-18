"""Convert physical AC-OPF quantities into comparable evaluation metrics."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from optimal_power_flow.power.metrics import OPFAwareMetric

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class OPFEvaluationMetrics:
    """
    Store accuracy, feasibility, objective, and timing metrics for one result.

    Parameters
    ----------
    sample_count : int
        Number of evaluated OPF scenarios.
    mean_squared_error : float
        Mean squared error across all target values.
    cost_gap_percent : float
        Mean signed generator-cost gap relative to target solutions, in percent.
    power_mismatch_mean : float
        Mean absolute active/reactive power-balance mismatch in per unit.
    reference_angle_violation_mean : float
        Mean absolute reference-angle deviation in radians.
    bound_violation_mean : float
        Mean target-bound violation in native per-unit/radian units.
    branch_flow_violation_mean : float
        Mean branch apparent-power violation in per unit.
    branch_angle_violation_mean : float
        Mean branch angle-difference violation in radians.
    inference_runtime_seconds : float or None, optional
        Wall-clock batched inference duration. Default is None.
    """

    sample_count: int
    mean_squared_error: float
    cost_gap_percent: float
    power_mismatch_mean: float
    reference_angle_violation_mean: float
    bound_violation_mean: float
    branch_flow_violation_mean: float
    branch_angle_violation_mean: float
    inference_runtime_seconds: float | None = None

    def as_dict(self) -> dict[str, float]:
        """
        Return stable metric names suitable for experiment result metadata.

        Returns
        -------
        dict[str, float]
            Scalar metrics using the notebook's comparison-table column names.
        """
        metrics = {
            "samples": float(self.sample_count),
            "mse": self.mean_squared_error,
            "cost_gap_pct": self.cost_gap_percent,
            "power_mismatch_mean": self.power_mismatch_mean,
            "ref_angle_vio_mean": self.reference_angle_violation_mean,
            "bound_vio_mean": self.bound_violation_mean,
            "branch_flow_vio_mean": self.branch_flow_violation_mean,
            "branch_angle_vio_mean": self.branch_angle_violation_mean,
        }
        if self.inference_runtime_seconds is not None:
            metrics["runtime_s"] = self.inference_runtime_seconds
        return metrics


def evaluate_opf_predictions(
    metric: OPFAwareMetric,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    inference_runtime_seconds: float | None = None,
) -> OPFEvaluationMetrics:
    """
    Evaluate AC-OPF accuracy, cost, equality residuals, and inequality limits.

    Parameters
    ----------
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Differentiable case-specific physical metric object.
    inputs : torch.Tensor
        Batched canonical active then reactive load features.
    targets : torch.Tensor
        Batched solver-generated canonical targets.
    predictions : torch.Tensor
        Batched model-predicted canonical targets.
    inference_runtime_seconds : float or None, optional
        External batched inference duration. Default is None.

    Returns
    -------
    OPFEvaluationMetrics
        Scalar accuracy, objective, feasibility, and timing metrics.

    Raises
    ------
    ValueError
        If the supplied batch sizes do not match.
    """
    if len(inputs) != len(targets) or len(targets) != len(predictions):
        raise ValueError("inputs, targets, and predictions must share batch size.")
    if len(inputs) == 0:
        raise ValueError("OPF evaluation requires at least one scenario.")

    device_inputs = inputs.to(metric.device)
    device_targets = targets.to(metric.device)
    device_predictions = predictions.to(metric.device)
    predicted_cost = metric.generator_cost(device_predictions)
    target_cost = metric.generator_cost(device_targets)
    active_mismatch, reactive_mismatch = metric.power_balance_violation(
        device_inputs, device_predictions
    )
    branch_flow, branch_angle = metric.branch_flow_violations(device_predictions)
    relative_cost_gap = (predicted_cost - target_cost) / target_cost
    return OPFEvaluationMetrics(
        sample_count=len(device_inputs),
        mean_squared_error=float(
            torch.mean(torch.square(device_predictions - device_targets)).detach().cpu()
        ),
        cost_gap_percent=float((100.0 * torch.mean(relative_cost_gap)).detach().cpu()),
        power_mismatch_mean=float(
            torch.cat((active_mismatch, reactive_mismatch), dim=1).mean().detach().cpu()
        ),
        reference_angle_violation_mean=float(
            metric.reference_angle_violation(device_predictions).mean().detach().cpu()
        ),
        bound_violation_mean=float(
            metric.bound_violation(device_predictions).mean().detach().cpu()
        ),
        branch_flow_violation_mean=float(branch_flow.mean().detach().cpu()),
        branch_angle_violation_mean=float(branch_angle.mean().detach().cpu()),
        inference_runtime_seconds=inference_runtime_seconds,
    )


def evaluate_opf_model(
    metric: OPFAwareMetric, model: nn.Module, data_loader: DataLoader[Batch]
) -> OPFEvaluationMetrics:
    """
    Run batched model inference and evaluate all predictions with AC-OPF metrics.

    Parameters
    ----------
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Differentiable case-specific physical metric object.
    model : torch.nn.Module
        Trained model mapping canonical inputs to canonical targets.
    data_loader : torch.utils.data.DataLoader
        Ordered feature-target batches for full-dataset evaluation.

    Returns
    -------
    OPFEvaluationMetrics
        Full-dataset physical evaluation with batched inference runtime.

    Raises
    ------
    ValueError
        If no batches are produced by the data loader.
    """
    model.to(metric.device)
    model.eval()
    all_inputs = []
    all_targets = []
    all_predictions = []
    started_at = perf_counter()
    with torch.no_grad():
        for inputs, targets in data_loader:
            device_inputs = inputs.to(metric.device)
            all_inputs.append(device_inputs)
            all_targets.append(targets.to(metric.device))
            all_predictions.append(model(device_inputs))
    if not all_inputs:
        raise ValueError("data_loader produced no batches.")
    runtime_seconds = perf_counter() - started_at
    return evaluate_opf_predictions(
        metric,
        torch.cat(all_inputs),
        torch.cat(all_targets),
        torch.cat(all_predictions),
        inference_runtime_seconds=runtime_seconds,
    )
