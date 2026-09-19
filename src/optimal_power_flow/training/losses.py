"""Define physics-aware training objectives for AC-OPF surrogate models."""

from dataclasses import dataclass

import torch
from torch import nn

from optimal_power_flow.power.metrics import OPFAwareMetric


@dataclass(frozen=True, slots=True)
class ConstraintItems:
    """
    Store equality residuals and non-negative inequality violations.

    Parameters
    ----------
    equality : torch.Tensor
        Signed active/reactive balance and reference-angle residual items.
    inequality : torch.Tensor
        Non-negative output-bound, branch-flow, and branch-angle violations.
    """

    equality: torch.Tensor
    inequality: torch.Tensor


def get_opf_constraint_items(
    metric: OPFAwareMetric, inputs: torch.Tensor, predictions: torch.Tensor
) -> ConstraintItems:
    """
    Assemble common equality and inequality items for OPF-aware losses.

    Parameters
    ----------
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Case-specific differentiable physical metric object.
    inputs : torch.Tensor
        Batched canonical active and reactive load features.
    predictions : torch.Tensor
        Batched canonical AC-OPF target predictions.

    Returns
    -------
    ConstraintItems
        Concatenated signed equality residuals and non-negative inequality
        violations for every prediction row.
    """
    active_residual, reactive_residual = metric.power_balance_residual(
        inputs, predictions
    )
    reference_residual = metric.reference_angle_residual(predictions)
    equality = torch.cat(
        (active_residual, reactive_residual, reference_residual), dim=1
    )
    bound_violation = metric.bound_violation(predictions)
    flow_violation, angle_violation = metric.branch_flow_violations(predictions)
    inequality = torch.cat((bound_violation, flow_violation, angle_violation), dim=1)
    return ConstraintItems(equality=equality, inequality=inequality)


def quadratic_constraint_penalties(
    items: ConstraintItems,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean squared equality and inequality penalty terms.

    Parameters
    ----------
    items : ConstraintItems
        Equality residuals and inequality violations for one prediction batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scalar equality and inequality quadratic penalties.
    """
    return torch.mean(torch.square(items.equality)), torch.mean(
        torch.square(items.inequality)
    )


class OPFPenaltyLoss(nn.Module):
    """
    Combine supervised MSE with fixed quadratic AC-OPF constraint penalties.

    Parameters
    ----------
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Case-specific differentiable physical metric object.
    equality_weight : float, optional
        Weight for squared equality residuals. Default is 0.01.
    inequality_weight : float, optional
        Weight for squared inequality violations. Default is 0.01.

    Raises
    ------
    ValueError
        If either weight is negative.
    """

    def __init__(
        self,
        metric: OPFAwareMetric,
        equality_weight: float = 0.01,
        inequality_weight: float = 0.01,
    ) -> None:
        """Initialize fixed penalty weights and the physical metric reference."""
        super().__init__()
        if equality_weight < 0.0 or inequality_weight < 0.0:
            raise ValueError("Constraint penalty weights must be non-negative.")
        self.metric = metric
        self.equality_weight = equality_weight
        self.inequality_weight = inequality_weight

    def forward(
        self, inputs: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate supervised MSE plus weighted physical constraint penalties.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched canonical active and reactive load features.
        predictions : torch.Tensor
            Batched canonical target predictions.
        targets : torch.Tensor
            Batched solver-generated canonical targets.

        Returns
        -------
        torch.Tensor
            Scalar differentiable combined training loss.
        """
        mean_squared_error = torch.mean(torch.square(predictions - targets))
        items = get_opf_constraint_items(self.metric, inputs, predictions)
        equality_penalty, inequality_penalty = quadratic_constraint_penalties(items)
        return (
            mean_squared_error
            + self.equality_weight * equality_penalty
            + self.inequality_weight * inequality_penalty
        )


class SelfSupervisedOPFLoss(nn.Module):
    """
    Train from AC-OPF objective and feasibility terms without label MSE.

    Parameters
    ----------
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Case-specific differentiable physical metric object.
    equality_weight : float, optional
        Weight for squared equality residuals. Default is 1.0.
    inequality_weight : float, optional
        Weight for squared inequality violations. Default is 1.0.
    objective_weight : float, optional
        Weight for mean generation cost. Default is 1e-5.

    Raises
    ------
    ValueError
        If any weight is negative.
    """

    def __init__(
        self,
        metric: OPFAwareMetric,
        equality_weight: float = 1.0,
        inequality_weight: float = 1.0,
        objective_weight: float = 1e-5,
    ) -> None:
        """Initialize objective and physical constraint weights."""
        super().__init__()
        if min(equality_weight, inequality_weight, objective_weight) < 0.0:
            raise ValueError("Self-supervised loss weights must be non-negative.")
        self.metric = metric
        self.equality_weight = equality_weight
        self.inequality_weight = inequality_weight
        self.objective_weight = objective_weight

    def forward(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate objective and feasibility loss without consuming target labels.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched canonical active and reactive load features.
        predictions : torch.Tensor
            Batched canonical target predictions.
        targets : torch.Tensor or None, optional
            Ignored supervised targets retained for shared training-loop
            compatibility. Default is None.

        Returns
        -------
        torch.Tensor
            Scalar differentiable self-supervised AC-OPF loss.
        """
        del targets
        items = get_opf_constraint_items(self.metric, inputs, predictions)
        equality_penalty, inequality_penalty = quadratic_constraint_penalties(items)
        objective = torch.mean(self.metric.generator_cost(predictions))
        return (
            self.equality_weight * equality_penalty
            + self.inequality_weight * inequality_penalty
            + self.objective_weight * objective
        )
