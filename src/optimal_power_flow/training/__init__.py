"""Model-training loops, objectives, and experiment workflows."""

from optimal_power_flow.training.emissions import EmissionsMeasurement, track_emissions
from optimal_power_flow.training.loops import (
    RegressionMetrics,
    TrainingHistory,
    evaluate_mse_regression,
    resolve_device,
    train_mse_regression,
    train_regression,
)
from optimal_power_flow.training.losses import (
    ConstraintItems,
    OPFPenaltyLoss,
    SelfSupervisedOPFLoss,
    get_opf_constraint_items,
    quadratic_constraint_penalties,
)
from optimal_power_flow.training.workflows import (
    run_baseline_mlp,
    run_bounded_penalty_mlp,
    run_bounded_self_supervised_mlp,
)

__all__ = [
    "RegressionMetrics",
    "TrainingHistory",
    "EmissionsMeasurement",
    "ConstraintItems",
    "OPFPenaltyLoss",
    "SelfSupervisedOPFLoss",
    "evaluate_mse_regression",
    "resolve_device",
    "get_opf_constraint_items",
    "quadratic_constraint_penalties",
    "run_baseline_mlp",
    "run_bounded_penalty_mlp",
    "run_bounded_self_supervised_mlp",
    "track_emissions",
    "train_regression",
    "train_mse_regression",
]
