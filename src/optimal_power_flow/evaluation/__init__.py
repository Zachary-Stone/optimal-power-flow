"""Experiment summaries, figures, and carbon-aware comparisons."""

from optimal_power_flow.evaluation.carbon import (
    CarbonOPFComparison,
    carbon_priced_case,
    compare_carbon_price_opf,
    generator_emissions,
    original_generation_cost,
)
from optimal_power_flow.evaluation.metrics import (
    OPFEvaluationMetrics,
    evaluate_opf_model,
    evaluate_opf_predictions,
)
from optimal_power_flow.evaluation.plotting import (
    plot_opf_metric_comparison,
    plot_training_mse,
)
from optimal_power_flow.evaluation.summaries import (
    count_trainable_parameters,
    summarize_opf_results,
    summarize_regression_results,
)

__all__ = [
    "OPFEvaluationMetrics",
    "CarbonOPFComparison",
    "carbon_priced_case",
    "count_trainable_parameters",
    "compare_carbon_price_opf",
    "evaluate_opf_model",
    "evaluate_opf_predictions",
    "generator_emissions",
    "original_generation_cost",
    "plot_opf_metric_comparison",
    "plot_training_mse",
    "summarize_opf_results",
    "summarize_regression_results",
]
