"""Summarize baseline regression metrics before OPF-aware evaluation is added."""

from collections.abc import Mapping

import pandas as pd
from torch import nn

from optimal_power_flow.structs.experiments import ExperimentResult

OPF_SUMMARY_COLUMNS = (
    "experiment",
    "model_size_params",
    "samples",
    "mse",
    "cost_gap_pct",
    "power_mismatch_mean",
    "ref_angle_vio_mean",
    "bound_vio_mean",
    "branch_flow_vio_mean",
    "branch_angle_vio_mean",
    "runtime_s",
    "train_runtime_s",
    "energy_kwh",
    "emissions_kg_co2eq",
)


def summarize_regression_results(
    results: Mapping[str, ExperimentResult[object]],
) -> pd.DataFrame:
    """
    Create one comparable MSE summary row for each completed experiment.

    Parameters
    ----------
    results : collections.abc.Mapping[str, ExperimentResult]
        Completed experiments keyed by stable display name.

    Returns
    -------
    pandas.DataFrame
        Table with ``experiment``, ``samples``, and ``mse`` columns.

    Raises
    ------
    ValueError
        If no results are supplied or a result has no baseline regression MSE.
    """
    if not results:
        raise ValueError("results must not be empty.")
    rows = []
    for name, result in results.items():
        try:
            sample_count = int(result.metrics["samples"])
            mean_squared_error = float(result.metrics["mse"])
        except KeyError as error:
            raise ValueError(
                f"Result {name!r} does not include baseline regression metrics."
            ) from error
        rows.append(
            {
                "experiment": name,
                "samples": sample_count,
                "mse": mean_squared_error,
            }
        )
    return pd.DataFrame(rows, columns=["experiment", "samples", "mse"])


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the trainable parameters in one PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose trainable parameters are counted.

    Returns
    -------
    int
        Number of parameter values with ``requires_grad=True``.
    """
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def summarize_opf_results(
    results: Mapping[str, ExperimentResult[object]],
) -> pd.DataFrame:
    """
    Create a stable, rounded comparison table for OPF-aware experiment results.

    Parameters
    ----------
    results : collections.abc.Mapping[str, ExperimentResult]
        Completed model experiments keyed by display name.

    Returns
    -------
    pandas.DataFrame
        One OPF-aware metrics row per experiment, with unavailable optional
        runtime and emissions fields represented by missing values.

    Raises
    ------
    ValueError
        If no results are supplied or a result lacks required OPF metrics.
    """
    if not results:
        raise ValueError("results must not be empty.")
    required_metrics = OPF_SUMMARY_COLUMNS[2:10]
    rows = []
    for name, result in results.items():
        missing_metrics = set(required_metrics).difference(result.metrics)
        if missing_metrics:
            raise ValueError(
                f"Result {name!r} is missing OPF metrics: {sorted(missing_metrics)}."
            )
        model_size = (
            count_trainable_parameters(result.model)
            if isinstance(result.model, nn.Module)
            else None
        )
        row = {
            "experiment": name,
            "model_size_params": model_size,
            **{metric: float(result.metrics[metric]) for metric in required_metrics},
            "runtime_s": result.metrics.get("runtime_s"),
            "train_runtime_s": result.metrics.get("train_runtime_s"),
            "energy_kwh": result.metadata.get("energy_kwh"),
            "emissions_kg_co2eq": result.metadata.get("emissions_kg_co2eq"),
        }
        rows.append(row)
    summary = pd.DataFrame(rows, columns=OPF_SUMMARY_COLUMNS)
    return summary.round(
        {
            "model_size_params": 0,
            "mse": 5,
            "cost_gap_pct": 3,
            "power_mismatch_mean": 5,
            "ref_angle_vio_mean": 5,
            "bound_vio_mean": 5,
            "branch_flow_vio_mean": 5,
            "branch_angle_vio_mean": 5,
            "runtime_s": 3,
            "train_runtime_s": 3,
            "energy_kwh": 6,
            "emissions_kg_co2eq": 6,
        }
    )
