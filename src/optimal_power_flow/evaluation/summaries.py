"""Summarize baseline regression metrics before OPF-aware evaluation is added."""

from collections.abc import Mapping

import pandas as pd

from optimal_power_flow.structs.experiments import ExperimentResult


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
