"""Create OPF evaluation figures without displaying or saving them."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from optimal_power_flow.training.loops import TrainingHistory


def plot_training_mse(history: TrainingHistory, title: str) -> Figure:
    """
    Plot mean training MSE by epoch.

    Parameters
    ----------
    history : optimal_power_flow.training.loops.TrainingHistory
        Per-epoch MSE values returned by a training loop.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Line figure containing training MSE history.

    Raises
    ------
    ValueError
        If no epoch values are available.
    """
    values = history.epoch_mean_squared_errors
    if not values:
        raise ValueError("Training history must contain at least one epoch.")
    figure, axis = plt.subplots(figsize=(5, 2.8))
    axis.plot(range(1, len(values) + 1), values)
    axis.set(title=title, xlabel="Epoch", ylabel="Training MSE")
    figure.tight_layout()
    return figure


def plot_opf_metric_comparison(
    summary: pd.DataFrame,
    metrics: Sequence[str] = (
        "mse",
        "power_mismatch_mean",
        "bound_vio_mean",
        "branch_flow_vio_mean",
    ),
) -> Figure:
    """
    Plot selected OPF-aware metrics as one bar chart per metric.

    Parameters
    ----------
    summary : pandas.DataFrame
        Comparison table with an ``"experiment"`` column and selected metrics.
    metrics : collections.abc.Sequence[str], optional
        Numeric summary columns to plot. Default is MSE, power mismatch, bound
        violation, and branch-flow violation.

    Returns
    -------
    matplotlib.figure.Figure
        Bar-chart figure with one subplot per selected metric.

    Raises
    ------
    ValueError
        If the summary is empty or requested columns are absent.
    """
    if summary.empty:
        raise ValueError("summary must not be empty.")
    required_columns = {"experiment", *metrics}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(
            f"summary is missing plotting columns: {sorted(missing_columns)}."
        )
    if not metrics:
        raise ValueError("metrics must contain at least one column.")

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(4 * len(metrics), 3),
        squeeze=False,
    )
    labels = summary["experiment"].tolist()
    positions = np.arange(len(labels))
    for axis, metric in zip(axes[0], metrics, strict=True):
        values = summary[metric].to_numpy(dtype=float)
        axis.bar(positions, values)
        axis.set(title=metric, ylabel=metric)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    figure.tight_layout()
    return figure
