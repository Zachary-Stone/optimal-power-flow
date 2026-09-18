"""Assemble configured baseline OPF surrogate-model workflows."""

from optimal_power_flow.dataset_io.datasets import OPFDataLoaders
from optimal_power_flow.models.mlp import BaselineMLP
from optimal_power_flow.structs.experiments import ExperimentConfig, ExperimentResult
from optimal_power_flow.training.loops import (
    evaluate_mse_regression,
    resolve_device,
    train_mse_regression,
)


def run_baseline_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
) -> ExperimentResult[BaselineMLP]:
    """
    Train and evaluate the configured baseline MLP on canonical OPF data.

    Parameters
    ----------
    config : optimal_power_flow.structs.experiments.ExperimentConfig
        Case, model, and training settings for this experiment.
    data_loaders : optimal_power_flow.dataset_io.datasets.OPFDataLoaders
        Deterministic training and test loaders.
    input_columns : tuple[str, ...]
        Canonical input-column names whose count sets the model input size.
    output_columns : tuple[str, ...]
        Canonical output-column names whose count sets the model output size.

    Returns
    -------
    optimal_power_flow.structs.experiments.ExperimentResult
        Trained MLP, test-set MSE, sample count, and epoch-loss history.

    Raises
    ------
    ValueError
        If the selected model is not the baseline MLP or column groups are
        empty.
    """
    if config.model.name != "baseline_mlp":
        raise ValueError(
            "run_baseline_mlp requires config.model.name to be 'baseline_mlp'."
        )
    if not input_columns or not output_columns:
        raise ValueError("Baseline MLP requires non-empty input and output columns.")

    device = resolve_device(config.training.device)
    model = BaselineMLP(
        input_size=len(input_columns),
        output_size=len(output_columns),
        hidden_layers=config.model.hidden_layers,
    )
    history = train_mse_regression(
        training_loader=data_loaders.training,
        model=model,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        device=device,
        track_energy=config.training.track_emissions,
        emissions_project_name=f"opf_{config.name}",
    )
    metrics = evaluate_mse_regression(data_loaders.test, model, device)
    return ExperimentResult(
        config=config,
        model=model,
        metrics={
            "samples": float(metrics.sample_count),
            "mse": metrics.mean_squared_error,
            "train_runtime_s": history.training_runtime_seconds,
        },
        metadata={
            "device": str(device),
            "training_mse": history.epoch_mean_squared_errors,
            "energy_kwh": history.energy_kwh,
            "emissions_kg_co2eq": history.emissions_kg_co2eq,
        },
    )
