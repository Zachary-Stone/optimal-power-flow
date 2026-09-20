"""Assemble configured baseline OPF surrogate-model workflows."""

from torch import nn

from optimal_power_flow.dataset_io.datasets import OPFDataLoaders
from optimal_power_flow.models.graph import TopologyGNN, build_graph_features
from optimal_power_flow.models.mlp import BaselineMLP
from optimal_power_flow.models.physics import PhysicalCompletionMLP
from optimal_power_flow.models.residual import BoundedResidualMLP
from optimal_power_flow.models.transformer import BusLevelTransformer
from optimal_power_flow.power.metrics import OPFAwareMetric
from optimal_power_flow.structs.experiments import ExperimentConfig, ExperimentResult
from optimal_power_flow.training.loops import (
    evaluate_mse_regression,
    resolve_device,
    train_augmented_lagrangian_regression,
    train_mse_regression,
    train_regression,
)
from optimal_power_flow.training.losses import (
    AugmentedLagrangianLoss,
    OPFPenaltyLoss,
    SelfSupervisedOPFLoss,
)


def _validate_metric_device(metric: OPFAwareMetric, device: object) -> None:
    """Ensure a physics metric and model workflow operate on one device."""
    if metric.device != device:
        raise ValueError(
            "OPF metric device must match the configured training device: "
            f"{metric.device} != {device}."
        )


def _run_mse_model(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    model: nn.Module,
    training_signal: str,
) -> ExperimentResult[nn.Module]:
    """Train a supplied architecture with MSE and package a standard result."""
    device = resolve_device(config.training.device)
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
            "training_signal": training_signal,
            "energy_kwh": history.energy_kwh,
            "emissions_kg_co2eq": history.emissions_kg_co2eq,
        },
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


def _run_bounded_constraint_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
    loss_function: OPFPenaltyLoss | SelfSupervisedOPFLoss,
    training_signal: str,
) -> ExperimentResult[BoundedResidualMLP]:
    """
    Train one bounded residual MLP with a supplied constraint-aware objective.

    Parameters
    ----------
    config : optimal_power_flow.structs.experiments.ExperimentConfig
        Model, loss, and training settings.
    data_loaders : optimal_power_flow.dataset_io.datasets.OPFDataLoaders
        Deterministic training and test loaders.
    input_columns : tuple[str, ...]
        Canonical input-column names.
    output_columns : tuple[str, ...]
        Canonical output-column names.
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Physical metric sharing the case-specific output bounds.
    loss_function : OPFPenaltyLoss or SelfSupervisedOPFLoss
        Differentiable three-argument training objective.
    training_signal : str
        Human-readable objective identifier retained in result metadata.

    Returns
    -------
    optimal_power_flow.structs.experiments.ExperimentResult
        Bounded trained model and its test MSE, runtime, and loss history.

    Raises
    ------
    ValueError
        If configured columns are empty or the metric uses a different device.
    """
    if not input_columns or not output_columns:
        raise ValueError("Constraint-aware MLP requires non-empty columns.")
    device = resolve_device(config.training.device)
    if metric.device != device:
        raise ValueError(
            "OPF metric device must match the configured training device: "
            f"{metric.device} != {device}."
        )
    model = BoundedResidualMLP(
        input_size=len(input_columns),
        output_size=len(output_columns),
        lower_bounds=metric.output_minimum,
        upper_bounds=metric.output_maximum,
        width=config.model.residual_width,
        depth=config.model.residual_depth,
    )
    history = train_regression(
        training_loader=data_loaders.training,
        model=model,
        loss_function=loss_function,
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
            "training_loss": history.epoch_mean_squared_errors,
            "training_signal": training_signal,
            "energy_kwh": history.energy_kwh,
            "emissions_kg_co2eq": history.emissions_kg_co2eq,
        },
    )


def run_bounded_penalty_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[BoundedResidualMLP]:
    """
    Train a bounded residual MLP with supervised MSE and OPF penalties.

    Parameters
    ----------
    config : optimal_power_flow.structs.experiments.ExperimentConfig
        Config whose model name is ``"bounded_penalty_mlp"``.
    data_loaders : optimal_power_flow.dataset_io.datasets.OPFDataLoaders
        Deterministic training and test loaders.
    input_columns : tuple[str, ...]
        Canonical input-column names.
    output_columns : tuple[str, ...]
        Canonical output-column names.
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Case-specific physical metric and output bounds.

    Returns
    -------
    optimal_power_flow.structs.experiments.ExperimentResult
        Completed bounded penalty-model result.

    Raises
    ------
    ValueError
        If the configured model name does not select this workflow.
    """
    if config.model.name != "bounded_penalty_mlp":
        raise ValueError(
            "run_bounded_penalty_mlp requires model name 'bounded_penalty_mlp'."
        )
    loss_function = OPFPenaltyLoss(
        metric,
        equality_weight=config.loss.equality_weight,
        inequality_weight=config.loss.inequality_weight,
    )
    return _run_bounded_constraint_mlp(
        config,
        data_loaders,
        input_columns,
        output_columns,
        metric,
        loss_function,
        training_signal="mse_plus_opf_penalties",
    )


def run_bounded_self_supervised_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[BoundedResidualMLP]:
    """
    Train a bounded residual MLP from OPF objective and feasibility terms.

    Parameters
    ----------
    config : optimal_power_flow.structs.experiments.ExperimentConfig
        Config whose model name is ``"bounded_self_supervised_mlp"``.
    data_loaders : optimal_power_flow.dataset_io.datasets.OPFDataLoaders
        Deterministic training and test loaders.
    input_columns : tuple[str, ...]
        Canonical input-column names.
    output_columns : tuple[str, ...]
        Canonical output-column names retained for held-out MSE evaluation.
    metric : optimal_power_flow.power.metrics.OPFAwareMetric
        Case-specific physical metric and output bounds.

    Returns
    -------
    optimal_power_flow.structs.experiments.ExperimentResult
        Completed bounded self-supervised-model result.

    Raises
    ------
    ValueError
        If the configured model name does not select this workflow.
    """
    if config.model.name != "bounded_self_supervised_mlp":
        raise ValueError(
            "run_bounded_self_supervised_mlp requires model name "
            "'bounded_self_supervised_mlp'."
        )
    loss_function = SelfSupervisedOPFLoss(
        metric,
        equality_weight=config.loss.equality_weight,
        inequality_weight=config.loss.inequality_weight,
        objective_weight=config.loss.objective_weight,
    )
    return _run_bounded_constraint_mlp(
        config,
        data_loaders,
        input_columns,
        output_columns,
        metric,
        loss_function,
        training_signal="opf_objective_and_constraints",
    )


def run_topology_gnn(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[TopologyGNN]:
    """Train a graph-convolutional surrogate over the case bus topology."""
    if config.model.name != "topology_gnn":
        raise ValueError("run_topology_gnn requires model name 'topology_gnn'.")
    if not input_columns or not output_columns:
        raise ValueError("Topology GNN requires non-empty columns.")
    device = resolve_device(config.training.device)
    _validate_metric_device(metric, device)
    model = TopologyGNN(
        build_graph_features(metric.context),
        output_size=len(output_columns),
        width=config.model.topology_width,
        depth=config.model.topology_depth,
    )
    return _run_mse_model(config, data_loaders, model, "topology_message_passing")


def run_bus_transformer(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[BusLevelTransformer]:
    """Train a Transformer surrogate over learned bus-position embeddings."""
    if config.model.name != "bus_transformer":
        raise ValueError("run_bus_transformer requires model name 'bus_transformer'.")
    if not input_columns or not output_columns:
        raise ValueError("Bus Transformer requires non-empty columns.")
    device = resolve_device(config.training.device)
    _validate_metric_device(metric, device)
    model = BusLevelTransformer(
        build_graph_features(metric.context),
        output_size=len(output_columns),
        width=config.model.transformer_width,
        heads=config.model.transformer_heads,
        depth=config.model.transformer_depth,
    )
    return _run_mse_model(config, data_loaders, model, "bus_attention")


def run_physical_completion_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[PhysicalCompletionMLP]:
    """Train a bounded voltage model with physically completed generator powers."""
    if config.model.name != "physical_completion_mlp":
        raise ValueError(
            "run_physical_completion_mlp requires model name 'physical_completion_mlp'."
        )
    if not input_columns or not output_columns:
        raise ValueError("Physical-completion MLP requires non-empty columns.")
    device = resolve_device(config.training.device)
    _validate_metric_device(metric, device)
    model = PhysicalCompletionMLP(
        metric,
        input_size=len(input_columns),
        width=config.model.residual_width,
        depth=config.model.residual_depth,
    )
    return _run_mse_model(config, data_loaders, model, "physical_generator_completion")


def run_augmented_lagrangian_mlp(
    config: ExperimentConfig,
    data_loaders: OPFDataLoaders,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    metric: OPFAwareMetric,
) -> ExperimentResult[BoundedResidualMLP]:
    """Train a bounded MLP with adaptive AC-OPF dual multipliers."""
    if config.model.name != "augmented_lagrangian_mlp":
        raise ValueError(
            "run_augmented_lagrangian_mlp requires model name "
            "'augmented_lagrangian_mlp'."
        )
    if not input_columns or not output_columns:
        raise ValueError("Augmented-Lagrangian MLP requires non-empty columns.")
    device = resolve_device(config.training.device)
    _validate_metric_device(metric, device)
    model = BoundedResidualMLP(
        input_size=len(input_columns),
        output_size=len(output_columns),
        lower_bounds=metric.output_minimum,
        upper_bounds=metric.output_maximum,
        width=config.model.residual_width,
        depth=config.model.residual_depth,
    )
    loss_function = AugmentedLagrangianLoss(
        metric,
        equality_penalty=config.loss.augmented_equality_penalty,
        inequality_penalty=config.loss.augmented_inequality_penalty,
    )
    history = train_augmented_lagrangian_regression(
        data_loaders.training,
        model,
        loss_function,
        config.training.epochs,
        config.training.learning_rate,
        config.loss.dual_learning_rate,
        device,
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
            "training_loss": history.epoch_mean_squared_errors,
            "training_signal": "adaptive_augmented_lagrangian",
            "equality_multipliers": tuple(loss_function.equality_multipliers.tolist()),
            "inequality_multipliers": tuple(
                loss_function.inequality_multipliers.tolist()
            ),
            "energy_kwh": history.energy_kwh,
            "emissions_kg_co2eq": history.emissions_kg_co2eq,
        },
    )
