"""Define typed configuration and result objects for OPF experiments."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    """
    Identify the electrical-network case used by an experiment.

    Parameters
    ----------
    name : str, optional
        Registered case identifier. Default is ``"case5_pjm"``.
    """

    name: str = "case5_pjm"

    def __post_init__(self) -> None:
        """Validate the case identifier."""
        if not self.name:
            raise ValueError("Case name must not be empty.")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """
    Define an OPF supervised-learning data source and split settings.

    Parameters
    ----------
    source : str, optional
        Registered dataset source. Default is ``"opflearn_case5"``.
    cache_directory : pathlib.Path, optional
        Directory where downloaded source files may be stored. Default is
        ``Path("artifacts/data")``.
    random_seed : int, optional
        Seed for deterministic splitting and sampling. Default is 2026.
    train_fraction : float, optional
        Fraction of rows used for training. Default is 0.8.
    """

    source: str = "opflearn_case5"
    cache_directory: Path = Path("artifacts/data")
    random_seed: int = 2026
    train_fraction: float = 0.8

    def __post_init__(self) -> None:
        """Validate dataset source and split settings."""
        if not self.source:
            raise ValueError("Dataset source must not be empty.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Define common model-training settings.

    Parameters
    ----------
    device : str, optional
        Requested PyTorch device or ``"auto"``. Default is ``"auto"``.
    batch_size : int, optional
        Number of rows per optimization batch. Default is 128.
    epochs : int, optional
        Number of complete training passes. Default is 10.
    learning_rate : float, optional
        Optimizer learning rate. Default is 0.001.
    track_emissions : bool, optional
        Whether training should enable CodeCarbon tracking. Default is False.
    """

    device: str = "auto"
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    track_emissions: bool = False

    def __post_init__(self) -> None:
        """Validate common training settings."""
        if not self.device:
            raise ValueError("device must not be empty.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Define a named neural-surrogate architecture configuration.

    Parameters
    ----------
    name : str, optional
        Registered model workflow identifier. Default is ``"baseline_mlp"``.
    hidden_layers : tuple[int, ...], optional
        Width of each hidden fully connected layer. Default is ``(128, 64)``.
    """

    name: str = "baseline_mlp"
    hidden_layers: tuple[int, ...] = (128, 64)

    def __post_init__(self) -> None:
        """Validate the model identifier and hidden-layer widths."""
        if not self.name:
            raise ValueError("Model name must not be empty.")
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Combine all inputs needed to run one named OPF experiment.

    Parameters
    ----------
    name : str
        Human-readable experiment identifier.
    case : CaseConfig, optional
        Electrical-network case selection. Default is a five-bus PJM case.
    dataset : DatasetConfig, optional
        Data acquisition and splitting settings. Default is OPFLearn Case 5.
    training : TrainingConfig, optional
        Common model-training settings. Default is tutorial-sized training.
    model : ModelConfig, optional
        Surrogate-model settings. Default is the baseline MLP.
    """

    name: str
    case: CaseConfig = field(default_factory=CaseConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        """Validate the experiment identifier."""
        if not self.name:
            raise ValueError("Experiment name must not be empty.")


@dataclass(slots=True)
class ExperimentResult[ModelT]:
    """
    Store model, numerical metrics, and artifacts from a completed experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration used to run the experiment.
    model : ModelT or None
        Trained model or ``None`` for a solver-only workflow.
    metrics : collections.abc.Mapping[str, float], optional
        Scalar metrics indexed by stable metric names. Default is empty.
    artifact_paths : tuple[pathlib.Path, ...], optional
        Saved output paths associated with this run. Default is empty.
    metadata : collections.abc.Mapping[str, object], optional
        Additional serializable workflow metadata. Default is empty.
    """

    config: ExperimentConfig
    model: ModelT | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifact_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
