"""Provide project paths used by Optimal Power Flow experiment workflows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Define repository-relative locations for configuration and runtime artifacts.

    Parameters
    ----------
    project_root : pathlib.Path
        Absolute repository root containing ``src`` and ``experiment.toml``.
    """

    project_root: Path

    @property
    def experiment_config_path(self) -> Path:
        """
        Return the root TOML experiment configuration path.

        Returns
        -------
        pathlib.Path
            Location of ``experiment.toml``.
        """
        return self.project_root / "experiment.toml"

    @property
    def artifact_directory(self) -> Path:
        """
        Return the root directory for generated runtime artifacts.

        Returns
        -------
        pathlib.Path
            Directory intended for downloaded data and experiment outputs.
        """
        return self.project_root / "artifacts"


def load_settings() -> Settings:
    """
    Construct settings for the repository containing this source directory.

    Returns
    -------
    Settings
        Paths resolved relative to the checked-out project.
    """
    return Settings(project_root=Path(__file__).resolve().parent.parent)


settings = load_settings()
