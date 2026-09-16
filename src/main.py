"""Provide the eventual command-line entry point for OPF experiments."""

from config import settings


def main() -> None:
    """
    Confirm that the project foundation and its root configuration are present.

    Full experiment dispatch is introduced after the case, data, model, and
    training workflows are implemented.
    """
    print(f"OPF package foundation is ready at {settings.project_root}")
    print(f"Experiment configuration: {settings.experiment_config_path}")


if __name__ == "__main__":
    main()
