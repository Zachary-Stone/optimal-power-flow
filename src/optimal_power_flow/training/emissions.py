"""Optionally measure training energy and emissions with CodeCarbon."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from codecarbon import EmissionsTracker


@dataclass(slots=True)
class EmissionsMeasurement:
    """
    Store optional energy and emissions from one measured training section.

    Parameters
    ----------
    energy_kwh : float or None, optional
        Measured energy consumption in kilowatt-hours. Default is None.
    emissions_kg_co2eq : float or None, optional
        Measured carbon dioxide equivalent in kilograms. Default is None.
    """

    energy_kwh: float | None = None
    emissions_kg_co2eq: float | None = None


@contextmanager
def track_emissions(
    enabled: bool, project_name: str
) -> Generator[EmissionsMeasurement, None, None]:
    """
    Measure one training section only when CodeCarbon tracking is enabled.

    Parameters
    ----------
    enabled : bool
        Whether a CodeCarbon tracker should be created and started.
    project_name : str
        In-memory CodeCarbon project label.

    Yields
    ------
    EmissionsMeasurement
        Mutable measurement populated after the wrapped block completes.
    """
    measurement = EmissionsMeasurement()
    if not enabled:
        yield measurement
        return

    tracker = EmissionsTracker(
        project_name=project_name,
        save_to_file=False,
        log_level="error",
    )
    tracker.start()
    try:
        yield measurement
    finally:
        tracker.stop()
        final_data = tracker.final_emissions_data
        measurement.energy_kwh = float(final_data.energy_consumed)
        measurement.emissions_kg_co2eq = float(final_data.emissions)
