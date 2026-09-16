"""Acquire and standardize the OPFLearn/PGLib five-bus dataset."""

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from optimal_power_flow.power.context import OPFContext

OPFLEARN_CASE5_FILENAME = "pglib_opf_case5_pjm.csv"
OPFLEARN_CASE5_URL = "https://data.nlr.gov/system/files/177/pglib_opf_case5_pjm.csv"
EXPECTED_FIRST_COLUMN = "load1:pl"
EXPECTED_FIRST_VOLTAGE_COLUMN = "bus1:v_bus"


def opflearn_case5_path(cache_directory: Path) -> Path:
    """
    Return the expected cache path for the OPFLearn five-bus CSV.

    Parameters
    ----------
    cache_directory : pathlib.Path
        Runtime directory used for downloaded source data.

    Returns
    -------
    pathlib.Path
        Expected path of the cached OPFLearn Case 5 CSV.
    """
    return cache_directory / OPFLEARN_CASE5_FILENAME


def validate_opflearn_case5_layout(columns: pd.Index) -> None:
    """
    Validate the fixed column layout required by this tutorial's adapter.

    This is deliberately not a general OPFLearn or PGLib schema validator. It
    ensures the source has the 26 columns needed for the Case 5 load,
    generator, generator-voltage-setpoint, and complex bus-voltage groups.

    Parameters
    ----------
    columns : pandas.Index
        Source CSV column names in their original order.

    Raises
    ------
    ValueError
        If the source does not match the Case 5 layout used by the tutorial.
    """
    if len(columns) < 26:
        raise ValueError(
            "OPFLearn Case 5 CSV must contain at least 26 columns; "
            f"received {len(columns)}."
        )
    if columns[0] != EXPECTED_FIRST_COLUMN:
        raise ValueError(
            "Unexpected first OPFLearn Case 5 column: "
            f"expected {EXPECTED_FIRST_COLUMN!r}, received {columns[0]!r}."
        )
    if columns[21] != EXPECTED_FIRST_VOLTAGE_COLUMN:
        raise ValueError(
            "Unexpected first OPFLearn Case 5 voltage column: "
            f"expected {EXPECTED_FIRST_VOLTAGE_COLUMN!r}, received "
            f"{columns[21]!r}."
        )


def download_opflearn_case5_dataset(
    cache_directory: Path,
    url: str = OPFLEARN_CASE5_URL,
    timeout_seconds: float = 60.0,
) -> Path:
    """
    Download and validate the OPFLearn Case 5 CSV when it is not cached.

    Existing cached files are never replaced. The source response is first
    written to a temporary file and moved into place only after a successful
    download, avoiding a partially downloaded file in the cache.

    Parameters
    ----------
    cache_directory : pathlib.Path
        Runtime directory where the CSV is cached.
    url : str, optional
        OPFLearn Case 5 download URL. Default is the NLR source URL.
    timeout_seconds : float, optional
        Network timeout in seconds. Default is 60.0.

    Returns
    -------
    pathlib.Path
        Validated path to the cached CSV.

    Raises
    ------
    ValueError
        If the timeout is not positive or the downloaded file has an
        unexpected Case 5 layout.
    RuntimeError
        If the source cannot be downloaded.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive.")

    path = opflearn_case5_path(cache_directory)
    if not path.exists():
        cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".download")
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                temporary_path.write_bytes(response.read())
        except URLError as error:
            temporary_path.unlink(missing_ok=True)
            message = f"Could not download OPFLearn Case 5 from {url!r}."
            raise RuntimeError(message) from error
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
        temporary_path.replace(path)

    columns = pd.read_csv(path, nrows=0).columns
    validate_opflearn_case5_layout(columns)
    return path


def parse_complex_value(value: object) -> complex:
    """
    Parse one complex-number value written in the OPFLearn CSV format.

    Parameters
    ----------
    value : object
        Complex number or string representation, such as ``"1.0 + 0.1j"``.

    Returns
    -------
    complex
        Parsed complex voltage value.

    Raises
    ------
    ValueError
        If the value cannot be parsed as a finite complex number.
    """
    normalized = str(value).replace(" + ", "+").replace(" - ", "-").strip()
    try:
        parsed = complex(normalized)
    except ValueError as error:
        message = f"Could not parse complex OPFLearn value {value!r}."
        raise ValueError(message) from error
    if not np.isfinite(parsed.real) or not np.isfinite(parsed.imag):
        raise ValueError(f"Complex OPFLearn value must be finite: {value!r}.")
    return parsed


def correct_opflearn_case5_voltage(value: object) -> complex:
    """
    Apply the notebook's dataset-specific Case 5 voltage-phase correction.

    OPFLearn's Case 5 ``v_bus`` phase convention differs from the
    PYPOWER/PGLib convention used by this tutorial. This preserves the
    notebook's empirical correction exactly and is not a general correction
    for other OPFLearn datasets.

    Parameters
    ----------
    value : object
        Source complex-voltage value.

    Returns
    -------
    complex
        Corrected complex voltage whose magnitude and phase can be converted
        to the tutorial's voltage-magnitude and voltage-angle labels.
    """
    voltage = parse_complex_value(value)
    magnitude = abs(voltage)
    corrected_angle = -np.rad2deg(np.angle(voltage))
    return complex(magnitude * np.exp(1j * corrected_angle))


def load_clean_opflearn_case5_csv(path: Path) -> pd.DataFrame:
    """
    Load a validated Case 5 CSV and correct its complex bus-voltage columns.

    Parameters
    ----------
    path : pathlib.Path
        Local OPFLearn Case 5 CSV path.

    Returns
    -------
    pandas.DataFrame
        Loaded data with corrected complex values in each ``":v_bus"`` column.

    Raises
    ------
    ValueError
        If the source layout is invalid or has no complex bus-voltage columns.
    """
    raw_data = pd.read_csv(path)
    validate_opflearn_case5_layout(raw_data.columns)
    voltage_columns = [
        column for column in raw_data.columns if column.endswith(":v_bus")
    ]
    if not voltage_columns:
        raise ValueError("OPFLearn Case 5 CSV has no complex bus-voltage columns.")
    cleaned_data = raw_data.copy()
    for column in voltage_columns:
        cleaned_data[column] = cleaned_data[column].map(correct_opflearn_case5_voltage)
    return cleaned_data


def standardize_opflearn_case5_csv(
    raw_data: pd.DataFrame, context: "OPFContext"
) -> pd.DataFrame:
    """
    Convert prepared OPFLearn Case 5 rows to the canonical tutorial schema.

    The input order is active then reactive load, and the target order is
    active then reactive generator power, voltage magnitude, and voltage angle.
    The source is intentionally positional because the Case 5 CSV's groups
    are fixed and its human-readable names differ from the tutorial schema.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Validated data after ``load_clean_opflearn_case5_csv`` processing.
    context : optimal_power_flow.power.context.OPFContext
        Case-derived canonical column and dimensionality metadata.

    Returns
    -------
    pandas.DataFrame
        Numeric canonical inputs and targets in the context's exact order.

    Raises
    ------
    ValueError
        If the source layout does not provide the expected voltage columns.
    """
    validate_opflearn_case5_layout(raw_data.columns)
    canonical_data = pd.DataFrame(index=raw_data.index)
    load_count = len(context.load_bus_indices)
    generator_count = context.generator_count
    bus_count = context.bus_count

    def copy_group(columns: tuple[str, ...], start: int) -> None:
        stop = start + len(columns)
        canonical_data.loc[:, list(columns)] = raw_data.iloc[:, start:stop].to_numpy()

    copy_group(context.active_load_columns, 0)
    copy_group(context.reactive_load_columns, load_count)
    copy_group(context.active_generator_columns, 2 * load_count)
    copy_group(context.reactive_generator_columns, 2 * load_count + generator_count)

    voltage_start = 2 * load_count + 3 * generator_count
    voltage_columns = raw_data.columns[voltage_start : voltage_start + bus_count]
    if len(voltage_columns) != bus_count or not all(
        column.endswith(":v_bus") for column in voltage_columns
    ):
        raise ValueError(
            "Could not find one complex bus-voltage column per Case 5 bus."
        )
    for magnitude_column, angle_column, source_column in zip(
        context.voltage_magnitude_columns,
        context.voltage_angle_columns,
        voltage_columns,
        strict=True,
    ):
        voltage_values = raw_data[source_column]
        canonical_data[magnitude_column] = voltage_values.map(abs)
        canonical_data[angle_column] = voltage_values.map(np.angle)

    ordered_columns = list(context.input_columns + context.output_columns)
    return canonical_data.loc[:, ordered_columns].apply(pd.to_numeric, errors="raise")
