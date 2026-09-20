"""Load optional PGLib MATPOWER cases without affecting default workflows."""

import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from optimal_power_flow.cases.case5_pjm import PowerFlowCase

PGLIB_CASE118_IEEE_URL = (
    "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"
    "pglib_opf_case118_ieee.m"
)
PGLIB_CASE118_IEEE_FILENAME = "pglib_opf_case118_ieee.m"
_REQUIRED_MATPOWER_FIELDS = ("bus", "gen", "branch", "gencost")


def parse_matpower_matrix(text: str, field_name: str) -> np.ndarray:
    """Parse one numeric matrix from a MATPOWER version-two case definition.

    Parameters
    ----------
    text : str
        Complete MATPOWER case-file text.
    field_name : str
        Matrix field name, such as ``"bus"`` or ``"gencost"``.

    Returns
    -------
    numpy.ndarray
        Finite two-dimensional float64 matrix from ``mpc.<field_name>``.

    Raises
    ------
    ValueError
        If the field is absent, empty, ragged, or contains a non-finite value.
    """
    if not field_name:
        raise ValueError("field_name must not be empty.")
    pattern = rf"mpc\.{re.escape(field_name)}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find mpc.{field_name} in the MATPOWER file.")
    rows = []
    for source_line in match.group(1).splitlines():
        values = source_line.split("%", maxsplit=1)[0].strip().rstrip(";")
        if values:
            rows.append([float(value) for value in values.split()])
    if not rows:
        raise ValueError(f"MATPOWER matrix mpc.{field_name} must not be empty.")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"MATPOWER matrix mpc.{field_name} must be rectangular.")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"MATPOWER matrix mpc.{field_name} must be finite.")
    return matrix


def parse_matpower_case(text: str) -> PowerFlowCase:
    """Convert a MATPOWER version-two case file into a PYPOWER case dictionary.

    Parameters
    ----------
    text : str
        Complete MATPOWER ``.m`` case-file text.

    Returns
    -------
    optimal_power_flow.cases.case5_pjm.PowerFlowCase
        Fresh PYPOWER-compatible base-MVA and network-table dictionary.

    Raises
    ------
    ValueError
        If ``baseMVA`` or a required numeric table is missing or invalid.
    """
    base_match = re.search(
        r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text, flags=re.DOTALL
    )
    if base_match is None:
        raise ValueError("Could not find mpc.baseMVA in the MATPOWER file.")
    base_mva = float(base_match.group(1))
    if not np.isfinite(base_mva) or base_mva <= 0.0:
        raise ValueError("MATPOWER mpc.baseMVA must be finite and positive.")
    return {
        "baseMVA": base_mva,
        **{
            field_name: parse_matpower_matrix(text, field_name)
            for field_name in _REQUIRED_MATPOWER_FIELDS
        },
    }


def pglib_case_path(cache_directory: Path, filename: str) -> Path:
    """Return a named PGLib case path beneath a caller-controlled cache directory."""
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a simple non-empty filename.")
    return cache_directory / filename


def download_pglib_matpower_case(
    cache_directory: Path,
    url: str,
    filename: str,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download one MATPOWER case into a cache if it is not already present.

    The network is contacted only when this function is called and the named
    cache entry is absent. A temporary download is moved into place only after
    a successful response; parsing occurs before the final path is returned.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive.")
    path = pglib_case_path(cache_directory, filename)
    if not path.exists():
        cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".download")
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                temporary_path.write_bytes(response.read())
        except URLError as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download PGLib case from {url!r}."
            ) from error
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
        temporary_path.replace(path)
    parse_matpower_case(path.read_text(encoding="utf-8"))
    return path


def load_pglib_matpower_case(
    cache_directory: Path,
    url: str,
    filename: str,
    timeout_seconds: float = 60.0,
) -> PowerFlowCase:
    """Download or reuse a PGLib case, then return a parsed fresh case dictionary."""
    path = download_pglib_matpower_case(cache_directory, url, filename, timeout_seconds)
    return parse_matpower_case(path.read_text(encoding="utf-8"))


def load_pglib_case118_ieee(
    cache_directory: Path,
    timeout_seconds: float = 60.0,
) -> PowerFlowCase:
    """Load the optional PGLib IEEE 118-bus case from a local runtime cache."""
    return load_pglib_matpower_case(
        cache_directory,
        PGLIB_CASE118_IEEE_URL,
        PGLIB_CASE118_IEEE_FILENAME,
        timeout_seconds,
    )
