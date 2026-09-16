"""Validate local OPFLearn Case 5 parsing and canonicalization."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from optimal_power_flow.dataset_io.opflearn import (
    correct_opflearn_case5_voltage,
    download_opflearn_case5_dataset,
    load_clean_opflearn_case5_csv,
    standardize_opflearn_case5_csv,
    validate_opflearn_case5_layout,
)


def make_raw_case5_data(rows: int = 2) -> pd.DataFrame:
    """Create a local fixture with the fixed OPFLearn Case 5 column layout."""
    columns = [
        "load1:pl",
        "load2:pl",
        "load3:pl",
        "load1:ql",
        "load2:ql",
        "load3:ql",
        *[f"gen{index}:pg" for index in range(1, 6)],
        *[f"gen{index}:qg" for index in range(1, 6)],
        *[f"gen{index}:vg" for index in range(1, 6)],
        *[f"bus{index}:v_bus" for index in range(1, 6)],
    ]
    data = {}
    for index, column in enumerate(columns):
        data[column] = [float(index + row) for row in range(rows)]
    for index in range(1, 6):
        data[f"bus{index}:v_bus"] = ["1.0 + 0.1j"] * rows
    return pd.DataFrame(data)


def make_context() -> SimpleNamespace:
    """Create the Case 5 schema attributes consumed by the data adapter."""
    return SimpleNamespace(
        load_bus_indices=np.array([1, 2, 3]),
        generator_count=5,
        bus_count=5,
        active_load_columns=("pd_bus_1", "pd_bus_2", "pd_bus_3"),
        reactive_load_columns=("qd_bus_1", "qd_bus_2", "qd_bus_3"),
        active_generator_columns=tuple(f"pg_gen_{index}" for index in range(5)),
        reactive_generator_columns=tuple(f"qg_gen_{index}" for index in range(5)),
        voltage_magnitude_columns=tuple(f"vm_bus_{index}" for index in range(5)),
        voltage_angle_columns=tuple(f"va_bus_{index}" for index in range(5)),
        input_columns=(
            "pd_bus_1",
            "pd_bus_2",
            "pd_bus_3",
            "qd_bus_1",
            "qd_bus_2",
            "qd_bus_3",
        ),
        output_columns=tuple(
            [f"pg_gen_{index}" for index in range(5)]
            + [f"qg_gen_{index}" for index in range(5)]
            + [f"vm_bus_{index}" for index in range(5)]
            + [f"va_bus_{index}" for index in range(5)]
        ),
    )


class OPFLearnCase5Tests(unittest.TestCase):
    """Validate the fixed Case 5 adapter without a network download."""

    def test_layout_rejects_unexpected_columns(self) -> None:
        """Reject an input that is not the expected OPFLearn Case 5 layout."""
        data = make_raw_case5_data()
        columns = data.columns.to_list()
        columns[0] = "unexpected"

        with self.assertRaises(ValueError):
            validate_opflearn_case5_layout(pd.Index(columns))

    def test_load_cleans_complex_bus_voltage_values(self) -> None:
        """Apply the notebook's phase correction to each complex bus voltage."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case5.csv"
            make_raw_case5_data().to_csv(path, index=False)
            loaded_data = load_clean_opflearn_case5_csv(path)

        expected = correct_opflearn_case5_voltage("1.0 + 0.1j")
        self.assertEqual(loaded_data.loc[0, "bus1:v_bus"], expected)

    def test_download_uses_and_validates_a_local_cache_file(self) -> None:
        """Download a fixture URL into the configured runtime cache once."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.csv"
            make_raw_case5_data().to_csv(source_path, index=False)

            cached_path = download_opflearn_case5_dataset(
                root / "cache", url=source_path.as_uri()
            )
            self.assertTrue(cached_path.exists())
            self.assertEqual(pd.read_csv(cached_path, nrows=0).shape[1], 26)

        self.assertTrue(cached_path.name.endswith("pglib_opf_case5_pjm.csv"))

    def test_standardization_matches_canonical_column_order(self) -> None:
        """Map fixed source groups to the Case 5 feature and target schema."""
        context = make_context()
        raw_data = make_raw_case5_data()
        for column in raw_data.columns[-5:]:
            raw_data[column] = raw_data[column].map(correct_opflearn_case5_voltage)

        canonical_data = standardize_opflearn_case5_csv(raw_data, context)

        self.assertEqual(
            canonical_data.columns.tolist(),
            list(context.input_columns + context.output_columns),
        )
        self.assertEqual(canonical_data.shape, (2, 26))
        self.assertEqual(canonical_data.loc[0, "pd_bus_1"], 0.0)
        self.assertAlmostEqual(canonical_data.loc[0, "vm_bus_0"], np.sqrt(1.01))
