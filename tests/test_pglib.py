"""Test optional PGLib MATPOWER parsing without downloading a public case."""

import tempfile
import unittest
from pathlib import Path

from optimal_power_flow.cases import (
    download_pglib_matpower_case,
    parse_matpower_case,
    parse_matpower_matrix,
)
from optimal_power_flow.power import OPFAwareMetric, build_opf_context

MATPOWER_FIXTURE = """
function mpc = case_fixture
mpc.version = '2';
mpc.baseMVA = 100.0;
mpc.bus = [
    1 3 0 0 0 0 1 1.0 0 230 1 1.1 0.9;
    2 1 50 20 0 0 1 1.0 0 230 1 1.1 0.9;
];
mpc.gen = [
    1 60 0 100 -100 1 100 1 120 0;
];
mpc.branch = [
    1 2 0.01 0.05 0.01 100 100 100 0 0 1 -30 30;
];
mpc.gencost = [
    2 0 0 3 0.02 10 0;
];
"""


class PGLibCaseTests(unittest.TestCase):
    """Validate local MATPOWER parsing and cache behavior for optional cases."""

    def test_parser_builds_context_and_metric_for_local_matpower_fixture(self) -> None:
        """Convert a local case definition into the shared context and metric schema."""
        case = parse_matpower_case(MATPOWER_FIXTURE)
        context = build_opf_context(case, "Fixture Case")
        metric = OPFAwareMetric(context)

        self.assertEqual(case["bus"].shape, (2, 13))
        self.assertEqual(context.bus_count, 2)
        self.assertEqual(len(context.input_columns), 2)
        self.assertEqual(len(context.output_columns), 6)
        self.assertEqual(metric.output_minimum.shape, (1, 6))

    def test_download_reuses_a_local_cache_source(self) -> None:
        """Download a file URL once and validate the cached MATPOWER text."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m"
            source.write_text(MATPOWER_FIXTURE, encoding="utf-8")
            cached = download_pglib_matpower_case(
                root / "cache", source.as_uri(), "fixture.m"
            )
            source.unlink()
            reused = download_pglib_matpower_case(
                root / "cache", "https://invalid.example/fixture.m", "fixture.m"
            )
            generator_shape = parse_matpower_matrix(cached.read_text(), "gen").shape

        self.assertEqual(cached, reused)
        self.assertEqual(generator_shape, (1, 10))

    def test_parser_rejects_missing_matrix(self) -> None:
        """Report the absent field instead of returning an incomplete case."""
        with self.assertRaisesRegex(ValueError, "gencost"):
            parse_matpower_case(MATPOWER_FIXTURE.replace("mpc.gencost", "mpc.cost"))
