"""Validate AC-OPF context construction when PYPOWER is installed."""

import importlib.util
import unittest

PYPOWER_AVAILABLE = importlib.util.find_spec("pypower") is not None

if PYPOWER_AVAILABLE:
    from optimal_power_flow.cases import case5_pjm
    from optimal_power_flow.power.context import build_opf_context


@unittest.skipUnless(PYPOWER_AVAILABLE, "PYPOWER is required for context tests.")
class OPFContextTests(unittest.TestCase):
    """Validate the tutorial's canonical model input and output layout."""

    def test_context_matches_case5_notebook_layout(self) -> None:
        """Build the six-feature, twenty-target schema used in the tutorial."""
        context = build_opf_context(case5_pjm(), "PGLib/PJM Case 5")

        self.assertEqual(context.case_name, "PGLib/PJM Case 5")
        self.assertEqual(context.bus_count, 5)
        self.assertEqual(context.generator_count, 5)
        self.assertEqual(context.branch_count, 6)
        self.assertEqual(context.load_bus_indices.tolist(), [1, 2, 3])
        self.assertEqual(context.reference_bus_index, 3)
        self.assertEqual(
            context.input_columns,
            (
                "pd_bus_1",
                "pd_bus_2",
                "pd_bus_3",
                "qd_bus_1",
                "qd_bus_2",
                "qd_bus_3",
            ),
        )
        self.assertEqual(len(context.output_columns), 20)
        self.assertEqual(context.output_slices.generator_active, slice(0, 5))
        self.assertEqual(context.output_slices.generator_reactive, slice(5, 10))
        self.assertEqual(context.output_slices.voltage_magnitude, slice(10, 15))
        self.assertEqual(context.output_slices.voltage_angle, slice(15, 20))
