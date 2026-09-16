"""Validate the reusable five-bus PJM network-case definition."""

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

    from optimal_power_flow.cases import case5_pjm


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required for case tests.")
class Case5PjmTests(unittest.TestCase):
    """Validate dimensions and isolation of the tutorial's network case."""

    def test_case_matches_notebook_schema(self) -> None:
        """Expose the five buses, generators, and six branches used in the notebook."""
        case = case5_pjm()

        self.assertEqual(case["baseMVA"], 100.0)
        self.assertEqual(case["bus"].shape, (5, 13))
        self.assertEqual(case["gen"].shape, (5, 10))
        self.assertEqual(case["gencost"].shape, (5, 7))
        self.assertEqual(case["branch"].shape, (6, 13))
        self.assertEqual(float(np.sum(case["bus"][:, 2])), 1000.0)

    def test_case_returns_fresh_arrays(self) -> None:
        """Prevent a solver or caller mutation from leaking into another experiment."""
        first_case = case5_pjm()
        first_case["bus"][0, 2] = 123.0

        second_case = case5_pjm()

        self.assertEqual(second_case["bus"][0, 2], 0.0)
