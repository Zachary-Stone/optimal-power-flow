"""Validate a quiet AC-OPF solve for the tutorial network case."""

import importlib.util
import unittest

PYPOWER_AVAILABLE = importlib.util.find_spec("pypower") is not None

if PYPOWER_AVAILABLE:
    from optimal_power_flow.cases import case5_pjm
    from optimal_power_flow.power.solver import solve_ac_opf


@unittest.skipUnless(PYPOWER_AVAILABLE, "PYPOWER is required for solver tests.")
class SolverTests(unittest.TestCase):
    """Validate the solver wrapper against the notebook's five-bus case."""

    def test_case5_solve_succeeds_with_finite_objective(self) -> None:
        """Solve the reference case without emitting the solver report."""
        result = solve_ac_opf(case5_pjm())

        self.assertTrue(result["success"])
        self.assertEqual(result["bus"].shape, (5, 17))
        self.assertEqual(result["gen"].shape, (5, 25))
        self.assertGreater(float(result["f"]), 0.0)
