"""Validate notebook-equivalent carbon-price AC-OPF comparisons."""

import copy
import unittest

import numpy as np
from pypower import idx_cost

from optimal_power_flow.cases import case5_pjm
from optimal_power_flow.evaluation import (
    carbon_priced_case,
    compare_carbon_price_opf,
    generator_emissions,
)


class CarbonAwareOPFTests(unittest.TestCase):
    """Test carbon-price objective adjustment and summary generation."""

    def setUp(self) -> None:
        """Create the notebook's illustrative Case 5 emission-rate vector."""
        self.case = case5_pjm()
        self.rates = np.array([0.10, 0.20, 0.55, 0.75, 0.95])

    def test_carbon_price_updates_only_linear_costs_of_a_case_copy(self) -> None:
        """Preserve the caller's cost table while adding price-times-rate terms."""
        original_case = copy.deepcopy(self.case)
        carbon_case = carbon_priced_case(self.case, self.rates, 80.0)
        linear_index = idx_cost.COST + 1

        np.testing.assert_allclose(
            carbon_case["gencost"][:, linear_index],
            original_case["gencost"][:, linear_index] + 80.0 * self.rates,
        )
        np.testing.assert_array_equal(self.case["gencost"], original_case["gencost"])

    def test_comparison_solves_both_objectives_and_reports_dispatch_metrics(
        self,
    ) -> None:
        """Return least-cost and carbon-price dispatches in a two-row table."""
        comparison = compare_carbon_price_opf(self.case, self.rates, 80.0)

        self.assertEqual(
            comparison.summary["case"].tolist(),
            [
                "least_cost_opf",
                "carbon_price_opf",
            ],
        )
        self.assertEqual(comparison.summary.shape[0], 2)
        self.assertIn("generation_cost", comparison.summary.columns)
        self.assertIn("dispatch_emissions_tco2", comparison.summary.columns)
        self.assertGreater(
            generator_emissions(comparison.least_cost_result, self.rates), 0.0
        )

    def test_invalid_emission_rates_are_rejected(self) -> None:
        """Require one non-negative documented rate per generator."""
        with self.assertRaises(ValueError):
            carbon_priced_case(self.case, self.rates[:-1], 80.0)
        with self.assertRaises(ValueError):
            carbon_priced_case(self.case, -self.rates, 80.0)
