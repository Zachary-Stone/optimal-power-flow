"""Validate optional CodeCarbon tracking without starting a real tracker."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from optimal_power_flow.training.emissions import track_emissions


class EmissionsTests(unittest.TestCase):
    """Validate opt-in tracker lifecycle and reported metrics."""

    def test_disabled_tracking_has_no_measurement(self) -> None:
        """Avoid starting CodeCarbon when a workflow disables tracking."""
        with track_emissions(False, "unit_test") as measurement:
            pass

        self.assertIsNone(measurement.energy_kwh)
        self.assertIsNone(measurement.emissions_kg_co2eq)

    @patch("optimal_power_flow.training.emissions.EmissionsTracker")
    def test_enabled_tracking_collects_final_measurement(self, tracker_class) -> None:
        """Start and stop the tracker once, then expose final in-memory values."""
        tracker = tracker_class.return_value
        tracker.final_emissions_data = SimpleNamespace(
            energy_consumed=0.25,
            emissions=0.125,
        )

        with track_emissions(True, "unit_test") as measurement:
            pass

        tracker.start.assert_called_once_with()
        tracker.stop.assert_called_once_with()
        self.assertEqual(measurement.energy_kwh, 0.25)
        self.assertEqual(measurement.emissions_kg_co2eq, 0.125)
