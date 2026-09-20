"""Compare least-cost and carbon-price AC-OPF dispatch without case mutation."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pypower import idx_cost, idx_gen

from optimal_power_flow.cases.case5_pjm import PowerFlowCase
from optimal_power_flow.power.solver import solve_ac_opf


@dataclass(frozen=True, slots=True)
class CarbonOPFComparison:
    """Store both solver results and a tabular carbon-price comparison."""

    least_cost_result: Mapping[str, Any]
    carbon_price_result: Mapping[str, Any]
    summary: pd.DataFrame


def _validate_emission_rates(
    emission_rates_ton_per_mwh: np.ndarray | list[float], generator_count: int
) -> np.ndarray:
    """Return a validated one-dimensional non-negative emission-rate vector."""
    rates = np.asarray(emission_rates_ton_per_mwh, dtype=np.float64).reshape(-1)
    if rates.shape != (generator_count,):
        raise ValueError("Emission rates must contain exactly one value per generator.")
    if not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("Emission rates must be finite and non-negative.")
    return rates


def _validate_polynomial_costs(case: PowerFlowCase) -> np.ndarray:
    """Validate and return the generator-cost table used for carbon pricing."""
    generator = case["gen"]
    generator_cost = case["gencost"]
    if len(generator_cost) != len(generator):
        raise ValueError("gencost must contain exactly one row per generator.")
    if generator_cost.ndim != 2 or generator_cost.shape[1] <= idx_cost.NCOST:
        raise ValueError("gencost has an invalid MATPOWER polynomial layout.")
    if np.any(generator_cost[:, idx_cost.MODEL] != idx_cost.POLYNOMIAL):
        raise ValueError("Carbon pricing requires polynomial generator costs.")
    coefficient_counts = generator_cost[:, idx_cost.NCOST].astype(int)
    if np.any(coefficient_counts < 2):
        raise ValueError("Carbon pricing requires each cost function to include Pg.")
    if np.any(idx_cost.COST + coefficient_counts > generator_cost.shape[1]):
        raise ValueError("gencost lacks the declared polynomial coefficients.")
    return generator_cost


def carbon_priced_case(
    case: PowerFlowCase,
    emission_rates_ton_per_mwh: np.ndarray | list[float],
    carbon_price_per_ton: float,
) -> PowerFlowCase:
    """Return a deep-copied case with carbon price added to linear cost terms.

    The carbon price has units of currency per metric ton CO2 and the emission
    rates have units of metric tons CO2 per MWh. The resulting coefficient has
    the same units as each generator's linear operating-cost coefficient.

    Parameters
    ----------
    case : PowerFlowCase
        Base least-cost PYPOWER-compatible case. It is never modified.
    emission_rates_ton_per_mwh : numpy.ndarray or list[float]
        One non-negative emission rate for each generator.
    carbon_price_per_ton : float
        Non-negative carbon price in currency per metric ton CO2.

    Returns
    -------
    PowerFlowCase
        Independent case whose linear generator-cost coefficients include the
        carbon-price term.
    """
    if not np.isfinite(carbon_price_per_ton) or carbon_price_per_ton < 0.0:
        raise ValueError("carbon_price_per_ton must be finite and non-negative.")
    carbon_case = copy.deepcopy(case)
    generator_cost = _validate_polynomial_costs(carbon_case)
    rates = _validate_emission_rates(
        emission_rates_ton_per_mwh, len(carbon_case["gen"])
    )
    coefficient_counts = generator_cost[:, idx_cost.NCOST].astype(int)
    linear_indices = idx_cost.COST + coefficient_counts - 2
    generator_cost[np.arange(len(generator_cost)), linear_indices] += (
        carbon_price_per_ton * rates
    )
    return carbon_case


def generator_emissions(
    result: Mapping[str, Any],
    emission_rates_ton_per_mwh: np.ndarray | list[float],
    hours: float = 1.0,
) -> float:
    """Calculate dispatch emissions in metric tons CO2 for a finite interval."""
    if not np.isfinite(hours) or hours < 0.0:
        raise ValueError("hours must be finite and non-negative.")
    active_generation = np.asarray(result["gen"])[:, idx_gen.PG]
    rates = _validate_emission_rates(emission_rates_ton_per_mwh, len(active_generation))
    return float(hours * np.sum(active_generation * rates))


def original_generation_cost(
    result: Mapping[str, Any], reference_case: PowerFlowCase
) -> float:
    """Evaluate a solved dispatch using the original polynomial cost curves."""
    generator_cost = _validate_polynomial_costs(reference_case)
    active_generation = np.asarray(result["gen"])[:, idx_gen.PG]
    if len(active_generation) != len(generator_cost):
        raise ValueError("Result generation count does not match reference gencost.")
    total_cost = 0.0
    for output, cost_row in zip(active_generation, generator_cost, strict=True):
        coefficient_count = int(cost_row[idx_cost.NCOST])
        coefficients = cost_row[idx_cost.COST : idx_cost.COST + coefficient_count]
        total_cost += np.polyval(coefficients, output)
    return float(total_cost)


def compare_carbon_price_opf(
    case: PowerFlowCase,
    emission_rates_ton_per_mwh: np.ndarray | list[float],
    carbon_price_per_ton: float,
    hours: float = 1.0,
) -> CarbonOPFComparison:
    """Solve and summarize least-cost and carbon-priced AC-OPF variants.

    The summary preserves the original generation cost for both dispatches,
    while the carbon-priced solver objective includes the carbon term.
    """
    carbon_case = carbon_priced_case(
        case, emission_rates_ton_per_mwh, carbon_price_per_ton
    )
    least_cost_result = solve_ac_opf(case)
    carbon_price_result = solve_ac_opf(carbon_case)
    rates = _validate_emission_rates(emission_rates_ton_per_mwh, len(case["gen"]))

    def summarize(label: str, result: Mapping[str, Any]) -> dict[str, float | str]:
        active_generation = np.asarray(result["gen"])[:, idx_gen.PG]
        return {
            "case": label,
            "generation_cost": original_generation_cost(result, case),
            "dispatch_emissions_tco2": generator_emissions(result, rates, hours),
            "carbon_price_objective": float(result["f"]),
            **{
                f"pg_gen_{index}_mw": float(output)
                for index, output in enumerate(active_generation)
            },
        }

    summary = pd.DataFrame(
        (
            summarize("least_cost_opf", least_cost_result),
            summarize("carbon_price_opf", carbon_price_result),
        )
    )
    return CarbonOPFComparison(least_cost_result, carbon_price_result, summary)
