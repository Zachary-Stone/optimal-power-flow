"""Build typed AC-OPF schemas from PYPOWER-compatible network cases."""

import copy
from dataclasses import dataclass

import numpy as np
from pypower import idx_bus
from pypower.api import ext2int

from optimal_power_flow.cases.case5_pjm import PowerFlowCase


@dataclass(frozen=True, slots=True)
class OutputSlices:
    """
    Identify contiguous target-vector sections for AC-OPF quantities.

    Parameters
    ----------
    generator_active : slice
        Active generator-power output positions.
    generator_reactive : slice
        Reactive generator-power output positions.
    voltage_magnitude : slice
        Bus voltage-magnitude output positions.
    voltage_angle : slice
        Bus voltage-angle output positions, stored in radians.
    """

    generator_active: slice
    generator_reactive: slice
    voltage_magnitude: slice
    voltage_angle: slice


@dataclass(frozen=True, slots=True)
class OPFContext:
    """
    Store a case's canonical supervised-learning and physical-system schema.

    Parameters
    ----------
    case_name : str
        Human-readable network-case identifier.
    case : PowerFlowCase
        External-numbered PYPOWER-compatible input case.
    internal_case : PowerFlowCase
        Copy of ``case`` converted to PYPOWER's zero-based internal ordering.
    bus_count : int
        Number of buses in the internal case.
    generator_count : int
        Number of generators in the internal case.
    branch_count : int
        Number of branches in the internal case.
    load_bus_indices : numpy.ndarray
        Internal bus indices with active or reactive demand.
    reference_bus_index : int
        Internal index of the reference bus.
    active_load_columns : tuple[str, ...]
        Canonical active-load feature names.
    reactive_load_columns : tuple[str, ...]
        Canonical reactive-load feature names.
    active_generator_columns : tuple[str, ...]
        Canonical active-generator target names.
    reactive_generator_columns : tuple[str, ...]
        Canonical reactive-generator target names.
    voltage_magnitude_columns : tuple[str, ...]
        Canonical voltage-magnitude target names.
    voltage_angle_columns : tuple[str, ...]
        Canonical voltage-angle target names.
    output_slices : OutputSlices
        Target-vector sections corresponding to each physical quantity.
    """

    case_name: str
    case: PowerFlowCase
    internal_case: PowerFlowCase
    bus_count: int
    generator_count: int
    branch_count: int
    load_bus_indices: np.ndarray
    reference_bus_index: int
    active_load_columns: tuple[str, ...]
    reactive_load_columns: tuple[str, ...]
    active_generator_columns: tuple[str, ...]
    reactive_generator_columns: tuple[str, ...]
    voltage_magnitude_columns: tuple[str, ...]
    voltage_angle_columns: tuple[str, ...]
    output_slices: OutputSlices

    @property
    def input_columns(self) -> tuple[str, ...]:
        """
        Return ordered active and reactive load feature names.

        Returns
        -------
        tuple[str, ...]
            Input-column names in model feature-vector order.
        """
        return self.active_load_columns + self.reactive_load_columns

    @property
    def output_columns(self) -> tuple[str, ...]:
        """
        Return ordered generator and voltage target names.

        Returns
        -------
        tuple[str, ...]
            Output-column names in model target-vector order.
        """
        return (
            self.active_generator_columns
            + self.reactive_generator_columns
            + self.voltage_magnitude_columns
            + self.voltage_angle_columns
        )


def build_opf_context(
    case: PowerFlowCase, case_name: str = "PYPOWER-compatible case"
) -> OPFContext:
    """
    Build the canonical schema shared by AC-OPF data, models, and metrics.

    Parameters
    ----------
    case : PowerFlowCase
        External-numbered MATPOWER/PYPOWER case.
    case_name : str, optional
        Human-readable case name. Default is ``"PYPOWER-compatible case"``.

    Returns
    -------
    OPFContext
        Typed internal-case metadata, canonical input/output names, and target
        vector slices.

    Raises
    ------
    ValueError
        If the case has no buses or reference bus.
    """
    internal_case = ext2int(copy.deepcopy(case))
    bus = internal_case["bus"]
    generator = internal_case["gen"]
    branch = internal_case["branch"]
    bus_count = len(bus)
    if bus_count == 0:
        raise ValueError("An OPF case must contain at least one bus.")

    reference_candidates = np.flatnonzero(bus[:, idx_bus.BUS_TYPE] == idx_bus.REF)
    if len(reference_candidates) == 0:
        raise ValueError("An OPF case must contain a reference bus.")
    load_bus_indices = np.flatnonzero(
        (bus[:, idx_bus.PD] != 0) | (bus[:, idx_bus.QD] != 0)
    ).astype(int)
    generator_count = len(generator)

    active_load_columns = tuple(f"pd_bus_{index}" for index in load_bus_indices)
    reactive_load_columns = tuple(f"qd_bus_{index}" for index in load_bus_indices)
    active_generator_columns = tuple(
        f"pg_gen_{index}" for index in range(generator_count)
    )
    reactive_generator_columns = tuple(
        f"qg_gen_{index}" for index in range(generator_count)
    )
    voltage_magnitude_columns = tuple(f"vm_bus_{index}" for index in range(bus_count))
    voltage_angle_columns = tuple(f"va_bus_{index}" for index in range(bus_count))

    active_end = generator_count
    reactive_end = 2 * generator_count
    magnitude_end = reactive_end + bus_count
    output_slices = OutputSlices(
        generator_active=slice(0, active_end),
        generator_reactive=slice(active_end, reactive_end),
        voltage_magnitude=slice(reactive_end, magnitude_end),
        voltage_angle=slice(magnitude_end, magnitude_end + bus_count),
    )
    return OPFContext(
        case_name=case_name,
        case=case,
        internal_case=internal_case,
        bus_count=bus_count,
        generator_count=generator_count,
        branch_count=len(branch),
        load_bus_indices=load_bus_indices,
        reference_bus_index=int(reference_candidates[0]),
        active_load_columns=active_load_columns,
        reactive_load_columns=reactive_load_columns,
        active_generator_columns=active_generator_columns,
        reactive_generator_columns=reactive_generator_columns,
        voltage_magnitude_columns=voltage_magnitude_columns,
        voltage_angle_columns=voltage_angle_columns,
        output_slices=output_slices,
    )
