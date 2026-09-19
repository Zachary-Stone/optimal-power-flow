"""Calculate differentiable physical AC-OPF residual and limit metrics."""

import numpy as np
import torch
from pypower import idx_brch, idx_bus, idx_gen
from pypower.api import makeYbus

from optimal_power_flow.power.context import OPFContext


class OPFAwareMetric:
    """
    Evaluate AC-OPF feasibility, limits, and objective values for predictions.

    Parameters
    ----------
    context : optimal_power_flow.power.context.OPFContext
        Canonical case metadata and internal PYPOWER network case.
    device : torch.device or str, optional
        Device that stores immutable metric tensors. Default is ``"cpu"``.
    """

    def __init__(self, context: OPFContext, device: torch.device | str = "cpu"):
        """Initialize fixed network, cost, and operating-limit tensors."""
        self.context = context
        self.device = torch.device(device)
        self.internal_case = context.internal_case
        self.base_mva = float(self.internal_case["baseMVA"])
        self.bus_count = context.bus_count
        self.generator_count = context.generator_count
        self.reference_bus_index = context.reference_bus_index
        self.output_slices = context.output_slices
        generator = self.internal_case["gen"]
        bus = self.internal_case["bus"]
        branch = self.internal_case["branch"]
        generator_cost = self.internal_case["gencost"]

        self.generator_bus_indices = generator[:, idx_gen.GEN_BUS].astype(int)
        self.load_bus_indices = context.load_bus_indices.astype(int)
        self.generator_to_bus = self._build_mapping_matrix(
            self.generator_bus_indices, self.generator_count
        )
        self.load_to_bus = self._build_mapping_matrix(
            self.load_bus_indices, len(self.load_bus_indices)
        )
        self.from_bus_indices = branch[:, idx_brch.F_BUS].astype(int)
        self.to_bus_indices = branch[:, idx_brch.T_BUS].astype(int)

        ybus, y_from, y_to = makeYbus(self.base_mva, bus, branch)
        self.ybus_real = self._to_tensor(np.asarray(ybus.todense().real))
        self.ybus_imaginary = self._to_tensor(np.asarray(ybus.todense().imag))
        self.y_from_real = self._to_tensor(np.asarray(y_from.todense().real))
        self.y_from_imaginary = self._to_tensor(np.asarray(y_from.todense().imag))
        self.y_to_real = self._to_tensor(np.asarray(y_to.todense().real))
        self.y_to_imaginary = self._to_tensor(np.asarray(y_to.todense().imag))

        coefficients = generator_cost[:, 4:7]
        self.cost_quadratic = self._to_tensor(coefficients[:, 0])
        self.cost_linear = self._to_tensor(coefficients[:, 1])
        self.cost_constant = self._to_tensor(coefficients[:, 2])
        raw_apparent_power_limit = branch[:, idx_brch.RATE_A] / self.base_mva
        self.apparent_power_limit = self._to_tensor(
            np.where(raw_apparent_power_limit > 0.0, raw_apparent_power_limit, np.inf)
        )
        self.angle_minimum = self._to_tensor(np.deg2rad(branch[:, idx_brch.ANGMIN]))
        self.angle_maximum = self._to_tensor(np.deg2rad(branch[:, idx_brch.ANGMAX]))

        active_minimum = generator[:, idx_gen.PMIN] / self.base_mva
        active_maximum = generator[:, idx_gen.PMAX] / self.base_mva
        reactive_minimum = generator[:, idx_gen.QMIN] / self.base_mva
        reactive_maximum = generator[:, idx_gen.QMAX] / self.base_mva
        voltage_minimum = bus[:, idx_bus.VMIN]
        voltage_maximum = bus[:, idx_bus.VMAX]
        angle_minimum = -np.pi * np.ones(self.bus_count)
        angle_maximum = np.pi * np.ones(self.bus_count)
        self.output_minimum = self._to_tensor(
            np.concatenate(
                [
                    active_minimum,
                    reactive_minimum,
                    voltage_minimum,
                    angle_minimum,
                ]
            )
        ).reshape(1, -1)
        self.output_maximum = self._to_tensor(
            np.concatenate(
                [
                    active_maximum,
                    reactive_maximum,
                    voltage_maximum,
                    angle_maximum,
                ]
            )
        ).reshape(1, -1)

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        """
        Convert fixed NumPy values to a float32 tensor on the metric device.

        Parameters
        ----------
        values : numpy.ndarray
            Numeric values to convert.

        Returns
        -------
        torch.Tensor
            Immutable-style tensor used in differentiable calculations.
        """
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _build_mapping_matrix(
        self, bus_indices: np.ndarray, item_count: int
    ) -> torch.Tensor:
        """
        Construct an item-to-bus aggregation matrix.

        Parameters
        ----------
        bus_indices : numpy.ndarray
            Internal bus index for each item.
        item_count : int
            Number of items represented by ``bus_indices``.

        Returns
        -------
        torch.Tensor
            Matrix that sums item quantities at each bus.
        """
        mapping = np.zeros((item_count, self.bus_count), dtype=np.float32)
        for item_index, bus_index in enumerate(bus_indices):
            mapping[item_index, int(bus_index)] = 1.0
        return self._to_tensor(mapping)

    def split_outputs(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split canonical targets into generator power and voltage components.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Per-unit active power, reactive power, voltage magnitude, and
            voltage angle in radians.
        """
        values = outputs.to(self.device)
        slices = self.output_slices
        return (
            values[:, slices.generator_active],
            values[:, slices.generator_reactive],
            values[:, slices.voltage_magnitude],
            values[:, slices.voltage_angle],
        )

    def voltage_rectangular(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert polar voltage outputs to real and imaginary components.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Real voltage, imaginary voltage, and original voltage angle.
        """
        _, _, magnitude, angle = self.split_outputs(outputs)
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle), angle

    @staticmethod
    def complex_power(
        voltage_real: torch.Tensor,
        voltage_imaginary: torch.Tensor,
        current_real: torch.Tensor,
        current_imaginary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate active and reactive complex power from voltage and current.

        Parameters
        ----------
        voltage_real : torch.Tensor
            Real voltage components.
        voltage_imaginary : torch.Tensor
            Imaginary voltage components.
        current_real : torch.Tensor
            Real current components.
        current_imaginary : torch.Tensor
            Imaginary current components.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Active and reactive power components.
        """
        active = voltage_real * current_real + voltage_imaginary * current_imaginary
        reactive = voltage_imaginary * current_real - voltage_real * current_imaginary
        return active, reactive

    def generator_cost(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate polynomial generator cost for each predicted solution.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors with per-unit active generation.

        Returns
        -------
        torch.Tensor
            Total generator cost for each batch row.
        """
        active_generation, _, _, _ = self.split_outputs(outputs)
        active_generation_mw = active_generation * self.base_mva
        return torch.sum(
            self.cost_quadratic * torch.square(active_generation_mw)
            + self.cost_linear * active_generation_mw
            + self.cost_constant,
            dim=1,
        )

    def bound_violation(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate non-negative lower and upper target-bound violations.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        torch.Tensor
            Per-output non-negative violation magnitudes.
        """
        values = outputs.to(self.device)
        return torch.relu(values - self.output_maximum) + torch.relu(
            self.output_minimum - values
        )

    def _bus_injections(
        self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate generation and load input quantities to their buses.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched active then reactive per-unit load features.
        outputs : torch.Tensor
            Batched canonical solution vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Bus active generation, reactive generation, active demand, and
            reactive demand.
        """
        values = inputs.to(self.device)
        load_count = len(self.load_bus_indices)
        active_demand = values[:, :load_count]
        reactive_demand = values[:, load_count : 2 * load_count]
        active_generation, reactive_generation, _, _ = self.split_outputs(outputs)
        return (
            active_generation @ self.generator_to_bus,
            reactive_generation @ self.generator_to_bus,
            active_demand @ self.load_to_bus,
            reactive_demand @ self.load_to_bus,
        )

    def power_balance_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate signed per-bus active and reactive AC power-balance residuals.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched active then reactive per-unit load features.
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Signed active and reactive mismatch at every bus.
        """
        (
            bus_active_generation,
            bus_reactive_generation,
            bus_active_demand,
            bus_reactive_demand,
        ) = self._bus_injections(inputs, outputs)
        voltage_real, voltage_imaginary, _ = self.voltage_rectangular(outputs)
        current_real = (
            voltage_real @ self.ybus_real - voltage_imaginary @ self.ybus_imaginary
        )
        current_imaginary = (
            voltage_imaginary @ self.ybus_real + voltage_real @ self.ybus_imaginary
        )
        bus_active_power, bus_reactive_power = self.complex_power(
            voltage_real, voltage_imaginary, current_real, current_imaginary
        )
        return (
            bus_active_generation - bus_active_demand - bus_active_power,
            bus_reactive_generation - bus_reactive_demand - bus_reactive_power,
        )

    def power_balance_violation(
        self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate absolute per-bus active and reactive power-balance violations.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched active then reactive per-unit load features.
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Non-negative active and reactive mismatch magnitudes.
        """
        active_residual, reactive_residual = self.power_balance_residual(
            inputs, outputs
        )
        return torch.abs(active_residual), torch.abs(reactive_residual)

    def reference_angle_violation(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate absolute reference-bus voltage-angle deviation from zero.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        torch.Tensor
            One non-negative reference-angle violation per batch row.
        """
        return torch.abs(self.reference_angle_residual(outputs))

    def reference_angle_residual(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Return the signed reference-bus voltage-angle residual.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        torch.Tensor
            Signed reference-bus voltage angle in radians for each batch row.
        """
        _, _, _, angle = self.split_outputs(outputs)
        return angle[:, self.reference_bus_index : self.reference_bus_index + 1]

    def branch_flow_violations(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate branch apparent-power and angle-difference violations.

        Parameters
        ----------
        outputs : torch.Tensor
            Batched canonical output vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Non-negative apparent-power and angle-limit violations per branch.
        """
        voltage_real, voltage_imaginary, angle = self.voltage_rectangular(outputs)
        from_current_real = (
            voltage_real @ self.y_from_real.T
            - voltage_imaginary @ self.y_from_imaginary.T
        )
        from_current_imaginary = (
            voltage_imaginary @ self.y_from_real.T
            + voltage_real @ self.y_from_imaginary.T
        )
        to_current_real = (
            voltage_real @ self.y_to_real.T - voltage_imaginary @ self.y_to_imaginary.T
        )
        to_current_imaginary = (
            voltage_imaginary @ self.y_to_real.T + voltage_real @ self.y_to_imaginary.T
        )
        from_active, from_reactive = self.complex_power(
            voltage_real[:, self.from_bus_indices],
            voltage_imaginary[:, self.from_bus_indices],
            from_current_real,
            from_current_imaginary,
        )
        to_active, to_reactive = self.complex_power(
            voltage_real[:, self.to_bus_indices],
            voltage_imaginary[:, self.to_bus_indices],
            to_current_real,
            to_current_imaginary,
        )
        apparent_from = torch.sqrt(
            torch.square(from_active) + torch.square(from_reactive)
        )
        apparent_to = torch.sqrt(torch.square(to_active) + torch.square(to_reactive))
        apparent_power = torch.maximum(apparent_from, apparent_to)
        branch_angle = angle[:, self.from_bus_indices] - angle[:, self.to_bus_indices]
        angle_violation = torch.relu(branch_angle - self.angle_maximum) + torch.relu(
            self.angle_minimum - branch_angle
        )
        return torch.relu(apparent_power - self.apparent_power_limit), angle_violation
