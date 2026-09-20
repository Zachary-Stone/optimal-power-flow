"""Physics-completing architectures that solve balancing generation exactly."""

from dataclasses import dataclass

import numpy as np
import torch
from pypower import idx_bus
from torch import nn

from optimal_power_flow.models.residual import ResidualMLP
from optimal_power_flow.power.metrics import OPFAwareMetric


@dataclass(frozen=True, slots=True)
class GeneratorBalanceMapping:
    """Partition generators into one balancing unit per generator bus and extras."""

    balancing_generator_indices: torch.Tensor
    balancing_bus_indices: torch.Tensor
    extra_generator_indices: torch.Tensor


def identify_balancing_generators(metric: OPFAwareMetric) -> GeneratorBalanceMapping:
    """Select the first generator at every generating bus as its balancing unit."""
    by_bus: dict[int, int] = {}
    balance_indices = []
    extra_indices = []
    for generator_index, bus_index in enumerate(metric.generator_bus_indices):
        bus_number = int(bus_index)
        if bus_number in by_bus:
            extra_indices.append(generator_index)
        else:
            by_bus[bus_number] = generator_index
            balance_indices.append(generator_index)
    device = metric.device
    return GeneratorBalanceMapping(
        balancing_generator_indices=torch.tensor(
            balance_indices, dtype=torch.long, device=device
        ),
        balancing_bus_indices=torch.tensor(
            list(by_bus), dtype=torch.long, device=device
        ),
        extra_generator_indices=torch.tensor(
            extra_indices, dtype=torch.long, device=device
        ),
    )


def complete_balancing_generators(
    metric: OPFAwareMetric,
    mapping: GeneratorBalanceMapping,
    inputs: torch.Tensor,
    voltage_magnitude: torch.Tensor,
    voltage_angle: torch.Tensor,
    extra_active_generation: torch.Tensor,
    extra_reactive_generation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete generator powers required by predicted voltages and load inputs.

    The selected balancing generator at each generator bus receives the power
    mismatch remaining after any other generator at that bus is assigned.
    Consequently, power balance is exact at every bus that has generation.
    """
    batch_size = inputs.shape[0]
    generator_count = metric.generator_count
    extra_count = mapping.extra_generator_indices.numel()
    expected_voltage_shape = (batch_size, metric.bus_count)
    if inputs.ndim != 2 or voltage_magnitude.shape != expected_voltage_shape:
        raise ValueError("inputs and voltage_magnitude have incompatible shapes.")
    if voltage_angle.shape != expected_voltage_shape:
        raise ValueError("voltage_angle must contain one angle per bus.")
    if extra_active_generation.shape != (batch_size, extra_count):
        raise ValueError("extra_active_generation has an incompatible shape.")
    if extra_reactive_generation.shape != (batch_size, extra_count):
        raise ValueError("extra_reactive_generation has an incompatible shape.")

    voltage_real = voltage_magnitude * torch.cos(voltage_angle)
    voltage_imaginary = voltage_magnitude * torch.sin(voltage_angle)
    current_real = (
        voltage_real @ metric.ybus_real - voltage_imaginary @ metric.ybus_imaginary
    )
    current_imaginary = (
        voltage_imaginary @ metric.ybus_real + voltage_real @ metric.ybus_imaginary
    )
    network_active, network_reactive = metric.complex_power(
        voltage_real, voltage_imaginary, current_real, current_imaginary
    )
    load_count = len(metric.load_bus_indices)
    active_demand = inputs[:, :load_count] @ metric.load_to_bus
    reactive_demand = inputs[:, load_count : 2 * load_count] @ metric.load_to_bus
    required_active = network_active + active_demand
    required_reactive = network_reactive + reactive_demand

    active_generation = torch.zeros(
        (batch_size, generator_count), dtype=inputs.dtype, device=inputs.device
    )
    reactive_generation = torch.zeros_like(active_generation)
    if extra_count:
        active_generation.index_copy_(
            1, mapping.extra_generator_indices, extra_active_generation
        )
        reactive_generation.index_copy_(
            1, mapping.extra_generator_indices, extra_reactive_generation
        )
    extra_active_by_bus = active_generation @ metric.generator_to_bus
    extra_reactive_by_bus = reactive_generation @ metric.generator_to_bus
    balancing_active = (
        required_active[:, mapping.balancing_bus_indices]
        - extra_active_by_bus[:, mapping.balancing_bus_indices]
    )
    balancing_reactive = (
        required_reactive[:, mapping.balancing_bus_indices]
        - extra_reactive_by_bus[:, mapping.balancing_bus_indices]
    )
    active_generation.index_copy_(
        1, mapping.balancing_generator_indices, balancing_active
    )
    reactive_generation.index_copy_(
        1, mapping.balancing_generator_indices, balancing_reactive
    )
    return active_generation, reactive_generation


class PhysicalCompletionMLP(nn.Module):
    """Predict bounded voltages and extra units, then complete balancing units."""

    def __init__(
        self,
        metric: OPFAwareMetric,
        input_size: int,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        """Initialize direct bounded predictions and balancing-generator mapping."""
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1.")
        self.metric = metric
        self.mapping = identify_balancing_generators(metric)
        extra_count = self.mapping.extra_generator_indices.numel()
        direct_output_size = 2 * extra_count + metric.bus_count + metric.bus_count - 1
        self.network = ResidualMLP(input_size, direct_output_size, width, depth)
        self.register_buffer("output_minimum", metric.output_minimum.clone())
        self.register_buffer("output_maximum", metric.output_maximum.clone())
        reference_angle = np.deg2rad(
            metric.internal_case["bus"][metric.reference_bus_index, idx_bus.VA]
        )
        self.register_buffer(
            "reference_angle", torch.tensor(reference_angle, dtype=torch.float32)
        )

    @staticmethod
    def _bound(
        raw_values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """Map unconstrained values into elementwise finite closed intervals."""
        return lower + torch.sigmoid(raw_values) * (upper - lower)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict a canonical target vector with completed balancing generation."""
        raw_values = self.network(inputs)
        metric = self.metric
        slices = metric.output_slices
        extra_indices = self.mapping.extra_generator_indices
        extra_count = extra_indices.numel()
        cursor = 0
        active_bounds = metric.output_minimum[:, slices.generator_active]
        active_lower = active_bounds[:, extra_indices]
        active_bounds = metric.output_maximum[:, slices.generator_active]
        active_upper = active_bounds[:, extra_indices]
        extra_active = self._bound(
            raw_values[:, cursor : cursor + extra_count], active_lower, active_upper
        )
        cursor += extra_count
        reactive_bounds = metric.output_minimum[:, slices.generator_reactive]
        reactive_lower = reactive_bounds[:, extra_indices]
        reactive_bounds = metric.output_maximum[:, slices.generator_reactive]
        reactive_upper = reactive_bounds[:, extra_indices]
        extra_reactive = self._bound(
            raw_values[:, cursor : cursor + extra_count], reactive_lower, reactive_upper
        )
        cursor += extra_count
        magnitude_lower = metric.output_minimum[:, slices.voltage_magnitude]
        magnitude_upper = metric.output_maximum[:, slices.voltage_magnitude]
        voltage_magnitude = self._bound(
            raw_values[:, cursor : cursor + metric.bus_count],
            magnitude_lower,
            magnitude_upper,
        )
        cursor += metric.bus_count
        non_reference_indices = torch.tensor(
            [
                index
                for index in range(metric.bus_count)
                if index != metric.reference_bus_index
            ],
            dtype=torch.long,
            device=inputs.device,
        )
        angle_bounds = metric.output_minimum[:, slices.voltage_angle]
        angle_lower = angle_bounds[:, non_reference_indices]
        angle_bounds = metric.output_maximum[:, slices.voltage_angle]
        angle_upper = angle_bounds[:, non_reference_indices]
        non_reference_angles = self._bound(
            raw_values[:, cursor:], angle_lower, angle_upper
        )
        voltage_angle = torch.full(
            (inputs.shape[0], metric.bus_count),
            self.reference_angle.item(),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        voltage_angle.index_copy_(1, non_reference_indices, non_reference_angles)
        active_generation, reactive_generation = complete_balancing_generators(
            metric,
            self.mapping,
            inputs,
            voltage_magnitude,
            voltage_angle,
            extra_active,
            extra_reactive,
        )
        return torch.cat(
            (
                active_generation,
                reactive_generation,
                voltage_magnitude,
                voltage_angle,
            ),
            dim=1,
        )
