"""Evaluation and validation for canonical assignment traces."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from state_bench.model import (
    AssignConstant,
    AssignIncrement,
    AssignVariable,
    Operation,
    StateTrackingInstance,
)


@dataclass(frozen=True, slots=True)
class DependencySummary:
    final_dependency_lengths: tuple[int, ...]
    max_dependency_length: int
    mean_dependency_length: float
    modal_dependency_length: int

    def to_record(self) -> dict[str, Any]:
        return {
            "final_dependency_lengths": list(self.final_dependency_lengths),
            "max_dependency_length": self.max_dependency_length,
            "mean_dependency_length": self.mean_dependency_length,
            "modal_dependency_length": self.modal_dependency_length,
        }


@dataclass(frozen=True, slots=True)
class SensitivitySummary:
    """Exact counterfactual influence of live program state on queried outputs."""

    sensitivity_state_count: int
    sensitivity_output_count: int
    sensitivity_density: float
    sensitivity_coverage: float
    mean_live_fanout: float

    def to_record(self) -> dict[str, Any]:
        return {
            "sensitivity_state_count": self.sensitivity_state_count,
            "sensitivity_output_count": self.sensitivity_output_count,
            "sensitivity_density": self.sensitivity_density,
            "sensitivity_coverage": self.sensitivity_coverage,
            "mean_live_fanout": self.mean_live_fanout,
        }


def evaluate(*, num_variables: int, num_values: int, operations: Sequence[Operation]) -> tuple[int, ...]:
    """Evaluate eager, copy-at-assignment-time variable semantics.

    Increments wrap modulo ``num_values``: ``x <- (x + delta) mod num_values``.
    """

    if num_variables < 1:
        raise ValueError("num_variables must be positive")
    if num_values < 1:
        raise ValueError("num_values must be positive")

    state: list[int | None] = [None] * num_variables
    for index, operation in enumerate(operations):
        _validate_variable(operation.target, num_variables)
        if isinstance(operation, AssignConstant):
            if not 0 <= operation.value < num_values:
                raise ValueError(f"operation {index} has out-of-range value {operation.value}")
            state[operation.target] = operation.value
        elif isinstance(operation, AssignVariable):
            _validate_variable(operation.source, num_variables)
            source_value = state[operation.source]
            if source_value is None:
                raise ValueError(f"operation {index} reads uninitialized variable {operation.source}")
            state[operation.target] = source_value
        elif isinstance(operation, AssignIncrement):
            if not 0 < operation.delta < num_values:
                raise ValueError(f"operation {index} has out-of-range delta {operation.delta}")
            current = state[operation.target]
            if current is None:
                raise ValueError(f"operation {index} increments uninitialized variable {operation.target}")
            state[operation.target] = (current + operation.delta) % num_values

    uninitialized = [index for index, value in enumerate(state) if value is None]
    if uninitialized:
        raise ValueError(f"final state has uninitialized variables: {uninitialized}")
    return tuple(value for value in state if value is not None)


def dependency_lengths(
    *,
    num_variables: int,
    operations: Sequence[Operation],
) -> tuple[int, ...]:
    """Return each final variable's eager-copy hop count back to a constant assignment.

    An increment is a self-referential hop: it extends the target's own chain by one.
    """

    if num_variables < 1:
        raise ValueError("num_variables must be positive")
    depths: list[int | None] = [None] * num_variables
    for index, operation in enumerate(operations):
        _validate_variable(operation.target, num_variables)
        if isinstance(operation, AssignConstant):
            depths[operation.target] = 0
        elif isinstance(operation, AssignVariable):
            _validate_variable(operation.source, num_variables)
            source_depth = depths[operation.source]
            if source_depth is None:
                raise ValueError(f"operation {index} reads uninitialized variable {operation.source}")
            depths[operation.target] = source_depth + 1
        elif isinstance(operation, AssignIncrement):
            current_depth = depths[operation.target]
            if current_depth is None:
                raise ValueError(f"operation {index} increments uninitialized variable {operation.target}")
            depths[operation.target] = current_depth + 1

    uninitialized = [index for index, depth in enumerate(depths) if depth is None]
    if uninitialized:
        raise ValueError(f"final state has uninitialized variables: {uninitialized}")
    return tuple(depth for depth in depths if depth is not None)


def dependency_summary(instance: StateTrackingInstance) -> DependencySummary:
    """Summarize dependency lengths over all variables in the final state."""

    register_lengths = dependency_lengths(
        num_variables=instance.num_registers,
        operations=instance.operations,
    )
    lengths = register_lengths[: instance.num_variables]
    frequencies = Counter(lengths)
    highest_frequency = max(frequencies.values())
    mode = min(depth for depth, count in frequencies.items() if count == highest_frequency)
    return DependencySummary(
        final_dependency_lengths=lengths,
        max_dependency_length=max(lengths),
        mean_dependency_length=statistics.fmean(lengths),
        modal_dependency_length=mode,
    )


def sensitivity_summary(instance: StateTrackingInstance) -> SensitivitySummary:
    """Summarize exact suffix influence at semantic operation boundaries.

    The scratch register is excluded from both inputs and queried outputs. Each
    three-assignment TMP swap is sampled only after the complete swap, while
    ordinary constant assignments remain individual semantic operations. A
    reverse pass propagates query-output bitsets through every primitive copy
    and kills them at constant assignments. Increments are bijections of their
    target's value, so they carry the target's bitset through unchanged.
    """

    start = _fully_initialized_boundary(
        num_variables=instance.num_registers,
        operations=instance.operations,
    )
    queried_variables = tuple(
        variable for variable in instance.query if variable != instance.temporary_variable
    )
    output_bits = [0] * instance.num_registers
    for output_index, variable in enumerate(queried_variables):
        _validate_variable(variable, instance.num_registers)
        output_bits[variable] |= 1 << output_index

    total_affected_outputs = sum(
        bits.bit_count() for bits in output_bits[: instance.num_variables]
    )
    live_variables = sum(bits != 0 for bits in output_bits[: instance.num_variables])
    sample_boundaries = _sensitivity_sample_boundaries(instance, start=start)
    aggregate_affected_outputs = 0
    aggregate_live_variables = 0
    state_count = 0
    if len(instance.operations) in sample_boundaries:
        aggregate_affected_outputs += total_affected_outputs
        aggregate_live_variables += live_variables
        state_count += 1

    # Hoist attribute lookups and inline the reverse-live bitset updates; this
    # loop runs once per operation and dominates sensitivity's cost.
    operations = instance.operations
    num_registers = instance.num_registers
    num_variables = instance.num_variables
    for index in range(len(operations) - 1, -1, -1):
        operation = operations[index]
        target = operation.target
        if not 0 <= target < num_registers:
            raise ValueError(f"variable index {target} is outside [0, {num_registers})")
        if type(operation) is AssignConstant:
            previous = output_bits[target]
            if previous:
                output_bits[target] = 0
                if target < num_variables:
                    total_affected_outputs -= previous.bit_count()
                    live_variables -= 1
        elif type(operation) is AssignIncrement:
            pass
        else:
            source = operation.source
            if not 0 <= source < num_registers:
                raise ValueError(f"variable index {source} is outside [0, {num_registers})")
            if source != target:
                target_bits = output_bits[target]
                if target_bits:
                    output_bits[target] = 0
                    if target < num_variables:
                        total_affected_outputs -= target_bits.bit_count()
                        live_variables -= 1
                previous_source = output_bits[source]
                merged_source = previous_source | target_bits
                if merged_source != previous_source:
                    output_bits[source] = merged_source
                    if source < num_variables:
                        total_affected_outputs += merged_source.bit_count() - previous_source.bit_count()
                        live_variables += (merged_source != 0) - (previous_source != 0)
        if index in sample_boundaries:
            aggregate_affected_outputs += total_affected_outputs
            aggregate_live_variables += live_variables
            state_count += 1

    denominator = state_count * instance.num_variables
    output_count = len(queried_variables)
    return SensitivitySummary(
        sensitivity_state_count=state_count,
        sensitivity_output_count=output_count,
        sensitivity_density=(
            aggregate_affected_outputs / (denominator * output_count) if output_count else 0.0
        ),
        sensitivity_coverage=aggregate_live_variables / denominator,
        mean_live_fanout=(
            aggregate_affected_outputs / aggregate_live_variables
            if aggregate_live_variables
            else 0.0
        ),
    )


def _sensitivity_sample_boundaries(
    instance: StateTrackingInstance,
    *,
    start: int,
) -> frozenset[int]:
    """Return operation-prefix lengths representing complete semantic steps."""

    temporary = instance.temporary_variable
    operations = instance.operations
    length = len(operations)
    if temporary is None:
        return frozenset(range(start, length + 1))

    num_variables = instance.num_variables
    boundaries = {start}
    index = start
    while index < length:
        # Inline TMP-swap recognition (was _is_tmp_swap) to avoid a per-index
        # call and operation-slice; the fields are frozen dataclasses, so
        # ``type(...) is`` matches the previous isinstance checks exactly.
        is_swap = False
        if index + 2 < length:
            load = operations[index]
            replace = operations[index + 1]
            restore = operations[index + 2]
            left = replace.target
            right = restore.target
            is_swap = (
                load.target == temporary
                and 0 <= left < num_variables
                and 0 <= right < num_variables
                and left != right
                and (
                    (
                        type(load) is AssignConstant
                        and type(replace) is AssignConstant
                        and type(restore) is AssignConstant
                    )
                    or (
                        type(load) is AssignVariable
                        and type(replace) is AssignVariable
                        and type(restore) is AssignVariable
                        and load.source == left
                        and replace.source == right
                        and restore.source == temporary
                    )
                )
            )
        if is_swap:
            index += 3
            boundaries.add(index)
        elif operations[index].target == temporary:
            # Budget-padding scratch loads do not change ordinary state.
            index += 1
        else:
            index += 1
            boundaries.add(index)
    return frozenset(boundaries)


def _fully_initialized_boundary(*, num_variables: int, operations: Sequence[Operation]) -> int:
    """Return the first program boundary at which every variable has a value."""

    initialized: set[int] = set()
    for index, operation in enumerate(operations):
        _validate_variable(operation.target, num_variables)
        initialized.add(operation.target)
        if len(initialized) == num_variables:
            return index + 1
    raise ValueError("sensitivity requires every variable to be initialized")


def _validate_variable(variable: int, num_variables: int) -> None:
    if not 0 <= variable < num_variables:
        raise ValueError(f"variable index {variable} is outside [0, {num_variables})")
