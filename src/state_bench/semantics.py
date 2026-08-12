"""Evaluation and validation for canonical assignment traces."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from state_bench.model import AssignConstant, AssignVariable, Operation, StateTrackingInstance


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
    """Evaluate eager, copy-at-assignment-time variable semantics."""

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

    uninitialized = [index for index, value in enumerate(state) if value is None]
    if uninitialized:
        raise ValueError(f"final state has uninitialized variables: {uninitialized}")
    return tuple(value for value in state if value is not None)


def dependency_lengths(
    *,
    num_variables: int,
    operations: Sequence[Operation],
) -> tuple[int, ...]:
    """Return each final variable's eager-copy hop count back to a constant assignment."""

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
    and kills them at constant assignments.
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

    for index in range(len(instance.operations) - 1, -1, -1):
        operation = instance.operations[index]
        _validate_variable(operation.target, instance.num_registers)
        if isinstance(operation, AssignConstant):
            total_affected_outputs, live_variables = _replace_influence(
                output_bits,
                operation.target,
                0,
                included=operation.target < instance.num_variables,
                total_affected_outputs=total_affected_outputs,
                live_variables=live_variables,
            )
        else:
            _validate_variable(operation.source, instance.num_registers)
            if operation.source != operation.target:
                target_bits = output_bits[operation.target]
                total_affected_outputs, live_variables = _replace_influence(
                    output_bits,
                    operation.target,
                    0,
                    included=operation.target < instance.num_variables,
                    total_affected_outputs=total_affected_outputs,
                    live_variables=live_variables,
                )
                total_affected_outputs, live_variables = _replace_influence(
                    output_bits,
                    operation.source,
                    output_bits[operation.source] | target_bits,
                    included=operation.source < instance.num_variables,
                    total_affected_outputs=total_affected_outputs,
                    live_variables=live_variables,
                )
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

    if instance.temporary_variable is None:
        return frozenset(range(start, len(instance.operations) + 1))

    boundaries = {start}
    index = start
    while index < len(instance.operations):
        if _is_tmp_swap(instance, index=index):
            index += 3
            boundaries.add(index)
        elif instance.operations[index].target == instance.temporary_variable:
            # Budget-padding scratch loads do not change ordinary state.
            index += 1
        else:
            index += 1
            boundaries.add(index)
    return frozenset(boundaries)


def _is_tmp_swap(instance: StateTrackingInstance, *, index: int) -> bool:
    """Recognize one TMP-expanded swap in a source or structurally derived trace."""

    if instance.temporary_variable is None or index + 2 >= len(instance.operations):
        return False
    load, replace, restore = instance.operations[index : index + 3]
    left = replace.target
    right = restore.target
    if not (
        load.target == instance.temporary_variable
        and 0 <= left < instance.num_variables
        and 0 <= right < instance.num_variables
        and left != right
    ):
        return False
    if all(isinstance(operation, AssignConstant) for operation in (load, replace, restore)):
        return True
    return (
        isinstance(load, AssignVariable)
        and isinstance(replace, AssignVariable)
        and isinstance(restore, AssignVariable)
        and load.source == left
        and replace.source == right
        and restore.source == instance.temporary_variable
    )


def _fully_initialized_boundary(*, num_variables: int, operations: Sequence[Operation]) -> int:
    """Return the first program boundary at which every variable has a value."""

    initialized: set[int] = set()
    for index, operation in enumerate(operations):
        _validate_variable(operation.target, num_variables)
        initialized.add(operation.target)
        if len(initialized) == num_variables:
            return index + 1
    raise ValueError("sensitivity requires every variable to be initialized")


def _replace_influence(
    output_bits: list[int],
    variable: int,
    replacement: int,
    *,
    included: bool,
    total_affected_outputs: int,
    live_variables: int,
) -> tuple[int, int]:
    """Replace one reverse-live bitset while maintaining aggregate counters."""

    previous = output_bits[variable]
    if previous == replacement:
        return total_affected_outputs, live_variables
    output_bits[variable] = replacement
    if not included:
        return total_affected_outputs, live_variables
    return (
        total_affected_outputs + replacement.bit_count() - previous.bit_count(),
        live_variables + (replacement != 0) - (previous != 0),
    )


def _validate_variable(variable: int, num_variables: int) -> None:
    if not 0 <= variable < num_variables:
        raise ValueError(f"variable index {variable} is outside [0, {num_variables})")
