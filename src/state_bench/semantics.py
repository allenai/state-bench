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

    lengths = dependency_lengths(
        num_variables=instance.num_variables,
        operations=instance.operations,
    )
    frequencies = Counter(lengths)
    highest_frequency = max(frequencies.values())
    mode = min(depth for depth, count in frequencies.items() if count == highest_frequency)
    return DependencySummary(
        final_dependency_lengths=lengths,
        max_dependency_length=max(lengths),
        mean_dependency_length=statistics.fmean(lengths),
        modal_dependency_length=mode,
    )


def _validate_variable(variable: int, num_variables: int) -> None:
    if not 0 <= variable < num_variables:
        raise ValueError(f"variable index {variable} is outside [0, {num_variables})")
