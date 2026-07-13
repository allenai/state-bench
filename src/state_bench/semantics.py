"""Evaluation and validation for canonical assignment traces."""

from __future__ import annotations

from collections.abc import Sequence

from state_bench.model import AssignConstant, AssignVariable, Operation


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


def _validate_variable(variable: int, num_variables: int) -> None:
    if not 0 <= variable < num_variables:
        raise ValueError(f"variable index {variable} is outside [0, {num_variables})")
