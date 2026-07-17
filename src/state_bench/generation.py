"""Constrained generators for the three state-tracking families."""

from __future__ import annotations

import random

from state_bench.model import (
    AssignConstant,
    AssignVariable,
    Complexity,
    Operation,
    QueryMode,
    StateTrackingInstance,
)
from state_bench.semantics import evaluate


def generate_instance(
    *,
    complexity: Complexity,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int = 0,
    max_extra_assignments: int = 0,
    query_mode: QueryMode = QueryMode.ALL,
    distinct_initial_values: bool = False,
    require_strict_witness: bool = True,
    allow_self_reference: bool = False,
    periodic_constant_probability: float = 0.1,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate one evaluated instance satisfying its family constraints."""

    _validate_parameters(
        complexity=complexity,
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        periodic_constant_probability=periodic_constant_probability,
    )

    initial_values = _initial_values(
        num_variables=num_variables,
        num_values=num_values,
        distinct=distinct_initial_values,
        require_diversity=complexity is Complexity.PERIODIC and require_strict_witness,
        rng=rng,
    )
    variable_order = list(range(num_variables))
    rng.shuffle(variable_order)
    operations: list[Operation] = [
        AssignConstant(target=variable, value=initial_values[variable]) for variable in variable_order
    ]
    state = list(evaluate(num_variables=num_variables, num_values=num_values, operations=operations))
    dependency_depths = [0] * num_variables

    extra_count = rng.randint(min_extra_assignments, max_extra_assignments)
    if complexity is Complexity.APERIODIC:
        non_witness_count = extra_count - 1 if require_strict_witness else extra_count
        for _ in range(non_witness_count):
            operation = AssignConstant(target=rng.randrange(num_variables), value=rng.randrange(num_values))
            operations.append(operation)
            state[operation.target] = operation.value
        if require_strict_witness:
            target = rng.randrange(num_variables)
            value = rng.choice([value for value in range(num_values) if value != state[target]])
            witness = AssignConstant(target=target, value=value)
            operations.append(witness)
            _apply(witness, state)
    elif complexity is Complexity.PERIODIC:
        remaining_count = extra_count
        if require_strict_witness:
            target, source = rng.choice(
                [
                    (target, source)
                    for target in range(num_variables)
                    for source in range(num_variables)
                    if state[target] != state[source] and (allow_self_reference or target != source)
                ]
            )
            witness = AssignVariable(target=target, source=source)
            operations.append(witness)
            _apply(witness, state)
            _apply_dependency(witness, dependency_depths)
            remaining_count -= 1
        for _ in range(remaining_count):
            operation = _dependency_biased_periodic_operation(
                num_variables=num_variables,
                dependency_depths=dependency_depths,
                num_values=num_values,
                allow_self_reference=allow_self_reference,
                constant_probability=periodic_constant_probability,
                rng=rng,
            )
            operations.append(operation)
            _apply(operation, state)
            _apply_dependency(operation, dependency_depths)

    final_state = evaluate(num_variables=num_variables, num_values=num_values, operations=operations)
    query = _choose_query(num_variables=num_variables, query_mode=query_mode, rng=rng)
    return StateTrackingInstance.create(
        complexity=complexity,
        num_variables=num_variables,
        num_values=num_values,
        operations=tuple(operations),
        query=query,
        final_state=final_state,
    )


def _validate_parameters(
    *,
    complexity: Complexity,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int,
    max_extra_assignments: int,
    distinct_initial_values: bool,
    require_strict_witness: bool,
    periodic_constant_probability: float,
) -> None:
    if num_variables < 1 or num_values < 1:
        raise ValueError("num_variables and num_values must be positive")
    if min_extra_assignments < 0 or max_extra_assignments < min_extra_assignments:
        raise ValueError("extra assignment bounds must satisfy 0 <= min <= max")
    if distinct_initial_values and num_values < num_variables:
        raise ValueError("distinct initial values require num_values >= num_variables")
    if not 0 <= periodic_constant_probability <= 1:
        raise ValueError("periodic constant probability must satisfy 0 <= probability <= 1")
    if complexity is Complexity.R_TRIVIAL and max_extra_assignments != 0:
        raise ValueError("r-trivial instances cannot have extra assignments")
    if complexity is Complexity.APERIODIC and require_strict_witness and min_extra_assignments < 1:
        raise ValueError("strict aperiodic instances require at least one extra assignment")
    if complexity is Complexity.APERIODIC and require_strict_witness and num_values < 2:
        raise ValueError("strict aperiodic instances require at least two values")
    if complexity is Complexity.PERIODIC and require_strict_witness:
        if min_extra_assignments < 1:
            raise ValueError("strict periodic instances require at least one extra assignment")
        if num_variables < 2 or num_values < 2:
            raise ValueError("strict periodic instances require at least two variables and values")


def _initial_values(
    *,
    num_variables: int,
    num_values: int,
    distinct: bool,
    require_diversity: bool,
    rng: random.Random,
) -> list[int]:
    if distinct:
        return rng.sample(range(num_values), num_variables)
    values = [rng.randrange(num_values) for _ in range(num_variables)]
    if require_diversity and len(set(values)) == 1:
        values[rng.randrange(num_variables)] = (values[0] + 1) % num_values
    return values


def _dependency_biased_periodic_operation(
    *,
    num_variables: int,
    dependency_depths: list[int],
    num_values: int,
    allow_self_reference: bool,
    constant_probability: float,
    rng: random.Random,
) -> Operation:
    """Prefer extending the deepest live provenance chain into a shallow variable."""

    copy_possible = num_variables > 1 or allow_self_reference
    if rng.random() < constant_probability or not copy_possible:
        return AssignConstant(
            target=rng.randrange(num_variables),
            value=rng.randrange(num_values),
        )

    max_depth = max(dependency_depths)
    source = rng.choice(
        [index for index, depth in enumerate(dependency_depths) if depth == max_depth]
    )
    target_choices = [index for index in range(num_variables) if index != source]
    if not target_choices:
        target_choices = [source]
    min_target_depth = min(dependency_depths[index] for index in target_choices)
    target = rng.choice(
        [index for index in target_choices if dependency_depths[index] == min_target_depth]
    )
    return AssignVariable(target=target, source=source)


def _apply(operation: Operation, state: list[int]) -> None:
    if isinstance(operation, AssignConstant):
        state[operation.target] = operation.value
    else:
        state[operation.target] = state[operation.source]


def _apply_dependency(operation: Operation, dependency_depths: list[int]) -> None:
    if isinstance(operation, AssignConstant):
        dependency_depths[operation.target] = 0
    else:
        dependency_depths[operation.target] = dependency_depths[operation.source] + 1


def _choose_query(*, num_variables: int, query_mode: QueryMode, rng: random.Random) -> tuple[int, ...]:
    if query_mode is QueryMode.ALL:
        return tuple(range(num_variables))
    if query_mode is QueryMode.ONE:
        return (rng.randrange(num_variables),)
    size = rng.randint(1, num_variables)
    return tuple(sorted(rng.sample(range(num_variables), size)))
