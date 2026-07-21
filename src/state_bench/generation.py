"""Constrained generators for the three state-tracking families."""

from __future__ import annotations

import random
from collections import Counter

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
    target_periodic_diversity = len(set(state))

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
            copy_candidates = _periodic_copy_candidates(
                state=state,
                desired_diversity=target_periodic_diversity,
            )
            diversity_preserving = [candidate for candidate in copy_candidates if candidate[2]]
            witness = _choose_dependency_biased_copy(
                diversity_preserving or copy_candidates,
                dependency_depths=dependency_depths,
                rng=rng,
            )
            operations.append(witness)
            _apply(witness, state)
            _apply_dependency(witness, dependency_depths)
            remaining_count -= 1
        for index in range(remaining_count):
            operation = _value_aware_periodic_operation(
                state=state,
                dependency_depths=dependency_depths,
                num_values=num_values,
                allow_self_reference=allow_self_reference,
                constant_probability=periodic_constant_probability,
                desired_diversity=target_periodic_diversity,
                remaining_after=remaining_count - index - 1,
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


def _value_aware_periodic_operation(
    *,
    state: list[int],
    dependency_depths: list[int],
    num_values: int,
    allow_self_reference: bool,
    constant_probability: float,
    desired_diversity: int,
    remaining_after: int,
    rng: random.Random,
) -> Operation:
    """Choose a value-changing update while retaining diverse live lineages.

    Copy candidates normally preserve the initial number of distinct live
    values. When that is impossible (notably with two variables), a copy may
    temporarily collapse the state only if a later operation can restore its
    diversity. Constants steer the trace back to the desired diversity and
    preferentially reset shallow variables so deep provenance survives.
    """

    copy_candidates = _periodic_copy_candidates(
        state=state,
        desired_diversity=desired_diversity,
    )
    if not allow_self_reference:
        copy_candidates = [
            candidate for candidate in copy_candidates if candidate[0] != candidate[1]
        ]
    constant_candidates = _periodic_constant_candidates(
        state=state,
        num_values=num_values,
        desired_diversity=desired_diversity,
    )
    exact_copies = [candidate for candidate in copy_candidates if candidate[2]]
    exact_constants = [candidate for candidate in constant_candidates if candidate[1]]
    current_diversity = len(set(state))

    if current_diversity != desired_diversity:
        if exact_constants:
            return _choose_dependency_preserving_constant(
                exact_constants,
                dependency_depths=dependency_depths,
                rng=rng,
            )
        if exact_copies:
            return _choose_dependency_biased_copy(
                exact_copies,
                dependency_depths=dependency_depths,
                rng=rng,
            )

    choose_constant = rng.random() < constant_probability
    if choose_constant and exact_constants:
        return _choose_dependency_preserving_constant(
            exact_constants,
            dependency_depths=dependency_depths,
            rng=rng,
        )
    if exact_copies:
        return _choose_dependency_biased_copy(
            exact_copies,
            dependency_depths=dependency_depths,
            rng=rng,
        )
    if copy_candidates and remaining_after > 0 and not choose_constant:
        return _choose_dependency_biased_copy(
            copy_candidates,
            dependency_depths=dependency_depths,
            rng=rng,
        )
    if exact_constants:
        return _choose_dependency_preserving_constant(
            exact_constants,
            dependency_depths=dependency_depths,
            rng=rng,
        )

    # Unusual non-strict configurations can begin with only one live value, or
    # request as many distinct values as the value vocabulary contains. Choose
    # the operation that gets closest to the target and let subsequent steps
    # repair the state.
    candidates: list[tuple[Operation, int]] = [
        (AssignVariable(target=target, source=source), result_diversity)
        for target, source, _, result_diversity in copy_candidates
    ]
    candidates.extend(
        (operation, result_diversity)
        for operation, _, result_diversity in constant_candidates
    )
    if not candidates:
        return AssignConstant(
            target=rng.randrange(len(state)),
            value=rng.randrange(num_values),
        )
    closest_distance = min(
        abs(result_diversity - desired_diversity) for _, result_diversity in candidates
    )
    closest = [
        operation
        for operation, result_diversity in candidates
        if abs(result_diversity - desired_diversity) == closest_distance
    ]
    return rng.choice(closest)


def _periodic_copy_candidates(
    *,
    state: list[int],
    desired_diversity: int,
) -> list[tuple[int, int, bool, int]]:
    """Return changing ``(target, source, is_exact, result_diversity)`` copies."""

    value_counts = Counter(state)
    current_diversity = len(value_counts)
    candidates = []
    for target in range(len(state)):
        for source in range(len(state)):
            if target == source or state[target] == state[source]:
                continue
            result_diversity = current_diversity - (value_counts[state[target]] == 1)
            candidates.append(
                (target, source, result_diversity == desired_diversity, result_diversity)
            )
    return candidates


def _periodic_constant_candidates(
    *,
    state: list[int],
    num_values: int,
    desired_diversity: int,
) -> list[tuple[AssignConstant, bool, int]]:
    """Return changing constants and their effect on live-value diversity."""

    value_counts = Counter(state)
    current_diversity = len(value_counts)
    candidates = []
    for target in range(len(state)):
        for value in range(num_values):
            if value == state[target]:
                continue
            result_diversity = (
                current_diversity
                - (value_counts[state[target]] == 1)
                + (value_counts[value] == 0)
            )
            candidates.append(
                (
                    AssignConstant(target=target, value=value),
                    result_diversity == desired_diversity,
                    result_diversity,
                )
            )
    return candidates


def _choose_dependency_biased_copy(
    candidates: list[tuple[int, int, bool, int]],
    *,
    dependency_depths: list[int],
    rng: random.Random,
) -> AssignVariable:
    max_source_depth = max(dependency_depths[source] for _, source, _, _ in candidates)
    deepest = [
        candidate
        for candidate in candidates
        if dependency_depths[candidate[1]] == max_source_depth
    ]
    min_target_depth = min(dependency_depths[target] for target, _, _, _ in deepest)
    target, source, _, _ = rng.choice(
        [
            candidate
            for candidate in deepest
            if dependency_depths[candidate[0]] == min_target_depth
        ]
    )
    return AssignVariable(target=target, source=source)


def _choose_dependency_preserving_constant(
    candidates: list[tuple[AssignConstant, bool, int]],
    *,
    dependency_depths: list[int],
    rng: random.Random,
) -> AssignConstant:
    min_target_depth = min(dependency_depths[operation.target] for operation, _, _ in candidates)
    return rng.choice(
        [
            operation
            for operation, _, _ in candidates
            if dependency_depths[operation.target] == min_target_depth
        ]
    )


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
