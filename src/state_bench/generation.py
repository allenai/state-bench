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
    """Generate a periodic source trace and return the requested derivation."""

    source = generate_periodic_source(
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        query_mode=query_mode,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        allow_self_reference=allow_self_reference,
        periodic_constant_probability=periodic_constant_probability,
        rng=rng,
    )
    return derive_instance(source, complexity=complexity)


def generate_periodic_source(
    *,
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
    """Generate the periodic source-of-truth for a three-family trace set."""

    _validate_parameters(
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
        require_diversity=require_strict_witness,
        rng=rng,
    )
    variable_order = list(range(num_variables))
    rng.shuffle(variable_order)
    operations: list[Operation] = [
        AssignConstant(target=variable, value=initial_values[variable]) for variable in variable_order
    ]
    state = list(initial_values)
    dependency_depths = [0] * num_variables
    target_periodic_diversity = len(set(state))

    extra_count = rng.randint(min_extra_assignments, max_extra_assignments)
    remaining_count = extra_count
    if require_strict_witness:
        witness = _choose_periodic_copy(
            state=state,
            desired_diversity=target_periodic_diversity,
            dependency_depths=dependency_depths,
            exact=True,
            rng=rng,
        )
        if witness is None:
            witness = _choose_periodic_copy(
                state=state,
                desired_diversity=target_periodic_diversity,
                dependency_depths=dependency_depths,
                exact=None,
                rng=rng,
            )
        assert witness is not None
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

    query = _choose_query(num_variables=num_variables, query_mode=query_mode, rng=rng)
    return StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=num_variables,
        num_values=num_values,
        operations=tuple(operations),
        query=query,
        final_state=tuple(state),
    )


def derive_instance(
    source: StateTrackingInstance,
    *,
    complexity: Complexity,
) -> StateTrackingInstance:
    """Derive a complexity variant without changing source trace length or query.

    Aperiodic traces eagerly resolve every reference to the source variable's
    value at that point while retaining literal assignments. R-trivial traces
    replace every assignment after a target's first assignment with that
    target's first value, regardless of whether the source assignment was a
    literal or a reference.
    """

    if source.complexity is not Complexity.PERIODIC:
        raise ValueError("derived instances require a periodic source")
    if complexity is Complexity.PERIODIC:
        return source

    source_state: list[int | None] = [None] * source.num_variables
    first_values: list[int | None] = [None] * source.num_variables
    operations: list[Operation] = []
    for operation in source.operations:
        if isinstance(operation, AssignConstant):
            source_value = operation.value
        else:
            source_value = source_state[operation.source]
            if source_value is None:
                raise ValueError(f"source variable {operation.source} is uninitialized")

        if complexity is Complexity.APERIODIC:
            derived = AssignConstant(target=operation.target, value=source_value)
        else:
            first_value = first_values[operation.target]
            if first_value is None:
                first_value = source_value
                first_values[operation.target] = first_value
            derived = AssignConstant(target=operation.target, value=first_value)

        operations.append(derived)
        source_state[operation.target] = source_value

    operation_tuple = tuple(operations)
    if complexity is Complexity.APERIODIC:
        final_state = source.final_state
    else:
        final_state = tuple(value for value in first_values if value is not None)
        if len(final_state) != source.num_variables:
            raise ValueError("source trace leaves a variable uninitialized")
    return StateTrackingInstance.create(
        complexity=complexity,
        num_variables=source.num_variables,
        num_values=source.num_values,
        operations=operation_tuple,
        query=source.query,
        final_state=final_state,
    )


def derive_variants(source: StateTrackingInstance) -> tuple[StateTrackingInstance, ...]:
    """Derive all families together with one traversal of the periodic trace."""

    if source.complexity is not Complexity.PERIODIC:
        raise ValueError("derived instances require a periodic source")

    source_state: list[int | None] = [None] * source.num_variables
    first_values: list[int | None] = [None] * source.num_variables
    aperiodic_operations: list[Operation] = []
    r_trivial_operations: list[Operation] = []
    for operation in source.operations:
        if isinstance(operation, AssignConstant):
            source_value = operation.value
        else:
            source_value = source_state[operation.source]
            if source_value is None:
                raise ValueError(f"source variable {operation.source} is uninitialized")

        aperiodic_operations.append(AssignConstant(target=operation.target, value=source_value))
        first_value = first_values[operation.target]
        if first_value is None:
            first_value = source_value
            first_values[operation.target] = first_value
        r_trivial_operations.append(AssignConstant(target=operation.target, value=first_value))
        source_state[operation.target] = source_value

    r_trivial_state = tuple(value for value in first_values if value is not None)
    if len(r_trivial_state) != source.num_variables:
        raise ValueError("source trace leaves a variable uninitialized")
    derived = {
        Complexity.R_TRIVIAL: StateTrackingInstance.create(
            complexity=Complexity.R_TRIVIAL,
            num_variables=source.num_variables,
            num_values=source.num_values,
            operations=tuple(r_trivial_operations),
            query=source.query,
            final_state=r_trivial_state,
        ),
        Complexity.APERIODIC: StateTrackingInstance.create(
            complexity=Complexity.APERIODIC,
            num_variables=source.num_variables,
            num_values=source.num_values,
            operations=tuple(aperiodic_operations),
            query=source.query,
            final_state=source.final_state,
        ),
        Complexity.PERIODIC: source,
    }
    return tuple(derived[complexity] for complexity in Complexity)


def _validate_parameters(
    *,
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
    if require_strict_witness:
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

    exact_copy = _choose_periodic_copy(
        state=state,
        desired_diversity=desired_diversity,
        dependency_depths=dependency_depths,
        exact=True,
        rng=rng,
        choose=False,
    )
    exact_constant = _choose_periodic_constant(
        state=state,
        num_values=num_values,
        desired_diversity=desired_diversity,
        dependency_depths=dependency_depths,
        rng=rng,
        choose=False,
    )
    current_diversity = len(set(state))

    if current_diversity != desired_diversity:
        if exact_constant is not None:
            chosen = _choose_periodic_constant(
                state=state,
                num_values=num_values,
                desired_diversity=desired_diversity,
                dependency_depths=dependency_depths,
                rng=rng,
            )
            assert chosen is not None
            return chosen
        if exact_copy is not None:
            chosen = _choose_periodic_copy(
                state=state,
                desired_diversity=desired_diversity,
                dependency_depths=dependency_depths,
                exact=True,
                rng=rng,
            )
            assert chosen is not None
            return chosen

    choose_constant = rng.random() < constant_probability
    if choose_constant and exact_constant is not None:
        chosen = _choose_periodic_constant(
            state=state,
            num_values=num_values,
            desired_diversity=desired_diversity,
            dependency_depths=dependency_depths,
            rng=rng,
        )
        assert chosen is not None
        return chosen
    if exact_copy is not None:
        chosen = _choose_periodic_copy(
            state=state,
            desired_diversity=desired_diversity,
            dependency_depths=dependency_depths,
            exact=True,
            rng=rng,
        )
        assert chosen is not None
        return chosen
    any_copy = _choose_periodic_copy(
        state=state,
        desired_diversity=desired_diversity,
        dependency_depths=dependency_depths,
        exact=None,
        rng=rng,
        choose=False,
    )
    if any_copy is not None and remaining_after > 0 and not choose_constant:
        chosen = _choose_periodic_copy(
            state=state,
            desired_diversity=desired_diversity,
            dependency_depths=dependency_depths,
            exact=None,
            rng=rng,
        )
        assert chosen is not None
        return chosen
    if exact_constant is not None:
        chosen = _choose_periodic_constant(
            state=state,
            num_values=num_values,
            desired_diversity=desired_diversity,
            dependency_depths=dependency_depths,
            rng=rng,
        )
        assert chosen is not None
        return chosen

    # Unusual non-strict configurations can begin with only one live value, or
    # request as many distinct values as the value vocabulary contains. Choose
    # the operation that gets closest to the target and let subsequent steps
    # repair the state.
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


def _choose_periodic_copy(
    *,
    state: list[int],
    desired_diversity: int,
    dependency_depths: list[int],
    exact: bool | None,
    rng: random.Random,
    choose: bool = True,
) -> AssignVariable | None:
    """Choose a dependency-biased copy without materializing O(n²) candidates."""

    value_counts = Counter(state)
    current_diversity = len(value_counts)
    eligible_targets = [
        target
        for target, value in enumerate(state)
        if exact is None
        or (current_diversity - (value_counts[value] == 1) == desired_diversity) is exact
    ]
    if not eligible_targets:
        return None

    target_values = {state[target] for target in eligible_targets}
    max_source_depth = max(
        (
            dependency_depths[source]
            for source, value in enumerate(state)
            if len(target_values) > 1 or value not in target_values
        ),
        default=None,
    )
    if max_source_depth is None:
        return None
    source_counts: Counter[int] = Counter(
        state[source]
        for source in range(len(state))
        if dependency_depths[source] == max_source_depth
    )
    total_sources = sum(source_counts.values())
    eligible_targets = [
        target
        for target in eligible_targets
        if total_sources > source_counts[state[target]]
    ]
    if not eligible_targets:
        return None
    min_target_depth = min(dependency_depths[target] for target in eligible_targets)
    eligible_targets = [
        target for target in eligible_targets if dependency_depths[target] == min_target_depth
    ]
    if not choose:
        target = eligible_targets[0]
        source = next(
            source
            for source, value in enumerate(state)
            if dependency_depths[source] == max_source_depth and value != state[target]
        )
        return AssignVariable(target=target, source=source)

    candidate_count = sum(total_sources - source_counts[state[target]] for target in eligible_targets)
    selected = rng.randrange(candidate_count)
    for target in eligible_targets:
        target_count = total_sources - source_counts[state[target]]
        if selected >= target_count:
            selected -= target_count
            continue
        for source, value in enumerate(state):
            if dependency_depths[source] != max_source_depth or value == state[target]:
                continue
            if selected == 0:
                return AssignVariable(target=target, source=source)
            selected -= 1
    raise AssertionError("copy candidate selection fell out of range")


def _choose_periodic_constant(
    *,
    state: list[int],
    num_values: int,
    desired_diversity: int,
    dependency_depths: list[int],
    rng: random.Random,
    choose: bool = True,
) -> AssignConstant | None:
    """Choose an exact diversity-preserving constant in O(n + num_values)."""

    value_counts = Counter(state)
    current_diversity = len(value_counts)
    present_values = set(value_counts)
    absent_count = num_values - current_diversity

    def candidate_count(target: int) -> int:
        required_absent = (
            desired_diversity - current_diversity + (value_counts[state[target]] == 1)
        )
        if required_absent == 1:
            return absent_count
        if required_absent == 0:
            return current_diversity - 1
        return 0

    eligible_targets = [target for target in range(len(state)) if candidate_count(target)]
    if not eligible_targets:
        return None
    min_target_depth = min(dependency_depths[target] for target in eligible_targets)
    eligible_targets = [
        target for target in eligible_targets if dependency_depths[target] == min_target_depth
    ]
    if not choose:
        target = eligible_targets[0]
        required_absent = (
            desired_diversity - current_diversity + (value_counts[state[target]] == 1)
        )
        value = next(
            value
            for value in range(num_values)
            if value != state[target]
            and ((value not in present_values) == bool(required_absent))
        )
        return AssignConstant(target=target, value=value)

    selected = rng.randrange(sum(candidate_count(target) for target in eligible_targets))
    for target in eligible_targets:
        count = candidate_count(target)
        if selected >= count:
            selected -= count
            continue
        required_absent = (
            desired_diversity - current_diversity + (value_counts[state[target]] == 1)
        )
        for value in range(num_values):
            if value == state[target] or ((value not in present_values) != bool(required_absent)):
                continue
            if selected == 0:
                return AssignConstant(target=target, value=value)
            selected -= 1
    raise AssertionError("constant candidate selection fell out of range")


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
