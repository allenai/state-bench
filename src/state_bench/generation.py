"""Constrained generators for the state-tracking families.

Two families act as *sources*: ``periodic`` (TMP-mediated swaps of variable
contents) and ``solvable`` (cyclic increments of single variables). Each source
derives the simpler ``aperiodic`` and ``r-trivial`` families from its own trace
without changing trace length, targets, or query.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from state_bench.model import (
    AssignConstant,
    AssignIncrement,
    AssignVariable,
    Complexity,
    Operation,
    QueryMode,
    StateTrackingInstance,
)

_SOURCE_FAMILIES: dict[Complexity, tuple[Complexity, ...]] = {
    Complexity.PERIODIC: (Complexity.R_TRIVIAL, Complexity.APERIODIC, Complexity.PERIODIC),
    Complexity.SOLVABLE: (Complexity.R_TRIVIAL, Complexity.APERIODIC, Complexity.SOLVABLE),
}

SOURCE_COMPLEXITIES: tuple[Complexity, ...] = tuple(_SOURCE_FAMILIES)


def derived_families(source_complexity: Complexity) -> tuple[Complexity, ...]:
    """Return, in emission order, every family derived from a source of this class."""

    try:
        return _SOURCE_FAMILIES[source_complexity]
    except KeyError:
        raise ValueError(
            f"{source_complexity.value!r} is not a source family; "
            f"sources are {[item.value for item in SOURCE_COMPLEXITIES]}"
        ) from None


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
    min_aperiodic_constants: int = 1,
    max_aperiodic_constants: int = 64,
    min_increment_delta: int = 1,
    max_increment_delta: int | None = None,
    source_complexity: Complexity | None = None,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate a source trace and return the requested derivation.

    The source defaults to ``periodic`` unless ``solvable`` itself is requested,
    in which case a solvable source is generated. Pass ``source_complexity`` to
    derive ``aperiodic`` or ``r-trivial`` from a solvable source instead.
    """

    if source_complexity is None:
        source_complexity = (
            Complexity.SOLVABLE if complexity is Complexity.SOLVABLE else Complexity.PERIODIC
        )
    source = generate_source(
        complexity=source_complexity,
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        query_mode=query_mode,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
        min_increment_delta=min_increment_delta,
        max_increment_delta=max_increment_delta,
        rng=rng,
    )
    return derive_instance(source, complexity=complexity)


def generate_source(
    *,
    complexity: Complexity,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int = 0,
    max_extra_assignments: int = 0,
    query_mode: QueryMode = QueryMode.ALL,
    distinct_initial_values: bool = False,
    require_strict_witness: bool = True,
    min_aperiodic_constants: int = 1,
    max_aperiodic_constants: int = 64,
    min_increment_delta: int = 1,
    max_increment_delta: int | None = None,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate the source-of-truth trace for one linked family set."""

    if complexity not in _SOURCE_FAMILIES:
        raise ValueError(
            f"{complexity.value!r} is not a source family; "
            f"sources are {[item.value for item in SOURCE_COMPLEXITIES]}"
        )
    _validate_parameters(
        complexity=complexity,
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
    )
    if complexity is Complexity.PERIODIC:
        return _generate_swap_periodic_source(
            num_variables=num_variables,
            num_values=num_values,
            min_extra_assignments=min_extra_assignments,
            max_extra_assignments=max_extra_assignments,
            min_aperiodic_constants=min_aperiodic_constants,
            max_aperiodic_constants=max_aperiodic_constants,
            query_mode=query_mode,
            distinct_initial_values=distinct_initial_values,
            rng=rng,
        )
    resolved_max_delta = num_values - 1 if max_increment_delta is None else max_increment_delta
    _validate_increment_deltas(
        num_values=num_values,
        min_increment_delta=min_increment_delta,
        max_increment_delta=resolved_max_delta,
    )
    return _generate_increment_solvable_source(
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
        min_increment_delta=min_increment_delta,
        max_increment_delta=resolved_max_delta,
        query_mode=query_mode,
        distinct_initial_values=distinct_initial_values,
        rng=rng,
    )


def generate_periodic_source(
    *,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int = 0,
    max_extra_assignments: int = 0,
    query_mode: QueryMode = QueryMode.ALL,
    distinct_initial_values: bool = False,
    require_strict_witness: bool = True,
    min_aperiodic_constants: int = 1,
    max_aperiodic_constants: int = 64,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate the periodic source-of-truth for a three-family trace set."""

    return generate_source(
        complexity=Complexity.PERIODIC,
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        query_mode=query_mode,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
        rng=rng,
    )


def derive_instance(
    source: StateTrackingInstance,
    *,
    complexity: Complexity,
) -> StateTrackingInstance:
    """Derive a complexity variant without changing source trace length or query.

    Aperiodic traces eagerly resolve every reference or increment to the value
    it produces at that point while retaining literal assignments. R-trivial
    traces replace every assignment after a target's first assignment with that
    target's first value, regardless of the source operation's kind.
    """

    families = derived_families(source.complexity)
    if complexity is source.complexity:
        return source
    if complexity not in families:
        raise ValueError(
            f"cannot derive {complexity.value!r} from a {source.complexity.value!r} source"
        )

    source_state: list[int | None] = [None] * source.num_registers
    first_values: list[int | None] = [None] * source.num_registers
    operations: list[Operation] = []
    for operation in source.operations:
        source_value = _resolved_value(operation, source_state, source.num_values)

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
        if len(final_state) != source.num_registers:
            raise ValueError("source trace leaves a variable uninitialized")
    return StateTrackingInstance.create(
        complexity=complexity,
        num_variables=source.num_variables,
        num_values=source.num_values,
        operations=operation_tuple,
        query=source.query,
        final_state=final_state,
        temporary_variable=source.temporary_variable,
    )


def derive_variants(source: StateTrackingInstance) -> tuple[StateTrackingInstance, ...]:
    """Derive all families together with one traversal of the source trace.

    The result is ordered like ``derived_families(source.complexity)``.
    """

    families = derived_families(source.complexity)

    source_state: list[int | None] = [None] * source.num_registers
    first_values: list[int | None] = [None] * source.num_registers
    aperiodic_operations: list[Operation] = []
    r_trivial_operations: list[Operation] = []
    for operation in source.operations:
        source_value = _resolved_value(operation, source_state, source.num_values)

        aperiodic_operations.append(AssignConstant(target=operation.target, value=source_value))
        first_value = first_values[operation.target]
        if first_value is None:
            first_value = source_value
            first_values[operation.target] = first_value
        r_trivial_operations.append(AssignConstant(target=operation.target, value=first_value))
        source_state[operation.target] = source_value

    r_trivial_state = tuple(value for value in first_values if value is not None)
    if len(r_trivial_state) != source.num_registers:
        raise ValueError("source trace leaves a variable uninitialized")
    derived = {
        Complexity.R_TRIVIAL: StateTrackingInstance.create(
            complexity=Complexity.R_TRIVIAL,
            num_variables=source.num_variables,
            num_values=source.num_values,
            operations=tuple(r_trivial_operations),
            query=source.query,
            final_state=r_trivial_state,
            temporary_variable=source.temporary_variable,
        ),
        Complexity.APERIODIC: StateTrackingInstance.create(
            complexity=Complexity.APERIODIC,
            num_variables=source.num_variables,
            num_values=source.num_values,
            operations=tuple(aperiodic_operations),
            query=source.query,
            final_state=source.final_state,
            temporary_variable=source.temporary_variable,
        ),
        source.complexity: source,
    }
    return tuple(derived[complexity] for complexity in families)


def _resolved_value(operation: Operation, state: list[int | None], num_values: int) -> int:
    """Return the value an operation writes given the state before it runs."""

    if isinstance(operation, AssignConstant):
        return operation.value
    if isinstance(operation, AssignVariable):
        source_value = state[operation.source]
        if source_value is None:
            raise ValueError(f"source variable {operation.source} is uninitialized")
        return source_value
    current = state[operation.target]
    if current is None:
        raise ValueError(f"incremented variable {operation.target} is uninitialized")
    return (current + operation.delta) % num_values


def _validate_parameters(
    *,
    complexity: Complexity,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int,
    max_extra_assignments: int,
    distinct_initial_values: bool,
    require_strict_witness: bool,
    min_aperiodic_constants: int,
    max_aperiodic_constants: int,
) -> None:
    if num_variables < 1 or num_values < 1:
        raise ValueError("num_variables and num_values must be positive")
    if min_extra_assignments < 0 or max_extra_assignments < min_extra_assignments:
        raise ValueError("extra assignment bounds must satisfy 0 <= min <= max")
    if distinct_initial_values and num_values < num_variables:
        raise ValueError("distinct initial values require num_values >= num_variables")
    if min_aperiodic_constants < 0 or max_aperiodic_constants < min_aperiodic_constants:
        raise ValueError("aperiodic constant bounds must satisfy 0 <= min <= max")
    if min_aperiodic_constants != max_aperiodic_constants and min_aperiodic_constants < 1:
        raise ValueError(
            "aperiodic constant ranges are sampled log-uniformly and must not include zero; "
            "use min = max = 0 for pure periodic traces"
        )
    if complexity is Complexity.PERIODIC:
        if require_strict_witness:
            if max_extra_assignments < 3:
                raise ValueError(
                    "strict swap-periodic instances require room for three extra assignments"
                )
            if num_variables < 2 or num_values < 2:
                raise ValueError("strict periodic instances require at least two variables and values")
    elif complexity is Complexity.SOLVABLE:
        if num_values < 2:
            raise ValueError("solvable instances require at least two values to increment through")
        if require_strict_witness and max_extra_assignments < 1:
            raise ValueError("strict solvable instances require room for one extra assignment")


def _validate_increment_deltas(
    *,
    num_values: int,
    min_increment_delta: int,
    max_increment_delta: int,
) -> None:
    if min_increment_delta < 1:
        raise ValueError("increment deltas must be positive; zero is the identity")
    if max_increment_delta < min_increment_delta:
        raise ValueError("increment delta bounds must satisfy 1 <= min <= max")
    if max_increment_delta >= num_values:
        raise ValueError(
            f"increment deltas must be smaller than num_values={num_values} so every "
            "increment changes its target"
        )


def _generate_swap_periodic_source(
    *,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int,
    max_extra_assignments: int,
    min_aperiodic_constants: int,
    max_aperiodic_constants: int,
    query_mode: QueryMode,
    distinct_initial_values: bool,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate a one-pool program whose real variables are permuted via TMP.

    Every swap is emitted as ``TMP <- a; a <- b; b <- TMP``. Configured
    ``num_variables`` counts only real variables; the scratch register lives at
    the following index. Constant mix-ins target real variables only and are
    placed early enough that balanced later swaps can rebuild their chains.
    The mix-in count is sampled log-uniformly per instance because sensitivity
    coverage is governed by the absolute count, independent of trace length.
    """

    extra_count = rng.randint(max(3, min_extra_assignments), max_extra_assignments)
    requested_constants = _sample_aperiodic_constant_count(
        minimum=min_aperiodic_constants,
        maximum=max_aperiodic_constants,
        rng=rng,
    )
    constant_count = min(requested_constants, extra_count - 3)
    reference_count = extra_count - constant_count
    swap_count = reference_count // 3
    scratch_load_count = reference_count - 3 * swap_count
    if swap_count < 1:
        raise ValueError("swap-periodic generation requires room for at least one complete swap")

    temporary = num_variables
    initial_real_values = _diverse_initial_values(
        num_variables=num_variables,
        num_values=num_values,
        distinct=distinct_initial_values,
        rng=rng,
    )
    state = [*initial_real_values, rng.randrange(num_values)]
    initial_order = list(range(num_variables + 1))
    rng.shuffle(initial_order)
    operations: list[Operation] = [
        AssignConstant(target=variable, value=state[variable]) for variable in initial_order
    ]

    pairs = _balanced_swap_pairs(
        num_variables=num_variables,
        count=swap_count,
        rng=rng,
    )
    future_swaps = [0] * num_variables
    for left, right in pairs:
        future_swaps[left] += 1
        future_swaps[right] += 1

    # Scratch-only prefix operations consume assignment budgets that are not a
    # multiple of three without changing the real-variable permutation.
    for _ in range(scratch_load_count):
        source = rng.randrange(num_variables)
        operation = AssignVariable(target=temporary, source=source)
        operations.append(operation)
        _apply(operation, state, num_values)

    injection_counts = _constant_injection_counts(
        constant_count=constant_count,
        update_count=swap_count,
    )
    reset_counts = [0] * num_variables

    for swap_index, (left, right) in enumerate(pairs):
        for _ in range(injection_counts[swap_index]):
            operation = _choose_constant_reset(
                state=state,
                num_variables=num_variables,
                num_values=num_values,
                future_updates=future_swaps,
                reset_counts=reset_counts,
                rng=rng,
            )
            operations.append(operation)
            _apply(operation, state, num_values)
            reset_counts[operation.target] += 1

        swap_operations = (
            AssignVariable(target=temporary, source=left),
            AssignVariable(target=left, source=right),
            AssignVariable(target=right, source=temporary),
        )
        for operation in swap_operations:
            operations.append(operation)
            _apply(operation, state, num_values)
        future_swaps[left] -= 1
        future_swaps[right] -= 1

    query = _choose_query(
        num_variables=num_variables + 1,
        query_mode=query_mode,
        rng=rng,
    )
    return StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=num_variables,
        num_values=num_values,
        operations=tuple(operations),
        query=query,
        final_state=tuple(state),
        temporary_variable=temporary,
    )


def _generate_increment_solvable_source(
    *,
    num_variables: int,
    num_values: int,
    min_extra_assignments: int,
    max_extra_assignments: int,
    min_aperiodic_constants: int,
    max_aperiodic_constants: int,
    min_increment_delta: int,
    max_increment_delta: int,
    query_mode: QueryMode,
    distinct_initial_values: bool,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate a program whose variables are advanced by modular increments.

    Every update is ``x <- (x + delta) mod num_values`` with ``delta`` sampled
    uniformly from the configured range. Registers never read one another, so
    the transition monoid is generated by constant resets and translations of
    the cyclic group of order ``num_values``: every group in its Krohn-Rhodes
    decomposition is cyclic of prime order, hence solvable. There is no scratch
    register. Update targets are balanced across variables, and constant
    mix-ins are scheduled exactly as in the swap-periodic generator so that the
    two source families share their sensitivity-coverage sweep.
    """

    extra_count = rng.randint(max(1, min_extra_assignments), max_extra_assignments)
    requested_constants = _sample_aperiodic_constant_count(
        minimum=min_aperiodic_constants,
        maximum=max_aperiodic_constants,
        rng=rng,
    )
    constant_count = min(requested_constants, extra_count - 1)
    increment_count = extra_count - constant_count
    if increment_count < 1:
        raise ValueError("solvable generation requires room for at least one increment")

    initial_values = _diverse_initial_values(
        num_variables=num_variables,
        num_values=num_values,
        distinct=distinct_initial_values,
        rng=rng,
    )
    state = list(initial_values)
    initial_order = list(range(num_variables))
    rng.shuffle(initial_order)
    operations: list[Operation] = [
        AssignConstant(target=variable, value=state[variable]) for variable in initial_order
    ]

    targets = _balanced_targets(num_variables=num_variables, count=increment_count, rng=rng)
    future_increments = [0] * num_variables
    for target in targets:
        future_increments[target] += 1

    injection_counts = _constant_injection_counts(
        constant_count=constant_count,
        update_count=increment_count,
    )
    reset_counts = [0] * num_variables

    for increment_index, target in enumerate(targets):
        for _ in range(injection_counts[increment_index]):
            operation = _choose_constant_reset(
                state=state,
                num_variables=num_variables,
                num_values=num_values,
                future_updates=future_increments,
                reset_counts=reset_counts,
                rng=rng,
            )
            operations.append(operation)
            _apply(operation, state, num_values)
            reset_counts[operation.target] += 1

        increment = AssignIncrement(
            target=target,
            delta=rng.randint(min_increment_delta, max_increment_delta),
        )
        operations.append(increment)
        _apply(increment, state, num_values)
        future_increments[target] -= 1

    query = _choose_query(num_variables=num_variables, query_mode=query_mode, rng=rng)
    return StateTrackingInstance.create(
        complexity=Complexity.SOLVABLE,
        num_variables=num_variables,
        num_values=num_values,
        operations=tuple(operations),
        query=query,
        final_state=tuple(state),
        temporary_variable=None,
    )


def _constant_injection_counts(*, constant_count: int, update_count: int) -> Counter[int]:
    """Spread constant resets over the first 80% of updates so later updates rebuild chains."""

    if not constant_count:
        return Counter()
    cutoff = max(1, min(update_count, round(0.8 * update_count)))
    return Counter(index * cutoff // constant_count for index in range(constant_count))


def _choose_constant_reset(
    *,
    state: list[int],
    num_variables: int,
    num_values: int,
    future_updates: list[int],
    reset_counts: list[int],
    rng: random.Random,
) -> AssignConstant:
    """Reset the variable with the most remaining updates (fewest prior resets) to a novel value."""

    best_future = max(future_updates)
    candidates = [
        variable for variable, count in enumerate(future_updates) if count == best_future
    ]
    fewest_resets = min(reset_counts[variable] for variable in candidates)
    candidates = [variable for variable in candidates if reset_counts[variable] == fewest_resets]
    target = rng.choice(candidates)
    live_values = set(state[:num_variables])
    novel_values = [value for value in range(num_values) if value not in live_values]
    choices = novel_values or [value for value in range(num_values) if value != state[target]]
    return AssignConstant(target=target, value=rng.choice(choices))


def _sample_aperiodic_constant_count(
    *,
    minimum: int,
    maximum: int,
    rng: random.Random,
) -> int:
    """Sample an integer log-uniformly from [minimum, maximum]."""

    if minimum == maximum:
        return minimum
    sampled = math.exp(rng.uniform(math.log(minimum), math.log(maximum + 1)))
    return min(maximum, int(sampled))


def _diverse_initial_values(
    *,
    num_variables: int,
    num_values: int,
    distinct: bool,
    rng: random.Random,
) -> list[int]:
    """Create a balanced, maximally diverse real-variable value pool."""

    if distinct:
        return rng.sample(range(num_values), num_variables)
    diversity = min(num_variables, num_values)
    vocabulary = rng.sample(range(num_values), diversity)
    values = [vocabulary[index % diversity] for index in range(num_variables)]
    rng.shuffle(values)
    return values


def _balanced_swap_pairs(
    *,
    num_variables: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Choose transpositions that keep real-variable participation balanced."""

    participation = [0] * num_variables
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        minimum = min(participation)
        first_candidates = [
            variable for variable, value in enumerate(participation) if value == minimum
        ]
        left = rng.choice(first_candidates)
        other_minimum = min(
            value for variable, value in enumerate(participation) if variable != left
        )
        second_candidates = [
            variable
            for variable, value in enumerate(participation)
            if variable != left and value == other_minimum
        ]
        right = rng.choice(second_candidates)
        participation[left] += 1
        participation[right] += 1
        pairs.append((left, right))
    return pairs


def _balanced_targets(*, num_variables: int, count: int, rng: random.Random) -> list[int]:
    """Choose increment targets that keep per-variable participation balanced."""

    participation = [0] * num_variables
    targets: list[int] = []
    for _ in range(count):
        minimum = min(participation)
        candidates = [variable for variable, value in enumerate(participation) if value == minimum]
        target = rng.choice(candidates)
        participation[target] += 1
        targets.append(target)
    return targets


def _apply(operation: Operation, state: list[int], num_values: int) -> None:
    if isinstance(operation, AssignConstant):
        state[operation.target] = operation.value
    elif isinstance(operation, AssignVariable):
        state[operation.target] = state[operation.source]
    else:
        state[operation.target] = (state[operation.target] + operation.delta) % num_values


def _choose_query(*, num_variables: int, query_mode: QueryMode, rng: random.Random) -> tuple[int, ...]:
    if query_mode is QueryMode.ALL:
        return tuple(range(num_variables))
    if query_mode is QueryMode.ONE:
        return (rng.randrange(num_variables),)
    size = rng.randint(1, num_variables)
    return tuple(sorted(rng.sample(range(num_variables), size)))
