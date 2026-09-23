"""Constrained generators for the state-tracking families.

Every trace starts as one ``periodic`` source (TMP-mediated swaps of variable
contents). The three simpler families are derived from that exact trace
without changing trace length, targets, query, or the value any assignment
writes:

- ``solvable`` rewrites every reference as a modular increment of its own
  target that lands on the value the reference would have copied;
- ``aperiodic`` rewrites every reference as the literal it copies;
- ``r-trivial`` rewrites every assignment as its target's first value.
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

DERIVED_FAMILIES: tuple[Complexity, ...] = (
    Complexity.R_TRIVIAL,
    Complexity.APERIODIC,
    Complexity.SOLVABLE,
    Complexity.PERIODIC,
)
"""Every family derived from a periodic source, in emission order."""


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
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
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
    min_aperiodic_constants: int = 1,
    max_aperiodic_constants: int = 64,
    rng: random.Random,
) -> StateTrackingInstance:
    """Generate the periodic source-of-truth for a four-family trace set."""

    _validate_parameters(
        num_variables=num_variables,
        num_values=num_values,
        min_extra_assignments=min_extra_assignments,
        max_extra_assignments=max_extra_assignments,
        distinct_initial_values=distinct_initial_values,
        require_strict_witness=require_strict_witness,
        min_aperiodic_constants=min_aperiodic_constants,
        max_aperiodic_constants=max_aperiodic_constants,
    )
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


def derive_instance(
    source: StateTrackingInstance,
    *,
    complexity: Complexity,
) -> StateTrackingInstance:
    """Derive one complexity variant of a periodic source; see ``derive_variants``."""

    (instance,) = derive_variants(source, families=(complexity,))
    return instance


def derive_variants(
    source: StateTrackingInstance,
    *,
    families: tuple[Complexity, ...] = DERIVED_FAMILIES,
) -> tuple[StateTrackingInstance, ...]:
    """Derive the requested families with one traversal of a periodic source.

    Literal assignments are kept by every family except r-trivial. For each
    reference ``a <- b`` that writes ``v`` over ``a``'s current value ``u``:

    - solvable emits ``a <- (a + ((v - u) mod num_values)) mod num_values``;
    - aperiodic emits ``a <- v``;
    - r-trivial emits ``a <- first`` where ``first`` is ``a``'s first value,
      for every assignment after ``a``'s first, whatever its kind.

    Solvable, aperiodic, and periodic therefore hold identical values in every
    register after every step and share the source's final state. Solvable
    increments may have ``delta = 0`` (for example when TMP is reloaded with
    the value it already holds); those are kept as identity increments so that
    every family has the same operation kinds at the same positions except for
    the reference-to-increment rewrite.

    The result is ordered like ``families``.
    """

    if source.complexity is not Complexity.PERIODIC:
        raise ValueError("derived instances require a periodic source")
    unknown = [family for family in families if family not in DERIVED_FAMILIES]
    if unknown:
        raise ValueError(f"cannot derive {[family.value for family in unknown]} from a periodic source")

    num_values = source.num_values
    state: list[int | None] = [None] * source.num_registers
    first_values: list[int | None] = [None] * source.num_registers
    aperiodic_operations: list[Operation] = []
    solvable_operations: list[Operation] = []
    r_trivial_operations: list[Operation] = []
    for operation in source.operations:
        target = operation.target
        if isinstance(operation, AssignConstant):
            value = operation.value
            solvable_operations.append(operation)
        elif isinstance(operation, AssignVariable):
            value = state[operation.source]
            if value is None:
                raise ValueError(f"source variable {operation.source} is uninitialized")
            current = state[target]
            if current is None:
                raise ValueError(f"referenced target {target} is uninitialized")
            solvable_operations.append(
                AssignIncrement(target=target, delta=(value - current) % num_values)
            )
        else:
            raise ValueError("periodic sources contain only constants and references")

        aperiodic_operations.append(AssignConstant(target=target, value=value))
        first_value = first_values[target]
        if first_value is None:
            first_value = value
            first_values[target] = first_value
        r_trivial_operations.append(AssignConstant(target=target, value=first_value))
        state[target] = value

    r_trivial_state = tuple(value for value in first_values if value is not None)
    if len(r_trivial_state) != source.num_registers:
        raise ValueError("source trace leaves a variable uninitialized")

    def derived(complexity: Complexity) -> StateTrackingInstance:
        if complexity is Complexity.PERIODIC:
            return source
        operations, final_state = {
            Complexity.R_TRIVIAL: (r_trivial_operations, r_trivial_state),
            Complexity.APERIODIC: (aperiodic_operations, source.final_state),
            Complexity.SOLVABLE: (solvable_operations, source.final_state),
        }[complexity]
        return StateTrackingInstance.create(
            complexity=complexity,
            num_variables=source.num_variables,
            num_values=num_values,
            operations=tuple(operations),
            query=source.query,
            final_state=final_state,
            temporary_variable=source.temporary_variable,
        )

    return tuple(derived(complexity) for complexity in families)


def _validate_parameters(
    *,
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
    if require_strict_witness:
        if max_extra_assignments < 3:
            raise ValueError("strict swap-periodic instances require room for three extra assignments")
        if num_variables < 2 or num_values < 2:
            raise ValueError("strict periodic instances require at least two variables and values")


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
        _apply(operation, state)

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
            _apply(operation, state)
            reset_counts[operation.target] += 1

        swap_operations = (
            AssignVariable(target=temporary, source=left),
            AssignVariable(target=left, source=right),
            AssignVariable(target=right, source=temporary),
        )
        for operation in swap_operations:
            operations.append(operation)
            _apply(operation, state)
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


def _apply(operation: Operation, state: list[int]) -> None:
    if isinstance(operation, AssignConstant):
        state[operation.target] = operation.value
    else:
        state[operation.target] = state[operation.source]


def _choose_query(*, num_variables: int, query_mode: QueryMode, rng: random.Random) -> tuple[int, ...]:
    if query_mode is QueryMode.ALL:
        return tuple(range(num_variables))
    if query_mode is QueryMode.ONE:
        return (rng.randrange(num_variables),)
    size = rng.randint(1, num_variables)
    return tuple(sorted(rng.sample(range(num_variables), size)))
