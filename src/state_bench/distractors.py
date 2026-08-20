"""Generate formatter-independent invalid answer states."""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations, permutations
from math import factorial, log1p, perm

from state_bench.model import AssignConstant, AssignVariable, Complexity, StateTrackingInstance
from state_bench.semantics import dependency_lengths


def generate_incorrect_states(
    instance: StateTrackingInstance,
    *,
    num_distractors: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    """Return unique wrong states that a value-bag heuristic cannot reject.

    Periodic distractors preserve the correct answer's value multiplicities and
    match its agreement with each variable's initial literal and a one-hop
    reading of each final copy source. Non-periodic distractors prefer the same
    value-preserving rearrangements, then substitutions drawn from constants
    the trace actually assigns, and fall back to equality-pattern-preserving
    relabelings only when the answer space is too small for anything better.
    These constraints prevent a multiple-choice solver from choosing an option
    merely because it uses unseen values, resembles the initial state less, or
    follows only the final copy operation. When a scratch register is present,
    its answer coordinate is fixed to the correct value in every alternative.
    Every complexity family fills the same way - value-preserving
    rearrangements first, then trace-seen substitutions - so all families
    return the full ``num_distractors`` (the two-variable case pairs its single
    non-identity permutation with substitution-based alternatives).
    """

    if num_distractors < 0:
        raise ValueError("num_distractors must be non-negative")
    correct_answer = tuple(instance.final_state[variable] for variable in instance.query)
    fixed_positions = {
        index
        for index, variable in enumerate(instance.query)
        if variable == instance.temporary_variable
    }
    movable_positions = tuple(
        index for index in range(len(correct_answer)) if index not in fixed_positions
    )
    distinct_values = tuple(dict.fromkeys(correct_answer))
    periodic = instance.complexity is Complexity.PERIODIC
    movable_answer = tuple(correct_answer[index] for index in movable_positions)
    periodic_available = max(0, _multiset_permutations(movable_answer) - 1)
    target_count = num_distractors
    rearrangement_target = min(num_distractors, periodic_available)
    cue_matched_periodic = periodic and rearrangement_target > 0
    available = perm(instance.num_values, len(distinct_values)) - 1
    if periodic:
        available += periodic_available
    if target_count > available:
        raise ValueError(
            f"requested {target_count} distractors, but only {available} equality-pattern-"
            "preserving wrong answers exist"
        )

    initial_values = _initial_values(instance) if cue_matched_periodic else ()
    correct_initial_agreement = (
        _initial_agreement(correct_answer, instance.query, initial_values)
        if cue_matched_periodic
        else None
    )
    one_hop_values = (
        _last_copy_source_initial_values(instance, initial_values) if cue_matched_periodic else ()
    )
    correct_one_hop_agreement = (
        sum(
            value == one_hop
            for value, one_hop in zip(correct_answer, one_hop_values, strict=True)
        )
        if cue_matched_periodic
        else None
    )
    wrong_answers: set[tuple[int, ...]] = set()
    if cue_matched_periodic:
        wrong_answers = _periodic_cue_matched_answers(
            correct_answer,
            initial_values=initial_values,
            query=instance.query,
            one_hop_values=one_hop_values,
            initial_agreement=correct_initial_agreement,
            one_hop_agreement=correct_one_hop_agreement,
            movable_positions=movable_positions,
        )
        attempts = 0
        while len(wrong_answers) < rearrangement_target:
            attempts += 1
            if attempts > 100_000:
                raise ValueError("could not generate enough TMP-fixed periodic permutations")
            shuffled = list(movable_answer)
            rng.shuffle(shuffled)
            candidate = list(correct_answer)
            for position, value in zip(movable_positions, shuffled, strict=True):
                candidate[position] = value
            candidate_tuple = tuple(candidate)
            if candidate_tuple != correct_answer:
                wrong_answers.add(candidate_tuple)
        final_depths = dependency_lengths(
            num_variables=instance.num_registers,
            operations=instance.operations,
        )
        queried_depths = tuple(final_depths[variable] for variable in instance.query)
        wrong_answers = set(
            _choose_depth_weighted_answers(
                wrong_answers,
                correct_answer=correct_answer,
                queried_depths=queried_depths,
                count=rearrangement_target,
                rng=rng,
            )
        )
    if not periodic:
        _add_multiset_rearrangements(
            wrong_answers,
            correct_answer=correct_answer,
            movable_positions=movable_positions,
            movable_answer=movable_answer,
            count=rearrangement_target,
            rng=rng,
        )
    _add_seen_value_substitutions(
        wrong_answers,
        instance=instance,
        correct_answer=correct_answer,
        movable_positions=movable_positions,
        count=target_count,
        rng=rng,
    )
    attempts = 0
    while len(wrong_answers) < target_count:
        attempts += 1
        if attempts > 100_000:
            raise ValueError("could not generate enough equality-pattern-preserving distractors")
        replacement_values = rng.sample(range(instance.num_values), len(distinct_values))
        replacements = dict(zip(distinct_values, replacement_values, strict=True))
        candidate = tuple(replacements[value] for value in correct_answer)
        if fixed_positions:
            candidate_values = list(candidate)
            for position in fixed_positions:
                candidate_values[position] = correct_answer[position]
            candidate = tuple(candidate_values)
        if candidate == correct_answer:
            continue
        wrong_answers.add(candidate)

    states = []
    for wrong_answer in sorted(wrong_answers):
        state = list(instance.final_state)
        for variable, value in zip(instance.query, wrong_answer, strict=True):
            state[variable] = value
        states.append(tuple(state))
    rng.shuffle(states)
    return tuple(states)


def _add_multiset_rearrangements(
    wrong_answers: set[tuple[int, ...]],
    *,
    correct_answer: tuple[int, ...],
    movable_positions: tuple[int, ...],
    movable_answer: tuple[int, ...],
    count: int,
    rng: random.Random,
) -> None:
    """Add wrong answers that reuse exactly the correct answer's values."""

    attempts = 0
    while len(wrong_answers) < count and attempts < 10_000:
        attempts += 1
        shuffled = list(movable_answer)
        rng.shuffle(shuffled)
        candidate = list(correct_answer)
        for position, value in zip(movable_positions, shuffled, strict=True):
            candidate[position] = value
        candidate_tuple = tuple(candidate)
        if candidate_tuple != correct_answer:
            wrong_answers.add(candidate_tuple)


def _add_seen_value_substitutions(
    wrong_answers: set[tuple[int, ...]],
    *,
    instance: StateTrackingInstance,
    correct_answer: tuple[int, ...],
    movable_positions: tuple[int, ...],
    count: int,
    rng: random.Random,
) -> None:
    """Add wrong answers whose values all appear as constants in the trace."""

    seen_values = sorted({
        operation.value
        for operation in instance.operations
        if isinstance(operation, AssignConstant)
    })
    if not seen_values or not movable_positions:
        return
    attempts = 0
    while len(wrong_answers) < count and attempts < 10_000:
        attempts += 1
        candidate = list(correct_answer)
        size = min(len(movable_positions), rng.randint(1, 2))
        changed = False
        for position in rng.sample(movable_positions, size):
            alternatives = [value for value in seen_values if value != correct_answer[position]]
            if alternatives:
                candidate[position] = rng.choice(alternatives)
                changed = True
        candidate_tuple = tuple(candidate)
        if changed and candidate_tuple != correct_answer:
            wrong_answers.add(candidate_tuple)


def _multiset_permutations(values: tuple[int, ...]) -> int:
    count = factorial(len(values))
    for multiplicity in Counter(values).values():
        count //= factorial(multiplicity)
    return count


def _periodic_cue_matched_answers(
    correct_answer: tuple[int, ...],
    *,
    initial_values: tuple[int, ...],
    query: tuple[int, ...],
    one_hop_values: tuple[int, ...],
    initial_agreement: int | None,
    one_hop_agreement: int | None,
    movable_positions: tuple[int, ...],
) -> set[tuple[int, ...]]:
    """Find local rearrangements that preserve every audited shallow score."""

    candidates: set[tuple[int, ...]] = set()
    for size in (2, 3):
        for selected in combinations(movable_positions, size):
            values = tuple(correct_answer[index] for index in selected)
            for rearrangement in permutations(values):
                candidate_list = list(correct_answer)
                for index, value in zip(selected, rearrangement, strict=True):
                    candidate_list[index] = value
                candidate = tuple(candidate_list)
                if candidate == correct_answer:
                    continue
                if _initial_agreement(candidate, query, initial_values) != initial_agreement:
                    continue
                if sum(
                    value == one_hop
                    for value, one_hop in zip(candidate, one_hop_values, strict=True)
                ) != one_hop_agreement:
                    continue
                candidates.add(candidate)
    return candidates


def _choose_depth_weighted_answers(
    candidates: set[tuple[int, ...]],
    *,
    correct_answer: tuple[int, ...],
    queried_depths: tuple[int, ...],
    count: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Choose alternatives that preferentially move deep queried coordinates.

    A moved coordinate contributes ``log(1 + depth)`` to its candidate's
    weight. Sampling without replacement from these weights leaves short,
    locally solvable coordinates unchanged in most alternatives while placing
    the uncertainty on the long dependency chains.
    """

    remaining = sorted(candidates)
    chosen: list[tuple[int, ...]] = []
    while len(chosen) < count:
        weights = [
            _depth_weight(candidate, correct_answer=correct_answer, queried_depths=queried_depths)
            for candidate in remaining
        ]
        selected = rng.choices(remaining, weights=weights, k=1)[0]
        remaining.remove(selected)
        chosen.append(selected)
    return chosen


def _depth_weight(
    candidate: tuple[int, ...],
    *,
    correct_answer: tuple[int, ...],
    queried_depths: tuple[int, ...],
) -> float:
    """Return the relative sampling weight for one non-identity alternative."""

    weight = 1.0
    for candidate_value, correct_value, depth in zip(
        candidate, correct_answer, queried_depths, strict=True
    ):
        if candidate_value != correct_value:
            # A tiny nonzero floor keeps generation feasible if a legacy trace
            # exposes only shallow mutable coordinates.
            weight *= max(log1p(depth), 0.05)
    return weight


def _initial_values(instance: StateTrackingInstance) -> tuple[int, ...]:
    """Return the first literal assigned to each variable in a periodic trace."""

    values: list[int | None] = [None] * instance.num_registers
    for operation in instance.operations:
        if values[operation.target] is None:
            if not isinstance(operation, AssignConstant):
                raise ValueError("periodic distractors require an initial literal for every variable")
            values[operation.target] = operation.value
    if any(value is None for value in values):
        raise ValueError("periodic distractors require every variable to be initialized")
    return tuple(value for value in values if value is not None)


def _initial_agreement(
    answer: tuple[int, ...],
    query: tuple[int, ...],
    initial_values: tuple[int, ...],
) -> int:
    return sum(
        value == initial_values[variable]
        for variable, value in zip(query, answer, strict=True)
    )


def _last_copy_source_initial_values(
    instance: StateTrackingInstance,
    initial_values: tuple[int, ...],
) -> tuple[int, ...]:
    """Initial value lookup obtained by inspecting only the final copy into each target."""

    last_sources: list[int | None] = [None] * instance.num_registers
    for operation in instance.operations:
        if isinstance(operation, AssignVariable):
            last_sources[operation.target] = operation.source
    values: list[int] = []
    for variable in instance.query:
        source = last_sources[variable]
        values.append(initial_values[source] if source is not None else initial_values[variable])
    return tuple(values)
