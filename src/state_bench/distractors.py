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
    """Return unique wrong states with the correct answer's equality pattern.

    Distractors preserve the correct answer's equality pattern, except that
    periodic distractors preserve its value multiplicities rather than the
    locations of equal values. They additionally use only correct-answer
    values and match its agreement with each variable's initial literal and a
    one-hop reading of each final copy source.
    These constraints prevent a multiple-choice solver from choosing an option
    merely because it uses unseen values, resembles the initial state less, or
    follows only the final copy operation.
    """

    if num_distractors < 0:
        raise ValueError("num_distractors must be non-negative")
    correct_answer = tuple(instance.final_state[variable] for variable in instance.query)
    distinct_values = tuple(dict.fromkeys(correct_answer))
    periodic = instance.complexity is Complexity.PERIODIC
    cue_matched_periodic = (
        periodic and _multiset_permutations(correct_answer) - 1 >= num_distractors
    )
    available = (
        _multiset_permutations(correct_answer) - 1
        if cue_matched_periodic
        else perm(instance.num_values, len(distinct_values)) - 1
    )
    if num_distractors > available:
        raise ValueError(
            f"requested {num_distractors} distractors, but only {available} equality-pattern-"
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
        )
        if len(wrong_answers) >= num_distractors:
            final_depths = dependency_lengths(
                num_variables=instance.num_variables,
                operations=instance.operations,
            )
            queried_depths = tuple(
                final_depths[variable] for variable in instance.query
            )
            chosen_answers = _choose_depth_weighted_answers(
                wrong_answers,
                correct_answer=correct_answer,
                queried_depths=queried_depths,
                count=num_distractors,
                rng=rng,
            )
            wrong_answers = set(chosen_answers)
        else:
            # Legacy periodic traces can have too little answer entropy to
            # balance all cues. Preserve their established generic behavior;
            # layered evaluation data is constructed to take the branch above.
            cue_matched_periodic = False
            wrong_answers.clear()
    attempts = 0
    while len(wrong_answers) < num_distractors:
        attempts += 1
        if attempts > 100_000:
            raise ValueError("could not generate enough leakage-matched periodic distractors")
        replacement_values = rng.sample(range(instance.num_values), len(distinct_values))
        replacements = dict(zip(distinct_values, replacement_values, strict=True))
        candidate = tuple(replacements[value] for value in correct_answer)
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
) -> set[tuple[int, ...]]:
    """Find local rearrangements that preserve every audited shallow score."""

    candidates: set[tuple[int, ...]] = set()
    indices = range(len(correct_answer))
    for size in (2, 3):
        for selected in combinations(indices, size):
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

    values: list[int | None] = [None] * instance.num_variables
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

    last_sources: list[int | None] = [None] * instance.num_variables
    for operation in instance.operations:
        if isinstance(operation, AssignVariable):
            last_sources[operation.target] = operation.source
    return tuple(
        initial_values[last_sources[variable]]
        if last_sources[variable] is not None
        else initial_values[variable]
        for variable in instance.query
    )
