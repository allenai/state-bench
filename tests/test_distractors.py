from __future__ import annotations

import random
from collections import Counter

import pytest

from state_bench.distractors import _choose_depth_weighted_answers, generate_incorrect_states
from state_bench.generation import derive_variants, generate_instance
from state_bench.model import AssignConstant, AssignVariable, Complexity, StateTrackingInstance


def _instance(*, num_values: int = 5) -> StateTrackingInstance:
    return StateTrackingInstance.create(
        complexity=Complexity.R_TRIVIAL,
        num_variables=1,
        num_values=num_values,
        operations=(AssignConstant(0, 0),),
        query=(0,),
        final_state=(0,),
    )


def test_generates_four_unique_incorrect_states_for_single_variable_query() -> None:
    states = generate_incorrect_states(
        _instance(),
        num_distractors=4,
        rng=random.Random(3),
    )
    assert len(states) == len(set(states)) == 4
    assert all(state != (0,) for state in states)


def test_non_periodic_distractors_prefer_rearrangements_then_seen_values() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.R_TRIVIAL,
        num_variables=4,
        num_values=5,
        operations=tuple(
            AssignConstant(target=target, value=value)
            for target, value in enumerate((0, 0, 2, 0))
        ),
        query=(0, 1, 2, 3),
        final_state=(0, 0, 2, 0),
    )
    states = generate_incorrect_states(instance, num_distractors=4, rng=random.Random(7))
    correct_counter = Counter(instance.final_state)
    rearranged = sum(Counter(state) == correct_counter for state in states)
    # (0, 0, 2, 0) admits exactly three rearrangements; the fourth distractor
    # must fall back to substituting values the trace actually assigns.
    assert rearranged == 3
    assert all(set(state) <= {0, 2} for state in states)


def test_two_variable_periodic_traces_still_fill_every_distractor_slot() -> None:
    instance = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=100,
        min_extra_assignments=64,
        max_extra_assignments=64,
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        rng=random.Random(61),
    )
    states = generate_incorrect_states(instance, num_distractors=4, rng=random.Random(67))
    assert len(states) == len(set(states)) == 4
    correct = tuple(instance.final_state[variable] for variable in instance.query)
    correct_counter = Counter(correct)
    answers = [tuple(state[variable] for variable in instance.query) for state in states]
    # Exactly one rearrangement of the correct values exists (the swap); the
    # remaining alternatives substitute values the trace actually assigns.
    assert sum(Counter(answer) == correct_counter for answer in answers) == 1
    seen = {
        operation.value
        for operation in instance.operations
        if isinstance(operation, AssignConstant)
    }
    assert all(set(answer) <= seen for answer in answers)
    assert all(
        state[instance.temporary_variable] == instance.final_state[instance.temporary_variable]
        for state in states
    )


def test_aperiodic_distractors_are_not_identifiable_by_value_multiset() -> None:
    source = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=8,
        num_values=100,
        min_extra_assignments=64,
        max_extra_assignments=64,
        min_aperiodic_constants=4,
        max_aperiodic_constants=4,
        rng=random.Random(51),
    )
    for instance in derive_variants(source):
        if instance.complexity is Complexity.PERIODIC:
            continue
        states = generate_incorrect_states(instance, num_distractors=4, rng=random.Random(53))
        correct_counter = Counter(
            instance.final_state[variable] for variable in instance.query
        )
        assert all(
            Counter(state[variable] for variable in instance.query) == correct_counter
            for state in states
        )


def test_rejects_more_distractors_than_the_answer_space_supports() -> None:
    with pytest.raises(ValueError, match="only 1 equality-pattern-preserving"):
        generate_incorrect_states(
            _instance(num_values=2),
            num_distractors=4,
            rng=random.Random(0),
        )


def test_periodic_distractors_match_initial_value_and_position_cues() -> None:
    instance = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=8,
        num_values=10,
        min_extra_assignments=64,
        max_extra_assignments=64,
        rng=random.Random(19),
    )
    initial = [None] * instance.num_registers
    for operation in instance.operations[: instance.num_initial_assignments]:
        assert isinstance(operation, AssignConstant)
        initial[operation.target] = operation.value
    correct = tuple(instance.final_state[variable] for variable in instance.query)
    correct_agreement = sum(
        value == initial[variable]
        for variable, value in zip(instance.query, correct, strict=True)
    )
    last_sources = [None] * instance.num_registers
    for operation in instance.operations:
        if isinstance(operation, AssignVariable):
            last_sources[operation.target] = operation.source
    one_hop = tuple(
        initial[last_sources[variable]]
        if last_sources[variable] is not None
        else initial[variable]
        for variable in instance.query
    )
    correct_one_hop_agreement = sum(
        value == predicted for value, predicted in zip(correct, one_hop, strict=True)
    )
    states = generate_incorrect_states(instance, num_distractors=4, rng=random.Random(11))

    for state in states:
        answer = tuple(state[variable] for variable in instance.query)
        assert state[instance.temporary_variable] == instance.final_state[instance.temporary_variable]
        assert set(answer) == set(correct)
        assert Counter(answer) == Counter(correct)
        assert sum(
            value == initial[variable]
            for variable, value in zip(instance.query, answer, strict=True)
        ) == correct_agreement
        assert sum(
            value == predicted for value, predicted in zip(answer, one_hop, strict=True)
        ) == correct_one_hop_agreement


def test_depth_weighted_periodic_distractors_prefer_to_move_deep_coordinates() -> None:
    correct = (0, 1, 2, 3)
    shallow_move = (1, 0, 2, 3)
    deep_move = (0, 1, 3, 2)
    deep_selections = 0

    for seed in range(400):
        selected = _choose_depth_weighted_answers(
            {shallow_move, deep_move},
            correct_answer=correct,
            queried_depths=(1, 1, 1_000, 1_000),
            count=1,
            rng=random.Random(seed),
        )
        deep_selections += selected == [deep_move]

    assert deep_selections > 390


def test_tmp_is_fixed_in_alternatives_for_every_derived_complexity() -> None:
    source = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=4,
        num_values=10,
        min_extra_assignments=64,
        max_extra_assignments=64,
        min_aperiodic_constants=6,
        max_aperiodic_constants=6,
        rng=random.Random(23),
    )
    for instance in derive_variants(source):
        states = generate_incorrect_states(
            instance,
            num_distractors=4,
            rng=random.Random(29),
        )
        assert states
        assert all(
            state[instance.temporary_variable] == instance.final_state[instance.temporary_variable]
            for state in states
        )
