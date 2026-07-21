from __future__ import annotations

import random

import pytest

from state_bench.distractors import generate_incorrect_states
from state_bench.model import AssignConstant, Complexity, StateTrackingInstance


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


def test_distractors_preserve_the_correct_answers_equality_pattern() -> None:
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
    correct_pattern = _equality_pattern(instance.final_state)
    assert all(_equality_pattern(state) == correct_pattern for state in states)


def test_rejects_more_distractors_than_the_answer_space_supports() -> None:
    with pytest.raises(ValueError, match="only 1 equality-pattern-preserving"):
        generate_incorrect_states(
            _instance(num_values=2),
            num_distractors=4,
            rng=random.Random(0),
        )


def _equality_pattern(values: tuple[int, ...]) -> tuple[int, ...]:
    labels: dict[int, int] = {}
    return tuple(labels.setdefault(value, len(labels)) for value in values)
