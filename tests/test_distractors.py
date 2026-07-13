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


def test_rejects_more_distractors_than_the_answer_space_supports() -> None:
    with pytest.raises(ValueError, match="only 1 wrong answers exist"):
        generate_incorrect_states(
            _instance(num_values=2),
            num_distractors=4,
            rng=random.Random(0),
        )
