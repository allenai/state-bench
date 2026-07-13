"""Generate formatter-independent invalid answer states."""

from __future__ import annotations

import random

from state_bench.model import StateTrackingInstance


def generate_incorrect_states(
    instance: StateTrackingInstance,
    *,
    num_distractors: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    """Return unique states whose queried values differ from the correct answer."""

    if num_distractors < 0:
        raise ValueError("num_distractors must be non-negative")
    available = instance.num_values ** len(instance.query) - 1
    if num_distractors > available:
        raise ValueError(
            f"requested {num_distractors} distractors, but only {available} wrong answers exist"
        )

    correct_answer = tuple(instance.final_state[variable] for variable in instance.query)
    wrong_answers: set[tuple[int, ...]] = set()
    while len(wrong_answers) < num_distractors:
        candidate = tuple(rng.randrange(instance.num_values) for _ in instance.query)
        if candidate != correct_answer:
            wrong_answers.add(candidate)

    states = []
    for wrong_answer in sorted(wrong_answers):
        state = list(instance.final_state)
        for variable, value in zip(instance.query, wrong_answer, strict=True):
            state[variable] = value
        states.append(tuple(state))
    rng.shuffle(states)
    return tuple(states)
