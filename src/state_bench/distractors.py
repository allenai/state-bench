"""Generate formatter-independent invalid answer states."""

from __future__ import annotations

import random
from math import perm

from state_bench.model import StateTrackingInstance


def generate_incorrect_states(
    instance: StateTrackingInstance,
    *,
    num_distractors: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    """Return unique wrong states with the correct answer's equality pattern.

    Distractors are injective relabelings of the queried values. Thus two
    queried variables are equal in every distractor exactly when they are equal
    in the correct answer. This prevents the shape of an answer (for example,
    all queried variables sharing one value) from identifying the correct
    multiple-choice option without reading the trace.
    """

    if num_distractors < 0:
        raise ValueError("num_distractors must be non-negative")
    correct_answer = tuple(instance.final_state[variable] for variable in instance.query)
    distinct_values = tuple(dict.fromkeys(correct_answer))
    available = perm(instance.num_values, len(distinct_values)) - 1
    if num_distractors > available:
        raise ValueError(
            f"requested {num_distractors} distractors, but only {available} equality-pattern-"
            "preserving wrong answers exist"
        )

    wrong_answers: set[tuple[int, ...]] = set()
    while len(wrong_answers) < num_distractors:
        replacement_values = rng.sample(
            range(instance.num_values),
            len(distinct_values),
        )
        replacements = dict(zip(distinct_values, replacement_values, strict=True))
        candidate = tuple(replacements[value] for value in correct_answer)
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
