from __future__ import annotations

import random

import pytest

from state_bench.generation import generate_instance
from state_bench.model import AssignConstant, AssignVariable, Complexity
from state_bench.semantics import evaluate


def test_prompt_examples() -> None:
    assert evaluate(
        num_variables=2,
        num_values=3,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1)),
    ) == (0, 1)
    assert evaluate(
        num_variables=2,
        num_values=3,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1), AssignConstant(0, 2)),
    ) == (2, 1)
    assert evaluate(
        num_variables=2,
        num_values=3,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1), AssignVariable(0, 1)),
    ) == (1, 1)


def test_reference_is_eager_copy_not_alias() -> None:
    operations = (
        AssignConstant(0, 0),
        AssignConstant(1, 1),
        AssignVariable(0, 1),
        AssignConstant(1, 2),
    )
    assert evaluate(num_variables=2, num_values=3, operations=operations) == (1, 2)


@pytest.mark.parametrize("complexity", list(Complexity))
def test_generation_is_reproducible_and_self_consistent(complexity: Complexity) -> None:
    kwargs = {
        "complexity": complexity,
        "num_variables": 4,
        "num_values": 5,
        "min_extra_assignments": 0 if complexity is Complexity.R_TRIVIAL else 1,
        "max_extra_assignments": 0 if complexity is Complexity.R_TRIVIAL else 5,
    }
    first = generate_instance(**kwargs, rng=random.Random(7))
    second = generate_instance(**kwargs, rng=random.Random(7))
    assert first == second
    assert first.final_state == evaluate(
        num_variables=first.num_variables,
        num_values=first.num_values,
        operations=first.operations,
    )


def test_strict_family_witnesses() -> None:
    aperiodic = generate_instance(
        complexity=Complexity.APERIODIC,
        num_variables=3,
        num_values=3,
        min_extra_assignments=1,
        max_extra_assignments=1,
        rng=random.Random(1),
    )
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=3,
        num_values=3,
        min_extra_assignments=3,
        max_extra_assignments=3,
        rng=random.Random(1),
    )
    assert len(aperiodic.operations) == 4
    assert all(isinstance(operation, AssignConstant) for operation in aperiodic.operations)
    assert isinstance(periodic.operations[-1], AssignVariable)


def test_uninitialized_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="uninitialized"):
        evaluate(num_variables=2, num_values=2, operations=(AssignVariable(0, 1),))
