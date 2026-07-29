from __future__ import annotations

import random

import pytest

from state_bench.generation import derive_instance, derive_variants, generate_instance
from state_bench.model import AssignConstant, AssignVariable, Complexity, StateTrackingInstance
from state_bench.semantics import dependency_lengths, dependency_summary, evaluate


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
    first = generate_instance(
        complexity=complexity,
        num_variables=4,
        num_values=5,
        min_extra_assignments=1,
        max_extra_assignments=5,
        rng=random.Random(7),
    )
    second = generate_instance(
        complexity=complexity,
        num_variables=4,
        num_values=5,
        min_extra_assignments=1,
        max_extra_assignments=5,
        rng=random.Random(7),
    )
    assert first == second
    assert first.final_state == evaluate(
        num_variables=first.num_variables,
        num_values=first.num_values,
        operations=first.operations,
    )


def test_derived_families_resolve_periodic_references_from_one_source() -> None:
    source = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=6,
        operations=(
            AssignConstant(0, 2),
            AssignConstant(1, 5),
            AssignVariable(0, 1),
            AssignConstant(1, 3),
            AssignVariable(1, 0),
        ),
        query=(0, 1),
        final_state=(5, 5),
    )

    aperiodic = derive_instance(source, complexity=Complexity.APERIODIC)
    r_trivial = derive_instance(source, complexity=Complexity.R_TRIVIAL)

    assert aperiodic.operations == (
        AssignConstant(0, 2),
        AssignConstant(1, 5),
        AssignConstant(0, 5),
        AssignConstant(1, 3),
        AssignConstant(1, 5),
    )
    assert r_trivial.operations == (
        AssignConstant(0, 2),
        AssignConstant(1, 5),
        AssignConstant(0, 2),
        AssignConstant(1, 5),
        AssignConstant(1, 5),
    )
    assert aperiodic.final_state == source.final_state
    assert r_trivial.final_state == (2, 5)
    assert len(source.operations) == len(aperiodic.operations) == len(r_trivial.operations)
    first_values: dict[int, int] = {}
    for operation in r_trivial.operations:
        assert isinstance(operation, AssignConstant)
        first_values.setdefault(operation.target, operation.value)
        assert operation.value == first_values[operation.target]

    assert derive_variants(source) == tuple(
        derive_instance(source, complexity=complexity) for complexity in Complexity
    )


@pytest.mark.parametrize("seed", range(20))
def test_batched_derivations_preserve_trace_alignment_and_final_states(seed: int) -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=2 + seed % 7,
        num_values=3 + seed % 9,
        min_extra_assignments=1,
        max_extra_assignments=40,
        rng=random.Random(seed),
    )
    variants = derive_variants(periodic)

    assert variants == tuple(
        derive_instance(periodic, complexity=complexity) for complexity in Complexity
    )
    for variant in variants:
        assert variant.query == periodic.query
        assert len(variant.operations) == len(periodic.operations)
        assert tuple(operation.target for operation in variant.operations) == tuple(
            operation.target for operation in periodic.operations
        )
        assert variant.final_state == evaluate(
            num_variables=variant.num_variables,
            num_values=variant.num_values,
            operations=variant.operations,
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
    state: list[int | None] = [None] * periodic.num_variables
    changing_copy_found = False
    for operation in periodic.operations:
        if isinstance(operation, AssignConstant):
            state[operation.target] = operation.value
        else:
            changing_copy_found |= state[operation.target] != state[operation.source]
            state[operation.target] = state[operation.source]
    assert changing_copy_found


def test_periodic_generation_can_build_deep_dependency_chains() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=4,
        num_values=5,
        min_extra_assignments=100,
        max_extra_assignments=100,
        periodic_constant_probability=0.0,
        rng=random.Random(3),
    )
    lengths = dependency_lengths(
        num_variables=periodic.num_variables,
        operations=periodic.operations,
    )
    assert min(lengths) >= 48
    assert max(lengths) >= 50
    assert len(set(periodic.final_state)) == 2


def test_periodic_generation_uses_value_changing_updates_and_live_lineages() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=6,
        num_values=10,
        min_extra_assignments=1000,
        max_extra_assignments=1000,
        periodic_constant_probability=0.1,
        rng=random.Random(3),
    )
    initial_state = evaluate(
        num_variables=periodic.num_variables,
        num_values=periodic.num_values,
        operations=periodic.operations[: periodic.num_variables],
    )
    state = list(initial_state)
    copy_count = 0
    for operation in periodic.operations[periodic.num_variables :]:
        old_value = state[operation.target]
        if isinstance(operation, AssignConstant):
            new_value = operation.value
        else:
            copy_count += 1
            new_value = state[operation.source]
        assert new_value != old_value
        state[operation.target] = new_value

    assert tuple(state) == periodic.final_state
    assert len(set(periodic.final_state)) == len(set(initial_state)) >= 2
    assert copy_count >= 800
    assert max(
        dependency_lengths(
            num_variables=periodic.num_variables,
            operations=periodic.operations,
        )
    ) >= 400


def test_two_variable_periodic_generation_repairs_copy_collapse() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=10,
        min_extra_assignments=100,
        max_extra_assignments=100,
        periodic_constant_probability=0.0,
        rng=random.Random(3),
    )
    assert len(set(periodic.final_state)) == 2
    assert max(
        dependency_lengths(
            num_variables=periodic.num_variables,
            operations=periodic.operations,
        )
    ) >= 45


def test_uninitialized_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="uninitialized"):
        evaluate(num_variables=2, num_values=2, operations=(AssignVariable(0, 1),))


def test_dependency_lengths_follow_eager_copy_provenance_and_overwrites() -> None:
    operations = (
        AssignConstant(0, 4),
        AssignVariable(1, 0),
        AssignVariable(2, 1),
        AssignVariable(0, 2),
        AssignConstant(1, 3),
    )
    assert dependency_lengths(num_variables=3, operations=operations) == (3, 0, 2)

    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=3,
        num_values=5,
        operations=operations,
        query=(0, 1, 2),
        final_state=(4, 3, 4),
    )
    summary = dependency_summary(instance)
    assert summary.max_dependency_length == 3
    assert summary.mean_dependency_length == pytest.approx(5 / 3)
    assert summary.modal_dependency_length == 0
