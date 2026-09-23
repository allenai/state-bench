from __future__ import annotations

import random

import pytest

from state_bench.generation import DERIVED_FAMILIES, derive_instance, derive_variants, generate_instance
from state_bench.model import AssignConstant, AssignVariable, Complexity, StateTrackingInstance
from state_bench.semantics import dependency_lengths, dependency_summary, evaluate, sensitivity_summary


def _instance(operations: tuple[object, ...], **overrides: object) -> StateTrackingInstance:
    defaults: dict[str, object] = {
        "complexity": Complexity.APERIODIC,
        "num_variables": 2,
        "num_values": 3,
        "query": (0, 1),
        "final_state": (1, 2),
        "temporary_variable": None,
    }
    defaults.update(overrides)
    return StateTrackingInstance.create(operations=operations, **defaults)  # type: ignore[arg-type]


def test_instance_id_is_deterministic_and_field_sensitive() -> None:
    base_ops = (AssignConstant(target=0, value=1), AssignConstant(target=1, value=2))
    base = _instance(base_ops)
    # Deterministic: identical semantics reproduce the same content hash.
    assert base.instance_id == _instance(base_ops).instance_id
    # Every semantic field participates in the digest.
    variants = [
        _instance(base_ops, complexity=Complexity.R_TRIVIAL),
        _instance(base_ops, query=(0,)),
        _instance(base_ops, final_state=(2, 1)),
        _instance((AssignConstant(target=0, value=2), AssignConstant(target=1, value=2))),
        _instance((AssignVariable(target=0, source=1), AssignConstant(target=1, value=2))),
    ]
    ids = {base.instance_id, *(variant.instance_id for variant in variants)}
    assert len(ids) == len(variants) + 1


def test_instance_ids_do_not_collide_across_generated_sources() -> None:
    ids = set()
    for seed in range(400):
        source = generate_instance(
            complexity=Complexity.PERIODIC,
            num_variables=3,
            num_values=5,
            max_extra_assignments=12,
            rng=random.Random(seed),
        )
        ids.update(instance.instance_id for instance in derive_variants(source))
    # No accidental collisions between distinct semantic traces.
    assert len(ids) > 400


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
        num_variables=first.num_registers,
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
        derive_instance(source, complexity=complexity)
        for complexity in DERIVED_FAMILIES
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
        derive_instance(periodic, complexity=complexity)
        for complexity in DERIVED_FAMILIES
    )
    for variant in variants:
        assert variant.query == periodic.query
        assert len(variant.operations) == len(periodic.operations)
        assert tuple(operation.target for operation in variant.operations) == tuple(
            operation.target for operation in periodic.operations
        )
        assert variant.final_state == evaluate(
            num_variables=variant.num_registers,
            num_values=variant.num_values,
            operations=variant.operations,
        )


def test_strict_family_witnesses() -> None:
    aperiodic = generate_instance(
        complexity=Complexity.APERIODIC,
        num_variables=3,
        num_values=3,
        min_extra_assignments=3,
        max_extra_assignments=3,
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
    assert len(aperiodic.operations) == 7
    assert all(isinstance(operation, AssignConstant) for operation in aperiodic.operations)
    state: list[int | None] = [None] * periodic.num_registers
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
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        rng=random.Random(3),
    )
    lengths = dependency_lengths(
        num_variables=periodic.num_registers,
        operations=periodic.operations,
    )
    assert min(lengths[: periodic.num_variables]) >= 20
    assert max(lengths[: periodic.num_variables]) >= 25
    assert len(set(periodic.final_state[: periodic.num_variables])) == 4


def test_periodic_generation_uses_value_changing_updates_and_live_lineages() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=6,
        num_values=10,
        min_extra_assignments=1000,
        max_extra_assignments=1000,
        min_aperiodic_constants=100,
        max_aperiodic_constants=100,
        rng=random.Random(3),
    )
    initial_state = evaluate(
        num_variables=periodic.num_registers,
        num_values=periodic.num_values,
        operations=periodic.operations[: periodic.num_initial_assignments],
    )
    state = list(initial_state)
    copy_count = 0
    for operation in periodic.operations[periodic.num_initial_assignments :]:
        if isinstance(operation, AssignConstant):
            new_value = operation.value
        else:
            copy_count += 1
            new_value = state[operation.source]
        state[operation.target] = new_value

    assert tuple(state) == periodic.final_state
    assert len(set(periodic.final_state[: periodic.num_variables])) >= 2
    assert copy_count >= 800
    assert max(
        dependency_lengths(
            num_variables=periodic.num_registers,
            operations=periodic.operations,
        )[: periodic.num_variables]
    ) >= 40


@pytest.mark.parametrize("num_variables", (2, 4, 8, 16))
def test_swap_periodic_generation_uses_one_extra_tmp_register(
    num_variables: int,
) -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=num_variables,
        num_values=10,
        min_extra_assignments=96,
        max_extra_assignments=96,
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        rng=random.Random(37),
    )
    assert periodic.num_variables == num_variables
    assert periodic.num_registers == num_variables + 1
    assert periodic.temporary_variable == num_variables
    assert periodic.query == tuple(range(num_variables + 1))
    initial_state = evaluate(
        num_variables=periodic.num_registers,
        num_values=periodic.num_values,
        operations=periodic.operations[: periodic.num_initial_assignments],
    )
    assert sorted(periodic.final_state[:num_variables]) == sorted(initial_state[:num_variables])
    extras = periodic.operations[periodic.num_initial_assignments :]
    assert len(extras) == 96
    assert all(isinstance(operation, AssignVariable) for operation in extras)
    for index in range(0, len(extras), 3):
        load, move, restore = extras[index : index + 3]
        assert load.target == periodic.temporary_variable
        assert move.target == load.source
        assert restore.target == move.source
        assert restore.source == periodic.temporary_variable


def test_swap_periodic_generation_mixes_exactly_the_requested_constant_count() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=8,
        num_values=10,
        min_extra_assignments=80,
        max_extra_assignments=80,
        min_aperiodic_constants=16,
        max_aperiodic_constants=16,
        rng=random.Random(41),
    )

    extras = periodic.operations[periodic.num_initial_assignments :]
    assert sum(isinstance(operation, AssignConstant) for operation in extras) == 16
    assert sum(isinstance(operation, AssignVariable) for operation in extras) == 64
    assert all(
        operation.target != periodic.temporary_variable
        for operation in extras
        if isinstance(operation, AssignConstant)
    )


def test_two_real_variables_swap_without_collapsing_the_value_pool() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=10,
        min_extra_assignments=100,
        max_extra_assignments=100,
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        rng=random.Random(3),
    )
    assert len(set(periodic.final_state[:2])) == 2
    assert max(
        dependency_lengths(
            num_variables=periodic.num_registers,
            operations=periodic.operations,
        )[:2]
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


def test_sensitivity_summary_is_an_exact_reverse_dataflow_measure() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=3,
        operations=(
            AssignConstant(0, 0),
            AssignConstant(1, 1),
            AssignVariable(0, 1),
        ),
        query=(0, 1),
        final_state=(1, 1),
    )

    summary = sensitivity_summary(instance)
    # At the final boundary, each variable reaches its own output. Before the
    # copy, variable 1 reaches both outputs and variable 0 reaches neither.
    assert summary.sensitivity_state_count == 2
    assert summary.sensitivity_output_count == 2
    assert summary.sensitivity_density == pytest.approx(0.5)
    assert summary.sensitivity_coverage == pytest.approx(0.75)
    assert summary.mean_live_fanout == pytest.approx(4 / 3)

    # The derived aperiodic program resolves the copy to a literal, so the
    # same structural trace has less runtime state-to-output influence.
    aperiodic = derive_instance(instance, complexity=Complexity.APERIODIC)
    assert sensitivity_summary(aperiodic).sensitivity_density == pytest.approx(0.375)


def test_sensitivity_excludes_tmp_and_samples_completed_swaps() -> None:
    periodic = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=3,
        operations=(
            AssignConstant(0, 0),
            AssignConstant(1, 1),
            AssignConstant(2, 2),
            AssignVariable(2, 0),  # Scratch-only budget padding: not a sample boundary.
            AssignVariable(2, 0),
            AssignVariable(0, 1),
            AssignVariable(1, 2),
        ),
        query=(0, 1, 2),
        final_state=(1, 0, 0),
        temporary_variable=2,
    )

    summary = sensitivity_summary(periodic)
    assert summary.sensitivity_state_count == 2
    assert summary.sensitivity_output_count == 2
    assert summary.sensitivity_density == pytest.approx(0.5)
    assert summary.sensitivity_coverage == pytest.approx(1.0)
    assert summary.mean_live_fanout == pytest.approx(1.0)

    # Derived traces retain the swap transaction boundary even though their
    # three primitive assignments have all been resolved to constants.
    for complexity in (Complexity.APERIODIC, Complexity.R_TRIVIAL):
        derived = derive_instance(periodic, complexity=complexity)
        derived_summary = sensitivity_summary(derived)
        assert derived_summary.sensitivity_state_count == 2
        assert derived_summary.sensitivity_output_count == 2
        assert derived_summary.sensitivity_density == pytest.approx(0.25)
        assert derived_summary.sensitivity_coverage == pytest.approx(0.5)
        assert derived_summary.mean_live_fanout == pytest.approx(1.0)


def test_sensitivity_samples_mixed_constants_as_semantic_operations() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=3,
        operations=(
            AssignConstant(0, 0),
            AssignConstant(1, 1),
            AssignConstant(2, 2),
            AssignConstant(0, 2),
            AssignVariable(2, 0),
            AssignVariable(0, 1),
            AssignVariable(1, 2),
        ),
        query=(0, 1, 2),
        final_state=(1, 2, 2),
        temporary_variable=2,
    )

    summary = sensitivity_summary(instance)
    assert summary.sensitivity_state_count == 3
    assert summary.sensitivity_output_count == 2
    assert summary.sensitivity_density == pytest.approx(5 / 12)
    assert summary.sensitivity_coverage == pytest.approx(5 / 6)
    assert summary.mean_live_fanout == pytest.approx(1.0)
