from __future__ import annotations

import random
from itertools import product
from pathlib import Path

import pytest
import yaml

from state_bench.config import FormatterSpec, load_experiment, load_formatter
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import (
    derive_instance,
    derive_variants,
    derived_families,
    generate_instance,
    generate_source,
)
from state_bench.model import (
    AssignConstant,
    AssignIncrement,
    AssignVariable,
    Complexity,
    StateTrackingInstance,
)
from state_bench.rendering import render_instance
from state_bench.semantics import dependency_lengths, evaluate, sensitivity_summary

ROOT = Path(__file__).resolve().parents[1]
MODULAR_FORMATTER = ROOT / "configs/formats/integer-code-modular.yaml"


def _solvable(seed: int, **overrides: object) -> StateTrackingInstance:
    parameters: dict[str, object] = {
        "complexity": Complexity.SOLVABLE,
        "num_variables": 3,
        "num_values": 10,
        "min_extra_assignments": 1,
        "max_extra_assignments": 30,
    }
    parameters.update(overrides)
    return generate_instance(rng=random.Random(seed), **parameters)  # type: ignore[arg-type]


def test_increment_semantics_wrap_modulo_num_values() -> None:
    final = evaluate(
        num_variables=1,
        num_values=100,
        operations=(AssignConstant(0, 98), AssignIncrement(0, 5)),
    )
    assert final == (3,)
    with pytest.raises(ValueError, match="uninitialized"):
        evaluate(num_variables=1, num_values=100, operations=(AssignIncrement(0, 5),))
    with pytest.raises(ValueError, match="out-of-range delta"):
        evaluate(
            num_variables=1,
            num_values=100,
            operations=(AssignConstant(0, 1), AssignIncrement(0, 100)),
        )


def test_increment_participates_in_instance_id_and_record_round_trip() -> None:
    def instance(operations: tuple[object, ...], final_state: tuple[int, ...]) -> StateTrackingInstance:
        return StateTrackingInstance.create(
            complexity=Complexity.SOLVABLE,
            num_variables=1,
            num_values=10,
            operations=operations,  # type: ignore[arg-type]
            query=(0,),
            final_state=final_state,
        )

    incremented = instance((AssignConstant(0, 4), AssignIncrement(0, 3)), (7,))
    other_delta = instance((AssignConstant(0, 4), AssignIncrement(0, 2)), (6,))
    # A constant that happens to write the same value is a different trace.
    literal = instance((AssignConstant(0, 4), AssignConstant(0, 7)), (7,))
    assert len({incremented.instance_id, other_delta.instance_id, literal.instance_id}) == 3
    round_tripped = StateTrackingInstance.from_record(incremented.to_record())
    assert round_tripped == incremented
    assert incremented.to_record()["operations"][1] == {"op": "increment", "target": 0, "delta": 3}


@pytest.mark.parametrize("seed", range(40))
def test_solvable_generation_is_self_consistent_and_uses_only_constants_and_increments(
    seed: int,
) -> None:
    num_variables = 1 + seed % 6
    source = generate_source(
        complexity=Complexity.SOLVABLE,
        num_variables=num_variables,
        num_values=max(num_variables, 2 + seed % 20),
        min_extra_assignments=1,
        max_extra_assignments=1 + (seed * 3) % 60,
        min_aperiodic_constants=0 if seed % 4 == 0 else 1,
        max_aperiodic_constants=0 if seed % 4 == 0 else 16,
        distinct_initial_values=seed % 2 == 0,
        rng=random.Random(seed),
    )
    assert source.complexity is Complexity.SOLVABLE
    assert source.temporary_variable is None
    assert source.query == tuple(range(source.num_variables))
    assert source.final_state == evaluate(
        num_variables=source.num_variables,
        num_values=source.num_values,
        operations=source.operations,
    )
    kinds = {type(operation) for operation in source.operations}
    assert AssignVariable not in kinds
    assert AssignIncrement in kinds, "strict solvable sources carry at least one increment"
    # Every register is initialised by a literal before anything else touches it.
    initial = source.operations[: source.num_variables]
    assert all(isinstance(operation, AssignConstant) for operation in initial)
    assert sorted(operation.target for operation in initial) == list(range(source.num_variables))
    extra = source.operations[source.num_variables :]
    assert 1 <= len(extra) <= 1 + (seed * 3) % 60
    for operation in extra:
        if isinstance(operation, AssignIncrement):
            assert 1 <= operation.delta < source.num_values


def test_pure_solvable_traces_have_no_constant_resets_after_initialisation() -> None:
    source = generate_source(
        complexity=Complexity.SOLVABLE,
        num_variables=4,
        num_values=100,
        min_extra_assignments=50,
        max_extra_assignments=50,
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        rng=random.Random(7),
    )
    extra = source.operations[4:]
    assert len(extra) == 50
    assert all(isinstance(operation, AssignIncrement) for operation in extra)
    # Increment targets are balanced across variables.
    counts = [sum(operation.target == variable for operation in extra) for variable in range(4)]
    assert max(counts) - min(counts) <= 1


def test_solvable_mixes_exactly_the_requested_constant_count() -> None:
    source = generate_source(
        complexity=Complexity.SOLVABLE,
        num_variables=3,
        num_values=100,
        min_extra_assignments=40,
        max_extra_assignments=40,
        min_aperiodic_constants=7,
        max_aperiodic_constants=7,
        rng=random.Random(3),
    )
    extra = source.operations[3:]
    assert sum(isinstance(operation, AssignConstant) for operation in extra) == 7
    assert sum(isinstance(operation, AssignIncrement) for operation in extra) == 33


def test_increment_deltas_are_sampled_within_the_configured_range() -> None:
    source = generate_source(
        complexity=Complexity.SOLVABLE,
        num_variables=2,
        num_values=100,
        min_extra_assignments=200,
        max_extra_assignments=200,
        min_aperiodic_constants=0,
        max_aperiodic_constants=0,
        min_increment_delta=1,
        max_increment_delta=9,
        rng=random.Random(11),
    )
    deltas = {operation.delta for operation in source.operations if isinstance(operation, AssignIncrement)}
    assert deltas <= set(range(1, 10))
    assert len(deltas) > 1
    with pytest.raises(ValueError, match="smaller than num_values"):
        generate_source(
            complexity=Complexity.SOLVABLE,
            num_variables=2,
            num_values=10,
            max_extra_assignments=4,
            max_increment_delta=10,
            rng=random.Random(0),
        )


def test_derived_families_from_a_solvable_source() -> None:
    assert derived_families(Complexity.SOLVABLE) == (
        Complexity.R_TRIVIAL,
        Complexity.APERIODIC,
        Complexity.SOLVABLE,
    )
    assert derived_families(Complexity.PERIODIC) == (
        Complexity.R_TRIVIAL,
        Complexity.APERIODIC,
        Complexity.PERIODIC,
    )
    with pytest.raises(ValueError, match="not a source family"):
        derived_families(Complexity.APERIODIC)


@pytest.mark.parametrize("seed", range(20))
def test_solvable_derivations_resolve_increments_and_preserve_alignment(seed: int) -> None:
    source = _solvable(seed, num_variables=1 + seed % 5, num_values=3 + seed % 9)
    r_trivial, aperiodic, solvable = derive_variants(source)
    assert solvable is source
    assert (r_trivial, aperiodic) == (
        derive_instance(source, complexity=Complexity.R_TRIVIAL),
        derive_instance(source, complexity=Complexity.APERIODIC),
    )
    assert aperiodic.complexity is Complexity.APERIODIC
    assert aperiodic.final_state == source.final_state
    assert r_trivial.complexity is Complexity.R_TRIVIAL
    for variant in (r_trivial, aperiodic):
        assert variant.query == source.query
        assert variant.temporary_variable is None
        assert len(variant.operations) == len(source.operations)
        assert all(isinstance(operation, AssignConstant) for operation in variant.operations)
        assert tuple(operation.target for operation in variant.operations) == tuple(
            operation.target for operation in source.operations
        )
        assert variant.final_state == evaluate(
            num_variables=variant.num_variables,
            num_values=variant.num_values,
            operations=variant.operations,
        )
    # Aperiodic literals reproduce the running solvable state step by step.
    state: dict[int, int] = {}
    for operation, literal in zip(source.operations, aperiodic.operations, strict=True):
        assert isinstance(literal, AssignConstant)
        if isinstance(operation, AssignConstant):
            expected = operation.value
        else:
            assert isinstance(operation, AssignIncrement)
            expected = (state[operation.target] + operation.delta) % source.num_values
        assert literal.value == expected
        state[operation.target] = expected


def test_solvable_cannot_be_derived_from_a_periodic_source_and_vice_versa() -> None:
    periodic = generate_instance(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=5,
        max_extra_assignments=6,
        rng=random.Random(0),
    )
    with pytest.raises(ValueError, match="cannot derive 'solvable'"):
        derive_instance(periodic, complexity=Complexity.SOLVABLE)
    solvable = _solvable(0)
    with pytest.raises(ValueError, match="cannot derive 'periodic'"):
        derive_instance(solvable, complexity=Complexity.PERIODIC)
    with pytest.raises(ValueError, match="not a source family"):
        derive_variants(derive_instance(solvable, complexity=Complexity.APERIODIC))


def test_generate_instance_routes_solvable_requests_to_a_solvable_source() -> None:
    instance = _solvable(5)
    assert instance.complexity is Complexity.SOLVABLE
    aperiodic = generate_instance(
        complexity=Complexity.APERIODIC,
        source_complexity=Complexity.SOLVABLE,
        num_variables=3,
        num_values=10,
        min_extra_assignments=1,
        max_extra_assignments=30,
        rng=random.Random(5),
    )
    # Same seed, same source, different derivation.
    assert aperiodic == derive_instance(instance, complexity=Complexity.APERIODIC)


def test_increments_extend_dependency_chains_and_pass_sensitivity_through() -> None:
    operations = (
        AssignConstant(0, 1),
        AssignConstant(1, 2),
        AssignIncrement(0, 3),
        AssignIncrement(0, 4),
        AssignConstant(1, 9),
    )
    assert dependency_lengths(num_variables=2, operations=operations) == (2, 0)
    instance = StateTrackingInstance.create(
        complexity=Complexity.SOLVABLE,
        num_variables=2,
        num_values=10,
        operations=operations,
        query=(0, 1),
        final_state=evaluate(num_variables=2, num_values=10, operations=operations),
    )
    summary = sensitivity_summary(instance)
    # Boundaries after ops 2..5 (fully initialised onwards). Variable 0 stays
    # live through both increments at every boundary; variable 1 is live only
    # at the final boundary because the last reset erases its earlier state.
    assert summary.sensitivity_state_count == 4
    assert summary.sensitivity_coverage == pytest.approx((1 + 1 + 1 + 2) / (4 * 2))
    # Each live variable feeds exactly one of the two queried outputs.
    assert summary.mean_live_fanout == 1.0
    assert summary.sensitivity_density == pytest.approx(summary.sensitivity_coverage / 2)


def test_solvable_distractors_prefer_increment_offsets() -> None:
    operations = (
        AssignConstant(0, 10),
        AssignConstant(1, 20),
        AssignIncrement(0, 7),
        AssignIncrement(1, 7),
    )
    instance = StateTrackingInstance.create(
        complexity=Complexity.SOLVABLE,
        num_variables=2,
        num_values=100,
        operations=operations,
        query=(0, 1),
        final_state=(17, 27),
    )
    states = generate_incorrect_states(instance, num_distractors=4, rng=random.Random(0))
    assert len(states) == 4
    assert len(set(states)) == 4
    assert all(state != (17, 27) for state in states)
    offsets = {value - truth for state in states for value, truth in zip(state, (17, 27), strict=True)}
    # The single rearrangement is (27, 17); everything else is off by the trace's delta.
    assert offsets <= {-10, 0, 10, -7, 7}
    assert {-7, 7} & offsets


def _modular_formatter() -> FormatterSpec:
    return load_formatter(MODULAR_FORMATTER)


def test_modular_formatter_renders_truthful_arithmetic() -> None:
    formatter = _modular_formatter()
    instance = StateTrackingInstance.create(
        complexity=Complexity.SOLVABLE,
        num_variables=2,
        num_values=100,
        operations=(AssignConstant(0, 98), AssignConstant(1, 5), AssignIncrement(0, 5)),
        query=(0, 1),
        final_state=(3, 5),
    )
    rendered = render_instance(instance, formatter, rng=random.Random(0), incorrect_states=())
    assert "a <- 98; b <- 5; a <- (a + 5) mod 100;" in rendered.prefix
    assert rendered.completion == "a ?= 3; b ?= 5;"
    assert rendered.temporary_variable is None


def test_formatters_without_an_increment_template_reject_increment_traces() -> None:
    formatter = load_formatter(ROOT / "configs/formats/integer-code.yaml")
    instance = StateTrackingInstance.create(
        complexity=Complexity.SOLVABLE,
        num_variables=1,
        num_values=10,
        operations=(AssignConstant(0, 1), AssignIncrement(0, 2)),
        query=(0,),
        final_state=(3,),
    )
    with pytest.raises(ValueError, match="assign_increment"):
        render_instance(instance, formatter, rng=random.Random(0))


def _write_experiment(tmp_path: Path, **overrides: object) -> Path:
    source = yaml.safe_load((ROOT / "configs/experiments/solvable_test.yaml").read_text(encoding="utf-8"))
    source["dataset"]["output_dir"] = str(tmp_path / "output")
    source["splits"] = {"test": {"samples_per_condition": 2}}
    source["generation"]["conditions"][0]["extra_assignments"] = {"min": 2, "max": 6}
    source["generation"]["conditions"][0]["num_variables"] = [1, 3]
    # Let the condition use whichever formatters the test supplies.
    source["generation"]["conditions"][0].pop("formatters", None)
    source["formatters"] = [{"config": str(MODULAR_FORMATTER)}]
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            source[section][field] = value
        else:
            source[section] = value
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def test_solvable_experiment_loads_and_builds_linked_families(tmp_path: Path) -> None:
    from state_bench.io import build_dataset, read_jsonl
    from state_bench.preview import available_config_names, generate_preview

    loaded = load_experiment(_write_experiment(tmp_path))
    assert available_config_names(loaded) == (
        "integer-code-modular--aperiodic",
        "integer-code-modular--r-trivial",
        "integer-code-modular--solvable",
    )
    preview = generate_preview(loaded, config_name="integer-code-modular--solvable")
    assert preview.instance.complexity is Complexity.SOLVABLE
    assert "mod 100" in preview.rendered.prefix

    root = build_dataset(loaded, workers=1)
    canonical = read_jsonl(root / "canonical/test.jsonl")
    assert sorted({row["complexity"] for row in canonical}) == ["aperiodic", "r-trivial", "solvable"]
    by_source: dict[str, set[str]] = {}
    for row in canonical:
        by_source.setdefault(row["source_example_id"], set()).add(row["complexity"])
    assert all(families == {"aperiodic", "r-trivial", "solvable"} for families in by_source.values())
    solvable_rows = read_jsonl(root / "rendered/integer-code-modular--solvable/test.jsonl")
    assert len(solvable_rows) == 4
    for row in solvable_rows:
        assert row["complexity"] == "solvable"
        assert row["temporary_variable"] is None
        assert any(operation["op"] == "increment" for operation in row["operations"])
        assert row["completion"] in row["choices"]
        assert len(set(row["choices"])) == 5


def test_solvable_conditions_require_a_zero_based_increment_formatter(tmp_path: Path) -> None:
    path = _write_experiment(
        tmp_path,
        formatters=[{"config": str(ROOT / "configs/formats/integer-code.yaml")}],
    )
    with pytest.raises(ValueError, match="assign_increment"):
        load_experiment(path)

    one_based = yaml.safe_load(MODULAR_FORMATTER.read_text(encoding="utf-8"))
    one_based["vocabulary"]["values"] = [str(value) for value in range(1, 101)]
    one_based_path = tmp_path / "one-based.yaml"
    one_based_path.write_text(yaml.safe_dump(one_based), encoding="utf-8")
    path = _write_experiment(tmp_path, formatters=[{"config": str(one_based_path)}])
    with pytest.raises(ValueError, match='values "0" through "99"'):
        load_experiment(path)


def test_experiments_reject_delta_ranges_that_reach_the_modulus(tmp_path: Path) -> None:
    path = _write_experiment(tmp_path, generation__increment_deltas={"min": 1, "max": 100})
    with pytest.raises(ValueError, match="increment_deltas.max"):
        load_experiment(path)


# --- Algebraic sanity check ---------------------------------------------------

type Transformation = tuple[int, ...]


def _compose(left: Transformation, right: Transformation) -> Transformation:
    return tuple(right[left[index]] for index in range(len(left)))


def _closure(generators: set[Transformation], identity: Transformation) -> set[Transformation]:
    monoid = {identity, *generators}
    frontier = set(monoid)
    while frontier:
        new = {_compose(left, right) for left in frontier for right in generators} - monoid
        monoid |= new
        frontier = new
    return monoid


def _maximal_subgroups(monoid: set[Transformation]) -> list[set[Transformation]]:
    """Return the group H-class of every idempotent (all maximal subgroups)."""

    groups = []
    for idempotent in monoid:
        if _compose(idempotent, idempotent) != idempotent:
            continue
        # x is in the H-class of e iff ex = xe = x and x has an inverse relative to e.
        h_class = {
            element
            for element in monoid
            if _compose(idempotent, element) == element
            and _compose(element, idempotent) == element
            and any(
                _compose(element, other) == idempotent and _compose(other, element) == idempotent
                for other in monoid
            )
        }
        groups.append(h_class)
    return groups


def test_increment_transition_monoid_has_only_abelian_groups_for_tiny_domain() -> None:
    """Constants + increments on Z_4^2 yield only abelian (hence solvable) groups.

    Every maximal subgroup of the transition monoid is a product of cyclic
    groups, so the Krohn-Rhodes decomposition uses only prime cyclic groups.
    """

    modulus = 4
    states = list(product(range(modulus), repeat=2))
    state_index = {state: index for index, state in enumerate(states)}

    def reset(target: int, value: int) -> Transformation:
        return tuple(
            state_index[tuple(value if index == target else current for index, current in enumerate(state))]
            for state in states
        )

    def increment(target: int, delta: int) -> Transformation:
        return tuple(
            state_index[
                tuple(
                    (current + delta) % modulus if index == target else current
                    for index, current in enumerate(state)
                )
            ]
            for state in states
        )

    identity = tuple(range(len(states)))
    generators = {reset(target, value) for target in range(2) for value in range(modulus)}
    generators |= {increment(target, delta) for target in range(2) for delta in range(1, modulus)}
    monoid = _closure(generators, identity)
    groups = _maximal_subgroups(monoid)
    assert any(len(group) == modulus**2 for group in groups), "the unit group should be Z_4 x Z_4"
    for group in groups:
        for left in group:
            for right in group:
                assert _compose(left, right) == _compose(right, left)


def test_swap_transition_monoid_has_a_non_abelian_group_for_three_variables() -> None:
    """The periodic contrast: swaps of three variables realise the non-abelian S_3."""

    values = 3
    states = list(product(range(values), repeat=3))
    state_index = {state: index for index, state in enumerate(states)}

    def swap(left: int, right: int) -> Transformation:
        transformed = []
        for state in states:
            result = list(state)
            result[left], result[right] = result[right], result[left]
            transformed.append(state_index[tuple(result)])
        return tuple(transformed)

    identity = tuple(range(len(states)))
    monoid = _closure({swap(0, 1), swap(1, 2)}, identity)
    assert len(monoid) == 6
    first, second = swap(0, 1), swap(1, 2)
    assert _compose(first, second) != _compose(second, first)
