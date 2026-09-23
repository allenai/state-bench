from __future__ import annotations

import random
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import yaml

from state_bench.config import FormatterSpec, load_experiment, load_formatter, split_families
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import (
    DERIVED_FAMILIES,
    derive_instance,
    derive_variants,
    generate_instance,
    generate_periodic_source,
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
EXPERIMENTS = ROOT / "configs/experiments"
MODULAR_FORMATTER = ROOT / "configs/formats/integer-code-modular.yaml"


def _periodic(seed: int, **overrides: object) -> StateTrackingInstance:
    parameters: dict[str, object] = {
        "num_variables": 3,
        "num_values": 10,
        "min_extra_assignments": 3,
        "max_extra_assignments": 40,
    }
    parameters.update(overrides)
    return generate_periodic_source(rng=random.Random(seed), **parameters)  # type: ignore[arg-type]


def _trajectory(instance: StateTrackingInstance) -> list[tuple[int | None, ...]]:
    """Return the register state after every operation."""

    state: list[int | None] = [None] * instance.num_registers
    states = []
    for operation in instance.operations:
        if isinstance(operation, AssignConstant):
            state[operation.target] = operation.value
        elif isinstance(operation, AssignVariable):
            state[operation.target] = state[operation.source]
        else:
            current = state[operation.target]
            assert current is not None
            state[operation.target] = (current + operation.delta) % instance.num_values
        states.append(tuple(state))
    return states


def test_increment_semantics_wrap_modulo_num_values() -> None:
    final = evaluate(
        num_variables=1,
        num_values=100,
        operations=(AssignConstant(0, 98), AssignIncrement(0, 5)),
    )
    assert final == (3,)
    # Zero is the identity increment the derivation emits for unchanged copies.
    assert evaluate(
        num_variables=1,
        num_values=100,
        operations=(AssignConstant(0, 42), AssignIncrement(0, 0)),
    ) == (42,)
    with pytest.raises(ValueError, match="uninitialized"):
        evaluate(num_variables=1, num_values=100, operations=(AssignIncrement(0, 5),))
    for delta in (-1, 100):
        with pytest.raises(ValueError, match="out-of-range delta"):
            evaluate(
                num_variables=1,
                num_values=100,
                operations=(AssignConstant(0, 1), AssignIncrement(0, delta)),
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


# --- Derivation from periodic sources ----------------------------------------


def test_solvable_rewrites_each_swap_copy_as_a_self_increment() -> None:
    # a=2, b=5, TMP=7; then swap(a, b): TMP <- a; a <- b; b <- TMP.
    source = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=10,
        operations=(
            AssignConstant(0, 2),
            AssignConstant(1, 5),
            AssignConstant(2, 7),
            AssignVariable(2, 0),
            AssignVariable(0, 1),
            AssignVariable(1, 2),
            AssignConstant(0, 9),
        ),
        query=(0, 1, 2),
        final_state=(9, 2, 2),
        temporary_variable=2,
    )

    solvable = derive_instance(source, complexity=Complexity.SOLVABLE)

    assert solvable.operations == (
        AssignConstant(0, 2),
        AssignConstant(1, 5),
        AssignConstant(2, 7),
        AssignIncrement(2, 5),  # TMP: 7 -> 2
        AssignIncrement(0, 3),  # a: 2 -> 5
        AssignIncrement(1, 7),  # b: 5 -> 2 (wraps)
        AssignConstant(0, 9),
    )
    assert solvable.final_state == source.final_state
    assert solvable.temporary_variable == source.temporary_variable
    assert solvable.query == source.query
    # The solvable and aperiodic rewrites write the same value at every step.
    aperiodic = derive_instance(source, complexity=Complexity.APERIODIC)
    assert _trajectory(solvable) == _trajectory(aperiodic) == _trajectory(source)


def test_copies_that_leave_their_target_unchanged_become_identity_increments() -> None:
    source = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=10,
        operations=(AssignConstant(0, 4), AssignConstant(1, 4), AssignVariable(0, 1)),
        query=(0, 1),
        final_state=(4, 4),
    )
    solvable = derive_instance(source, complexity=Complexity.SOLVABLE)
    assert solvable.operations[-1] == AssignIncrement(0, 0)
    assert evaluate(
        num_variables=2, num_values=10, operations=solvable.operations
    ) == source.final_state


@pytest.mark.parametrize("seed", range(40))
def test_solvable_derivation_tracks_the_periodic_state_exactly(seed: int) -> None:
    source = _periodic(
        seed,
        num_variables=2 + seed % 7,
        num_values=3 + seed % 98,
        min_aperiodic_constants=seed % 5,
        max_aperiodic_constants=seed % 5,
    )
    solvable = derive_instance(source, complexity=Complexity.SOLVABLE)

    assert solvable.complexity is Complexity.SOLVABLE
    assert len(solvable.operations) == len(source.operations)
    assert _trajectory(solvable) == _trajectory(source)
    assert solvable.final_state == source.final_state == evaluate(
        num_variables=solvable.num_registers,
        num_values=solvable.num_values,
        operations=solvable.operations,
    )
    for original, rewritten in zip(source.operations, solvable.operations, strict=True):
        assert rewritten.target == original.target
        if isinstance(original, AssignConstant):
            assert rewritten == original
        else:
            assert isinstance(rewritten, AssignIncrement)
            assert 0 <= rewritten.delta < source.num_values
    # Registers never read one another, so no copy survives.
    assert not any(isinstance(operation, AssignVariable) for operation in solvable.operations)


def test_derive_variants_returns_requested_families_in_order() -> None:
    source = _periodic(3)
    everything = derive_variants(source)
    assert tuple(instance.complexity for instance in everything) == DERIVED_FAMILIES
    assert everything[-1] is source
    assert derive_variants(source, families=(Complexity.SOLVABLE, Complexity.R_TRIVIAL)) == (
        everything[2],
        everything[0],
    )
    assert everything == tuple(
        derive_instance(source, complexity=complexity) for complexity in DERIVED_FAMILIES
    )
    assert generate_instance(
        complexity=Complexity.SOLVABLE,
        num_variables=3,
        num_values=10,
        min_extra_assignments=3,
        max_extra_assignments=40,
        rng=random.Random(3),
    ) == everything[2]


def test_only_periodic_sources_can_be_derived() -> None:
    solvable = derive_instance(_periodic(4), complexity=Complexity.SOLVABLE)
    with pytest.raises(ValueError, match="periodic source"):
        derive_instance(solvable, complexity=Complexity.APERIODIC)


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


# --- Experiments ---------------------------------------------------------------


def _write_experiment(
    tmp_path: Path,
    name: str,
    *,
    families: list[str] | None,
    formatter: Path = MODULAR_FORMATTER,
    disjoint: bool = True,
    train_families: list[str] | None = None,
) -> Path:
    """Write a tiny two-split experiment whose sources collide often across splits.

    Two variables over three values with exactly one swap and no constant resets
    leave few distinct traces, so train keeps resampling sources seen in test.
    """

    generation: dict[str, Any] = {
        "aperiodic_constants": {"min": 0, "max": 0},
        "conditions": [
            {
                "complexity": "periodic",
                "num_variables": [2],
                "num_values": 3,
                "extra_assignments": {"min": 3, "max": 3},
            }
        ],
    }
    if families is not None:
        generation["families"] = families
    experiment = {
        "schema_version": 1,
        "dataset": {"name": "linked", "seed": 5, "output_dir": str(tmp_path / name)},
        "splits": {
            "test": {"samples_per_condition": 200},
            "train": {
                "samples_per_condition": 12,
                **({"disjoint_from": ["test"]} if disjoint else {}),
                **({"families": train_families} if train_families is not None else {}),
            },
        },
        "generation": generation,
        "formatters": [{"config": str(formatter)}],
        "output": {
            "minimal_splits": ["train"],
            "omit_canonical_splits": ["train"],
            "multiple_choice": {"num_distractors": 4, "shuffle_choices": True},
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return path


def _sources(root: Path, config: str, split: str) -> dict[str, str]:
    from state_bench.io import read_jsonl

    rows = read_jsonl(root / "rendered" / config / f"{split}.jsonl")
    return {row["source_example_id"]: row["source_instance_id"] for row in rows}


def test_a_solvable_only_build_regenerates_the_periodic_builds_sources(tmp_path: Path) -> None:
    from state_bench.io import build_dataset, read_jsonl

    periodic_root = build_dataset(
        load_experiment(
            _write_experiment(tmp_path, "periodic", families=["r-trivial", "aperiodic", "periodic"])
        ),
        workers=1,
    )
    solvable_loaded = load_experiment(_write_experiment(tmp_path, "solvable", families=["solvable"]))
    solvable_root = build_dataset(solvable_loaded, workers=1)
    unconstrained_root = build_dataset(
        load_experiment(
            _write_experiment(tmp_path, "unconstrained", families=["periodic"], disjoint=False)
        ),
        workers=1,
    )

    assert sorted(path.name for path in (solvable_root / "rendered").iterdir()) == [
        "integer-code-modular--solvable"
    ]
    for split in ("test", "train"):
        periodic = _sources(periodic_root, "integer-code-modular--periodic", split)
        assert _sources(periodic_root, "integer-code-modular--aperiodic", split) == periodic
        assert _sources(solvable_root, "integer-code-modular--solvable", split) == periodic
    # The fixture really exercises disjointness resampling, which the
    # solvable-only build must reproduce although it renders no periodic rows.
    assert _sources(unconstrained_root, "integer-code-modular--periodic", "train") != _sources(
        periodic_root, "integer-code-modular--periodic", "train"
    )
    # Persisted test IDs cover every family, emitted or not.
    periodic_ids = set((periodic_root / "semantics/test.ids").read_text().split())
    solvable_ids = set((solvable_root / "semantics/test.ids").read_text().split())
    assert solvable_ids == periodic_ids

    canonical = read_jsonl(solvable_root / "canonical/test.jsonl")
    assert {row["complexity"] for row in canonical} == {"solvable"}
    for row in read_jsonl(solvable_root / "rendered/integer-code-modular--solvable/test.jsonl"):
        assert "mod 3;" in row["prefix"]
        assert row["temporary_variable"] == "TMP"
        assert row["completion"] in row["choices"]


def test_splits_can_emit_a_subset_of_the_generated_families(tmp_path: Path) -> None:
    from state_bench.io import build_dataset

    periodic_root = build_dataset(
        load_experiment(
            _write_experiment(tmp_path, "periodic", families=["r-trivial", "aperiodic", "periodic"])
        ),
        workers=1,
    )
    loaded = load_experiment(
        _write_experiment(tmp_path, "split", families=None, train_families=["solvable"])
    )
    assert split_families(loaded.spec, "test") == DERIVED_FAMILIES
    assert split_families(loaded.spec, "train") == (Complexity.SOLVABLE,)
    root = build_dataset(loaded, workers=1)

    rendered = {
        (path.parent.name, path.stem) for path in (root / "rendered").glob("*/*.jsonl")
    }
    assert rendered == {
        *(("integer-code-modular--" + family.value, "test") for family in DERIVED_FAMILIES),
        ("integer-code-modular--solvable", "train"),
    }
    assert _sources(root, "integer-code-modular--solvable", "train") == _sources(
        periodic_root, "integer-code-modular--periodic", "train"
    )

    with pytest.raises(ValueError, match="missing from generation.families"):
        load_experiment(
            _write_experiment(
                tmp_path, "bad", families=["periodic"], train_families=["solvable"]
            )
        )


def test_experiments_emit_every_family_by_default(tmp_path: Path) -> None:
    from state_bench.preview import available_config_names, generate_preview

    loaded = load_experiment(_write_experiment(tmp_path, "all", families=None))
    assert loaded.spec.generation.emitted_families() == DERIVED_FAMILIES
    assert available_config_names(loaded) == (
        "integer-code-modular--aperiodic",
        "integer-code-modular--periodic",
        "integer-code-modular--r-trivial",
        "integer-code-modular--solvable",
    )
    preview = generate_preview(loaded, config_name="integer-code-modular--solvable")
    assert preview.instance.complexity is Complexity.SOLVABLE
    assert "mod 3;" in preview.rendered.prefix


def test_the_solvable_family_requires_a_zero_based_increment_formatter(tmp_path: Path) -> None:
    integer_code = ROOT / "configs/formats/integer-code.yaml"
    with pytest.raises(ValueError, match="assign_increment"):
        load_experiment(_write_experiment(tmp_path, "one", families=None, formatter=integer_code))
    load_experiment(
        _write_experiment(
            tmp_path, "two", families=["r-trivial", "aperiodic", "periodic"], formatter=integer_code
        )
    )

    one_based = yaml.safe_load(MODULAR_FORMATTER.read_text(encoding="utf-8"))
    one_based["vocabulary"]["values"] = [str(value) for value in range(1, 101)]
    one_based_path = tmp_path / "one-based.yaml"
    one_based_path.write_text(yaml.safe_dump(one_based), encoding="utf-8")
    with pytest.raises(ValueError, match='values "0" through "2"'):
        load_experiment(_write_experiment(tmp_path, "three", families=None, formatter=one_based_path))


def test_conditions_describe_periodic_sources_only(tmp_path: Path) -> None:
    path = _write_experiment(tmp_path, "bad", families=None)
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["generation"]["conditions"][0]["complexity"] = "solvable"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    with pytest.raises(ValueError, match="periodic source traces"):
        load_experiment(path)

    for families in ([], ["solvable", "solvable"]):
        path = _write_experiment(tmp_path, "bad", families=families)
        with pytest.raises(ValueError, match="families"):
            load_experiment(path)


@pytest.mark.parametrize(
    ("solvable_name", "periodic_name"),
    [("solvable.yaml", "default.yaml"), ("solvable_test.yaml", "periodic_test_mixed.yaml")],
)
def test_solvable_experiments_stay_in_lockstep_with_their_periodic_twins(
    solvable_name: str, periodic_name: str
) -> None:
    """Everything that determines a source must match, or rows stop joining."""

    def normalized(name: str) -> dict[str, Any]:
        experiment = yaml.safe_load((EXPERIMENTS / name).read_text(encoding="utf-8"))
        experiment["dataset"].pop("output_dir")
        experiment["generation"].pop("families")
        experiment.pop("formatters")
        for split in experiment["splits"].values():
            split.pop("formatters", None)
            split.pop("families", None)
        for condition in experiment["generation"]["conditions"]:
            condition.pop("formatters", None)
        experiment.get("huggingface", {}).get("repository", {}).pop("repo_id", None)
        return experiment

    assert normalized(solvable_name) == normalized(periodic_name)
    solvable = load_experiment(EXPERIMENTS / solvable_name)
    assert solvable.spec.generation.emitted_families() == DERIVED_FAMILIES
    assert [formatter.name for formatter in solvable.formatters] == ["integer-code-modular"]
    if "train" in solvable.spec.splits:
        assert split_families(solvable.spec, "train") == (Complexity.SOLVABLE,)


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
