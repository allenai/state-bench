from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from state_bench.config import FillerSpec, LogSpacedAssignmentSchedule, load_experiment
from state_bench.scheduling import (
    allocate_bucket_counts,
    build_assignment_plan,
    build_filler_plan,
    build_stratified_filler_plan,
    iter_randomized_assignment_plan,
    log_spaced_buckets,
)

ROOT = Path(__file__).resolve().parents[1]


def _schedule(**updates: object) -> LogSpacedAssignmentSchedule:
    values = {
        "distribution": "log-spaced",
        "min": 1,
        "max": 4096,
        "growth_factor": 2.0,
        "sample_decay": 0.65,
        "jitter_fraction": 0.10,
        "min_samples_per_bucket": 8,
        **updates,
    }
    return LogSpacedAssignmentSchedule.model_validate(values)


def test_log_spaced_buckets_reach_thousands_exponentially() -> None:
    assert log_spaced_buckets(_schedule()) == (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    )


def test_bucket_allocation_preserves_budget_and_sparsifies_long_examples() -> None:
    counts = allocate_bucket_counts(_schedule(), total_samples=6000)
    assert sum(counts.values()) == 6000
    assert counts[1] > counts[128] > counts[4096]
    assert counts[4096] >= 8


def test_assignment_plan_is_deterministic_and_respects_jitter() -> None:
    schedule = _schedule(max=16, min_samples_per_bucket=1)
    first = build_assignment_plan(schedule, total_samples=20, rng=random.Random(5))
    second = build_assignment_plan(schedule, total_samples=20, rng=random.Random(5))
    assert first == second
    assert len(first) == 20
    counts = Counter(item.target_extra_assignments for item in first)
    assert set(counts) == {1, 2, 4, 8, 16}
    for item in first:
        assert item.target_extra_assignments is not None
        assert item.min_extra_assignments <= item.target_extra_assignments
        assert item.max_extra_assignments >= item.target_extra_assignments


def test_streaming_assignment_plan_is_deterministic_and_preserves_exact_counts() -> None:
    schedule = _schedule(max=16, min_samples_per_bucket=1)
    expected_counts = allocate_bucket_counts(schedule, total_samples=1000)
    first = tuple(
        iter_randomized_assignment_plan(schedule, total_samples=1000, rng=random.Random(5))
    )
    second = tuple(
        iter_randomized_assignment_plan(schedule, total_samples=1000, rng=random.Random(5))
    )

    assert first == second
    assert Counter(item.target_extra_assignments for item in first) == expected_counts


def test_too_small_sample_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="need at least"):
        allocate_bucket_counts(_schedule(), total_samples=10)


def test_long_context_experiment_config_loads() -> None:
    loaded = load_experiment(ROOT / "configs/experiments/long-context.yaml")
    assert loaded.spec.dataset.name == "state-tracking-long-context-v1"
    assert loaded.spec.splits["test"].samples_per_condition == 340
    train = loaded.spec.splits["train"]
    assert train.samples_per_condition == 15_400_000
    assert train.formatters == ["integer-code"]
    assert train.disjoint_from == ["test"]
    assert len(loaded.spec.generation.conditions) == 4
    padded, long, training, padded_long = loaded.spec.generation.conditions
    assert padded.splits == ["test"]
    assert padded.formatters is None
    assert padded.filler.proportions == [0.0, 0.25, 0.5, 0.75, 0.9]
    assert long.splits == ["test"]
    assert isinstance(long.extra_assignments, LogSpacedAssignmentSchedule)
    assert long.extra_assignments.max == 64_000
    assert long.extra_assignments.sample_decay == 1.0
    assert long.extra_assignments.jitter_fraction == 0.0
    assert long.extra_assignments.min_samples_per_bucket == 20
    buckets = log_spaced_buckets(long.extra_assignments)
    assert len(buckets) == 17
    assert set(allocate_bucket_counts(long.extra_assignments, total_samples=340).values()) == {20}
    assert training.splits == ["train"]
    assert training.formatters == ["integer-code"]
    assert isinstance(training.extra_assignments, LogSpacedAssignmentSchedule)
    assert training.extra_assignments.min == 1
    assert training.extra_assignments.max == 256
    assert training.extra_assignments.growth_factor == 2.0
    assert training.extra_assignments.sample_decay == 1.0
    assert training.extra_assignments.jitter_fraction == 0.0
    assert training.extra_assignments.min_samples_per_bucket == 1
    assert training.filler.proportions == [0.0, 0.25]
    assert training.filler.distraction_levels == ["easy"]
    assert padded_long.splits == ["test"]
    assert padded_long.formatters == ["integer-code"]
    assert isinstance(padded_long.extra_assignments, LogSpacedAssignmentSchedule)
    assert padded_long.extra_assignments.max == 64_000
    assert padded_long.filler.proportions == [0.25]
    assert padded_long.filler.distraction_levels == ["easy", "medium", "hard"]
    assert padded_long.filler.stratify_by_assignment_bucket
    token_strata = loaded.spec.huggingface.token_strata
    assert token_strata is not None
    assert loaded.spec.huggingface.staging_workers == 2
    assert token_strata.mode == "exact"
    assert token_strata.batch_size == 64
    assert token_strata.max_batch_characters == 16_000_000
    assert loaded.spec.output.target_shard_bytes == 128_000_000
    assert [item.max_tokens for item in token_strata.ranges] == [
        2_048,
        6_144,
        12_288,
        24_576,
        49_152,
        98_304,
        196_608,
        393_216,
        786_432,
        1_572_864,
        None,
    ]
    assert next(item for item in token_strata.ranges if item.name == "tokens_128k").min_tokens == 98_304


def test_train_profile_experiment_matches_long_context_train_settings() -> None:
    production = load_experiment(ROOT / "configs/experiments/long-context.yaml").spec
    profile = load_experiment(
        ROOT / "configs/experiments/long-context-train-profile.yaml"
    ).spec
    production_train = next(
        condition
        for condition in production.generation.conditions
        if condition.splits == ["train"]
    )

    assert profile.splits["train"].samples_per_condition == (
        production.splits["train"].samples_per_condition
    )
    assert profile.generation.model_copy(update={"conditions": []}) == (
        production.generation.model_copy(update={"conditions": []})
    )
    assert profile.generation.conditions == [production_train]
    assert profile.output == production.output


def test_filler_plan_preserves_budget_and_evenly_covers_axis() -> None:
    spec = FillerSpec(
        proportions=[0.0, 0.9],
        distraction_levels=["easy", "medium", "hard"],
    )
    first = build_filler_plan(spec, total_samples=10, rng=random.Random(7))
    second = build_filler_plan(spec, total_samples=10, rng=random.Random(7))
    assert first == second
    assert len(first) == 10
    counts = Counter((item.proportion, item.distraction) for item in first)
    assert set(counts) == {
        (0.0, None),
        (0.9, "easy"),
        (0.9, "medium"),
        (0.9, "hard"),
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def test_filler_plan_can_cover_every_variant_within_each_assignment_bucket() -> None:
    assignment_plan = build_assignment_plan(
        _schedule(max=8, min_samples_per_bucket=6),
        total_samples=24,
        rng=random.Random(3),
    )
    filler = FillerSpec(
        proportions=[0.25],
        distraction_levels=["easy", "medium", "hard"],
        stratify_by_assignment_bucket=True,
    )
    filler_plan = build_stratified_filler_plan(
        filler,
        assignment_plan=assignment_plan,
        rng=random.Random(4),
    )

    by_bucket: dict[int, Counter[tuple[float, str | None]]] = {}
    for assignment, padding in zip(assignment_plan, filler_plan, strict=True):
        assert assignment.target_extra_assignments is not None
        counts = by_bucket.setdefault(assignment.target_extra_assignments, Counter())
        counts[(padding.proportion, padding.distraction)] += 1

    assert set(by_bucket) == {1, 2, 4, 8}
    for counts in by_bucket.values():
        assert set(counts) == {
            (0.25, "easy"),
            (0.25, "medium"),
            (0.25, "hard"),
        }
        assert max(counts.values()) - min(counts.values()) <= 1
