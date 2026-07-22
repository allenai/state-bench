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


def test_too_small_sample_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="need at least"):
        allocate_bucket_counts(_schedule(), total_samples=10)


def test_long_context_experiment_config_loads() -> None:
    loaded = load_experiment(ROOT / "configs/experiments/long-context.yaml")
    assert loaded.spec.dataset.name == "state-tracking-long-context-v1"
    assert loaded.spec.splits["test"].samples_per_condition == 380
    assert len(loaded.spec.generation.conditions) == 2
    padded, long = loaded.spec.generation.conditions
    assert padded.formatters is None
    assert max(padded.filler.proportions) == 0.99995
    assert isinstance(long.extra_assignments, LogSpacedAssignmentSchedule)
    assert long.extra_assignments.max == 192_000
    assert long.extra_assignments.sample_decay == 1.0
    assert long.extra_assignments.jitter_fraction == 0.0
    assert long.extra_assignments.min_samples_per_bucket == 20
    token_strata = loaded.spec.huggingface.token_strata
    assert token_strata is not None
    assert loaded.spec.huggingface.staging_workers == 2
    assert token_strata.mode == "sampled"
    assert token_strata.batch_size == 4
    assert token_strata.max_batch_characters == 16_000_000
    assert token_strata.calibration_scan_rows == 128
    assert token_strata.calibration_samples_per_group == 16
    assert loaded.spec.output.target_shard_bytes == 128_000_000
    assert [item.max_tokens for item in token_strata.ranges] == [
        100_000,
        500_000,
        1_000_000,
        None,
    ]


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
