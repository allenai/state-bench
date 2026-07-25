"""Allocate fixed sample budgets across assignment-depth schedules."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from state_bench.config import (
    AssignmentRange,
    ExtraAssignmentsSpec,
    FillerDistraction,
    FillerSpec,
    LogSpacedAssignmentSchedule,
    filler_variants,
)


@dataclass(frozen=True, slots=True)
class AssignmentPlanItem:
    min_extra_assignments: int
    max_extra_assignments: int
    target_extra_assignments: int | None


@dataclass(frozen=True, slots=True)
class FillerPlanItem:
    proportion: float
    distraction: FillerDistraction | None


def build_filler_plan(
    spec: FillerSpec,
    *,
    total_samples: int,
    rng: random.Random,
) -> tuple[FillerPlanItem, ...]:
    """Distribute a fixed sample budget as evenly as possible across filler variants."""

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    variants = filler_variants(spec)
    if 0 < total_samples < len(variants):
        raise ValueError(
            f"samples_per_condition={total_samples} cannot cover {len(variants)} filler variants"
        )
    quotient, remainder = divmod(total_samples, len(variants))
    plan = [
        FillerPlanItem(proportion, distraction)
        for index, (proportion, distraction) in enumerate(variants)
        for _ in range(quotient + (index < remainder))
    ]
    rng.shuffle(plan)
    return tuple(plan)


def iter_balanced_filler_plan(
    spec: FillerSpec,
    *,
    total_samples: int,
    rng: random.Random,
) -> Iterator[FillerPlanItem]:
    """Yield an exactly balanced filler schedule without an O(n) list.

    Each small cycle is shuffled independently. This preserves balanced counts
    and avoids a giant allocation for corpus-scale range conditions.
    """

    variants = filler_variants(spec)
    if 0 < total_samples < len(variants):
        raise ValueError(
            f"samples_per_condition={total_samples} cannot cover {len(variants)} filler variants"
        )
    quotient, remainder = divmod(total_samples, len(variants))
    for _ in range(quotient):
        cycle = list(variants)
        rng.shuffle(cycle)
        yield from (FillerPlanItem(proportion, distraction) for proportion, distraction in cycle)
    if remainder:
        cycle = list(variants)
        rng.shuffle(cycle)
        yield from (FillerPlanItem(proportion, distraction) for proportion, distraction in cycle[:remainder])


def build_stratified_filler_plan(
    spec: FillerSpec,
    *,
    assignment_plan: tuple[AssignmentPlanItem, ...],
    rng: random.Random,
) -> tuple[FillerPlanItem, ...]:
    """Evenly cover filler variants within every assignment-depth bucket."""

    indices_by_bucket: dict[int | None, list[int]] = {}
    for index, item in enumerate(assignment_plan):
        indices_by_bucket.setdefault(item.target_extra_assignments, []).append(index)

    result = [FillerPlanItem(0.0, None)] * len(assignment_plan)
    for indices in indices_by_bucket.values():
        bucket_plan = build_filler_plan(spec, total_samples=len(indices), rng=rng)
        for index, filler_item in zip(indices, bucket_plan, strict=True):
            result[index] = filler_item
    return tuple(result)


def sample_filler_plan_item(spec: FillerSpec, *, rng: random.Random) -> FillerPlanItem:
    proportion, distraction = rng.choice(filler_variants(spec))
    return FillerPlanItem(proportion, distraction)


def build_assignment_plan(
    spec: ExtraAssignmentsSpec,
    *,
    total_samples: int,
    rng: random.Random,
) -> tuple[AssignmentPlanItem, ...]:
    """Build a shuffled plan whose length exactly matches the requested budget."""

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if isinstance(spec, AssignmentRange):
        return tuple(
            AssignmentPlanItem(
                min_extra_assignments=spec.min,
                max_extra_assignments=spec.max,
                target_extra_assignments=None,
            )
            for _ in range(total_samples)
        )

    buckets = log_spaced_buckets(spec)
    counts = allocate_bucket_counts(spec, total_samples=total_samples)
    plan = [
        _bucket_item(spec, bucket)
        for bucket in buckets
        for _ in range(counts[bucket])
    ]
    rng.shuffle(plan)
    return tuple(plan)


def sample_assignment_plan_item(
    spec: ExtraAssignmentsSpec,
    *,
    rng: random.Random,
) -> AssignmentPlanItem:
    """Sample one preview item according to the configured depth weights."""

    if isinstance(spec, AssignmentRange):
        return AssignmentPlanItem(spec.min, spec.max, None)
    buckets = log_spaced_buckets(spec)
    bucket = rng.choices(
        buckets,
        weights=[spec.sample_decay**index for index in range(len(buckets))],
        k=1,
    )[0]
    return _bucket_item(spec, bucket)


def log_spaced_buckets(spec: LogSpacedAssignmentSchedule) -> tuple[int, ...]:
    buckets = [spec.min]
    while buckets[-1] < spec.max:
        next_bucket = max(buckets[-1] + 1, math.ceil(buckets[-1] * spec.growth_factor))
        if next_bucket >= spec.max:
            break
        buckets.append(next_bucket)
    if buckets[-1] != spec.max:
        buckets.append(spec.max)
    return tuple(buckets)


def allocate_bucket_counts(
    spec: LogSpacedAssignmentSchedule,
    *,
    total_samples: int,
) -> Counter[int]:
    buckets = log_spaced_buckets(spec)
    required = len(buckets) * spec.min_samples_per_bucket
    if total_samples < required:
        raise ValueError(
            f"samples_per_condition={total_samples} cannot give {len(buckets)} buckets at least "
            f"{spec.min_samples_per_bucket} samples each; need at least {required}"
        )

    remaining = total_samples - required
    weights = [spec.sample_decay**index for index in range(len(buckets))]
    weight_sum = sum(weights)
    quotas = [remaining * weight / weight_sum for weight in weights]
    allocated = [math.floor(quota) for quota in quotas]
    leftover = remaining - sum(allocated)
    order = sorted(
        range(len(buckets)),
        key=lambda index: (-(quotas[index] - allocated[index]), index),
    )
    for index in order[:leftover]:
        allocated[index] += 1
    return Counter(
        {
            bucket: spec.min_samples_per_bucket + allocated[index]
            for index, bucket in enumerate(buckets)
        }
    )


def _bucket_item(spec: LogSpacedAssignmentSchedule, bucket: int) -> AssignmentPlanItem:
    jitter = math.floor(bucket * spec.jitter_fraction)
    return AssignmentPlanItem(
        min_extra_assignments=max(spec.min, bucket - jitter),
        max_extra_assignments=min(spec.max, bucket + jitter),
        target_extra_assignments=bucket,
    )
