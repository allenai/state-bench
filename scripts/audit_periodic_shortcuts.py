#!/usr/bin/env python3
"""Audit periodic canonical records for shallow multiple-choice shortcuts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from state_bench.distractors import generate_incorrect_states
from state_bench.io import derive_seed
from state_bench.model import AssignVariable, Complexity, StateTrackingInstance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical JSONL containing periodic rows.")
    parser.add_argument("--seed", type=int, required=True, help="Dataset seed used for distractors.")
    return parser.parse_args(argv)


def _initial_values(record: dict[str, Any]) -> tuple[int, ...]:
    values: list[int | None] = [None] * int(record["num_variables"])
    for operation in record["operations"]:
        if values[operation["target"]] is None:
            if operation["op"] != "constant":
                raise ValueError("every periodic variable must begin with a literal")
            values[operation["target"]] = int(operation["value"])
    return tuple(value for value in values if value is not None)


def _expected_choice_accuracy(scores: list[int], *, prefer: str) -> tuple[float, int]:
    best = max(scores) if prefer == "max" else min(scores)
    ties = sum(score == best for score in scores)
    return 1 / ties, ties


def audit_records(records: Iterable[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    ties: dict[str, Counter[int]] = defaultdict(Counter)
    widths: Counter[int] = Counter()
    multiplicities: Counter[tuple[int, ...]] = Counter()
    by_extra_assignments: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_extra_rows: Counter[int] = Counter()

    for record in records:
        instance = StateTrackingInstance.from_record(record)
        if instance.complexity is not Complexity.PERIODIC:
            continue
        widths[instance.num_variables] += 1
        initial = _initial_values(record)
        correct = tuple(instance.final_state[variable] for variable in instance.query)
        incorrect_states = generate_incorrect_states(
            instance,
            num_distractors=4,
            rng=random.Random(derive_seed(seed, "distractors", str(record["example_id"]))),
        )
        candidates = [
            correct,
            *(tuple(state[variable] for variable in instance.query) for state in incorrect_states),
        ]
        initial_set = set(initial)
        set_scores = [len(set(candidate) & initial_set) for candidate in candidates]
        position_scores = [
            sum(
                value == initial[variable]
                for variable, value in zip(instance.query, candidate, strict=True)
            )
            for candidate in candidates
        ]
        character_scores = [sum(len(str(value)) for value in candidate) for candidate in candidates]
        last_sources: list[int | None] = [None] * instance.num_variables
        for operation in instance.operations:
            if isinstance(operation, AssignVariable):
                last_sources[operation.target] = operation.source
        one_hop = tuple(
            initial[last_sources[variable]] if last_sources[variable] is not None else initial[variable]
            for variable in instance.query
        )
        one_hop_scores = [
            sum(value == prediction for value, prediction in zip(candidate, one_hop, strict=True))
            for candidate in candidates
        ]
        extra_assignments = int(record.get("target_extra_assignments") or 0)
        if not extra_assignments:
            extra_assignments = len(instance.operations) - instance.num_variables
        for name, scores, prefer in (
            ("initial_value_set_overlap", set_scores, "max"),
            ("fewest_initial_position_matches", position_scores, "min"),
            ("answer_value_character_length", character_scores, "max"),
            ("last_copy_source_initial_value", one_hop_scores, "max"),
        ):
            accuracy, tie_count = _expected_choice_accuracy(scores, prefer=prefer)
            totals[name] += accuracy
            ties[name][tie_count] += 1
            by_extra_assignments[extra_assignments][name] += accuracy
        by_extra_assignments[extra_assignments]["one_hop_coordinate_accuracy"] += (
            sum(value == prediction for value, prediction in zip(correct, one_hop, strict=True))
            / len(correct)
        )
        by_extra_rows[extra_assignments] += 1
        multiplicities[tuple(sorted(Counter(correct).values()))] += 1
        totals["rows"] += 1

    rows = int(totals["rows"])
    if not rows:
        raise ValueError("no periodic rows found")
    return {
        "periodic_rows": rows,
        "widths": dict(sorted(widths.items())),
        "candidate_baseline_expected_accuracy": {
            name: total / rows for name, total in sorted(totals.items()) if name != "rows"
        },
        "candidate_baseline_tie_sizes": {
            name: dict(sorted(counts.items())) for name, counts in sorted(ties.items())
        },
        "final_value_multiplicities": {
            ",".join(map(str, pattern)): count for pattern, count in sorted(multiplicities.items())
        },
        "by_extra_assignments": {
            str(extra): {
                metric: total / by_extra_rows[extra]
                for metric, total in sorted(metrics.items())
            }
            for extra, metrics in sorted(by_extra_assignments.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    def periodic_records() -> Iterable[dict[str, Any]]:
        with args.input.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["complexity"] == "periodic":
                    yield record

    print(json.dumps(audit_records(periodic_records(), seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
