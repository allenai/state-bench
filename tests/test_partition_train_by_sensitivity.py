"""Tests for scripts/partition_train_by_sensitivity.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "partition_train_by_sensitivity.py"
_spec = importlib.util.spec_from_file_location("partition_train_by_sensitivity", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
partition = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(partition)


def _write_train_jsonl(path: Path, coverages: list[float]) -> list[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, coverage in enumerate(coverages):
        record = {
            "example_id": f"example-{index}",
            "sensitivity_coverage": coverage,
            "id": f"example-{index}",
            "text": f"document {index} " + "x" * (index % 7),
            "source": "state-bench",
        }
        lines.append((json.dumps(record, separators=(",", ":")) + "\n").encode())
    path.write_bytes(b"".join(lines))
    return lines


def test_partition_is_exhaustive_and_sensitivity_ordered(tmp_path: Path) -> None:
    train_path = tmp_path / "rendered" / "integer-code--periodic" / "train.jsonl"
    # Interleave low/high coverages so bucket assignment must reorder lines.
    coverages = [(index * 37 % 100) / 100 for index in range(200)]
    lines = _write_train_jsonl(train_path, coverages)
    output = tmp_path / "rendered-sensitivity"

    argv = [
        str(_SCRIPT_PATH),
        str(train_path),
        "--output",
        str(output),
        "--buckets",
        "4",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        assert partition.main() == 0
    finally:
        sys.argv = old_argv

    bucket_paths = sorted(output.glob("integer-code--periodic--sens-*/train.jsonl"))
    assert len(bucket_paths) == 4

    seen_lines: list[bytes] = []
    previous_max = float("-inf")
    for bucket_path in bucket_paths:
        bucket_lines = bucket_path.read_bytes().splitlines(keepends=True)
        assert bucket_lines, f"{bucket_path} is empty"
        seen_lines.extend(bucket_lines)
        bucket_coverages = [json.loads(line)["sensitivity_coverage"] for line in bucket_lines]
        # Buckets are contiguous runs of the sensitivity-sorted order.
        assert min(bucket_coverages) >= previous_max
        previous_max = max(bucket_coverages)

    # Every input line lands in exactly one bucket, byte-for-byte.
    assert sorted(seen_lines) == sorted(lines)

    manifest = json.loads((output / "sensitivity-buckets.json").read_text())
    assert manifest["field"] == "sensitivity_coverage"
    assert sum(summary["documents"] for summary in manifest["buckets"]) == len(lines)
    assert sum(summary["bytes"] for summary in manifest["buckets"]) == sum(len(line) for line in lines)


def test_assign_buckets_balances_bytes() -> None:
    values = [index / 1000 for index in range(1000)]
    lengths = [100] * 1000
    bucket_of_line = partition.assign_buckets(values, lengths, 8)
    counts = [bucket_of_line.count(bucket) for bucket in range(8)]
    assert counts == [125] * 8
    # Ascending values in line order means bucket ids are non-decreasing.
    assert bucket_of_line == sorted(bucket_of_line)


def test_assign_buckets_breaks_ties_by_line_order() -> None:
    bucket_of_line = partition.assign_buckets([0.5] * 10, [10] * 10, 2)
    assert bucket_of_line == [0] * 5 + [1] * 5


def test_missing_field_raises(tmp_path: Path) -> None:
    train_path = tmp_path / "rendered" / "integer-code--periodic" / "train.jsonl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b'{"id":"a","text":"t","source":"state-bench"}\n')
    with pytest.raises(ValueError, match="sensitivity_coverage"):
        partition.read_sort_values(train_path, "sensitivity_coverage")
