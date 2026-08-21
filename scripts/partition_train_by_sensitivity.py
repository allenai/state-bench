#!/usr/bin/env python3
"""Partition a rendered train split into sensitivity-ordered bucket datasets.

Reads a rendered ``train.jsonl`` (minimal rendered records carry per-document
sensitivity fields alongside the Dolma ``id``/``text``/``source`` fields), sorts the
documents by a sensitivity field, and cuts the sorted order into contiguous buckets
of approximately equal byte mass (a proxy for token mass, so equal per-bucket token
budgets downstream preserve the marginal sensitivity distribution).

Bucket ``k`` is written to ``<output>/<config>--sens-<k>/train.jsonl``, where
``<config>`` is the name of the rendered config directory containing the input
(e.g. ``integer-code--periodic``). That layout is exactly what
``scripts/tokenize_dataset.py`` expects, so the buckets tokenize with:

    uv run scripts/tokenize_dataset.py <output>

Buckets are ordered by ascending sensitivity: ``--sens-00`` holds the least
sensitive (most aperiodic-like) documents. Within a bucket, documents keep their
original line order; training shuffles within buckets, so only the bucket
assignment matters. Input lines are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

DEFAULT_BUCKETS = 16
DEFAULT_FIELD = "sensitivity_coverage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_jsonl",
        type=Path,
        help="Rendered train split to partition, e.g. rendered/integer-code--periodic/train.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory that will contain one <config>--sens-NN/train.jsonl per bucket.",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=DEFAULT_BUCKETS,
        help=f"Number of sensitivity buckets (default: {DEFAULT_BUCKETS}).",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=f"Numeric record field to order by (default: {DEFAULT_FIELD}).",
    )
    return parser.parse_args()


def read_sort_values(source: Path, field: str) -> tuple[list[float], list[int]]:
    """Return each document's sort value and byte length, in line order."""
    values: list[float] = []
    lengths: list[int] = []
    with (
        source.open("rb") as input_file,
        tqdm(
            total=source.stat().st_size,
            desc=f"Scanning {source.parent.name}/{source.name}",
            unit="B",
            unit_scale=True,
            mininterval=0.5,
            disable=not sys.stderr.isatty(),
        ) as progress,
    ):
        for line_number, line in enumerate(input_file, start=1):
            progress.update(len(line))
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank lines cannot be partitioned")
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            value = record.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{source}:{line_number}: field {field!r} is missing or not numeric "
                    f"(got {value!r}); pass --field for a different sensitivity field"
                )
            values.append(float(value))
            lengths.append(len(line))
    return values, lengths


def assign_buckets(values: list[float], lengths: list[int], buckets: int) -> list[int]:
    """Map each document (in line order) to a bucket of ascending sensitivity.

    Documents are ordered by (value, line index) — a deterministic tie-break — and the
    sorted order is cut into ``buckets`` contiguous runs of approximately equal bytes.
    """
    if buckets < 1:
        raise ValueError("--buckets must be positive")
    if len(values) < buckets:
        raise ValueError(f"cannot cut {len(values)} documents into {buckets} buckets")
    sorted_order = sorted(range(len(values)), key=values.__getitem__)
    total_bytes = sum(lengths)
    bucket_of_line = [0] * len(values)
    consumed_bytes = 0
    for line_index in sorted_order:
        # Cut on the byte mass consumed *before* each document so bucket 0 is never
        # skipped and the final documents land in bucket `buckets - 1`.
        bucket_of_line[line_index] = min(buckets - 1, consumed_bytes * buckets // total_bytes)
        consumed_bytes += lengths[line_index]
    return bucket_of_line


def write_buckets(
    source: Path,
    output: Path,
    config_name: str,
    bucket_of_line: list[int],
    buckets: int,
) -> list[Path]:
    """Stream the input once, appending each line to its bucket's train.jsonl."""
    bucket_paths = [output / f"{config_name}--sens-{bucket:02d}" / "train.jsonl" for bucket in range(buckets)]
    for path in bucket_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("wb") for path in bucket_paths]
    try:
        with (
            source.open("rb") as input_file,
            tqdm(
                total=source.stat().st_size,
                desc=f"Writing {buckets} buckets",
                unit="B",
                unit_scale=True,
                mininterval=0.5,
                disable=not sys.stderr.isatty(),
            ) as progress,
        ):
            for line_index, line in enumerate(input_file):
                progress.update(len(line))
                handles[bucket_of_line[line_index]].write(line)
    finally:
        for handle in handles:
            handle.close()
    return bucket_paths


def bucket_manifest(
    values: list[float],
    lengths: list[int],
    bucket_of_line: list[int],
    buckets: int,
    field: str,
    source: Path,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = [
        {"documents": 0, "bytes": 0, "min_value": None, "max_value": None} for _ in range(buckets)
    ]
    for value, length, bucket in zip(values, lengths, bucket_of_line, strict=True):
        summary = summaries[bucket]
        summary["documents"] += 1
        summary["bytes"] += length
        if summary["min_value"] is None or value < summary["min_value"]:
            summary["min_value"] = value
        if summary["max_value"] is None or value > summary["max_value"]:
            summary["max_value"] = value
    return {
        "source": str(source),
        "field": field,
        "buckets": summaries,
    }


def main() -> int:
    args = parse_args()
    source = args.train_jsonl.resolve()
    if not source.is_file():
        raise SystemExit(f"train split does not exist: {source}")
    config_name = source.parent.name
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}\nChoose --output or remove it first.")

    values, lengths = read_sort_values(source, args.field)
    bucket_of_line = assign_buckets(values, lengths, args.buckets)
    output.mkdir(parents=True)
    write_buckets(source, output, config_name, bucket_of_line, args.buckets)
    manifest = bucket_manifest(values, lengths, bucket_of_line, args.buckets, args.field, source)
    manifest_path = output / "sensitivity-buckets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for bucket, summary in enumerate(manifest["buckets"]):
        print(
            f"sens-{bucket:02d}: {summary['documents']:,} documents, {summary['bytes']:,} bytes, "
            f"{args.field} in [{summary['min_value']:.6g}, {summary['max_value']:.6g}]"
        )
    print(f"Partitioned {len(values):,} documents into {args.buckets} buckets under {output}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
