"""Focused token-length histograms for rendered state-bench datasets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import batched
from pathlib import Path
from typing import Any, Literal

from state_bench.stats import (
    DEFAULT_TOKENIZER,
    TokenizerLike,
    iter_rendered_records,
    load_tokenizer,
    token_lengths,
)

TextField = Literal["prefix", "completion", "prefix+completion"]
Binning = Literal["log2", "linear"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report a compact histogram of rendered-example token lengths."
    )
    parser.add_argument("dataset", type=Path, help="Rendered JSONL/Parquet file or directory.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--field",
        choices=("prefix", "completion", "prefix+completion"),
        default="prefix",
        help="Text to tokenize. Defaults to prompt prefixes only.",
    )
    parser.add_argument(
        "--joiner",
        default="",
        help="Separator between prefix and completion for --field prefix+completion.",
    )
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tokenizer batch size. The conservative default supports million-token prompts.",
    )
    parser.add_argument(
        "--binning",
        choices=("log2", "linear"),
        default="log2",
        help="Log2 bins are readable from short prompts through million-token prompts.",
    )
    parser.add_argument("--bin-width", type=int, default=10_000, help="Width of linear bins.")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        metavar="FIELD",
        help="Report separate histograms for fields such as formatter complexity.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer, revision=args.revision)
    report = analyze_token_lengths(
        iter_rendered_records(args.dataset, limit=args.limit),
        tokenizer=tokenizer,
        dataset=str(args.dataset),
        field=args.field,
        joiner=args.joiner,
        add_special_tokens=args.add_special_tokens,
        batch_size=args.batch_size,
        binning=args.binning,
        bin_width=args.bin_width,
        group_by=tuple(args.group_by),
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_histogram_report(report))
    return 0


def analyze_token_lengths(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: TokenizerLike,
    dataset: str = "<records>",
    field: TextField = "prefix",
    joiner: str = "",
    add_special_tokens: bool = False,
    batch_size: int = 1,
    binning: Binning = "log2",
    bin_width: int = 10_000,
    group_by: tuple[str, ...] = (),
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if bin_width < 1:
        raise ValueError("bin_width must be positive")

    lengths_by_group: dict[tuple[str, ...], list[int]] = defaultdict(list)
    all_lengths: list[int] = []
    for record_batch in batched(records, batch_size, strict=False):
        batch = list(record_batch)
        for index, record in enumerate(batch):
            missing = {"prefix", "completion", *group_by} - record.keys()
            if missing:
                raise ValueError(f"record {index} is missing fields: {sorted(missing)}")
        texts = [_record_text(record, field=field, joiner=joiner) for record in batch]
        lengths = token_lengths(tokenizer, texts, add_special_tokens)
        all_lengths.extend(lengths)
        for record, length in zip(batch, lengths, strict=True):
            key = tuple(str(record[name]) for name in group_by)
            lengths_by_group[key].append(length)

    if not all_lengths:
        raise ValueError(f"no rendered records found in {dataset}")

    histograms = {
        "all": summarize_token_lengths(all_lengths, binning=binning, bin_width=bin_width)
    }
    histograms.update(
        {
            _format_group_key(group_by, key): summarize_token_lengths(
                lengths,
                binning=binning,
                bin_width=bin_width,
            )
            for key, lengths in sorted(lengths_by_group.items())
        }
        if group_by
        else {}
    )
    return {
        "dataset": dataset,
        "tokenizer": tokenizer.name_or_path,
        "field": field,
        "joiner": joiner,
        "add_special_tokens": add_special_tokens,
        "binning": binning,
        "bin_width": bin_width if binning == "linear" else None,
        "rows": len(all_lengths),
        "group_by": list(group_by),
        "histograms": histograms,
    }


def summarize_token_lengths(
    lengths: Sequence[int],
    *,
    binning: Binning,
    bin_width: int,
) -> dict[str, Any]:
    ordered = sorted(lengths)
    counts = Counter(_bucket_bounds(length, binning=binning, bin_width=bin_width) for length in lengths)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "histogram": {
            _format_bucket(lower, upper): counts[(lower, upper)]
            for lower, upper in sorted(counts)
        },
    }


def format_histogram_report(report: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {report['dataset']}",
        f"Tokenizer: {report['tokenizer']}",
        f"Field: {report['field']}",
        f"Rows: {report['rows']}",
        f"Binning: {report['binning']}",
    ]
    for group, summary in report["histograms"].items():
        lines.extend(("", f"Histogram: {group}"))
        lines.append(
            "  "
            f"count={summary['count']} mean={summary['mean']} min={summary['min']} "
            f"p50={summary['p50']} p90={summary['p90']} p95={summary['p95']} "
            f"p99={summary['p99']} max={summary['max']}"
        )
        maximum = max(summary["histogram"].values())
        for bucket, count in summary["histogram"].items():
            bar_length = max(1, round(30 * count / maximum))
            lines.append(f"  {bucket:>17}: {count:>7} {'#' * bar_length}")
    return "\n".join(lines)


def _record_text(record: dict[str, Any], *, field: TextField, joiner: str) -> str:
    if field == "prefix":
        return str(record["prefix"])
    if field == "completion":
        return str(record["completion"])
    return str(record["prefix"]) + joiner + str(record["completion"])


def _bucket_bounds(length: int, *, binning: Binning, bin_width: int) -> tuple[int, int]:
    if binning == "linear":
        lower = (length // bin_width) * bin_width
        return lower, lower + bin_width - 1
    if length == 0:
        return 0, 0
    lower = 1 << (length.bit_length() - 1)
    return lower, 2 * lower - 1


def _format_bucket(lower: int, upper: int) -> str:
    return f"{lower:,}" if lower == upper else f"{lower:,}-{upper:,}"


def _percentile(ordered: Sequence[int], quantile: float) -> float | int:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 2)


def _format_group_key(fields: tuple[str, ...], values: tuple[str, ...]) -> str:
    return ", ".join(f"{field}={value}" for field, value in zip(fields, values, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
