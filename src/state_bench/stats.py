"""Summary statistics for rendered state-bench datasets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_TOKENIZER = "allenai/Olmo-3-1025-7B"
DEFAULT_GROUP_FIELDS = (
    "formatter",
    "complexity",
    "num_variables",
    "target_extra_assignments",
)


class TokenizerLike(Protocol):
    name_or_path: str

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, Any]: ...


class _GigatokenAdapter:
    """Use gigatoken for exact lengths while retaining the HF tokenizer API."""

    def __init__(self, tokenizer: Any) -> None:
        import gigatoken

        self.name_or_path = str(tokenizer.name_or_path)
        self._fallback = tokenizer
        self._accelerated = gigatoken.Tokenizer(tokenizer)

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._fallback(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
            ),
        )

    def token_lengths(self, texts: list[str], *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            encodings = self._fallback.backend_tokenizer.encode_batch(
                texts,
                add_special_tokens=True,
            )
            return [len(encoding) for encoding in encodings]

        import awkward as ak

        encoded = self._accelerated.encode_batch(texts)
        return [int(length) for length in ak.to_list(ak.num(encoded, axis=1))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize rendered dataset fields and tokenizer-dependent lengths."
    )
    parser.add_argument("dataset", type=Path, help="Rendered JSONL/Parquet file or directory.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument(
        "--joiner",
        default="",
        help="Text inserted between prefix and completion before combined tokenization.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--histogram-bin-width", type=int, default=10)
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        metavar="FIELD",
        help="Also report token distributions for groups such as formatter complexity.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_path(
        args.dataset,
        tokenizer_name=args.tokenizer,
        revision=args.revision,
        add_special_tokens=args.add_special_tokens,
        joiner=args.joiner,
        limit=args.limit,
        batch_size=args.batch_size,
        histogram_bin_width=args.histogram_bin_width,
        group_by=tuple(args.group_by),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


def analyze_path(
    path: Path,
    *,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    revision: str | None = None,
    add_special_tokens: bool = False,
    joiner: str = "",
    limit: int | None = None,
    batch_size: int = 1024,
    histogram_bin_width: int = 10,
    group_by: tuple[str, ...] = (),
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_name, revision=revision)
    return analyze_records(
        iter_rendered_records(path, limit=limit),
        tokenizer=tokenizer,
        dataset=str(path),
        add_special_tokens=add_special_tokens,
        joiner=joiner,
        batch_size=batch_size,
        histogram_bin_width=histogram_bin_width,
        group_by=group_by,
    )


def analyze_records(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: TokenizerLike,
    dataset: str = "<records>",
    add_special_tokens: bool = False,
    joiner: str = "",
    batch_size: int = 1024,
    histogram_bin_width: int = 10,
    group_by: tuple[str, ...] = (),
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if histogram_bin_width < 1:
        raise ValueError("histogram_bin_width must be positive")

    metrics: dict[str, list[int]] = defaultdict(list)
    grouped_metrics: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    field_counts = {field: Counter() for field in DEFAULT_GROUP_FIELDS}
    numeric_fields: dict[str, list[int]] = {
        "num_assignments": [],
        "num_extra_assignments": [],
        "target_extra_assignments": [],
    }
    row_count = 0

    for record_batch in batched(records, batch_size, strict=False):
        batch = list(record_batch)
        if not batch:
            continue
        _validate_rendered_batch(batch, group_by)
        prefixes = [str(record["prefix"]) for record in batch]
        completions = [str(record["completion"]) for record in batch]
        combined = [
            prefix + joiner + completion
            for prefix, completion in zip(prefixes, completions, strict=True)
        ]
        prefix_lengths = token_lengths(tokenizer, prefixes, add_special_tokens)
        completion_lengths = token_lengths(tokenizer, completions, add_special_tokens)
        combined_lengths = token_lengths(tokenizer, combined, add_special_tokens)

        choices: list[str] = []
        choice_owners: list[int] = []
        for row_index, record in enumerate(batch):
            for choice in record.get("choices", []):
                choices.append(str(choice))
                choice_owners.append(row_index)
        choice_lengths = token_lengths(tokenizer, choices, add_special_tokens) if choices else []
        prompt_choice_lengths = (
            token_lengths(
                tokenizer,
                [
                    prefixes[owner] + joiner + choice
                    for owner, choice in zip(choice_owners, choices, strict=True)
                ],
                add_special_tokens,
            )
            if choices
            else []
        )

        for index, record in enumerate(batch):
            row_count += 1
            row_metrics = {
                "prefix_tokens": prefix_lengths[index],
                "completion_tokens": completion_lengths[index],
                "prefix_plus_completion_tokens": combined_lengths[index],
            }
            for name, value in row_metrics.items():
                metrics[name].append(value)
            group_key = tuple(str(record[field]) for field in group_by)
            for name, value in row_metrics.items():
                grouped_metrics[group_key][name].append(value)
            for field in DEFAULT_GROUP_FIELDS:
                if field in record and record[field] is not None:
                    field_counts[field][str(record[field])] += 1
            for field in numeric_fields:
                if field in record and record[field] is not None:
                    numeric_fields[field].append(int(record[field]))

        for length in choice_lengths:
            metrics["choice_tokens"].append(length)
        for length in prompt_choice_lengths:
            metrics["prefix_plus_choice_tokens"].append(length)
        for choice_index, owner in enumerate(choice_owners):
            group_key = tuple(str(batch[owner][field]) for field in group_by)
            grouped_metrics[group_key]["choice_tokens"].append(choice_lengths[choice_index])
            grouped_metrics[group_key]["prefix_plus_choice_tokens"].append(
                prompt_choice_lengths[choice_index]
            )

    if row_count == 0:
        raise ValueError(f"no rendered records found in {dataset}")

    return {
        "dataset": dataset,
        "tokenizer": tokenizer.name_or_path,
        "add_special_tokens": add_special_tokens,
        "joiner": joiner,
        "rows": row_count,
        "field_counts": {
            field: dict(sorted(counts.items())) for field, counts in field_counts.items() if counts
        },
        "assignment_distributions": {
            field: summarize_distribution(values, histogram_bin_width=histogram_bin_width)
            for field, values in numeric_fields.items()
            if values
        },
        "token_distributions": {
            name: summarize_distribution(values, histogram_bin_width=histogram_bin_width)
            for name, values in sorted(metrics.items())
        },
        "group_by": list(group_by),
        "groups": {
            _format_group_key(group_by, key): {
                name: summarize_distribution(values, histogram_bin_width=histogram_bin_width)
                for name, values in sorted(group_metrics.items())
            }
            for key, group_metrics in sorted(grouped_metrics.items())
        }
        if group_by
        else {},
    }


def summarize_distribution(
    values: Sequence[int],
    *,
    histogram_bin_width: int,
) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    histogram = Counter((value // histogram_bin_width) * histogram_bin_width for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "p50": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "stdev": round(statistics.pstdev(ordered), 2),
        "histogram": {
            f"{lower}-{lower + histogram_bin_width - 1}": histogram[lower]
            for lower in sorted(histogram)
        },
    }


def iter_rendered_records(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    files = _discover_files(path)
    emitted = 0
    for file_path in files:
        for record in _read_file(file_path):
            if "prefix" not in record or "completion" not in record:
                continue
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield record


def load_tokenizer(name_or_path: str, *, revision: str | None = None) -> TokenizerLike:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("token statistics require: uv sync --extra analysis") from error
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision, use_fast=True)
    if tokenizer is None:
        raise RuntimeError(f"failed to load tokenizer: {name_or_path}")
    try:
        return _GigatokenAdapter(tokenizer)
    except (ImportError, NotImplementedError, ValueError):
        return cast("TokenizerLike", tokenizer)


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {report['dataset']}",
        f"Tokenizer: {report['tokenizer']}",
        f"Rows: {report['rows']}",
        f"Add special tokens: {report['add_special_tokens']}",
        f"Joiner: {report['joiner']!r}",
    ]
    for field, counts in report["field_counts"].items():
        lines.extend(("", f"{field} distribution:"))
        lines.extend(f"  {value}: {count}" for value, count in counts.items())
    for name, distribution in report["assignment_distributions"].items():
        lines.extend(("", f"{name} distribution:"))
        lines.extend(_format_distribution(distribution))
    for name, distribution in report["token_distributions"].items():
        lines.extend(("", f"{name} distribution:"))
        lines.extend(_format_distribution(distribution))
    for group, distributions in report["groups"].items():
        lines.extend(("", f"Group: {group}"))
        for name, distribution in distributions.items():
            lines.append(f"  {name}: {_compact_distribution(distribution)}")
    return "\n".join(lines)


def token_lengths(
    tokenizer: TokenizerLike,
    texts: list[str],
    add_special_tokens: bool,
) -> list[int]:
    if not texts:
        return []
    accelerated = getattr(tokenizer, "token_lengths", None)
    if callable(accelerated):
        return cast(
            "list[int]",
            accelerated(texts, add_special_tokens=add_special_tokens),
        )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        encodings = backend.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [len(encoding) for encoding in encodings]
    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
    )
    return [len(token_ids) for token_ids in encoded["input_ids"]]
def _validate_rendered_batch(batch: list[dict[str, Any]], group_by: tuple[str, ...]) -> None:
    for index, record in enumerate(batch):
        missing = {"prefix", "completion"} - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing rendered fields: {sorted(missing)}")
        missing_groups = set(group_by) - record.keys()
        if missing_groups:
            raise ValueError(f"record {index} is missing group fields: {sorted(missing_groups)}")


def _discover_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in {".jsonl", ".parquet"}:
            raise ValueError(f"unsupported dataset file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    jsonl_files = sorted(path.rglob("*.jsonl"))
    parquet_files = sorted(path.rglob("*.parquet"))
    if jsonl_files and parquet_files:
        raise ValueError("dataset directory mixes JSONL and Parquet; select a narrower directory")
    files = jsonl_files or parquet_files
    if not files:
        raise FileNotFoundError(f"no JSONL or Parquet files found under {path}")
    return files


def _read_file(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise RuntimeError("Parquet statistics require: uv sync --extra analysis") from error
    yield from parquet.read_table(path).to_pylist()


def _percentile(ordered: Sequence[int], quantile: float) -> float | int:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 2)


def _format_distribution(distribution: dict[str, Any]) -> list[str]:
    if distribution["count"] == 0:
        return ["  count: 0"]
    lines = [f"  {_compact_distribution(distribution)}", "  histogram:"]
    maximum = max(distribution["histogram"].values())
    for bucket, count in distribution["histogram"].items():
        bar_length = max(1, round(30 * count / maximum))
        lines.append(f"    {bucket:>9}: {count:>7} {'#' * bar_length}")
    return lines


def _compact_distribution(distribution: dict[str, Any]) -> str:
    return (
        f"count={distribution['count']} mean={distribution['mean']} stdev={distribution['stdev']} "
        f"min={distribution['min']} p25={distribution['p25']} p50={distribution['p50']} "
        f"p75={distribution['p75']} p90={distribution['p90']} p95={distribution['p95']} "
        f"p99={distribution['p99']} max={distribution['max']}"
    )


def _format_group_key(fields: tuple[str, ...], values: tuple[str, ...]) -> str:
    return ", ".join(f"{field}={value}" for field, value in zip(fields, values, strict=True))
