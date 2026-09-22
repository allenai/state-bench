"""Build, validate, and explicitly upload Hugging Face dataset repositories."""

from __future__ import annotations

import json
import shutil
import statistics
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from state_bench.config import LoadedExperiment, TokenStratificationSpec, TokenStratumSpec
from state_bench.io import sha256_file
from state_bench.stats import TokenizerLike, load_tokenizer, token_lengths


def stage_hub_repository(
    loaded: LoadedExperiment,
    *,
    source_dir: Path,
    repo_dir: Path,
    workers: int | None = None,
    only_splits: frozenset[str] | None = None,
) -> Path:
    """Create a complete local Hub repository without network access.

    ``only_splits`` restricts staging to those base splits (for example
    ``{"test"}``), leaving other generated splits such as a large ``train``
    out of the repository entirely.
    """

    if only_splits is not None and not only_splits:
        raise ValueError("only_splits must not be empty")

    # Freely overwrite any prior staging: clear the directory so stale shards or
    # metadata from an earlier run cannot linger alongside the new artifacts.
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    data_dir = repo_dir / "data"
    data_dir.mkdir()

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    # Hub-only settings can change without regenerating rendered JSONL. Record
    # the settings used for this staging run rather than the source build's copy.
    source_manifest["huggingface"] = loaded.spec.huggingface.model_dump(mode="json")
    include_canonical = loaded.spec.huggingface.include_canonical_config
    total_rows = sum(
        int(count)
        for name, count in source_manifest["counts"].items()
        if (include_canonical or not name.startswith("canonical/"))
        and (only_splits is None or name.rpartition("/")[2] in only_splits)
    )
    progress = tqdm(
        total=total_rows,
        desc="Staging Hub data",
        unit="row",
        dynamic_ncols=True,
        disable=None,
    )

    configs: list[dict[str, Any]] = []
    staging_workers = workers if workers is not None else loaded.spec.huggingface.staging_workers
    if staging_workers < 1:
        raise ValueError("staging workers must be positive")
    token_strata = loaded.spec.huggingface.token_strata
    tokenizer = (
        load_tokenizer(token_strata.tokenizer, revision=token_strata.revision)
        if token_strata is not None
        else None
    )
    try:
        if include_canonical:
            configs.append(
                _stage_config(
                    source_dir / "canonical",
                    data_dir,
                    "canonical",
                    shard_size=loaded.spec.output.shard_size,
                    target_shard_bytes=loaded.spec.output.target_shard_bytes,
                    progress=progress,
                    only_splits=only_splits,
                )
            )
        rendered_dirs = [
            config_dir for config_dir in sorted((source_dir / "rendered").iterdir()) if config_dir.is_dir()
        ]

        def stage_rendered_config(config_dir: Path) -> dict[str, Any]:
            return _stage_config(
                config_dir,
                data_dir,
                config_dir.name,
                shard_size=loaded.spec.output.shard_size,
                target_shard_bytes=loaded.spec.output.target_shard_bytes,
                token_strata=token_strata,
                tokenizer=tokenizer,
                progress=progress,
                only_splits=only_splits,
            )

        if staging_workers == 1:
            configs.extend(stage_rendered_config(config_dir) for config_dir in rendered_dirs)
        else:
            with ThreadPoolExecutor(max_workers=staging_workers) as executor:
                configs.extend(executor.map(stage_rendered_config, rendered_dirs))
    finally:
        progress.close()

    metadata: dict[str, Any] = {
        "pretty_name": loaded.spec.dataset.name,
        "language": loaded.spec.huggingface.card.language,
        "tags": loaded.spec.huggingface.card.tags,
        "configs": configs,
    }
    if loaded.spec.huggingface.card.license is not None:
        metadata["license"] = loaded.spec.huggingface.card.license
    readme = _dataset_card(metadata=metadata, loaded=loaded)
    (repo_dir / "README.md").write_text(readme, encoding="utf-8")
    (repo_dir / "generation.yaml").write_text(loaded.path.read_text(encoding="utf-8"), encoding="utf-8")
    formatter_dir = repo_dir / "formats"
    formatter_dir.mkdir()
    for reference, formatter in zip(loaded.spec.formatters, loaded.formatters, strict=True):
        source = (loaded.path.parent / reference.config).resolve()
        (formatter_dir / f"{formatter.name}.yaml").write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    staged_files = [path for path in sorted(repo_dir.rglob("*")) if path.is_file()]
    hub_hashes: dict[str, str] = {}
    hash_progress = tqdm(
        total=len(staged_files),
        desc="Hashing staged files",
        unit="file",
        dynamic_ncols=True,
        disable=None,
    )
    try:
        if staging_workers == 1:
            hashes: Iterable[str] = (sha256_file(path) for path in staged_files)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=staging_workers)
            hashes = executor.map(sha256_file, staged_files)
        for path, digest in zip(staged_files, hashes, strict=True):
            relative_path = str(path.relative_to(repo_dir))
            hash_progress.set_postfix(file=relative_path, refresh=False)
            hub_hashes[relative_path] = digest
            hash_progress.update()
    finally:
        if executor is not None:
            executor.shutdown()
        hash_progress.close()
    source_manifest["hub_files"] = hub_hashes
    (repo_dir / "manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return repo_dir


def validate_hub_repository(repo_dir: Path) -> dict[str, Any]:
    """Validate staged files, schemas, IDs, and split relationships offline."""

    _require_hub_dependencies()
    import pyarrow.parquet as parquet

    required = (repo_dir / "README.md", repo_dir / "manifest.json", repo_dir / "generation.yaml")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing staged Hub files: {missing}")
    manifest = json.loads((repo_dir / "manifest.json").read_text(encoding="utf-8"))
    for relative_path, expected_hash in manifest.get("hub_files", {}).items():
        path = repo_dir / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"staged file does not match manifest: {relative_path}")

    parquet_files = sorted((repo_dir / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError("staged Hub repository contains no Parquet files")

    counts: dict[str, int] = {}
    examples_by_config_split: dict[tuple[str, str], set[str]] = {}
    semantics_by_split: dict[str, set[str]] = defaultdict(set)
    schemas: dict[str, Any] = {}
    for path in parquet_files:
        table = parquet.read_table(path)
        config_name = path.parent.name
        split = path.name.split("-", 1)[0]
        key = (config_name, split)
        records = table.to_pylist()
        _validate_multiple_choice_records(records, path)
        example_ids = {str(record["example_id"]) for record in records}
        if len(example_ids) != len(records):
            raise ValueError(f"duplicate example_id values in {path}")
        if example_ids & examples_by_config_split.get(key, set()):
            raise ValueError(f"duplicate example_id values across shards for {config_name}/{split}")
        count_key = f"{config_name}/{split}"
        counts[count_key] = counts.get(count_key, 0) + len(records)
        examples_by_config_split.setdefault(key, set()).update(example_ids)
        semantics_by_split[_base_split(split)].update(str(record["instance_id"]) for record in records)
        if config_name in schemas and schemas[config_name] != table.schema:
            raise ValueError(f"schema mismatch across splits for config {config_name!r}")
        schemas[config_name] = table.schema

    if any(key[0] == "canonical" for key in examples_by_config_split):
        canonical = {
            split: ids for (config, split), ids in examples_by_config_split.items() if config == "canonical"
        }
        for (config, split), ids in examples_by_config_split.items():
            if config != "canonical" and not ids <= canonical.get(_base_split(split), set()):
                raise ValueError(f"rendered config {config!r} contains IDs absent from canonical/{split}")

    overlap: dict[str, int] = {}
    splits = sorted(semantics_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap[f"{left}--{right}"] = len(semantics_by_split[left] & semantics_by_split[right])
    return {"files": len(parquet_files), "counts": counts, "semantic_overlap": overlap}


def push_hub_repository(
    repo_dir: Path,
    *,
    repo_id: str,
    private: bool,
    dry_run: bool,
    sync: bool = False,
    commit_message: str = "Upload state-bench dataset",
) -> dict[str, Any]:
    """Upload a staged repository; optionally replace the remote file tree."""

    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must have the form namespace/name")
    files = [path for path in sorted(repo_dir.rglob("*")) if path.is_file()]
    if sync:
        _validate_sync_source(repo_dir, files)
    plan = {
        "repo_id": repo_id,
        "private": private,
        "sync": sync,
        "delete_patterns": ["*"] if sync else [],
        "files": [str(path.relative_to(repo_dir)) for path in files],
    }
    if dry_run:
        return plan

    _require_hub_dependencies()
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=repo_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        delete_patterns="*" if sync else None,
    )
    return plan


def _validate_sync_source(repo_dir: Path, files: list[Path]) -> None:
    relative_files = {str(path.relative_to(repo_dir)) for path in files}
    required = {"README.md", "generation.yaml", "manifest.json"}
    missing = sorted(required - relative_files)
    if missing:
        raise ValueError(f"sync requires a complete staged repository; missing files: {missing}")
    if not any(path.startswith("data/") and path.endswith(".parquet") for path in relative_files):
        raise ValueError("sync requires a complete staged repository with Parquet data files")


def _stage_config(
    source_dir: Path,
    data_dir: Path,
    config_name: str,
    *,
    shard_size: int,
    target_shard_bytes: int,
    token_strata: TokenStratificationSpec | None = None,
    tokenizer: TokenizerLike | None = None,
    progress: Any | None = None,
    only_splits: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing generated config directory: {source_dir}")
    destination = data_dir / config_name
    destination.mkdir()
    data_files: list[dict[str, str]] = []
    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        split = jsonl_path.stem
        if only_splits is not None and split not in only_splits:
            continue
        if token_strata is None:
            _set_progress_context(progress, config=config_name, split=split, phase="parquet")
            writer = _ParquetShardWriter(
                destination,
                split=split,
                max_rows=shard_size,
                target_bytes=target_shard_bytes,
                progress=progress,
            )
            for record, estimated_bytes in _iter_jsonl_records(jsonl_path):
                writer.add(record, estimated_bytes=estimated_bytes)
            if not writer.finish():
                raise ValueError(f"cannot infer a Parquet schema from empty split: {jsonl_path}")
            data_files.append({"split": split, "path": f"data/{config_name}/{split}-*.parquet"})
            continue
        if tokenizer is None:
            raise ValueError("tokenizer is required when token strata are configured")
        writers = {
            stratum.name: _ParquetShardWriter(
                destination,
                split=f"{split}__{stratum.name}",
                max_rows=shard_size,
                target_bytes=target_shard_bytes,
                progress=progress,
                progress_weight=0.5,
            )
            for stratum in token_strata.ranges
        }
        if token_strata.mode == "sampled":
            _set_progress_context(progress, config=config_name, split=split, phase="calibrate")
            estimator = _calibrate_token_length_estimator(jsonl_path, token_strata, tokenizer)
            _set_progress_context(progress, config=config_name, split=split, phase="estimate")
            for record, estimated_bytes in _iter_jsonl_records(jsonl_path):
                text = _tokenization_text(record, token_strata)
                token_length = estimator.estimate(record, text)
                stratum = _token_stratum(token_length, token_strata)
                writers[stratum.name].add(
                    {
                        **record,
                        "token_length": token_length,
                        "token_length_is_estimate": True,
                        "token_stratum": stratum.name,
                    },
                    estimated_bytes=estimated_bytes,
                )
                if progress is not None:
                    progress.update(0.5)
        else:
            _set_progress_context(progress, config=config_name, split=split, phase="tokenize")
            for batch in _character_bounded_batches(
                _iter_jsonl_records(jsonl_path),
                spec=token_strata,
            ):
                texts = [text for _, _, text in batch]
                lengths = _token_lengths(tokenizer, texts, token_strata)
                for (record, estimated_bytes, _), token_length in zip(batch, lengths, strict=True):
                    stratum = _token_stratum(token_length, token_strata)
                    writers[stratum.name].add(
                        {
                            **record,
                            "token_length": token_length,
                            "token_length_is_estimate": False,
                            "token_stratum": stratum.name,
                        },
                        estimated_bytes=estimated_bytes,
                    )
                if progress is not None:
                    progress.update(len(batch) * 0.5)
        for stratum in token_strata.ranges:
            writer = writers[stratum.name]
            _set_progress_context(
                progress,
                config=config_name,
                split=writer.split,
                phase="parquet",
            )
            if writer.finish():
                data_files.append(
                    {
                        "split": writer.split,
                        "path": f"data/{config_name}/{writer.split}-*.parquet",
                    }
                )
    if not data_files:
        raise FileNotFoundError(f"no generated splits selected under {source_dir}")
    return {"config_name": config_name, "data_files": data_files}


class _ParquetShardWriter:
    """Buffer bounded data and finalize deterministic shard names."""

    def __init__(
        self,
        destination: Path,
        *,
        split: str,
        max_rows: int,
        target_bytes: int,
        progress: Any | None = None,
        progress_weight: float = 1.0,
    ) -> None:
        self.destination = destination
        self.split = split
        self.max_rows = max_rows
        self.target_bytes = target_bytes
        self.progress = progress
        self.progress_weight = progress_weight
        self.buffer: list[dict[str, Any]] = []
        self.buffer_bytes = 0
        self.parts: list[Path] = []

    def add(self, record: dict[str, Any], *, estimated_bytes: int) -> None:
        if self.buffer and (
            len(self.buffer) >= self.max_rows or self.buffer_bytes + estimated_bytes > self.target_bytes
        ):
            self._flush()
        self.buffer.append(record)
        self.buffer_bytes += estimated_bytes

    def finish(self) -> bool:
        if self.buffer:
            self._flush()
        num_shards = len(self.parts)
        for shard_index, part_path in enumerate(self.parts):
            final_path = self.destination / (f"{self.split}-{shard_index:05d}-of-{num_shards:05d}.parquet")
            part_path.rename(final_path)
        return bool(self.parts)

    def _flush(self) -> None:
        _require_hub_dependencies()
        import pyarrow as pa
        import pyarrow.parquet as parquet

        table = _arrow_table(pa, self.buffer)
        part_path = self.destination / f".{self.split}-part-{len(self.parts):05d}.parquet"
        parquet.write_table(table, part_path, compression="zstd")
        self.parts.append(part_path)
        if self.progress is not None:
            self.progress.update(len(self.buffer) * self.progress_weight)
        self.buffer = []
        self.buffer_bytes = 0


def _iter_jsonl_records(path: Path) -> Iterator[tuple[dict[str, Any], int]]:
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line), len(line)


def _character_bounded_batches(
    records: Iterable[tuple[dict[str, Any], int]],
    *,
    spec: TokenStratificationSpec,
) -> Iterator[list[tuple[dict[str, Any], int, str]]]:
    batch: list[tuple[dict[str, Any], int, str]] = []
    characters = 0
    for record, estimated_bytes in records:
        text = _tokenization_text(record, spec)
        if batch and (len(batch) >= spec.batch_size or characters + len(text) > spec.max_batch_characters):
            yield batch
            batch = []
            characters = 0
        batch.append((record, estimated_bytes, text))
        characters += len(text)
    if batch:
        yield batch


@dataclass(frozen=True, slots=True)
class _TokenLengthEstimator:
    points_by_group: dict[str, tuple[tuple[int, int], ...]]
    global_points: tuple[tuple[int, int], ...]

    def estimate(self, record: dict[str, Any], text: str) -> int:
        points = self.points_by_group.get(_calibration_group(record), self.global_points)
        return _interpolate_token_length(points, len(text))


def _calibrate_token_length_estimator(
    path: Path,
    spec: TokenStratificationSpec,
    tokenizer: TokenizerLike,
) -> _TokenLengthEstimator:
    scanned: list[tuple[dict[str, Any], int, str]] = []
    for index, (record, estimated_bytes) in enumerate(_iter_jsonl_records(path)):
        if index >= spec.calibration_scan_rows:
            break
        scanned.append((record, estimated_bytes, _tokenization_text(record, spec)))
    if not scanned:
        raise ValueError(f"cannot calibrate token lengths from empty split: {path}")

    candidates: list[tuple[dict[str, Any], int, str]] = []
    by_group: dict[str, list[tuple[dict[str, Any], int, str]]] = defaultdict(list)
    for item in scanned:
        by_group[_calibration_group(item[0])].append(item)
    for items in by_group.values():
        candidates.extend(_quantile_samples(items, spec.calibration_samples_per_group))

    samples_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    global_samples: list[tuple[int, int]] = []
    sized_candidates = ((record, estimated_bytes) for record, estimated_bytes, _ in candidates)
    for batch in _character_bounded_batches(sized_candidates, spec=spec):
        lengths = _token_lengths(tokenizer, [text for _, _, text in batch], spec)
        for (record, _, text), token_length in zip(batch, lengths, strict=True):
            point = (len(text), token_length)
            samples_by_group[_calibration_group(record)].append(point)
            global_samples.append(point)
    return _TokenLengthEstimator(
        points_by_group={group: _consolidate_points(points) for group, points in samples_by_group.items()},
        global_points=_consolidate_points(global_samples),
    )


def _quantile_samples(
    items: list[tuple[dict[str, Any], int, str]],
    count: int,
) -> list[tuple[dict[str, Any], int, str]]:
    ordered = sorted(items, key=lambda item: len(item[2]))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
    return [ordered[index] for index in sorted(indices)]


def _calibration_group(record: dict[str, Any]) -> str:
    return str(record.get("filler_distraction") or "none")


def _consolidate_points(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    tokens_by_characters: dict[int, list[int]] = defaultdict(list)
    for characters, tokens in points:
        tokens_by_characters[characters].append(tokens)
    return tuple(
        (characters, round(statistics.median(tokens)))
        for characters, tokens in sorted(tokens_by_characters.items())
    )


def _interpolate_token_length(points: tuple[tuple[int, int], ...], characters: int) -> int:
    if not points:
        raise ValueError("token-length estimator has no calibration points")
    if len(points) == 1:
        sample_characters, sample_tokens = points[0]
        if sample_characters == 0:
            return sample_tokens
        return max(0, round(characters * sample_tokens / sample_characters))
    position = bisect_left([point[0] for point in points], characters)
    if position == 0:
        left, right = points[0], points[1]
    elif position == len(points):
        left, right = points[-2], points[-1]
    else:
        left, right = points[position - 1], points[position]
    if left[0] == right[0]:
        return left[1]
    estimate = left[1] + (characters - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
    return max(0, round(estimate))


def _partition_records_by_token_stratum(
    records: list[dict[str, Any]],
    *,
    spec: TokenStratificationSpec,
    tokenizer: TokenizerLike,
    progress: Any | None = None,
    progress_weight: float = 1.0,
    progress_context: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Annotate and partition in-memory rows; Hub staging streams from disk."""

    result = {stratum.name: [] for stratum in spec.ranges}
    if progress_context is not None:
        _set_progress_context(progress, **progress_context)
    sized_records = ((record, len(_tokenization_text(record, spec))) for record in records)
    for batch in _character_bounded_batches(sized_records, spec=spec):
        lengths = _token_lengths(tokenizer, [text for _, _, text in batch], spec)
        for (record, _, _), token_length in zip(batch, lengths, strict=True):
            stratum = _token_stratum(token_length, spec)
            annotated = {**record, "token_length": token_length, "token_stratum": stratum.name}
            result[stratum.name].append(annotated)
        if progress is not None:
            progress.update(len(batch) * progress_weight)
    return result


def _tokenization_text(record: dict[str, Any], spec: TokenStratificationSpec) -> str:
    if "prefix" not in record:
        raise ValueError("token stratification requires a prefix field")
    text = str(record["prefix"])
    if spec.field == "prefix+completion":
        if "completion" not in record:
            raise ValueError("prefix+completion stratification requires a completion field")
        text += spec.joiner + str(record["completion"])
    return text


def _token_lengths(
    tokenizer: TokenizerLike,
    texts: list[str],
    spec: TokenStratificationSpec,
) -> list[int]:
    return token_lengths(tokenizer, texts, spec.add_special_tokens)


def _token_stratum(token_length: int, spec: TokenStratificationSpec) -> TokenStratumSpec:
    return next(
        item
        for item in spec.ranges
        if token_length >= item.min_tokens and (item.max_tokens is None or token_length < item.max_tokens)
    )


def _set_progress_context(progress: Any | None, **context: str) -> None:
    if progress is not None:
        progress.set_postfix(context, refresh=False)


def _base_split(split: str) -> str:
    return split.partition("__")[0]


def _arrow_table(pa: Any, records: list[dict[str, Any]]) -> Any:
    normalized = []
    for record in records:
        normalized_record = dict(record)
        if "operations" in record:
            normalized_record["operations"] = [
                {
                    "op": operation["op"],
                    "target": operation["target"],
                    "value": operation.get("value"),
                    "source": operation.get("source"),
                }
                for operation in record["operations"]
            ]
        normalized.append(normalized_record)

    inferred = pa.Table.from_pylist(normalized)
    explicit_types = {
        "operations": pa.list_(
            pa.struct(
                [
                    pa.field("op", pa.string(), nullable=False),
                    pa.field("target", pa.int64(), nullable=False),
                    pa.field("value", pa.int64()),
                    pa.field("source", pa.int64()),
                ]
            )
        ),
        "filler_distraction": pa.string(),
        "target_extra_assignments": pa.int64(),
    }
    for name, data_type in explicit_types.items():
        if name not in inferred.column_names:
            continue
        index = inferred.column_names.index(name)
        values = [record.get(name) for record in normalized]
        inferred = inferred.set_column(index, pa.field(name, data_type), pa.array(values, type=data_type))
    return inferred


def _dataset_card(*, metadata: dict[str, Any], loaded: LoadedExperiment) -> str:
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    formatter_names = ", ".join(formatter.name for formatter in loaded.formatters)
    strata = loaded.spec.huggingface.token_strata
    strata_description = ""
    if strata is not None:
        interval_text = ", ".join(
            f"`{item.name}`=[{item.min_tokens:,}, "
            f"{item.max_tokens if item.max_tokens is not None else 'infinity'})"
            for item in strata.ranges
        )
        length_method = (
            f"estimated from {strata.calibration_samples_per_group} length-quantile "
            "calibration rows per filler style"
            if strata.mode == "sampled"
            else "measured by exact per-row tokenization"
        )
        strata_description = f"""

Rendered configs are partitioned into tokenizer-length splits named
`<split>__<stratum>` using `{strata.tokenizer}` over `{strata.field}`. The
half-open strata are {interval_text}. Each row records its `token_length`
and `token_stratum`; token lengths are {length_method}. The
`token_length_is_estimate` field distinguishes sampled estimates from exact
measurements. Empty tokenizer ranges are omitted, so consumers should discover
the available split names for each rendered config. The canonical config
retains the unstratified base splits.
"""
    return f"""---
{frontmatter}
---

# {loaded.spec.dataset.name}

Synthetic evaluations of state tracking across four algebraic families:
`r-trivial`, `aperiodic`, `solvable`, and `periodic`. Solvable traces advance
variables with modular increments (`x <- (x + d) mod n`), whose transition
monoid contains only solvable (cyclic) groups; periodic traces permute variable
contents via swaps.

Available surface formatters: {formatter_names}.

Each rendered record contains `prefix`, `completion`, stable `example_id` and
semantic `instance_id` fields, four `incorrect_completions`, shuffled `choices`,
a zero-based `correct_choice_index`, its complexity class, and generation parameters.
The `canonical` config contains formatter-independent operation traces and final
states. See `generation.yaml`, `formats/`, and `manifest.json` for exact
reproduction metadata.
{strata_description}

Reference assignments use eager copy semantics: assigning one variable from
another copies its value at that point in the trace and does not create an alias.
Increments wrap modulo the condition's value count.
"""


def _validate_multiple_choice_records(records: list[dict[str, Any]], path: Path) -> None:
    for record in records:
        if "choices" not in record:
            continue
        choices = record["choices"]
        incorrect = record["incorrect_completions"]
        correct_index = record["correct_choice_index"]
        if len(set(choices)) != len(choices):
            raise ValueError(f"duplicate multiple-choice options in {path}")
        if len(set(incorrect)) != len(incorrect) or record["completion"] in incorrect:
            raise ValueError(f"invalid incorrect completions in {path}")
        if not 0 <= correct_index < len(choices) or choices[correct_index] != record["completion"]:
            raise ValueError(f"incorrect answer key in {path}")


def _require_hub_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("Hugging Face support requires: uv sync --extra hub") from error
