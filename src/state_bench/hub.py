"""Build, validate, and explicitly upload Hugging Face dataset repositories."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import batched
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from state_bench.config import LoadedExperiment, TokenStratificationSpec
from state_bench.io import read_jsonl, sha256_file
from state_bench.stats import TokenizerLike, load_tokenizer


def stage_hub_repository(
    loaded: LoadedExperiment,
    *,
    source_dir: Path,
    repo_dir: Path,
) -> Path:
    """Create a complete local Hub repository without network access."""

    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise FileExistsError(f"Hub staging directory is not empty: {repo_dir}")
    repo_dir.mkdir(parents=True, exist_ok=True)
    data_dir = repo_dir / "data"
    data_dir.mkdir()

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    include_canonical = loaded.spec.huggingface.include_canonical_config
    total_rows = sum(
        int(count)
        for name, count in source_manifest["counts"].items()
        if include_canonical or not name.startswith("canonical/")
    )
    progress = tqdm(
        total=total_rows,
        desc="Staging Hub data",
        unit="row",
        dynamic_ncols=True,
        disable=None,
    )

    configs: list[dict[str, Any]] = []
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
                    progress=progress,
                )
            )
        for config_dir in sorted((source_dir / "rendered").iterdir()):
            if config_dir.is_dir():
                configs.append(
                    _stage_config(
                        config_dir,
                        data_dir,
                        config_dir.name,
                        shard_size=loaded.spec.output.shard_size,
                        token_strata=token_strata,
                        tokenizer=tokenizer,
                        progress=progress,
                    )
                )
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
        for path in staged_files:
            relative_path = str(path.relative_to(repo_dir))
            hash_progress.set_postfix(file=relative_path, refresh=False)
            hub_hashes[relative_path] = sha256_file(path)
            hash_progress.update()
    finally:
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
    token_strata: TokenStratificationSpec | None = None,
    tokenizer: TokenizerLike | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing generated config directory: {source_dir}")
    destination = data_dir / config_name
    destination.mkdir()
    data_files: list[dict[str, str]] = []
    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        split = jsonl_path.stem
        if token_strata is None:
            _set_progress_context(progress, config=config_name, split=split, phase="parquet")
            _jsonl_to_parquet_shards(
                jsonl_path,
                destination,
                split=split,
                shard_size=shard_size,
                progress=progress,
            )
            data_files.append({"split": split, "path": f"data/{config_name}/{split}-*.parquet"})
            continue
        if tokenizer is None:
            raise ValueError("tokenizer is required when token strata are configured")
        records_by_stratum = _partition_records_by_token_stratum(
            read_jsonl(jsonl_path),
            spec=token_strata,
            tokenizer=tokenizer,
            progress=progress,
            progress_weight=0.5,
            progress_context={"config": config_name, "split": split, "phase": "tokenize"},
        )
        for stratum in token_strata.ranges:
            records = records_by_stratum[stratum.name]
            if not records:
                continue
            stratified_split = f"{split}__{stratum.name}"
            _set_progress_context(
                progress,
                config=config_name,
                split=stratified_split,
                phase="parquet",
            )
            _records_to_parquet_shards(
                records,
                destination,
                split=stratified_split,
                shard_size=shard_size,
                progress=progress,
                progress_weight=0.5,
            )
            data_files.append(
                {
                    "split": stratified_split,
                    "path": f"data/{config_name}/{stratified_split}-*.parquet",
                }
            )
    if not data_files:
        raise FileNotFoundError(f"no generated splits found under {source_dir}")
    return {"config_name": config_name, "data_files": data_files}


def _jsonl_to_parquet_shards(
    source: Path,
    destination: Path,
    *,
    split: str,
    shard_size: int,
    progress: Any | None = None,
) -> None:
    records = read_jsonl(source)
    if not records:
        raise ValueError(f"cannot infer a Parquet schema from empty split: {source}")
    _records_to_parquet_shards(
        records,
        destination,
        split=split,
        shard_size=shard_size,
        progress=progress,
    )


def _records_to_parquet_shards(
    records: list[dict[str, Any]],
    destination: Path,
    *,
    split: str,
    shard_size: int,
    progress: Any | None = None,
    progress_weight: float = 1.0,
) -> None:
    _require_hub_dependencies()
    import pyarrow as pa
    import pyarrow.parquet as parquet

    num_shards = (len(records) + shard_size - 1) // shard_size
    for shard_index in range(num_shards):
        chunk = records[shard_index * shard_size : (shard_index + 1) * shard_size]
        table = _arrow_table(pa, chunk)
        name = f"{split}-{shard_index:05d}-of-{num_shards:05d}.parquet"
        parquet.write_table(table, destination / name, compression="zstd")
        if progress is not None:
            progress.update(len(chunk) * progress_weight)


def _partition_records_by_token_stratum(
    records: list[dict[str, Any]],
    *,
    spec: TokenStratificationSpec,
    tokenizer: TokenizerLike,
    progress: Any | None = None,
    progress_weight: float = 1.0,
    progress_context: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Annotate and partition rendered rows using exact tokenizer lengths."""

    result = {stratum.name: [] for stratum in spec.ranges}
    if progress_context is not None:
        _set_progress_context(progress, **progress_context)
    for record_batch in batched(records, spec.batch_size, strict=False):
        batch = list(record_batch)
        texts = []
        for record in batch:
            if "prefix" not in record:
                raise ValueError("token stratification requires a prefix field")
            text = str(record["prefix"])
            if spec.field == "prefix+completion":
                if "completion" not in record:
                    raise ValueError("prefix+completion stratification requires a completion field")
                text += spec.joiner + str(record["completion"])
            texts.append(text)
        encoded = tokenizer(
            texts,
            add_special_tokens=spec.add_special_tokens,
            padding=False,
            truncation=False,
        )
        lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        for record, token_length in zip(batch, lengths, strict=True):
            stratum = next(
                item
                for item in spec.ranges
                if token_length >= item.min_tokens
                and (item.max_tokens is None or token_length < item.max_tokens)
            )
            annotated = {**record, "token_length": token_length, "token_stratum": stratum.name}
            result[stratum.name].append(annotated)
        if progress is not None:
            progress.update(len(batch) * progress_weight)
    return result


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
    operation_type = pa.list_(
        pa.struct(
            [
                pa.field("op", pa.string(), nullable=False),
                pa.field("target", pa.int64(), nullable=False),
                pa.field("value", pa.int64()),
                pa.field("source", pa.int64()),
            ]
        )
    )
    schema = pa.schema(
        [
            pa.field(field.name, operation_type) if field.name == "operations" else field
            for field in inferred.schema
        ]
    )
    return pa.Table.from_pylist(normalized, schema=schema)


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
        strata_description = f"""

Rendered configs are partitioned into tokenizer-length splits named
`<split>__<stratum>` using `{strata.tokenizer}` over `{strata.field}`. The
half-open strata are {interval_text}. Each row records its exact `token_length`
and `token_stratum`. The canonical config retains the unstratified base splits.
"""
    return f"""---
{frontmatter}
---

# {loaded.spec.dataset.name}

Synthetic evaluations of state tracking across three algebraic families:
`r-trivial`, `aperiodic`, and `periodic`.

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
