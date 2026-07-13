"""Build, validate, and explicitly upload Hugging Face dataset repositories."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from state_bench.config import LoadedExperiment
from state_bench.io import read_jsonl, sha256_file


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

    configs: list[dict[str, Any]] = []
    if loaded.spec.huggingface.include_canonical_config:
        configs.append(_stage_config(source_dir / "canonical", data_dir, "canonical"))
    for config_dir in sorted((source_dir / "rendered").iterdir()):
        if config_dir.is_dir():
            configs.append(_stage_config(config_dir, data_dir, config_dir.name))

    metadata = {
        "pretty_name": loaded.spec.dataset.name,
        "tags": ["state-tracking", "formal-languages", "llm-evaluation"],
        "configs": configs,
    }
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

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_manifest["hub_files"] = {
        str(path.relative_to(repo_dir)): sha256_file(path)
        for path in sorted(repo_dir.rglob("*"))
        if path.is_file()
    }
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
        example_ids = {str(record["example_id"]) for record in records}
        if len(example_ids) != len(records):
            raise ValueError(f"duplicate example_id values in {path}")
        counts[f"{config_name}/{split}"] = len(records)
        examples_by_config_split[key] = example_ids
        semantics_by_split[split].update(str(record["instance_id"]) for record in records)
        if config_name in schemas and schemas[config_name] != table.schema:
            raise ValueError(f"schema mismatch across splits for config {config_name!r}")
        schemas[config_name] = table.schema

    if any(key[0] == "canonical" for key in examples_by_config_split):
        canonical = {
            split: ids
            for (config, split), ids in examples_by_config_split.items()
            if config == "canonical"
        }
        for (config, split), ids in examples_by_config_split.items():
            if config != "canonical" and not ids <= canonical.get(split, set()):
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
    commit_message: str = "Upload state-bench dataset",
) -> dict[str, Any]:
    """Upload a staged repository; dry-run performs no authentication or writes."""

    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must have the form namespace/name")
    files = [path for path in sorted(repo_dir.rglob("*")) if path.is_file()]
    plan = {
        "repo_id": repo_id,
        "private": private,
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
    )
    return plan


def _stage_config(source_dir: Path, data_dir: Path, config_name: str) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing generated config directory: {source_dir}")
    destination = data_dir / config_name
    destination.mkdir()
    data_files: list[dict[str, str]] = []
    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        split = jsonl_path.stem
        parquet_name = f"{split}-00000-of-00001.parquet"
        _jsonl_to_parquet(jsonl_path, destination / parquet_name)
        data_files.append({"split": split, "path": f"data/{config_name}/{parquet_name}"})
    if not data_files:
        raise FileNotFoundError(f"no generated splits found under {source_dir}")
    return {"config_name": config_name, "data_files": data_files}


def _jsonl_to_parquet(source: Path, destination: Path) -> None:
    _require_hub_dependencies()
    import pyarrow as pa
    import pyarrow.parquet as parquet

    records = read_jsonl(source)
    if not records:
        raise ValueError(f"cannot infer a Parquet schema from empty split: {source}")
    table = pa.Table.from_pylist(records)
    parquet.write_table(table, destination, compression="zstd")


def _dataset_card(*, metadata: dict[str, Any], loaded: LoadedExperiment) -> str:
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    formatter_names = ", ".join(formatter.name for formatter in loaded.formatters)
    return f"""---
{frontmatter}
---

# {loaded.spec.dataset.name}

Synthetic evaluations of state tracking across three algebraic families:
`r-trivial`, `aperiodic`, and `periodic`.

Available surface formatters: {formatter_names}.

Each rendered record contains `prefix`, `completion`, stable `example_id` and
semantic `instance_id` fields, its complexity class, and generation parameters.
The `canonical` config contains formatter-independent operation traces and final
states. See `generation.yaml`, `formats/`, and `manifest.json` for exact
reproduction metadata.

Reference assignments use eager copy semantics: assigning one variable from
another copies its value at that point in the trace and does not create an alias.
"""


def _require_hub_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("Hugging Face support requires: uv sync --extra hub") from error
