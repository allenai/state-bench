"""Dataset build pipeline and local artifact I/O."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from state_bench.config import LoadedExperiment
from state_bench.generation import generate_instance
from state_bench.rendering import render_instance


def build_dataset(loaded: LoadedExperiment, *, output_dir: Path | None = None) -> Path:
    """Generate canonical and rendered JSONL artifacts from an experiment."""

    spec = loaded.spec
    root = output_dir or _resolve_relative(loaded.path.parent, spec.dataset.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_dir = root / "canonical"
    rendered_dir = root / "rendered"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    unique_semantics: dict[str, set[str]] = defaultdict(set)
    split_semantics: dict[str, set[str]] = defaultdict(set)
    generated_files: list[Path] = []

    canonical_handles: dict[str, Any] = {}
    rendered_handles: dict[tuple[str, str], Any] = {}
    try:
        for split in spec.splits:
            path = canonical_dir / f"{split}.jsonl"
            canonical_handles[split] = path.open("w", encoding="utf-8")
            generated_files.append(path)

        for formatter in loaded.formatters:
            config_names = {
                hub_config_name(
                    spec.huggingface.config_naming,
                    formatter=formatter.name,
                    complexity=condition.complexity.value,
                )
                for condition in spec.generation.conditions
            }
            for config_name in sorted(config_names):
                for split in spec.splits:
                    path = rendered_dir / config_name / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rendered_handles[(config_name, split)] = path.open("w", encoding="utf-8")
                    generated_files.append(path)

        for split_name, split_spec in spec.splits.items():
            for condition_index, condition in enumerate(spec.generation.conditions):
                for num_variables in condition.num_variables:
                    for sample_index in range(split_spec.samples_per_condition):
                        example_id = _example_id(
                            dataset=spec.dataset.name,
                            split=split_name,
                            complexity=condition.complexity.value,
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            condition_index=condition_index,
                            sample_index=sample_index,
                        )
                        semantic_seed = derive_seed(spec.dataset.seed, "semantic", example_id)
                        instance = generate_instance(
                            complexity=condition.complexity,
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            min_extra_assignments=condition.extra_assignments.min,
                            max_extra_assignments=condition.extra_assignments.max,
                            query_mode=spec.generation.query_mode,
                            distinct_initial_values=spec.generation.distinct_initial_values,
                            require_strict_witness=spec.generation.require_strict_witness,
                            allow_self_reference=spec.generation.allow_self_reference,
                            rng=random.Random(semantic_seed),
                        )
                        canonical_record = {"example_id": example_id, **instance.to_record()}
                        write_json_line(canonical_handles[split_name], canonical_record)
                        counts[f"canonical/{split_name}"] += 1
                        unique_semantics[f"canonical/{split_name}"].add(instance.instance_id)
                        split_semantics[split_name].add(instance.instance_id)

                        for formatter in loaded.formatters:
                            config_name = hub_config_name(
                                spec.huggingface.config_naming,
                                formatter=formatter.name,
                                complexity=condition.complexity.value,
                            )
                            surface_seed = derive_seed(
                                spec.dataset.seed,
                                "surface",
                                example_id,
                                formatter.name,
                            )
                            rendered = render_instance(
                                instance,
                                formatter,
                                rng=random.Random(surface_seed),
                            )
                            rendered_record = {"example_id": example_id, **rendered.to_record()}
                            if spec.output.include_canonical_trace:
                                rendered_record["operations"] = canonical_record["operations"]
                                rendered_record["query"] = canonical_record["query"]
                            if spec.output.include_final_state:
                                rendered_record["final_state"] = canonical_record["final_state"]
                            write_json_line(rendered_handles[(config_name, split_name)], rendered_record)
                            counts[f"{config_name}/{split_name}"] += 1
                            unique_semantics[f"{config_name}/{split_name}"].add(instance.instance_id)
    finally:
        for handle in (*canonical_handles.values(), *rendered_handles.values()):
            handle.close()

    manifest = {
        "schema_version": 1,
        "dataset": spec.dataset.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_config": str(loaded.path),
        "experiment_config_hash": normalized_hash(spec.model_dump(mode="json")),
        "formatter_config_hashes": {
            formatter.name: formatter.config_hash for formatter in loaded.formatters
        },
        "counts": dict(sorted(counts.items())),
        "unique_semantic_counts": {
            key: len(value) for key, value in sorted(unique_semantics.items())
        },
        "cross_split_semantic_overlap": _split_overlap(split_semantics),
        "files": {
            str(path.relative_to(root)): sha256_file(path) for path in sorted(generated_files)
        },
        "huggingface": spec.huggingface.model_dump(mode="json"),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_line(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def hub_config_name(pattern: str, *, formatter: str, complexity: str) -> str:
    return pattern.format(formatter=formatter, complexity=complexity)


def derive_seed(root_seed: int, *parts: str) -> int:
    content = "\0".join((str(root_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "big")


def normalized_hash(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _example_id(
    *,
    dataset: str,
    split: str,
    complexity: str,
    num_variables: int,
    num_values: int,
    condition_index: int,
    sample_index: int,
) -> str:
    value = (
        f"{dataset}\0{split}\0{complexity}\0{num_variables}\0{num_values}\0"
        f"{condition_index}\0{sample_index}"
    )
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _split_overlap(split_semantics: dict[str, set[str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    splits = sorted(split_semantics)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            result[f"{left}--{right}"] = len(split_semantics[left] & split_semantics[right])
    return result


def _resolve_relative(parent: Path, path: Path) -> Path:
    return path if path.is_absolute() else (parent / path).resolve()
