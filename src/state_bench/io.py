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

from tqdm.auto import tqdm

from state_bench.config import LoadedExperiment, condition_uses_formatter
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import derive_instance, generate_periodic_source
from state_bench.model import Complexity
from state_bench.rendering import render_instance
from state_bench.scheduling import build_assignment_plan, build_filler_plan
from state_bench.semantics import dependency_summary


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
    total_examples = sum(
        len(Complexity)
        * split.samples_per_condition
        * sum(len(condition.num_variables) for condition in spec.generation.conditions)
        for split in spec.splits.values()
    )
    progress = tqdm(
        total=total_examples,
        desc="Building dataset",
        unit="example",
        dynamic_ncols=True,
        disable=None,
    )
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
                    complexity=complexity.value,
                )
                for condition in spec.generation.conditions
                for complexity in Complexity
                if condition_uses_formatter(condition, formatter.name)
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
                    progress.set_postfix(
                        split=split_name,
                        source="periodic",
                        variables=num_variables,
                        refresh=False,
                    )
                    assignment_plan = build_assignment_plan(
                        condition.extra_assignments,
                        total_samples=split_spec.samples_per_condition,
                        rng=random.Random(
                            derive_seed(
                                spec.dataset.seed,
                                "assignment-plan",
                                split_name,
                                str(condition_index),
                                str(num_variables),
                            )
                        ),
                    )
                    filler_plan = build_filler_plan(
                        condition.filler,
                        total_samples=split_spec.samples_per_condition,
                        rng=random.Random(
                            derive_seed(
                                spec.dataset.seed,
                                "filler-plan",
                                split_name,
                                str(condition_index),
                                str(num_variables),
                            )
                        ),
                    )
                    for sample_index, (plan_item, filler_item) in enumerate(
                        zip(assignment_plan, filler_plan, strict=True)
                    ):
                        source_example_id = _example_id(
                            dataset=spec.dataset.name,
                            split=split_name,
                            complexity="periodic-source",
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            condition_index=condition_index,
                            sample_index=sample_index,
                            target_extra_assignments=plan_item.target_extra_assignments,
                            filler_proportion=filler_item.proportion,
                            filler_distraction=filler_item.distraction,
                        )
                        semantic_seed = derive_seed(spec.dataset.seed, "semantic", source_example_id)
                        source = generate_periodic_source(
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            min_extra_assignments=plan_item.min_extra_assignments,
                            max_extra_assignments=plan_item.max_extra_assignments,
                            query_mode=spec.generation.query_mode,
                            distinct_initial_values=spec.generation.distinct_initial_values,
                            require_strict_witness=spec.generation.require_strict_witness,
                            allow_self_reference=spec.generation.allow_self_reference,
                            periodic_constant_probability=(spec.generation.periodic_constant_probability),
                            rng=random.Random(semantic_seed),
                        )
                        for complexity in Complexity:
                            instance = derive_instance(source, complexity=complexity)
                            example_id = _derived_example_id(source_example_id, complexity)
                            canonical_record = {
                                "example_id": example_id,
                                "source_example_id": source_example_id,
                                "source_instance_id": source.instance_id,
                                **instance.to_record(),
                                **dependency_summary(instance).to_record(),
                                "num_extra_assignments": len(instance.operations) - num_variables,
                                "target_extra_assignments": plan_item.target_extra_assignments,
                                "filler_proportion": filler_item.proportion,
                                "filler_distraction": filler_item.distraction,
                            }
                            write_json_line(canonical_handles[split_name], canonical_record)
                            counts[f"canonical/{split_name}"] += 1
                            unique_semantics[f"canonical/{split_name}"].add(instance.instance_id)
                            split_semantics[split_name].add(instance.instance_id)

                            for formatter in loaded.formatters:
                                if not condition_uses_formatter(condition, formatter.name):
                                    continue
                                config_name = hub_config_name(
                                    spec.huggingface.config_naming,
                                    formatter=formatter.name,
                                    complexity=complexity.value,
                                )
                                surface_seed = derive_seed(
                                    spec.dataset.seed,
                                    "surface",
                                    source_example_id,
                                    formatter.name,
                                )
                                incorrect_states = generate_incorrect_states(
                                    instance,
                                    num_distractors=spec.output.multiple_choice.num_distractors,
                                    rng=random.Random(
                                        derive_seed(spec.dataset.seed, "distractors", example_id)
                                    ),
                                )
                                rendered = render_instance(
                                    instance,
                                    formatter,
                                    rng=random.Random(surface_seed),
                                    incorrect_states=incorrect_states,
                                    shuffle_choices=spec.output.multiple_choice.shuffle_choices,
                                    filler_proportion=filler_item.proportion,
                                    filler_distraction=filler_item.distraction,
                                )
                                rendered_record = {
                                    "example_id": example_id,
                                    "source_example_id": source_example_id,
                                    "source_instance_id": source.instance_id,
                                    **rendered.to_record(),
                                    "num_extra_assignments": len(instance.operations) - num_variables,
                                    "target_extra_assignments": plan_item.target_extra_assignments,
                                }
                                if spec.output.include_canonical_trace:
                                    rendered_record["operations"] = canonical_record["operations"]
                                    rendered_record["source_operations"] = [
                                        operation.to_record() for operation in source.operations
                                    ]
                                    rendered_record["query"] = canonical_record["query"]
                                if spec.output.include_final_state:
                                    rendered_record["final_state"] = canonical_record["final_state"]
                                write_json_line(
                                    rendered_handles[(config_name, split_name)], rendered_record
                                )
                                counts[f"{config_name}/{split_name}"] += 1
                                unique_semantics[f"{config_name}/{split_name}"].add(
                                    instance.instance_id
                                )
                            progress.update()
    finally:
        progress.close()
        for handle in (*canonical_handles.values(), *rendered_handles.values()):
            handle.close()

    file_hashes: dict[str, str] = {}
    hash_progress = tqdm(
        total=len(generated_files),
        desc="Hashing dataset files",
        unit="file",
        dynamic_ncols=True,
        disable=None,
    )
    try:
        for path in sorted(generated_files):
            relative_path = str(path.relative_to(root))
            hash_progress.set_postfix(file=relative_path, refresh=False)
            file_hashes[relative_path] = sha256_file(path)
            hash_progress.update()
    finally:
        hash_progress.close()

    manifest = {
        "schema_version": 1,
        "dataset": spec.dataset.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_config": str(loaded.path),
        "experiment_config_hash": normalized_hash(spec.model_dump(mode="json")),
        "formatter_config_hashes": {formatter.name: formatter.config_hash for formatter in loaded.formatters},
        "counts": dict(sorted(counts.items())),
        "unique_semantic_counts": {key: len(value) for key, value in sorted(unique_semantics.items())},
        "cross_split_semantic_overlap": _split_overlap(split_semantics),
        "files": file_hashes,
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
    target_extra_assignments: int | None,
    filler_proportion: float,
    filler_distraction: str | None,
) -> str:
    value = (
        f"{dataset}\0{split}\0{complexity}\0{num_variables}\0{num_values}\0{condition_index}\0{sample_index}"
    )
    if target_extra_assignments is not None:
        value += f"\0{target_extra_assignments}"
    if filler_proportion != 0 or filler_distraction is not None:
        value += f"\0{filler_proportion:.17g}\0{filler_distraction or 'none'}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _derived_example_id(source_example_id: str, complexity: Complexity) -> str:
    """Give each family row a unique ID while preserving its source linkage."""

    return hashlib.sha256(f"{source_example_id}\0{complexity.value}".encode()).hexdigest()[:24]


def _split_overlap(split_semantics: dict[str, set[str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    splits = sorted(split_semantics)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            result[f"{left}--{right}"] = len(split_semantics[left] & split_semantics[right])
    return result


def _resolve_relative(parent: Path, path: Path) -> Path:
    return path if path.is_absolute() else (parent / path).resolve()
