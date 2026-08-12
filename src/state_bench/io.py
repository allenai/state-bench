"""Dataset build pipeline and local artifact I/O."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched, repeat
from pathlib import Path
from typing import Any, cast

from tqdm.auto import tqdm

from state_bench.config import (
    AssignmentRange,
    FormatterSpec,
    LoadedExperiment,
    condition_uses_formatter,
    condition_uses_split,
    split_uses_formatter,
)
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import derive_variants, generate_periodic_source
from state_bench.model import Complexity, QueryMode, StateTrackingInstance
from state_bench.rendering import RenderedExample, render_instance
from state_bench.scheduling import (
    AssignmentPlanItem,
    FillerPlanItem,
    build_assignment_plan,
    build_filler_plan,
    build_stratified_filler_plan,
    iter_balanced_filler_plan,
    iter_randomized_assignment_plan,
)
from state_bench.semantics import dependency_summary, sensitivity_summary

_MINIMAL_SOURCE_CHUNK_SIZE = 1024


def _available_cpu_count() -> int | None:
    """Return CPUs available to this process on every supported Python version."""

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count()
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        return len(get_affinity(0))
    return os.cpu_count()


def _executor_map(
    executor: ProcessPoolExecutor,
    function: Any,
    items: Any,
    *,
    chunksize: int,
    buffersize: int,
) -> Any:
    """Map with bounded submission when the running Python supports it."""

    if "buffersize" in inspect.signature(executor.map).parameters:
        map_with_buffer = cast("Any", executor.map)
        return map_with_buffer(
            function,
            items,
            chunksize=chunksize,
            buffersize=buffersize,
        )
    return _bounded_executor_map(
        executor,
        function,
        items,
        chunksize=chunksize,
        buffersize=buffersize,
    )


def _bounded_executor_map(
    executor: ProcessPoolExecutor,
    function: Any,
    items: Any,
    *,
    chunksize: int,
    buffersize: int,
) -> Iterator[Any]:
    """Backport bounded ``Executor.map`` behavior for Python before 3.14."""

    batches = iter(batched(items, chunksize))
    pending = deque()
    for _ in range(buffersize):
        batch = next(batches, None)
        if batch is None:
            break
        pending.append(executor.submit(_run_map_batch, function, batch))

    while pending:
        yield from pending.popleft().result()
        batch = next(batches, None)
        if batch is not None:
            pending.append(executor.submit(_run_map_batch, function, batch))


def _run_map_batch(function: Any, batch: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(map(function, batch))


@dataclass(frozen=True, slots=True)
class _GenerationTask:
    """All information needed to generate one deterministic source triplet."""

    source_example_id: str
    num_variables: int
    num_values: int
    min_extra_assignments: int
    max_extra_assignments: int
    query_mode: QueryMode
    distinct_initial_values: bool
    require_strict_witness: bool
    periodic_constant_probability: float
    root_seed: int
    excluded_semantics: frozenset[str]
    plan_item: AssignmentPlanItem
    filler_item: FillerPlanItem


@dataclass(frozen=True, slots=True)
class _GeneratedSample:
    """Compact worker result; large shared task context is not returned over IPC."""

    source_example_id: str
    plan_item: AssignmentPlanItem
    filler_item: FillerPlanItem
    source: StateTrackingInstance
    instances: tuple[StateTrackingInstance, ...]


@dataclass(frozen=True, slots=True)
class _MinimalChunkTask:
    """A bounded source range rendered and serialized entirely by one worker."""

    chunk_index: int
    samples: tuple[_GenerationTask, ...]
    formatters: tuple[FormatterSpec, ...]
    config_naming: str
    chunk_dir: Path


@dataclass(frozen=True, slots=True)
class _MinimalChunkResult:
    chunk_index: int
    source_count: int
    files: tuple[tuple[str, Path, int], ...]


def _generate_instances(
    task: _GenerationTask,
) -> _GeneratedSample:
    """Generate a source and its derived variants in an isolated worker."""

    semantic_seed = derive_seed(task.root_seed, "semantic", task.source_example_id)
    for retry_index in range(10_000):
        retry_seed = (
            semantic_seed
            if retry_index == 0
            else derive_seed(task.root_seed, "semantic-retry", task.source_example_id, str(retry_index))
        )
        source = generate_periodic_source(
            num_variables=task.num_variables,
            num_values=task.num_values,
            min_extra_assignments=task.min_extra_assignments,
            max_extra_assignments=task.max_extra_assignments,
            query_mode=task.query_mode,
            distinct_initial_values=task.distinct_initial_values,
            require_strict_witness=task.require_strict_witness,
            periodic_constant_probability=task.periodic_constant_probability,
            rng=random.Random(retry_seed),
        )
        instances = derive_variants(source)
        if not ({instance.instance_id for instance in instances} & task.excluded_semantics):
            return _GeneratedSample(
                source_example_id=task.source_example_id,
                plan_item=task.plan_item,
                filler_item=task.filler_item,
                source=source,
                instances=instances,
            )
    raise RuntimeError(f"could not generate source {task.source_example_id!r} after 10000 attempts")


def _write_minimal_chunk(task: _MinimalChunkTask) -> _MinimalChunkResult:
    """Generate, render, serialize, and write one deterministic minimal chunk."""

    handles: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    counts: Counter[str] = Counter()
    try:
        for sample_task in task.samples:
            generated = _generate_instances(sample_task)
            for complexity, instance in zip(Complexity, generated.instances, strict=True):
                example_id = _derived_example_id(generated.source_example_id, complexity)
                for formatter in task.formatters:
                    config_name = hub_config_name(
                        task.config_naming,
                        formatter=formatter.name,
                        complexity=complexity.value,
                    )
                    if config_name not in handles:
                        path = task.chunk_dir / config_name / f"{task.chunk_index:08d}.jsonl"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        paths[config_name] = path
                        handles[config_name] = path.open("w", encoding="utf-8")
                    rendered = render_instance(
                        instance,
                        formatter,
                        rng=random.Random(
                            derive_seed(
                                sample_task.root_seed,
                                "surface",
                                generated.source_example_id,
                                formatter.name,
                            )
                        ),
                        incorrect_states=(),
                        shuffle_choices=False,
                        filler_proportion=generated.filler_item.proportion,
                        filler_distraction=generated.filler_item.distraction,
                        include_diagnostics=False,
                    )
                    write_json_line(
                        handles[config_name],
                        _minimal_rendered_record(
                            example_id=example_id,
                            source_example_id=generated.source_example_id,
                            source_instance_id=generated.source.instance_id,
                            rendered=rendered,
                        ),
                    )
                    counts[config_name] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return _MinimalChunkResult(
        chunk_index=task.chunk_index,
        source_count=len(task.samples),
        files=tuple(
            (config_name, paths[config_name], counts[config_name])
            for config_name in sorted(paths)
        ),
    )


def build_dataset(
    loaded: LoadedExperiment,
    *,
    output_dir: Path | None = None,
    workers: int | None = None,
    only_splits: frozenset[str] | None = None,
) -> Path:
    """Generate canonical and rendered JSONL artifacts from an experiment.

    When ``only_splits`` is given, only those splits are generated; any split
    they must stay disjoint from that is not part of this run is enforced from
    the semantic IDs a prior build persisted under ``<output>/semantics``.
    """

    spec = loaded.spec
    if only_splits is not None:
        unknown_splits = only_splits - spec.splits.keys()
        if unknown_splits:
            raise ValueError(f"unknown splits requested: {sorted(unknown_splits)}")
        if not only_splits:
            raise ValueError("only_splits must not be empty")

    def selected(split: str) -> bool:
        return only_splits is None or split in only_splits

    auto_workers = workers is None
    workers = max(1, (_available_cpu_count() or 1) - 1) if auto_workers else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    root = output_dir or _resolve_relative(loaded.path.parent, spec.dataset.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_dir = root / "canonical"
    rendered_dir = root / "rendered"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    unique_semantics: dict[str, set[str]] = defaultdict(set)
    split_semantics: dict[str, set[str]] = defaultdict(set)
    referenced_splits = {
        referenced for split in spec.splits.values() for referenced in split.disjoint_from
    }
    generated_files: list[Path] = []

    canonical_handles: dict[str, Any] = {}
    rendered_handles: dict[tuple[str, str], Any] = {}
    total_examples = sum(
        len(Complexity)
        * split.samples_per_condition
        * sum(
            len(condition.num_variables)
            for condition in spec.generation.conditions
            if condition_uses_split(condition, split_name)
        )
        for split_name, split in spec.splits.items()
        if selected(split_name)
    )
    progress = tqdm(
        total=total_examples,
        desc="Building dataset",
        unit="example",
        dynamic_ncols=True,
        disable=None,
    )
    if workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except (NotImplementedError, PermissionError):
            # Some restricted execution environments disallow OS semaphores.
            # The automatic setting should remain usable there; an explicit
            # worker request still reports the platform limitation.
            if not auto_workers:
                raise
            executor = None
    else:
        executor = None
    chunk_root: Path | None = None
    try:
        for split in spec.splits:
            if not selected(split):
                continue
            if split in spec.output.omit_canonical_splits:
                # A rebuild may reuse an output directory created before this
                # split was made minimal; do not leave stale canonical rows.
                (canonical_dir / f"{split}.jsonl").unlink(missing_ok=True)
                continue
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
                for split, split_spec in spec.splits.items():
                    if not selected(split):
                        continue
                    if not split_uses_formatter(split_spec, formatter.name):
                        continue
                    if not any(
                        condition_uses_split(condition, split)
                        and condition_uses_formatter(condition, formatter.name)
                        for condition in spec.generation.conditions
                    ):
                        continue
                    path = rendered_dir / config_name / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rendered_handles[(config_name, split)] = path.open("w", encoding="utf-8")
                    generated_files.append(path)

        for split_name, split_spec in spec.splits.items():
            if not selected(split_name):
                continue
            excluded_semantics: set[str] = set()
            for disjoint_name in split_spec.disjoint_from:
                if selected(disjoint_name):
                    # Built earlier in this run; the validator guarantees a
                    # disjoint_from split precedes the split that references it.
                    excluded_semantics |= split_semantics[disjoint_name]
                else:
                    excluded_semantics |= _load_persisted_semantics(root, disjoint_name)
            for condition_index, condition in enumerate(spec.generation.conditions):
                if not condition_uses_split(condition, split_name):
                    continue
                for num_variables in condition.num_variables:
                    progress.set_postfix(
                        split=split_name,
                        source="periodic",
                        variables=num_variables,
                        refresh=False,
                    )
                    if isinstance(condition.extra_assignments, AssignmentRange):
                        assignment_plan = repeat(
                            # Ranges use the same bounds for every source, so
                            # one shared item is sufficient.
                            _assignment_range_item(condition.extra_assignments),
                            split_spec.samples_per_condition,
                        )
                    elif split_spec.samples_per_condition >= 100_000:
                        assignment_plan = iter_randomized_assignment_plan(
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
                    else:
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
                    filler_rng = random.Random(
                        derive_seed(
                            spec.dataset.seed,
                            "filler-plan",
                            split_name,
                            str(condition_index),
                            str(num_variables),
                        )
                    )
                    if condition.filler.stratify_by_assignment_bucket:
                        # Stratification needs bucket membership, which is
                        # currently represented by the finite assignment plan.
                        assignment_plan = tuple(assignment_plan)
                        filler_plan = build_stratified_filler_plan(
                            condition.filler,
                            assignment_plan=assignment_plan,
                            rng=filler_rng,
                        )
                    elif split_spec.samples_per_condition >= 100_000:
                        filler_plan = iter_balanced_filler_plan(
                            condition.filler,
                            total_samples=split_spec.samples_per_condition,
                            rng=filler_rng,
                        )
                    else:
                        filler_plan = build_filler_plan(
                            condition.filler,
                            total_samples=split_spec.samples_per_condition,
                            rng=filler_rng,
                        )
                    planned_samples = enumerate(zip(assignment_plan, filler_plan, strict=True))
                    frozen_excluded_semantics = frozenset(excluded_semantics)
                    tasks = (
                        _GenerationTask(
                            source_example_id=_example_id(
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
                            ),
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            min_extra_assignments=plan_item.min_extra_assignments,
                            max_extra_assignments=plan_item.max_extra_assignments,
                            query_mode=spec.generation.query_mode,
                            distinct_initial_values=spec.generation.distinct_initial_values,
                            require_strict_witness=spec.generation.require_strict_witness,
                            periodic_constant_probability=spec.generation.periodic_constant_probability,
                            root_seed=spec.dataset.seed,
                            excluded_semantics=frozen_excluded_semantics,
                            plan_item=plan_item,
                            filler_item=filler_item,
                        )
                        for sample_index, (plan_item, filler_item) in planned_samples
                    )
                    use_minimal_chunks = (
                        executor is not None
                        and split_name in spec.output.minimal_splits
                        and split_name in spec.output.omit_canonical_splits
                        and not spec.output.track_unique_semantics
                        and split_name not in referenced_splits
                    )
                    if use_minimal_chunks:
                        if chunk_root is None:
                            chunk_root = Path(tempfile.mkdtemp(prefix=".state-bench-chunks-", dir=root))
                        selected_formatters = tuple(
                            formatter
                            for formatter in loaded.formatters
                            if condition_uses_formatter(condition, formatter.name)
                            and split_uses_formatter(split_spec, formatter.name)
                        )
                        chunk_tasks = (
                            _MinimalChunkTask(
                                chunk_index=chunk_index,
                                samples=tuple(sample_batch),
                                formatters=selected_formatters,
                                config_naming=spec.huggingface.config_naming,
                                chunk_dir=chunk_root,
                            )
                                for chunk_index, sample_batch in enumerate(
                                    batched(tasks, _MINIMAL_SOURCE_CHUNK_SIZE)
                                )
                        )
                        chunk_results = _executor_map(
                            executor,
                            _write_minimal_chunk,
                            chunk_tasks,
                            chunksize=1,
                            buffersize=max(1, workers * 2),
                        )
                        for result in chunk_results:
                            for config_name, chunk_path, row_count in result.files:
                                with chunk_path.open(encoding="utf-8") as chunk_handle:
                                    shutil.copyfileobj(
                                        chunk_handle,
                                        rendered_handles[(config_name, split_name)],
                                    )
                                chunk_path.unlink()
                                counts[f"{config_name}/{split_name}"] += row_count
                            progress.update(len(Complexity) * result.source_count)
                        continue
                    generated_samples = (
                        _executor_map(
                            executor,
                            _generate_instances,
                            tasks,
                            chunksize=256,
                            buffersize=max(1, workers * 4),
                        )
                        if executor is not None
                        else map(_generate_instances, tasks)
                    )
                    for generated in generated_samples:
                        source_example_id = generated.source_example_id
                        plan_item = generated.plan_item
                        filler_item = generated.filler_item
                        source = generated.source
                        instances = generated.instances
                        for complexity, instance in zip(Complexity, instances, strict=True):
                            example_id = _derived_example_id(source_example_id, complexity)
                            needs_canonical_record = (
                                split_name in canonical_handles
                                or split_name not in spec.output.minimal_splits
                            )
                            canonical_record: dict[str, Any] | None = None
                            if needs_canonical_record:
                                canonical_record = {
                                    "example_id": example_id,
                                    "source_example_id": source_example_id,
                                    "source_instance_id": source.instance_id,
                                    **instance.to_record(),
                                    **dependency_summary(instance).to_record(),
                                    **sensitivity_summary(instance).to_record(),
                                    "num_extra_assignments": (
                                        len(instance.operations) - instance.num_initial_assignments
                                    ),
                                    "target_extra_assignments": plan_item.target_extra_assignments,
                                    "filler_proportion": filler_item.proportion,
                                    "filler_distraction": filler_item.distraction,
                                }
                            if split_name in canonical_handles:
                                assert canonical_record is not None
                                write_json_line(canonical_handles[split_name], canonical_record)
                                counts[f"canonical/{split_name}"] += 1
                            if spec.output.track_unique_semantics:
                                unique_semantics[f"canonical/{split_name}"].add(instance.instance_id)
                            # Retain only semantics that a later split needs for
                            # disjointness. Keeping every training ID is prohibitive
                            # for multi-billion-token builds.
                            if spec.output.track_unique_semantics or split_name in referenced_splits:
                                split_semantics[split_name].add(instance.instance_id)

                            include_trace = (
                                split_name not in spec.output.minimal_splits
                                and spec.output.include_canonical_trace
                            )
                            source_operations = (
                                [operation.to_record() for operation in source.operations]
                                if include_trace
                                else None
                            )
                            minimal_output = split_name in spec.output.minimal_splits
                            incorrect_states = (
                                ()
                                if minimal_output
                                else generate_incorrect_states(
                                    instance,
                                    num_distractors=spec.output.multiple_choice.num_distractors,
                                    rng=random.Random(
                                        derive_seed(spec.dataset.seed, "distractors", example_id)
                                    ),
                                )
                            )

                            for formatter in loaded.formatters:
                                if not (
                                    condition_uses_formatter(condition, formatter.name)
                                    and split_uses_formatter(split_spec, formatter.name)
                                ):
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
                                rendered = render_instance(
                                    instance,
                                    formatter,
                                    rng=random.Random(surface_seed),
                                    incorrect_states=incorrect_states,
                                    shuffle_choices=spec.output.multiple_choice.shuffle_choices,
                                    filler_proportion=filler_item.proportion,
                                    filler_distraction=filler_item.distraction,
                                    include_diagnostics=not minimal_output,
                                )
                                rendered_record: dict[str, Any]
                                if minimal_output:
                                    rendered_record = cast(
                                        "dict[str, Any]",
                                        _minimal_rendered_record(
                                            example_id=example_id,
                                            source_example_id=source_example_id,
                                            source_instance_id=source.instance_id,
                                            rendered=rendered,
                                        ),
                                    )
                                else:
                                    rendered_record = {
                                    "example_id": example_id,
                                    "source_example_id": source_example_id,
                                    "source_instance_id": source.instance_id,
                                    **rendered.to_record(),
                                    "num_extra_assignments": (
                                        len(instance.operations) - instance.num_initial_assignments
                                    ),
                                    "target_extra_assignments": plan_item.target_extra_assignments,
                                    }
                                if include_trace:
                                    assert canonical_record is not None
                                    assert source_operations is not None
                                    rendered_record["operations"] = canonical_record["operations"]
                                    rendered_record["source_operations"] = source_operations
                                    rendered_record["query"] = canonical_record["query"]
                                if (
                                    split_name not in spec.output.minimal_splits
                                    and spec.output.include_final_state
                                ):
                                    assert canonical_record is not None
                                    rendered_record["final_state"] = canonical_record["final_state"]
                                # Keep the training payload alongside the evaluation metadata so
                                # raw rendered JSONL is directly consumable by Dolma and HF JSON.
                                rendered_record.update(
                                    _document_fields(
                                        example_id=example_id,
                                        prefix=rendered.prefix,
                                        completion=rendered.completion,
                                    )
                                )
                                write_json_line(
                                    rendered_handles[(config_name, split_name)], rendered_record
                                )
                                counts[f"{config_name}/{split_name}"] += 1
                                if spec.output.track_unique_semantics:
                                    unique_semantics[f"{config_name}/{split_name}"].add(
                                        instance.instance_id
                                    )
                            progress.update()
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        for handle in (*canonical_handles.values(), *rendered_handles.values()):
            handle.close()
        if chunk_root is not None:
            shutil.rmtree(chunk_root, ignore_errors=True)

    # Persist the semantic IDs of every generated split that a later split must
    # stay disjoint from, so a subsequent independent build (for example a large
    # train split) can enforce disjointness from this run's test split.
    for split_name in spec.splits:
        if selected(split_name) and split_name in referenced_splits:
            _persist_split_semantics(root, split_name, split_semantics[split_name])

    file_hashes: dict[str, str] = {}
    if spec.output.hash_files:
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
        "unique_semantic_counts": (
            {key: len(value) for key, value in sorted(unique_semantics.items())}
            if spec.output.track_unique_semantics
            else None
        ),
        "cross_split_semantic_overlap": _split_overlap(split_semantics),
        "files": file_hashes,
        "huggingface": spec.huggingface.model_dump(mode="json"),
    }
    manifest_path = root / "manifest.json"
    if only_splits is not None and manifest_path.exists():
        # A partial build only knows about the splits it generated; fold its
        # counts and file hashes into the manifest a prior run left behind so
        # the record for untouched splits survives.
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _merge_manifest(existing_manifest, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _merge_manifest(existing: dict[str, Any], updated: dict[str, Any]) -> dict[str, Any]:
    """Overlay a partial build's manifest onto a prior full or partial manifest."""

    merged = {**existing, **updated}
    merged["counts"] = dict(sorted({**existing.get("counts", {}), **updated["counts"]}.items()))
    merged["files"] = dict(sorted({**existing.get("files", {}), **(updated["files"] or {})}.items()))
    existing_unique = existing.get("unique_semantic_counts") or {}
    updated_unique = updated["unique_semantic_counts"]
    merged["unique_semantic_counts"] = (
        dict(sorted({**existing_unique, **updated_unique}.items()))
        if updated_unique is not None
        else (existing_unique or None)
    )
    # A partial run cannot recompute overlaps against splits it did not build;
    # keep the richer record whenever this run produced none.
    if not updated["cross_split_semantic_overlap"]:
        merged["cross_split_semantic_overlap"] = existing.get("cross_split_semantic_overlap", {})
    return merged


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_line(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def _minimal_rendered_record(
    *,
    example_id: str,
    source_example_id: str,
    source_instance_id: str,
    rendered: RenderedExample,
) -> dict[str, str | int | float]:
    record = {
        "example_id": example_id,
        "source_example_id": source_example_id,
        "source_instance_id": source_instance_id,
        "instance_id": rendered.instance_id,
        "formatter": rendered.formatter,
        "complexity": rendered.complexity,
        "prefix": rendered.prefix,
        "completion": rendered.completion,
        "sensitivity_state_count": rendered.sensitivity_state_count,
        "sensitivity_output_count": rendered.sensitivity_output_count,
        "sensitivity_density": rendered.sensitivity_density,
        "sensitivity_coverage": rendered.sensitivity_coverage,
        "mean_live_fanout": rendered.mean_live_fanout,
    }
    record.update(
        _document_fields(
            example_id=example_id,
            prefix=rendered.prefix,
            completion=rendered.completion,
        )
    )
    return record


def _document_fields(*, example_id: str, prefix: str, completion: str) -> dict[str, str]:
    """Return the standard document fields shared by training-data consumers."""

    return {
        "id": example_id,
        "text": f"{prefix}\n{completion}",
        "source": "state-bench",
    }


def _assignment_range_item(spec: AssignmentRange) -> AssignmentPlanItem:
    return AssignmentPlanItem(
        min_extra_assignments=spec.min,
        max_extra_assignments=spec.max,
        target_extra_assignments=None,
    )


def hub_config_name(pattern: str, *, formatter: str, complexity: str) -> str:
    return pattern.format(formatter=formatter, complexity=complexity)


def _semantics_dir(root: Path) -> Path:
    return root / "semantics"


def _persist_split_semantics(root: Path, split_name: str, instance_ids: set[str]) -> None:
    """Write a split's semantic IDs so later independent builds can exclude them."""

    directory = _semantics_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split_name}.ids"
    with path.open("w", encoding="utf-8") as handle:
        for instance_id in sorted(instance_ids):
            handle.write(f"{instance_id}\n")


def _load_persisted_semantics(root: Path, split_name: str) -> set[str]:
    """Load a prior build's persisted semantic IDs for a disjoint_from split."""

    path = _semantics_dir(root) / f"{split_name}.ids"
    if not path.exists():
        raise FileNotFoundError(
            f"split {split_name!r} must be generated before a split that is disjoint from it, "
            f"but no persisted semantics were found at {path}. Build {split_name!r} into this "
            f"output directory first (for example `build <config> --splits {split_name}`)."
        )
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


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
