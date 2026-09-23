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

import orjson
from tqdm.auto import tqdm

from state_bench.config import (
    AssignmentRange,
    FormatterSpec,
    LoadedExperiment,
    condition_uses_formatter,
    condition_uses_split,
    split_families,
    split_uses_formatter,
)
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import DERIVED_FAMILIES, derive_variants, generate_periodic_source
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
# Full (non-minimal) chunks carry the richer per-row payload and may include very
# long sequences. Chunks are formed per (split, condition, variable-count) block,
# which holds relatively few sources, so keep the batch small enough to produce
# several chunks per worker for even load-balancing.
_FULL_SOURCE_CHUNK_SIZE = 16


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
    """All information needed to generate one deterministic source and its families."""

    source_example_id: str
    families: tuple[Complexity, ...]
    num_variables: int
    num_values: int
    min_extra_assignments: int
    max_extra_assignments: int
    query_mode: QueryMode
    distinct_initial_values: bool
    require_strict_witness: bool
    min_aperiodic_constants: int
    max_aperiodic_constants: int
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
    # Emitted instances, ordered like the task's families.
    families: tuple[Complexity, ...]
    instances: tuple[StateTrackingInstance, ...]
    # IDs of every derived family, emitted or not, for split disjointness.
    semantic_ids: tuple[str, ...]


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


@dataclass(frozen=True, slots=True)
class _EmitContext:
    """Everything needed to turn one generated sample into its output rows.

    Shared across a split's samples and picklable, so the same emission logic
    runs unchanged in the serial main loop and in the parallel chunk worker.
    """

    formatters: tuple[FormatterSpec, ...]
    config_naming: str
    split_name: str
    dataset_seed: int
    write_canonical: bool
    minimal_output: bool
    include_canonical_trace: bool
    include_final_state: bool
    track_unique: bool
    num_distractors: int
    shuffle_choices: bool


@dataclass(frozen=True, slots=True)
class _SampleEmission:
    """Serialized output rows for one source, plus the ids the main loop tracks.

    Each line is ``(target, json)`` where target is ``"canonical"`` or a rendered
    config name; ``unique_pairs`` is ``(count_key, instance_id)`` when unique
    semantics are tracked.
    """

    lines: tuple[tuple[str, str], ...]
    instance_ids: tuple[str, ...]
    unique_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _FullChunkTask:
    """A bounded source range fully rendered and serialized by one worker."""

    chunk_index: int
    samples: tuple[_GenerationTask, ...]
    context: _EmitContext
    chunk_dir: Path


@dataclass(frozen=True, slots=True)
class _FullChunkResult:
    chunk_index: int
    source_count: int
    files: tuple[tuple[str, Path, int], ...]
    instance_ids: tuple[str, ...]
    unique_pairs: tuple[tuple[str, str], ...]


def _generate_instances(
    task: _GenerationTask,
) -> _GeneratedSample:
    """Generate a source and its derived variants in an isolated worker.

    Disjointness is checked against every derived family, not only the emitted
    ones, so a build that emits a subset of families (for example only
    ``solvable``) regenerates exactly the sources of a build that emitted the
    others from the same experiment.
    """

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
            min_aperiodic_constants=task.min_aperiodic_constants,
            max_aperiodic_constants=task.max_aperiodic_constants,
            rng=random.Random(retry_seed),
        )
        variants = dict(zip(DERIVED_FAMILIES, derive_variants(source), strict=True))
        semantic_ids = tuple(instance.instance_id for instance in variants.values())
        if not (set(semantic_ids) & task.excluded_semantics):
            return _GeneratedSample(
                source_example_id=task.source_example_id,
                plan_item=task.plan_item,
                filler_item=task.filler_item,
                source=source,
                families=task.families,
                instances=tuple(variants[family] for family in task.families),
                semantic_ids=semantic_ids,
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
            for complexity, instance in zip(generated.families, generated.instances, strict=True):
                example_id = _derived_example_id(generated.source_example_id, complexity)
                # Sensitivity depends only on the instance; compute it once and
                # reuse it across every formatter rendered from this instance.
                sensitivity = sensitivity_summary(instance)
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
                        sensitivity=sensitivity,
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


def _serialize_line(record: Mapping[str, Any]) -> str:
    return orjson.dumps(record).decode()


def _emit_sample_records(generated: _GeneratedSample, context: _EmitContext) -> _SampleEmission:
    """Build every output row for one generated source (canonical + rendered).

    This is the single source of truth for non-minimal-chunk emission; the serial
    main loop and the parallel chunk worker both call it, so their output is
    identical by construction.
    """

    source_example_id = generated.source_example_id
    plan_item = generated.plan_item
    filler_item = generated.filler_item
    source = generated.source
    lines: list[tuple[str, str]] = []
    unique_pairs: list[tuple[str, str]] = []
    needs_canonical_record = context.write_canonical or not context.minimal_output
    for complexity, instance in zip(generated.families, generated.instances, strict=True):
        example_id = _derived_example_id(source_example_id, complexity)
        sensitivity = sensitivity_summary(instance)
        dependencies = (
            dependency_summary(instance)
            if needs_canonical_record or not context.minimal_output
            else None
        )
        canonical_record: dict[str, Any] | None = None
        if needs_canonical_record:
            assert dependencies is not None
            canonical_record = {
                "example_id": example_id,
                "source_example_id": source_example_id,
                "source_instance_id": source.instance_id,
                **instance.to_record(),
                **dependencies.to_record(),
                **sensitivity.to_record(),
                "num_extra_assignments": (
                    len(instance.operations) - instance.num_initial_assignments
                ),
                "target_extra_assignments": plan_item.target_extra_assignments,
                "filler_proportion": filler_item.proportion,
                "filler_distraction": filler_item.distraction,
            }
        if context.write_canonical:
            assert canonical_record is not None
            lines.append(("canonical", _serialize_line(canonical_record)))
        if context.track_unique:
            unique_pairs.append((f"canonical/{context.split_name}", instance.instance_id))

        include_trace = not context.minimal_output and context.include_canonical_trace
        source_operations = (
            [operation.to_record() for operation in source.operations] if include_trace else None
        )
        incorrect_states = (
            ()
            if context.minimal_output
            else generate_incorrect_states(
                instance,
                num_distractors=context.num_distractors,
                rng=random.Random(derive_seed(context.dataset_seed, "distractors", example_id)),
            )
        )

        for formatter in context.formatters:
            config_name = hub_config_name(
                context.config_naming, formatter=formatter.name, complexity=complexity.value
            )
            surface_seed = derive_seed(
                context.dataset_seed, "surface", source_example_id, formatter.name
            )
            rendered = render_instance(
                instance,
                formatter,
                rng=random.Random(surface_seed),
                incorrect_states=incorrect_states,
                shuffle_choices=context.shuffle_choices,
                filler_proportion=filler_item.proportion,
                filler_distraction=filler_item.distraction,
                include_diagnostics=not context.minimal_output,
                dependencies=dependencies,
                sensitivity=sensitivity,
            )
            rendered_record: dict[str, Any]
            if context.minimal_output:
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
            if not context.minimal_output and context.include_final_state:
                assert canonical_record is not None
                rendered_record["final_state"] = canonical_record["final_state"]
            # Keep the training payload alongside the evaluation metadata so raw
            # rendered JSONL is directly consumable by Dolma and HF JSON.
            rendered_record.update(
                _document_fields(
                    example_id=example_id,
                    prefix=rendered.prefix,
                    completion=rendered.completion,
                )
            )
            lines.append((config_name, _serialize_line(rendered_record)))
            if context.track_unique:
                unique_pairs.append((f"{config_name}/{context.split_name}", instance.instance_id))

    return _SampleEmission(tuple(lines), generated.semantic_ids, tuple(unique_pairs))


def _write_full_chunk(task: _FullChunkTask) -> _FullChunkResult:
    """Generate, render, serialize, and write one non-minimal chunk in a worker."""

    handles: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    counts: Counter[str] = Counter()
    instance_ids: list[str] = []
    unique_pairs: list[tuple[str, str]] = []
    try:
        for sample_task in task.samples:
            emission = _emit_sample_records(_generate_instances(sample_task), task.context)
            for target, line in emission.lines:
                handle = handles.get(target)
                if handle is None:
                    path = task.chunk_dir / target / f"{task.chunk_index:08d}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    paths[target] = path
                    handle = path.open("w", encoding="utf-8")
                    handles[target] = handle
                handle.write(line)
                handle.write("\n")
                counts[target] += 1
            instance_ids.extend(emission.instance_ids)
            unique_pairs.extend(emission.unique_pairs)
    finally:
        for handle in handles.values():
            handle.close()
    return _FullChunkResult(
        chunk_index=task.chunk_index,
        source_count=len(task.samples),
        files=tuple((target, paths[target], counts[target]) for target in sorted(paths)),
        instance_ids=tuple(instance_ids),
        unique_pairs=tuple(unique_pairs),
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
        split.samples_per_condition
        * sum(
            len(split_families(spec, split_name)) * len(condition.num_variables)
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
            if not any(
                condition_uses_formatter(condition, formatter.name)
                for condition in spec.generation.conditions
            ):
                continue
            config_names = {
                complexity: hub_config_name(
                    spec.huggingface.config_naming,
                    formatter=formatter.name,
                    complexity=complexity.value,
                )
                for complexity in spec.generation.emitted_families()
            }
            for complexity, config_name in sorted(config_names.items(), key=lambda item: item[1]):
                for split, split_spec in spec.splits.items():
                    if not selected(split):
                        continue
                    if complexity not in split_families(spec, split):
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
            families = split_families(spec, split_name)
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
                family_count = len(families)
                for num_variables in condition.num_variables:
                    progress.set_postfix(
                        split=split_name,
                        source=condition.complexity.value,
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
                                complexity=f"{condition.complexity.value}-source",
                                num_variables=num_variables,
                                num_values=condition.num_values,
                                condition_index=condition_index,
                                sample_index=sample_index,
                                target_extra_assignments=plan_item.target_extra_assignments,
                                filler_proportion=filler_item.proportion,
                                filler_distraction=filler_item.distraction,
                            ),
                            families=families,
                            num_variables=num_variables,
                            num_values=condition.num_values,
                            min_extra_assignments=plan_item.min_extra_assignments,
                            max_extra_assignments=plan_item.max_extra_assignments,
                            query_mode=spec.generation.query_mode,
                            distinct_initial_values=spec.generation.distinct_initial_values,
                            require_strict_witness=spec.generation.require_strict_witness,
                            min_aperiodic_constants=spec.generation.aperiodic_constants.min,
                            max_aperiodic_constants=spec.generation.aperiodic_constants.max,
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
                            progress.update(family_count * result.source_count)
                        continue
                    emit_context = _EmitContext(
                        formatters=tuple(
                            formatter
                            for formatter in loaded.formatters
                            if condition_uses_formatter(condition, formatter.name)
                            and split_uses_formatter(split_spec, formatter.name)
                        ),
                        config_naming=spec.huggingface.config_naming,
                        split_name=split_name,
                        dataset_seed=spec.dataset.seed,
                        write_canonical=split_name in canonical_handles,
                        minimal_output=split_name in spec.output.minimal_splits,
                        include_canonical_trace=spec.output.include_canonical_trace,
                        include_final_state=spec.output.include_final_state,
                        track_unique=spec.output.track_unique_semantics,
                        num_distractors=spec.output.multiple_choice.num_distractors,
                        shuffle_choices=spec.output.multiple_choice.shuffle_choices,
                    )
                    # Retain only semantics that a later split needs for disjointness;
                    # keeping every training ID is prohibitive for large builds.
                    track_split_semantics = (
                        spec.output.track_unique_semantics or split_name in referenced_splits
                    )

                    def _handle_for(target: str, split: str = split_name) -> Any:
                        if target == "canonical":
                            return canonical_handles[split]
                        return rendered_handles[(target, split)]

                    if executor is not None:
                        # Render and serialize in workers, writing per-chunk files
                        # concatenated here; only compact ids travel back over IPC.
                        if chunk_root is None:
                            chunk_root = Path(tempfile.mkdtemp(prefix=".state-bench-chunks-", dir=root))
                        full_chunk_tasks = (
                            _FullChunkTask(
                                chunk_index=chunk_index,
                                samples=tuple(sample_batch),
                                context=emit_context,
                                chunk_dir=chunk_root,
                            )
                            for chunk_index, sample_batch in enumerate(
                                batched(tasks, _FULL_SOURCE_CHUNK_SIZE)
                            )
                        )
                        chunk_results = _executor_map(
                            executor,
                            _write_full_chunk,
                            full_chunk_tasks,
                            chunksize=1,
                            buffersize=max(1, workers * 2),
                        )
                        for result in chunk_results:
                            for target, chunk_path, row_count in result.files:
                                with chunk_path.open(encoding="utf-8") as chunk_handle:
                                    shutil.copyfileobj(chunk_handle, _handle_for(target))
                                chunk_path.unlink()
                                count_key = (
                                    f"canonical/{split_name}"
                                    if target == "canonical"
                                    else f"{target}/{split_name}"
                                )
                                counts[count_key] += row_count
                            if track_split_semantics:
                                split_semantics[split_name].update(result.instance_ids)
                            for unique_key, instance_id in result.unique_pairs:
                                unique_semantics[unique_key].add(instance_id)
                            progress.update(family_count * result.source_count)
                    else:
                        for generated in map(_generate_instances, tasks):
                            emission = _emit_sample_records(generated, emit_context)
                            for target, line in emission.lines:
                                _write_line(_handle_for(target), line)
                                count_key = (
                                    f"canonical/{split_name}"
                                    if target == "canonical"
                                    else f"{target}/{split_name}"
                                )
                                counts[count_key] += 1
                            if track_split_semantics:
                                split_semantics[split_name].update(emission.instance_ids)
                            for unique_key, instance_id in emission.unique_pairs:
                                unique_semantics[unique_key].add(instance_id)
                            progress.update(family_count)
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
    # orjson emits the same compact UTF-8 JSON as json.dumps(ensure_ascii=False,
    # separators=(",", ":")) — including shortest-round-trip floats — but several
    # times faster, which dominates the write cost for long-sequence rows.
    _write_line(handle, _serialize_line(record))


def _write_line(handle: Any, line: str) -> None:
    handle.write(line)
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
