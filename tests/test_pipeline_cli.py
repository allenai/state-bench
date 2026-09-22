from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from state_bench import io as benchmark_io
from state_bench.cli import _default_repo_dir, main
from state_bench.config import TokenStratificationSpec, load_experiment
from state_bench.hub import (
    _calibrate_token_length_estimator,
    _character_bounded_batches,
    _ParquetShardWriter,
    _partition_records_by_token_stratum,
    push_hub_repository,
    stage_hub_repository,
    validate_hub_repository,
)
from state_bench.io import build_dataset, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class FakeLengthTokenizer:
    name_or_path = "fake-length-tokenizer"

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, object]:
        del padding, truncation
        offset = int(add_special_tokens)
        return {"input_ids": [[0] * (len(text) + offset) for text in texts]}


class RecordingProgress:
    def __init__(self, *, total: int, **kwargs: object) -> None:
        del kwargs
        self.total = total
        self.n = 0.0
        self.closed = False

    def set_postfix(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def update(self, amount: float = 1) -> None:
        self.n += amount

    def close(self) -> None:
        self.closed = True


class BackendLengthTokenizer(FakeLengthTokenizer):
    class Backend:
        class Encoding:
            def __init__(self, length: int) -> None:
                self.length = length

            def __len__(self) -> int:
                return self.length

        def encode_batch(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[Encoding]:
            offset = int(add_special_tokens)
            return [self.Encoding(len(text) + offset) for text in texts]

    backend_tokenizer = Backend()

    def __call__(self, *args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise AssertionError("the Python token-ID path should not be used")


class CountingLengthTokenizer(FakeLengthTokenizer):
    def __init__(self) -> None:
        self.tokenized_texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, object]:
        self.tokenized_texts.extend(texts)
        return super().__call__(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
        )


def _small_experiment(tmp_path: Path) -> Path:
    source = yaml.safe_load((ROOT / "configs/experiments/default.yaml").read_text(encoding="utf-8"))
    source["dataset"]["output_dir"] = str(tmp_path / "output")
    source["splits"] = {"train": {"samples_per_condition": 2}, "test": {"samples_per_condition": 1}}
    # default.yaml is the full long-context experiment; reset the output and Hub
    # blocks to a plain, non-minimal base so each test opts into the behavior it
    # exercises.
    source["output"] = {
        "include_canonical_trace": True,
        "include_final_state": True,
        "shard_size": 1,
        "multiple_choice": {"num_distractors": 4, "shuffle_choices": True},
    }
    source["huggingface"] = {
        "include_canonical_config": True,
        "config_naming": "{formatter}--{complexity}",
        "staging_workers": 2,
        "repository": {"repo_id": None, "private": True},
    }
    source["generation"]["conditions"] = [
        {
            "complexity": "periodic",
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 3, "max": 4},
        }
    ]
    source["formatters"] = [
        {"config": str(ROOT / "configs/formats/integer-code.yaml")},
        {"config": str(ROOT / "configs/formats/cube-painting.yaml")},
    ]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def test_build_preserves_ids_across_formatters(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    output = build_dataset(load_experiment(experiment))
    canonical = read_jsonl(output / "canonical/train.jsonl")
    integer = read_jsonl(output / "rendered/integer-code--periodic/train.jsonl")
    cubes = read_jsonl(output / "rendered/cube-painting--periodic/train.jsonl")
    assert len(canonical) == 6
    assert len(integer) == len(cubes) == 2
    periodic_canonical = [row for row in canonical if row["complexity"] == "periodic"]
    assert [row["example_id"] for row in periodic_canonical] == [
        row["example_id"] for row in integer
    ]
    assert [row["instance_id"] for row in integer] == [row["instance_id"] for row in cubes]
    for canonical_row, rendered_row in zip(periodic_canonical, integer, strict=True):
        assert rendered_row["final_dependency_lengths"] == canonical_row["final_dependency_lengths"]
        assert rendered_row["max_dependency_length"] == canonical_row["max_dependency_length"]
        assert rendered_row["mean_dependency_length"] == canonical_row["mean_dependency_length"]
        assert rendered_row["modal_dependency_length"] == canonical_row["modal_dependency_length"]
        assert rendered_row["sensitivity_density"] == canonical_row["sensitivity_density"]
        assert rendered_row["sensitivity_coverage"] == canonical_row["sensitivity_coverage"]
        assert rendered_row["mean_live_fanout"] == canonical_row["mean_live_fanout"]
    for row in (*integer, *cubes):
        # Two real variables have one non-identity permutation once TMP is held
        # fixed; substitutions of trace-seen values fill the remaining slots so
        # every complexity family presents the same number of choices.
        assert len(row["incorrect_completions"]) == 4
        assert len(set(row["incorrect_completions"])) == 4
        assert row["completion"] not in row["incorrect_completions"]
        assert len(row["choices"]) == 5
        assert row["choices"][row["correct_choice_index"]] == row["completion"]
    source_groups = {}
    for row in canonical:
        source_groups.setdefault(row["source_example_id"], []).append(row)
    assert len(source_groups) == 2
    assert all(
        {row["complexity"] for row in rows} == {"periodic", "aperiodic", "r-trivial"}
        for rows in source_groups.values()
    )
    assert all(
        len({len(row["operations"]) for row in rows}) == 1
        for rows in source_groups.values()
    )
    assert json.loads((output / "manifest.json").read_text())["counts"]["canonical/train"] == 6


def test_parallel_generation_preserves_rendered_bytes(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    loaded = load_experiment(experiment)
    serial = build_dataset(loaded, output_dir=tmp_path / "serial", workers=1)
    try:
        parallel = build_dataset(loaded, output_dir=tmp_path / "parallel", workers=2)
    except PermissionError as error:
        pytest.skip(f"process pools are unavailable in this sandbox: {error}")
    serial_files = {
        str(path.relative_to(serial)): path.read_bytes()
        for path in serial.rglob("*.jsonl")
    }
    parallel_files = {
        str(path.relative_to(parallel)): path.read_bytes()
        for path in parallel.rglob("*.jsonl")
    }
    assert parallel_files == serial_files


def test_build_defaults_to_cpus_minus_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    monkeypatch.setattr("state_bench.io._available_cpu_count", lambda: 5)
    created_workers: list[int] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            created_workers.append(max_workers)

        def map(
            self,
            function: object,
            items: object,
            *,
            chunksize: int,
            buffersize: int,
        ) -> object:
            del chunksize, buffersize
            return map(function, items)  # ty: ignore[invalid-argument-type]

        def shutdown(self, *, cancel_futures: bool = False) -> None:
            del cancel_futures

    monkeypatch.setattr("state_bench.io.ProcessPoolExecutor", RecordingExecutor)
    build_dataset(load_experiment(experiment))
    assert created_workers == [4]


def test_available_cpu_count_falls_back_on_python_before_313(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr("state_bench.io.os.process_cpu_count", raising=False)
    monkeypatch.setattr("state_bench.io.os.sched_getaffinity", lambda _: {1, 3, 7}, raising=False)

    assert benchmark_io._available_cpu_count() == 3


def test_executor_map_backport_bounds_input_consumption() -> None:
    consumed: list[int] = []

    class ImmediateFuture:
        def __init__(self, value: tuple[int, ...]) -> None:
            self.value = value

        def result(self) -> tuple[int, ...]:
            return self.value

    class Pre314Executor:
        def map(self, function: object, items: object, *, chunksize: int) -> object:
            raise AssertionError("the eager pre-3.14 map must not be called")

        def submit(
            self,
            function: object,
            mapped_function: object,
            batch: tuple[int, ...],
        ) -> ImmediateFuture:
            return ImmediateFuture(function(mapped_function, batch))  # ty: ignore[call-non-callable]

    def items() -> Iterator[int]:
        for value in range(20):
            consumed.append(value)
            yield value

    results = benchmark_io._executor_map(
        Pre314Executor(),  # ty: ignore[invalid-argument-type]
        lambda value: value * 2,
        items(),
        chunksize=2,
        buffersize=2,
    )
    assert consumed == []
    assert next(results) == 0
    assert consumed == [0, 1, 2, 3]
    assert list(results) == [value * 2 for value in range(1, 20)]


def test_minimal_split_skips_distractors_and_dependency_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"train": {"samples_per_condition": 2}}
    source["output"]["minimal_splits"] = ["train"]
    source["output"]["omit_canonical_splits"] = ["train"]
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("minimal rendering should skip diagnostics")

    monkeypatch.setattr("state_bench.io.generate_incorrect_states", fail)
    monkeypatch.setattr("state_bench.rendering.dependency_summary", fail)
    output = build_dataset(load_experiment(experiment), workers=1)
    rows = read_jsonl(output / "rendered/integer-code--periodic/train.jsonl")
    assert rows
    assert set(rows[0]) == {
        "example_id",
        "source_example_id",
        "source_instance_id",
        "instance_id",
        "formatter",
        "complexity",
        "prefix",
        "completion",
        "id",
        "text",
        "source",
        "sensitivity_state_count",
        "sensitivity_output_count",
        "sensitivity_density",
        "sensitivity_coverage",
        "mean_live_fanout",
    }
    assert rows[0]["id"] == rows[0]["example_id"]
    assert rows[0]["text"] == f"{rows[0]['prefix']}\n{rows[0]['completion']}"
    assert rows[0]["source"] == "state-bench"
    assert not (output / "canonical/train.jsonl").exists()


def test_parallel_minimal_chunks_preserve_bytes_and_cleanup(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"train": {"samples_per_condition": 20}}
    source["output"]["minimal_splits"] = ["train"]
    source["output"]["omit_canonical_splits"] = ["train"]
    source["output"]["track_unique_semantics"] = False
    source["output"]["hash_files"] = False
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    loaded = load_experiment(experiment)
    serial = build_dataset(loaded, output_dir=tmp_path / "minimal-serial", workers=1)
    parallel = build_dataset(loaded, output_dir=tmp_path / "minimal-parallel", workers=2)
    serial_files = {
        str(path.relative_to(serial)): path.read_bytes()
        for path in serial.rglob("*.jsonl")
    }
    parallel_files = {
        str(path.relative_to(parallel)): path.read_bytes()
        for path in parallel.rglob("*.jsonl")
    }
    assert parallel_files == serial_files
    assert not list(parallel.glob(".state-bench-chunks-*"))


def test_split_scoping_and_disjoint_generation_preserve_existing_test_bytes(
    tmp_path: Path,
) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"test": {"samples_per_condition": 2}}
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    baseline = build_dataset(
        load_experiment(experiment),
        output_dir=tmp_path / "test-only",
    )
    baseline_test_files = {
        str(path.relative_to(baseline)): path.read_bytes() for path in baseline.rglob("test.jsonl")
    }

    source["splits"]["train"] = {
        "samples_per_condition": 2,
        "formatters": ["integer-code"],
        "disjoint_from": ["test"],
    }
    source["generation"]["conditions"][0]["splits"] = ["test"]
    source["generation"]["conditions"].append(
        {
            "complexity": "periodic",
            "splits": ["train"],
            "formatters": ["integer-code"],
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 3, "max": 3},
            "filler": {
                "proportions": [0.0, 0.5],
                "distraction_levels": ["easy"],
            },
        }
    )
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    combined = build_dataset(
        load_experiment(experiment),
        output_dir=tmp_path / "test-and-train",
    )

    assert {
        str(path.relative_to(combined)): path.read_bytes() for path in combined.rglob("test.jsonl")
    } == baseline_test_files
    assert (combined / "rendered/integer-code--periodic/train.jsonl").is_file()
    assert not (combined / "rendered/cube-painting--periodic/train.jsonl").exists()
    train_rows = read_jsonl(combined / "rendered/integer-code--periodic/train.jsonl")
    assert {row["filler_distraction"] for row in train_rows} == {None, "easy"}
    assert {row["num_extra_assignments"] for row in train_rows} == {3}

    test_semantics = {row["instance_id"] for row in read_jsonl(combined / "canonical/test.jsonl")}
    train_semantics = {row["instance_id"] for row in read_jsonl(combined / "canonical/train.jsonl")}
    assert test_semantics.isdisjoint(train_semantics)
    manifest = json.loads((combined / "manifest.json").read_text())
    assert manifest["cross_split_semantic_overlap"] == {"test--train": 0}
    assert "cube-painting--periodic/train" not in manifest["counts"]


def _test_and_train_experiment(tmp_path: Path) -> Path:
    """A two-split experiment: test in every formatter, train in integer-code only."""

    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {
        "test": {"samples_per_condition": 2},
        "train": {
            "samples_per_condition": 2,
            "formatters": ["integer-code"],
            "disjoint_from": ["test"],
        },
    }
    source["generation"]["conditions"][0]["splits"] = ["test"]
    source["generation"]["conditions"].append(
        {
            "complexity": "periodic",
            "splits": ["train"],
            "formatters": ["integer-code"],
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 3, "max": 3},
            "filler": {"proportions": [0.0], "distraction_levels": ["easy"]},
        }
    )
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return experiment


def test_independent_split_builds_share_disjointness_via_persisted_ids(
    tmp_path: Path,
) -> None:
    experiment = _test_and_train_experiment(tmp_path)
    output = tmp_path / "shared-output"

    # Build the test split alone; its semantic IDs are persisted for later runs.
    build_dataset(load_experiment(experiment), output_dir=output, only_splits=frozenset({"test"}))
    assert (output / "semantics" / "test.ids").is_file()
    assert not (output / "canonical" / "train.jsonl").exists()
    test_bytes = {
        str(path.relative_to(output)): path.read_bytes() for path in output.rglob("test.jsonl")
    }

    # Build the train split independently into the same directory; disjointness
    # is enforced from the persisted test IDs, not from an in-memory set.
    build_dataset(load_experiment(experiment), output_dir=output, only_splits=frozenset({"train"}))

    assert {
        str(path.relative_to(output)): path.read_bytes() for path in output.rglob("test.jsonl")
    } == test_bytes, "an independent train build must not touch existing test rows"

    test_semantics = {row["instance_id"] for row in read_jsonl(output / "canonical/test.jsonl")}
    train_semantics = {row["instance_id"] for row in read_jsonl(output / "canonical/train.jsonl")}
    assert train_semantics
    assert test_semantics.isdisjoint(train_semantics)

    # The merged manifest keeps the record for both splits.
    manifest = json.loads((output / "manifest.json").read_text())
    assert any(key.endswith("/test") for key in manifest["counts"])
    assert "integer-code--periodic/train" in manifest["counts"]


def test_disjoint_build_without_persisted_ids_is_rejected(tmp_path: Path) -> None:
    experiment = _test_and_train_experiment(tmp_path)
    with pytest.raises(FileNotFoundError, match="must be generated before"):
        build_dataset(
            load_experiment(experiment),
            output_dir=tmp_path / "train-first",
            only_splits=frozenset({"train"}),
        )


def test_unknown_requested_split_is_rejected(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    with pytest.raises(ValueError, match="unknown splits requested"):
        build_dataset(load_experiment(experiment), only_splits=frozenset({"nonexistent"}))


def test_padded_length_sweep_appends_rows_without_changing_existing_test_rows(
    tmp_path: Path,
) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"test": {"samples_per_condition": 6}}
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    baseline = build_dataset(
        load_experiment(experiment),
        output_dir=tmp_path / "unpadded-test",
    )
    baseline_files = {
        str(path.relative_to(baseline)): path.read_bytes()
        for path in baseline.rglob("test.jsonl")
    }

    source["generation"]["conditions"].append(
        {
            "complexity": "periodic",
            "splits": ["test"],
            "formatters": ["integer-code"],
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {
                "distribution": "log-spaced",
                "min": 3,
                "max": 6,
                "min_samples_per_bucket": 3,
            },
            "filler": {
                "proportions": [0.25],
                "distraction_levels": ["easy", "medium", "hard"],
                "stratify_by_assignment_bucket": True,
            },
        }
    )
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    combined = build_dataset(
        load_experiment(experiment),
        output_dir=tmp_path / "padded-test",
    )

    assert (combined / "canonical/test.jsonl").read_bytes().startswith(
        baseline_files["canonical/test.jsonl"]
    )
    for complexity in ("periodic", "aperiodic", "r-trivial"):
        integer_path = f"rendered/integer-code--{complexity}/test.jsonl"
        cube_path = f"rendered/cube-painting--{complexity}/test.jsonl"
        assert (combined / integer_path).read_bytes().startswith(baseline_files[integer_path])
        assert (combined / cube_path).read_bytes() == baseline_files[cube_path]

    periodic = read_jsonl(combined / "rendered/integer-code--periodic/test.jsonl")
    added = periodic[6:]
    assert len(added) == 6
    by_depth: dict[int, set[str]] = {}
    for row in added:
        by_depth.setdefault(row["target_extra_assignments"], set()).add(
            row["filler_distraction"]
        )
    assert by_depth == {
        3: {"easy", "medium", "hard"},
        6: {"easy", "medium", "hard"},
    }


def test_cli_config_check_and_build(tmp_path: Path, capsys: object) -> None:
    experiment = _small_experiment(tmp_path)
    assert main(["config", "check", str(experiment)]) == 0
    assert main(["build", str(experiment)]) == 0
    assert (tmp_path / "output/canonical/train.jsonl").is_file()


def test_build_progress_tracks_generated_examples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[RecordingProgress] = []

    def record_progress(**kwargs: object) -> RecordingProgress:
        progress = RecordingProgress(**kwargs)  # ty: ignore[invalid-argument-type]
        created.append(progress)
        return progress

    monkeypatch.setattr("state_bench.io.tqdm", record_progress)
    build_dataset(load_experiment(_small_experiment(tmp_path)))
    assert len(created) == 2
    assert created[0].total == 9
    assert created[0].n == 9
    assert created[0].closed
    assert created[1].n == created[1].total
    assert created[1].closed


def test_default_hub_repo_dir_mirrors_configured_data_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "data" / "state-tracking-long-context-v1"
    assert _default_repo_dir(source_dir) == (tmp_path / "build" / "state-tracking-long-context-v1")


def test_default_hub_repo_dir_uses_source_name_for_an_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _default_repo_dir(tmp_path / "custom-source") == tmp_path / "build" / "custom-source"


def test_hub_push_dry_run_needs_no_optional_dependencies(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    plan = push_hub_repository(
        tmp_path,
        repo_id="example/state-bench",
        private=True,
        dry_run=True,
    )
    assert plan["repo_id"] == "example/state-bench"
    assert plan["sync"] is False
    assert plan["delete_patterns"] == []
    assert plan["files"] == ["README.md"]


def test_hub_sync_dry_run_reports_remote_deletion_plan(tmp_path: Path) -> None:
    _write_complete_staged_repository(tmp_path)
    plan = push_hub_repository(
        tmp_path,
        repo_id="example/state-bench",
        private=True,
        dry_run=True,
        sync=True,
    )
    assert plan["sync"] is True
    assert plan["delete_patterns"] == ["*"]


def test_hub_sync_cli_wires_the_opt_in_flag(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_complete_staged_repository(tmp_path)
    manifest = {
        "huggingface": {
            "repository": {
                "repo_id": "example/state-bench",
                "private": True,
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["hub", "push", str(tmp_path), "--sync", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["sync"] is True
    assert plan["delete_patterns"] == ["*"]


def test_hub_sync_rejects_an_incomplete_staging_directory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    with pytest.raises(ValueError, match="complete staged repository"):
        push_hub_repository(
            tmp_path,
            repo_id="example/state-bench",
            private=True,
            dry_run=True,
            sync=True,
        )


def test_hub_sync_passes_delete_pattern_to_huggingface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_complete_staged_repository(tmp_path)
    calls: list[dict[str, object]] = []

    class FakeApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.append({"method": "create_repo", **kwargs})

        def upload_folder(self, **kwargs: object) -> None:
            calls.append({"method": "upload_folder", **kwargs})

    fake_module = ModuleType("huggingface_hub")
    fake_module.HfApi = FakeApi  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr("state_bench.hub._require_hub_dependencies", lambda: None)

    push_hub_repository(
        tmp_path,
        repo_id="example/state-bench",
        private=True,
        dry_run=False,
        sync=True,
    )
    upload_call = next(call for call in calls if call["method"] == "upload_folder")
    assert upload_call["delete_patterns"] == "*"


def test_stage_only_selected_splits(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    experiment = _small_experiment(tmp_path)
    loaded = load_experiment(experiment)
    output = build_dataset(loaded)
    repo_dir = stage_hub_repository(
        loaded,
        source_dir=output,
        repo_dir=tmp_path / "test-only-hub",
        only_splits=frozenset({"test"}),
    )
    report = validate_hub_repository(repo_dir)
    staged_splits = {key.rpartition("/")[2] for key in report["counts"]}
    assert staged_splits == {"test"}
    assert not list((repo_dir / "data").glob("*/train-*.parquet"))
    with pytest.raises(ValueError, match="only_splits must not be empty"):
        stage_hub_repository(
            loaded, source_dir=output, repo_dir=tmp_path / "empty-hub", only_splits=frozenset()
        )
    with pytest.raises(FileNotFoundError, match="no generated splits selected"):
        stage_hub_repository(
            loaded,
            source_dir=output,
            repo_dir=tmp_path / "missing-hub",
            only_splits=frozenset({"validation"}),
        )


def test_stage_and_validate_hub_repository(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    experiment = _small_experiment(tmp_path)
    loaded = load_experiment(experiment)
    output = build_dataset(loaded)
    repo_dir = stage_hub_repository(
        loaded,
        source_dir=output,
        repo_dir=tmp_path / "hub",
    )
    report = validate_hub_repository(repo_dir)
    assert report["files"] == 27
    assert report["counts"]["canonical/train"] == 6
    assert "configs:" in (repo_dir / "README.md").read_text(encoding="utf-8")
    canonical_rows = [
        row
        for path in sorted((repo_dir / "data/canonical").glob("train-*.parquet"))
        for row in parquet.read_table(path).to_pylist()
    ]
    periodic_row = next(row for row in canonical_rows if row["complexity"] == "periodic")
    assert any(operation["source"] is not None for operation in periodic_row["operations"])


def test_stage_overwrites_a_populated_repo_directory(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    loaded = load_experiment(_small_experiment(tmp_path))
    output = build_dataset(loaded)
    repo_dir = tmp_path / "hub"

    # A prior staging (plus an unrelated stale file) must not block a re-stage.
    stage_hub_repository(loaded, source_dir=output, repo_dir=repo_dir)
    stale = repo_dir / "data" / "leftover.parquet"
    stale.write_text("stale", encoding="utf-8")

    stage_hub_repository(loaded, source_dir=output, repo_dir=repo_dir)
    assert not stale.exists(), "re-staging must clear stale artifacts"
    assert validate_hub_repository(repo_dir)["files"] == 27


def test_stage_progress_tracks_rows_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow.parquet")
    loaded = load_experiment(_small_experiment(tmp_path))
    output = build_dataset(loaded)
    created: list[RecordingProgress] = []

    def record_progress(**kwargs: object) -> RecordingProgress:
        progress = RecordingProgress(**kwargs)  # ty: ignore[invalid-argument-type]
        created.append(progress)
        return progress

    monkeypatch.setattr("state_bench.hub.tqdm", record_progress)
    stage_hub_repository(loaded, source_dir=output, repo_dir=tmp_path / "progress-hub")
    assert len(created) == 2
    assert created[0].total == 27
    assert created[0].n == 27
    assert created[0].closed
    assert created[1].n == created[1].total
    assert created[1].closed


def test_token_strata_use_half_open_bounds_and_annotate_rows() -> None:
    spec = TokenStratificationSpec.model_validate(
        {
            "tokenizer": "fake",
            "ranges": [
                {"name": "short", "min_tokens": 0, "max_tokens": 5},
                {"name": "long", "min_tokens": 5, "max_tokens": None},
            ],
        }
    )
    partitioned = _partition_records_by_token_stratum(
        [
            {"example_id": "a", "prefix": "1234", "completion": "x"},
            {"example_id": "b", "prefix": "12345", "completion": "x"},
        ],
        spec=spec,
        tokenizer=FakeLengthTokenizer(),
    )
    assert [row["example_id"] for row in partitioned["short"]] == ["a"]
    assert [row["example_id"] for row in partitioned["long"]] == ["b"]
    assert partitioned["short"][0]["token_length"] == 4
    assert partitioned["long"][0]["token_stratum"] == "long"


def test_token_strata_use_backend_lengths_without_python_token_ids() -> None:
    spec = TokenStratificationSpec.model_validate(
        {
            "tokenizer": "fake",
            "add_special_tokens": True,
            "ranges": [
                {"name": "short", "min_tokens": 0, "max_tokens": 5},
                {"name": "long", "min_tokens": 5, "max_tokens": None},
            ],
        }
    )
    partitioned = _partition_records_by_token_stratum(
        [{"example_id": "a", "prefix": "1234", "completion": "x"}],
        spec=spec,
        tokenizer=BackendLengthTokenizer(),
    )
    assert partitioned["long"][0]["token_length"] == 5


def test_token_batches_respect_character_budget() -> None:
    spec = TokenStratificationSpec.model_validate(
        {
            "tokenizer": "fake",
            "batch_size": 4,
            "max_batch_characters": 5,
            "ranges": [{"name": "all", "min_tokens": 0, "max_tokens": None}],
        }
    )
    records = [
        ({"prefix": "123"}, 3),
        ({"prefix": "456"}, 3),
        ({"prefix": "78"}, 2),
    ]
    batches = list(_character_bounded_batches(records, spec=spec))
    assert [[item[0]["prefix"] for item in batch] for batch in batches] == [
        ["123"],
        ["456", "78"],
    ]


def test_sampled_token_estimator_only_tokenizes_calibration_quantiles(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    rows = [
        {"example_id": str(length), "prefix": "x" * length, "filler_distraction": None}
        for length in range(10, 110, 10)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    spec = TokenStratificationSpec.model_validate(
        {
            "tokenizer": "fake",
            "mode": "sampled",
            "calibration_scan_rows": 8,
            "calibration_samples_per_group": 4,
            "ranges": [{"name": "all", "min_tokens": 0, "max_tokens": None}],
        }
    )
    tokenizer = CountingLengthTokenizer()
    estimator = _calibrate_token_length_estimator(path, spec, tokenizer)
    assert len(tokenizer.tokenized_texts) == 4
    assert estimator.estimate(rows[-1], str(rows[-1]["prefix"])) == 100


def test_parquet_writer_flushes_at_estimated_byte_target(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    writer = _ParquetShardWriter(
        tmp_path,
        split="test",
        max_rows=100,
        target_bytes=5,
    )
    for index in range(3):
        writer.add(
            {"example_id": str(index), "instance_id": str(index)},
            estimated_bytes=4,
        )
    assert writer.finish()
    assert [path.name for path in sorted(tmp_path.glob("*.parquet"))] == [
        "test-00000-of-00003.parquet",
        "test-00001-of-00003.parquet",
        "test-00002-of-00003.parquet",
    ]


@pytest.mark.parametrize(
    ("mode", "is_estimate"),
    [("sampled", True), ("exact", False)],
)
def test_hub_stage_publishes_rendered_token_strata_as_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    is_estimate: bool,
) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["huggingface"]["token_strata"] = {
        "tokenizer": "fake",
        "mode": mode,
        "calibration_scan_rows": 2,
        "calibration_samples_per_group": 2,
        "ranges": [
            {"name": "short", "min_tokens": 0, "max_tokens": 5},
            {"name": "long", "min_tokens": 5, "max_tokens": None},
        ],
    }
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    loaded = load_experiment(experiment)
    output = build_dataset(loaded)
    for path in (output / "rendered").glob("*/*.jsonl"):
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            row["prefix"] = "1234" if index % 2 == 0 else "12345"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    monkeypatch.setattr("state_bench.hub.load_tokenizer", lambda *args, **kwargs: FakeLengthTokenizer())
    repo_dir = stage_hub_repository(
        loaded,
        source_dir=output,
        repo_dir=tmp_path / f"{mode}-stratified-hub",
    )
    report = validate_hub_repository(repo_dir)
    assert report["counts"]["integer-code--periodic/train__short"] == 1
    assert report["counts"]["integer-code--periodic/train__long"] == 1
    row = parquet.read_table(
        repo_dir / "data/integer-code--periodic/train__long-00000-of-00001.parquet"
    ).to_pylist()[0]
    assert row["token_length"] == 5
    assert row["token_length_is_estimate"] is is_estimate
    assert row["token_stratum"] == "long"


def test_mixed_complexities_have_stable_parquet_schema_across_shards(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["generation"]["conditions"] = [
        {
            "complexity": "periodic",
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 3, "max": 3},
        }
    ]
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    loaded = load_experiment(experiment)
    output = build_dataset(loaded)
    repo_dir = stage_hub_repository(loaded, source_dir=output, repo_dir=tmp_path / "mixed-hub")
    report = validate_hub_repository(repo_dir)
    assert report["counts"]["canonical/train"] == 6


def test_log_spaced_schedule_uses_fixed_budget_and_records_bucket(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"train": {"samples_per_condition": 8}}
    source["generation"]["conditions"][0]["extra_assignments"] = {
        "distribution": "log-spaced",
        "min": 3,
        "max": 24,
        "growth_factor": 2,
        "sample_decay": 0.5,
        "jitter_fraction": 0,
        "min_samples_per_bucket": 1,
    }
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    output = build_dataset(load_experiment(experiment))
    rows = read_jsonl(output / "canonical/train.jsonl")
    assert len(rows) == 24
    assert {row["target_extra_assignments"] for row in rows} == {3, 6, 12, 24}
    assert all(row["num_extra_assignments"] == row["target_extra_assignments"] for row in rows)


def test_filler_axes_share_budget_and_can_be_scoped_to_one_formatter(tmp_path: Path) -> None:
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["splits"] = {"test": {"samples_per_condition": 8}}
    source["generation"]["conditions"] = [
        {
            "complexity": "periodic",
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 3, "max": 3},
            "filler": {
                "proportions": [0.0, 0.5],
                "distraction_levels": ["easy", "medium", "hard"],
            },
            "formatters": ["integer-code"],
        }
    ]
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    output = build_dataset(load_experiment(experiment))
    canonical = read_jsonl(output / "canonical/test.jsonl")
    rendered = read_jsonl(output / "rendered/integer-code--r-trivial/test.jsonl")
    assert len(canonical) == 24
    assert len(rendered) == 8
    assert not (output / "rendered/cube-painting--r-trivial").exists()
    variants = {(row["filler_proportion"], row["filler_distraction"]) for row in rendered}
    assert variants == {
        (0.0, None),
        (0.5, "easy"),
        (0.5, "medium"),
        (0.5, "hard"),
    }
    assert all(row["actual_filler_proportion"] == row["filler_proportion"] for row in rendered)


def _write_complete_staged_repository(path: Path) -> None:
    (path / "README.md").write_text("card", encoding="utf-8")
    (path / "generation.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (path / "manifest.json").write_text("{}\n", encoding="utf-8")
    data = path / "data/canonical"
    data.mkdir(parents=True)
    (data / "test-00000-of-00001.parquet").write_bytes(b"parquet")
