from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from state_bench.cli import main
from state_bench.config import load_experiment
from state_bench.hub import (
    push_hub_repository,
    stage_hub_repository,
    validate_hub_repository,
)
from state_bench.io import build_dataset, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _small_experiment(tmp_path: Path) -> Path:
    source = yaml.safe_load((ROOT / "configs/experiments/default.yaml").read_text(encoding="utf-8"))
    source["dataset"]["output_dir"] = str(tmp_path / "output")
    source["splits"] = {"train": {"samples_per_condition": 2}, "test": {"samples_per_condition": 1}}
    source["output"]["shard_size"] = 1
    source["generation"]["conditions"] = [
        {
            "complexity": "periodic",
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 1, "max": 2},
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
    assert len(canonical) == len(integer) == len(cubes) == 2
    assert [row["example_id"] for row in canonical] == [row["example_id"] for row in integer]
    assert [row["instance_id"] for row in integer] == [row["instance_id"] for row in cubes]
    assert json.loads((output / "manifest.json").read_text())["counts"]["canonical/train"] == 2


def test_cli_config_check_and_build(tmp_path: Path, capsys: object) -> None:
    experiment = _small_experiment(tmp_path)
    assert main(["config", "check", str(experiment)]) == 0
    assert main(["build", str(experiment)]) == 0
    assert (tmp_path / "output/canonical/train.jsonl").is_file()


def test_hub_push_dry_run_needs_no_optional_dependencies(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    plan = push_hub_repository(
        tmp_path,
        repo_id="example/state-bench",
        private=True,
        dry_run=True,
    )
    assert plan["repo_id"] == "example/state-bench"
    assert plan["files"] == ["README.md"]


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
    assert report["files"] == 9
    assert report["counts"]["canonical/train"] == 2
    assert "configs:" in (repo_dir / "README.md").read_text(encoding="utf-8")
    canonical_rows = parquet.read_table(
        repo_dir / "data/canonical/train-00000-of-00002.parquet"
    ).to_pylist()
    assert canonical_rows[0]["operations"][-1]["source"] is not None


def test_mixed_complexities_have_stable_parquet_schema_across_shards(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    experiment = _small_experiment(tmp_path)
    source = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    source["generation"]["conditions"] = [
        {
            "complexity": complexity,
            "num_variables": [2],
            "num_values": 10,
            "extra_assignments": {"min": 0 if complexity == "r-trivial" else 1, "max": 1},
        }
        for complexity in ("r-trivial", "aperiodic", "periodic")
    ]
    source["generation"]["conditions"][0]["extra_assignments"]["max"] = 0
    experiment.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    loaded = load_experiment(experiment)
    output = build_dataset(loaded)
    repo_dir = stage_hub_repository(loaded, source_dir=output, repo_dir=tmp_path / "mixed-hub")
    report = validate_hub_repository(repo_dir)
    assert report["counts"]["canonical/train"] == 6
