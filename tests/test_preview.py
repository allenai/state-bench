from __future__ import annotations

import json
from pathlib import Path

import pytest

from state_bench.cli import main
from state_bench.config import load_experiment
from state_bench.preview import available_config_names, generate_preview

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/default.yaml"


def test_preview_is_deterministic_and_uses_requested_config() -> None:
    loaded = load_experiment(EXPERIMENT)
    first = generate_preview(
        loaded,
        config_name="integer-code--periodic",
        seed=7,
        num_variables=4,
    )
    second = generate_preview(
        loaded,
        config_name="integer-code--periodic",
        seed=7,
        num_variables=4,
    )
    assert first == second
    assert first.rendered.formatter == "integer-code"
    assert first.instance.complexity.value == "periodic"
    assert first.instance.num_variables == 4
    assert len(first.rendered.incorrect_completions) == 4
    assert len(first.rendered.choices) == 5
    assert first.rendered.choices[first.rendered.correct_choice_index] == first.rendered.completion


def test_available_config_names_cover_each_formatter_and_complexity() -> None:
    names = available_config_names(load_experiment(EXPERIMENT))
    assert len(names) == 6
    assert "cube-painting--r-trivial" in names
    assert "integer-code--periodic" in names


def test_sample_cli_prints_human_readable_preview(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", str(EXPERIMENT), "cube-painting--aperiodic", "--seed", "3"]) == 0
    output = capsys.readouterr().out
    assert "Canonical trace:" in output
    assert "Prefix:" in output
    assert "Completion:" in output
    assert "Paint cube" in output


def test_sample_cli_can_print_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", str(EXPERIMENT), "integer-code--r-trivial", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["config_name"] == "integer-code--r-trivial"
    assert "prefix" in record["rendered"]


def test_unknown_config_reports_available_names() -> None:
    with pytest.raises(ValueError, match="available configs"):
        generate_preview(load_experiment(EXPERIMENT), config_name="missing")
