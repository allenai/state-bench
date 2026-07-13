from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from state_bench.cli import main as cli_main
from state_bench.stats import analyze_records, format_report, iter_rendered_records


class WhitespaceTokenizer:
    name_or_path = "test/whitespace"

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, Any]:
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(text.split()) + extra) for text in texts]}


def _records() -> list[dict[str, Any]]:
    return [
        {
            "prefix": "one two",
            "completion": "three",
            "choices": ["three", "four five"],
            "formatter": "integer-code",
            "complexity": "r-trivial",
            "num_variables": 2,
            "num_assignments": 2,
        },
        {
            "prefix": "one two three four",
            "completion": "five six",
            "choices": ["five six", "seven eight nine"],
            "formatter": "integer-code",
            "complexity": "aperiodic",
            "num_variables": 2,
            "num_assignments": 3,
        },
    ]


def test_analyze_records_reports_token_distributions_and_groups() -> None:
    report = analyze_records(
        _records(),
        tokenizer=WhitespaceTokenizer(),
        joiner=" ",
        histogram_bin_width=2,
        group_by=("complexity",),
    )
    assert report["rows"] == 2
    assert report["tokenizer"] == "test/whitespace"
    assert report["field_counts"]["complexity"] == {"aperiodic": 1, "r-trivial": 1}
    assert report["token_distributions"]["prefix_tokens"]["mean"] == 3.0
    assert report["token_distributions"]["completion_tokens"]["p50"] == 1.5
    assert report["token_distributions"]["prefix_plus_completion_tokens"]["max"] == 6
    assert report["token_distributions"]["choice_tokens"]["count"] == 4
    assert "complexity=aperiodic" in report["groups"]
    assert "prefix_tokens distribution:" in format_report(report)


def test_special_tokens_are_counted_when_requested() -> None:
    report = analyze_records(
        _records()[:1],
        tokenizer=WhitespaceTokenizer(),
        add_special_tokens=True,
    )
    assert report["token_distributions"]["prefix_tokens"]["min"] == 4
    assert report["token_distributions"]["completion_tokens"]["min"] == 3


def test_jsonl_reader_skips_canonical_rows_and_honors_limit(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [{"operations": []}, *_records()]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert list(iter_rendered_records(path, limit=1)) == [_records()[0]]


def test_empty_rendered_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="no rendered records"):
        analyze_records([], tokenizer=WhitespaceTokenizer())


def test_stats_cli_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    report = analyze_records(_records(), tokenizer=WhitespaceTokenizer())
    monkeypatch.setattr("state_bench.cli.analyze_path", lambda *args, **kwargs: report)
    assert cli_main(["stats", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 2
