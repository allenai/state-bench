from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from state_bench.stats import DEFAULT_TOKENIZER
from state_bench.token_histogram import (
    analyze_token_lengths,
    format_histogram_report,
    main,
)


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
            "formatter": "integer-code",
            "complexity": "r-trivial",
        },
        {
            "prefix": "one two three four five",
            "completion": "six seven",
            "formatter": "integer-code",
            "complexity": "aperiodic",
        },
        {
            "prefix": " ".join(["token"] * 9),
            "completion": "ten",
            "formatter": "status-lights",
            "complexity": "periodic",
        },
    ]


def test_log_histogram_reports_summary_and_groups() -> None:
    report = analyze_token_lengths(
        _records(),
        tokenizer=WhitespaceTokenizer(),
        group_by=("complexity",),
    )
    assert report["rows"] == 3
    assert report["histograms"]["all"]["histogram"] == {
        "2-3": 1,
        "4-7": 1,
        "8-15": 1,
    }
    assert report["histograms"]["complexity=periodic"]["max"] == 9
    assert "Histogram: complexity=aperiodic" in format_histogram_report(report)


def test_linear_combined_histogram_honors_joiner_and_special_tokens() -> None:
    report = analyze_token_lengths(
        _records()[:1],
        tokenizer=WhitespaceTokenizer(),
        field="prefix+completion",
        joiner=" ",
        add_special_tokens=True,
        binning="linear",
        bin_width=5,
    )
    assert report["histograms"]["all"]["min"] == 5
    assert report["histograms"]["all"]["histogram"] == {"5-9": 1}


def test_cli_defaults_to_olmo_3_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "rendered.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in _records()) + "\n", encoding="utf-8")
    requested: list[str] = []

    def fake_load_tokenizer(name: str, *, revision: str | None = None) -> WhitespaceTokenizer:
        requested.append(name)
        return WhitespaceTokenizer()

    monkeypatch.setattr("state_bench.token_histogram.load_tokenizer", fake_load_tokenizer)
    assert main([str(path)]) == 0
    assert requested == [DEFAULT_TOKENIZER]
    assert "Histogram: all" in capsys.readouterr().out
