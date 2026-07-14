from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from state_bench.config import FormatterSpec, load_experiment, load_formatter
from state_bench.model import AssignConstant, AssignVariable, Complexity, StateTrackingInstance
from state_bench.rendering import render_instance

ROOT = Path(__file__).resolve().parents[1]


def test_default_experiment_loads_and_validates_capacities() -> None:
    loaded = load_experiment(ROOT / "configs/experiments/default.yaml")
    assert loaded.spec.schema_version == 1
    assert [formatter.name for formatter in loaded.formatters] == [
        "integer-code",
        "cube-painting",
        "spreadsheet-cells",
        "status-lights",
        "people-in-rooms",
    ]


def test_integer_code_golden_render() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=2,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1), AssignVariable(0, 1)),
        query=(0, 1),
        final_state=(1, 1),
    )
    formatter = load_formatter(ROOT / "configs/formats/integer-code.yaml")
    rendered = render_instance(instance, formatter, rng=random.Random(0))
    assert rendered.prefix == (
        "Track the current value of each variable after executing the assignments.\n\n"
        "a <- 1; b <- 2; a <- b;"
    )
    assert rendered.completion == "a ?= 2; b ?= 2;"


def test_filler_lines_hit_requested_ratio_without_changing_program_order() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.R_TRIVIAL,
        num_variables=2,
        num_values=2,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1)),
        query=(0, 1),
        final_state=(0, 1),
    )
    formatter = load_formatter(ROOT / "configs/formats/integer-code.yaml")
    rendered = render_instance(
        instance,
        formatter,
        rng=random.Random(4),
        filler_proportion=0.5,
        filler_distraction="hard",
    )
    assert rendered.num_filler_lines == 3
    assert rendered.actual_filler_proportion == 0.5
    assert rendered.filler_distraction == "hard"
    assert rendered.completion == "a ?= 1; b ?= 2;"
    assert rendered.prefix.index("a <- 1;") < rendered.prefix.index("b <- 2;")
    assert rendered.prefix.count("// stale log; do not execute:") == 3


def test_cube_golden_render() -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=2,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1), AssignVariable(0, 1)),
        query=(0, 1),
        final_state=(1, 1),
    )
    formatter = load_formatter(ROOT / "configs/formats/cube-painting.yaml")
    rendered = render_instance(instance, formatter, rng=random.Random(0))
    assert "Paint cube 1 the same color as cube 2." in rendered.prefix
    assert rendered.completion == "cube 1 is orange. cube 2 is orange."


@pytest.mark.parametrize(
    ("formatter_name", "reference_text", "completion"),
    [
        (
            "spreadsheet-cells",
            "Copy the current contents of cell B1 into cell A1.",
            "cell A1 contains 2. cell B1 contains 2.",
        ),
        (
            "status-lights",
            "Set light A to the color currently displayed by light B.",
            "light A displays orange. light B displays orange.",
        ),
        (
            "people-in-rooms",
            "Alice moves to Bob's current location.",
            "Alice is in the garden. Bob is in the garden.",
        ),
    ],
)
def test_additional_natural_language_formatters(
    formatter_name: str,
    reference_text: str,
    completion: str,
) -> None:
    instance = StateTrackingInstance.create(
        complexity=Complexity.PERIODIC,
        num_variables=2,
        num_values=2,
        operations=(AssignConstant(0, 0), AssignConstant(1, 1), AssignVariable(0, 1)),
        query=(0, 1),
        final_state=(1, 1),
    )
    formatter = load_formatter(ROOT / f"configs/formats/{formatter_name}.yaml")
    rendered = render_instance(instance, formatter, rng=random.Random(0))
    assert reference_text in rendered.prefix
    assert rendered.completion == completion


def test_unknown_formatter_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FormatterSpec.model_validate(
            {
                "schema_version": 1,
                "name": "bad",
                "renderer": "template",
                "vocabulary": {"variables": ["a"], "values": ["1"]},
                "templates": {
                    "assign_constant": "$target=$value",
                    "assign_reference": "$target=$source",
                    "state_assertion": "$variable=$value",
                },
                "unknown": True,
            }
        )


def test_unknown_placeholder_is_rejected_at_render_time() -> None:
    formatter = load_formatter(ROOT / "configs/formats/integer-code.yaml")
    invalid = formatter.model_copy(
        update={"templates": formatter.templates.model_copy(update={"assign_constant": "$mystery"})}
    )
    instance = StateTrackingInstance.create(
        complexity=Complexity.R_TRIVIAL,
        num_variables=1,
        num_values=1,
        operations=(AssignConstant(0, 0),),
        query=(0,),
        final_state=(0,),
    )
    with pytest.raises(ValueError, match="unknown template placeholders"):
        render_instance(instance, invalid, rng=random.Random(0))


def test_unknown_placeholder_is_rejected_when_parsing_config() -> None:
    formatter = load_formatter(ROOT / "configs/formats/integer-code.yaml")
    record = formatter.model_dump()
    record["templates"]["assign_constant"] = "$unknown"
    with pytest.raises(ValidationError, match="unknown placeholders"):
        FormatterSpec.model_validate(record)
