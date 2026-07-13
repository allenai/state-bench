"""Render canonical traces using declarative formatter specifications."""

from __future__ import annotations

import random
from dataclasses import dataclass
from string import Template
from typing import Any

from state_bench.config import FormatterSpec, TemplateValue
from state_bench.model import AssignConstant, AssignVariable, StateTrackingInstance


@dataclass(frozen=True, slots=True)
class RenderedExample:
    instance_id: str
    formatter: str
    formatter_config_hash: str
    prefix: str
    completion: str
    complexity: str
    num_variables: int
    num_values: int
    num_assignments: int
    incorrect_completions: tuple[str, ...]
    choices: tuple[str, ...]
    correct_choice_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "formatter": self.formatter,
            "formatter_config_hash": self.formatter_config_hash,
            "prefix": self.prefix,
            "completion": self.completion,
            "complexity": self.complexity,
            "num_variables": self.num_variables,
            "num_values": self.num_values,
            "num_assignments": self.num_assignments,
            "incorrect_completions": list(self.incorrect_completions),
            "choices": list(self.choices),
            "correct_choice_index": self.correct_choice_index,
        }


def render_instance(
    instance: StateTrackingInstance,
    formatter: FormatterSpec,
    *,
    rng: random.Random,
    incorrect_states: tuple[tuple[int, ...], ...] = (),
    shuffle_choices: bool = True,
) -> RenderedExample:
    """Render an instance without recomputing or changing its semantics."""

    _validate_capacity(instance, formatter)
    metadata = {
        "num_variables": str(instance.num_variables),
        "num_values": str(instance.num_values),
        "num_assignments": str(len(instance.operations)),
        "complexity": instance.complexity.value,
    }

    instructions: list[str] = []
    for operation in instance.operations:
        if isinstance(operation, AssignConstant):
            template = _choose(formatter.templates.assign_constant, rng)
            instructions.append(
                _substitute(
                    template,
                    allowed={"target", "value", *metadata},
                    target=formatter.vocabulary.variables[operation.target],
                    value=formatter.vocabulary.values[operation.value],
                    **metadata,
                )
            )
        elif isinstance(operation, AssignVariable):
            template = _choose(formatter.templates.assign_reference, rng)
            instructions.append(
                _substitute(
                    template,
                    allowed={"target", "source", *metadata},
                    target=formatter.vocabulary.variables[operation.target],
                    source=formatter.vocabulary.variables[operation.source],
                    **metadata,
                )
            )

    components = {
        "task_description": _optional_render(formatter.templates.task_description, metadata, rng),
        "program": formatter.layout.instruction_separator.join(instructions),
        "answer_prompt": _optional_render(formatter.templates.answer_prompt, metadata, rng),
    }
    prefix = formatter.layout.block_separator.join(
        components[block] for block in formatter.layout.prompt_blocks if components[block]
    )

    assertion_template = _choose(formatter.templates.state_assertion, rng)
    completion = _render_completion(
        instance=instance,
        state=instance.final_state,
        formatter=formatter,
        assertion_template=assertion_template,
        metadata=metadata,
    )
    incorrect_completions = tuple(
        _render_completion(
            instance=instance,
            state=state,
            formatter=formatter,
            assertion_template=assertion_template,
            metadata=metadata,
        )
        for state in incorrect_states
    )
    choices = [completion, *incorrect_completions]
    if shuffle_choices:
        rng.shuffle(choices)
    correct_choice_index = choices.index(completion)
    if formatter.layout.trailing_newline:
        prefix += "\n"
        completion += "\n"
        incorrect_completions = tuple(value + "\n" for value in incorrect_completions)
        choices = [value + "\n" for value in choices]

    return RenderedExample(
        instance_id=instance.instance_id,
        formatter=formatter.name,
        formatter_config_hash=formatter.config_hash,
        prefix=prefix,
        completion=completion,
        complexity=instance.complexity.value,
        num_variables=instance.num_variables,
        num_values=instance.num_values,
        num_assignments=len(instance.operations),
        incorrect_completions=incorrect_completions,
        choices=tuple(choices),
        correct_choice_index=correct_choice_index,
    )


def _choose(value: TemplateValue, rng: random.Random) -> str:
    if isinstance(value, str):
        return value
    choices = value.choices
    return rng.choices(
        [choice.text for choice in choices],
        weights=[choice.weight for choice in choices],
        k=1,
    )[0]


def _optional_render(
    value: TemplateValue | None,
    metadata: dict[str, str],
    rng: random.Random,
) -> str:
    if value is None:
        return ""
    return _substitute(_choose(value, rng), allowed=set(metadata), **metadata)


def _substitute(template: str, *, allowed: set[str], **values: str) -> str:
    identifiers = {
        match.group("named") or match.group("braced")
        for match in Template.pattern.finditer(template)
        if match.group("named") or match.group("braced")
    }
    unknown = identifiers - allowed
    if unknown:
        raise ValueError(f"unknown template placeholders: {sorted(unknown)}")
    try:
        return Template(template).substitute(values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid template: {template!r}") from error


def _validate_capacity(instance: StateTrackingInstance, formatter: FormatterSpec) -> None:
    if len(formatter.vocabulary.variables) < instance.num_variables:
        raise ValueError(f"formatter {formatter.name!r} has too few variables")
    if len(formatter.vocabulary.values) < instance.num_values:
        raise ValueError(f"formatter {formatter.name!r} has too few values")


def _render_completion(
    *,
    instance: StateTrackingInstance,
    state: tuple[int, ...],
    formatter: FormatterSpec,
    assertion_template: str,
    metadata: dict[str, str],
) -> str:
    assertions = [
        _substitute(
            assertion_template,
            allowed={"variable", "value", *metadata},
            variable=formatter.vocabulary.variables[variable],
            value=formatter.vocabulary.values[state[variable]],
            **metadata,
        )
        for variable in instance.query
    ]
    return formatter.layout.completion_separator.join(assertions)
