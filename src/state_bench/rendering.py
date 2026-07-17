"""Render canonical traces using declarative formatter specifications."""

from __future__ import annotations

import random
from dataclasses import dataclass
from string import Template
from typing import Any

from state_bench.config import FillerDistraction, FormatterSpec, TemplateValue
from state_bench.model import AssignConstant, AssignVariable, StateTrackingInstance
from state_bench.semantics import dependency_summary


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
    filler_proportion: float
    actual_filler_proportion: float
    filler_distraction: str | None
    num_filler_lines: int
    final_dependency_lengths: tuple[int, ...]
    max_dependency_length: int
    mean_dependency_length: float
    modal_dependency_length: int

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
            "filler_proportion": self.filler_proportion,
            "actual_filler_proportion": self.actual_filler_proportion,
            "filler_distraction": self.filler_distraction,
            "num_filler_lines": self.num_filler_lines,
            "final_dependency_lengths": list(self.final_dependency_lengths),
            "max_dependency_length": self.max_dependency_length,
            "mean_dependency_length": self.mean_dependency_length,
            "modal_dependency_length": self.modal_dependency_length,
        }


def render_instance(
    instance: StateTrackingInstance,
    formatter: FormatterSpec,
    *,
    rng: random.Random,
    incorrect_states: tuple[tuple[int, ...], ...] = (),
    shuffle_choices: bool = True,
    filler_proportion: float = 0.0,
    filler_distraction: FillerDistraction | None = None,
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

    task_description = _optional_render(formatter.templates.task_description, metadata, rng)
    answer_prompt = _optional_render(formatter.templates.answer_prompt, metadata, rng)
    num_non_filler_lines = (
        _content_line_count(task_description) + len(instructions) + _content_line_count(answer_prompt)
    )
    num_filler_lines = _filler_line_count(
        proportion=filler_proportion,
        num_non_filler_lines=num_non_filler_lines,
    )
    if num_filler_lines:
        if filler_distraction is None:
            raise ValueError("filler_distraction is required when filler_proportion is positive")
        program = _render_program_with_filler(
            instructions,
            formatter=formatter,
            metadata=metadata,
            level=filler_distraction,
            num_filler_lines=num_filler_lines,
            rng=rng,
        )
    else:
        program = formatter.layout.instruction_separator.join(instructions)
    components = {
        "task_description": task_description,
        "program": program,
        "answer_prompt": answer_prompt,
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

    dependencies = dependency_summary(instance)
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
        filler_proportion=filler_proportion,
        actual_filler_proportion=(
            num_filler_lines / (num_non_filler_lines + num_filler_lines)
            if num_filler_lines
            else 0.0
        ),
        filler_distraction=filler_distraction,
        num_filler_lines=num_filler_lines,
        final_dependency_lengths=dependencies.final_dependency_lengths,
        max_dependency_length=dependencies.max_dependency_length,
        mean_dependency_length=dependencies.mean_dependency_length,
        modal_dependency_length=dependencies.modal_dependency_length,
    )


def _content_line_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


def _filler_line_count(*, proportion: float, num_non_filler_lines: int) -> int:
    if proportion < 0 or proportion >= 1:
        raise ValueError("filler_proportion must satisfy 0 <= proportion < 1")
    if proportion == 0:
        return 0
    return max(1, round(num_non_filler_lines * proportion / (1 - proportion)))


def _render_program_with_filler(
    instructions: list[str],
    *,
    formatter: FormatterSpec,
    metadata: dict[str, str],
    level: FillerDistraction,
    num_filler_lines: int,
    rng: random.Random,
) -> str:
    template_value = getattr(formatter.templates, f"filler_{level}")
    if template_value is None:
        raise ValueError(f"formatter {formatter.name!r} has no filler_{level} template")
    filler_lines = [
        _substitute(
            _choose(template_value, rng),
            allowed={"variable", "value", "filler_index", *metadata},
            variable=rng.choice(formatter.vocabulary.variables[: int(metadata["num_variables"])]),
            value=rng.choice(formatter.vocabulary.values[: int(metadata["num_values"])]),
            filler_index=str(index + 1),
            **metadata,
        )
        for index in range(num_filler_lines)
    ]
    total_lines = len(instructions) + num_filler_lines
    instruction_positions = set(rng.sample(range(total_lines), len(instructions)))
    instruction_iterator = iter(instructions)
    filler_iterator = iter(filler_lines)
    return "\n".join(
        next(instruction_iterator) if position in instruction_positions else next(filler_iterator)
        for position in range(total_lines)
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
