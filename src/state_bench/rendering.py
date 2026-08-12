"""Render canonical traces using declarative formatter specifications."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache
from string import Template
from typing import Any

from state_bench.config import FillerDistraction, FormatterSpec, TemplateValue
from state_bench.model import AssignConstant, AssignVariable, StateTrackingInstance
from state_bench.semantics import (
    DependencySummary,
    SensitivitySummary,
    dependency_summary,
    sensitivity_summary,
)


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
    surface_variables: tuple[str, ...]
    temporary_variable: str | None
    surface_values: tuple[str, ...]
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
    sensitivity_state_count: int
    sensitivity_output_count: int
    sensitivity_density: float
    sensitivity_coverage: float
    mean_live_fanout: float

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
            "surface_variables": list(self.surface_variables),
            "temporary_variable": self.temporary_variable,
            "surface_values": list(self.surface_values),
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
            "sensitivity_state_count": self.sensitivity_state_count,
            "sensitivity_output_count": self.sensitivity_output_count,
            "sensitivity_density": self.sensitivity_density,
            "sensitivity_coverage": self.sensitivity_coverage,
            "mean_live_fanout": self.mean_live_fanout,
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
    include_diagnostics: bool = True,
    dependencies: DependencySummary | None = None,
    sensitivity: SensitivitySummary | None = None,
) -> RenderedExample:
    """Render an instance without recomputing or changing its semantics.

    ``dependencies`` and ``sensitivity`` depend only on ``instance``; callers
    that render one instance in several formatters (or also need them for a
    canonical record) can compute them once and pass them in to avoid repeating
    the traversal per render.
    """

    _validate_capacity(instance, formatter)
    surface_variables, surface_values = _surface_vocabulary(instance, formatter, rng)
    metadata = {
        "num_variables": str(instance.num_variables),
        "num_values": str(instance.num_values),
        "num_assignments": str(len(instance.operations)),
        "complexity": instance.complexity.value,
        "query_count": str(len(instance.query)),
        "query_variables": _natural_list(
            [surface_variables[variable] for variable in instance.query]
        ),
    }

    # The set of legal placeholders is fixed across a render, so build each once
    # here. The value dicts start from the shared metadata and are mutated in
    # place per operation, avoiding a fresh metadata merge on every assignment.
    metadata_keys = frozenset(metadata)
    constant_allowed = metadata_keys | {"target", "value"}
    reference_allowed = metadata_keys | {"target", "source"}
    constant_values = dict(metadata)
    reference_values = dict(metadata)

    instructions: list[str] = []
    for operation in instance.operations:
        if isinstance(operation, AssignConstant):
            template = _choose(formatter.templates.assign_constant, rng)
            constant_values["target"] = surface_variables[operation.target]
            constant_values["value"] = surface_values[operation.value]
            instructions.append(_substitute(template, constant_allowed, constant_values))
        elif isinstance(operation, AssignVariable):
            template = _choose(formatter.templates.assign_reference, rng)
            reference_values["target"] = surface_variables[operation.target]
            reference_values["source"] = surface_variables[operation.source]
            instructions.append(_substitute(template, reference_allowed, reference_values))

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
            filler_width=_mean_line_width(instructions),
            surface_variables=surface_variables,
            surface_values=surface_values,
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
        surface_variables=surface_variables,
        surface_values=surface_values,
    )
    if include_diagnostics:
        incorrect_completions = tuple(
            _render_completion(
                instance=instance,
                state=state,
                formatter=formatter,
                assertion_template=assertion_template,
                metadata=metadata,
                surface_variables=surface_variables,
                surface_values=surface_values,
            )
            for state in incorrect_states
        )
        choices = [completion, *incorrect_completions]
        if shuffle_choices:
            rng.shuffle(choices)
        correct_choice_index = choices.index(completion)
    else:
        incorrect_completions = ()
        choices = []
        correct_choice_index = 0
    if formatter.layout.trailing_newline:
        prefix += "\n"
        completion += "\n"
        incorrect_completions = tuple(value + "\n" for value in incorrect_completions)
        choices = [value + "\n" for value in choices]

    if include_diagnostics and dependencies is None:
        dependencies = dependency_summary(instance)
    elif not include_diagnostics:
        dependencies = None
    if sensitivity is None:
        sensitivity = sensitivity_summary(instance)
    return RenderedExample(
        instance_id=instance.instance_id,
        formatter=formatter.name,
        formatter_config_hash=formatter.config_hash,
        prefix=prefix,
        completion=completion,
        complexity=instance.complexity.value,
        num_variables=instance.num_variables,
        num_values=instance.num_values,
        surface_variables=surface_variables,
        temporary_variable=(
            surface_variables[instance.temporary_variable]
            if instance.temporary_variable is not None
            else None
        ),
        surface_values=surface_values,
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
        final_dependency_lengths=(dependencies.final_dependency_lengths if dependencies else ()),
        max_dependency_length=(dependencies.max_dependency_length if dependencies else 0),
        mean_dependency_length=(dependencies.mean_dependency_length if dependencies else 0.0),
        modal_dependency_length=(dependencies.modal_dependency_length if dependencies else 0),
        sensitivity_state_count=sensitivity.sensitivity_state_count,
        sensitivity_output_count=sensitivity.sensitivity_output_count,
        sensitivity_density=sensitivity.sensitivity_density,
        sensitivity_coverage=sensitivity.sensitivity_coverage,
        mean_live_fanout=sensitivity.mean_live_fanout,
    )


def _content_line_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


def _mean_line_width(lines: list[str]) -> int:
    if not lines:
        return 1
    return max(1, round(sum(len(line) for line in lines) / len(lines)))


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
    filler_width: int,
    surface_variables: tuple[str, ...],
    surface_values: tuple[str, ...],
    rng: random.Random,
) -> str:
    template_value = getattr(formatter.templates, f"filler_{level}")
    if template_value is None:
        raise ValueError(f"formatter {formatter.name!r} has no filler_{level} template")
    corpus_lines = (
        _corpus_filler_lines(
            formatter.filler_corpus,
            num_lines=num_filler_lines,
            target_width=filler_width,
            rng=rng,
        )
        if formatter.filler_corpus is not None
        else [""] * num_filler_lines
    )
    if (
        isinstance(template_value, str)
        and template_value == "$filler_text"
        and formatter.filler_corpus is not None
    ):
        filler_lines = corpus_lines
    elif (
        isinstance(template_value, str)
        and set(Template(template_value).get_identifiers()) == {"filler_dots"}
    ):
        filler_line = Template(template_value).substitute(filler_dots="." * filler_width)
        filler_lines = [filler_line] * num_filler_lines
    else:
        filler_allowed = frozenset(metadata) | {
            "variable",
            "value",
            "filler_index",
            "filler_dots",
            "filler_text",
        }
        filler_values = dict(metadata)
        filler_lines = []
        for index in range(num_filler_lines):
            # Preserve the original RNG draw order: template choice, then the
            # variable and value picks.
            chosen = _choose(template_value, rng)
            filler_values["variable"] = rng.choice(surface_variables)
            filler_values["value"] = rng.choice(surface_values)
            filler_values["filler_index"] = str(index + 1)
            filler_values["filler_dots"] = "." * filler_width
            filler_values["filler_text"] = corpus_lines[index]
            filler_lines.append(_substitute(chosen, filler_allowed, filler_values))
    total_lines = len(instructions) + num_filler_lines
    instruction_positions = set(rng.sample(range(total_lines), len(instructions)))
    instruction_iterator = iter(instructions)
    filler_iterator = iter(filler_lines)
    return "\n".join(
        next(instruction_iterator) if position in instruction_positions else next(filler_iterator)
        for position in range(total_lines)
    )


def _corpus_filler_lines(
    corpus: str,
    *,
    num_lines: int,
    target_width: int,
    rng: random.Random,
) -> list[str]:
    words = _corpus_words(corpus)
    if not words:
        raise ValueError("filler corpus must contain at least one word")

    word_index = rng.randrange(len(words))
    lines: list[str] = []
    for _ in range(num_lines):
        line_words: list[str] = []
        line_width = 0
        while True:
            word = words[word_index]
            next_width = line_width + len(word) + bool(line_words)
            if line_words and next_width > target_width:
                break
            line_words.append(word)
            line_width = next_width
            word_index = (word_index + 1) % len(words)
        lines.append(" ".join(line_words))
    return lines


@cache
def _corpus_words(corpus: str) -> tuple[str, ...]:
    return tuple(corpus.split())


def _natural_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


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
    return _substitute(_choose(value, rng), frozenset(metadata), metadata)


def _substitute(template: str, allowed: frozenset[str], values: dict[str, str]) -> str:
    # FormatterSpec validates every template once while loading configuration.
    # Cache the defensive runtime check for programmatically constructed specs.
    # ``allowed`` is a hashable frozenset, so the cache keys on it directly.
    # ``values`` may be reused across calls (only read here), which lets the
    # per-operation render loop avoid rebuilding the shared metadata each time.
    _validate_runtime_template(template, allowed)
    try:
        return "".join(
            literal + (values[identifier] if identifier is not None else "")
            for literal, identifier in _template_parts(template)
        )
    except KeyError as error:
        raise ValueError(f"invalid template: {template!r}") from error


@cache
def _validate_runtime_template(template: str, allowed: frozenset[str]) -> None:
    identifiers = {
        identifier
        for _, identifier in _template_parts(template)
        if identifier is not None
    }
    unknown = identifiers - allowed
    if unknown:
        raise ValueError(f"unknown template placeholders: {sorted(unknown)}")


@cache
def _template_parts(template: str) -> tuple[tuple[str, str | None], ...]:
    """Parse a ``string.Template`` once for allocation-light substitutions."""

    parts: list[tuple[str, str | None]] = []
    cursor = 0
    for match in Template.pattern.finditer(template):
        literal = template[cursor : match.start()]
        identifier = match.group("named") or match.group("braced")
        if match.group("escaped") is not None:
            literal += "$"
        elif match.group("invalid") is not None:
            raise ValueError(f"invalid template: {template!r}")
        parts.append((literal, identifier))
        cursor = match.end()
    parts.append((template[cursor:], None))
    return tuple(parts)


def _validate_capacity(instance: StateTrackingInstance, formatter: FormatterSpec) -> None:
    if len(formatter.vocabulary.variables) < instance.num_variables:
        raise ValueError(f"formatter {formatter.name!r} has too few variables")
    if (
        formatter.vocabulary.value_selection == "prefix"
        and len(formatter.vocabulary.values) < instance.num_values
    ):
        raise ValueError(f"formatter {formatter.name!r} has too few values")


def _surface_vocabulary(
    instance: StateTrackingInstance,
    formatter: FormatterSpec,
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if formatter.vocabulary.variable_selection == "sample-without-replacement":
        variables = tuple(rng.sample(formatter.vocabulary.variables, instance.num_variables))
    else:
        variables = tuple(formatter.vocabulary.variables[: instance.num_variables])
    if instance.temporary_variable is not None:
        variables = (*variables, formatter.vocabulary.temporary_variable)

    if formatter.vocabulary.value_selection == "random-seven-digit":
        values = tuple(
            str(value)
            for value in rng.sample(range(1_000_000, 10_000_000), instance.num_values)
        )
    else:
        values = tuple(formatter.vocabulary.values[: instance.num_values])
    return variables, values


def _render_completion(
    *,
    instance: StateTrackingInstance,
    state: tuple[int, ...],
    formatter: FormatterSpec,
    assertion_template: str,
    metadata: dict[str, str],
    surface_variables: tuple[str, ...],
    surface_values: tuple[str, ...],
) -> str:
    completion_allowed = frozenset(metadata) | {"variable", "value"}
    completion_values = dict(metadata)
    assertions: list[str] = []
    for variable in instance.query:
        completion_values["variable"] = surface_variables[variable]
        completion_values["value"] = surface_values[state[variable]]
        assertions.append(_substitute(assertion_template, completion_allowed, completion_values))
    return formatter.layout.completion_separator.join(assertions)
