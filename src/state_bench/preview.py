"""Generate human-readable previews without writing a dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from state_bench.config import ConditionSpec, FormatterSpec, LoadedExperiment
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import generate_instance
from state_bench.io import derive_seed, hub_config_name
from state_bench.model import StateTrackingInstance
from state_bench.rendering import RenderedExample, render_instance


@dataclass(frozen=True, slots=True)
class SamplePreview:
    config_name: str
    seed: int
    instance: StateTrackingInstance
    rendered: RenderedExample

    def to_record(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "seed": self.seed,
            "canonical": self.instance.to_record(),
            "rendered": self.rendered.to_record(),
        }


def available_config_names(loaded: LoadedExperiment) -> tuple[str, ...]:
    """Return the rendered Hugging Face config names defined by an experiment."""

    names = {
        hub_config_name(
            loaded.spec.huggingface.config_naming,
            formatter=formatter.name,
            complexity=condition.complexity.value,
        )
        for formatter in loaded.formatters
        for condition in loaded.spec.generation.conditions
    }
    return tuple(sorted(names))


def generate_preview(
    loaded: LoadedExperiment,
    *,
    config_name: str,
    seed: int | None = None,
    num_variables: int | None = None,
) -> SamplePreview:
    """Generate one sample for a rendered config without creating output files."""

    resolved_seed = loaded.spec.dataset.seed if seed is None else seed
    formatter, conditions = _resolve_config(loaded, config_name)
    candidates = [
        (condition, candidate_num_variables)
        for condition in conditions
        for candidate_num_variables in condition.num_variables
        if num_variables is None or candidate_num_variables == num_variables
    ]
    if not candidates:
        supported = sorted(
            {candidate for condition in conditions for candidate in condition.num_variables}
        )
        raise ValueError(
            f"config {config_name!r} does not support num_variables={num_variables}; "
            f"choose one of {supported}"
        )

    selection_rng = random.Random(derive_seed(resolved_seed, "preview", config_name, "selection"))
    condition, selected_num_variables = selection_rng.choice(candidates)
    instance = _generate(
        loaded,
        condition=condition,
        num_variables=selected_num_variables,
        seed=derive_seed(resolved_seed, "preview", config_name, "semantic"),
    )
    incorrect_states = generate_incorrect_states(
        instance,
        num_distractors=loaded.spec.output.multiple_choice.num_distractors,
        rng=random.Random(derive_seed(resolved_seed, "preview", config_name, "distractors")),
    )
    rendered = render_instance(
        instance,
        formatter,
        rng=random.Random(derive_seed(resolved_seed, "preview", config_name, "surface")),
        incorrect_states=incorrect_states,
        shuffle_choices=loaded.spec.output.multiple_choice.shuffle_choices,
    )
    return SamplePreview(
        config_name=config_name,
        seed=resolved_seed,
        instance=instance,
        rendered=rendered,
    )


def format_preview(preview: SamplePreview) -> str:
    """Format a preview for terminal display."""

    operations = "\n".join(
        f"  {index + 1}. {operation}"
        for index, operation in enumerate(_canonical_operations(preview.instance))
    )
    choices = "\n".join(
        f"  {index}. {choice}" for index, choice in enumerate(preview.rendered.choices)
    )
    return (
        f"Config: {preview.config_name}\n"
        f"Seed: {preview.seed}\n"
        f"Complexity: {preview.instance.complexity.value}\n"
        f"Variables: {preview.instance.num_variables}\n"
        f"Assignments: {len(preview.instance.operations)}\n\n"
        f"Canonical trace:\n{operations}\n\n"
        f"Prefix:\n{preview.rendered.prefix}\n\n"
        f"Completion:\n{preview.rendered.completion}\n\n"
        f"Multiple-choice options:\n{choices}\n"
        f"Correct choice index: {preview.rendered.correct_choice_index}"
    )


def _resolve_config(
    loaded: LoadedExperiment,
    config_name: str,
) -> tuple[FormatterSpec, list[ConditionSpec]]:
    matches: list[tuple[FormatterSpec, list[ConditionSpec]]] = []
    for formatter in loaded.formatters:
        grouped: dict[str, list[ConditionSpec]] = {}
        for condition in loaded.spec.generation.conditions:
            name = hub_config_name(
                loaded.spec.huggingface.config_naming,
                formatter=formatter.name,
                complexity=condition.complexity.value,
            )
            grouped.setdefault(name, []).append(condition)
        if config_name in grouped:
            matches.append((formatter, grouped[config_name]))

    if not matches:
        available = ", ".join(available_config_names(loaded))
        raise ValueError(f"unknown config {config_name!r}; available configs: {available}")
    if len(matches) > 1:
        raise ValueError(f"config naming pattern is ambiguous for {config_name!r}")
    return matches[0]


def _generate(
    loaded: LoadedExperiment,
    *,
    condition: ConditionSpec,
    num_variables: int,
    seed: int,
) -> StateTrackingInstance:
    generation = loaded.spec.generation
    return generate_instance(
        complexity=condition.complexity,
        num_variables=num_variables,
        num_values=condition.num_values,
        min_extra_assignments=condition.extra_assignments.min,
        max_extra_assignments=condition.extra_assignments.max,
        query_mode=generation.query_mode,
        distinct_initial_values=generation.distinct_initial_values,
        require_strict_witness=generation.require_strict_witness,
        allow_self_reference=generation.allow_self_reference,
        rng=random.Random(seed),
    )


def _canonical_operations(instance: StateTrackingInstance) -> list[str]:
    operations = []
    for operation in instance.operations:
        record = operation.to_record()
        if record["op"] == "constant":
            operations.append(f"variable[{record['target']}] <- value[{record['value']}]")
        else:
            operations.append(f"variable[{record['target']}] <- variable[{record['source']}]")
    return operations
