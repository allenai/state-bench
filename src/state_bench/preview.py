"""Generate human-readable previews without writing a dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from state_bench.config import (
    ConditionSpec,
    FillerDistraction,
    FormatterSpec,
    LoadedExperiment,
    condition_uses_formatter,
    filler_variants,
)
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import generate_instance
from state_bench.io import derive_seed, hub_config_name
from state_bench.model import StateTrackingInstance
from state_bench.rendering import RenderedExample, render_instance
from state_bench.scheduling import (
    AssignmentPlanItem,
    FillerPlanItem,
    sample_assignment_plan_item,
    sample_filler_plan_item,
)


@dataclass(frozen=True, slots=True)
class SamplePreview:
    config_name: str
    seed: int
    instance: StateTrackingInstance
    rendered: RenderedExample
    target_extra_assignments: int | None

    def to_record(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "seed": self.seed,
            "canonical": self.instance.to_record(),
            "rendered": self.rendered.to_record(),
            "num_extra_assignments": len(self.instance.operations) - self.instance.num_variables,
            "target_extra_assignments": self.target_extra_assignments,
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
        if condition_uses_formatter(condition, formatter.name)
    }
    return tuple(sorted(names))


def generate_preview(
    loaded: LoadedExperiment,
    *,
    config_name: str,
    seed: int | None = None,
    num_variables: int | None = None,
    extra_assignments: int | None = None,
    filler_proportion: float | None = None,
    filler_distraction: FillerDistraction | None = None,
) -> SamplePreview:
    """Generate one sample for a rendered config without creating output files."""

    resolved_seed = loaded.spec.dataset.seed if seed is None else seed
    formatter, conditions = _resolve_config(loaded, config_name)
    candidates = [
        (condition, candidate_num_variables, candidate_proportion, candidate_distraction)
        for condition in conditions
        for candidate_num_variables in condition.num_variables
        for candidate_proportion, candidate_distraction in filler_variants(condition.filler)
        if num_variables is None or candidate_num_variables == num_variables
        if extra_assignments is None
        or condition.extra_assignments.min <= extra_assignments <= condition.extra_assignments.max
        if filler_proportion is None or candidate_proportion == filler_proportion
        if filler_distraction is None or candidate_distraction == filler_distraction
    ]
    if not candidates:
        supported = sorted(
            {candidate for condition in conditions for candidate in condition.num_variables}
        )
        raise ValueError(
            f"config {config_name!r} does not support num_variables={num_variables} and "
            f"extra_assignments={extra_assignments}; variable counts are {supported}"
        )

    selection_rng = random.Random(derive_seed(resolved_seed, "preview", config_name, "selection"))
    condition, selected_num_variables, selected_proportion, selected_distraction = (
        selection_rng.choice(candidates)
    )
    if extra_assignments is None:
        plan_item = sample_assignment_plan_item(
            condition.extra_assignments,
            rng=random.Random(derive_seed(resolved_seed, "preview", config_name, "depth")),
        )
    else:
        plan_item = AssignmentPlanItem(extra_assignments, extra_assignments, extra_assignments)
    if filler_proportion is None and filler_distraction is None:
        filler_item = sample_filler_plan_item(
            condition.filler,
            rng=random.Random(derive_seed(resolved_seed, "preview", config_name, "filler")),
        )
    else:
        filler_item = FillerPlanItem(selected_proportion, selected_distraction)
    instance = _generate(
        loaded,
        condition=condition,
        num_variables=selected_num_variables,
        seed=derive_seed(resolved_seed, "preview", config_name, "semantic"),
        plan_item=plan_item,
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
        filler_proportion=filler_item.proportion,
        filler_distraction=filler_item.distraction,
    )
    return SamplePreview(
        config_name=config_name,
        seed=resolved_seed,
        instance=instance,
        rendered=rendered,
        target_extra_assignments=plan_item.target_extra_assignments,
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
        f"Assignments: {len(preview.instance.operations)}\n"
        f"Extra assignments: {len(preview.instance.operations) - preview.instance.num_variables}\n"
        f"Target extra-assignment bucket: {preview.target_extra_assignments}\n\n"
        f"Filler proportion: {preview.rendered.filler_proportion}\n"
        f"Actual filler proportion: {preview.rendered.actual_filler_proportion}\n"
        f"Filler distraction: {preview.rendered.filler_distraction}\n"
        f"Filler lines: {preview.rendered.num_filler_lines}\n\n"
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
            if not condition_uses_formatter(condition, formatter.name):
                continue
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
    plan_item: AssignmentPlanItem,
) -> StateTrackingInstance:
    generation = loaded.spec.generation
    return generate_instance(
        complexity=condition.complexity,
        num_variables=num_variables,
        num_values=condition.num_values,
        min_extra_assignments=plan_item.min_extra_assignments,
        max_extra_assignments=plan_item.max_extra_assignments,
        query_mode=generation.query_mode,
        distinct_initial_values=generation.distinct_initial_values,
        require_strict_witness=generation.require_strict_witness,
        allow_self_reference=generation.allow_self_reference,
        periodic_constant_probability=generation.periodic_constant_probability,
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
