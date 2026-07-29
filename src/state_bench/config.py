"""Versioned, strict YAML configuration models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Formatter, Template
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from state_bench.model import Complexity, QueryMode

_FORMATTER_CONFIG_HASH_CACHE: dict[int, tuple[object, str]] = {}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssignmentRange(StrictModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> AssignmentRange:
        if self.max < self.min:
            raise ValueError("max must be greater than or equal to min")
        return self


class LogSpacedAssignmentSchedule(StrictModel):
    distribution: Literal["log-spaced"]
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    growth_factor: float = Field(default=2.0, gt=1.0)
    sample_decay: float = Field(default=0.65, gt=0.0, le=1.0)
    jitter_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    min_samples_per_bucket: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def ordered(self) -> LogSpacedAssignmentSchedule:
        if self.max < self.min:
            raise ValueError("max must be greater than or equal to min")
        return self


type ExtraAssignmentsSpec = AssignmentRange | LogSpacedAssignmentSchedule


FillerDistraction = Literal["easy", "medium", "hard"]


class FillerSpec(StrictModel):
    """Surface-only no-op lines varied within a condition's sample budget."""

    proportions: list[float] = [0.0]
    distraction_levels: list[FillerDistraction] = ["easy"]
    stratify_by_assignment_bucket: bool = False

    @model_validator(mode="after")
    def validate_axes(self) -> FillerSpec:
        if not self.proportions:
            raise ValueError("filler proportions must not be empty")
        if not self.distraction_levels:
            raise ValueError("filler distraction_levels must not be empty")
        if len(set(self.proportions)) != len(self.proportions):
            raise ValueError("filler proportions must not contain duplicates")
        if len(set(self.distraction_levels)) != len(self.distraction_levels):
            raise ValueError("filler distraction_levels must not contain duplicates")
        if any(proportion < 0.0 or proportion >= 1.0 for proportion in self.proportions):
            raise ValueError("filler proportions must satisfy 0 <= proportion < 1")
        return self


class ConditionSpec(StrictModel):
    complexity: Complexity
    num_variables: list[PositiveInt]
    num_values: PositiveInt
    extra_assignments: ExtraAssignmentsSpec
    filler: FillerSpec = FillerSpec()
    formatters: list[str] | None = None
    splits: list[str] | None = None

    @model_validator(mode="after")
    def nonempty_variables(self) -> ConditionSpec:
        if not self.num_variables:
            raise ValueError("num_variables must not be empty")
        if len(set(self.num_variables)) != len(self.num_variables):
            raise ValueError("num_variables must not contain duplicates")
        if self.formatters is not None:
            if not self.formatters:
                raise ValueError("condition formatters must not be empty")
            if len(set(self.formatters)) != len(self.formatters):
                raise ValueError("condition formatters must not contain duplicates")
        if self.splits is not None:
            if not self.splits:
                raise ValueError("condition splits must not be empty")
            if len(set(self.splits)) != len(self.splits):
                raise ValueError("condition splits must not contain duplicates")
        return self


class GenerationSpec(StrictModel):
    query_mode: QueryMode = QueryMode.ALL
    require_strict_witness: bool = True
    allow_self_reference: bool = False
    distinct_initial_values: bool = False
    periodic_constant_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    conditions: list[ConditionSpec]

    @model_validator(mode="after")
    def validate_conditions(self) -> GenerationSpec:
        if not self.conditions:
            raise ValueError("conditions must not be empty")
        seen: set[
            tuple[
                Complexity,
                int,
                int,
                str,
                str,
                tuple[str, ...] | None,
                tuple[str, ...] | None,
            ]
        ] = set()
        for condition in self.conditions:
            if condition.complexity is not Complexity.PERIODIC:
                raise ValueError(
                    "generation conditions must describe periodic source traces; "
                    "aperiodic and r-trivial traces are derived automatically"
                )
            for num_variables in condition.num_variables:
                key = (
                    condition.complexity,
                    num_variables,
                    condition.num_values,
                    condition.extra_assignments.model_dump_json(),
                    condition.filler.model_dump_json(),
                    tuple(condition.formatters) if condition.formatters is not None else None,
                    tuple(condition.splits) if condition.splits is not None else None,
                )
                if key in seen:
                    raise ValueError(f"duplicate condition: {key}")
                seen.add(key)
                if self.distinct_initial_values and condition.num_values < num_variables:
                    raise ValueError("distinct initial values require num_values >= num_variables")
                if (
                    self.require_strict_witness
                    and condition.extra_assignments.min < 1
                ):
                    raise ValueError(
                        "strict periodic source conditions need at least one extra assignment"
                    )
                if (
                    self.require_strict_witness
                    and (num_variables < 2 or condition.num_values < 2)
                ):
                    raise ValueError("strict periodic conditions need at least two variables and values")
        return self


class DatasetSpec(StrictModel):
    name: str = Field(min_length=1)
    seed: int = 0
    output_dir: Path


class SplitSpec(StrictModel):
    samples_per_condition: int = Field(ge=0)
    formatters: list[str] | None = None
    disjoint_from: list[str] = []

    @model_validator(mode="after")
    def validate_scope(self) -> SplitSpec:
        if self.formatters is not None:
            if not self.formatters:
                raise ValueError("split formatters must not be empty")
            if len(set(self.formatters)) != len(self.formatters):
                raise ValueError("split formatters must not contain duplicates")
        if len(set(self.disjoint_from)) != len(self.disjoint_from):
            raise ValueError("split disjoint_from must not contain duplicates")
        return self


class FormatterReference(StrictModel):
    config: Path


class MultipleChoiceSpec(StrictModel):
    num_distractors: int = Field(default=4, ge=0)
    shuffle_choices: bool = True


class OutputSpec(StrictModel):
    local_format: Literal["jsonl"] = "jsonl"
    include_canonical_trace: bool = True
    include_final_state: bool = True
    # Large training corpora generally only need prompt/completion pairs. These
    # switches keep the richer evaluation artifact as the default.
    minimal_splits: list[str] = []
    omit_canonical_splits: list[str] = []
    track_unique_semantics: bool = True
    hash_files: bool = True
    shard_size: PositiveInt = 100_000
    target_shard_bytes: PositiveInt = 256_000_000
    multiple_choice: MultipleChoiceSpec = MultipleChoiceSpec()


class HubRepositorySpec(StrictModel):
    repo_id: str | None = None
    private: bool = True


class HubCardSpec(StrictModel):
    license: str | None = None
    language: list[str] = ["en"]
    tags: list[str] = ["state-tracking", "formal-languages", "llm-evaluation"]


class TokenStratumSpec(StrictModel):
    """One half-open token-length interval used as a published Hub split."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    min_tokens: int = Field(ge=0)
    max_tokens: PositiveInt | None = None

    @model_validator(mode="after")
    def ordered(self) -> TokenStratumSpec:
        if self.max_tokens is not None and self.max_tokens <= self.min_tokens:
            raise ValueError("max_tokens must be greater than min_tokens")
        return self


class TokenStratificationSpec(StrictModel):
    """Tokenizer-dependent rendered-data splits created during Hub staging."""

    tokenizer: str = Field(min_length=1)
    revision: str | None = None
    mode: Literal["exact", "sampled"] = "exact"
    field: Literal["prefix", "prefix+completion"] = "prefix"
    joiner: str = ""
    add_special_tokens: bool = False
    batch_size: PositiveInt = 1
    max_batch_characters: PositiveInt = 16_000_000
    calibration_scan_rows: PositiveInt = 128
    calibration_samples_per_group: PositiveInt = 16
    ranges: list[TokenStratumSpec]

    @model_validator(mode="after")
    def contiguous_ranges(self) -> TokenStratificationSpec:
        if not self.ranges:
            raise ValueError("token strata ranges must not be empty")
        if len({item.name for item in self.ranges}) != len(self.ranges):
            raise ValueError("token strata names must be unique")
        if self.ranges[0].min_tokens != 0:
            raise ValueError("token strata must start at zero")
        for left, right in zip(self.ranges, self.ranges[1:], strict=False):
            if left.max_tokens is None or left.max_tokens != right.min_tokens:
                raise ValueError("token strata must be ordered, contiguous, and non-overlapping")
        if self.ranges[-1].max_tokens is not None:
            raise ValueError("the final token stratum must have max_tokens: null")
        return self


class HuggingFaceSpec(StrictModel):
    include_canonical_config: bool = True
    config_naming: str = "{formatter}--{complexity}"
    data_format: Literal["parquet"] = "parquet"
    staging_workers: PositiveInt = 1
    token_strata: TokenStratificationSpec | None = None
    repository: HubRepositorySpec = HubRepositorySpec()
    card: HubCardSpec = HubCardSpec()

    @model_validator(mode="after")
    def validate_config_naming(self) -> HuggingFaceSpec:
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.config_naming)
            if field_name is not None
        }
        if fields - {"formatter", "complexity"}:
            raise ValueError("config_naming may only interpolate formatter and complexity")
        try:
            rendered = self.config_naming.format(formatter="format", complexity="complexity")
        except (KeyError, ValueError) as error:
            raise ValueError("config_naming may only interpolate formatter and complexity") from error
        if not rendered or "/" in rendered:
            raise ValueError("config_naming must produce a nonempty path-safe name")
        return self


class ExperimentSpec(StrictModel):
    schema_version: Literal[1]
    dataset: DatasetSpec
    splits: dict[str, SplitSpec]
    generation: GenerationSpec
    formatters: list[FormatterReference]
    output: OutputSpec = OutputSpec()
    huggingface: HuggingFaceSpec = HuggingFaceSpec()

    @model_validator(mode="after")
    def nonempty_sections(self) -> ExperimentSpec:
        if not self.splits:
            raise ValueError("splits must not be empty")
        if not self.formatters:
            raise ValueError("formatters must not be empty")
        if any(not split for split in self.splits):
            raise ValueError("split names must not be empty")
        preceding_splits: set[str] = set()
        for split_name, split in self.splits.items():
            if split_name in split.disjoint_from:
                raise ValueError(f"split {split_name!r} cannot be disjoint from itself")
            unknown = set(split.disjoint_from) - self.splits.keys()
            if unknown:
                raise ValueError(f"split {split_name!r} has unknown disjoint_from splits: {sorted(unknown)}")
            unavailable = set(split.disjoint_from) - preceding_splits
            if unavailable:
                raise ValueError(
                    f"split {split_name!r} must appear after its disjoint_from splits: {sorted(unavailable)}"
                )
            preceding_splits.add(split_name)
        for condition in self.generation.conditions:
            unknown = set(condition.splits or self.splits) - self.splits.keys()
            if unknown:
                raise ValueError(f"condition references unknown splits: {sorted(unknown)}")
        for split_name in self.splits:
            if not any(
                condition.splits is None or split_name in condition.splits
                for condition in self.generation.conditions
            ):
                raise ValueError(f"split {split_name!r} has no generation conditions")
        output_splits = set(self.output.minimal_splits) | set(self.output.omit_canonical_splits)
        unknown_output_splits = output_splits - self.splits.keys()
        if unknown_output_splits:
            raise ValueError(f"output references unknown splits: {sorted(unknown_output_splits)}")
        if len(set(self.output.minimal_splits)) != len(self.output.minimal_splits):
            raise ValueError("output.minimal_splits must not contain duplicates")
        if len(set(self.output.omit_canonical_splits)) != len(self.output.omit_canonical_splits):
            raise ValueError("output.omit_canonical_splits must not contain duplicates")
        return self


class TemplateChoice(StrictModel):
    text: str
    weight: float = Field(default=1.0, gt=0)


class TemplateChoices(StrictModel):
    choices: list[TemplateChoice]

    @model_validator(mode="after")
    def nonempty_choices(self) -> TemplateChoices:
        if not self.choices:
            raise ValueError("template choices must not be empty")
        return self


type TemplateValue = str | TemplateChoices


class VocabularySpec(StrictModel):
    variables: list[str]
    values: list[str] = []
    variable_selection: Literal["prefix", "sample-without-replacement"] = "prefix"
    value_selection: Literal["prefix", "random-seven-digit"] = "prefix"

    @model_validator(mode="after")
    def validate_vocabulary(self) -> VocabularySpec:
        if not self.variables:
            raise ValueError("variable vocabulary must not be empty")
        if self.value_selection == "prefix" and not self.values:
            raise ValueError("fixed value vocabulary must not be empty")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable vocabulary must be unique")
        if len(set(self.values)) != len(self.values):
            raise ValueError("value vocabulary must be unique")
        return self


class TemplatesSpec(StrictModel):
    task_description: TemplateValue | None = None
    assign_constant: TemplateValue
    assign_reference: TemplateValue
    answer_prompt: TemplateValue | None = None
    state_assertion: TemplateValue
    filler_easy: TemplateValue | None = None
    filler_medium: TemplateValue | None = None
    filler_hard: TemplateValue | None = None


PromptBlock = Literal["task_description", "program", "answer_prompt"]


class LayoutSpec(StrictModel):
    prompt_blocks: list[PromptBlock] = ["task_description", "program", "answer_prompt"]
    block_separator: str = "\n\n"
    instruction_separator: str = " "
    completion_separator: str = " "
    trailing_newline: bool = False

    @model_validator(mode="after")
    def unique_blocks(self) -> LayoutSpec:
        if len(set(self.prompt_blocks)) != len(self.prompt_blocks):
            raise ValueError("prompt_blocks must not contain duplicates")
        return self


class FormatterSpec(StrictModel):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    renderer: Literal["template"]
    filler_corpus: str | None = None
    vocabulary: VocabularySpec
    templates: TemplatesSpec
    layout: LayoutSpec = LayoutSpec()

    @model_validator(mode="after")
    def validate_templates(self) -> FormatterSpec:
        metadata = {
            "num_variables",
            "num_values",
            "num_assignments",
            "complexity",
            "query_count",
            "query_variables",
        }
        template_fields = {
            "task_description": (self.templates.task_description, metadata),
            "assign_constant": (self.templates.assign_constant, metadata | {"target", "value"}),
            "assign_reference": (self.templates.assign_reference, metadata | {"target", "source"}),
            "answer_prompt": (self.templates.answer_prompt, metadata),
            "state_assertion": (self.templates.state_assertion, metadata | {"variable", "value"}),
            "filler_easy": (
                self.templates.filler_easy,
                metadata
                | {"variable", "value", "filler_index", "filler_dots", "filler_text"},
            ),
            "filler_medium": (
                self.templates.filler_medium,
                metadata
                | {"variable", "value", "filler_index", "filler_dots", "filler_text"},
            ),
            "filler_hard": (
                self.templates.filler_hard,
                metadata
                | {"variable", "value", "filler_index", "filler_dots", "filler_text"},
            ),
        }
        for field_name, (value, allowed) in template_fields.items():
            if value is None:
                continue
            for text in _template_texts(value):
                template = Template(text)
                if not template.is_valid():
                    raise ValueError(f"invalid template syntax in {field_name}")
                unknown = set(template.get_identifiers()) - allowed
                if unknown:
                    raise ValueError(f"unknown placeholders in {field_name}: {sorted(unknown)}")
                if field_name.startswith("filler_") and (not text.strip() or "\n" in text):
                    raise ValueError(f"{field_name} templates must contain exactly one nonempty line")
        return self

    @property
    def config_hash(self) -> str:
        cached = _FORMATTER_CONFIG_HASH_CACHE.get(id(self))
        if cached is not None and cached[0] is self:
            return cached[1]
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(content.encode()).hexdigest()[:20]
        _FORMATTER_CONFIG_HASH_CACHE[id(self)] = (self, digest)
        return digest


class LoadedExperiment:
    def __init__(self, *, path: Path, spec: ExperimentSpec, formatters: tuple[FormatterSpec, ...]) -> None:
        self.path = path
        self.spec = spec
        self.formatters = formatters


def load_experiment(path: str | Path) -> LoadedExperiment:
    experiment_path = Path(path).resolve()
    spec = ExperimentSpec.model_validate(_load_yaml(experiment_path))
    _validate_schedule_budgets(spec)
    formatters = tuple(
        load_formatter(experiment_path.parent / formatter.config) for formatter in spec.formatters
    )
    names = [formatter.name for formatter in formatters]
    if len(set(names)) != len(names):
        raise ValueError("formatter names must be unique")
    _validate_condition_formatters(spec, formatters)
    _validate_split_formatters(spec, formatters)
    _validate_formatter_capacity(spec, formatters)
    return LoadedExperiment(path=experiment_path, spec=spec, formatters=formatters)


def load_formatter(path: str | Path) -> FormatterSpec:
    formatter_path = Path(path).resolve()
    record = _load_yaml(formatter_path)
    if not isinstance(record, dict):
        return FormatterSpec.model_validate(record)
    formatter_record = cast(dict[str, object], record.copy())
    if "filler_corpus_path" in formatter_record:
        corpus_path = formatter_path.parent / str(formatter_record.pop("filler_corpus_path"))
        formatter_record["filler_corpus"] = corpus_path.read_text(encoding="utf-8")
    return FormatterSpec.model_validate(formatter_record)


def _load_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        raise ValueError(f"empty YAML file: {path}")
    return content


def _validate_formatter_capacity(
    experiment: ExperimentSpec,
    formatters: tuple[FormatterSpec, ...],
) -> None:
    num_distractors = experiment.output.multiple_choice.num_distractors
    for condition in experiment.generation.conditions:
        for num_variables in condition.num_variables:
            queried_variables = num_variables if experiment.generation.query_mode is QueryMode.ALL else 1
            available_wrong_answers = condition.num_values**queried_variables - 1
            if available_wrong_answers < num_distractors:
                raise ValueError(
                    f"source condition {num_variables} variables has only "
                    f"{available_wrong_answers} distinct wrong answers, fewer than the requested "
                    f"{num_distractors}"
                )
    for formatter in formatters:
        selected_conditions = [
            condition
            for condition in experiment.generation.conditions
            if condition_uses_formatter(condition, formatter.name)
        ]
        if not selected_conditions:
            continue
        max_variables = max(
            num_variables for condition in selected_conditions for num_variables in condition.num_variables
        )
        max_values = max(condition.num_values for condition in selected_conditions)
        if len(formatter.vocabulary.variables) < max_variables:
            raise ValueError(f"formatter {formatter.name!r} does not define {max_variables} variables")
        if (
            formatter.vocabulary.value_selection == "prefix"
            and len(formatter.vocabulary.values) < max_values
        ):
            raise ValueError(f"formatter {formatter.name!r} does not define {max_values} values")


def _validate_condition_formatters(
    experiment: ExperimentSpec,
    formatters: tuple[FormatterSpec, ...],
) -> None:
    available = {formatter.name: formatter for formatter in formatters}
    for condition in experiment.generation.conditions:
        selected = set(condition.formatters or available)
        unknown = selected - available.keys()
        if unknown:
            raise ValueError(f"condition references unknown formatters: {sorted(unknown)}")
        if any(proportion > 0 for proportion in condition.filler.proportions):
            for name in selected:
                formatter = available[name]
                for level in condition.filler.distraction_levels:
                    if getattr(formatter.templates, f"filler_{level}") is None:
                        raise ValueError(
                            f"formatter {name!r} has no filler_{level} template required by condition"
                        )


def _validate_split_formatters(
    experiment: ExperimentSpec,
    formatters: tuple[FormatterSpec, ...],
) -> None:
    available = {formatter.name for formatter in formatters}
    for split_name, split in experiment.splits.items():
        unknown = set(split.formatters or available) - available
        if unknown:
            raise ValueError(f"split {split_name!r} references unknown formatters: {sorted(unknown)}")


def condition_uses_formatter(condition: ConditionSpec, formatter_name: str) -> bool:
    return condition.formatters is None or formatter_name in condition.formatters


def condition_uses_split(condition: ConditionSpec, split_name: str) -> bool:
    return condition.splits is None or split_name in condition.splits


def split_uses_formatter(split: SplitSpec, formatter_name: str) -> bool:
    return split.formatters is None or formatter_name in split.formatters


def filler_variants(spec: FillerSpec) -> tuple[tuple[float, FillerDistraction | None], ...]:
    """Expand axes, collapsing the zero-filler baseline to one distraction-free variant."""

    variants: list[tuple[float, FillerDistraction | None]] = []
    for proportion in spec.proportions:
        if proportion == 0:
            variants.append((proportion, None))
        else:
            variants.extend((proportion, level) for level in spec.distraction_levels)
    return tuple(variants)


def _template_texts(value: TemplateValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(choice.text for choice in value.choices)


def _validate_schedule_budgets(experiment: ExperimentSpec) -> None:
    from state_bench.scheduling import (
        allocate_bucket_counts,
    )

    for split_name, split in experiment.splits.items():
        for condition in experiment.generation.conditions:
            if not condition_uses_split(condition, split_name):
                continue
            bucket_counts: list[int]
            if isinstance(condition.extra_assignments, LogSpacedAssignmentSchedule):
                try:
                    bucket_counts = list(
                        allocate_bucket_counts(
                            condition.extra_assignments,
                            total_samples=split.samples_per_condition,
                        ).values()
                    )
                except ValueError as error:
                    raise ValueError(f"invalid assignment schedule for split {split_name!r}") from error
            else:
                bucket_counts = [split.samples_per_condition]

            variant_count = len(filler_variants(condition.filler))
            budgets = (
                bucket_counts
                if condition.filler.stratify_by_assignment_bucket
                else [split.samples_per_condition]
            )
            if any(0 < budget < variant_count for budget in budgets):
                raise ValueError(
                    f"invalid filler schedule for split {split_name!r}: "
                    f"each covered bucket needs at least {variant_count} samples"
                )
