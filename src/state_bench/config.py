"""Versioned, strict YAML configuration models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from state_bench.model import Complexity, QueryMode


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


class ConditionSpec(StrictModel):
    complexity: Complexity
    num_variables: list[PositiveInt]
    num_values: PositiveInt
    extra_assignments: AssignmentRange

    @model_validator(mode="after")
    def nonempty_variables(self) -> ConditionSpec:
        if not self.num_variables:
            raise ValueError("num_variables must not be empty")
        if len(set(self.num_variables)) != len(self.num_variables):
            raise ValueError("num_variables must not contain duplicates")
        return self


class GenerationSpec(StrictModel):
    query_mode: QueryMode = QueryMode.ALL
    require_strict_witness: bool = True
    allow_self_reference: bool = False
    distinct_initial_values: bool = False
    conditions: list[ConditionSpec]

    @model_validator(mode="after")
    def validate_conditions(self) -> GenerationSpec:
        if not self.conditions:
            raise ValueError("conditions must not be empty")
        seen: set[tuple[Complexity, int, int]] = set()
        for condition in self.conditions:
            for num_variables in condition.num_variables:
                key = (condition.complexity, num_variables, condition.num_values)
                if key in seen:
                    raise ValueError(f"duplicate condition: {key}")
                seen.add(key)
                if self.distinct_initial_values and condition.num_values < num_variables:
                    raise ValueError("distinct initial values require num_values >= num_variables")
                if condition.complexity is Complexity.R_TRIVIAL and condition.extra_assignments.max != 0:
                    raise ValueError("r-trivial conditions cannot have extra assignments")
                if (
                    self.require_strict_witness
                    and condition.complexity is not Complexity.R_TRIVIAL
                    and condition.extra_assignments.min < 1
                ):
                    raise ValueError(
                        f"strict {condition.complexity.value} conditions need at least one extra "
                        "assignment"
                    )
                if (
                    self.require_strict_witness
                    and condition.complexity is Complexity.PERIODIC
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


class FormatterReference(StrictModel):
    config: Path


class OutputSpec(StrictModel):
    local_format: Literal["jsonl"] = "jsonl"
    include_canonical_trace: bool = True
    include_final_state: bool = True
    shard_size: PositiveInt = 100_000


class HubRepositorySpec(StrictModel):
    repo_id: str | None = None
    private: bool = True


class HuggingFaceSpec(StrictModel):
    include_canonical_config: bool = True
    config_naming: str = "{formatter}--{complexity}"
    data_format: Literal["parquet"] = "parquet"
    repository: HubRepositorySpec = HubRepositorySpec()

    @model_validator(mode="after")
    def validate_config_naming(self) -> HuggingFaceSpec:
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
    values: list[str]

    @model_validator(mode="after")
    def validate_vocabulary(self) -> VocabularySpec:
        if not self.variables or not self.values:
            raise ValueError("variable and value vocabularies must not be empty")
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


PromptBlock = Literal["task_description", "program", "answer_prompt"]


class LayoutSpec(StrictModel):
    prompt_blocks: list[PromptBlock] = ["task_description", "program", "answer_prompt"]
    block_separator: str = "\n\n"
    instruction_separator: str = " "
    completion_separator: str = " "
    trailing_newline: bool = False


class FormatterSpec(StrictModel):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    renderer: Literal["template"]
    vocabulary: VocabularySpec
    templates: TemplatesSpec
    layout: LayoutSpec = LayoutSpec()

    @property
    def config_hash(self) -> str:
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()[:20]


class LoadedExperiment:
    def __init__(self, *, path: Path, spec: ExperimentSpec, formatters: tuple[FormatterSpec, ...]) -> None:
        self.path = path
        self.spec = spec
        self.formatters = formatters


def load_experiment(path: str | Path) -> LoadedExperiment:
    experiment_path = Path(path).resolve()
    spec = ExperimentSpec.model_validate(_load_yaml(experiment_path))
    formatters = tuple(
        load_formatter(experiment_path.parent / formatter.config) for formatter in spec.formatters
    )
    names = [formatter.name for formatter in formatters]
    if len(set(names)) != len(names):
        raise ValueError("formatter names must be unique")
    _validate_formatter_capacity(spec, formatters)
    return LoadedExperiment(path=experiment_path, spec=spec, formatters=formatters)


def load_formatter(path: str | Path) -> FormatterSpec:
    return FormatterSpec.model_validate(_load_yaml(Path(path)))


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
    max_variables = max(
        num_variables
        for condition in experiment.generation.conditions
        for num_variables in condition.num_variables
    )
    max_values = max(condition.num_values for condition in experiment.generation.conditions)
    for formatter in formatters:
        if len(formatter.vocabulary.variables) < max_variables:
            raise ValueError(f"formatter {formatter.name!r} does not define {max_variables} variables")
        if len(formatter.vocabulary.values) < max_values:
            raise ValueError(f"formatter {formatter.name!r} does not define {max_values} values")
