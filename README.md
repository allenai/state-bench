# state-bench

Generate reproducible state-tracking evaluations from canonical assignment
traces, render the same traces into multiple surface forms, and stage complete
Hugging Face dataset repositories.

The semantic families are ordered by increasing state-tracking complexity:

- `r-trivial`: every variable receives one constant assignment.
- `aperiodic`: variables can be overwritten with constants.
- `periodic`: assignments may copy the current value of another variable.

Reference assignments use eager copy semantics. `a <- b` copies the value of
`b` at that point; it does not create a persistent alias.

## Install

```bash
uv sync
```

Install optional Hugging Face and Parquet support when staging or publishing:

```bash
uv sync --extra hub
```

## Configuration

An experiment YAML controls semantic generation, splits, selected formatters,
local artifacts, and Hugging Face metadata. See
`configs/experiments/default.yaml` for a complete example.

```yaml
schema_version: 1

dataset:
  name: state-tracking-v1
  seed: 13
  output_dir: ../../data/state-tracking-v1

splits:
  train: {samples_per_condition: 10000}
  validation: {samples_per_condition: 1000}
  test: {samples_per_condition: 2000}

generation:
  query_mode: all
  require_strict_witness: true
  allow_self_reference: false
  distinct_initial_values: false
  conditions:
    - complexity: r-trivial
      num_variables: [2, 4, 6]
      num_values: 10
      extra_assignments: {min: 0, max: 0}
    - complexity: aperiodic
      num_variables: [2, 4, 6]
      num_values: 10
      extra_assignments: {min: 1, max: 20}
    - complexity: periodic
      num_variables: [2, 4, 6]
      num_values: 10
      extra_assignments: {min: 1, max: 20}

formatters:
  - config: ../formats/integer-code.yaml
  - config: ../formats/cube-painting.yaml
  - config: ../formats/spreadsheet-cells.yaml
  - config: ../formats/status-lights.yaml
  - config: ../formats/people-in-rooms.yaml
```

Paths in an experiment are resolved relative to that experiment file.
Configuration models are strict and versioned: unknown keys, invalid family
parameters, insufficient formatter vocabularies, and invalid placeholders fail
during `config check`.

### Formatter YAML

Most surface forms use the built-in template renderer and require no Python:

```yaml
schema_version: 1
name: integer-code
renderer: template

vocabulary:
  variables: [a, b, c, d]
  values: ["1", "2", "3", "4"]

templates:
  task_description: >-
    Track the current value of each variable after executing the assignments.
  assign_constant: "$target <- $value;"
  assign_reference: "$target <- $source;"
  answer_prompt: null
  state_assertion: "$variable ?= $value;"

layout:
  prompt_blocks: [task_description, program, answer_prompt]
  block_separator: "\n\n"
  instruction_separator: " "
  completion_separator: " "
  trailing_newline: false
```

Set `task_description: null` to omit the description. Template fields may also
select deterministic weighted variants:

```yaml
assign_constant:
  choices:
    - {text: "Paint $target $value.", weight: 3}
    - {text: "Apply $value paint to $target.", weight: 1}
```

Assignment templates support `$target`, `$value`, or `$source` as applicable.
State assertions support `$variable` and `$value`. All templates may use
`$num_variables`, `$num_values`, `$num_assignments`, and `$complexity`.

## Generate locally

Validate the expanded experiment first:

```bash
uv run state-bench config check configs/experiments/default.yaml
```

Generate canonical and rendered JSONL:

```bash
uv run state-bench build configs/experiments/default.yaml
```

Preview one generated example without writing a dataset:

```bash
uv run state-bench sample \
  configs/experiments/default.yaml \
  integer-code--periodic \
  --num-variables 4 \
  --seed 7
```

This prints the canonical trace, rendered prefix, and expected completion. Use
`--json` for machine-readable output, or list available names with:

```bash
uv run state-bench sample configs/experiments/default.yaml --list-configs
```

The output contains:

```text
canonical/{split}.jsonl
rendered/{formatter}--{complexity}/{split}.jsonl
manifest.json
```

Every row has a unique, generation-stable `example_id`. Its content-derived
`instance_id` identifies semantic duplicates. The manifest reports unique
semantic counts and cross-split semantic overlap so dataset leakage is visible.

Rendered rows also contain four unique invalid `incorrect_completions`, a
shuffled five-item `choices` list, and zero-based `correct_choice_index`.
Configure these under `output.multiple_choice`:

```yaml
output:
  multiple_choice:
    num_distractors: 4
    shuffle_choices: true
```

Distractors are generated from alternative queried states and rendered with the
same answer template as the correct completion.

## Hugging Face

Stage a complete local dataset repository containing sharded Parquet files, a
dataset card, source YAML files, and a checksummed manifest:

```bash
uv run --extra hub state-bench hub stage \
  configs/experiments/default.yaml \
  --repo-dir build/state-tracking-hf

uv run --extra hub state-bench hub validate build/state-tracking-hf
```

Each `(formatter, complexity)` pair becomes a Hugging Face config, such as
`integer-code--periodic`. Variable counts and assignment depths remain row
columns. An optional `canonical` config contains the underlying traces.

Preview an upload without authentication or network writes:

```bash
uv run state-bench hub push build/state-tracking-hf \
  --repo-id organization/state-bench \
  --dry-run
```

After `hf auth login`, upload explicitly:

```bash
uv run --extra hub state-bench hub push build/state-tracking-hf \
  --repo-id organization/state-bench
```

The repository ID and default visibility can instead be set under
`huggingface.repository` in the experiment YAML. Authentication tokens are not
stored in configuration files.

Load a published config normally:

```python
from datasets import load_dataset

test = load_dataset(
    "organization/state-bench",
    "integer-code--periodic",
    split="test",
)
```

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run ty check src
```

Tests cover evaluator semantics, strict family witnesses, deterministic
generation and rendering, algebraic properties on exhaustive tiny domains,
configuration validation, Parquet sharding, Hub metadata, and upload dry runs.
