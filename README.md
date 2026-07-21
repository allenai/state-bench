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

Install tokenizer-based dataset analysis support:

```bash
uv sync --extra analysis
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
  periodic_constant_probability: 0.1
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

Periodic generation is dependency- and value-biased: copy operations extend
the deepest currently live provenance chain into one of the shallowest
variables, and every generated update changes its target's value. Operations
preserve the initial number of distinct live values whenever possible so one
lineage does not collapse the final answer to a repeated value. The
`periodic_constant_probability` controls how often periodic steps write a new
constant instead, resetting a shallow variable's dependency length while
retaining the other deep lineages. A strict value-changing copy is always
inserted first when `require_strict_witness` is enabled. With two variables, a
changing copy necessarily collapses the state temporarily, so the generator
uses a later constant to restore diversity. Lower constant probabilities
therefore produce substantially deeper final dependency chains where the
diversity constraint permits them.

Paths in an experiment are resolved relative to that experiment file.
Configuration models are strict and versioned: unknown keys, invalid family
parameters, insufficient formatter vocabularies, and invalid placeholders fail
during `config check`.

### Long-context assignment schedules

Use `distribution: log-spaced` to spread a fixed sample budget over
exponentially increasing assignment depths:

```yaml
splits:
  test:
    samples_per_condition: 6000

generation:
  conditions:
    - complexity: periodic
      num_variables: [2, 4, 6]
      num_values: 10
      extra_assignments:
        distribution: log-spaced
        min: 1
        max: 4096
        growth_factor: 2.0
        sample_decay: 0.65
        jitter_fraction: 0.10
        min_samples_per_bucket: 8
```

This creates target buckets `1, 2, 4, ..., 4096`. The
`samples_per_condition` value remains the total budget for each
complexity/variable-count pair; it is not generated separately at every depth.
Bucket weights decay by `sample_decay`, and `min_samples_per_bucket` guarantees
coverage of the longest contexts. `jitter_fraction` samples nearby assignment
counts rather than putting every row at the exact bucket center. Set
`sample_decay: 1.0` for uniform sampling across the log-spaced target buckets.

The provided long-context experiment uses 380 samples per
complexity/variable-count condition, uniform weights across 19 assignment
buckets up to 192,000 assignments, and exactly 20 samples at every target
assignment length. It disables assignment jitter so those per-length sample
counts are exact. Hub Parquet shards are capped at 50 long rows apiece.
Its r-trivial condition reaches a comparable scale with a maximum filler
proportion of `0.99995`.

Rows include `num_extra_assignments` and `target_extra_assignments` metadata.
They also include final-state dependency provenance metadata:

- `final_dependency_lengths` gives the copy-hop count for every variable in
  variable-index order. A direct constant assignment has length zero, and each
  eager variable copy adds one to the source's length at that moment.
- `max_dependency_length`, `mean_dependency_length`, and
  `modal_dependency_length` summarize those per-variable lengths. If several
  lengths tie for the mode, the smallest is recorded.

R-trivial traces cannot become long through reassignment because each variable
is assigned exactly once. They can now be lengthened with surface-only filler;
the provided long-context experiment varies filler for one-variable r-trivial
examples in every configured format while retaining assignment-depth sweeps for
aperiodic and periodic traces:

```bash
uv run state-bench config check configs/experiments/long-context.yaml
uv run state-bench build configs/experiments/long-context.yaml
```

The preset also configures exact tokenizer-length strata for Hub staging:
`[0, 100k)`, `[100k, 500k)`, `[500k, 1m)`, and `[1m, infinity)`. Token counts
use the OLMo 3 tokenizer over each rendered prompt prefix. Stage this preset
with both optional dependency groups because stratification requires the
tokenizer as well as Parquet support:

```bash
uv run --extra hub --extra analysis state-bench hub stage \
  configs/experiments/long-context.yaml \
  --repo-dir build/state-tracking-long-context-hf
```

Rendered configs then expose independently loadable splits such as
`test__tokens_0_100k`, `test__tokens_100k_500k`, `test__tokens_500k_1m`, and
`test__tokens_1m_plus`. Every staged row includes `token_length` and
`token_stratum`; the canonical config retains the base `test` split.

### No-op filler lines

Each condition can vary the requested proportion of nonblank prompt lines that
are no-op filler and how distracting those lines look:

```yaml
generation:
  conditions:
    - complexity: r-trivial
      num_variables: [1]
      num_values: 10
      extra_assignments: {min: 0, max: 0}
      filler:
        proportions: [0.0, 0.9, 0.99, 0.999]
        distraction_levels: [easy, medium, hard]
      formatters: [integer-code]
```

The filler axes share `samples_per_condition`; they do not multiply it. A zero
proportion creates one distraction-free baseline, while every positive
proportion is crossed with each distraction level. If a prompt has `L`
non-filler content lines and requests proportion `p`, rendering adds the nearest
integer to `L * p / (1 - p)` filler lines. Filler is interleaved with program
lines without changing assignment order, the canonical trace, or the answer.

`formatters` is an optional condition-level filter. Omit it to render the
condition with every experiment formatter. Each selected formatter supplies
`filler_easy`, `filler_medium`, and `filler_hard` templates. Filler templates
may use `$variable`, `$value`, `$filler_index`, and the standard metadata
placeholders.

Rendered rows record `filler_proportion`, `actual_filler_proportion`,
`filler_distraction`, and `num_filler_lines`. Preview an exact setting with:

```bash
uv run state-bench sample configs/experiments/long-context.yaml \
  integer-code--r-trivial --num-variables 1 \
  --filler-proportion 0.99 --filler-distraction hard
```

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

For a specific long-context depth, add `--extra-assignments 1024`.

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

Distractors are generated by injectively relabeling the correct queried values
and rendered with the same answer template as the correct completion. This
preserves the correct answer's equality pattern: if two queried variables have
the same value in the correct answer, they also have the same value in every
distractor, and vice versa.

### Rendered dataset statistics

Summarize a rendered JSONL file or directory using the OLMo 3 tokenizer:

```bash
uv run --extra analysis state-bench stats \
  data/state-tracking-v1/rendered/integer-code--periodic/test.jsonl
```

The default tokenizer is `allenai/Olmo-3-1025-7B`. Only tokenizer files are
loaded; model weights are not downloaded. The report includes categorical row
counts, assignment-count statistics, and token-count distributions for:

- Prefixes
- Correct completions
- Prefix plus correct completion
- Individual multiple-choice options
- Prefix plus each multiple-choice option

Each distribution reports its count, mean, population standard deviation,
minimum, maximum, percentiles, and a text histogram. Useful options include:

`prefix_plus_completion_tokens` tokenizes `prefix + joiner + completion` as one
string so tokens can cross the field boundary. The joiner is empty by default;
set `--joiner` if the evaluation harness inserts a separator.

```bash
# Compare groups inside a directory containing several rendered configs.
uv run --extra analysis state-bench stats data/state-tracking-v1/rendered \
  --group-by formatter complexity num_variables

# Select another Hugging Face tokenizer and emit machine-readable output.
uv run --extra analysis state-bench stats path/to/test.parquet \
  --tokenizer meta-llama/Llama-3.1-8B \
  --json

# Analyze a sample or count special tokens as well.
uv run --extra analysis state-bench stats path/to/rendered.jsonl \
  --limit 1000 \
  --add-special-tokens
```

The standalone `state-bench-stats` entry point accepts the same options.

For a compact prompt-length histogram, use the focused utility. It defaults to
the OLMo 3 tokenizer, log2 bins, prompt prefixes only, and a batch size of one
so million-token examples do not create unexpectedly large tokenizer batches:

```bash
uv run --extra analysis python scripts/token_length_histogram.py \
  data/state-tracking-long-context-v1/rendered \
  --group-by formatter complexity
```

Use linear bins or include the completion when needed:

```bash
uv run --extra analysis state-bench-token-histogram path/to/rendered \
  --field prefix+completion --joiner " " \
  --binning linear --bin-width 50000
```

The long-context preset supplies the local rendered-data path and groups by
formatter and complexity automatically:

```bash
uv run --extra analysis python scripts/long_context_token_histograms.py
```

Both utilities accept JSONL files, Parquet files, or directories containing one
of those formats. Pass `--json` for machine-readable histogram output and
`--tokenizer` to override `allenai/Olmo-3-1025-7B`.

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

Add `--sync` to make the remote file tree match the staged repository. This
deletes remote files that are absent locally before uploading replacements;
the previous version remains recoverable from the Hub commit history. Sync is
rejected unless the local directory looks like a complete staged repository.
Preview the upload and deletion pattern without changing the Hub first:

```bash
uv run state-bench hub push build/state-tracking-hf \
  --repo-id organization/state-bench --sync --dry-run

uv run --extra hub state-bench hub push build/state-tracking-hf \
  --repo-id organization/state-bench --sync
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

For a token-stratified long-context config, select only the desired cost tier:

```python
short_test = load_dataset(
    "organization/state-bench",
    "integer-code--periodic",
    split="test__tokens_0_100k",
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
