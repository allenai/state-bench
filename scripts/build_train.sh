#!/usr/bin/env bash
# Build every training distribution and put it in place, end to end:
#
#   1. Build the `test` split (needed for train disjointness IDs).
#   2. Build the `train` split (r-trivial, aperiodic, periodic rendered configs).
#   3. Partition the periodic train split into sensitivity-quantile bucket
#      datasets under <output_dir>/rendered-sensitivity/ (kept out of rendered/
#      so hub staging never uploads them).
#   4. Tokenize the train splits of rendered/ and rendered-sensitivity/, and
#      merge the bucket token directories into <output_dir>/rendered-tokenized/
#      tokens/ alongside the base distributions -- the layout the olmo-core
#      StateBench ladder reads.
#
# Training is `disjoint_from: [test]` in the experiment config, and
# disjointness is enforced from semantic IDs a prior build persisted under
# <output_dir>/semantics/test.ids (see state_bench.io.build_dataset). That
# file is only written by a build that includes the `test` split, so this
# script builds `test` first unless --skip-test says the persisted IDs are
# already current.

set -euo pipefail

CONFIG="${CONFIG:-configs/experiments/default.yaml}"
SENS_BUCKETS="${SENS_BUCKETS:-16}"
SENS_FIELD="${SENS_FIELD:-sensitivity_coverage}"
PERIODIC_CONFIG="${PERIODIC_CONFIG:-integer-code--periodic}"
SKIP_TEST=0
SKIP_TOKENIZE=0

TOKENIZE_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/build_train.sh [--skip-test] [--skip-tokenize] [--tokenize-only] [--config PATH]

Build the `test` split (for its persisted semantic IDs), build the `train`
split, partition the periodic train split into sensitivity buckets, and
tokenize everything into <output_dir>/rendered-tokenized/tokens/.

Options:
  --skip-test       Skip building `test`; reuse the semantics/test.ids already
                    persisted in the output directory from a prior build.
  --skip-tokenize   Stop after the sensitivity partition; leave tokenization
                    for a separate run (e.g. on a high-CPU machine).
  --tokenize-only   Skip the builds and the sensitivity partition and run just
                    the tokenization steps, reusing the existing rendered/ and
                    rendered-sensitivity/ trees. Use to resume after a
                    tokenization failure.
  --config PATH     Experiment YAML (default: configs/experiments/default.yaml,
                    or the CONFIG environment variable).
  -h, --help        Show this help.

Environment:
  SENS_BUCKETS      Number of sensitivity buckets (default: 16). Must match
                    STATE_BENCH_SENSITIVITY_BUCKETS in the olmo-core ladder.
  SENS_FIELD        Record field the buckets are ordered by
                    (default: sensitivity_coverage).
  PERIODIC_CONFIG   Rendered config to partition (default: integer-code--periodic).

Derived artifacts (rendered-sensitivity/, rendered-tokenized/,
rendered-sensitivity-tokenized/) are deleted and regenerated on each run.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-test) SKIP_TEST=1; shift ;;
        --skip-tokenize) SKIP_TOKENIZE=1; shift ;;
        --tokenize-only) TOKENIZE_ONLY=1; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Resolve the dataset root exactly as state_bench.io.build_dataset does.
DATASET_ROOT="$(uv run python -c "
import sys
from pathlib import Path
from state_bench.config import load_experiment
loaded = load_experiment(sys.argv[1])
print((Path(sys.argv[1]).resolve().parent / loaded.spec.dataset.output_dir).resolve())
" "$CONFIG")"
echo ">> Dataset root: $DATASET_ROOT"

if [[ "$TOKENIZE_ONLY" -eq 1 && "$SKIP_TOKENIZE" -eq 1 ]]; then
    echo "--tokenize-only and --skip-tokenize are mutually exclusive" >&2
    exit 1
fi

SENS_DIR="$DATASET_ROOT/rendered-sensitivity"

if [[ "$TOKENIZE_ONLY" -eq 1 ]]; then
    echo ">> Skipping builds and partition (--tokenize-only); reusing rendered/ and rendered-sensitivity/"
    if [[ ! -d "$SENS_DIR" ]]; then
        echo "missing $SENS_DIR; run without --tokenize-only first" >&2
        exit 1
    fi
else

if [[ "$SKIP_TEST" -eq 1 ]]; then
    echo ">> Skipping test build; reusing persisted semantics/test.ids"
else
    echo ">> Building test split from $CONFIG (needed for train disjointness)"
    uv run state-bench build "$CONFIG" --splits test
fi

echo ">> Building train split from $CONFIG"
uv run state-bench build "$CONFIG" --splits train

echo ">> Partitioning $PERIODIC_CONFIG train split into $SENS_BUCKETS sensitivity buckets"
rm -rf "$SENS_DIR"
uv run scripts/partition_train_by_sensitivity.py \
    "$DATASET_ROOT/rendered/$PERIODIC_CONFIG/train.jsonl" \
    --output "$SENS_DIR" \
    --buckets "$SENS_BUCKETS" \
    --field "$SENS_FIELD"

fi  # TOKENIZE_ONLY

if [[ "$SKIP_TOKENIZE" -eq 1 ]]; then
    echo ">> Skipping tokenization (--skip-tokenize)"
    echo ">> Later, resume with: bash scripts/build_train.sh --tokenize-only"
    exit 0
fi

TOKENS_DEST="$DATASET_ROOT/rendered-tokenized"
SENS_TOKENS_DEST="$DATASET_ROOT/rendered-sensitivity-tokenized"

echo ">> Tokenizing train splits into $TOKENS_DEST"
rm -rf "$TOKENS_DEST"
uv run --extra tokenization scripts/tokenize_dataset.py "$DATASET_ROOT/rendered" \
    --destination "$TOKENS_DEST" \
    --only-splits train

echo ">> Tokenizing sensitivity buckets into $SENS_TOKENS_DEST"
rm -rf "$SENS_TOKENS_DEST"
uv run --extra tokenization scripts/tokenize_dataset.py "$SENS_DIR" --destination "$SENS_TOKENS_DEST"

echo ">> Merging bucket token directories into $TOKENS_DEST/tokens/"
mv "$SENS_TOKENS_DEST/tokens/"* "$TOKENS_DEST/tokens/"
rm -rf "$SENS_TOKENS_DEST"

echo ">> Done. Training distributions in $TOKENS_DEST/tokens/:"
ls -1 "$TOKENS_DEST/tokens/"
