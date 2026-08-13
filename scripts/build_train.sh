#!/usr/bin/env bash
# Build the train split locally. There is no Hugging Face upload here; see
# build_test.sh for staging + pushing the test split.
#
# Training is `disjoint_from: [test]` in the experiment config, and
# disjointness is enforced from semantic IDs a prior build persisted under
# <output_dir>/semantics/test.ids (see state_bench.io.build_dataset). That
# file is only written by a build that includes the `test` split, so this
# script builds `test` first unless --skip-test says the persisted IDs are
# already current.

set -euo pipefail

CONFIG="${CONFIG:-configs/experiments/default.yaml}"
SKIP_TEST=0

usage() {
    cat <<'EOF'
Usage: bash scripts/build_train.sh [--skip-test] [--config PATH]

Build the `test` split (for its persisted semantic IDs) and then the `train`
split, which is generated disjoint from `test`.

Options:
  --skip-test     Skip building `test`; reuse the semantics/test.ids already
                   persisted in the output directory from a prior build.
  --config PATH   Experiment YAML (default: configs/experiments/default.yaml,
                  or the CONFIG environment variable).
  -h, --help      Show this help.

Steps:
  1. state-bench build CONFIG --splits test    (skipped with --skip-test)
  2. state-bench build CONFIG --splits train
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-test) SKIP_TEST=1; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ "$SKIP_TEST" -eq 1 ]]; then
    echo ">> Skipping test build; reusing persisted semantics/test.ids"
else
    echo ">> Building test split from $CONFIG (needed for train disjointness)"
    uv run state-bench build "$CONFIG" --splits test
fi

echo ">> Building train split from $CONFIG"
uv run state-bench build "$CONFIG" --splits train
