#!/usr/bin/env bash
# Build and tokenize the ~500M-token train mixture inside a Beaker job.
#
# Runs from the root of a state-bench checkout (configs/beaker/train-mixture-500m.yaml
# clones one and calls this script). Each of the three slices -- integer code,
# spreadsheet cells, and cube painting -- is built and tokenized by
# scripts/build_train.sh into its own directory under data/, which is a symlink
# to OUTPUT_ROOT on Weka. The slices' token directories are then gathered into
# one data/state-tracking-mixture-500m/rendered-tokenized/tokens/ tree.

set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent directory, e.g. on Weka.}"
WORKERS="${WORKERS:-$(nproc)}"
MIXTURE_CONFIGS=(
    configs/experiments/train_mixture_integer_code.yaml
    configs/experiments/train_mixture_spreadsheet.yaml
    configs/experiments/train_mixture_cube.yaml
)
MIXTURE_ROOT=data/state-tracking-mixture-500m

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo ">> uv not found; installing it"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# The tokenization extra builds jopetty/dolma's Rust extension from source,
# and its libz-ng-sys dependency shells out to cmake. jacksonp/beaker-dev ships
# cmake and a C toolchain but no Rust, so install a minimal one when missing.
if ! command -v cargo >/dev/null; then
    echo ">> cargo not found; installing a minimal Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
for tool in cargo cmake; do
    if ! command -v "$tool" >/dev/null; then
        echo "$tool is required to build dolma; use an image that provides it" >&2
        exit 1
    fi
done

echo ">> Commit: $(git rev-parse HEAD)"
echo ">> Writing data to $OUTPUT_ROOT with $WORKERS workers"
mkdir -p "$OUTPUT_ROOT"
rm -rf data
ln -s "$OUTPUT_ROOT" data

# Build dolma now so a toolchain problem fails in minutes, not after generation.
# build_train.sh re-syncs this environment from uv's cache before tokenizing.
echo ">> Building the tokenization environment"
uv sync --extra tokenization
uv run --no-sync python -c "import huggingface_hub; from dolma.tokenizer import tokenize_in_parallel"

for config in "${MIXTURE_CONFIGS[@]}"; do
    echo ">> Building $config"
    WORKERS="$WORKERS" bash scripts/build_train.sh --config "$config" --skip-partition
done

echo ">> Gathering slice token directories into $MIXTURE_ROOT/rendered-tokenized/tokens/"
rm -rf "$MIXTURE_ROOT/rendered-tokenized"
mkdir -p "$MIXTURE_ROOT/rendered-tokenized/tokens"
for slice in integer-code spreadsheet cube; do
    mv "$MIXTURE_ROOT/$slice/rendered-tokenized/tokens/"* "$MIXTURE_ROOT/rendered-tokenized/tokens/"
done

bash scripts/token_counts.sh "$MIXTURE_ROOT/rendered-tokenized" | tee /results/token_counts.tsv
awk -F'\t' 'NR > 1 { total += $3 } END { printf ">> Total tokens: %.0f\n", total }' \
    /results/token_counts.tsv
