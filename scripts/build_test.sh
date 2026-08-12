#!/usr/bin/env bash
# Build the test split locally, stage a Hugging Face repo, and (optionally)
# push it with --sync.
#
# By default this builds + stages and then previews the sync upload with
# --dry-run, so nothing is written to the Hub. Pass --push to actually upload.
# The repo id and visibility come from `huggingface.repository` in the config.

set -euo pipefail

CONFIG="${CONFIG:-configs/experiments/default.yaml}"
PUSH=0

usage() {
    cat <<'EOF'
Usage: bash scripts/build_test.sh [--push] [--config PATH]

Build the `test` split, stage a local Hugging Face repository, and preview (or
perform) a --sync upload.

Options:
  --push          Perform the real --sync upload (default: --dry-run preview).
  --config PATH   Experiment YAML (default: configs/experiments/default.yaml,
                  or the CONFIG environment variable).
  -h, --help      Show this help.

Steps:
  1. state-bench build CONFIG --splits test
  2. state-bench hub stage CONFIG        (needs the hub + analysis extras)
  3. state-bench hub push REPO --sync    (--dry-run unless --push is given)

Authenticate with `hf auth login` before running with --push.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH=1; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

echo ">> Building test split from $CONFIG"
uv run state-bench build "$CONFIG" --splits test

echo ">> Staging Hugging Face repository"
# hub stage prints the staged repo directory on stdout; keep only the last line
# so tqdm progress on stderr does not interfere.
repo_dir=$(uv run --extra hub --extra analysis state-bench hub stage "$CONFIG" | tail -n1)
echo ">> Staged at: $repo_dir"

if [[ "$PUSH" -eq 1 ]]; then
    echo ">> Pushing to the Hub with --sync"
    uv run --extra hub state-bench hub push "$repo_dir" --sync
else
    echo ">> Previewing --sync upload (--dry-run); pass --push to upload for real"
    uv run state-bench hub push "$repo_dir" --sync --dry-run
fi
