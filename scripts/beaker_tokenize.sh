#!/usr/bin/env bash
# Start an interactive Beaker session for tokenizing a state-bench dataset.
# Mirrors the local `sinteractive` setup, with the development image pinned.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/beaker_tokenize.sh [cpus] [gpus] [cluster] [priority]

Start a remote, interactive Beaker session using jacksonp/beaker-dev.

Defaults: 1 CPU, 0 GPUs, ai2/saturn, urgent priority.
Example: scripts/beaker_tokenize.sh 64 0 ai2/saturn urgent
EOF
    exit 0
fi

cpus="${1:-1}"
gpus="${2:-0}"
cluster="${3:-ai2/saturn}"
priority="${4:-urgent}"
: "${WORKSPACE:?Set WORKSPACE to the Beaker workspace before running this script.}"

exec beaker session create \
    --cluster "$cluster" \
    --remote \
    --bare \
    --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
    --mount src=secret,ref=JACKSONP_ID_ED25519,dst=/root/.ssh/id_ed25519 \
    --mount src=secret,ref=JACKSONP_BASHRC,dst=/root/.bashrc \
    --mount src=secret,ref=JACKSONP_SSH_CONFIG,dst=/root/.ssh/config \
    --mount src=secret,ref=JACKSONP_GIT_CONFIG,dst=/root/.gitconfig \
    --priority "$priority" \
    --host-networking \
    --workspace "$WORKSPACE" \
    --timeout 8h \
    --cpus "$cpus" \
    --gpus "$gpus" \
    --image jacksonp/beaker-dev
