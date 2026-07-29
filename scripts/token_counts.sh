#!/usr/bin/env bash
# Report Dolma token totals per rendered dataset and split.
#
# Each split has its own uint32 .npy shards. Their exact token count is the
# sum of their file sizes divided by four, avoiding a full scan of per-document
# .csv.gz metadata.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/token_counts.sh TOKENIZED_DIR [DATASET] [SPLIT]

Print exact token totals from split-specific Dolma uint32 .npy shards.

Examples:
  scripts/token_counts.sh data/state-tracking-long-context-v1/rendered-tokenized
  scripts/token_counts.sh data/state-tracking-long-context-v1/rendered-tokenized integer-code--aperiodic
  scripts/token_counts.sh data/state-tracking-long-context-v1/rendered-tokenized integer-code--aperiodic train
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 || $# -gt 3 ]]; then
    usage
    [[ $# -eq 0 ]] && exit 1
    exit 0
fi

tokenized_dir=$1
dataset_filter=${2:-}
split_filter=${3:-}
metadata_dir="$tokenized_dir/tokens"

if [[ ! -d "$metadata_dir" ]]; then
    echo "Dolma token directory not found: $metadata_dir" >&2
    exit 1
fi

if ! find "$metadata_dir" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "No Dolma token shards (*.npy) found under $metadata_dir" >&2
    exit 1
fi

if stat -c %s /dev/null >/dev/null 2>&1; then
    file_size() { stat -c %s "$1"; }
else
    file_size() { stat -f %z "$1"; }
fi

printf 'dataset\tsplit\ttokens\n'
while IFS= read -r -d '' shard; do
    relative_path=${shard#"$metadata_dir"/}
    IFS=/ read -r dataset split_name _ <<< "$relative_path"
    if [[ -z "$dataset" || -z "$split_name" ]]; then
        echo "Token shard is not in tokens/<dataset>/<split>/: $shard" >&2
        exit 1
    fi
    size=$(file_size "$shard")
    if ((size % 4 != 0)); then
        echo "Token shard size is not divisible by uint32 width: $shard" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$dataset" "$split_name" "$((size / 4))"
done < <(find "$metadata_dir" -type f -name '*.npy' -print0) \
    | awk -F'\t' -v dataset_filter="$dataset_filter" -v split_filter="$split_filter" '
        (dataset_filter == "" || $1 == dataset_filter) &&
        (split_filter == "" || $2 == split_filter) {
            totals[$1 SUBSEP $2] += $3
        }
        END {
            for (key in totals) {
                split(key, fields, SUBSEP)
                printf "%s\t%s\t%.0f\n", fields[1], fields[2], totals[key]
            }
        }
    ' \
    | LC_ALL=C sort -t $'\t' -k1,1 -k2,2
