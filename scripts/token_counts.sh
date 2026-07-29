#!/usr/bin/env bash
# Report Dolma token totals per rendered dataset and split.
#
# Dolma's .npy shards may mix source files. Their matching .csv.gz metadata
# retains each sequence's source path and [start, end) token offsets, which is
# what this script sums.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/token_counts.sh TOKENIZED_DIR [DATASET] [SPLIT]

Print token totals from Dolma output, grouped by rendered dataset and split.

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

if ! find "$metadata_dir" -type f -name '*.csv.gz' -print -quit | grep -q .; then
    echo "No Dolma metadata files (*.csv.gz) found under $metadata_dir" >&2
    exit 1
fi

printf 'dataset\tsplit\ttokens\n'
find "$metadata_dir" -type f -name '*.csv.gz' -print0 \
    | xargs -0 gzip -cd \
    | awk -F, -v dataset_filter="$dataset_filter" -v split_filter="$split_filter" '
        # Dolma MemmapMetadata is written as: start,end,id,src,loc.
        NF == 5 {
            path_components = split($4, path, "/")
            if (path[path_components] ~ /^part-[0-9]+\.jsonl$/) {
                dataset = path[path_components - 2]
                split_name = path[path_components - 1]
            } else {
                dataset = path[path_components - 1]
                split_name = path[path_components]
                sub(/\.jsonl$/, "", split_name)
            }
            if ((dataset_filter == "" || dataset == dataset_filter) &&
                (split_filter == "" || split_name == split_filter)) {
                totals[dataset SUBSEP split_name] += $2 - $1
            }
        }
        END {
            for (key in totals) {
                split(key, fields, SUBSEP)
                printf "%s\t%s\t%.0f\n", fields[1], fields[2], totals[key]
            }
        }
    ' \
    | LC_ALL=C sort -t $'\t' -k1,1 -k2,2
