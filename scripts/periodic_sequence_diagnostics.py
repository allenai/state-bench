#!/usr/bin/env python3
"""Create periodic-generation diagnostics from a canonical JSONL split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical JSONL file containing periodic records.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] == "periodic":
                records.append(record)
    if not records:
        raise ValueError(f"no periodic canonical records found in {path}")
    return records


def last_literal_agreement(record: dict[str, object]) -> float:
    final_state = list(record["final_state"])
    last_literals: list[int | None] = [None] * len(final_state)
    for operation in record["operations"]:
        if operation["op"] == "constant":
            last_literals[int(operation["target"])] = int(operation["value"])
    query = [int(variable) for variable in record["query"]]
    return float(
        np.mean([final_state[variable] == last_literals[variable] for variable in query])
    )


def sequence_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for record in records:
        depths = [int(depth) for depth in record["final_dependency_lengths"]]
        query = [int(variable) for variable in record["query"]]
        queried_depths = [depths[variable] for variable in query]
        final_state = list(record["final_state"])
        rows.append(
            {
                "variables": int(record["num_variables"]),
                "assignments": int(record["num_assignments"]),
                "min_depth": min(queried_depths),
                "median_depth": float(np.median(queried_depths)),
                "max_depth": max(queried_depths),
                "distinct_final_values": len({final_state[variable] for variable in query}),
                "last_literal_agreement": last_literal_agreement(record),
            }
        )
    return pd.DataFrame(rows)


def plot_minimum_depth_histogram(frame: pd.DataFrame, output: Path) -> None:
    """Plot one minimum queried-variable depth per periodic sequence."""

    transformed = np.log2(frame["min_depth"].to_numpy() + 1)
    upper = int(np.ceil(transformed.max())) + 1
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.histplot(transformed, bins=np.arange(-0.5, upper + 0.5, 1), color="#6f3fa0", ax=axis)
    ticks = np.arange(upper)
    axis.set_xticks(ticks, ["0" if tick == 0 else f"{2**tick - 1:,}" for tick in ticks], rotation=45)
    axis.set(
        title="Minimum queried-variable dependency length",
        xlabel="Minimum dependency length per sequence (log2-binned)",
        ylabel="Periodic sequences",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dependency_profiles(frame: pd.DataFrame, output: Path) -> None:
    profile = frame.melt(
        id_vars="variables",
        value_vars=["min_depth", "median_depth", "max_depth"],
        var_name="summary",
        value_name="depth",
    )
    profile["log_depth"] = np.log2(profile["depth"] + 1)
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=profile,
        x="variables",
        y="log_depth",
        hue="summary",
        palette="deep",
        ax=axis,
    )
    axis.set(
        title="Per-sequence dependency profile",
        xlabel="Variables in state",
        ylabel="Dependency length",
    )
    ticks = np.arange(0, int(np.ceil(profile["log_depth"].max())) + 1)
    axis.set_yticks(ticks, [str(2**tick - 1) for tick in ticks])
    axis.legend(title="Queried-variable summary", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_shortcut_diagnostics(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=frame, x="variables", y="last_literal_agreement", color="#6f3fa0", ax=axes[0])
    axes[0].set(
        title="Last-literal heuristic agreement",
        xlabel="Variables in state",
        ylabel="Fraction of queried coordinates",
        ylim=(-0.02, 1.02),
    )
    sns.boxplot(data=frame, x="variables", y="distinct_final_values", color="#3d8f7a", ax=axes[1])
    axes[1].set(
        title="Final value diversity",
        xlabel="Variables in state",
        ylabel="Distinct queried values",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = read_records(args.input)
    frame = sequence_frame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    max_assignments = max(int(record["target_extra_assignments"]) for record in records)
    suffix = f"max-assignments-{max_assignments}"
    plot_minimum_depth_histogram(frame, args.output_dir / f"dependency_histogram-{suffix}.png")
    plot_dependency_profiles(frame, args.output_dir / f"dependency_profiles-{suffix}.png")
    plot_shortcut_diagnostics(frame, args.output_dir / f"shortcut_diagnostics-{suffix}.png")
    summary = {
        "periodic_sequences": len(frame),
        "minimum_query_depth": int(frame["min_depth"].min()),
        "median_query_depth": float(frame["median_depth"].median()),
        "maximum_query_depth": int(frame["max_depth"].max()),
        "mean_last_literal_agreement": float(frame["last_literal_agreement"].mean()),
        "minimum_distinct_final_values": int(frame["distinct_final_values"].min()),
    }
    (args.output_dir / f"summary-{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
