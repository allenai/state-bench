#!/usr/bin/env python3
"""Compare periodic-program sensitivity distributions with Seaborn facets."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

METRICS = {
    "sensitivity_density": "Sensitivity density",
    "sensitivity_coverage": "Sensitivity coverage",
    "mean_live_fanout": "Mean live fanout",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Canonical JSONL inputs; labels are inferred from their dataset directory names.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels, in the same order as the canonical JSONL inputs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG path.")
    return parser.parse_args(argv)


def read_periodic_rows(path: Path, *, mixture: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            for field, label in METRICS.items():
                rows.append(
                    {
                        "Aperiodic mix": mixture,
                        "Sensitivity measure": label,
                        "Value": float(record[field]),
                    }
                )
    if not rows:
        raise ValueError(f"no periodic records found in {path}")
    return rows


def infer_label(path: Path) -> str:
    """Infer an aperiodic-mixture label from a periodic dataset directory."""

    dataset_name = path.parents[1].name
    if dataset_name.endswith("-pure"):
        return "0% aperiodic mix"
    match = re.search(r"mixed-(\d+)pct", dataset_name)
    if match:
        return f"{match.group(1)}% aperiodic mix"
    return dataset_name


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels or [infer_label(path) for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must provide exactly one label for each input")
    if len(set(labels)) != len(labels):
        raise ValueError("dataset labels must be unique")
    frame = pd.DataFrame(
        [
            row
            for path, label in zip(args.inputs, labels, strict=True)
            for row in read_periodic_rows(path, mixture=label)
        ]
    )
    sns.set_theme(style="whitegrid", context="talk")
    mixtures = labels
    palette = dict(zip(mixtures, sns.color_palette("deep", n_colors=len(mixtures)), strict=True))
    figure = plt.figure(figsize=(15, 5), layout="constrained")
    grid = GridSpec(1, len(METRICS), figure=figure, wspace=0.1)
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[0, index], sharey=axes[0])
        for index in range(1, len(METRICS))
    )

    for axis, metric in zip(axes, METRICS.values(), strict=True):
        facet = frame[frame["Sensitivity measure"] == metric]
        values = facet["Value"].to_numpy()
        bins = np.linspace(values.min(), values.max(), 33)
        for mixture in mixtures:
            sns.histplot(
                data=facet[facet["Aperiodic mix"] == mixture],
                x="Value",
                bins=bins,
                stat="count",
                element="bars",
                fill=True,
                alpha=0.4,
                color=palette[mixture],
                label=mixture,
                ax=axis,
            )
        axis.set(title=metric, xlabel="Program sensitivity")
    axes[0].set_ylabel("Periodic sequences")
    for axis in axes[1:]:
        axis.set_ylabel("")
        axis.tick_params(axis="y", labelleft=False)
    axes[0].legend(title="Aperiodic mix")
    figure.suptitle("Periodic-program sensitivity by aperiodic mixture", y=1.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
