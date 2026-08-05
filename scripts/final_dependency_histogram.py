#!/usr/bin/env python3
"""Plot the distribution of final dependency lengths in a rendered JSONL split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Rendered JSONL file containing final_dependency_lengths.")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path.")
    return parser.parse_args(argv)


def read_depths(path: Path) -> list[int]:
    depths: list[int] = []
    with path.open() as handle:
        for line in handle:
            depths.extend(json.loads(line)["final_dependency_lengths"])
    if not depths:
        raise ValueError(f"no final dependency lengths found in {path}")
    return depths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    depths = read_depths(args.input)
    log_depths = np.log2(np.asarray(depths) + 1)
    maximum = max(depths)
    upper_edge = int(np.ceil(np.log2(maximum + 1))) + 1
    bins = np.arange(-0.5, upper_edge + 0.5, 1)

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.histplot(log_depths, bins=bins, color="#6f3fa0", edgecolor="white", linewidth=0.7, ax=axis)
    tick_positions = np.arange(upper_edge)
    tick_labels = ["0" if value == 0 else f"{2**value - 1:,}" for value in tick_positions]
    axis.set(
        title="Periodic integer-code test: final dependency lengths",
        xlabel="Final dependency length (log2-binned)",
        ylabel="Final variables",
        xticks=tick_positions,
        xticklabels=tick_labels,
    )
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.text(
        0.99,
        0.96,
        f"{len(depths):,} final variables\n{depths.count(0) / len(depths):.1%} at depth 0",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize="medium",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
