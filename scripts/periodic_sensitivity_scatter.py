#!/usr/bin/env python3
"""Scatter sensitivity coverage against sequence length across periodic datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import aesthetics as aes
import matplotlib.pyplot as plt
import seaborn as sns


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
    parser.add_argument("--output", type=Path, required=True, help="Destination path stem or file.")
    return parser.parse_args(argv)


def infer_label(path: Path) -> str:
    """Infer an aperiodic-mixture label from a periodic dataset directory."""

    dataset_name = path.parents[1].name
    if dataset_name.endswith("-pure"):
        return "0% aperiodic mix"
    match = re.search(r"mixed-(\d+)pct", dataset_name)
    if match:
        return f"{match.group(1)}% aperiodic mix"
    return dataset_name


def read_points(path: Path) -> tuple[list[int], list[float]]:
    lengths: list[int] = []
    coverages: list[float] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            lengths.append(int(record["num_assignments"]))
            coverages.append(float(record["sensitivity_coverage"]))
    if not lengths:
        raise ValueError(f"no periodic records found in {path}")
    return lengths, coverages


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels or [infer_label(path) for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must provide exactly one label for each input")
    if len(set(labels)) != len(labels):
        raise ValueError("dataset labels must be unique")

    palette = dict(zip(labels, sns.color_palette("deep", n_colors=len(labels)), strict=True))
    figure, axis = plt.subplots(figsize=(aes.PAPER_WIDTH_IN, aes.FIG_HEIGHT_DOUBLE_ROW_IN))
    for path, label in zip(args.inputs, labels, strict=True):
        lengths, coverages = read_points(path)
        axis.scatter(
            lengths,
            coverages,
            s=6,
            alpha=0.25,
            linewidths=0,
            color=palette[label],
            label=label,
        )
    axis.set(
        xscale="log",
        xlabel="Assignments per sequence",
        ylabel="Sensitivity coverage",
        ylim=(-0.02, 1.02),
    )
    legend = axis.legend(
        title="Aperiodic mix",
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([20.0])
    sns.despine(fig=figure)
    figure.subplots_adjust(top=0.82)
    aes.set_figure_title(figure, "Sensitivity coverage by sequence length")
    aes.save_figure(args.output, fig=figure, save_png=True)
    plt.close(figure)


if __name__ == "__main__":
    main()
