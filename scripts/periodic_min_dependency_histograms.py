#!/usr/bin/env python3
"""Compare minimum queried dependency lengths across periodic datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import aesthetics as aes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def read_minimum_depths(path: Path, *, mixture: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            depths = record["final_dependency_lengths"]
            temporary = record.get("temporary_variable")
            query = [variable for variable in record["query"] if variable != temporary]
            minimum = min(int(depths[variable]) for variable in query)
            rows.append({"Aperiodic mix": mixture, "Minimum dependency length": minimum})
    if not rows:
        raise ValueError(f"no periodic records found in {path}")
    return rows


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
            for row in read_minimum_depths(path, mixture=label)
        ]
    )
    frame["Log minimum dependency length"] = np.log2(
        frame["Minimum dependency length"].to_numpy() + 1
    )
    max_depth = int(frame["Minimum dependency length"].max())
    bins = np.arange(-0.5, int(np.ceil(np.log2(max_depth + 1))) + 1.5, 1)

    palette = dict(zip(labels, sns.color_palette("deep", n_colors=len(labels)), strict=True))
    figure, axis = plt.subplots(figsize=(aes.PAPER_WIDTH_IN, aes.FIG_HEIGHT_DOUBLE_ROW_IN))
    for label in labels:
        sns.histplot(
            data=frame[frame["Aperiodic mix"] == label],
            x="Log minimum dependency length",
            bins=bins,
            stat="proportion",
            element="bars",
            fill=True,
            alpha=0.4,
            color=palette[label],
            label=label,
            ax=axis,
        )
    ticks = np.arange(0, len(bins) - 1)
    axis.set_xticks(ticks, ["0" if tick == 0 else f"{2**tick - 1:,}" for tick in ticks])
    axis.set(
        xlabel="Minimum queried dependency length (log2-binned)",
        ylabel="Proportion of periodic sequences",
    )
    axis.tick_params(axis="x", labelrotation=45)
    axis.legend(title="Aperiodic mix", frameon=False)
    sns.despine(fig=figure)
    figure.subplots_adjust(top=0.82)
    aes.set_figure_title(figure, "Minimum queried dependency length by aperiodic mixture")
    aes.save_figure(args.output, fig=figure, save_png=True)
    plt.close(figure)


if __name__ == "__main__":
    main()
