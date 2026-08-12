#!/usr/bin/env python3
"""Render a canonical state-tracking sequence as a value-flow braid diagram."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical JSONL dataset input.")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--instance-id",
        help="Render the periodic record with this instance ID.",
    )
    selection.add_argument(
        "--index",
        type=int,
        help="Render this zero-based periodic-record index.",
    )
    selection.add_argument(
        "--shortest",
        action="store_true",
        help="Render the shortest periodic record (preferring fewer variables on ties).",
    )
    parser.add_argument("--title", help="Optional plot title.")
    return parser.parse_args(argv)


def select_record(
    path: Path,
    *,
    instance_id: str | None,
    index: int | None,
    shortest: bool,
) -> dict[str, Any]:
    """Select one periodic record without retaining the full dataset in memory."""

    selected: dict[str, Any] | None = None
    selected_key: tuple[int, int] | None = None
    periodic_index = 0
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            if instance_id is not None and record["instance_id"] == instance_id:
                return record
            if index is not None and periodic_index == index:
                return record
            if shortest or (instance_id is None and index is None):
                key = (int(record["num_extra_assignments"]), int(record["num_variables"]))
                if selected_key is None or key < selected_key:
                    selected = record
                    selected_key = key
            periodic_index += 1

    if instance_id is not None:
        raise ValueError(f"no periodic record has instance ID {instance_id!r}")
    if index is not None:
        raise IndexError(f"periodic-record index {index} is out of range")
    if selected is None:
        raise ValueError(f"no periodic records found in {path}")
    return selected


def variable_labels(num_variables: int, temporary_variable: int | None) -> list[str]:
    """Use integer-code-style names for real variables and the scratch register."""

    labels = [chr(ord("a") + index) if index < 26 else f"v{index}" for index in range(num_variables)]
    if temporary_variable is not None:
        if temporary_variable != num_variables:
            raise ValueError("temporary variable must follow the real variables")
        labels.append("TMP")
    return labels


def initial_state(record: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    """Evaluate initialization and return the state plus remaining operations."""

    num_registers = int(record["num_variables"]) + (record.get("temporary_variable") is not None)
    operations = record["operations"]
    state: list[int | None] = [None] * num_registers
    for operation in operations[:num_registers]:
        if operation["op"] != "constant":
            raise ValueError("expected every initialization operation to assign a constant")
        state[int(operation["target"])] = int(operation["value"])
    if any(value is None for value in state):
        raise ValueError("initialization does not assign every register")
    return [int(value) for value in state], operations[num_registers:]


def render_braid(record: dict[str, Any], *, output: Path, title: str | None) -> None:
    """Draw live values flowing through registers across extra assignments."""

    state, operations = initial_state(record)
    labels = variable_labels(int(record["num_variables"]), record.get("temporary_variable"))
    observed_values = sorted(
        set(state)
        | {
            int(operation["value"])
            for operation in operations
            if operation["op"] == "constant"
        }
    )
    colors = dict(
        zip(
            observed_values,
            sns.color_palette("husl", n_colors=len(observed_values)),
            strict=True,
        )
    )

    sns.set_theme(style="whitegrid", context="talk")
    width = min(40.0, max(12.0, 0.22 * (len(operations) + 1)))
    height = max(4.0, 0.72 * len(labels) + 1.8)
    figure, axis = plt.subplots(figsize=(width, height))

    for row, value in enumerate(state):
        axis.scatter(0, row, s=42, color=colors[value], edgecolor="white", linewidth=0.6, zorder=4)

    for step, operation in enumerate(operations, start=1):
        previous = state.copy()
        target = int(operation["target"])
        for row, value in enumerate(previous):
            if row != target:
                axis.plot(
                    [step - 1, step],
                    [row, row],
                    color=colors[value],
                    linewidth=1.35,
                    alpha=0.72,
                    solid_capstyle="round",
                    zorder=2,
                )

        if operation["op"] == "variable":
            source = int(operation["source"])
            state[target] = previous[source]
            axis.plot(
                [step - 1, step],
                [source, target],
                color=colors[state[target]],
                linewidth=2.5,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )
            marker = "o"
        elif operation["op"] == "constant":
            state[target] = int(operation["value"])
            marker = "s"
        else:
            raise ValueError(f"unknown operation type: {operation['op']!r}")
        axis.scatter(
            step,
            target,
            s=34,
            marker=marker,
            color=colors[state[target]],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

    if state != [int(value) for value in record["final_state"]]:
        raise ValueError("evaluated braid state does not match the record's final state")

    max_step = len(operations)
    tick_count = min(12, max_step + 1)
    ticks = np.unique(np.linspace(0, max_step, tick_count, dtype=int))
    axis.set_xticks(ticks)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlim(-0.5, max_step + 0.5)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set(
        xlabel="Extra assignment (time step)",
        ylabel="Variable",
        title=title
        or (
            f"Value-flow braid: {record['num_variables']} real variables, "
            f"{len(operations):,} extra assignments"
        ),
    )
    axis.grid(axis="x", alpha=0.12)
    axis.grid(axis="y", alpha=0.25)

    value_handles = [
        Line2D(
            [0],
            [0],
            color=colors[value],
            marker="o",
            linewidth=2,
            markersize=6,
            label=str(value),
        )
        for value in observed_values
    ]
    figure.legend(
        handles=value_handles,
        title="Value",
        bbox_to_anchor=(1.0, 0.9),
        loc="upper left",
        frameon=False,
        ncols=max(1, (len(value_handles) + 9) // 10),
    )
    operation_handles = [
        Line2D([0], [0], marker="o", color="0.35", linewidth=0, label="Copied value"),
        Line2D([0], [0], marker="s", color="0.35", linewidth=0, label="New constant"),
    ]
    axis.legend(
        handles=operation_handles,
        title="Assignment",
        bbox_to_anchor=(0.5, -0.43),
        loc="upper center",
        frameon=False,
        ncols=2,
    )

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    record = select_record(
        args.input,
        instance_id=args.instance_id,
        index=args.index,
        shortest=args.shortest,
    )
    render_braid(record, output=args.output, title=args.title)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "instance_id": record["instance_id"],
                "num_variables": record["num_variables"],
                "num_extra_assignments": record["num_extra_assignments"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
