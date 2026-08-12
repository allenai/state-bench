#!/usr/bin/env python3
"""Render a canonical state-tracking sequence as a value-flow braid diagram."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical JSONL dataset input.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination image path; use a .pdf or .svg suffix for vector output.",
    )
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


_BACKGROUND = "#ffffff"
_TRACK_COLOR = "#e9e9ee"
_INK = "#2a2a30"
_STRAND_WIDTH = 3.4
_CASING_EXTRA = 3.0


def _value_palette(values: Sequence[int]) -> dict[int, tuple[float, float, float]]:
    """A calm, colorblind-aware qualitative palette keyed by observed value."""

    count = len(values)
    name = "colorblind" if count <= 10 else "husl"
    palette = sns.color_palette(name, n_colors=max(count, 1))
    return dict(zip(values, palette, strict=False))


def _smooth_crossing(x0: float, x1: float, y0: float, y1: float, samples: int = 72) -> tuple[Any, Any]:
    """Sample a smootherstep S-curve with horizontal tangents at both ends."""

    fraction = np.linspace(0.0, 1.0, samples)
    eased = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0)
    return x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * eased


def _draw_strand(axis: Any, xs: Any, ys: Any, color: Any, *, zorder: float) -> None:
    """Draw one colored strand cased in the background color for a woven look."""

    axis.plot(
        xs,
        ys,
        color=color,
        linewidth=_STRAND_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
        path_effects=[
            pe.Stroke(linewidth=_STRAND_WIDTH + _CASING_EXTRA, foreground=_BACKGROUND),
            pe.Normal(),
        ],
    )


def render_braid(record: dict[str, Any], *, output: Path, title: str | None) -> None:
    """Draw live values flowing through registers across extra assignments."""

    state, operations = initial_state(record)
    labels = variable_labels(int(record["num_variables"]), record.get("temporary_variable"))
    temporary_row = record.get("temporary_variable")
    observed_values = sorted(
        set(state)
        | {int(operation["value"]) for operation in operations if operation["op"] == "constant"}
    )
    colors = _value_palette(observed_values)

    sns.set_theme(style="white", context="paper")
    # Keep text as editable glyphs in vector output (.pdf / .svg).
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    max_step = len(operations)
    width = min(30.0, max(6.5, 0.3 * (max_step + 1) + 1.6))
    height = max(1.1, 0.17 * len(labels) + 0.6)
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(_BACKGROUND)
    axis.set_facecolor(_BACKGROUND)

    # Faint full-width track behind every register; the scratch register is
    # dashed so it reads as auxiliary rather than a task variable.
    for row in range(len(labels)):
        axis.plot(
            [-0.15, max_step + 0.15],
            [row, row],
            color=_TRACK_COLOR,
            linewidth=1.4,
            linestyle=(0, (1, 2)) if row == temporary_row else "-",
            zorder=0.5,
            solid_capstyle="round",
        )

    # Simulate once, collecting whole-run horizontal strands (so casing does not
    # chop them into dashes), smooth copy crossings, and constant markers.
    initial = state.copy()
    run_start = [0] * len(state)
    horizontal_runs: list[tuple[int, int, int, int]] = []
    crossings: list[tuple[int, int, int, int, int]] = []
    constants: list[tuple[int, int, int]] = []
    for step, operation in enumerate(operations, start=1):
        target = int(operation["target"])
        horizontal_runs.append((target, run_start[target], step - 1, state[target]))
        if operation["op"] == "variable":
            source = int(operation["source"])
            new_value = state[source]
            crossings.append((step - 1, step, source, target, new_value))
        elif operation["op"] == "constant":
            new_value = int(operation["value"])
            constants.append((step, target, new_value))
        else:
            raise ValueError(f"unknown operation type: {operation['op']!r}")
        state[target] = new_value
        run_start[target] = step
    for row in range(len(state)):
        horizontal_runs.append((row, run_start[row], max_step, state[row]))

    if state != [int(value) for value in record["final_state"]]:
        raise ValueError("evaluated braid state does not match the record's final state")

    # Carried values (drawn first, so copy crossings weave over them).
    for row, x_start, x_end, value in horizontal_runs:
        if x_end > x_start:
            _draw_strand(axis, [x_start, x_end], [row, row], colors[value], zorder=2)
    for x0, x1, source, target, value in crossings:
        crossing_x, crossing_y = _smooth_crossing(x0, x1, source, target)
        _draw_strand(axis, crossing_x, crossing_y, colors[value], zorder=3)
    # Squares mark every value introduced by a constant assignment: the initial
    # register values and each mid-trace constant. Copies carry no marker.
    for step, target, value in [*((0, row, value) for row, value in enumerate(initial)), *constants]:
        axis.scatter(
            step, target, s=54, marker="s", color=colors[value],
            edgecolor=_BACKGROUND, linewidth=1.1, zorder=5,
        )

    # Read off the final state at the right edge, colored to match each strand.
    final_label_x = max_step + 0.35
    for row, value in enumerate(state):
        axis.text(
            final_label_x, row, str(value), va="center", ha="left",
            color=colors[value], fontweight="bold", fontsize=9, zorder=6,
        )

    tick_count = min(12, max_step + 1)
    ticks = np.unique(np.linspace(0, max_step, tick_count, dtype=int))
    axis.set_xticks(ticks)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    for tick_label, row in zip(axis.get_yticklabels(), range(len(labels)), strict=True):
        if row == temporary_row:
            tick_label.set_style("italic")
            tick_label.set_color("#8a8a92")
    axis.set_xlim(-0.6, max_step + 1.1)
    axis.set_ylim(len(labels) - 0.4, -0.6)
    axis.set_xlabel("assignment step", color=_INK)
    axis.tick_params(axis="both", length=0, colors=_INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if title:
        axis.set_title(title, color=_INK, pad=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor=_BACKGROUND)
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
