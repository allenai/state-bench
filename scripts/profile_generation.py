"""Run a bounded CPU profile of the production-scale train generation path."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import signal
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from state_bench.config import load_experiment
from state_bench.io import build_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "configs/experiments/long-context-train-profile.yaml"


class _ProfileDeadline(Exception):
    """Stop a build at the requested profiling deadline."""


def _deadline(_signum: int, _frame: FrameType | None) -> None:
    raise _ProfileDeadline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", type=Path, help="Optional output path for cProfile data.")
    parser.add_argument("--limit", type=int, default=40, help="Number of profile rows to print.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    loaded = load_experiment(args.experiment)
    profiler = cProfile.Profile()
    previous_handler = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        with tempfile.TemporaryDirectory(prefix="state-bench-profile-") as output_dir:
            output_path = Path(output_dir)
            profiler.enable()
            try:
                build_dataset(loaded, output_dir=output_path, workers=args.workers)
            except _ProfileDeadline:
                pass
            finally:
                profiler.disable()
            rendered_files = tuple((output_path / "rendered").glob("*/*.jsonl"))
            rendered_rows = sum(_count_lines(path) for path in rendered_files)
            rendered_bytes = sum(path.stat().st_size for path in rendered_files)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    if args.profile is not None:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile)
    print(
        f"profiled_seconds={args.seconds:g} rendered_rows={rendered_rows} "
        f"rendered_bytes={rendered_bytes}"
    )
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(args.limit)
    return 0


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


if __name__ == "__main__":
    raise SystemExit(main())
