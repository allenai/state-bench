"""Command-line interface for state-bench."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from state_bench.config import load_experiment
from state_bench.hub import (
    push_hub_repository,
    stage_hub_repository,
    validate_hub_repository,
)
from state_bench.io import build_dataset
from state_bench.preview import available_config_names, format_preview, generate_preview
from state_bench.stats import DEFAULT_TOKENIZER, analyze_path, format_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate formatter-independent state-tracking datasets.")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Configuration operations.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    check = config_commands.add_parser("check", help="Validate and summarize an experiment YAML file.")
    check.add_argument("config", type=Path)

    build = commands.add_parser("build", help="Generate canonical and rendered JSONL files.")
    build.add_argument("config", type=Path)
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--workers", type=int, default=1, help="Process workers for source generation.")

    sample = commands.add_parser("sample", help="Print one generated sample for a rendered config.")
    sample.add_argument("config", type=Path, help="Experiment YAML file.")
    sample.add_argument("config_name", nargs="?", help="Config such as integer-code--periodic.")
    sample.add_argument("--seed", type=int, help="Preview seed; defaults to the experiment seed.")
    sample.add_argument("--num-variables", type=int)
    sample.add_argument(
        "--extra-assignments",
        type=int,
        help="Generate an exact assignment depth instead of sampling the configured schedule.",
    )
    sample.add_argument("--filler-proportion", type=float)
    sample.add_argument("--filler-distraction", choices=("easy", "medium", "hard"))
    sample.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    sample.add_argument("--list-configs", action="store_true", help="List available config names.")

    stats = commands.add_parser("stats", help="Summarize a rendered JSONL or Parquet dataset.")
    stats.add_argument("dataset", type=Path)
    stats.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    stats.add_argument("--revision")
    stats.add_argument("--add-special-tokens", action="store_true")
    stats.add_argument("--joiner", default="")
    stats.add_argument("--limit", type=int)
    stats.add_argument("--batch-size", type=int, default=1024)
    stats.add_argument("--histogram-bin-width", type=int, default=10)
    stats.add_argument("--group-by", nargs="*", default=[])
    stats.add_argument("--json", action="store_true")

    hub = commands.add_parser("hub", help="Hugging Face staging and publishing.")
    hub_commands = hub.add_subparsers(dest="hub_command", required=True)

    stage = hub_commands.add_parser("stage", help="Build a local Hugging Face repository.")
    stage.add_argument("config", type=Path)
    stage.add_argument("--source-dir", type=Path)
    stage.add_argument("--workers", type=int, help="Override huggingface.staging_workers.")
    stage.add_argument(
        "--repo-dir",
        type=Path,
        help="Defaults to a sibling build/<source-directory-name> path.",
    )

    validate = hub_commands.add_parser("validate", help="Validate a staged repository offline.")
    validate.add_argument("repo_dir", type=Path)

    push = hub_commands.add_parser("push", help="Explicitly upload a staged repository.")
    push.add_argument("repo_dir", type=Path)
    push.add_argument("--repo-id")
    visibility = push.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true")
    visibility.add_argument("--private", action="store_true")
    push.add_argument("--dry-run", action="store_true")
    push.add_argument(
        "--sync",
        action="store_true",
        help="Delete remote files absent from the staged repository before uploading.",
    )
    push.add_argument("--commit-message", default="Upload state-bench dataset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config" and args.config_command == "check":
        loaded = load_experiment(args.config)
        summary = {
            "dataset": loaded.spec.dataset.name,
            "splits": list(loaded.spec.splits),
            "formatters": [formatter.name for formatter in loaded.formatters],
            "conditions": sum(
                len(condition.num_variables) for condition in loaded.spec.generation.conditions
            ),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "build":
        loaded = load_experiment(args.config)
        print(build_dataset(loaded, output_dir=args.output_dir, workers=args.workers))
        return 0

    if args.command == "sample":
        loaded = load_experiment(args.config)
        if args.list_configs:
            print("\n".join(available_config_names(loaded)))
            return 0
        if args.config_name is None:
            raise ValueError("config_name is required unless --list-configs is used")
        preview = generate_preview(
            loaded,
            config_name=args.config_name,
            seed=args.seed,
            num_variables=args.num_variables,
            extra_assignments=args.extra_assignments,
            filler_proportion=args.filler_proportion,
            filler_distraction=args.filler_distraction,
        )
        if args.json:
            print(json.dumps(preview.to_record(), indent=2))
        else:
            print(format_preview(preview))
        return 0

    if args.command == "stats":
        report = analyze_path(
            args.dataset,
            tokenizer_name=args.tokenizer,
            revision=args.revision,
            add_special_tokens=args.add_special_tokens,
            joiner=args.joiner,
            limit=args.limit,
            batch_size=args.batch_size,
            histogram_bin_width=args.histogram_bin_width,
            group_by=tuple(args.group_by),
        )
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_report(report))
        return 0

    if args.command == "hub" and args.hub_command == "stage":
        loaded = load_experiment(args.config)
        source_dir = args.source_dir or _default_output_dir(loaded.path, loaded.spec.dataset.output_dir)
        repo_dir = args.repo_dir or _default_repo_dir(source_dir)
        print(
            stage_hub_repository(
                loaded,
                source_dir=source_dir,
                repo_dir=repo_dir,
                workers=args.workers,
            )
        )
        return 0

    if args.command == "hub" and args.hub_command == "validate":
        print(json.dumps(validate_hub_repository(args.repo_dir), indent=2, sort_keys=True))
        return 0

    if args.command == "hub" and args.hub_command == "push":
        manifest = json.loads((args.repo_dir / "manifest.json").read_text(encoding="utf-8"))
        repository = manifest["huggingface"]["repository"]
        repo_id = args.repo_id or repository["repo_id"]
        if repo_id is None:
            raise ValueError("repo_id is required in YAML or via --repo-id")
        if args.public:
            private = False
        elif args.private:
            private = True
        else:
            private = bool(repository["private"])
        plan = push_hub_repository(
            args.repo_dir,
            repo_id=repo_id,
            private=private,
            dry_run=args.dry_run,
            sync=args.sync,
            commit_message=args.commit_message,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unhandled command")


def _default_output_dir(experiment_path: Path, output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return (experiment_path.resolve().parent / output_dir).resolve()


def _default_repo_dir(source_dir: Path) -> Path:
    """Mirror data/<dataset> as build/<dataset> when no staging path is given."""

    resolved = source_dir.resolve()
    if resolved.parent.name == "data":
        return resolved.parent.with_name("build") / resolved.name
    return Path.cwd().resolve() / "build" / resolved.name


if __name__ == "__main__":
    raise SystemExit(main())
