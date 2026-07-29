#!/usr/bin/env python3
"""Prepare rendered state-bench JSONL as Dolma documents and tokenize it.

This follows Step 3 of the OLMo cookbook's tokenization guide with the
configured Hugging Face tokenizer and the same tokenization settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any  # noqa: I001

DEFAULT_TOKENIZER = "allenai/Olmo-3-1025-7B"
EOS_TOKEN_ID = 100257
PAD_TOKEN_ID = 100277
MAX_SIZE = 4_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing generated JSONL sequences.")
    parser.add_argument(
        "--destination",
        type=Path,
        help="Output directory (default: a sibling directory named <dataset>-tokenized).",
    )
    parser.add_argument(
        "--text-fields",
        default="prefix,completion",
        help="Comma-separated fields joined into each training document (default: prefix,completion).",
    )
    parser.add_argument("--separator", default="\n", help="Text inserted between --text-fields.")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER, help=f"HF tokenizer (default: {DEFAULT_TOKENIZER})."
    )
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-size", type=int, default=MAX_SIZE)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write Dolma-formatted documents but do not download a tokenizer or run Dolma.",
    )
    return parser.parse_args()


def make_document(record: dict[str, Any], fields: list[str], separator: str) -> dict[str, Any]:
    """Convert a state-bench rendered record to the minimal Dolma document schema."""
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"record is missing requested text field(s): {', '.join(missing)}")
    values = [record[field] for field in fields]
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"requested text fields must all be strings: {', '.join(fields)}")
    identifier = record.get("example_id") or record.get("id")
    if not isinstance(identifier, str):
        raise ValueError("record needs a string example_id or id")
    return {"id": identifier, "text": separator.join(values), "source": "state-bench"}


def prepare_document(source: Path, destination: Path, fields: list[str], separator: str) -> int:
    """Convert one rendered JSONL split into one Dolma JSONL document file."""
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as input_file,
        destination.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                document = make_document(record, fields, separator)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            output_file.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def split_sort_key(dataset: Path, source: Path) -> tuple[int, str]:
    """Order all train splits before every non-train split."""
    relative = source.relative_to(dataset)
    return (0 if source.stem == "train" else 1, relative.as_posix())


def remove_prepared_document(document_path: Path, documents_dir: Path) -> None:
    """Remove a tokenized intermediate file and its empty parent directories."""
    document_path.unlink()
    parent = document_path.parent
    while parent != documents_dir:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset}")
    destination = (args.destination or dataset.with_name(f"{dataset.name}-tokenized")).resolve()
    if destination.exists():
        raise SystemExit(
            f"destination already exists: {destination}\nChoose --destination or remove it first."
        )
    fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
    if not fields:
        raise SystemExit("--text-fields must contain at least one field")

    documents_dir = destination / "documents"
    source_files = sorted(
        (path for path in dataset.rglob("*.jsonl") if path.is_file()),
        key=lambda path: split_sort_key(dataset, path),
    )
    if not source_files:
        raise SystemExit(f"no JSONL files found under {dataset}")
    destination.mkdir(parents=True)
    try:
        if args.prepare_only:
            document_count = 0
            for source_path in source_files:
                relative_path = source_path.relative_to(dataset)
                print(f"Preparing {relative_path}", flush=True)
                document_count += prepare_document(
                    source_path, documents_dir / relative_path, fields, args.separator
                )
            print(f"Prepared {document_count:,} Dolma documents in {documents_dir}")
            return 0

        from huggingface_hub import snapshot_download

        tokenizer_dir = destination / "tokenizer"
        print(f"Downloading tokenizer {args.tokenizer}", flush=True)
        snapshot_download(repo_id=args.tokenizer, local_dir=tokenizer_dir)
        for index, source_path in enumerate(source_files, start=1):
            relative_path = source_path.relative_to(dataset)
            document_path = documents_dir / relative_path
            split_destination = destination / "tokens" / relative_path.parent / relative_path.stem
            print(f"Preparing {relative_path} ({index}/{len(source_files)})", flush=True)
            document_count = prepare_document(source_path, document_path, fields, args.separator)
            print(
                f"Prepared {document_count:,} documents for {relative_path} ({index}/{len(source_files)})",
                flush=True,
            )
            command = [
                "dolma",
                "tokens",
                "--documents",
                str(document_path),
                "--destination",
                str(split_destination),
                "--tokenizer.name_or_path",
                str(tokenizer_dir / "tokenizer.json"),
                "--tokenizer.eos_token_id",
                str(EOS_TOKEN_ID),
                "--tokenizer.pad_token_id",
                str(PAD_TOKEN_ID),
                "--no-tokenizer.segment_before_tokenization",
                "--tokenizer.encode_special_tokens",
                "--processes",
                str(args.processes),
                "--max_size",
                str(args.max_size),
                "--sample_ring_prop",
                "--dtype",
                "uint32",
            ]
            print(
                f"Tokenizing {relative_path} ({index}/{len(source_files)}) into {split_destination}",
                flush=True,
            )
            subprocess.run(command, check=True)
            remove_prepared_document(document_path, documents_dir)
    except BaseException:
        # Do not leave a partial output that a subsequent run mistakes for success.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
