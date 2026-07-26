#!/usr/bin/env python3
"""Prepare rendered state-bench JSONL as Dolma documents and tokenize it.

This follows Step 3 of the OLMo cookbook's tokenization guide, using the
Dolma 2 tokenizer and the same tokenization settings.
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


def prepare_documents(dataset: Path, documents_dir: Path, fields: list[str], separator: str) -> int:
    jsonl_files = sorted(path for path in dataset.rglob("*.jsonl") if path.is_file())
    if not jsonl_files:
        raise ValueError(f"no JSONL files found under {dataset}")

    count = 0
    for source in jsonl_files:
        relative = source.relative_to(dataset)
        destination = documents_dir / relative
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
    destination.mkdir(parents=True)
    try:
        document_count = prepare_documents(dataset, documents_dir, fields, args.separator)
        print(f"Prepared {document_count:,} Dolma documents in {documents_dir}")
        if args.prepare_only:
            return 0

        from huggingface_hub import snapshot_download

        tokenizer_dir = destination / "tokenizer"
        snapshot_download(repo_id=args.tokenizer, local_dir=tokenizer_dir)
        command = [
            "dolma",
            "tokens",
            "--documents",
            str(documents_dir),
            "--destination",
            str(destination / "tokens"),
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
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    except BaseException:
        # Do not leave a partial output that a subsequent run mistakes for success.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
