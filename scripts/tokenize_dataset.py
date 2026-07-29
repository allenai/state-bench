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

from tqdm import tqdm

DEFAULT_TOKENIZER = "allenai/Olmo-3-1025-7B"
EOS_TOKEN_ID = 100257
PAD_TOKEN_ID = 100277
MAX_SIZE = 4_000_000_000
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024


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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum bytes per prepared JSONL shard (default: 64 MiB).",
    )
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


def prepare_document(
    source: Path,
    destination_dir: Path,
    fields: list[str],
    separator: str,
    chunk_size: int,
) -> tuple[int, int]:
    """Convert one rendered split into document-preserving JSONL shards."""
    count = 0
    pending_bytes = 0
    chunk_count = 0
    chunk_bytes = 0
    output_file = None
    destination_dir.mkdir(parents=True, exist_ok=True)

    def open_chunk() -> None:
        nonlocal chunk_bytes, chunk_count, output_file
        if output_file is not None:
            output_file.close()
        output_file = (destination_dir / f"part-{chunk_count:05d}.jsonl").open("wb")
        chunk_count += 1
        chunk_bytes = 0

    with (
        source.open("rb") as input_file,
        tqdm(
            total=source.stat().st_size,
            desc=f"Preparing {source.parent.name}/{source.name}",
            unit="B",
            unit_scale=True,
            mininterval=0.5,
            disable=not sys.stderr.isatty(),
        ) as progress,
    ):
        for line_number, line in enumerate(input_file, start=1):
            pending_bytes += len(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                document = make_document(record, fields, separator)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            encoded_document = (
                json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
            if output_file is None or (chunk_bytes and chunk_bytes + len(encoded_document) > chunk_size):
                open_chunk()
            output_file.write(encoded_document)
            chunk_bytes += len(encoded_document)
            count += 1
            if line_number % 10_000 == 0:
                progress.update(pending_bytes)
                pending_bytes = 0
        progress.update(pending_bytes)
    if output_file is not None:
        output_file.close()
    return count, chunk_count


def split_sort_key(dataset: Path, source: Path) -> tuple[int, str]:
    """Order all train splits before every non-train split."""
    relative = source.relative_to(dataset)
    return (0 if source.stem == "train" else 1, relative.as_posix())


def remove_prepared_documents(document_dir: Path, documents_dir: Path) -> None:
    """Remove tokenized intermediate shards and their empty parent directories."""
    shutil.rmtree(document_dir)
    parent = document_dir.parent
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
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

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
                prepared_count, _ = prepare_document(
                    source_path,
                    documents_dir / relative_path.parent / relative_path.stem,
                    fields,
                    args.separator,
                    args.chunk_size,
                )
                document_count += prepared_count
            print(f"Prepared {document_count:,} Dolma documents in {documents_dir}")
            return 0

        from huggingface_hub import snapshot_download

        tokenizer_dir = destination / "tokenizer"
        print(f"Downloading tokenizer {args.tokenizer}", flush=True)
        snapshot_download(
            repo_id=args.tokenizer,
            local_dir=tokenizer_dir,
            allow_patterns=["tokenizer.json"],
        )
        tokenizer_path = tokenizer_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise RuntimeError(f"{args.tokenizer} does not provide a tokenizer.json file")
        for index, source_path in enumerate(source_files, start=1):
            relative_path = source_path.relative_to(dataset)
            document_dir = documents_dir / relative_path.parent / relative_path.stem
            split_destination = destination / "tokens" / relative_path.parent / relative_path.stem
            print(f"Preparing {relative_path} ({index}/{len(source_files)})", flush=True)
            document_count, chunk_count = prepare_document(
                source_path, document_dir, fields, args.separator, args.chunk_size
            )
            print(
                f"Prepared {document_count:,} documents in {chunk_count:,} shards for {relative_path} "
                f"({index}/{len(source_files)})",
                flush=True,
            )
            command = [
                "dolma",
                "tokens",
                "--documents",
                str(document_dir / "*.jsonl"),
                "--destination",
                str(split_destination),
                "--tokenizer.name_or_path",
                str(tokenizer_path),
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
            remove_prepared_documents(document_dir, documents_dir)
    except BaseException:
        # Do not leave a partial output that a subsequent run mistakes for success.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
