#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_COLUMNS = (
    "cluster_key",
    "magnification",
    "fine_cluster_id",
    "audit_buckets",
    "run_config_hash",
    "run_input_fingerprint",
    "audit_selection_sha256",
    "reviewer",
    "decision",
    "notes",
)
AUDIT_REVIEW_BINDING_FIELDS = (
    "run_config_hash",
    "run_input_fingerprint",
    "audit_selection_sha256",
)
AUTOMATIC_REVIEWER = "automatic_all_pass"
AUTOMATIC_NOTE = (
    "AUTOMATIC ALL-PASS: decisions were generated programmatically; "
    "no per-cluster human review was performed."
)


def load_and_validate_template(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input review template does not exist: {path}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path}: missing CSV header")
        duplicate_columns = sorted(
            column for column, count in Counter(fieldnames).items() if count > 1
        )
        if duplicate_columns:
            raise ValueError(f"{path}: duplicate columns: {duplicate_columns}")
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        if missing_columns:
            raise ValueError(f"{path}: missing required columns: {missing_columns}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(column) is None for column in REQUIRED_COLUMNS):
                raise ValueError(f"{path}: malformed CSV row {row_number}")
            rows.append(dict(row))

    if not rows:
        raise ValueError(f"{path}: review template contains no cluster rows")

    normalized_keys = [str(row["cluster_key"]).strip() for row in rows]
    if any(not key for key in normalized_keys):
        raise ValueError(f"{path}: cluster_key values must be nonempty")
    duplicate_keys = sorted(
        key for key, count in Counter(normalized_keys).items() if count > 1
    )
    if duplicate_keys:
        preview = duplicate_keys[:10]
        suffix = " ..." if len(duplicate_keys) > len(preview) else ""
        raise ValueError(f"{path}: duplicate cluster_key values: {preview}{suffix}")

    for field in AUDIT_REVIEW_BINDING_FIELDS:
        values = {str(row[field]).strip() for row in rows}
        if "" in values:
            raise ValueError(f"{path}: {field} must be nonempty on every row")
        if len(values) != 1:
            raise ValueError(f"{path}: {field} must be identical on every row")

    prefilled = [
        str(row["cluster_key"]).strip()
        for row in rows
        if str(row["reviewer"]).strip() or str(row["decision"]).strip()
    ]
    if prefilled:
        preview = prefilled[:10]
        suffix = " ..." if len(prefilled) > len(preview) else ""
        raise ValueError(
            f"{path}: refuses to replace existing reviewer or decision values for "
            f"cluster(s): {preview}{suffix}"
        )
    return fieldnames, rows


def write_csv_exclusive(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def complete_review(
    input_path: str | Path,
    output_path: str | Path,
    *,
    acknowledge_no_human_review: bool,
) -> int:
    if not acknowledge_no_human_review:
        raise ValueError(
            "Automatic all-pass requires explicit acknowledgement that no per-cluster "
            "human review was performed"
        )
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("Input and output must be different files; the template is never modified")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    fieldnames, rows = load_and_validate_template(source)
    for row in rows:
        row["reviewer"] = AUTOMATIC_REVIEWER
        row["decision"] = "pass"
        row["notes"] = AUTOMATIC_NOTE
    write_csv_exclusive(destination, fieldnames, rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new all-pass audit review CSV while explicitly recording that "
            "the decisions were generated automatically."
        )
    )
    parser.add_argument("--input", required=True, help="Unmodified audit_review_template.csv.")
    parser.add_argument("--output", required=True, help="New completed review CSV to create.")
    parser.add_argument(
        "--acknowledge-no-human-review",
        action="store_true",
        required=True,
        help="Required acknowledgement that no per-cluster human review was performed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        count = complete_review(
            args.input,
            args.output,
            acknowledge_no_human_review=args.acknowledge_no_human_review,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    destination = Path(args.output).expanduser().resolve()
    print(f"Created automatic all-pass review with {count:,} clusters: {destination}")
    print(
        "Next: python run_remote_offline.py finalize "
        f"--config CONFIG --review-csv {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
