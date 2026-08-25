#!/usr/bin/env python3
"""
Reconcile manually downloaded audio files into download_log_pilot.csv.

This is a one-time migration for rows that are still marked
download_success=False even though their validated audio files exist on disk.

The script is intentionally not part of the embedding pipeline. It:
    - updates only currently unsuccessful rows;
    - requires every unsuccessful row to have its expected audio file;
    - marks those rows as successful;
    - clears stale error_msg values;
    - preserves all rows, columns, and row order;
    - creates a backup before replacing the CSV atomically.

It does not inspect or modify audio content. The audio files are assumed to
have already been manually validated.

Example:
    python scripts/audio-acquisition/reconcile_manual_downloads.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = PROJECT_ROOT / "data" / "logs" / "download_log_pilot.csv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"
EXPECTED_ROWS = 10_000

REQUIRED_COLUMNS = {
    "row_idx",
    "download_success",
    "error_msg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark manually validated existing audio files successful in the download log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Download log CSV (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Audio directory (default: {DEFAULT_AUDIO_DIR})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_row_idx(value: str) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid row_idx: {value!r}") from exc
    if number < 0:
        raise ValueError(f"row_idx must be non-negative: {number}")
    return number


def expected_audio_path(audio_dir: Path, row_idx: int) -> Path:
    return audio_dir / f"{row_idx:06d}_opus.webm"


def read_log(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise SystemExit(
                "Download log is missing required columns: " + ", ".join(missing)
            )
        return list(reader), fieldnames


def validate_log_shape(rows: Sequence[Dict[str, str]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"Expected {EXPECTED_ROWS:,} log rows, found {len(rows):,}; refusing to modify the log."
        )

    row_indices = []
    for row in rows:
        try:
            row_indices.append(parse_row_idx(row.get("row_idx", "")))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    expected = set(range(EXPECTED_ROWS))
    actual = set(row_indices)
    if len(actual) != len(row_indices) or actual != expected:
        missing = sorted(expected - actual)
        outside = sorted(actual - expected)
        duplicates = sorted(
            index for index in actual if row_indices.count(index) > 1
        )
        raise SystemExit(
            "row_idx values are not exactly 0..9999: "
            f"missing={missing[:20]}, outside={outside[:20]}, duplicates={duplicates[:20]}"
        )


def reconcile_rows(rows: List[Dict[str, str]], audio_dir: Path) -> tuple[int, List[int]]:
    failed_rows = [row for row in rows if not parse_bool(row.get("download_success", ""))]
    missing_files: List[int] = []

    for row in failed_rows:
        row_idx = parse_row_idx(row["row_idx"])
        path = expected_audio_path(audio_dir, row_idx)
        if not path.is_file():
            missing_files.append(row_idx)

    if missing_files:
        raise SystemExit(
            "Refusing to modify the log because these unsuccessful rows have no "
            "expected audio file: "
            + ", ".join(map(str, sorted(missing_files)))
        )

    for row in failed_rows:
        row_idx = parse_row_idx(row["row_idx"])
        path = expected_audio_path(audio_dir, row_idx)
        row["download_success"] = "True"
        row["error_msg"] = ""

    return len(failed_rows), []


def write_log_atomically(
    path: Path,
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(
            temporary,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    log_path = resolve_path(args.log)
    audio_dir = resolve_path(args.audio_dir)

    if not log_path.is_file():
        raise SystemExit(f"Download log not found: {log_path}")
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    rows, fieldnames = read_log(log_path)
    validate_log_shape(rows)

    print("=" * 70)
    print("RECONCILE MANUAL DOWNLOADS")
    print("=" * 70)
    print(f"Log: {log_path}")
    print(f"Audio directory: {audio_dir}")
    print("Checking unsuccessful rows against existing audio files...")

    updated, _ = reconcile_rows(rows, audio_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = log_path.with_name(
        f"{log_path.stem}.before_manual_reconciliation_{timestamp}{log_path.suffix}"
    )
    shutil.copy2(log_path, backup_path)
    print(f"Backup created: {backup_path}")

    write_log_atomically(log_path, rows, fieldnames)

    successful = sum(parse_bool(row.get("download_success", "")) for row in rows)
    print(f"Rows updated: {updated:,}")
    print(f"Successful rows after reconciliation: {successful:,}/{len(rows):,}")
    print(f"Updated log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
