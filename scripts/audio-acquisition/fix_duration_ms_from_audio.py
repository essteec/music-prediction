#!/usr/bin/env python3
"""
Rewrite songs.csv duration_ms values from the corresponding audio files.

Mapping:
    metadata rank 1     -> 000000_opus.webm
    metadata rank 10000  -> 009999_opus.webm

The script probes every row before changing the CSV. If any rank, audio file,
or duration is invalid, it stops without replacing the metadata file. On a
successful run it creates a backup and atomically replaces songs.csv.

This script is intentionally separate from validation so the duration repair
can be reviewed and run explicitly.

Requirements:
    ffprobe must be available on PATH.

Example:
    python scripts/audio-acquisition/fix_duration_ms_from_audio.py
    python scripts/audio-acquisition/fix_duration_ms_from_audio.py --ranks 545 588 1026
    python scripts/audio-acquisition/fix_duration_ms_from_audio.py --ranks 545,588,1026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "songs.csv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"
RANK_MIN = 1
RANK_MAX = 10_000

REQUIRED_COLUMNS = {"rank", "duration_ms"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set songs.csv duration_ms from the matching audio files."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Audio directory (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create the default backup before replacing the metadata CSV.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        metavar="RANK",
        help=(
            "Only fix these metadata ranks. Accepts space-separated or comma-separated "
            "values, e.g. --ranks 545 588 1026 or --ranks 545,588,1026."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_rank(value: str) -> Optional[int]:
    try:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            return None
        return int(number)
    except (TypeError, ValueError):
        return None


def parse_rank_selection(values: Optional[Sequence[str]]) -> Optional[set[int]]:
    """Parse an optional mix of space-separated and comma-separated ranks."""
    if not values:
        return None

    selected: set[int] = set()
    invalid: List[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            rank = parse_rank(token)
            if rank is None or not RANK_MIN <= rank <= RANK_MAX:
                invalid.append(token)
            else:
                selected.add(rank)

    if invalid:
        raise SystemExit(
            "Invalid rank selection: " + ", ".join(invalid) +
            f". Ranks must be integers from {RANK_MIN} to {RANK_MAX}."
        )
    if not selected:
        raise SystemExit("--ranks was provided but no ranks were supplied.")
    return selected


def audio_path(audio_dir: Path, rank: int) -> Path:
    return audio_dir / f"{rank - 1:06d}_opus.webm"


def probe_duration_ms(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    seconds = float(payload["format"]["duration"])
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("ffprobe returned an invalid duration")
    return int(round(seconds * 1000))


def read_metadata(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise SystemExit(
                "Metadata CSV is missing required columns: " + ", ".join(missing)
            )
        return list(reader), fieldnames


def validate_ranks(rows: Sequence[Dict[str, str]]) -> None:
    ranks = [parse_rank(row.get("rank", "")) for row in rows]
    if any(rank is None for rank in ranks):
        raise SystemExit("Cannot map audio files: at least one rank is invalid.")

    integer_ranks = [rank for rank in ranks if rank is not None]
    expected = set(range(RANK_MIN, RANK_MAX + 1))
    actual = set(integer_ranks)
    if len(rows) != RANK_MAX or actual != expected or len(integer_ranks) != len(actual):
        missing = sorted(expected - actual)
        outside = sorted(actual - expected)
        duplicates = sorted(
            rank for rank in actual if integer_ranks.count(rank) > 1
        )
        raise SystemExit(
            "Rank shape is not exactly 1..10000: "
            f"rows={len(rows)}, missing={missing[:20]}, "
            f"outside={outside[:20]}, duplicates={duplicates[:20]}"
        )


def collect_duration_updates(
    rows: List[Dict[str, str]],
    audio_dir: Path,
    selected_ranks: Optional[set[int]],
) -> int:
    targets = [
        (index, row)
        for index, row in enumerate(rows, start=1)
        if selected_ranks is None or parse_rank(row.get("rank", "")) in selected_ranks
    ]
    if selected_ranks is not None:
        found_ranks = {
            rank
            for _, row in targets
            if (rank := parse_rank(row.get("rank", ""))) is not None
        }
        missing_ranks = sorted(selected_ranks - found_ranks)
        if missing_ranks:
            raise SystemExit(
                "Selected ranks are not present in the metadata: "
                + ", ".join(map(str, missing_ranks))
            )

    updates = 0
    for index, (metadata_index, row) in enumerate(targets, start=1):
        rank = parse_rank(row.get("rank", ""))
        if rank is None:
            raise SystemExit(f"Invalid rank at metadata row {metadata_index}.")

        path = audio_path(audio_dir, rank)
        if not path.exists():
            raise SystemExit(f"Audio file not found for rank {rank}: {path}")

        try:
            duration_ms = probe_duration_ms(path)
        except Exception as exc:
            raise SystemExit(f"Could not read duration for rank {rank}: {exc}") from exc

        new_value = str(duration_ms)
        if row.get("duration_ms", "") != new_value:
            updates += 1
        row["duration_ms"] = new_value

        if index == 1 or index % 100 == 0 or index == len(targets):
            print(f"Probed {index:,}/{len(targets):,} selected audio files...")

    return updates


def write_csv_atomically(
    metadata_path: Path,
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=metadata_path.parent,
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
        os.replace(temporary_path, metadata_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    audio_dir = resolve_path(args.audio_dir)

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required but was not found on PATH.")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    rows, fieldnames = read_metadata(metadata_path)
    validate_ranks(rows)
    selected_ranks = parse_rank_selection(args.ranks)

    print("=" * 70)
    print("FIX duration_ms FROM AUDIO")
    print("=" * 70)
    print(f"Metadata: {metadata_path}")
    print(f"Audio directory: {audio_dir}")
    if selected_ranks is None:
        print("Ranks: all")
    else:
        print(f"Ranks: {', '.join(map(str, sorted(selected_ranks)))}")
    print("Collecting all audio durations before changing the CSV...")
    updates = collect_duration_updates(rows, audio_dir, selected_ranks)

    backup_path = metadata_path.with_name(metadata_path.name + ".before_duration_ms_fix")
    if not args.no_backup:
        if backup_path.exists():
            print(f"Backup already exists; keeping it: {backup_path}")
        else:
            shutil.copy2(metadata_path, backup_path)
            print(f"Backup created: {backup_path}")

    write_csv_atomically(metadata_path, rows, fieldnames)
    print(f"Updated duration_ms values: {updates:,}")
    print(f"Rows written: {len(rows):,}")
    print(f"Updated metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
