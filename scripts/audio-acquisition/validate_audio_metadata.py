#!/usr/bin/env python3
"""
Validate the 10k audio files against the extended songs metadata dataset.

The validator is read-only by default. It compares audio duration with the
``duration`` and ``duration_ms`` columns in data/processed/songs.csv, reports
the first songs without lyrics, and ranks songs by lyric punctuation frequency.

Use --write-tags explicitly to write standard metadata tags into the WebM
files. The download log and metadata CSV are never modified.

Audio filenames are zero-based while metadata ranks are one-based:
    rank 1     -> 000000_opus.webm
    rank 10000  -> 009999_opus.webm

Requirements:
    ffprobe and ffmpeg must be available on PATH.

Examples:
    python scripts/audio-acquisition/validate_audio_metadata.py
    python scripts/audio-acquisition/validate_audio_metadata.py --write-tags
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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "songs.csv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"

RANK_MIN = 1
RANK_MAX = 10_000
DURATION_TOLERANCE_SECONDS = 10.0
LYRIC_PUNCTUATION = ("*", "–", "—", "?", ".")
REQUIRED_COLUMNS = {
    "rank",
    "track_name",
    "track_id",
    "artist_names",
    "album_name",
    "album_id",
    "release_date",
    "isrc",
    "main_genres",
    "duration",
    "duration_ms",
    "lyrics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate audio durations and lyric data against songs.csv."
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
        "--tolerance-seconds",
        type=float,
        default=DURATION_TOLERANCE_SECONDS,
        help="Maximum allowed duration difference (default: 10).",
    )
    parser.add_argument(
        "--write-tags",
        action="store_true",
        help="Write metadata tags into audio files. Without this flag, no audio files are changed.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise SystemExit(
                "Metadata CSV is missing required columns: " + ", ".join(missing)
            )
        return list(reader)


def parse_rank(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_duration_seconds(value: str) -> Optional[float]:
    """Parse duration values represented as seconds or HH:MM:SS/MM:SS."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        if not 2 <= len(parts) <= 3:
            return None
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            return None
        if len(numbers) == 2:
            minutes, seconds = numbers
            if minutes < 0 or not 0 <= seconds < 60:
                return None
            return minutes * 60 + seconds
        hours, minutes, seconds = numbers
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            return None
        return hours * 3600 + minutes * 60 + seconds

    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def parse_duration_ms(value: str) -> Optional[float]:
    if value is None or not str(value).strip():
        return None
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        return None
    return milliseconds if math.isfinite(milliseconds) and milliseconds >= 0 else None


def format_seconds(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}s"


def audio_path(audio_dir: Path, rank: int) -> Path:
    return audio_dir / f"{rank - 1:06d}_opus.webm"


def probe_duration_seconds(path: Path) -> float:
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
    duration = float(payload["format"]["duration"])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("ffprobe returned an invalid duration")
    return duration


def probe_container_format(path: Path) -> str:
    """Return the actual media container, independent of the filename suffix."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def output_muxer_for_container(container: str) -> Optional[str]:
    """Choose a muxer from the actual input container, not its .webm suffix."""
    names = {name.strip().lower() for name in container.split(",")}
    if "webm" in names or "matroska" in names:
        return "webm"
    if names.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
        return "mp4"
    return None


def normalize_artists(value: str) -> str:
    """Make the pipe-delimited dataset value readable in audio tags."""
    return " | ".join(part.strip() for part in str(value or "").split("|") if part.strip())


def metadata_tags(row: Dict[str, str]) -> Dict[str, str]:
    """Return standard tags plus useful identifiers from songs.csv."""
    tags = {
        "title": row.get("track_name", "").strip(),
        "artist": normalize_artists(row.get("artist_names", "")),
        "album": row.get("album_name", "").strip(),
        "album_artist": normalize_artists(row.get("artist_names", "")),
        "track": row.get("rank", "").strip(),
        "date": row.get("release_date", "").strip(),
        "genre": row.get("main_genres", "").strip(),
        "comment": (
            f"track_id={row.get('track_id', '').strip()}; "
            f"album_id={row.get('album_id', '').strip()}; "
            f"isrc={row.get('isrc', '').strip()}"
        ),
    }
    return {key: value for key, value in tags.items() if value}


def write_audio_tags(path: Path, tags: Dict[str, str]) -> None:
    """Remux a file with metadata while copying its existing audio stream."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --write-tags")

    container = probe_container_format(path)
    output_muxer = output_muxer_for_container(container)

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    command = [ffmpeg, "-y", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy"]
    for key, value in tags.items():
        command.extend(["-metadata", f"{key}={value}"])
    if output_muxer:
        command.extend(["-f", output_muxer])
    command.append(str(temporary_path))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary_path, path)
    except subprocess.CalledProcessError as exc:
        temporary_path.unlink(missing_ok=True)
        details = (exc.stderr or "").strip()
        raise RuntimeError(
            f"FFmpeg failed for {path.name} (container={container or 'unknown'}): "
            f"{details or f'exit status {exc.returncode}'}"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_ranks(rows: Sequence[Dict[str, str]]) -> None:
    ranks = [parse_rank(row.get("rank", "")) for row in rows]
    invalid = [value for value in ranks if value is None]
    valid = [value for value in ranks if value is not None]
    duplicates = sorted(rank for rank, count in Counter(valid).items() if count > 1)
    missing = sorted(set(range(RANK_MIN, RANK_MAX + 1)) - set(valid))
    outside = sorted(set(valid) - set(range(RANK_MIN, RANK_MAX + 1)))

    print("\nDATASET SHAPE")
    print(f"Metadata rows: {len(rows):,}")
    print(f"Expected rows: {RANK_MAX:,}")
    print(f"Invalid ranks: {len(invalid):,}")
    print(f"Duplicate ranks: {duplicates[:20] if duplicates else 'none'}")
    print(f"Missing ranks: {missing[:20] if missing else 'none'}")
    print(f"Out-of-range ranks: {outside[:20] if outside else 'none'}")


def validate_durations(
    rows: Sequence[Dict[str, str]],
    audio_dir: Path,
    tolerance_seconds: float,
) -> None:
    audio_vs_duration: List[Tuple[int, float, float, float]] = []
    audio_vs_duration_ms: List[Tuple[int, float, float, float]] = []
    metadata_duration_mismatch: List[Tuple[int, float, float, float]] = []
    invalid_metadata_duration: List[int] = []
    invalid_metadata_duration_ms: List[int] = []

    for row in rows:
        rank = parse_rank(row.get("rank", ""))
        if rank is None or not RANK_MIN <= rank <= RANK_MAX:
            continue

        duration_seconds = parse_duration_seconds(row.get("duration", ""))
        duration_ms = parse_duration_ms(row.get("duration_ms", ""))
        if duration_seconds is None:
            invalid_metadata_duration.append(rank)
        if duration_ms is None:
            invalid_metadata_duration_ms.append(rank)
        if duration_seconds is not None and duration_ms is not None:
            difference = abs(duration_seconds - duration_ms / 1000.0)
            if difference > tolerance_seconds:
                metadata_duration_mismatch.append(
                    (rank, duration_seconds, duration_ms / 1000.0, difference)
                )

        path = audio_path(audio_dir, rank)
        if not path.exists():
            continue

        audio_seconds = probe_duration_seconds(path)
        if duration_seconds is not None:
            difference = abs(audio_seconds - duration_seconds)
            if difference > tolerance_seconds:
                audio_vs_duration.append((rank, audio_seconds, duration_seconds, difference))
        if duration_ms is not None:
            metadata_seconds = duration_ms / 1000.0
            difference = abs(audio_seconds - metadata_seconds)
            if difference > tolerance_seconds:
                audio_vs_duration_ms.append(
                    (rank, audio_seconds, metadata_seconds, difference)
                )

    print("\nDURATION VALIDATION")
    print(f"Tolerance: {tolerance_seconds:.2f}s")
    print(f"Invalid duration values: {invalid_metadata_duration[:20] or 'none'}")
    print(f"Invalid duration_ms values: {invalid_metadata_duration_ms[:20] or 'none'}")
    print("\nRanks where audio differs from duration by more than the tolerance:")
    print([item[0] for item in audio_vs_duration] or "none")
    print("Ranks where audio differs from duration_ms by more than the tolerance:")
    print([item[0] for item in audio_vs_duration_ms] or "none")
    print("Ranks where duration differs from duration_ms by more than the tolerance:")
    print([item[0] for item in metadata_duration_mismatch] or "none")

    details = audio_vs_duration + audio_vs_duration_ms + metadata_duration_mismatch
    if details:
        print("\nMismatch details: rank | left | right | absolute difference")
        for rank, left, right, difference in details:
            print(f"{rank}: {format_seconds(left)} | {format_seconds(right)} | {difference:.2f}s")


def report_lyrics(rows: Sequence[Dict[str, str]]) -> None:
    missing = []
    punctuation_rankings = []

    for row in rows:
        rank = parse_rank(row.get("rank", ""))
        if rank is None:
            continue
        lyrics = row.get("lyrics", "") or ""
        if not lyrics.strip():
            missing.append(row)
        counts = Counter(character for character in lyrics if character in LYRIC_PUNCTUATION)
        punctuation_rankings.append((sum(counts.values()), rank, row, counts))

    print("\nFIRST 20 SONGS WITHOUT LYRICS")
    for row in missing[:20]:
        print(f"{row.get('rank', '')}: {row.get('track_name', '')}")
    if not missing:
        print("none")

    print("\nTOP 30 SONGS BY LYRIC PUNCTUATION FREQUENCY")
    print("Characters counted: '*', '–', '—', '?', '.'")
    print("rank | total | * | – | — | ? | . | track_name")
    punctuation_rankings.sort(key=lambda item: (-item[0], item[1]))
    for total, rank, row, counts in punctuation_rankings[:30]:
        breakdown = " | ".join(str(counts.get(character, 0)) for character in LYRIC_PUNCTUATION)
        print(f"{rank} | {total} | {breakdown} | {row.get('track_name', '')}")


def write_tags(rows: Sequence[Dict[str, str]], audio_dir: Path) -> None:
    print("\nWRITING AUDIO TAGS")
    print("This modifies audio files only; metadata CSV and download log are unchanged.")
    for row in rows:
        rank = parse_rank(row.get("rank", ""))
        if rank is None or not RANK_MIN <= rank <= RANK_MAX:
            continue
        path = audio_path(audio_dir, rank)
        if not path.exists():
            continue
        write_audio_tags(path, metadata_tags(row))
        print(f"Tagged rank {rank}: {path.name}")


def main() -> int:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    audio_dir = resolve_path(args.audio_dir)

    if args.tolerance_seconds < 0:
        raise SystemExit("--tolerance-seconds must be non-negative")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required for duration validation")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    rows = read_metadata(metadata_path)
    print("=" * 70)
    print("AUDIO / METADATA VALIDATION")
    print("=" * 70)
    print(f"Metadata: {metadata_path}")
    print(f"Audio directory: {audio_dir}")
    print(f"Mode: {'WRITE TAGS' if args.write_tags else 'READ ONLY'}")

    validate_ranks(rows)
    validate_durations(rows, audio_dir, args.tolerance_seconds)
    report_lyrics(rows)

    if args.write_tags:
        write_tags(rows, audio_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
