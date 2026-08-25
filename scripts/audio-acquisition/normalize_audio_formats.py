#!/usr/bin/env python3
"""
Detect and normalize audio files whose content does not match the collection's
standard format.

Observed standard format in data/audio/pilot:
    Container:    WebM / Matroska
    Codec:        Opus
    Sample rate:  48,000 Hz
    Channels:     2, stereo

The script is report-only by default. Use --fix to re-encode only outliers.
Files keep their existing names. Repaired files are written to a temporary
file, verified, and atomically moved over the original only after successful
conversion.

The Opus encoder uses compression_level=10, the slowest/highest-compression
setting supported by libopus, with VBR enabled at 128 kbps.

Requirements:
    ffprobe and ffmpeg must be available on PATH.

Examples:
    python scripts/audio-acquisition/normalize_audio_formats.py
    python scripts/audio-acquisition/normalize_audio_formats.py --fix
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"

TARGET_CONTAINER = "webm"
TARGET_CODEC = "opus"
TARGET_SAMPLE_RATE = 48_000
TARGET_CHANNELS = 2
TARGET_LAYOUT = "stereo"
TARGET_BITRATE = "128k"
TARGET_COMPRESSION_LEVEL = "10"


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    container: str
    codec: str
    sample_rate: Optional[int]
    channels: Optional[int]
    channel_layout: str
    duration: Optional[float]
    error: Optional[str] = None

    @property
    def signature(self) -> tuple[str, str, Optional[int], Optional[int], str]:
        return (
            self.container,
            self.codec,
            self.sample_rate,
            self.channels,
            self.channel_layout,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and optionally normalize non-standard audio formats."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Audio directory (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Re-encode detected outliers to WebM/Opus/48 kHz/stereo.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def container_family(format_name: str) -> str:
    names = {part.strip().lower() for part in format_name.split(",")}
    if "webm" in names or "matroska" in names:
        return "webm"
    if names.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
        return "mp4"
    return format_name or "unknown"


def probe_audio(path: Path) -> AudioInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=format_name,duration:stream=codec_name,sample_rate,channels,channel_layout",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        format_data = payload.get("format", {})
        stream = (payload.get("streams") or [None])[0]
        if stream is None:
            raise ValueError("no audio stream found")

        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(format_data["duration"])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid duration")

        return AudioInfo(
            path=path,
            container=container_family(format_data.get("format_name", "")),
            codec=stream.get("codec_name", "unknown"),
            sample_rate=sample_rate,
            channels=channels,
            channel_layout=stream.get("channel_layout", "unknown"),
            duration=duration,
        )
    except Exception as exc:
        return AudioInfo(
            path=path,
            container="unknown",
            codec="unknown",
            sample_rate=None,
            channels=None,
            channel_layout="unknown",
            duration=None,
            error=str(exc),
        )


def target_signature() -> tuple[str, str, int, int, str]:
    return (
        TARGET_CONTAINER,
        TARGET_CODEC,
        TARGET_SAMPLE_RATE,
        TARGET_CHANNELS,
        TARGET_LAYOUT,
    )


def is_standard(info: AudioInfo) -> bool:
    return info.error is None and info.signature == target_signature()


def format_signature(info: AudioInfo) -> str:
    if info.error:
        return f"ERROR: {info.error}"
    return (
        f"container={info.container}, codec={info.codec}, "
        f"rate={info.sample_rate}, channels={info.channels}, "
        f"layout={info.channel_layout}"
    )


def reencode_to_target(info: AudioInfo) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --fix")

    with tempfile.NamedTemporaryFile(
        prefix=f".{info.path.stem}.",
        suffix=".webm",
        dir=info.path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(info.path),
        "-map",
        "0:a:0",
        "-map_metadata",
        "0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "libopus",
        "-application",
        "audio",
        "-compression_level",
        TARGET_COMPRESSION_LEVEL,
        "-b:a",
        TARGET_BITRATE,
        "-vbr",
        "on",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-f",
        "webm",
        str(temporary_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        converted = probe_audio(temporary_path)
        if not is_standard(converted):
            raise RuntimeError(
                f"converted file did not match target: {format_signature(converted)}"
            )
        os.replace(temporary_path, info.path)
    except subprocess.CalledProcessError as exc:
        temporary_path.unlink(missing_ok=True)
        details = (exc.stderr or "").strip()
        raise RuntimeError(
            f"FFmpeg failed for {info.path.name}: "
            f"{details or f'exit status {exc.returncode}'}"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    audio_dir = resolve_path(args.audio_dir)
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required but was not found on PATH.")
    if args.fix and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required with --fix but was not found on PATH.")
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    files = sorted(audio_dir.glob("*.webm"))
    if not files:
        raise SystemExit(f"No .webm files found in {audio_dir}")

    print("=" * 70)
    print("AUDIO FORMAT NORMALIZATION")
    print("=" * 70)
    print(f"Audio directory: {audio_dir}")
    print(f"Mode: {'FIX OUTLIERS' if args.fix else 'REPORT ONLY'}")
    print(
        "Target: "
        f"container={TARGET_CONTAINER}, codec={TARGET_CODEC}, "
        f"rate={TARGET_SAMPLE_RATE}, channels={TARGET_CHANNELS}, "
        f"layout={TARGET_LAYOUT}, bitrate={TARGET_BITRATE}, "
        f"compression_level={TARGET_COMPRESSION_LEVEL}"
    )

    infos = [probe_audio(path) for path in files]
    signatures = Counter(format_signature(info) for info in infos)
    outliers = [info for info in infos if not is_standard(info)]

    print(f"\nFiles inspected: {len(files):,}")
    print("Observed formats:")
    for signature, count in signatures.most_common():
        print(f"  {count:,}  {signature}")

    print(f"\nOutliers: {len(outliers):,}")
    for info in outliers:
        print(f"  {info.path.name}: {format_signature(info)}")

    if not args.fix or not outliers:
        return 0

    print("\nRe-encoding outliers...")
    repaired = 0
    failures = 0
    for index, info in enumerate(outliers, start=1):
        try:
            reencode_to_target(info)
            repaired += 1
            print(f"  Repaired {index:,}/{len(outliers):,}: {info.path.name}")
        except Exception as exc:
            failures += 1
            print(f"  FAILED {info.path.name}: {exc}")

    print(f"\nRepaired: {repaired:,}")
    print(f"Failed: {failures:,}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
