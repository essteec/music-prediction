"""
Audio Quality Control and Duration Verification Script.
Checks 10,000 downloaded Opus audio files against Spotify metadata duration.
Outputs: data/features/qc/audio_qc.parquet
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "features" / "qc"
OUTPUT_FILE = OUTPUT_DIR / "audio_qc.parquet"

def get_audio_duration_ffprobe(filepath: str) -> float:
    if not os.path.exists(filepath):
        return np.nan
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = res.stdout.strip()
        return float(out) if out else np.nan
    except Exception:
        return np.nan

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)
    print(f"Loaded {n_songs} songs.")

    file_paths = [str(AUDIO_DIR / f"{i:06d}_opus.webm") for i in range(n_songs)]
    file_exists = [os.path.exists(fp) for fp in file_paths]
    file_sizes = [os.path.getsize(fp) if exists else 0 for fp, exists in zip(file_paths, file_exists)]

    print("Probing audio durations with ffprobe in parallel (16 workers)...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        durations = list(tqdm(executor.map(get_audio_duration_ffprobe, file_paths), total=n_songs))

    spotify_duration_s = df['duration_ms'].fillna(0).values / 1000.0
    audio_duration_s = np.array(durations, dtype=np.float32)
    duration_delta_s = audio_duration_s - spotify_duration_s
    abs_delta_s = np.abs(duration_delta_s)

    qc_df = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'has_audio_file': file_exists,
        'file_size_bytes': file_sizes,
        'spotify_duration_s': spotify_duration_s.astype(np.float32),
        'audio_duration_s': audio_duration_s,
        'duration_delta_s': duration_delta_s.astype(np.float32),
        'abs_duration_delta_s': abs_delta_s.astype(np.float32),
        'duration_mismatch_10s': (abs_delta_s > 10.0) | np.isnan(audio_duration_s),
        'duration_mismatch_30s': (abs_delta_s > 30.0) | np.isnan(audio_duration_s),
    })

    qc_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Audio QC parquet to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # Summary Stats
    has_audio_count = sum(file_exists)
    valid_dur_count = (~np.isnan(audio_duration_s)).sum()
    mismatch_10s_count = qc_df['duration_mismatch_10s'].sum()
    mismatch_30s_count = qc_df['duration_mismatch_30s'].sum()

    print("\n--- Audio QC Summary ---")
    print(f"Total songs: {n_songs}")
    print(f"Audio files present: {has_audio_count} ({has_audio_count/n_songs*100:.1f}%)")
    print(f"Valid duration probed: {valid_dur_count} ({valid_dur_count/n_songs*100:.1f}%)")
    print(f"Mismatched (>10s delta): {mismatch_10s_count} ({mismatch_10s_count/n_songs*100:.1f}%)")
    print(f"Severe mismatch (>30s delta): {mismatch_30s_count} ({mismatch_30s_count/n_songs*100:.1f}%)")
    print(f"Median duration delta: {np.nanmedian(abs_delta_s):.2f}s")
    print(f"Mean duration delta: {np.nanmean(abs_delta_s):.2f}s")

if __name__ == "__main__":
    main()
