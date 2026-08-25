"""
Silero VAD (Vocal Activity Detection) Feature Extractor.
Extracts:
- vocal_duration_s: total seconds of speech/singing detected
- vocal_ratio: proportion of track containing vocals
- silent_ratio: proportion of silence
Outputs: data/features/audio/vad.parquet
"""

import os
import gc
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "features" / "audio"
OUTPUT_FILE = OUTPUT_DIR / "vad.parquet"

SAMPLE_RATE = 16000

def extract_track_vad(row_idx: int) -> dict:
    audio_path = AUDIO_DIR / f"{row_idx:06d}_opus.webm"
    empty_res = {
        'row_idx': row_idx,
        'vocal_duration_s': 0.0,
        'vocal_ratio': 0.0,
        'has_vocals': False
    }

    if not audio_path.exists():
        return empty_res

    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True, onnx=False)
        (get_speech_timestamps, _, _, _, _) = utils

        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        total_dur = len(audio) / SAMPLE_RATE
        if total_dur <= 0:
            return empty_res

        wav_tensor = torch.from_numpy(audio)
        speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=SAMPLE_RATE, threshold=0.45)
        speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        vocal_dur = speech_samples / SAMPLE_RATE
        vocal_ratio = vocal_dur / max(total_dur, 1.0)

        return {
            'row_idx': row_idx,
            'vocal_duration_s': round(float(vocal_dur), 2),
            'vocal_ratio': round(float(vocal_ratio), 4),
            'has_vocals': vocal_ratio > 0.05
        }
    except Exception:
        return empty_res

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Extract Silero VAD")
    parser.add_argument("--workers", type=int, default=4, help="Worker count")
    args = parser.parse_args()

    print(f"Loading songs from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    print(f"Extracting Silero VAD features across {n_songs} songs with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(tqdm(executor.map(extract_track_vad, range(n_songs), chunksize=50), total=n_songs))

    out_df = pd.DataFrame(records)
    out_df.insert(1, 'track_id', df['track_id'].values)

    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Silero VAD features to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"Tracks with vocals (>5% vocal ratio): {out_df['has_vocals'].sum()} / {n_songs}")

if __name__ == "__main__":
    main()
