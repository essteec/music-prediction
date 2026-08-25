"""
Comprehensive Librosa DSP & Classical MIR Feature Extractor.
Extracts 88 interpretable acoustic, rhythmic, harmonic, timbral, dynamic, and stereo descriptors.
Outputs: data/features/audio/dsp_librosa.parquet
"""

import os
import gc
import json
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "features" / "audio"
CHECKPOINT_DIR = DATA_DIR / "features" / "checkpoints"
OUTPUT_FILE = OUTPUT_DIR / "dsp_librosa.parquet"
CHECKPOINT_FILE = CHECKPOINT_DIR / "dsp_librosa_checkpoint.parquet"

SAMPLE_RATE = 22050

def extract_track_dsp(row_idx: int) -> dict:
    audio_path = AUDIO_DIR / f"{row_idx:06d}_opus.webm"
    empty_feats = {'row_idx': row_idx}

    if not audio_path.exists():
        return empty_feats

    try:
        y_stereo, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=False)
        if y_stereo.ndim == 1:
            y_mono = y_stereo
            y_stereo = np.stack([y_mono, y_mono])
        else:
            y_mono = np.mean(y_stereo, axis=0)

        feats = {'row_idx': row_idx}

        # 1. Energy & Dynamics
        rms = librosa.feature.rms(y=y_mono)[0]
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_std'] = float(np.std(rms))
        feats['rms_max'] = float(np.max(rms))
        feats['rms_q10'] = float(np.percentile(rms, 10))
        feats['rms_q90'] = float(np.percentile(rms, 90))
        feats['crest_factor'] = float(np.max(np.abs(y_mono)) / (np.mean(rms) + 1e-8))

        # Integrated LUFS loudness
        meter = pyln.Meter(sr)
        try:
            feats['lufs_integrated'] = float(meter.integrated_loudness(y_stereo.T))
        except Exception:
            feats['lufs_integrated'] = -24.0

        # 2. Rhythm
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        feats['tempo_librosa'] = float(np.atleast_1d(tempo)[0])
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        dur_s = max(len(y_mono) / sr, 1.0)
        feats['onset_rate'] = float(len(onsets) / dur_s)
        feats['onset_strength_mean'] = float(np.mean(onset_env))
        feats['onset_strength_std'] = float(np.std(onset_env))
        feats['onset_strength_max'] = float(np.max(onset_env))

        # 3. Timbre (MFCCs & Spectral)
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=20)
        for k in range(20):
            feats[f'mfcc_{k+1}_mean'] = float(np.mean(mfcc[k]))
            feats[f'mfcc_{k+1}_std'] = float(np.std(mfcc[k]))

        spec_cent = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
        feats['spectral_centroid_mean'] = float(np.mean(spec_cent))
        feats['spectral_centroid_std'] = float(np.std(spec_cent))

        spec_bw = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0]
        feats['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        feats['spectral_bandwidth_std'] = float(np.std(spec_bw))

        spec_flat = librosa.feature.spectral_flatness(y=y_mono)[0]
        feats['spectral_flatness_mean'] = float(np.mean(spec_flat))

        spec_roll = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)[0]
        feats['spectral_rolloff_mean'] = float(np.mean(spec_roll))

        spec_contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
        feats['spectral_contrast_mean'] = float(np.mean(spec_contrast))
        feats['spectral_contrast_std'] = float(np.std(spec_contrast))

        zcr = librosa.feature.zero_crossing_rate(y=y_mono)[0]
        feats['zcr_mean'] = float(np.mean(zcr))
        feats['zcr_std'] = float(np.std(zcr))

        # 4. Harmony
        chroma = librosa.feature.chroma_cens(y=y_mono, sr=sr)
        for k in range(12):
            feats[f'chroma_{k}_mean'] = float(np.mean(chroma[k]))
        feats['chroma_entropy'] = float(np.mean(-np.sum(chroma * np.log(chroma + 1e-8), axis=0)))

        tonnetz = librosa.feature.tonnetz(y=y_mono, sr=sr)
        for k in range(6):
            feats[f'tonnetz_{k}_mean'] = float(np.mean(tonnetz[k]))
            feats[f'tonnetz_{k}_std'] = float(np.std(tonnetz[k]))

        # 5. Stereo
        if y_stereo.shape[0] >= 2:
            left, right = y_stereo[0], y_stereo[1]
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            feats['stereo_width'] = float(np.std(side) / (np.std(mid) + 1e-8))
            feats['lr_correlation'] = float(np.corrcoef(left, right)[0, 1]) if len(left) > 0 else 1.0
        else:
            feats['stereo_width'] = 0.0
            feats['lr_correlation'] = 1.0

        return feats
    except Exception as e:
        print(f"Error on track {row_idx}: {e}")
        return empty_feats

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Extract Librosa DSP Features")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    print(f"Extracting Librosa DSP features across {n_songs} songs with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(tqdm(executor.map(extract_track_dsp, range(n_songs), chunksize=25), total=n_songs))

    out_df = pd.DataFrame(records)
    out_df.insert(1, 'track_id', df['track_id'].values)

    # Fill any NaNs with column median
    for col in out_df.columns:
        if col not in ('row_idx', 'track_id') and out_df[col].dtype.kind in 'fc':
            out_df[col] = out_df[col].fillna(out_df[col].median()).astype(np.float32)

    out_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Librosa DSP features to: {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
