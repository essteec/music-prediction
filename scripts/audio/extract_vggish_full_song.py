"""
Full-Song VGGish (128-D) Audio Embedding Extraction.
Model: Google VGGish (TF Hub, 16kHz)

Feeds 100% of each audio file @ 16kHz into VGGish, which extracts 128-D embeddings
across all 0.96s frames, and averages over time to produce (128,) vector per song.

Output:
  data/embeddings/audio/vggish_embeddings_128d.npy (10000, 128) float32
"""

import os
import gc
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
OUTPUT_FILE = OUTPUT_DIR / "vggish_embeddings_128d.npy"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "vggish_full_checkpoint.npy"

SAMPLE_RATE = 16000

def decode_audio_full(path: Path) -> np.ndarray:
    """Decode full audio to 16kHz float32 mono via ffmpeg pipe."""
    try:
        cmd = [
            "ffmpeg", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return np.frombuffer(res.stdout, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    print(f"\n=======================================================")
    print(f"Full-Song VGGish (128-D) Extraction")
    print(f"Sampling Rate: {SAMPLE_RATE} Hz | Tracks: {n_songs}")
    print(f"Target: {OUTPUT_FILE}")
    print(f"=======================================================\n")

    if CHECKPOINT_FILE.exists():
        embeddings = np.load(CHECKPOINT_FILE)
        done = np.where(np.any(embeddings != 0, axis=1))[0]
        start_idx = int(done[-1]) + 1 if len(done) > 0 else 0
        print(f"Resuming from checkpoint at track {start_idx}/{n_songs}...")
    else:
        embeddings = np.zeros((n_songs, 128), dtype=np.float32)
        start_idx = 0

    print("Loading VGGish model from TF Hub...")
    model = hub.load("https://tfhub.dev/google/vggish/1")

    pbar = tqdm(total=n_songs, initial=start_idx, desc="Extracting VGGish")

    for i in range(start_idx, n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            pbar.update(1)
            continue

        try:
            audio = decode_audio_full(audio_path)
            if len(audio) >= SAMPLE_RATE:
                audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
                frame_embs = model(audio_tensor).numpy()
                embeddings[i] = frame_embs.mean(axis=0).astype(np.float32)
        except Exception as e:
            pass

        pbar.update(1)

        if (i + 1) % 500 == 0 or (i + 1) == n_songs:
            np.save(CHECKPOINT_FILE, embeddings)
            gc.collect()

    pbar.close()

    np.save(OUTPUT_FILE, embeddings)
    filled = int(np.any(embeddings != 0, axis=1).sum())
    print(f"\n=======================================================")
    print(f"VGGish Extraction Complete!")
    print(f"Saved: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Filled: {filled}/{n_songs} | NaNs: {np.isnan(embeddings).sum()}")
    print(f"=======================================================")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
