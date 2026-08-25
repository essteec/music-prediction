"""
PANNs 527-Class AudioSet Tag Probabilities Extractor.
Extracts interpretable audio tag probabilities (527-D) across all 10,000 tracks.
Outputs:
- data/embeddings/audio/panns_tags_527d.npy (shape: 10000, 527, float32)
- data/embeddings/audio/panns_tags_labels.json (AudioSet class label names)
"""

import os
import gc
import json
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from panns_inference import AudioTagging, labels

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
OUTPUT_FILE = OUTPUT_DIR / "panns_tags_527d.npy"
LABELS_FILE = OUTPUT_DIR / "panns_tags_labels.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "panns_tags_checkpoint.npy"

SAMPLE_RATE = 32000

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Extract PANNs 527 tag probabilities")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if exists")
    args = parser.parse_args()

    # Save labels JSON
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(labels)} AudioSet class labels to {LABELS_FILE}")

    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PANNs model on device: {device}...")
    model = AudioTagging(checkpoint_path=None, device=device)

    # Allocate or load embeddings array
    if args.resume and CHECKPOINT_FILE.exists():
        tag_probs = np.load(CHECKPOINT_FILE)
        nonzero_mask = ~np.all(tag_probs == 0, axis=1)
        start_idx = np.where(~nonzero_mask)[0]
        start_idx = int(start_idx[0]) if len(start_idx) > 0 else n_songs
        print(f"Resuming from checkpoint at index {start_idx}/{n_songs}...")
    else:
        tag_probs = np.zeros((n_songs, 527), dtype=np.float32)
        start_idx = 0

    if start_idx >= n_songs:
        print("All PANNs tag probabilities already extracted!")
        np.save(OUTPUT_FILE, tag_probs)
        return

    print(f"Extracting PANNs tag probabilities for {n_songs - start_idx} tracks...")
    for i in tqdm(range(start_idx, n_songs), initial=start_idx, total=n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            tag_probs[i] = np.zeros(527, dtype=np.float32)
            continue

        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            audio_batch = audio[np.newaxis, :]
            clipwise_output, _ = model.inference(audio_batch)
            tag_probs[i] = clipwise_output[0].astype(np.float32)
        except Exception as e:
            print(f"\nError processing track index {i}: {e}")
            tag_probs[i] = np.zeros(527, dtype=np.float32)

        if i % 100 == 0:
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if i % 500 == 0 or i == n_songs - 1:
            np.save(CHECKPOINT_FILE, tag_probs)

    np.save(OUTPUT_FILE, tag_probs)
    print(f"\nSaved PANNs tag probabilities to: {OUTPUT_FILE}")
    print(f"Shape: {tag_probs.shape}, Dtype: {tag_probs.dtype}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
