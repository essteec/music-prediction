"""
LAION-CLAP Audio Embedding Extraction Script.
Extracts 512-D cross-modal embeddings for 10,000 Opus audio files.
Model: LAION-CLAP HTSAT-tiny / HTSAT-base (Apache-2.0)
Outputs: data/embeddings/audio/clap_512d.npy
"""

import os
import gc
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import laion_clap

# Suppress audio loading warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
OUTPUT_FILE = OUTPUT_DIR / "clap_512d.npy"
CHECKPOINT_FILE = CHECKPOINT_DIR / "clap_512d_checkpoint.npy"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Extract LAION-CLAP embeddings")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for extraction")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if exists")
    args = parser.parse_args()

    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    print("Initializing LAION-CLAP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    if device == "cuda":
        model = model.to(device)
    model.eval()

    # Allocate or load embeddings array
    if args.resume and CHECKPOINT_FILE.exists():
        embeddings = np.load(CHECKPOINT_FILE)
        # Find first non-zero row
        nonzero_mask = ~np.all(embeddings == 0, axis=1)
        start_idx = np.where(~nonzero_mask)[0]
        start_idx = int(start_idx[0]) if len(start_idx) > 0 else n_songs
        print(f"Resuming from checkpoint at index {start_idx}/{n_songs}...")
    else:
        embeddings = np.zeros((n_songs, 512), dtype=np.float32)
        start_idx = 0

    if start_idx >= n_songs:
        print("All embeddings already extracted!")
        np.save(OUTPUT_FILE, embeddings)
        return

    batch_files = []
    batch_indices = []

    print(f"Extracting CLAP audio embeddings for {n_songs - start_idx} tracks (batch_size={args.batch_size})...")
    pbar = tqdm(total=n_songs, initial=start_idx)

    for i in range(start_idx, n_songs):
        audio_path = str(AUDIO_DIR / f"{i:06d}_opus.webm")
        if os.path.exists(audio_path):
            batch_files.append(audio_path)
            batch_indices.append(i)
        else:
            embeddings[i] = np.zeros(512, dtype=np.float32)
            pbar.update(1)

        if len(batch_files) >= args.batch_size or (i == n_songs - 1 and batch_files):
            try:
                with torch.no_grad():
                    embeds = model.get_audio_embedding_from_filelist(x=batch_files, use_tensor=False)
                    for idx, emb in zip(batch_indices, embeds):
                        embeddings[idx] = emb.astype(np.float32)
            except Exception as e:
                # Fallback: one by one
                for idx, fp in zip(batch_indices, batch_files):
                    try:
                        with torch.no_grad():
                            emb = model.get_audio_embedding_from_filelist(x=[fp], use_tensor=False)
                            embeddings[idx] = emb[0].astype(np.float32)
                    except Exception as e2:
                        embeddings[idx] = np.zeros(512, dtype=np.float32)

            pbar.update(len(batch_indices))
            batch_files = []
            batch_indices = []

            if device == "cuda" and (i % 100 == 0):
                torch.cuda.empty_cache()
                gc.collect()

            if i % 500 == 0 or i == n_songs - 1:
                np.save(CHECKPOINT_FILE, embeddings)

    pbar.close()

    np.save(OUTPUT_FILE, embeddings)
    print(f"\nSaved CLAP embeddings to: {OUTPUT_FILE}")
    print(f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
