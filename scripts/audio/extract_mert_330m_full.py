"""
Full 10,000 Song MERT-v1-330M (1024-D) Audio Embedding Extraction Script.
Model: m-a-p/MERT-v1-330M (Apache-2.0)
Outputs: data/embeddings/audio/mert_330m_embeddings_1024d.npy (10000, 1024)
"""

import os
import gc
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
OUTPUT_FILE = OUTPUT_DIR / "mert_330m_embeddings_1024d.npy"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "mert_330m_checkpoint.npy"

MODEL_ID = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
CHUNK_SEC = 30
MAX_CHUNKS = 4

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Extracting {MODEL_ID} (1024-D) on {device} (FP16)")
    print(f"Target: {OUTPUT_FILE}")
    print(f"=======================================================")

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    if CHECKPOINT_FILE.exists():
        embeddings = np.load(CHECKPOINT_FILE)
        nonzero_mask = np.any(embeddings != 0, axis=1)
        nonzero_idx = np.where(nonzero_mask)[0]
        start_idx = int(nonzero_idx[-1]) + 1 if len(nonzero_idx) > 0 else 0
        print(f"Resuming from checkpoint at track {start_idx}/{n_songs}...")
    else:
        embeddings = np.zeros((n_songs, 1024), dtype=np.float32)
        start_idx = 0

    pbar = tqdm(total=n_songs, initial=start_idx)

    for i in range(start_idx, n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            pbar.update(1)
            continue

        try:
            # Load up to 2 minutes of audio in 30s chunks
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=CHUNK_SEC * MAX_CHUNKS, mono=True)
            chunk_samples = SAMPLE_RATE * CHUNK_SEC
            n_chunks = max(1, min(len(audio) // chunk_samples, MAX_CHUNKS))
            
            chunk_vectors = []
            for c_i in range(n_chunks):
                chunk = audio[c_i * chunk_samples : (c_i + 1) * chunk_samples]
                if len(chunk) < SAMPLE_RATE * 5:  # Skip chunks shorter than 5s
                    continue
                inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_values = inputs["input_values"].to(device)
                if device == "cuda":
                    input_values = input_values.half()

                with torch.no_grad():
                    outputs = model(input_values, output_hidden_states=True)
                    last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, 1024)
                    chunk_pooled = torch.mean(last_hidden, dim=0).cpu().float().numpy()
                    chunk_vectors.append(chunk_pooled)

            if chunk_vectors:
                embeddings[i] = np.mean(chunk_vectors, axis=0)

        except Exception as e:
            pass

        pbar.update(1)

        if (i + 1) % 25 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 500 == 0 or (i + 1) == n_songs:
            np.save(CHECKPOINT_FILE, embeddings)
            gc.collect()

    pbar.close()

    np.save(OUTPUT_FILE, embeddings)
    print(f"\nSaved MERT-330M embeddings to: {OUTPUT_FILE}")
    print(f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}, Size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
