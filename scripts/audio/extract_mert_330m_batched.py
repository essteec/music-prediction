"""
Optimized Batched 10,000 Song MERT-v1-330M (1024-D) Audio Embedding Extraction Script.
Model: m-a-p/MERT-v1-330M (Apache-2.0)
Extracts 30s standardized representations with batch_size=3 in FP16 on GPU.
Outputs: data/embeddings/audio/mert_330m_embeddings_1024d.npy (10000, 1024)
"""

import os
import gc
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

warnings.filterwarnings('ignore')

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
BATCH_SIZE = 3

def load_audio_30s(audio_path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=CHUNK_SEC, mono=True)
        # Pad to exactly 30s if shorter
        target_len = SAMPLE_RATE * CHUNK_SEC
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len].astype(np.float32)
    except Exception:
        return np.zeros(SAMPLE_RATE * CHUNK_SEC, dtype=np.float32)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Batched Extraction: {MODEL_ID} (1024-D) on {device} (FP16)")
    print(f"Batch Size: {BATCH_SIZE} | Duration: {CHUNK_SEC}s | Target: {OUTPUT_FILE}")
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

    for b_start in range(start_idx, n_songs, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, n_songs)
        batch_indices = list(range(b_start, b_end))

        # Preload batch audio
        batch_audios = []
        for idx in batch_indices:
            a_path = AUDIO_DIR / f"{idx:06d}_opus.webm"
            batch_audios.append(load_audio_30s(a_path))

        # Filter out completely silent / missing tracks
        valid_local = [i for i, a in enumerate(batch_audios) if np.any(a != 0)]
        if valid_local:
            try:
                valid_audios = [batch_audios[i] for i in valid_local]
                inputs = processor(valid_audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_values = inputs["input_values"].to(device)
                if device == "cuda":
                    input_values = input_values.half()

                with torch.no_grad():
                    outputs = model(input_values, output_hidden_states=True)
                    last_hidden = outputs.last_hidden_state  # (B, T, 1024)
                    pooled = torch.mean(last_hidden, dim=1).cpu().float().numpy()

                for local_i, pool_vec in zip(valid_local, pooled):
                    global_i = batch_indices[local_i]
                    embeddings[global_i] = pool_vec

            except Exception as e:
                # Fallback item by item
                for local_i in valid_local:
                    global_i = batch_indices[local_i]
                    try:
                        inp = processor([batch_audios[local_i]], sampling_rate=SAMPLE_RATE, return_tensors="pt")
                        inp_val = inp["input_values"].to(device)
                        if device == "cuda":
                            inp_val = inp_val.half()
                        with torch.no_grad():
                            out = model(inp_val, output_hidden_states=True)
                            embeddings[global_i] = torch.mean(out.last_hidden_state, dim=1).squeeze(0).cpu().float().numpy()
                    except Exception:
                        pass

        pbar.update(len(batch_indices))

        if (b_end % 30 == 0) and device == "cuda":
            torch.cuda.empty_cache()

        if (b_end % 500 == 0) or (b_end == n_songs):
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
