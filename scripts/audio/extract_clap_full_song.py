"""
Full-Song LAION-CLAP (512-D) Audio Embedding Extraction.
Model: laion_clap (HTSAT-base, 512-D, 48kHz)

Slices 100% of each song into 10s consecutive chunks (480,000 samples @ 48kHz),
computes 512-D embedding for each chunk, and mean-pools across the full track.

Output:
  data/embeddings/audio/clap_512d.npy (10000, 512) float32
"""

import os
import gc
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import laion_clap
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
OUTPUT_FILE = OUTPUT_DIR / "clap_512d.npy"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "clap_512d_checkpoint.npy"

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 480000  # 10s window
GPU_CHUNK_BATCH = 8     # 8 chunks per forward pass on GPU

def decode_audio_full(path: Path) -> np.ndarray:
    """Decode full audio to 48kHz float32 mono via ffmpeg pipe."""
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

def chunkify_10s(audio: np.ndarray) -> list:
    """Slice full audio into 10s chunks."""
    if len(audio) < SAMPLE_RATE * 2:
        return []
    chunks = []
    for s in range(0, len(audio), CHUNK_SAMPLES):
        c = audio[s : s + CHUNK_SAMPLES]
        if len(c) < SAMPLE_RATE * 2:
            break
        if len(c) < CHUNK_SAMPLES:
            c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c)
    return chunks

def embed_chunks(chunks: list, model, device: str) -> np.ndarray:
    """Embed 10s chunks through CLAP audio branch and mean-pool."""
    all_embs = []
    for b in range(0, len(chunks), GPU_CHUNK_BATCH):
        batch_c = chunks[b : b + GPU_CHUNK_BATCH]
        audio_inputs = []
        for c in batch_c:
            w = int16_to_float32(float32_to_int16(c))
            w = torch.from_numpy(w).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, w, CHUNK_SAMPLES,
                data_truncating='rand_trunc',
                data_filling='repeatpad',
                audio_cfg=model.model_cfg['audio_cfg'],
                require_grad=False
            )
            audio_inputs.append(temp_dict)
        
        with torch.no_grad():
            emb = model.model.get_audio_embedding(audio_inputs)
            all_embs.append(emb.cpu().numpy())
            
    return np.concatenate(all_embs, axis=0).mean(axis=0)  # (512,)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Full-Song LAION-CLAP (512-D) Extraction ({device})")
    print(f"Sampling Rate: {SAMPLE_RATE} Hz | Chunk: 10s | Tracks: {n_songs}")
    print(f"Target: {OUTPUT_FILE}")
    print(f"=======================================================\n")

    if CHECKPOINT_FILE.exists():
        embeddings = np.load(CHECKPOINT_FILE)
        done = np.where(np.any(embeddings != 0, axis=1))[0]
        start_idx = int(done[-1]) + 1 if len(done) > 0 else 0
        print(f"Resuming from checkpoint at track {start_idx}/{n_songs}...")
    else:
        embeddings = np.zeros((n_songs, 512), dtype=np.float32)
        start_idx = 0

    print("Loading LAION-CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    if device == "cuda":
        model.model.to(device)
    model.model.eval()

    pbar = tqdm(total=n_songs, initial=start_idx, desc="Extracting CLAP")

    for i in range(start_idx, n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            pbar.update(1)
            continue

        try:
            audio = decode_audio_full(audio_path)
            chunks = chunkify_10s(audio)
            if chunks:
                embeddings[i] = embed_chunks(chunks, model, device)
        except Exception as e:
            pass

        pbar.update(1)

        if (i + 1) % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 250 == 0 or (i + 1) == n_songs:
            np.save(CHECKPOINT_FILE, embeddings)
            gc.collect()

    pbar.close()

    np.save(OUTPUT_FILE, embeddings)
    filled = int(np.any(embeddings != 0, axis=1).sum())
    print(f"\n=======================================================")
    print(f"CLAP Extraction Complete!")
    print(f"Saved: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Shape: {embeddings.shape} | Filled: {filled}/{n_songs}")
    print(f"NaNs: {np.isnan(embeddings).sum()} | Infs: {np.isinf(embeddings).sum()}")
    print(f"=======================================================\n")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
