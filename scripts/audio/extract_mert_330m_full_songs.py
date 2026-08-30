"""
MERT-v1-330M (1024-D) Full-Song Audio Embedding Extraction.
Model: m-a-p/MERT-v1-330M (Apache-2.0)

Per song:
  - Decode full audio via ffmpeg pipe (fast, no audioread overhead)
  - Slice entire track into consecutive 30s chunks (100% coverage)
  - Run chunks through GPU in batches of 3
  - Mean-pool all chunk embeddings → 1 × 1024 vector per song

Expected speed: ~3.5s/song → ~10h for 10,000 songs on GTX 1660 Ti (6GB).
Output: data/embeddings/audio/mert_330m_embeddings_1024d.npy  (10000, 1024)
"""

import gc
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
SONGS_CSV     = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR     = DATA_DIR / "audio" / "pilot"
OUTPUT_FILE   = DATA_DIR / "embeddings" / "audio" / "mert_330m_embeddings_1024d.npy"
CKPT_FILE     = DATA_DIR / "embeddings" / "checkpoints" / "mert_330m_checkpoint.npy"

MODEL_ID      = "m-a-p/MERT-v1-330M"
SAMPLE_RATE   = 24000
CHUNK_SAMPLES = SAMPLE_RATE * 30   # 30 seconds
GPU_BATCH     = 3                  # chunks per forward pass — benchmarked safe on 6GB VRAM


def decode_audio(path: Path) -> np.ndarray:
    """Decode full audio file to float32 mono at SAMPLE_RATE using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.frombuffer(r.stdout, dtype=np.float32)


def chunkify(audio: np.ndarray) -> list:
    """Slice audio array into 30s chunks, keeping any tail ≥ 5s."""
    chunks = []
    for s in range(0, len(audio), CHUNK_SAMPLES):
        c = audio[s : s + CHUNK_SAMPLES]
        if len(c) < SAMPLE_RATE * 5:
            break
        if len(c) < CHUNK_SAMPLES:
            c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c.copy())
    return chunks


def embed_song(chunks: list, processor, model, device: str) -> np.ndarray:
    """Embed all 30s chunks of one song and return mean-pooled 1024-D vector."""
    pooled_list = []
    for b in range(0, len(chunks), GPU_BATCH):
        batch = chunks[b : b + GPU_BATCH]
        inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        iv = inputs["input_values"].to(device)
        if device == "cuda":
            iv = iv.half()
        with torch.no_grad():
            out = model(iv, output_hidden_states=True)
            p = torch.mean(out.last_hidden_state, dim=1).cpu().float().numpy()
        pooled_list.append(p)
    all_pooled = np.concatenate(pooled_list, axis=0)   # (N_chunks, 1024)
    return all_pooled.mean(axis=0)                      # (1024,)


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n  = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"MERT-v1-330M Full-Track Extraction  [{device} FP16]")
    print(f"Coverage: 100% of audio  |  GPU batch: {GPU_BATCH} chunks  |  {n} songs")
    print(f"Output → {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # Resume
    if CKPT_FILE.exists():
        emb = np.load(CKPT_FILE)
        done = np.where(np.any(emb != 0, axis=1))[0]
        start = int(done[-1]) + 1 if len(done) else 0
        print(f"Resuming from track {start}/{n}")
    else:
        emb   = np.zeros((n, 1024), dtype=np.float32)
        start = 0

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    for i in tqdm(range(start, n)):
        path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not path.exists():
            continue
        try:
            audio  = decode_audio(path)
            chunks = chunkify(audio)
            if not chunks:
                continue
            emb[i] = embed_song(chunks, processor, model, device)
        except Exception:
            pass

        if device == "cuda" and (i + 1) % 30 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 500 == 0 or (i + 1) == n:
            np.save(CKPT_FILE, emb)
            gc.collect()

    np.save(OUTPUT_FILE, emb)
    filled = int(np.any(emb != 0, axis=1).sum())
    print(f"\nDone. {OUTPUT_FILE}")
    print(f"Shape: {emb.shape} | Filled: {filled}/{n} | Size: {OUTPUT_FILE.stat().st_size/(1024**2):.1f} MB")
    CKPT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
