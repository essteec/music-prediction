"""
Lyric Embedding Extraction using microsoft/harrier-oss-v1-0.6b (MIT License).

Model specs:
  - Architecture: Decoder-only, 0.6B params
  - Output: 1024-D, L2-normalised (cosine-ready)
  - Pooling: Last-token
  - Max tokens: 32,768
  - sentence-transformers compatible

Encoding strategy (per model card):
  - Lyrics are corpus/document side -> encode plain, NO instruction prefix.
  - Instructions are only for query-side text; not applicable here.

Benchmark results on GTX 1660 Ti (6GB VRAM, FP16):
  batch=8  -> peak 1,267 MB | 0.86 songs/s
  batch=16 -> peak 1,386 MB | 1.12 songs/s  (optimal)
  batch=32 -> peak 1,622 MB | 1.14 songs/s

Output:
  data/embeddings/lyric/harrier_embeddings_1024d.npy  (10000, 1024) float32
  - Zero-vector rows for the 203 tracks with missing lyrics.
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SONGS_CSV    = DATA_DIR / "processed" / "songs.csv"
OUTPUT_DIR   = DATA_DIR / "embeddings" / "lyric"
OUTPUT_FILE  = OUTPUT_DIR / "harrier_embeddings_1024d.npy"
CKPT_FILE    = DATA_DIR / "embeddings" / "checkpoints" / "harrier_checkpoint.npy"

MODEL_ID   = "microsoft/harrier-oss-v1-0.6b"
BATCH_SIZE = 16   # Optimal: 1,386 MB peak VRAM, 1.12 songs/s on GTX 1660 Ti FP16
DIM        = 1024


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n  = len(df)
    lyrics = df["lyrics"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Harrier-OSS-v1-0.6B Lyric Embedding Extraction  [{device} FP16]")
    print(f"Tracks: {n} | Batch size: {BATCH_SIZE} | Output dim: {DIM}")
    print(f"Output -> {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # Resume from checkpoint
    if CKPT_FILE.exists():
        emb = np.load(CKPT_FILE)
        done = np.where(np.any(emb != 0, axis=1))[0]
        start = int(done[-1]) + 1 if len(done) else 0
        print(f"Resuming from track {start}/{n}")
    else:
        emb   = np.zeros((n, DIM), dtype=np.float32)
        start = 0

    model = SentenceTransformer(MODEL_ID, device=device)
    model.half()   # FP16 inference

    # Identify which indices have valid lyrics from start onward
    valid_indices = [
        i for i in range(start, n)
        if isinstance(lyrics[i], str) and lyrics[i].strip()
    ]
    print(f"Valid lyric tracks from index {start}: {len(valid_indices)}")

    for batch_start in tqdm(range(0, len(valid_indices), BATCH_SIZE),
                            desc="Encoding lyrics"):
        batch_indices = valid_indices[batch_start : batch_start + BATCH_SIZE]
        batch_lyrics  = [lyrics[i] for i in batch_indices]

        try:
            # No instruction prefix -- lyrics are corpus side (per model card)
            batch_emb = model.encode(
                batch_lyrics,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,   # L2 norm (cosine-ready)
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

            for local_j, global_i in enumerate(batch_indices):
                emb[global_i] = batch_emb[local_j]

        except Exception as e:
            print(f"  [WARN] batch starting at idx {batch_indices[0]} failed: {e}")

        # Checkpoint every ~500 songs
        last_idx = batch_indices[-1]
        if (last_idx + 1) % 500 < BATCH_SIZE or (last_idx + 1) == n:
            np.save(CKPT_FILE, emb)
            gc.collect()

    np.save(OUTPUT_FILE, emb)
    filled = int(np.any(emb != 0, axis=1).sum())
    print(f"\nDone.")
    print(f"Shape: {emb.shape} | Filled: {filled}/{n} | Size: {OUTPUT_FILE.stat().st_size/(1024**2):.1f} MB")
    print(f"NaNs: {np.isnan(emb).sum()} | Infs: {np.isinf(emb).sum()}")

    CKPT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
