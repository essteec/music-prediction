"""
Full 10,000 Song Lyric Embedding Extraction Script (Robust VRAM & Checkpointing).
Extracts:
1. BAAI/bge-m3 -> embeddings/lyrics/bge_m3_1024d.npy (1024-D, 8192 context, MIT)
2. intfloat/multilingual-e5-large -> embeddings/lyrics/multilingual_e5_large_1024d.npy (1024-D, 512 context, MIT)
"""

import os
import gc
import re
import argparse
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Optimize CUDA allocator to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
OUTPUT_DIR = DATA_DIR / "embeddings" / "lyrics"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"

def clean_lyric(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[.*?\]', '', text)
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_embeddings_for_model(model_name: str, model_id: str, output_filename: str, max_seq_len: int = 4096, is_e5: bool = False, batch_size: int = 8):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / output_filename
    checkpoint_path = CHECKPOINT_DIR / f"{output_filename}.checkpoint.npy"

    if output_path.exists():
        arr = np.load(output_path)
        if np.count_nonzero(np.any(arr != 0, axis=1)) >= 9700:
            print(f"Skipping {model_name}: already fully extracted at {output_path}")
            return

    print(f"\n=======================================================")
    print(f"Extracting {model_name} ({model_id})")
    print(f"Output: {output_path}")
    print(f"=======================================================")

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on device: {device}...")
    model = SentenceTransformer(model_id, device=device)
    if hasattr(model, 'max_seq_length'):
        model.max_seq_length = max_seq_len

    dim = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {dim}")

    if checkpoint_path.exists():
        embeddings = np.load(checkpoint_path)
        nonzero_mask = np.any(embeddings != 0, axis=1)
        nonzero_indices = np.where(nonzero_mask)[0]
        # Find first gap or zero
        if len(nonzero_indices) > 0:
            start_idx = int(nonzero_indices[-1]) + 1
        else:
            start_idx = 0
        print(f"Resuming from checkpoint at index {start_idx}/{n_songs} (non-zero rows: {len(nonzero_indices)})...")
    else:
        embeddings = np.zeros((n_songs, dim), dtype=np.float32)
        start_idx = 0

    if start_idx >= n_songs:
        print("All embeddings already extracted!")
        np.save(output_path, embeddings)
        return

    # Pre-clean texts
    print(f"Cleaning lyric texts for remaining {n_songs - start_idx} tracks...")
    all_texts = []
    for i in range(start_idx, n_songs):
        raw = df.iloc[i]['lyrics']
        cleaned = clean_lyric(raw)
        if is_e5 and cleaned:
            cleaned = "passage: " + cleaned
        all_texts.append(cleaned)

    print(f"Encoding with batch_size={batch_size} and periodic GPU cache clearing...")
    pbar = tqdm(total=n_songs, initial=start_idx)
    step_cnt = 0

    for b_start in range(0, len(all_texts), batch_size):
        b_end = min(b_start + batch_size, len(all_texts))
        batch_slice = all_texts[b_start:b_end]
        global_indices = list(range(start_idx + b_start, start_idx + b_end))

        non_empty_texts = []
        non_empty_local_idx = []
        for local_i, t in enumerate(batch_slice):
            if t.strip():
                non_empty_texts.append(t)
                non_empty_local_idx.append(local_i)

        if non_empty_texts:
            try:
                with torch.no_grad():
                    batch_embeds = model.encode(non_empty_texts, batch_size=len(non_empty_texts), show_progress_bar=False, normalize_embeddings=True)
                    for local_i, emb in zip(non_empty_local_idx, batch_embeds):
                        g_idx = global_indices[local_i]
                        embeddings[g_idx] = emb.astype(np.float32)
            except Exception as e:
                # Fallback one-by-one with CPU fallback if needed
                for local_i, t in zip(non_empty_local_idx, non_empty_texts):
                    g_idx = global_indices[local_i]
                    try:
                        with torch.no_grad():
                            emb = model.encode([t], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
                            embeddings[g_idx] = emb[0].astype(np.float32)
                    except Exception:
                        embeddings[g_idx] = np.zeros(dim, dtype=np.float32)

        pbar.update(len(batch_slice))
        step_cnt += 1

        if step_cnt % 25 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # Checkpoint every 500 songs
        if (start_idx + b_end) % 500 == 0 or (start_idx + b_end) == n_songs:
            np.save(checkpoint_path, embeddings)
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    pbar.close()

    np.save(output_path, embeddings)
    print(f"Saved {model_name} embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Extract Full Lyric Embeddings")
    parser.add_argument("--model", type=str, choices=['bge-m3', 'e5-large', 'both'], default='both', help="Model to extract")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    if args.model in ('e5-large', 'both'):
        extract_embeddings_for_model(
            model_name="multilingual-E5-large",
            model_id="intfloat/multilingual-e5-large",
            output_filename="multilingual_e5_large_1024d.npy",
            max_seq_len=512,
            is_e5=True,
            batch_size=args.batch_size
        )

    if args.model in ('bge-m3', 'both'):
        extract_embeddings_for_model(
            model_name="BGE-M3",
            model_id="BAAI/bge-m3",
            output_filename="bge_m3_1024d.npy",
            max_seq_len=4096,
            is_e5=False,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
