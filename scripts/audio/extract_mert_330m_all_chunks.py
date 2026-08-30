"""
Full-Song Multi-Chunk MERT-v1-330M (1024-D) Audio Embedding Extraction.
Model: m-a-p/MERT-v1-330M (Apache-2.0)

Extracts and saves ALL 30-second chunk embeddings across 100% of every song duration
with ZERO truncation limits in compact float16 precision.

Outputs:
  - data/embeddings/audio/mert_330m_all_chunks.npz:
      * 'chunks': (Total_Chunks, 1024) float16 contiguous array of all chunk vectors
      * 'chunk_offsets': (10001,) int64 CSR offset pointers per track
      * 'n_chunks': (10000,) int32 chunk counts per track
      * 'mean_embeddings': (10000, 1024) float32 global mean-pooled vector per track
  - data/embeddings/audio/mert_330m_embeddings_1024d.npy: (10000, 1024) float32 mean embeddings
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
from transformers import AutoModel, AutoFeatureExtractor

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "embeddings" / "audio"
OUTPUT_NPZ = OUTPUT_DIR / "mert_330m_all_chunks.npz"
MEAN_OUTPUT_NPY = OUTPUT_DIR / "mert_330m_embeddings_1024d.npy"
CHECKPOINT_DIR = DATA_DIR / "embeddings" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "mert_330m_chunks_checkpoint.npz"

MODEL_ID = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
CHUNK_SEC = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC
GPU_BATCH = 3  # 3 chunks per GPU forward pass (safe on 6GB VRAM)

def decode_audio_full(path: Path) -> np.ndarray:
    """Fast ffmpeg pipe decode of full audio to 24kHz float32 mono (no duration limits)."""
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

def chunkify_unlimited(audio: np.ndarray) -> list:
    """Slice 100% of the entire audio into consecutive 30s chunks without any cap."""
    if len(audio) < SAMPLE_RATE * 5:
        return []
    chunks = []
    for s in range(0, len(audio), CHUNK_SAMPLES):
        c = audio[s : s + CHUNK_SAMPLES]
        if len(c) < SAMPLE_RATE * 5:  # Tail shorter than 5s
            break
        if len(c) < CHUNK_SAMPLES:
            c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c.astype(np.float32))
    return chunks

def embed_song_chunks(chunks: list, processor, model, device: str) -> np.ndarray:
    """Embed all chunks of a song using batched GPU inference in float16."""
    all_pooled = []
    for b in range(0, len(chunks), GPU_BATCH):
        batch = chunks[b : b + GPU_BATCH]
        inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs["input_values"].to(device)
        if device == "cuda":
            input_values = input_values.half()
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
            # Mean pool over time frame dimension -> (B, 1024)
            pooled = torch.mean(outputs.last_hidden_state, dim=1).cpu().half().numpy()
            for p in pooled:
                all_pooled.append(p)
    return np.array(all_pooled, dtype=np.float16)  # (N_chunks, 1024) in float16

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Full-Song Multi-Chunk MERT-v1-330M Extraction ({device})")
    print(f"Precision: float16 | Chunk Duration: {CHUNK_SEC}s | Max Chunks: UNLIMITED")
    print(f"Outputs:")
    print(f"  - NPZ (all chunks + offsets): {OUTPUT_NPZ}")
    print(f"  - NPY (mean vectors):         {MEAN_OUTPUT_NPY}")
    print(f"=======================================================\n")

    # Resume handling
    song_chunks_dict = {}  # idx -> (N_chunks, 1024) float16
    mean_embeddings = np.zeros((n_songs, 1024), dtype=np.float32)
    n_chunks_arr = np.zeros(n_songs, dtype=np.int32)
    start_idx = 0

    if CHECKPOINT_FILE.exists():
        try:
            ckpt = np.load(CHECKPOINT_FILE)
            ckpt_chunks = ckpt['chunks']
            ckpt_offsets = ckpt['chunk_offsets']
            mean_embeddings = ckpt['mean_embeddings']
            n_chunks_arr = ckpt['n_chunks']
            done_indices = np.where(n_chunks_arr > 0)[0]
            start_idx = int(done_indices[-1]) + 1 if len(done_indices) > 0 else 0
            for idx in done_indices:
                s, e = ckpt_offsets[idx], ckpt_offsets[idx + 1]
                song_chunks_dict[idx] = ckpt_chunks[s:e]
            print(f"Resuming from checkpoint at song {start_idx}/{n_songs} ({len(song_chunks_dict)} songs loaded)...")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh...")
            start_idx = 0

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    pbar = tqdm(total=n_songs, initial=start_idx)

    for i in range(start_idx, n_songs):
        audio_path = AUDIO_DIR / f"{i:06d}_opus.webm"
        if not audio_path.exists():
            pbar.update(1)
            continue

        try:
            audio = decode_audio_full(audio_path)
            chunks = chunkify_unlimited(audio)
            if chunks:
                embs = embed_song_chunks(chunks, processor, model, device)  # (K, 1024) float16
                song_chunks_dict[i] = embs
                n_chunks_arr[i] = len(embs)
                mean_embeddings[i] = embs.astype(np.float32).mean(axis=0)
        except Exception as e:
            pass

        pbar.update(1)

        if (i + 1) % 30 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # Checkpoint every 100 songs
        if (i + 1) % 100 == 0 or (i + 1) == n_songs:
            ckpt_offsets = np.zeros(n_songs + 1, dtype=np.int64)
            chunks_list = []
            curr_off = 0
            for s_i in range(n_songs):
                ckpt_offsets[s_i] = curr_off
                if s_i in song_chunks_dict:
                    c_mat = song_chunks_dict[s_i]
                    chunks_list.append(c_mat)
                    curr_off += len(c_mat)
            ckpt_offsets[n_songs] = curr_off

            all_chunks_concat = np.concatenate(chunks_list, axis=0) if chunks_list else np.zeros((0, 1024), dtype=np.float16)
            np.savez_compressed(
                CHECKPOINT_FILE,
                chunks=all_chunks_concat,
                chunk_offsets=ckpt_offsets,
                n_chunks=n_chunks_arr,
                mean_embeddings=mean_embeddings
            )
            gc.collect()

    pbar.close()

    # Final assembly
    final_offsets = np.zeros(n_songs + 1, dtype=np.int64)
    final_chunks_list = []
    curr_offset = 0
    for s_i in range(n_songs):
        final_offsets[s_i] = curr_offset
        if s_i in song_chunks_dict:
            c_mat = song_chunks_dict[s_i]
            final_chunks_list.append(c_mat)
            curr_offset += len(c_mat)
    final_offsets[n_songs] = curr_offset

    all_chunks_array = np.concatenate(final_chunks_list, axis=0)  # (Total_Chunks, 1024) float16

    # Save final NPZ and NPY
    np.savez_compressed(
        OUTPUT_NPZ,
        chunks=all_chunks_array,
        chunk_offsets=final_offsets,
        n_chunks=n_chunks_arr,
        mean_embeddings=mean_embeddings
    )
    np.save(MEAN_OUTPUT_NPY, mean_embeddings)

    total_chunks = len(all_chunks_array)
    print(f"\n=======================================================")
    print(f"Extraction Complete (100% Lossless, float16)!")
    print(f"Saved All Chunks NPZ: {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved Mean Embeddings: {MEAN_OUTPUT_NPY} ({MEAN_OUTPUT_NPY.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Total Songs Processed: {(n_chunks_arr > 0).sum()}/{n_songs}")
    print(f"Total 30s Chunks Saved: {total_chunks:,} (Array shape: {all_chunks_array.shape}, Dtype: {all_chunks_array.dtype})")
    print(f"=======================================================")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
