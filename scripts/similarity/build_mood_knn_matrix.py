"""
Build Top-100 kNN Graph and 2D UMAP Projection for Mood, Emotion, and Acoustic Vibe.

Input Sources:
- data/embeddings/metadata/spotify_audio_11d.npy
- data/embeddings/metadata/emotion_sentiment_36d.npy
- data/embeddings/metadata/vocal_dsp_12d.npy

Outputs:
- data/similarity/knn_mood_top100.parquet
- data/similarity/umap_2d_mood.parquet
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
META_EMB_DIR = DATA_DIR / "embeddings" / "metadata"
SIM_DIR = DATA_DIR / "similarity"

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(t, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return t / norms

def compute_top_k(tensor: torch.Tensor, k: int = 100, device: str = "cuda") -> tuple:
    t_cuda = tensor.to(device)
    n = t_cuda.shape[0]
    top_indices = np.zeros((n, k), dtype=np.int32)
    top_sims = np.zeros((n, k), dtype=np.float32)
    
    block_size = 2000
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sim_block = torch.matmul(t_cuda[start:end], t_cuda.T)
        for i in range(start, end):
            sim_block[i - start, i] = -1e9  # Exclude self
        vals, idxs = torch.topk(sim_block, k=k, dim=1, largest=True)
        top_indices[start:end] = idxs.cpu().numpy().astype(np.int32)
        top_sims[start:end] = vals.cpu().numpy().astype(np.float32)
        
    return top_indices, top_sims

def save_knn_parquet(output_path: Path, spotify_ids: list, top_indices: np.ndarray, top_sims: np.ndarray):
    n = len(spotify_ids)
    spotify_ids_arr = np.array(spotify_ids, dtype=object)
    
    neighbor_ids = [spotify_ids_arr[top_indices[i]].tolist() for i in range(n)]
    neighbor_indices = [top_indices[i].tolist() for i in range(n)]
    similarities = [top_sims[i].tolist() for i in range(n)]
    
    table_dict = {
        'row_idx': np.arange(n, dtype=np.int32),
        'track_id': spotify_ids,
        'top100_neighbor_indices': neighbor_indices,
        'top100_neighbor_track_ids': neighbor_ids,
        'top100_similarities': similarities
    }
    
    df_out = pd.DataFrame(table_dict)
    df_out.to_parquet(output_path, index=False, engine='pyarrow')
    sz_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {output_path.name} ({sz_mb:.2f} MB, shape: {df_out.shape})")

def compute_and_save_umap(output_path: Path, spotify_ids: list, tensor: torch.Tensor, random_state: int = 42):
    import umap
    print(f"  Computing 2D UMAP projection for {output_path.name}...")
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=random_state, n_neighbors=30, min_dist=0.1)
    proj = reducer.fit_transform(tensor.numpy()).astype(np.float32)
    
    df_umap = pd.DataFrame({
        'row_idx': np.arange(len(spotify_ids), dtype=np.int32),
        'track_id': spotify_ids,
        'proj_x': proj[:, 0],
        'proj_y': proj[:, 1]
    })
    df_umap.to_parquet(output_path, index=False, engine='pyarrow')
    sz_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {output_path.name} ({sz_mb:.2f} MB in {time.time()-t0:.1f}s)")

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    df_songs = pd.read_csv(SONGS_CSV)
    spotify_ids = df_songs['track_id'].tolist()
    n_songs = len(spotify_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*75)
    print(f"BUILDING TOP-100 MOOD KNN MATRIX & 2D UMAP ({device})")
    print(f"Total Tracks: {n_songs:,}")
    print("="*75 + "\n")

    # Load component matrices
    print("Loading normalized metadata matrices...")
    t_spotify = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "spotify_audio_11d.npy").astype(np.float32)))
    t_emotion = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "emotion_sentiment_36d.npy").astype(np.float32)))
    t_vocal   = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "vocal_dsp_12d.npy").astype(np.float32)))

    # Fused Mood & Vibe Representation (59-D)
    fused_mood = normalize_tensor(torch.cat([t_spotify, t_emotion, t_vocal], dim=1))
    print(f"Fused Mood & Vibe representation shape: {fused_mood.shape}")

    # Compute Top-100 kNN
    print("\n[1/2] Computing Top-100 Mood & Vibe Graph...")
    mood_top100_idx, mood_top100_sims = compute_top_k(fused_mood, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_mood_top100.parquet", spotify_ids, mood_top100_idx, mood_top100_sims)

    # Compute 2D UMAP
    print("\n[2/2] Computing 2D Mood UMAP Map...")
    compute_and_save_umap(SIM_DIR / "umap_2d_mood.parquet", spotify_ids, fused_mood)

    print("\n" + "="*75)
    print("ALL MOOD SIMILARITY & UMAP ARTIFACTS GENERATED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
