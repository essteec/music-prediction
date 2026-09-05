"""
Build Top-250 kNN Graph and 2D UMAP Projection for Unified Mood, Vibe & Context.

Weights:
- Genre Hybrid (50-D): 40%
- Spotify Audio (11-D): 30%
- Temporal & Collab (10-D): 15%
- Vocal & DSP Dynamics (12-D): 15%
Total: 83-D

Outputs:
- data/similarity/knn_mood_top250.parquet
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

def compute_top_k(tensor: torch.Tensor, k: int = 250, device: str = "cuda") -> tuple:
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

def save_knn_parquet(output_path: Path, spotify_ids: list, top_indices: np.ndarray, top_sims: np.ndarray, k: int = 250):
    n = len(spotify_ids)
    spotify_ids_arr = np.array(spotify_ids, dtype=object)
    
    neighbor_ids = [spotify_ids_arr[top_indices[i]].tolist() for i in range(n)]
    neighbor_indices = [top_indices[i].tolist() for i in range(n)]
    similarities = [top_sims[i].tolist() for i in range(n)]
    
    table_dict = {
        'row_idx': np.arange(n, dtype=np.int32),
        'track_id': spotify_ids,
        f'top{k}_neighbor_indices': neighbor_indices,
        f'top{k}_neighbor_track_ids': neighbor_ids,
        f'top{k}_similarities': similarities
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
    print(f"BUILDING TOP-250 UNIFIED MOOD & CONTEXT KNN MATRIX & 2D UMAP ({device})")
    print(f"Total Tracks: {n_songs:,}")
    print("="*75 + "\n")

    # Load component matrices
    print("Loading normalized metadata matrices...")
    t_genre    = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "genre_hybrid_50d.npy").astype(np.float32)))
    t_spotify  = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "spotify_audio_11d.npy").astype(np.float32)))
    t_temporal = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "temporal_collab_10d.npy").astype(np.float32)))
    t_vocal    = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "vocal_dsp_12d.npy").astype(np.float32)))

    # Fused Unified Mood, Vibe & Context Representation (83-D)
    # Weights: Genre 40%, Spotify 30%, Temporal 15%, Vocal 15%
    mood_blocks = [
        np.sqrt(0.40) * t_genre,
        np.sqrt(0.30) * t_spotify,
        np.sqrt(0.15) * t_temporal,
        np.sqrt(0.15) * t_vocal
    ]
    fused_mood = normalize_tensor(torch.cat(mood_blocks, dim=1))
    print(f"Unified Mood & Context representation shape: {fused_mood.shape}")
    assert fused_mood.shape[1] == 83, f"Expected 83-D, got {fused_mood.shape[1]}-D"

    # Compute Top-250 kNN
    print("\n[1/2] Computing Top-250 Unified Mood & Context Graph...")
    mood_top_idx, mood_top_sims = compute_top_k(fused_mood, k=250, device=device)
    save_knn_parquet(SIM_DIR / "knn_mood_top250.parquet", spotify_ids, mood_top_idx, mood_top_sims, k=250)

    # Compute 2D UMAP
    print("\n[2/2] Computing 2D Unified Mood UMAP Map...")
    compute_and_save_umap(SIM_DIR / "umap_2d_mood.parquet", spotify_ids, fused_mood)

    print("\n" + "="*75)
    print("ALL MOOD SIMILARITY & UMAP ARTIFACTS GENERATED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
