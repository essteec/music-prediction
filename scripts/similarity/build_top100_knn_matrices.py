"""
Build Top-100 kNN Graph Matrices & 2D UMAP Projections.

Optimal Formulations:
  1. Audio Graph:   CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D) -> 1664-D
  2. Lyric Graph:   Harrier-0.6B (1024-D) + Multilingual E5-Large (1024-D) -> 2048-D
  3. Combined Graph: 0.5 * Audio (1664-D) + 0.5 * Lyric (2048-D) -> 3712-D

Outputs:
  - data/similarity/knn_audio_top100.parquet
  - data/similarity/knn_lyric_top100.parquet
  - data/similarity/knn_combined_top100.parquet
  - data/similarity/umap_2d_audio.parquet
  - data/similarity/umap_2d_lyric.parquet
  - data/similarity/umap_2d_combined.parquet
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_EMB_DIR = DATA_DIR / "embeddings" / "audio"
LYRIC_EMB_DIR = DATA_DIR / "embeddings" / "lyric"
SIM_DIR = DATA_DIR / "similarity"

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(t, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return t / norms

def compute_top_k(tensor: torch.Tensor, k: int = 100, device: str = "cuda") -> tuple:
    """Compute Top-K cosine similarity indices and scores on GPU."""
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
    """Save Top-100 kNN graph as a high-performance Parquet table."""
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
    """Compute 2D UMAP projections and save to Parquet."""
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
    print(f"BUILDING TOP-100 KNN MATRICES & UMAP 2D PROJECTIONS ({device})")
    print(f"Total Tracks: {n_songs:,}")
    print("="*75 + "\n")
    
    # 1. Optimal Audio Representation
    print("[1/3] Processing Audio (CLAP 512 + MERT-330M 1024 + VGGish 128 -> 1664-D)...")
    clap = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "clap_512d.npy").astype(np.float32)))
    mert = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "mert_330m_embeddings_1024d.npy").astype(np.float32)))
    vgg  = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "vggish_embeddings_128d.npy").astype(np.float32)))
    
    audio_fused = normalize_tensor(torch.cat([clap, mert, vgg], dim=1))
    audio_top100_idx, audio_top100_sims = compute_top_k(audio_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_audio_top100.parquet", spotify_ids, audio_top100_idx, audio_top100_sims)
    
    # 2. Optimal Lyric Representation
    print("\n[2/3] Processing Lyric (Harrier 1024 + E5-Large 1024 -> 2048-D)...")
    harrier = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "harrier_embeddings_1024d.npy").astype(np.float32)))
    e5      = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "multilingual_e5_large_1024d.npy").astype(np.float32)))
    
    has_lyrics = (torch.norm(harrier, p=2, dim=1, keepdim=True) > 1e-6).float()
    lyric_fused = normalize_tensor(torch.cat([harrier, e5], dim=1)) * has_lyrics
    lyric_top100_idx, lyric_top100_sims = compute_top_k(lyric_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_lyric_top100.parquet", spotify_ids, lyric_top100_idx, lyric_top100_sims)
    
    # 3. Combined Multimodal Representation
    print("\n[3/3] Processing Combined Multimodal (0.5 Audio + 0.5 Lyric -> 3712-D)...")
    audio_weighted = audio_fused * np.sqrt(0.5)
    lyric_weighted = lyric_fused * np.sqrt(0.5)
    combined_fused = normalize_tensor(torch.cat([audio_weighted, lyric_weighted], dim=1))
    combined_top100_idx, combined_top100_sims = compute_top_k(combined_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_combined_top100.parquet", spotify_ids, combined_top100_idx, combined_top100_sims)
    
    # 4. Clean up legacy top50 files
    print("\n" + "="*75)
    print("CLEANING UP LEGACY TOP-50 FILES")
    print("="*75)
    for legacy in ["knn_audio_top50.parquet", "knn_lyric_top50.parquet"]:
        p = SIM_DIR / legacy
        if p.exists():
            p.unlink()
            print(f"  Removed legacy: {p.name}")
            
    # 5. UMAP 2D Projections
    print("\n" + "="*75)
    print("COMPUTING 2D UMAP VISUALIZATION PROJECTIONS")
    print("="*75)
    compute_and_save_umap(SIM_DIR / "umap_2d_audio.parquet", spotify_ids, audio_fused)
    compute_and_save_umap(SIM_DIR / "umap_2d_lyric.parquet", spotify_ids, lyric_fused)
    compute_and_save_umap(SIM_DIR / "umap_2d_combined.parquet", spotify_ids, combined_fused)
    
    print("\n" + "="*75)
    print("ALL SIMILARITY & PROJECTION ARTIFACTS GENERATED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
