"""
Build Complete Suite of Top-250 kNN Graph Matrices.

Suite of Similarity Graphs:
1. knn_audio_top250.parquet:    CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D) -> 1664-D
2. knn_lyric_top250.parquet:    Harrier-0.6B (1024-D) + Multilingual E5-Large (1024-D) -> 2048-D
3. knn_mood_top250.parquet:     Unified Mood & Context: Genre (40%) + Spotify (30%) + Temporal (15%) + Vocal (15%) -> 83-D
4. knn_combined_top250.parquet: Audio (38%) + Lyric (35%) + Genre (11%) + Spotify (8%) + Temporal (4%) + Vocal (4%) -> 3795-D
"""

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
META_EMB_DIR = DATA_DIR / "embeddings" / "metadata"
SIM_DIR = DATA_DIR / "similarity"

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    nonzero = (torch.norm(t, p=2, dim=1, keepdim=True) > 1e-6).float()
    norms = torch.norm(t, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return (t / norms) * nonzero

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

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    df_songs = pd.read_csv(SONGS_CSV)
    spotify_ids = df_songs['track_id'].tolist()
    n_songs = len(spotify_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*85)
    print(f"BUILDING OFFICIAL 4-FACET TOP-250 KNN MATRICES ({device})")
    print(f"Total Tracks: {n_songs:,}")
    print("="*85 + "\n")

    # 1. Neural Audio Representation (1664-D)
    print("[1/4] Processing Acoustic Similarity (CLAP + MERT + VGGish -> 1664-D)...")
    clap = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "clap_512d.npy").astype(np.float32)))
    mert = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "mert_330m_embeddings_1024d.npy").astype(np.float32)))
    vgg  = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "vggish_embeddings_128d.npy").astype(np.float32)))
    audio_fused = normalize_tensor(torch.cat([clap, mert, vgg], dim=1))
    audio_top_idx, audio_top_sims = compute_top_k(audio_fused, k=250, device=device)
    save_knn_parquet(SIM_DIR / "knn_audio_top250.parquet", spotify_ids, audio_top_idx, audio_top_sims, k=250)

    # 2. Neural Lyric Representation (2048-D)
    print("\n[2/4] Processing Lyrical Storytelling (Harrier-0.6B + E5-Large -> 2048-D)...")
    harrier = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "harrier_embeddings_1024d.npy").astype(np.float32)))
    e5      = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "multilingual_e5_large_1024d.npy").astype(np.float32)))
    has_lyrics = (torch.norm(harrier, p=2, dim=1, keepdim=True) > 1e-6).float()
    lyric_fused = normalize_tensor(torch.cat([harrier, e5], dim=1)) * has_lyrics
    lyric_top_idx, lyric_top_sims = compute_top_k(lyric_fused, k=250, device=device)
    save_knn_parquet(SIM_DIR / "knn_lyric_top250.parquet", spotify_ids, lyric_top_idx, lyric_top_sims, k=250)

    # 3. Unified Mood, Vibe & Context Representation (83-D)
    # Weights: Genre 40%, Spotify 30%, Temporal 15%, Vocal 15%
    print("\n[3/4] Processing Unified Mood & Context (Genre 50-D + Spotify 11-D + Temporal 10-D + Vocal 12-D -> 83-D)...")
    t_genre    = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "genre_hybrid_50d.npy").astype(np.float32)))
    t_spotify  = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "spotify_audio_11d.npy").astype(np.float32)))
    t_temporal = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "temporal_collab_10d.npy").astype(np.float32)))
    t_vocal    = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "vocal_dsp_12d.npy").astype(np.float32)))
    
    mood_blocks = [
        np.sqrt(0.40) * t_genre,
        np.sqrt(0.30) * t_spotify,
        np.sqrt(0.15) * t_temporal,
        np.sqrt(0.15) * t_vocal
    ]
    mood_fused = normalize_tensor(torch.cat(mood_blocks, dim=1))
    print(f"  Exact Mood & Context dimension: {mood_fused.shape[1]}-D")
    assert mood_fused.shape[1] == 83, f"Expected 83-D, got {mood_fused.shape[1]}-D"
    
    mood_top_idx, mood_top_sims = compute_top_k(mood_fused, k=250, device=device)
    save_knn_parquet(SIM_DIR / "knn_mood_top250.parquet", spotify_ids, mood_top_idx, mood_top_sims, k=250)

    # 4. Master Combined Multi-Modal Representation (3,795-D)
    # Weights: Audio 38%, Lyric 35%, Genre 11%, Spotify 8%, Temporal 4%, Vocal 4%
    print("\n[4/4] Processing Master Multimodal Fusion (Audio 38% + Lyric 35% + Genre 11% + Spotify 8% + Temporal 4% + Vocal 4% -> 3795-D)...")
    combined_blocks = [
        np.sqrt(0.38) * audio_fused,
        np.sqrt(0.35) * lyric_fused,
        np.sqrt(0.11) * t_genre,
        np.sqrt(0.08) * t_spotify,
        np.sqrt(0.04) * t_temporal,
        np.sqrt(0.04) * t_vocal
    ]
    combined_fused = normalize_tensor(torch.cat(combined_blocks, dim=1))
    print(f"  Exact master combined feature dimension: {combined_fused.shape[1]}-D")
    assert combined_fused.shape[1] == 3795, f"Expected 3795-D, got {combined_fused.shape[1]}-D"

    combined_top_idx, combined_top_sims = compute_top_k(combined_fused, k=250, device=device)
    save_knn_parquet(SIM_DIR / "knn_combined_top250.parquet", spotify_ids, combined_top_idx, combined_top_sims, k=250)

    print("\n" + "="*85)
    print("ALL 4 TOP-250 KNN MATRICES GENERATED SUCCESSFULLY!")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
