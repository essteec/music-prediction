"""
Build Complete Suite of Top-100 kNN Graph Matrices & 2D Manifold Projections.

Suite of Similarity Graphs:
1. knn_audio_top100.parquet:    CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D) -> 1664-D
2. knn_lyric_top100.parquet:    Harrier-0.6B (1024-D) + Multilingual E5-Large (1024-D) -> 2048-D
3. knn_mood_top100.parquet:     Spotify Audio (11-D) + GoEmotions/NRC (36-D) + Vocal DSP (12-D) -> 59-D
4. knn_combined_top100.parquet: Audio (1664-D) + Lyric (2048-D) + Spotify (11-D) +
                               Vocal DSP (12-D) + Genre Hybrid (50-D) + Temporal (10-D) -> 3795-D
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

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    df_songs = pd.read_csv(SONGS_CSV)
    spotify_ids = df_songs['track_id'].tolist()
    n_songs = len(spotify_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*85)
    print(f"BUILDING OFFICIAL 4-FACET TOP-100 KNN MATRICES ({device})")
    print(f"Total Tracks: {n_songs:,}")
    print("="*85 + "\n")

    # 1. Neural Audio Representation (1664-D)
    print("[1/4] Processing Acoustic Similarity (CLAP + MERT + VGGish -> 1664-D)...")
    clap = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "clap_512d.npy").astype(np.float32)))
    mert = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "mert_330m_embeddings_1024d.npy").astype(np.float32)))
    vgg  = normalize_tensor(torch.from_numpy(np.load(AUDIO_EMB_DIR / "vggish_embeddings_128d.npy").astype(np.float32)))
    audio_fused = normalize_tensor(torch.cat([clap, mert, vgg], dim=1))
    audio_top100_idx, audio_top100_sims = compute_top_k(audio_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_audio_top100.parquet", spotify_ids, audio_top100_idx, audio_top100_sims)

    # 2. Neural Lyric Representation (2048-D)
    print("\n[2/4] Processing Lyrical Storytelling (Harrier-0.6B + E5-Large -> 2048-D)...")
    harrier = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "harrier_embeddings_1024d.npy").astype(np.float32)))
    e5      = normalize_tensor(torch.from_numpy(np.load(LYRIC_EMB_DIR / "multilingual_e5_large_1024d.npy").astype(np.float32)))
    has_lyrics = (torch.norm(harrier, p=2, dim=1, keepdim=True) > 1e-6).float()
    lyric_fused = normalize_tensor(torch.cat([harrier, e5], dim=1)) * has_lyrics
    lyric_top100_idx, lyric_top100_sims = compute_top_k(lyric_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_lyric_top100.parquet", spotify_ids, lyric_top100_idx, lyric_top100_sims)

    # 3. Dedicated Mood & Vibe Representation (59-D)
    print("\n[3/4] Processing Mood & Vibe (Spotify 11-D + GoEmotions/NRC 36-D + Vocal DSP 12-D -> 59-D)...")
    t_spotify = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "spotify_audio_11d.npy").astype(np.float32)))
    t_emotion = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "emotion_sentiment_36d.npy").astype(np.float32)))
    t_vocal   = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "vocal_dsp_12d.npy").astype(np.float32)))
    mood_fused = normalize_tensor(torch.cat([t_spotify, t_emotion, t_vocal], dim=1))
    mood_top100_idx, mood_top100_sims = compute_top_k(mood_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_mood_top100.parquet", spotify_ids, mood_top100_idx, mood_top100_sims)

    # 4. Master Combined Multi-Modal Representation (3795-D)
    print("\n[4/4] Processing Master Multimodal Fusion (Audio 1664 + Lyric 2048 + Spotify 11 + Vocal 12 + Genre 50 + Temporal 10 -> 3795-D)...")
    t_genre    = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "genre_hybrid_50d.npy").astype(np.float32)))
    t_temporal = normalize_tensor(torch.from_numpy(np.load(META_EMB_DIR / "temporal_collab_10d.npy").astype(np.float32)))
    
    combined_blocks = [
        audio_fused,
        lyric_fused,
        t_spotify,
        t_vocal,
        t_genre,
        t_temporal
    ]
    combined_fused = normalize_tensor(torch.cat(combined_blocks, dim=1))
    print(f"  Exact master combined feature dimension: {combined_fused.shape[1]}-D")
    assert combined_fused.shape[1] == 3795, f"Expected 3795-D, got {combined_fused.shape[1]}-D"

    combined_top100_idx, combined_top100_sims = compute_top_k(combined_fused, k=100, device=device)
    save_knn_parquet(SIM_DIR / "knn_combined_top100.parquet", spotify_ids, combined_top100_idx, combined_top100_sims)

    print("\n" + "="*85)
    print("ALL 4 TOP-100 KNN MATRICES GENERATED SUCCESSFULLY!")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
