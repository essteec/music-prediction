"""
2D Projection Coordinates Generator for Interactive Song Map.
Uses PCA-initialized t-SNE / 2D projection on normalized audio, lyric, and multimodal embeddings.
Outputs:
- data/similarity/umap_2d_audio.parquet
- data/similarity/umap_2d_lyric.parquet
- data/similarity/umap_2d_combined.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SIM_DIR = DATA_DIR / "similarity"

def compute_2d_projection(embeddings: np.ndarray) -> np.ndarray:
    # 1. PCA to 50 dims first for fast and stable projection
    pca_50 = PCA(n_components=min(50, embeddings.shape[1]), random_state=42).fit_transform(embeddings)
    # 2. t-SNE with PCA initialization for smooth cluster layout
    tsne = TSNE(n_components=2, perplexity=35, init='pca', learning_rate='auto', random_state=42, n_iter_without_progress=150)
    coords_2d = tsne.fit_transform(pca_50)
    # Normalize coordinates to [-100, 100] range for WebGL canvas
    min_val, max_val = coords_2d.min(), coords_2d.max()
    norm_coords = ((coords_2d - min_val) / (max_val - min_val) * 200.0) - 100.0
    return norm_coords.astype(np.float32)

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    clap = np.load(EMBEDDINGS_DIR / "audio" / "clap_512d.npy")
    mert = np.load(EMBEDDINGS_DIR / "audio" / "mert_embeddings_768d.npy")
    e5 = np.load(EMBEDDINGS_DIR / "lyrics" / "multilingual_e5_large_1024d.npy")
    bge = np.load(EMBEDDINGS_DIR / "lyrics" / "bge_m3_1024d.npy")

    c_norm = clap / np.maximum(np.linalg.norm(clap, axis=1, keepdims=True), 1e-8)
    m_norm = mert / np.maximum(np.linalg.norm(mert, axis=1, keepdims=True), 1e-8)
    e_norm = e5 / np.maximum(np.linalg.norm(e5, axis=1, keepdims=True), 1e-8)
    b_norm = bge / np.maximum(np.linalg.norm(bge, axis=1, keepdims=True), 1e-8)

    audio_comb = np.concatenate([c_norm, m_norm], axis=1)
    lyric_comb = np.concatenate([e_norm, b_norm], axis=1)
    all_comb = np.concatenate([audio_comb, lyric_comb], axis=1)

    print("Computing 2D Projection for Audio Embeddings (CLAP + MERT)...")
    a_2d = compute_2d_projection(audio_comb)
    pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'proj_x': np.round(a_2d[:, 0], 3),
        'proj_y': np.round(a_2d[:, 1], 3)
    }).to_parquet(SIM_DIR / "umap_2d_audio.parquet", index=False)
    print(f"Saved: {SIM_DIR / 'umap_2d_audio.parquet'}")

    print("Computing 2D Projection for Lyric Embeddings (E5 + BGE-M3)...")
    l_2d = compute_2d_projection(lyric_comb)
    pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'proj_x': np.round(l_2d[:, 0], 3),
        'proj_y': np.round(l_2d[:, 1], 3)
    }).to_parquet(SIM_DIR / "umap_2d_lyric.parquet", index=False)
    print(f"Saved: {SIM_DIR / 'umap_2d_lyric.parquet'}")

    print("Computing 2D Projection for Multimodal Combined Embeddings...")
    comb_2d = compute_2d_projection(all_comb)
    pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'proj_x': np.round(comb_2d[:, 0], 3),
        'proj_y': np.round(comb_2d[:, 1], 3)
    }).to_parquet(SIM_DIR / "umap_2d_combined.parquet", index=False)
    print(f"Saved: {SIM_DIR / 'umap_2d_combined.parquet'}")

    print("\n2D Projection coordinate generation completed successfully!")

if __name__ == "__main__":
    main()
