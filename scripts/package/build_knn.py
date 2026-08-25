"""
Pre-computed Similarity Graph & Top-50 Nearest Neighbors Generator.
Uses FAISS on normalized multi-modal embeddings to build fast nearest-neighbor indexes.
Outputs:
- data/similarity/knn_audio_top50.parquet
- data/similarity/knn_lyric_top50.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SIM_DIR = DATA_DIR / "similarity"

def build_faiss_knn(embeddings: np.ndarray, k: int = 50):
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_emb = (embeddings / norms).astype(np.float32)

    dim = norm_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(norm_emb)

    # Search top k+1 (first one is self)
    distances, indices = index.search(norm_emb, k + 1)
    
    # Exclude self
    clean_indices = indices[:, 1:k+1]
    clean_distances = distances[:, 1:k+1]
    return clean_indices, clean_distances

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    # 1. Audio Similarity Graph (CLAP + MERT normalized concatenation)
    clap_fp = EMBEDDINGS_DIR / "audio" / "clap_512d.npy"
    mert_fp = EMBEDDINGS_DIR / "audio" / "mert_embeddings_768d.npy"

    if clap_fp.exists() and mert_fp.exists():
        print("Building Top-50 Audio Nearest Neighbors Graph (CLAP + MERT)...")
        clap = np.load(clap_fp)
        mert = np.load(mert_fp)
        # Normalize each modality before concatenating
        c_norm = clap / np.maximum(np.linalg.norm(clap, axis=1, keepdims=True), 1e-8)
        m_norm = mert / np.maximum(np.linalg.norm(mert, axis=1, keepdims=True), 1e-8)
        audio_combined = np.concatenate([c_norm, m_norm], axis=1)

        a_idx, a_dist = build_faiss_knn(audio_combined, k=50)

        df_audio_knn = pd.DataFrame({
            'row_idx': np.arange(n_songs, dtype=np.int32),
            'track_id': df['track_id'].values,
            'top50_neighbor_indices': [a_idx[i].tolist() for i in range(n_songs)],
            'top50_neighbor_track_ids': [[df['track_id'].iloc[idx] for idx in a_idx[i]] for i in range(n_songs)],
            'top50_similarities': [np.round(a_dist[i], 4).tolist() for i in range(n_songs)]
        })
        audio_out = SIM_DIR / "knn_audio_top50.parquet"
        df_audio_knn.to_parquet(audio_out, index=False)
        print(f"Saved Audio kNN to: {audio_out} (Size: {audio_out.stat().st_size / 1024:.1f} KB)")

    # 2. Lyric Similarity Graph (Multilingual-E5 + BGE-M3)
    e5_fp = EMBEDDINGS_DIR / "lyrics" / "multilingual_e5_large_1024d.npy"
    bge_fp = EMBEDDINGS_DIR / "lyrics" / "bge_m3_1024d.npy"

    if e5_fp.exists() and bge_fp.exists():
        print("\nBuilding Top-50 Lyric Nearest Neighbors Graph (E5 + BGE-M3)...")
        e5 = np.load(e5_fp)
        bge = np.load(bge_fp)
        e_norm = e5 / np.maximum(np.linalg.norm(e5, axis=1, keepdims=True), 1e-8)
        b_norm = bge / np.maximum(np.linalg.norm(bge, axis=1, keepdims=True), 1e-8)
        lyric_combined = np.concatenate([e_norm, b_norm], axis=1)

        l_idx, l_dist = build_faiss_knn(lyric_combined, k=50)

        df_lyric_knn = pd.DataFrame({
            'row_idx': np.arange(n_songs, dtype=np.int32),
            'track_id': df['track_id'].values,
            'top50_neighbor_indices': [l_idx[i].tolist() for i in range(n_songs)],
            'top50_neighbor_track_ids': [[df['track_id'].iloc[idx] for idx in l_idx[i]] for i in range(n_songs)],
            'top50_similarities': [np.round(l_dist[i], 4).tolist() for i in range(n_songs)]
        })
        lyric_out = SIM_DIR / "knn_lyric_top50.parquet"
        df_lyric_knn.to_parquet(lyric_out, index=False)
        print(f"Saved Lyric kNN to: {lyric_out} (Size: {lyric_out.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
