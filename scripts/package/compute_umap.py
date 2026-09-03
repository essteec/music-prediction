"""
2D Projection Coordinates Generator for Multi-Modal Maps.
Uses PCA-initialized 2D manifold projection on normalized audio, lyric, mood, and master multimodal representations.

Outputs:
- data/similarity/umap_2d_audio.parquet
- data/similarity/umap_2d_lyric.parquet
- data/similarity/umap_2d_mood.parquet
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
AUDIO_EMB_DIR = DATA_DIR / "embeddings" / "audio"
LYRIC_EMB_DIR = DATA_DIR / "embeddings" / "lyric"
META_EMB_DIR = DATA_DIR / "embeddings" / "metadata"
SIM_DIR = DATA_DIR / "similarity"

def l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

def compute_2d_projection(embeddings: np.ndarray) -> np.ndarray:
    n_comp = min(50, embeddings.shape[1], embeddings.shape[0])
    pca_50 = PCA(n_components=n_comp, random_state=42).fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=35, init='pca', learning_rate='auto', random_state=42, max_iter=1000)
    coords_2d = tsne.fit_transform(pca_50)
    min_val, max_val = coords_2d.min(), coords_2d.max()
    norm_coords = ((coords_2d - min_val) / (max_val - min_val) * 200.0) - 100.0
    return norm_coords.astype(np.float32)

def main():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)
    spotify_ids = df['track_id'].values

    # 1. Audio (1664-D)
    print("Loading Audio Embeddings (CLAP 512 + MERT 1024 + VGGish 128)...")
    clap = np.load(AUDIO_EMB_DIR / "clap_512d.npy")
    mert = np.load(AUDIO_EMB_DIR / "mert_330m_embeddings_1024d.npy")
    vgg  = np.load(AUDIO_EMB_DIR / "vggish_embeddings_128d.npy")
    audio_fused = l2_norm(np.concatenate([l2_norm(clap), l2_norm(mert), l2_norm(vgg)], axis=1))

    # 2. Lyric (2048-D)
    print("Loading Lyric Embeddings (Harrier 1024 + E5 1024)...")
    harrier = np.load(LYRIC_EMB_DIR / "harrier_embeddings_1024d.npy")
    e5      = np.load(LYRIC_EMB_DIR / "multilingual_e5_large_1024d.npy")
    has_lyrics = (np.linalg.norm(harrier, axis=1, keepdims=True) > 1e-6).astype(np.float32)
    lyric_fused = l2_norm(np.concatenate([l2_norm(harrier), l2_norm(e5)], axis=1)) * has_lyrics

    # 3. Mood (59-D)
    print("Loading Mood Matrices (Spotify 11 + Emotion 36 + Vocal DSP 12)...")
    t_spotify = np.load(META_EMB_DIR / "spotify_audio_11d.npy")
    t_emotion = np.load(META_EMB_DIR / "emotion_sentiment_36d.npy")
    t_vocal   = np.load(META_EMB_DIR / "vocal_dsp_12d.npy")
    mood_fused = l2_norm(np.concatenate([l2_norm(t_spotify), l2_norm(t_emotion), l2_norm(t_vocal)], axis=1))

    # 4. Master Combined (3795-D)
    print("Loading Master Combined Representations (3795-D)...")
    t_genre    = l2_norm(np.load(META_EMB_DIR / "genre_hybrid_50d.npy"))
    t_temporal = l2_norm(np.load(META_EMB_DIR / "temporal_collab_10d.npy"))

    combined_fused = l2_norm(np.concatenate([
        audio_fused,
        lyric_fused,
        l2_norm(t_spotify),
        l2_norm(t_vocal),
        t_genre,
        t_temporal
    ], axis=1))
    print(f"  Master Multimodal Dimension: {combined_fused.shape[1]}-D")
    assert combined_fused.shape[1] == 3795, f"Expected 3795-D, got {combined_fused.shape[1]}-D"

    # Compute 2D Projections
    print("\n[1/4] Computing 2D Projection for Audio Space...")
    a_2d = compute_2d_projection(audio_fused)
    pd.DataFrame({'row_idx': np.arange(n_songs, dtype=np.int32), 'track_id': spotify_ids, 'proj_x': np.round(a_2d[:, 0], 3), 'proj_y': np.round(a_2d[:, 1], 3)}).to_parquet(SIM_DIR / "umap_2d_audio.parquet", index=False)
    print(f"  -> Saved: {SIM_DIR / 'umap_2d_audio.parquet'}")

    print("\n[2/4] Computing 2D Projection for Lyric Space...")
    l_2d = compute_2d_projection(lyric_fused)
    pd.DataFrame({'row_idx': np.arange(n_songs, dtype=np.int32), 'track_id': spotify_ids, 'proj_x': np.round(l_2d[:, 0], 3), 'proj_y': np.round(l_2d[:, 1], 3)}).to_parquet(SIM_DIR / "umap_2d_lyric.parquet", index=False)
    print(f"  -> Saved: {SIM_DIR / 'umap_2d_lyric.parquet'}")

    print("\n[3/4] Computing 2D Projection for Mood & Vibe Space...")
    m_2d = compute_2d_projection(mood_fused)
    pd.DataFrame({'row_idx': np.arange(n_songs, dtype=np.int32), 'track_id': spotify_ids, 'proj_x': np.round(m_2d[:, 0], 3), 'proj_y': np.round(m_2d[:, 1], 3)}).to_parquet(SIM_DIR / "umap_2d_mood.parquet", index=False)
    print(f"  -> Saved: {SIM_DIR / 'umap_2d_mood.parquet'}")

    print("\n[4/4] Computing 2D Projection for Master Multimodal Space...")
    c_2d = compute_2d_projection(combined_fused)
    pd.DataFrame({'row_idx': np.arange(n_songs, dtype=np.int32), 'track_id': spotify_ids, 'proj_x': np.round(c_2d[:, 0], 3), 'proj_y': np.round(c_2d[:, 1], 3)}).to_parquet(SIM_DIR / "umap_2d_combined.parquet", index=False)
    print(f"  -> Saved: {SIM_DIR / 'umap_2d_combined.parquet'}")

    print("\n" + "="*75)
    print("ALL 4 2D PROJECTIONS GENERATED AND SAVED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
