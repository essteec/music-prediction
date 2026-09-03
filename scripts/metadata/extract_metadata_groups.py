"""
Extract and normalize tabular metadata into 8 standardized .npy matrices.

Covers 100% of audited metadata & feature tables:
1. data/embeddings/metadata/spotify_audio_11d.npy      (songs.parquet)
2. data/embeddings/metadata/emotion_sentiment_36d.npy  (go_emotions.parquet + lyric_stats NRC)
3. data/embeddings/metadata/vocal_dsp_12d.npy          (vad.parquet + dsp_librosa.parquet)
4. data/embeddings/metadata/lyric_stats_12d.npy        (lyric_stats.parquet)
5. data/embeddings/metadata/genre_hybrid_50d.npy       (17-D Main + 17-D Subgenre Rollup + 16-D Latent SVD)
6. data/embeddings/metadata/language_27d.npy           (language_id.parquet)
7. data/embeddings/metadata/temporal_collab_10d.npy    (derived.parquet)
8. data/embeddings/metadata/bertopic_32d.npy           (bertopic_topics.parquet - archival table, excluded from global fusion)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "embeddings" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def robust_scale(series: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    """Robust Min-Max scale between 1st and 99th percentile, clipped to [0, 1]."""
    val = series.fillna(series.median()).values.astype(np.float32)
    low = np.percentile(val, q_low * 100)
    high = np.percentile(val, q_high * 100)
    if high - low < 1e-6:
        return np.zeros_like(val, dtype=np.float32)
    scaled = (val - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)

def normalize_genre_str(s: str) -> str:
    """Standardize genre string: lowercase, stripped whitespace, normalized spaces."""
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().lower().split())

def main():
    print("\n" + "="*75)
    print("EXTRACTING COMPLETE 8-GROUP METADATA FEATURE MATRICES (50-D GENRE HYBRID)")
    print("="*75 + "\n")

    # Load all source tables
    songs = pd.read_parquet(DATA_DIR / "metadata" / "songs.parquet")
    genres_df = pd.read_parquet(DATA_DIR / "metadata" / "genres.parquet")
    derived = pd.read_parquet(DATA_DIR / "features" / "metadata" / "derived.parquet")
    dsp = pd.read_parquet(DATA_DIR / "features" / "audio" / "dsp_librosa.parquet")
    vad = pd.read_parquet(DATA_DIR / "features" / "audio" / "vad.parquet")
    go_emo = pd.read_parquet(DATA_DIR / "features" / "lyric" / "go_emotions.parquet")
    lyric_stats = pd.read_parquet(DATA_DIR / "features" / "lyric" / "lyric_stats.parquet")
    lang_id = pd.read_parquet(DATA_DIR / "features" / "lyric" / "language_id.parquet")
    topics = pd.read_parquet(DATA_DIR / "features" / "lyric" / "bertopic_topics.parquet")

    n_songs = len(songs)
    print(f"Loaded {n_songs:,} tracks across all 8 feature tables.\n")

    # 1. Spotify Audio & Vibe (11-D)
    print("1. Building Spotify Audio & Vibe Matrix (11-D)...")
    spotify_cols = [
        songs['danceability'].fillna(0.5).values,
        songs['energy'].fillna(0.5).values,
        songs['valence'].fillna(0.5).values,
        songs['acousticness'].fillna(0.0).values,
        songs['instrumentalness'].fillna(0.0).values,
        songs['speechiness'].fillna(0.0).values,
        songs['liveness'].fillna(0.0).values,
        songs['mode'].fillna(1.0).values,
        robust_scale(songs['loudness']),
        robust_scale(songs['tempo']),
        songs['time_signature'].fillna(4).clip(1, 5).values / 5.0,
    ]
    spotify_audio_11d = np.column_stack(spotify_cols).astype(np.float32)
    np.save(OUT_DIR / "spotify_audio_11d.npy", spotify_audio_11d)
    print(f"   -> Saved: {OUT_DIR / 'spotify_audio_11d.npy'} | Shape: {spotify_audio_11d.shape}")

    # 2. Emotion & Sentiment (36-D)
    print("2. Building Emotion & Sentiment Matrix (36-D)...")
    emo_cols = [c for c in go_emo.columns if c.startswith('emotion_')]
    go_emo_matrix = go_emo[emo_cols].fillna(0.0).values.astype(np.float32)
    nrc_cols = ['nrc_anger', 'nrc_fear', 'nrc_anticipation', 'nrc_trust', 'nrc_surprise', 'nrc_sadness', 'nrc_joy', 'nrc_disgust']
    nrc_matrix = lyric_stats[nrc_cols].fillna(0.0).values.astype(np.float32)
    for i in range(nrc_matrix.shape[1]):
        nrc_matrix[:, i] = robust_scale(pd.Series(nrc_matrix[:, i]))
    emotion_sentiment_36d = np.hstack([go_emo_matrix, nrc_matrix]).astype(np.float32)
    np.save(OUT_DIR / "emotion_sentiment_36d.npy", emotion_sentiment_36d)
    print(f"   -> Saved: {OUT_DIR / 'emotion_sentiment_36d.npy'} | Shape: {emotion_sentiment_36d.shape}")

    # 3. Vocal & DSP Dynamics (12-D)
    print("3. Building Vocal & DSP Dynamics Matrix (12-D)...")
    vocal_dsp_cols = [
        vad['vocal_ratio'].fillna(0.0).values,
        vad['has_vocals'].astype(float).fillna(0.0).values,
        robust_scale(dsp['crest_factor']),
        robust_scale(dsp['lufs_integrated']),
        robust_scale(dsp['onset_rate']),
        robust_scale(dsp['onset_strength_mean']),
        robust_scale(dsp['spectral_centroid_mean']),
        robust_scale(dsp['spectral_contrast_mean']),
        robust_scale(dsp['zcr_mean']),
        robust_scale(dsp['chroma_entropy']),
        robust_scale(dsp['stereo_width']),
        robust_scale(dsp['lr_correlation']),
    ]
    vocal_dsp_12d = np.column_stack(vocal_dsp_cols).astype(np.float32)
    np.save(OUT_DIR / "vocal_dsp_12d.npy", vocal_dsp_12d)
    print(f"   -> Saved: {OUT_DIR / 'vocal_dsp_12d.npy'} | Shape: {vocal_dsp_12d.shape}")

    # 4. Lyric Structure & Complexity (12-D)
    print("4. Building Lyric Structure & Complexity Matrix (12-D)...")
    lyric_cols = [
        lyric_stats['unique_line_ratio'].fillna(0.0).values,
        lyric_stats['repeated_line_ratio'].fillna(0.0).values,
        lyric_stats['ttr'].fillna(0.0).values,
        lyric_stats['hapax_ratio'].fillna(0.0).values,
        robust_scale(lyric_stats['avg_line_char_len']),
        robust_scale(lyric_stats['line_char_len_std']),
        robust_scale(lyric_stats['flesch_reading_ease']),
        robust_scale(lyric_stats['stanza_count']),
        robust_scale(lyric_stats['vader_compound']),
        lyric_stats['vader_pos'].fillna(0.0).values,
        lyric_stats['vader_neg'].fillna(0.0).values,
        lyric_stats['vader_neu'].fillna(0.0).values,
    ]
    lyric_stats_12d = np.column_stack(lyric_cols).astype(np.float32)
    np.save(OUT_DIR / "lyric_stats_12d.npy", lyric_stats_12d)
    print(f"   -> Saved: {OUT_DIR / 'lyric_stats_12d.npy'} | Shape: {lyric_stats_12d.shape}")

    # 5. Hybrid Genre Vector (50-D: 17-D Main + 17-D Subgenre Rollup + 16-D Latent SVD Subgenre)
    print("5. Building Hybrid Genre Vector (50-D: 17 Main + 17 Sub Rollup + 16 Latent SVD)...")
    sub2main = {}
    for _, r in genres_df.iterrows():
        sub_norm = normalize_genre_str(r['subgenre'])
        mains = [normalize_genre_str(m) for m in str(r['main_genre']).split('|') if m.strip()]
        sub2main[sub_norm] = mains

    MAIN_GENRES = sorted(list({g for mains in sub2main.values() for g in mains}))
    genre2idx = {g: i for i, g in enumerate(MAIN_GENRES)}
    assert len(MAIN_GENRES) == 17, f"Expected 17 canonical main genres, found {len(MAIN_GENRES)}: {MAIN_GENRES}"

    # Block A: main_17d (song main genre multi-hot)
    main_17d = np.zeros((n_songs, 17), dtype=np.float32)
    unmapped_mains = set()
    for i, mg in enumerate(songs['main_genres']):
        if pd.notna(mg):
            for g in str(mg).split(','):
                g_norm = normalize_genre_str(g)
                if g_norm in genre2idx:
                    main_17d[i, genre2idx[g_norm]] = 1.0
                else:
                    unmapped_mains.add(g_norm)

    if unmapped_mains:
        print(f"   [WARNING] Unmapped main_genres strings encountered: {unmapped_mains}")
    else:
        print("   ✓ 100% Main Genre Coverage: 0 unmapped strings.")

    # Block B: sub_affinity_17d (artist subgenres rolled up into 17-D main taxonomy)
    sub_affinity_17d = np.zeros((n_songs, 17), dtype=np.float32)
    unmapped_subs = set()
    unmapped_occurrences = 0
    for i, ag in enumerate(songs['artist_genres']):
        if pd.notna(ag):
            subs = [normalize_genre_str(s) for s in str(ag).split(',') if s.strip()]
            for s in subs:
                if s in sub2main:
                    for parent_g in sub2main[s]:
                        sub_affinity_17d[i, genre2idx[parent_g]] += 1.0
                else:
                    unmapped_subs.add(s)
                    unmapped_occurrences += 1

    if unmapped_subs:
        print(f"   [WARNING] Unmapped artist subgenres encountered ({unmapped_occurrences} occurrences): {unmapped_subs}")
    else:
        print("   ✓ 100% Subgenre Taxonomy Coverage across all 1,276 subgenres: 0 unmapped strings.")

    # L1 normalize sub_affinity_17d across rows
    row_sums = sub_affinity_17d.sum(axis=1, keepdims=True)
    sub_affinity_17d_norm = np.divide(sub_affinity_17d, row_sums, out=np.zeros_like(sub_affinity_17d), where=row_sums > 0)

    # Block C: subgenre_svd_16d (16-D Latent Subgenre Co-Occurrence Space)
    all_subgenres = sorted(list(sub2main.keys()))
    sub2vocab_idx = {s: idx for idx, s in enumerate(all_subgenres)}
    X_sub = np.zeros((n_songs, len(all_subgenres)), dtype=np.float32)
    for i, ag in enumerate(songs['artist_genres']):
        if pd.notna(ag):
            subs = [normalize_genre_str(s) for s in str(ag).split(',') if s.strip()]
            for s in subs:
                if s in sub2vocab_idx:
                    X_sub[i, sub2vocab_idx[s]] = 1.0

    svd_16 = TruncatedSVD(n_components=16, random_state=42)
    subgenre_svd_16d = svd_16.fit_transform(X_sub).astype(np.float32)
    cum_var_16 = np.sum(svd_16.explained_variance_ratio_) * 100.0
    print(f"   ✓ Fitted TruncatedSVD(16): captures {cum_var_16:.2f}% of subgenre co-occurrence variance.")

    # Concatenate into 50-D matrix: 34-D Macro Taxonomy Anchor + 16-D Latent SVD Subgenre Space
    macro_34d = np.hstack([main_17d, sub_affinity_17d_norm])
    def l2_norm_block(a):
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        nonzero = (norms > 1e-6).astype(np.float32)
        return (a / np.maximum(norms, 1e-12)) * nonzero

    macro_norm = l2_norm_block(macro_34d)
    svd_norm = l2_norm_block(subgenre_svd_16d)
    genre_hybrid_50d = np.hstack([macro_norm, svd_norm]).astype(np.float32)
    np.save(OUT_DIR / "genre_hybrid_50d.npy", genre_hybrid_50d)
    print(f"   -> Saved: {OUT_DIR / 'genre_hybrid_50d.npy'} | Shape: {genre_hybrid_50d.shape}")

    # Remove old 34-D file if present to prevent stale artifacts
    old_34d_file = OUT_DIR / "genre_taxonomy_34d.npy"
    if old_34d_file.exists():
        old_34d_file.unlink()
        print(f"   -> Cleaned up obsolete: {old_34d_file.name}")

    # 6. Linguistic & Language Identity (27-D)
    print("6. Building Linguistic & Language Matrix (27-D)...")
    lang_is_cols = [c for c in lang_id.columns if c.startswith('is_')]
    conf = lang_id['lang_confidence'].fillna(0.0).values[:, None]
    lang_matrix = lang_id[lang_is_cols].astype(float).fillna(0.0).values * conf
    language_27d = lang_matrix.astype(np.float32)
    np.save(OUT_DIR / "language_27d.npy", language_27d)
    print(f"   -> Saved: {OUT_DIR / 'language_27d.npy'} | Shape: {language_27d.shape}")

    # 7. Temporal & Collaboration Context (10-D)
    print("7. Building Temporal & Collaboration Matrix (10-D)...")
    temp_collab_cols = [
        robust_scale(derived['release_year']),
        derived['is_2020s'].astype(float).fillna(0.0).values,
        derived['is_2010s'].astype(float).fillna(0.0).values,
        derived['is_2000s'].astype(float).fillna(0.0).values,
        derived['is_pre_2000'].astype(float).fillna(0.0).values,
        derived['is_collaboration'].astype(float).fillna(0.0).values,
        robust_scale(derived['n_artists']),
        robust_scale(derived['log_total_artist_followers']),
        robust_scale(derived['duration_min']),
        derived['is_explicit'].astype(float).fillna(0.0).values,
    ]
    temporal_collab_10d = np.column_stack(temp_collab_cols).astype(np.float32)
    np.save(OUT_DIR / "temporal_collab_10d.npy", temporal_collab_10d)
    print(f"   -> Saved: {OUT_DIR / 'temporal_collab_10d.npy'} | Shape: {temporal_collab_10d.shape}")

    # 8. BERTopic Distribution (32-D) - Archival
    print("8. Building BERTopic One-Hot Matrix (32-D, Archival Table)...")
    topic_ids = topics['topic_id'].fillna(0).astype(int).clip(0, 31).values
    bertopic_32d = np.zeros((n_songs, 32), dtype=np.float32)
    bertopic_32d[np.arange(n_songs), topic_ids] = 1.0
    has_lyric_mask = (lang_id['has_lyrics'].astype(float).values)[:, None]
    bertopic_32d = bertopic_32d * has_lyric_mask
    np.save(OUT_DIR / "bertopic_32d.npy", bertopic_32d)
    print(f"   -> Saved: {OUT_DIR / 'bertopic_32d.npy'} | Shape: {bertopic_32d.shape}")

    print("\n" + "="*75)
    print("ALL 8 METADATA FEATURE GROUPS EXTRACTED AND SAVED SUCCESSFULLY")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
