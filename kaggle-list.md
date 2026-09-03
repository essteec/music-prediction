# Kaggle Dataset Upload File List

This document lists all curated files included in the Kaggle dataset upload package (`spotify-10k-music-features`), along with their exact paths, dimensions, dtypes, file sizes, and descriptions.

---

## 📂 1. Metadata & Pre-Rendered Tables (`metadata/` & `processed/`)

> **Note on CSVs**: The `.csv` versions of the metadata tables are included alongside the optimized `.parquet` files to enable instant visual table rendering and column filtering in the Kaggle web UI.

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `metadata/songs.parquet` | Parquet | (10000, 32) | 9.65 MB | Primary Spotify track metadata (titles, artists, genres, popularity, release date, tempo, valence, etc.) |
| `metadata/artists.parquet` | Parquet | (5015, 6) | 0.30 MB | Unique artist catalog (artist ID, name, followers, popularity, genres) |
| `metadata/genres.parquet` | Parquet | (1276, 2) | 0.02 MB | 1,276 Subgenre to 17 Main Genre taxonomy mapping table |
| `processed/songs.csv` | CSV | (10000, 32) | 19.89 MB | CSV format for Kaggle in-browser table preview |
| `processed/artists.csv` | CSV | (5015, 6) | 0.45 MB | CSV format for Kaggle in-browser table preview |
| `processed/genres.csv` | CSV | (1276, 2) | 0.03 MB | CSV format for Kaggle in-browser table preview |

---

## 🎵 2. Extracted Tabular Features (`features/`)

### Audio Acoustic Descriptors (`features/audio/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/audio/dsp_librosa.parquet` | Parquet | (10000, 91) | 5.29 MB | 88 Librosa acoustic descriptors (MFCCs 1–20 mean/std, spectral contrast/rolloff/flatness, chroma pitch, tonnetz, onset rate, tempo, LUFS integrated loudness, stereo width) |
| `features/audio/vad.parquet` | Parquet | (10000, 5) | 0.39 MB | Silero Vocal Activity Detection (vocal duration in seconds, vocal ratio 0–1, vocal presence flag) |

### Lyric NLP Descriptors (`features/lyric/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/lyric/language_id.parquet` | Parquet | (10000, 34) | 0.36 MB | 34 Language & script flags (Devanagari Hindi, Romanized Hindi, Indonesian, Japanese, Korean, Chinese, European, Turkish, etc.) |
| `features/lyric/lyric_stats.parquet` | Parquet | (10000, 30) | 1.04 MB | Lexical statistics (TTR, Root-TTR, MTLD, HD-D, Hapax legomena ratio, Flesch reading ease, VADER compound/pos/neg, NRC EmoLex 8 emotions, YAKE top keywords JSON) |
| `features/lyric/go_emotions.parquet` | Parquet | (10000, 31) | 0.73 MB | 28 Fine-grained RoBERTa emotion probability scores on English tracks (`admiration`, `joy`, `love`, `sadness`, `anger`, `optimism`, etc.) |
| `features/lyric/bertopic_topics.parquet` | Parquet | (10000, 3) | 0.32 MB | 32 Thematic lyric clusters derived from BGE-M3 representations |
| `features/lyric/bertopic_topic_labels.json` | JSON | 32 topics | 0.01 MB | Topic names, document counts, and top c-TF-IDF keyword labels per topic |

### Metadata Contextual Features (`features/metadata/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/metadata/derived.parquet` | Parquet | (10000, 17) | 0.41 MB | Structural features (release decade, collaboration flag, total artist followers log, genre counts, key/mode name) |

### Acoustic Fingerprints & Recognition (`features/qc/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/qc/chromaprint_fingerprints.parquet` | Parquet | (10000, 6) | 35.57 MB | AcoustID Chromaprint raw base64 fingerprints & duplicate clusters for Shazam-like audio recognition |

---

## 🧠 3. High-Dimensional Embeddings & Normalized Matrices (`embeddings/`)

### Deep Audio Embeddings (`embeddings/audio/`)
| File Path | Format | Shape / Dtype | Size | Description |
|---|---|---|---|---|
| `embeddings/audio/clap_512d.npy` | NumPy | (10000, 512) float32 | 19.53 MB | LAION-CLAP zero-shot text-audio full-song mean embeddings (48 kHz mono) |
| `embeddings/audio/mert_330m_embeddings_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | MERT-v1-330M self-supervised music transformer full-song mean embeddings (24 kHz mono) |
| `embeddings/audio/mert_330m_all_chunks.npz` | NPZ | (74559, 1024) float16 | 167.05 MB | MERT-v1-330M CSR-indexed tensor of all consecutive 30s chunk representations |
| `embeddings/audio/vggish_embeddings_128d.npy` | NumPy | (10000, 128) float32 | 4.88 MB | Google VGGish deep acoustic full-song embeddings (16 kHz mono) |
| `embeddings/audio/panns_embeddings_2048d.npy` | NumPy | (10000, 2048) float32 | 78.13 MB | PANNs Cnn14 deep audio representations |
| `embeddings/audio/panns_tags_527d.npy` | NumPy | (10000, 527) float32 | 20.10 MB | PANNs 527 AudioSet sound & music class probability vectors |
| `embeddings/audio/panns_tags_labels.json` | JSON | 527 classes | 0.02 MB | Human-readable AudioSet class name mappings |
| `embeddings/audio/mel_stats_embeddings_512d.npy` | NumPy | (10000, 512) float32 | 19.53 MB | Mel-frequency spectral statistics embeddings |

### Deep Lyric Embeddings (`embeddings/lyric/`)
| File Path | Format | Shape / Dtype | Size | Description |
|---|---|---|---|---|
| `embeddings/lyric/harrier_embeddings_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | Microsoft Harrier-OSS-v1-0.6B embeddings (32k context, full song lyrics without truncation) |
| `embeddings/lyric/multilingual_e5_large_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | Multilingual-E5-Large embeddings (Pilot benchmark winner, nDCG@10: 0.3001) |
| `embeddings/lyric/bge_m3_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | BAAI/BGE-M3 embeddings (8,192-token document context, 100+ languages) |

### Normalized Tabular Feature Matrices (`embeddings/metadata/`)
| File Path | Format | Shape / Dtype | Size | Description |
|---|---|---|---|---|
| `embeddings/metadata/genre_hybrid_50d.npy` | NumPy | (10000, 50) float32 | 1.91 MB | Hybrid Genre Representation (17-D Main + 17-D Sub Rollup + 16-D Latent SVD) |
| `embeddings/metadata/spotify_audio_11d.npy` | NumPy | (10000, 11) float32 | 0.42 MB | Normalized Spotify audio descriptors (danceability, valence, energy, tempo, loudness, etc.) |
| `embeddings/metadata/vocal_dsp_12d.npy` | NumPy | (10000, 12) float32 | 0.46 MB | Normalized Vocal Activity (VAD ratio) + 10 Librosa dynamic descriptors |
| `embeddings/metadata/temporal_collab_10d.npy` | NumPy | (10000, 10) float32 | 0.38 MB | Normalized release decade, collaboration flags, artist follower counts |
| `embeddings/metadata/emotion_sentiment_36d.npy` | NumPy | (10000, 36) float32 | 1.37 MB | 28 RoBERTa GoEmotions probabilities + 8 NRC Lexicon emotion densities |
| `embeddings/metadata/lyric_stats_12d.npy` | NumPy | (10000, 12) float32 | 0.46 MB | Normalized TTR, readability index, VADER sentiment, line repetition ratios |
| `embeddings/metadata/language_27d.npy` | NumPy | (10000, 27) float32 | 1.03 MB | Confidence-weighted binary indicators for 26 languages + multilingual flag |
| `embeddings/metadata/bertopic_32d.npy` | NumPy | (10000, 32) float32 | 1.22 MB | 32-D one-hot thematic lyric cluster vectors (Archival table) |

---

## 🔍 4. Similarity Graphs & 2D Map Projections (`similarity/`)

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `similarity/knn_combined_top100.parquet` | Parquet | (10000, 5) | 9.49 MB | Pre-computed Top-100 nearest neighbors for Master Multimodal Fusion (3,795-D: Audio + Lyric + Spotify + Vocal DSP + Genre + Temporal) |
| `similarity/knn_audio_top100.parquet` | Parquet | (10000, 5) | 9.32 MB | Pre-computed Top-100 nearest neighbors for Pure Acoustic Sound (1,664-D: CLAP + MERT-330M + VGGish) |
| `similarity/knn_lyric_top100.parquet` | Parquet | (10000, 5) | 9.19 MB | Pre-computed Top-100 nearest neighbors for Lyric Storytelling (2,048-D: Harrier-0.6B + Multilingual-E5) |
| `similarity/knn_mood_top100.parquet` | Parquet | (10000, 5) | 9.49 MB | Pre-computed Top-100 nearest neighbors for Mood & Emotion (59-D: Spotify 11-D + Emotion 36-D + Vocal DSP 12-D) |
| `similarity/umap_2d_combined.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D Multimodal map projection coordinates (`proj_x`, `proj_y`) for visual mapping |
| `similarity/umap_2d_audio.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D Audio map projection coordinates (`proj_x`, `proj_y`) for visual mapping |
| `similarity/umap_2d_lyric.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D Lyric map projection coordinates (`proj_x`, `proj_y`) for visual mapping |
| `similarity/umap_2d_mood.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D Mood map projection coordinates (`proj_x`, `proj_y`) for visual mapping |

---

## ⚖️ 5. Evaluation Splits (`splits/`)

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `splits/artist_grouped_5fold.parquet` | Parquet | (10000, 4) | 0.05 MB | 5-Fold GroupKFold by `artist_id` for zero-leakage cross-validation evaluation |
| `splits/temporal_split.parquet` | Parquet | (10000, 4) | 0.05 MB | Chronological split: Train (≤2022, 6,549 songs), Val (2023, 843 songs), Test (2024–2025, 2,608 songs) |

---

## 📋 6. Manifests & Documentation

| File Path | Format | Size | Description |
|---|---|---|---|
| `track_ids.npy` | NumPy (10000,) | 0.08 MB | Master Spotify track ID array (0 to 9,999) ensuring 1:1 alignment across all files |
| `manifests/extraction_manifest.json` | JSON | 0.02 MB | Full schema, column list, shapes, and extraction metadata |
| `manifests/checksums.json` | JSON | 0.01 MB | SHA-256 cryptographic checksums for data integrity verification |
| `DATA_DICTIONARY.md` | Markdown | 0.01 MB | Detailed column-by-column documentation and metric definitions |
| `KAGGLE_README.md` | Markdown | 0.01 MB | Dataset card, citations, methodology, and quickstart documentation |
