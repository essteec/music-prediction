# Kaggle Dataset Upload File List

This document lists all 40 files included in the Kaggle dataset package (`spotify-10k-music-features`), along with their exact paths, dimensions, dtypes, file sizes, and descriptions.

---

## 📂 1. Metadata (`metadata/`)

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `metadata/songs.parquet` | Parquet | (10000, 32) | 9.65 MB | Primary Spotify track metadata (titles, artists, genres, popularity, release date, tempo, valence, etc.) |
| `metadata/artists.parquet` | Parquet | (5015, 6) | 0.30 MB | Unique artist catalog (artist ID, name, followers, popularity, genres) |
| `metadata/genres.parquet` | Parquet | (1276, 2) | 0.02 MB | Spotify genre taxonomy & mapping table |

---

## 🎵 2. Extracted Features (`features/`)

### Audio Features (`features/audio/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/audio/dsp_librosa.parquet` | Parquet | (10000, 91) | 5.29 MB | 88 Librosa acoustic descriptors (MFCCs 1–20 mean/std, spectral contrast/rolloff/flatness, chroma pitch, tonnetz, onset rate, tempo, LUFS integrated loudness, stereo width) |
| `features/audio/vad.parquet` | Parquet | (10000, 5) | 0.39 MB | Silero Vocal Activity Detection (vocal duration in seconds, vocal ratio 0–1, vocal presence flag) |

### Lyric Features (`features/lyric/`)
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/lyric/language_id.parquet` | Parquet | (10000, 35) | 0.36 MB | 35 Language & script flags (Devanagari Hindi, Romanized Hindi, Indonesian, Japanese, Korean, Chinese, European, Turkish, etc.) |
| `features/lyric/lyric_stats.parquet` | Parquet | (10000, 30) | 1.04 MB | Lexical statistics (TTR, Root-TTR, MTLD, HD-D, Hapax legomena ratio, Flesch reading ease, VADER compound/pos/neg, NRC EmoLex 8 emotions, YAKE top keywords JSON) |
| `features/lyric/go_emotions.parquet` | Parquet | (10000, 31) | 0.73 MB | 28 Fine-grained RoBERTa emotion probability scores on English tracks (`admiration`, `joy`, `love`, `sadness`, `anger`, `optimism`, etc.) |
| `features/lyric/bertopic_topics.parquet` | Parquet | (10000, 3) | 0.32 MB | 32 Thematic lyric clusters derived from BGE-M3 representations |
| `features/lyric/bertopic_topic_labels.json` | JSON | 32 topics | 0.01 MB | Topic names, document counts, and top c-TF-IDF keyword labels per topic |

### Metadata Derived & QC Features
| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `features/metadata/derived.parquet` | Parquet | (10000, 17) | 0.41 MB | Structural features (release decade, collaboration flag, total artist followers log, genre counts, key/mode name) |
| `features/qc/audio_qc.parquet` | Parquet | (10000, 10) | 0.48 MB | Audio file QC verification metrics (duration match, sample rate, bitrate, channel count) |
| `features/qc/chromaprint_fingerprints.parquet` | Parquet | (10000, 6) | 35.57 MB | Acoustic Chromaprint fingerprints and 115 identified duplicate track clusters |

---

## 🧠 3. Neural Embeddings (`embeddings/`)

### Audio Embeddings (`embeddings/audio/`)
| File Path | Format | Shape / Dtype | Size | Description |
|---|---|---|---|---|
| `embeddings/audio/clap_512d.npy` | NumPy | (10000, 512) float32 | 19.53 MB | LAION-CLAP zero-shot text-audio cross-modal full-song mean embeddings |
| `embeddings/audio/mert_330m_embeddings_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | MERT-v1-330M self-supervised music transformer full-song mean embeddings |
| `embeddings/audio/mert_330m_all_chunks.npz` | NPZ | (74559, 1024) float16 | 167.05 MB | MERT-v1-330M CSR-indexed tensor of all consecutive 30s chunk representations |
| `embeddings/audio/mert_embeddings_768d.npy` | NumPy | (10000, 768) float32 | 29.30 MB | MERT-v1-95M representations |
| `embeddings/audio/panns_embeddings_2048d.npy` | NumPy | (10000, 2048) float32 | 78.13 MB | PANNs Cnn14 deep audio representations |
| `embeddings/audio/panns_tags_527d.npy` | NumPy | (10000, 527) float32 | 20.10 MB | PANNs 527 AudioSet sound & music class probability vectors |
| `embeddings/audio/panns_tags_labels.json` | JSON | 527 classes | 0.02 MB | Human-readable AudioSet class name mappings |
| `embeddings/audio/vggish_embeddings_128d.npy` | NumPy | (10000, 128) float32 | 4.88 MB | VGGish deep acoustic full-song embeddings |
| `embeddings/audio/mel_stats_embeddings_512d.npy` | NumPy | (10000, 512) float32 | 19.53 MB | Mel-frequency spectral statistics embeddings |

### Lyric Embeddings (`embeddings/lyric/`)
| File Path | Format | Shape / Dtype | Size | Description |
|---|---|---|---|---|
| `embeddings/lyric/harrier_embeddings_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | Microsoft Harrier-OSS-v1-0.6B embeddings (MTEB rank #10, 32k context) |
| `embeddings/lyric/multilingual_e5_large_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | Multilingual-E5-Large embeddings (Pilot benchmark winner, nDCG@10: 0.3001) |
| `embeddings/lyric/bge_m3_1024d.npy` | NumPy | (10000, 1024) float32 | 39.06 MB | BAAI/BGE-M3 embeddings (8,192-token full document context, 100+ languages) |

---

## 🔍 4. Similarity Graphs & 2D Map Projections (`similarity/`)

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `similarity/knn_audio_top100.parquet` | Parquet | (10000, 5) | 9.32 MB | Pre-computed Top-100 nearest neighbors for Audio (CLAP + MERT-330M + VGGish) with track IDs & similarity scores |
| `similarity/knn_lyric_top100.parquet` | Parquet | (10000, 5) | 9.19 MB | Pre-computed Top-100 nearest neighbors for Lyrics (Harrier + E5-Large) with track IDs & similarity scores |
| `similarity/knn_combined_top100.parquet` | Parquet | (10000, 5) | 9.30 MB | Pre-computed Top-100 nearest neighbors for Combined Multimodal (50% Audio + 50% Lyric) |
| `similarity/umap_2d_audio.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D audio map projection coordinates (`proj_x`, `proj_y`) normalized for WebGL canvas |
| `similarity/umap_2d_lyric.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D lyric map projection coordinates (`proj_x`, `proj_y`) normalized for WebGL canvas |
| `similarity/umap_2d_combined.parquet` | Parquet | (10000, 4) | 0.41 MB | 2D multimodal map projection coordinates (`proj_x`, `proj_y`) normalized for WebGL canvas |

---

## ⚖️ 5. Evaluation Splits (`splits/`)

| File Path | Format | Shape | Size | Description |
|---|---|---|---|---|
| `splits/artist_grouped_5fold.parquet` | Parquet | (10000, 4) | 0.05 MB | 5-Fold GroupKFold by `artist_id` for zero-leakage cross-validation evaluation |
| `splits/temporal_split.parquet` | Parquet | (10000, 4) | 0.05 MB | Chronological split: Train (≤2022, 6,549 songs), Val (2023, 843 songs), Test (2024–2025, 2,608 songs) |

---

## 📋 6. Manifests & Identification

| File Path | Format | Shape / Details | Size | Description |
|---|---|---|---|---|
| `track_ids.npy` | NumPy | (10000,) string | 0.08 MB | Master Spotify track ID array (0 to 9,999) ensuring 1:1 alignment across all files |
| `manifests/extraction_manifest.json` | JSON | Schema specs | 0.02 MB | Full schema, column list, shapes, and extraction metadata |
| `manifests/checksums.json` | JSON | 40 SHA-256 hashes | 0.01 MB | SHA-256 cryptographic checksums for data integrity verification |
| `README.md` | Markdown | Dataset card | 0.01 MB | Dataset card, citations, and quickstart documentation |
| `DATA_DICTIONARY.md` | Markdown | Full dictionary | 0.01 MB | Detailed column-by-column documentation and metric definitions |

---

### Total Package Summary:
- **Total Track Count:** 10,000 songs
- **Total Upload Payload Size:** **349.32 MB**
- **Total Files Tracked:** 40 files
