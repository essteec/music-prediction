# Spotify Top-10,000 Feature Quality & Null/Zero Validation Report

> **Audit Date:** 2026-08-25  
> **Total Files Audited:** 19 Parquet feature tables + 13 NumPy embedding matrices (10,000 tracks)  
> **Integrity Status:** ✅ **PASSED (0 NaNs, 0 Infs, 0 Corrupted Matrices)**

---

## 1. Executive Summary Table

| Category | File Path | Shape | NaNs / Infs | All-Zero Rows | All-Zero Columns | Notes & Status |
|---|---|---|---|---|---|---|
| **Audio Features** | `features/audio/dsp_librosa.parquet` | (10000, 91) | **0 / 0** | **0** | **0** | ✅ 100% complete across all 88 Librosa descriptors |
| | `features/audio/vad.parquet` | (10000, 5) | **0 / 0** | **0** | **0** | ✅ 5,637 tracks with vocals detected (>5% ratio) |
| **Lyric Features** | `features/lyric/lyric_stats.parquet` | (10000, 30) | **0 / 0** | **0** | **0** | ✅ Non-zero across all TTR, MTLD, Hapax & NRC emotion scores |
| | `features/lyric/go_emotions.parquet` | (10000, 31) | **0 / 0** | **0** | **0** | ✅ 6,218 English tracks annotated; non-English flagged via `is_english_annotated` |
| | `features/lyric/language_id.parquet` | (10000, 34) | **0 / 0** | **0** | **0** | ✅ 34 language/script flags (Devanagari, Romanized Hindi, Asian, European, etc.) |
| | `features/lyric/bertopic_topics.parquet` | (10000, 3) | **0 / 0** | **0** | **0** | ✅ 32 topic assignments across all tracks |
| **Metadata Features** | `features/metadata/derived.parquet` | (10000, 17) | **0 / 0** | **0** | **0** | ✅ 100% complete across structural features |
| **QC Features** | `features/qc/audio_qc.parquet` | (10000, 10) | **0 / 0** | **0** | 2 (`mismatch_10s/30s`) | ℹ️ 0% audio duration mismatch across all 10k tracks |
| | `features/qc/chromaprint_fingerprints.parquet` | (10000, 6) | **0 / 0** | **0** | **0** | ✅ 100% acoustic fingerprints computed |
| **Audio Embeddings** | `embeddings/audio/clap_512d.npy` | (10000, 512) | **0 / 0** | **0** | **0** | ✅ LAION-CLAP full-song mean vectors |
| | `embeddings/audio/mert_330m_embeddings_1024d.npy` | (10000, 1024) | **0 / 0** | **0** | **0** | ✅ MERT-v1-330M full-song mean representations |
| | `embeddings/audio/mert_330m_all_chunks.npz` | (74559, 1024) | **0 / 0** | **0** | **0** | ✅ CSR-indexed tensor of all consecutive 30s chunk vectors |
| | `embeddings/audio/mert_embeddings_768d.npy` | (10000, 768) | **0 / 0** | **0** | **0** | ✅ MERT-95M representations |
| | `embeddings/audio/panns_embeddings_2048d.npy` | (10000, 2048) | **0 / 0** | **0** | **0** | ✅ PANNs Cnn14 deep embeddings |
| | `embeddings/audio/panns_tags_527d.npy` | (10000, 527) | **0 / 0** | **0** | **0** | ✅ AudioSet probabilities, range `[0.0, 0.9917]` |
| | `embeddings/audio/vggish_embeddings_128d.npy` | (10000, 128) | **0 / 0** | **0** | **0** | ✅ VGGish full-song embeddings |
| | `embeddings/audio/mel_stats_embeddings_512d.npy` | (10000, 512) | **0 / 0** | **0** | **0** | ✅ Mel spectral statistics |
| **Lyric Embeddings** | `embeddings/lyric/harrier_embeddings_1024d.npy` | (10000, 1024) | **0 / 0** | 203 | **0** | ✅ 9,797 lyric tracks encoded; 203 no-lyric tracks = 0 |
| | `embeddings/lyric/multilingual_e5_large_1024d.npy` | (10000, 1024) | **0 / 0** | 203 | **0** | ✅ 9,797 lyric tracks encoded; 203 no-lyric tracks = 0 |
| | `embeddings/lyric/bge_m3_1024d.npy` | (10000, 1024) | **0 / 0** | 203 | **0** | ✅ 9,797 lyric tracks encoded; 203 no-lyric tracks = 0 |
| **Similarity & Splits** | `similarity/knn_audio_top100.parquet` | (10000, 5) | **0 / 0** | **0** | **0** | ✅ Top-100 audio nearest neighbors (CLAP + MERT-330M + VGGish) |
| | `similarity/knn_lyric_top100.parquet` | (10000, 5) | **0 / 0** | **0** | **0** | ✅ Top-100 lyric nearest neighbors (Harrier + E5-Large) |
| | `similarity/knn_combined_top100.parquet` | (10000, 5) | **0 / 0** | **0** | **0** | ✅ Top-100 multimodal nearest neighbors (50% Audio + 50% Lyric) |
| | `similarity/umap_2d_*.parquet` | (10000, 4) | **0 / 0** | **0** | **0** | ✅ 2D coordinates normalized to `[-100, 100]` |
| | `splits/artist_grouped_5fold.parquet` | (10000, 4) | **0 / 0** | **0** | **0** | ✅ Exactly 2,000 tracks per fold (0 leakage) |
| | `splits/temporal_split.parquet` | (10000, 4) | **0 / 0** | **0** | **0** | ✅ 6,549 train, 843 val, 2,608 test |

---

## 2. Detailed Verification of Zero & Null Values

### A. Missing Values in Raw Metadata (`metadata/songs.parquet`)
- **`lyrics` (203 nulls = 2.03%):** 203 songs in the dataset are instrumental or lacked Genius lyric submissions.
- **Spotify API Audio Features (31 nulls = 0.31%):** 31 songs were missing raw Spotify Web API audio feature endpoints (`danceability`, `energy`, `valence`, etc.).
- **Impact on Extracted Features:** **None.** Our direct audio extraction (`dsp_librosa.parquet`, `clap_512d.npy`, `panns_tags_527d.npy`, `vad.parquet`) was performed directly on the 10,000 Opus audio files and has **0 nulls and 100% coverage**.

### B. Expected Zero Rows in Lyric Embeddings
- `bge_m3_1024d.npy` and `multilingual_e5_large_1024d.npy` have **exactly 203 all-zero rows** matching the 203 tracks with no lyrics.
- All 9,797 songs with lyrics have non-zero, normalized dense embeddings.

### C. Zero Values in Emotion Features (`go_emotions.parquet`)
- 6,218 English tracks have fine-grained RoBERTa emotion probability scores.
- Non-English songs have `0.0` with `is_english_annotated = False`.

### D. Audio QC Zeros
- `duration_mismatch_10s` (0) and `duration_mismatch_30s` (0) are all False because all 10,000 local Opus files match the Spotify duration catalog.
