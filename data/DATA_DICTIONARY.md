# Spotify 10,000 Song Feature Dataset — Data Dictionary

This document details every feature table, embedding matrix, similarity graph, and evaluation split included in this dataset.

---

## 1. Directory Structure

```
spotify-10k-music-features/
├── metadata/
│   ├── songs.parquet                   (10,000 rows × 32 columns - Spotify track metadata)
│   ├── artists.parquet                 (Artist catalog metadata)
│   └── genres.parquet                  (Genre taxonomy mappings)
│
├──                                     (CSV format for in-browser Kaggle table previews)
│   ├── songs.csv                       (10,000 rows × 32 columns)
│   ├── artists.csv                     (5,015 rows × 6 columns)
│   └── genres.csv                      (1,276 rows × 2 columns)
│
├── features/
│   ├── audio/
│   │   ├── dsp_librosa.parquet         (10,000 × 91 - Librosa DSP acoustic descriptors)
│   │   └── vad.parquet                 (10,000 × 5 - Silero Vocal Activity Detection)
│   │
│   ├── lyric/
│   │   ├── language_id.parquet         (10,000 × 34 - Language & Script detection flags)
│   │   ├── lyric_stats.parquet         (10,000 × 30 - Lexical richness, VADER, NRC EmoLex, YAKE)
│   │   ├── go_emotions.parquet         (10,000 × 31 - 28 Fine-grained RoBERTa emotion scores)
│   │   ├── bertopic_topics.parquet     (10,000 × 3 - 32 Thematic lyric clusters)
│   │   └── bertopic_topic_labels.json  (Topic names & c-TF-IDF keyword definitions)
│   │
│   ├── metadata/
│   │   └── derived.parquet             (10,000 × 17 - Structural, decade, collaboration features)
│   │
│   └── qc/
│       └── chromaprint_fingerprints.parquet (10,000 × 6 - AcoustID Chromaprint raw fingerprints & duplicate flags)
│
├── embeddings/
│   ├── audio/
│   │   ├── clap_512d.npy               (10000, 512, float32 - LAION-CLAP full-song mean vectors)
│   │   ├── mert_330m_embeddings_1024d.npy (10000, 1024, float32 - MERT-v1-330M full-song mean vectors)
│   │   ├── mert_330m_all_chunks.npz    (74559, 1024, float16 - All consecutive 30s chunk representations)
│   │   ├── panns_embeddings_2048d.npy  (10000, 2048, float32 - PANNs Cnn14 deep embeddings)
│   │   ├── panns_tags_527d.npy         (10000, 527, float32 - PANNs AudioSet tag probabilities)
│   │   ├── panns_tags_labels.json      (527 AudioSet label definitions)
│   │   ├── vggish_embeddings_128d.npy  (10000, 128, float32 - VGGish full-song embeddings)
│   │   └── mel_stats_embeddings_512d.npy (10000, 512, float32 - Mel spectral statistics)
│   │
│   ├── lyric/
│   │   ├── harrier_embeddings_1024d.npy (10000, 1024, float32 - Harrier-OSS-v1-0.6B 32k context embeddings)
│   │   ├── multilingual_e5_large_1024d.npy (10000, 1024, float32 - Multilingual E5-Large embeddings)
│   │   └── bge_m3_1024d.npy            (10000, 1024, float32 - BGE-M3 full context 8k embeddings)
│   │
│   └── metadata/
│       ├── spotify_audio_11d.npy       (10000, 11, float32 - Normalized Spotify audio & vibe descriptors)
│       ├── emotion_sentiment_36d.npy   (10000, 36, float32 - 28 GoEmotions + 8 NRC emotion densities)
│       ├── vocal_dsp_12d.npy           (10000, 12, float32 - Vocal ratio + 10 Librosa dynamics)
│       ├── lyric_stats_12d.npy         (10000, 12, float32 - TTR, readability, VADER, structure)
│       ├── genre_hybrid_50d.npy        (10000, 50, float32 - 17-D Main + 17-D Sub Rollup + 16-D Latent SVD)
│       ├── language_27d.npy            (10000, 27, float32 - 26 Language one-hots + confidence)
│       ├── temporal_collab_10d.npy     (10000, 10, float32 - Release decade, collaboration, followers)
│       └── bertopic_32d.npy            (10000, 32, float32 - 32-D Topic cluster one-hot vectors - Archival)
│
├── similarity/
│   ├── knn_audio_top100.parquet        (10,000 × 5 - Top-100 nearest neighbors for Acoustic Sound: 1664-D)
│   ├── knn_lyric_top100.parquet        (10,000 × 5 - Top-100 nearest neighbors for Lyric Storytelling: 2048-D)
│   ├── knn_mood_top100.parquet         (10,000 × 5 - Top-100 nearest neighbors for Mood & Emotion: 59-D)
│   ├── knn_combined_top100.parquet     (10,000 × 5 - Top-100 nearest neighbors for Master Multimodal: 3795-D)
│   ├── umap_2d_audio.parquet           (10,000 × 4 - 2D Visualization coordinates for Acoustic Space)
│   ├── umap_2d_lyric.parquet           (10,000 × 4 - 2D Visualization coordinates for Lyric Space)
│   ├── umap_2d_mood.parquet            (10,000 × 4 - 2D Visualization coordinates for Mood Space)
│   └── umap_2d_combined.parquet        (10,000 × 4 - 2D Visualization coordinates for Multimodal Space)
│
├── splits/
│   ├── artist_grouped_5fold.parquet    (10,000 × 4 - 5-fold cross-validation grouped by artist)
│   └── temporal_split.parquet          (10,000 × 4 - Train / Val / Test chronological split)
│
├── manifests/
│   ├── extraction_manifest.json        (Schema, shapes, dtypes, and extraction metadata)
│   └── checksums.json                  (SHA-256 integrity checksums for all files)
│
└── track_ids.npy                       (10,000 Spotify Track IDs master alignment array)
```

---

## 2. Similarity Graphs & 2D Manifolds (`similarity/`)

### A. Pre-Computed Top-100 kNN Matrices
- **`knn_audio_top100.parquet` (1664-D)**: Fused `CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D)`.
- **`knn_lyric_top100.parquet` (2048-D)**: Fused `Harrier-OSS-v1-0.6B (1024-D) + Multilingual E5-Large (1024-D)`.
- **`knn_mood_top100.parquet` (59-D)**: Fused `Spotify Audio (11-D) + GoEmotions/NRC (36-D) + Vocal DSP (12-D)`.
- **`knn_combined_top100.parquet` (3795-D)**: Master fusion of Audio (1664-D), Lyric (2048-D), Spotify Audio (11-D), Vocal DSP (12-D), Genre Hybrid (50-D: 17 Main + 17 Sub Rollup + 16 Latent SVD), and Temporal Context (10-D).

### B. 2D UMAP Visualization Projections
- **`umap_2d_audio.parquet`**, **`umap_2d_lyric.parquet`**, **`umap_2d_mood.parquet`**, **`umap_2d_combined.parquet`**:
  
> **Interpretation Disclaimer**: The 2D coordinates in `similarity/umap_2d_*.parquet` are qualitative non-linear dimensionality reduction projections intended for visual exploration, genre cluster maps, and playlist trajectory plotting. For quantitative metric distance or true mathematical similarity, always use the high-dimensional embeddings or the Top-100 kNN graph tables.

---

## 3. Audio Features (`dsp_librosa.parquet`)
- **Energy & Dynamics (7 dims):** `rms_mean`, `rms_std`, `rms_max`, `rms_q10`, `rms_q90`, `crest_factor`, `lufs_integrated` (BS.1770 compliant).
- **Rhythm (5 dims):** `tempo_librosa`, `onset_rate`, `onset_strength_mean`, `onset_strength_std`, `onset_strength_max`.
- **Timbre (58 dims):** `mfcc_1_mean` to `mfcc_20_std` (40 dims), `spectral_centroid_mean/std`, `spectral_bandwidth_mean/std`, `spectral_flatness_mean`, `spectral_rolloff_mean`, `spectral_contrast_mean/std`, `zcr_mean/std`.
- **Harmony (25 dims):** `chroma_0_mean` to `chroma_11_mean`, `chroma_entropy`, `tonnetz_0_mean` to `tonnetz_5_std` (12 dims).
- **Stereo & Dynamics (2 dims):** `stereo_width`, `lr_correlation`.

---

## 4. Language & Script Taxonomy (`language_id.parquet`)
- **Primary / Secondary Codes:** `primary_language` (ISO 639-1), `secondary_language`, `lang_confidence`, `primary_script`.
- **Explicit Language Indicators:** `is_english`, `is_spanish`, `is_hindi_any`, `is_hindi_devanagari`, `is_hindi_romanized`, `is_indonesian`, `is_japanese`, `is_chinese`, `is_dutch`, `is_german`, `is_russian`, `is_italian`, `is_french`, `is_arabic`, `is_portuguese`, `is_turkish`, `is_korean`, `is_tagalog`, `is_scandinavian` (Swedish, Norwegian, Danish, Finnish), `is_punjabi`, `is_tamil`, `is_telugu`, `is_polish`, `is_multilingual`.
