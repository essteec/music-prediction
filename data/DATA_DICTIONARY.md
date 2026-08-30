# Spotify 10,000 Song Feature Dataset — Data Dictionary

This document details every feature table, embedding matrix, descriptor, and split included in this dataset.

---

## 1. Directory Structure

```
spotify-10k-music-features/
├── metadata/
│   ├── songs.parquet                   (10,000 rows × 32 columns - Spotify track metadata)
│   ├── artists.parquet                 (Artist catalog metadata)
│   └── genres.parquet                  (Genre taxonomy mappings)
│
├── features/
│   ├── audio/
│   │   ├── dsp_librosa.parquet         (10,000 × 91 - Librosa DSP acoustic descriptors)
│   │   └── vad.parquet                 (10,000 × 5 - Silero Vocal Activity Detection)
│   │
│   ├── lyric/
│   │   ├── language_id.parquet         (10,000 × 35 - Language & Script detection flags)
│   │   ├── lyric_stats.parquet         (10,000 × 30 - Lexical richness, VADER, NRC EmoLex, YAKE)
│   │   ├── go_emotions.parquet         (10,000 × 31 - 28 Fine-grained RoBERTa emotion scores)
│   │   ├── bertopic_topics.parquet     (10,000 × 3 - 32 Thematic lyric clusters)
│   │   └── bertopic_topic_labels.json  (Topic names & c-TF-IDF keyword definitions)
│   │
│   ├── metadata/
│   │   └── derived.parquet             (10,000 × 17 - Structural, decade, collaboration features)
│   │
│   └── qc/
│       ├── audio_qc.parquet            (10,000 × 10 - Duration validation metrics)
│       └── chromaprint_fingerprints.parquet (10,000 × 6 - Acoustic fingerprints & duplicate flags)
│
├── embeddings/
│   ├── audio/
│   │   ├── clap_512d.npy               (10000, 512, float32 - LAION-CLAP full-song mean vectors)
│   │   ├── mert_330m_embeddings_1024d.npy (10000, 1024, float32 - MERT-v1-330M full-song mean vectors)
│   │   ├── mert_330m_all_chunks.npz    (74559, 1024, float16 - All consecutive 30s chunk representations)
│   │   ├── mert_embeddings_768d.npy    (10000, 768, float32 - MERT-v1-95M representations)
│   │   ├── panns_embeddings_2048d.npy  (10000, 2048, float32 - PANNs Cnn14 deep embeddings)
│   │   ├── panns_tags_527d.npy         (10000, 527, float32 - PANNs AudioSet tag probabilities)
│   │   ├── panns_tags_labels.json      (527 AudioSet label definitions)
│   │   ├── vggish_embeddings_128d.npy  (10000, 128, float32 - VGGish full-song embeddings)
│   │   └── mel_stats_embeddings_512d.npy (10000, 512, float32 - Mel spectral statistics)
│   │
│   └── lyric/
│       ├── harrier_embeddings_1024d.npy (10000, 1024, float32 - Harrier-OSS-v1-0.6B 32k context embeddings)
│       ├── multilingual_e5_large_1024d.npy (10000, 1024, float32 - Multilingual E5-Large embeddings)
│       └── bge_m3_1024d.npy            (10000, 1024, float32 - BGE-M3 full context 8k embeddings)
│
├── similarity/
│   ├── knn_audio_top100.parquet        (10,000 × 5 - Top-100 nearest neighbors for Optimal Audio Fusion)
│   ├── knn_lyric_top100.parquet        (10,000 × 5 - Top-100 nearest neighbors for Optimal Lyric Fusion)
│   ├── knn_combined_top100.parquet     (10,000 × 5 - Top-100 nearest neighbors for Combined Multimodal)
│   ├── umap_2d_audio.parquet           (10,000 × 4 - 2D Projection coordinates for Audio)
│   ├── umap_2d_lyric.parquet           (10,000 × 4 - 2D Projection coordinates for Lyrics)
│   └── umap_2d_combined.parquet        (10,000 × 4 - 2D Projection coordinates for Multimodal)
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

## 2. Audio Features (`dsp_librosa.parquet`)
- **Energy & Dynamics (7 dims):** `rms_mean`, `rms_std`, `rms_max`, `rms_q10`, `rms_q90`, `crest_factor`, `lufs_integrated` (BS.1770 compliant).
- **Rhythm (5 dims):** `tempo_librosa`, `onset_rate`, `onset_strength_mean`, `onset_strength_std`, `onset_strength_max`.
- **Timbre (58 dims):** `mfcc_1_mean` to `mfcc_20_std` (40 dims), `spectral_centroid_mean/std`, `spectral_bandwidth_mean/std`, `spectral_flatness_mean`, `spectral_rolloff_mean`, `spectral_contrast_mean/std`, `zcr_mean/std`.
- **Harmony (25 dims):** `chroma_0_mean` to `chroma_11_mean`, `chroma_entropy`, `tonnetz_0_mean` to `tonnetz_5_std` (12 dims).
- **Stereo & Dynamics (2 dims):** `stereo_width`, `lr_correlation`.

---

## 3. Language & Script Taxonomy (`language_id.parquet`)
- **Primary / Secondary Codes:** `primary_language` (ISO 639-1), `secondary_language`, `lang_confidence`, `primary_script`.
- **Explicit Language Indicators:** `is_english`, `is_spanish`, `is_hindi_any`, `is_hindi_devanagari`, `is_hindi_romanized`, `is_indonesian`, `is_japanese`, `is_chinese`, `is_dutch`, `is_german`, `is_russian`, `is_italian`, `is_french`, `is_arabic`, `is_portuguese`, `is_turkish`, `is_korean`, `is_tagalog`, `is_scandinavian` (Swedish, Norwegian, Danish, Finnish), `is_punjabi`, `is_tamil`, `is_telugu`, `is_polish`, `is_multilingual`.
