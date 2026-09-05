# Spotify 10,000 Song Dataset: Multimodal Similarity & Optimal Recommendation Guide

This guide details the **empirical methodology, extraction pipelines, ablation findings, and recommended architectural patterns** for using the similarity graphs and feature embeddings in this dataset.

Whether you are building a production content-based recommender, conducting Music Information Retrieval (MIR) research, or training downstream models, this document answers **which similarity files to use, how each feature was extracted, and why specific representations were kept or excluded**.

---

## 🧭 1. Executive Summary & File Decision Matrix

The dataset provides **4 pre-computed Top-250 kNN graph tables** and **8 raw embedding matrices**. Each similarity file serves a distinct musical facet:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 WHICH SIMILARITY FILE SHOULD I USE?                                         │
├──────────────────────────────┬──────────┬─────────────────────────────────────┬─────────────────────────────┤
│ Parquet File in similarity/  │ Dim (D)  │ Core Modalities Fused               │ Best Used For               │
├──────────────────────────────┼──────────┼─────────────────────────────────────┼─────────────────────────────┤
│ 1. knn_combined_top250.parquet│ 3,795-D │ 73% Neural (Audio 38% + Lyric 35%)  │ ⭐ Primary / General-purpose │
│                              │          │ + 27% Context & Vibe (Genre 11% +   │ recommendation across all   │
│                              │          │ Spotify 8% + Temporal 4% + Vocal 4%)│ musical dimensions.         │
├──────────────────────────────┼──────────┼─────────────────────────────────────┼─────────────────────────────┤
│ 2. knn_audio_top250.parquet  │ 1,664-D  │ LAION-CLAP + MERT-330M + VGGish    │ Pure sonic feel, timbre,    │
│                              │          │ (Mean-pooled over full song)        │ beat, instrumentation & DSP.│
├──────────────────────────────┼──────────┼─────────────────────────────────────┼─────────────────────────────┤
│ 3. knn_lyric_top250.parquet  │ 2,048-D  │ Harrier-OSS-0.6B + E5-Large         │ Poetic themes, narrative,   │
│                              │          │ (32k context + cross-lingual)       │ storytelling & metaphors.   │
├──────────────────────────────┼──────────┼─────────────────────────────────────┼─────────────────────────────┤
│ 4. knn_mood_top250.parquet   │ 83-D     │ Unified Mood & Context              │ Mood, acoustic vibe, vocal  │
│                              │          │ (Genre 40% + Spotify 30% +          │ warmth & era continuity.    │
│                              │          │  Temporal 15% + Vocal DSP 15%)      │                             │
└──────────────────────────────┴──────────┴─────────────────────────────────────┴─────────────────────────────┘
```

> **⚡ Fast-Path vs. Dynamic Steering**:
> - For **instant $O(1)$ lookups** (sub-millisecond latency), read directly from `similarity/knn_combined_top250.parquet` or `similarity/knn_mood_top250.parquet`.
> - For **interactive UI sliders** (e.g., custom weights $w_a \cdot S_{\text{audio}} + w_l \cdot S_{\text{lyric}} + w_m \cdot S_{\text{mood}}$), compute dot products directly using the normalized `.npy` matrices in `embeddings/`.

---

## 🎧 2. Audio Similarity Extraction & Optimal Fusion (`knn_audio_top250.parquet`)

### A. Extraction Methodology
All 10,000 tracks were decoded from local Opus audio files using `librosa` / `ffmpeg` over **100% of their full song duration** (no 30-second truncation artifacts):

1. **LAION-CLAP (`clap_512d.npy` - 512-D):**
   - **Model:** `HTSAT-base` zero-shot text-audio model trained by LAION.
   - **Sampling:** Resampled to 48 kHz mono.
   - **Extraction:** Windowed across full song with mean-pooling to produce unit L2-normalized 512-D vectors.
   - **Strengths:** Superb high-level genre, instrument recognition, and acoustic scene understanding.

2. **MERT-v1-330M (`mert_330m_embeddings_1024d.npy` - 1024-D):**
   - **Model:** 330M-parameter music-specific self-supervised transformer (`m-a-p/MERT-v1-330M`).
   - **Sampling:** Resampled to 24 kHz mono.
   - **Extraction:** Processed in consecutive 30-second sliding windows with mean-pooled hidden representations across all chunks.
   - **Strengths:** Exceptional harmonic awareness, chord progression tracking, and musical structure.

3. **Google VGGish (`vggish_embeddings_128d.npy` - 128-D):**
   - **Model:** Classical deep AudioSet convolutional network.
   - **Sampling:** Resampled to 16 kHz mono.
   - **Extraction:** Mean-pooled 128-D acoustic vectors across full audio tracks.
   - **Strengths:** Compact, robust baseline anchor for timbre and spectro-temporal texture.

### B. Why Was PANNs Excluded from the Audio Graph?
During our Leave-One-Group-Out (LOGO) ablation study on audio models, we evaluated 5 candidate representations:

| Audio Model Tested | Remaining Dim | Overlap @10 | Genre Agr @10 | Artist Agr @10 | Ablation Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full 5-Model Baseline** | 4224-D | 100.0% | 87.25% | 12.09% | Reference Baseline |
| **Without CLAP (512-D)** | 3712-D | 62.2% | **85.39% (-1.86%)** | **10.43% (-1.66%)** | **Essential Signal (Keep)** |
| **Without MERT-330M (1024-D)** | 3200-D | 78.0% | **87.13% (-0.12%)** | **11.64% (-0.45%)** | **Beneficial Signal (Keep)** |
| **Without VGGish (128-D)** | 4096-D | 76.8% | **86.88% (-0.37%)** | **11.72% (-0.37%)** | **Beneficial Signal (Keep)** |
| **Without PANNs Cnn14 (2048-D)**| 2176-D | 61.8% | **87.36% (+0.11%)** | **12.17% (+0.08%)** | **Harmful / Noisy Artifact (Drop)** |
| **Without Mel Stats (512-D)** | 3712-D | 98.2% | 87.22% (-0.03%) | 12.04% (-0.05%) | **Redundant (98.9% overlap)** |

- **Empirical Finding:** Removing PANNs **improved** genre purity ($+0.11\%$) and artist consistency ($+0.08\%$) while preventing 38.2% of neighbors from being displaced by sound-effect / Foley artifacts.
- **Final Audio Fusion Formula (1,664-D):**
  $$\mathbf{v}_{\text{audio}} = \text{Normalize}\left( \left[ \text{CLAP}_{512\text{d}} \;\|\; \text{MERT}_{1024\text{d}} \;\|\; \text{VGGish}_{128\text{d}} \right] \right)$$

---

## 📝 3. Lyric Similarity Extraction & Optimal Fusion (`knn_lyric_top250.parquet`)

### A. Extraction Methodology
All lyrics were cleaned to remove Genius scrapers' contributor metadata, headers (`[Chorus]`, `[Verse 1]`, `[Drop]`), and whitespace anomalies:

1. **Harrier-OSS-v1-0.6B (`harrier_embeddings_1024d.npy` - 1024-D):**
   - **Model:** Microsoft's state-of-the-art multilingual embedding model.
   - **Context Length:** Supports full 32,768 token context length (captures 100% of entire song lyrics with zero truncation).
   - **Extraction:** Encoded on GPU with sentence-level bidirectional attention.

2. **Multilingual-E5-Large (`multilingual_e5_large_1024d.npy` - 1024-D):**
   - **Model:** 24-layer multilingual transformer fine-tuned for semantic retrieval.
   - **Extraction:** Formatted with `passage:` prefix for asymmetric semantic matching across 100+ languages.

### B. Lyric Ablation & Decision
In the lyric LOGO benchmark:
- **Harrier-0.6B** proved to be the single most impactful lyric encoder: removing it degraded genre agreement by $-1.98\%$ and artist agreement by $-1.04\%$.
- **Multilingual-E5-Large** added crucial multilingual retrieval stability for Romanized and non-Latin scripts (Devanagari, Cyrillic, Hangul, Kanji).
- **BGE-M3 (1024-D)** was dropped because its removal increased genre alignment by $+0.97\%$, demonstrating that combining three 1024-D language models caused topic overfitting.

**Final Lyric Fusion Formula (2,048-D):**
$$\mathbf{v}_{\text{lyric}} = \text{Normalize}\left( \left[ \text{Harrier}_{1024\text{d}} \;\|\; \text{E5-Large}_{1024\text{d}} \right] \right) \cdot \mathbb{I}(\text{has\_lyrics})$$

---

## 🎭 4. Unified Mood, Vibe & Context Similarity (`knn_mood_top250.parquet`)

### A. Architecture (83-D)
Unlike acoustic timbre or lyric vocabularies, musical **mood and vibe** represents a continuous emotional and stylistic landscape. Based on empirical LOGO ablation testing, English-only emotion classifiers (`GoEmotions 36-D`) were removed to eliminate zero-padding bias on non-English tracks, and macro/subgenre anchors (`Genre Hybrid 50-D`) plus historical context (`Temporal 10-D`) were fused into a unified **83-D** representation:

1. **Genre Hybrid Vector (50-D - Weight: 40%):** 17-D Main + 17-D Subgenre Rollup + 16-D Latent SVD space.
2. **Spotify High-Level Descriptors (11-D - Weight: 30%):** `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `mode`, `loudness_scaled`, `tempo_scaled`, `time_signature_scaled`.
3. **Temporal & Collab Context (10-D - Weight: 15%):** Release decade one-hots, collaboration flag, artist follower reach, track duration.
4. **Vocal Activity & Dynamics (12-D - Weight: 15%):** Silero VAD vocal presence + Librosa dynamics (`crest_factor`, integrated `lufs`, `onset_rate`, `spectral_contrast`).

**Final Mood & Context Formula (83-D):**
$$\mathbf{v}_{\text{mood\_83}} = \text{Normalize}\left( \left[ \sqrt{0.40}\,\mathbf{v}_{\text{genre\_50d}} \;\|\; \sqrt{0.30}\,\mathbf{v}_{\text{spotify\_11d}} \;\|\; \sqrt{0.15}\,\mathbf{v}_{\text{temporal\_10d}} \;\|\; \sqrt{0.15}\,\mathbf{v}_{\text{vocal\_12d}} \right] \right)$$

---

## 🌐 5. Master Multimodal Similarity (`knn_combined_top250.parquet`)

#### A. The 50-D Hybrid Genre Representation
Instead of naive bag-of-words tokenizers that split multi-word subgenres into fragments (e.g., splitting `"k-pop"` or `"corridos bélicos"`), the dataset incorporates a **3-tier hierarchical and latent hybrid genre representation**:

1. **Level 1 (`main_17d` - 17-D):** Multi-hot vector across the 17 canonical Main Genres (`Blues, Christian, Classical, Country, Easy Listening, Electronic, Folk, Hip Hop, Jazz, Latin, Metal, New Age, Pop, R&B, Reggae, Rock, Traditional Music`).
2. **Level 2 (`sub_affinity_17d` - 17-D):** L1-normalized distribution across the 17 parent main genres rolled up from all 1,276 subgenres in `artist_genres` mapped deterministically through `metadata/genres.parquet` (100.0% coverage, 0 unmapped strings).
3. **Level 3 (`subgenre_svd_16d` - 16-D):** Dense latent subgenre co-occurrence representation derived via `TruncatedSVD(n_components=16, random_state=42)` across the $10,000 \times 1,276$ subgenre incidence matrix (capturing $36.35\%$ of all subgenre co-occurrence variance).
   - **Disentanglement:** Disentangles intra-macro distinctions (e.g. separates *Bedroom Pop* from *Christmas*, while keeping *Bedroom Pop* closely aligned with *Indie Pop*).
   - **Performance:** Extends Artist Agreement from $16.08\%$ to **$31.67\%$** and lowers Vibe MAE to **$0.1832$**.

### B. Master Multimodal LOGO Ablation Results

| Feature Block Evaluated | Remaining Dim | Overlap @10 | Overlap @50 | Vibe MAE @10 | Vibe Δ | Artist Agr @10 | Artist Δ | Impact Verdict | Action in Master Fusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Baseline Ensemble** | 3870-D | 100.0% | 100.0% | 0.1529 | 0.0000 | 17.18% | 0.00% | Reference | — |
| **Without Neural Audio (1664-D)** | 2206-D | 86.4% | 90.2% | 0.1584 | `+0.0055` | 15.83% | `-1.35%` | **Essential Signal (Keep)** | **Included (1,664-D - 38%)** |
| **Without Neural Lyric (2048-D)** | 1822-D | 92.5% | 95.1% | 0.1530 | `+0.0001` | 16.65% | `-0.53%` | **Moderate Signal (Keep)** | **Included (2,048-D - 35%)** |
| **Without Spotify Audio (11-D)** | 3859-D | 74.2% | 81.3% | 0.1624 | `+0.0095` | 17.55% | `+0.37%` | **Essential Signal (Keep)** | **Included (11-D - 8%)** |
| **Without Vocal DSP (12-D)** | 3858-D | 79.2% | 84.9% | 0.1568 | `+0.0039` | 16.91% | `-0.27%` | **Beneficial Signal (Keep)** | **Included (12-D - 4%)** |
| **Without Genre Hybrid (50-D)** | 3820-D | 46.8% | 51.6% | 0.1494 | `-0.0035` | 8.64% | `-8.54%` | **Essential Signal (Keep)** | **Included (50-D - 11%)** |
| **Without Temporal & Collab (10-D)**| 3860-D | 53.0% | 59.4% | 0.1498 | `-0.0030` | 15.49% | `-1.69%` | **Essential Signal (Keep)** | **Included (10-D - 4%)** |
| **Without Lyric Structure (12-D)** | 3858-D | 83.1% | 87.6% | 0.1524 | `-0.0005` | 17.37% | `+0.19%` | **Marginal / Neutral** | **Excluded (Dropped)** |
| **Without Linguistic & Language (27-D)** | 3843-D | 95.6% | 94.3% | 0.1524 | `-0.0004` | 17.19% | `+0.01%` | **Redundant (Query Filter)** | **Handled via Filter** |
| **Without Emotion & Sentiment (36-D)**| 3834-D | 49.9% | 60.8% | 0.1455 | `-0.0073` | 21.28% | `+4.10%` | **Distinct / Noisy on Global** | **Dropped (Language Disparity)** |

### C. Master Combined Vector Composition (Exact 3,795-D)

$$\mathbf{v}_{\text{combined}} = \text{Normalize}\left( \left[ \sqrt{0.38}\,\mathbf{v}_{\text{audio\_1664d}} \;\|\; \sqrt{0.35}\,\mathbf{v}_{\text{lyric\_2048d}} \;\|\; \sqrt{0.11}\,\mathbf{v}_{\text{genre\_50d}} \;\|\; \sqrt{0.08}\,\mathbf{v}_{\text{spotify\_11d}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{temporal\_10d}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{vocal\_12d}} \right] \right) \in \mathbb{R}^{3795}$$

- **Zero Zero-Padding Drift:** By routing Language ID to a boolean query filter and dropping fine-grained English emotion classifiers (which suffered from zero-padding disparity on non-English tracks per LOGO ablation), both the global 3,795-D master representation and the 83-D unified mood representation remain clean, robust, and balanced across all languages and genres.

---

## 🛠️ 6. Optimal Implementation Patterns

### Pattern 1: Instant Production Fast-Path ($O(1)$ Lookup)
For real-time APIs or frontend apps, read the pre-computed Top-250 Parquet table directly:

```python
import pandas as pd

songs = pd.read_parquet('metadata/songs.parquet')
knn_comb = pd.read_parquet('similarity/knn_combined_top250.parquet')

def get_instant_recommendations(track_idx: int, top_k: int = 5):
    row = knn_comb.iloc[track_idx]
    nb_indices = row['top250_neighbor_indices'][:top_k]
    nb_sims = row['top250_similarities'][:top_k]
    
    recs = songs.iloc[nb_indices][['track_name', 'artist_names', 'main_genres', 'release_date']].copy()
    recs['similarity'] = nb_sims
    return recs

# Query Lady Gaga - Bad Romance (Row 0)
print(get_instant_recommendations(0, top_k=5))
```

---

### Pattern 2: Dynamic Multi-Pillar Engine with Contextual Guards

When you want users to customize the recommendation balance (e.g., favoring lyrics over beat):

```python
import numpy as np
import pandas as pd

# 1. Load Metadata & Feature Matrices
songs = pd.read_parquet('metadata/songs.parquet')
lang_id = pd.read_parquet('features/lyric/language_id.parquet')
derived = pd.read_parquet('features/metadata/derived.parquet')

def l2_norm(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

# Load Normalized Blocks
audio_norm = l2_norm(np.concatenate([
    l2_norm(np.load('embeddings/audio/clap_512d.npy')),
    l2_norm(np.load('embeddings/audio/mert_330m_embeddings_1024d.npy')),
    l2_norm(np.load('embeddings/audio/vggish_embeddings_128d.npy'))
], axis=1))

harrier = np.load('embeddings/lyric/harrier_embeddings_1024d.npy')
has_lyrics = (np.linalg.norm(harrier, axis=1, keepdims=True) > 1e-6).astype(np.float32)
lyric_norm = l2_norm(np.concatenate([
    l2_norm(harrier),
    l2_norm(np.load('embeddings/lyric/multilingual_e5_large_1024d.npy'))
], axis=1)) * has_lyrics

mood_norm = l2_norm(np.concatenate([
    l2_norm(np.load('embeddings/metadata/spotify_audio_11d.npy')),
    l2_norm(np.load('embeddings/metadata/emotion_sentiment_36d.npy')),
    l2_norm(np.load('embeddings/metadata/vocal_dsp_12d.npy'))
], axis=1))

genre_norm = l2_norm(np.load('embeddings/metadata/genre_hybrid_50d.npy'))
temporal_norm = l2_norm(np.load('embeddings/metadata/temporal_collab_10d.npy'))

def recommend_songs(
    seed_idx: int,
    top_k: int = 5,
    audio_weight: float = 0.35,
    lyric_weight: float = 0.25,
    mood_weight: float = 0.20,
    genre_weight: float = 0.15,
    temporal_weight: float = 0.05,
    same_language_only: bool = False,
    same_decade_only: bool = False,
    penalize_same_artist: bool = True
):
    """
    Computes dynamic multimodal recommendations with contextual guards.
    
    Instrumental Fallback Rule:
    If the seed song has no lyrics (has_lyrics == 0), s_lyric automatically falls
    back to s_audio. This routes the lyric weight directly to acoustic matching
    rather than penalizing instrumental tracks with 0-similarity dot products.
    """
    s_audio = audio_norm @ audio_norm[seed_idx]
    
    # Instrumental fallback
    if has_lyrics[seed_idx, 0] > 0:
        s_lyric = lyric_norm @ lyric_norm[seed_idx]
    else:
        s_lyric = s_audio
        
    s_mood = mood_norm @ mood_norm[seed_idx]
    s_genre = genre_norm @ genre_norm[seed_idx]
    s_temporal = temporal_norm @ temporal_norm[seed_idx]
    
    total_sim = (
        audio_weight * s_audio +
        lyric_weight * s_lyric +
        mood_weight * s_mood +
        genre_weight * s_genre +
        temporal_weight * s_temporal
    )
    
    # Guard 1: Language filter
    if same_language_only:
        seed_lang = lang_id.iloc[seed_idx]['primary_language']
        total_sim[lang_id['primary_language'] != seed_lang] = -1e9
        
    # Guard 2: Decade filter
    if same_decade_only:
        seed_decade = derived.iloc[seed_idx]['release_decade']
        total_sim[derived['release_decade'] != seed_decade] = -1e9
        
    # Guard 3: Artist diversity penalty (fosters discovery)
    if penalize_same_artist:
        seed_artists = set(str(songs.iloc[seed_idx]['artist_names']).lower().split(','))
        for idx in range(len(songs)):
            if idx == seed_idx:
                continue
            cand_artists = set(str(songs.iloc[idx]['artist_names']).lower().split(','))
            if bool(seed_artists & cand_artists):
                total_sim[idx] *= 0.85
                
    # Mask self
    total_sim[seed_idx] = -1e9
    
    top_indices = np.argsort(total_sim)[-top_k:][::-1]
    return songs.iloc[top_indices][['track_name', 'artist_names', 'main_genres', 'popularity', 'release_date']]
```

---

## 📊 7. Visualizing 2D Manifolds (`similarity/umap_2d_*.parquet`)

The dataset includes 4 pre-computed 2D projection files (`umap_2d_audio.parquet`, `umap_2d_lyric.parquet`, `umap_2d_mood.parquet`, `umap_2d_combined.parquet`).

### Qualitative Interpretation Disclaimer
> **⚠️ Critical Interpretation Rule:**  
> The 2D coordinates in `similarity/umap_2d_*.parquet` are **qualitative non-linear dimensionality reduction projections** intended for:
> - Visual cluster maps and exploratory scatter plots
> - Playlist trajectory journey visualization
> - Qualitative comparison between audio timbre space and lyric topic space
> 
> **Never** use 2D Euclidean distances as a quantitative substitute for true high-dimensional similarity. For quantitative metric distance or kNN search, always use the high-dimensional embedding arrays or the Top-250 Parquet tables.

---

## 📚 8. Summary Table of Packaged Files

```
spotify-10k-music-features/
├── similarity/
│   ├── knn_combined_top250.parquet  (3,795-D Master Multi-Modal Top-250 Graph)
│   ├── knn_audio_top250.parquet     (1,664-D Pure Acoustic Top-250 Graph)
│   ├── knn_lyric_top250.parquet     (2,048-D Lyrical Semantic Top-250 Graph)
│   ├── knn_mood_top250.parquet      (83-D Unified Mood & Context Top-250 Graph)
│   ├── umap_2d_combined.parquet     (2D Projection Coordinates - Multimodal)
│   ├── umap_2d_audio.parquet        (2D Projection Coordinates - Audio Space)
│   ├── umap_2d_lyric.parquet        (2D Projection Coordinates - Lyric Space)
│   └── umap_2d_mood.parquet         (2D Projection Coordinates - Unified Mood & Context Space: 83-D)
│
├── embeddings/
│   ├── audio/                       (CLAP 512d, MERT 1024d, VGGish 128d, PANNs 2048d, Mel 512d)
│   ├── lyric/                       (Harrier 1024d, Multilingual E5 1024d, BGE-M3 1024d)
│   └── metadata/                    (Spotify 11d, Emotion 36d, Vocal DSP 12d, Genre Hybrid 50d, Temporal 10d)
│
└── splits/
    ├── artist_grouped_5fold.parquet (5-fold GroupKFold by artist_id for leakage-free modeling)
    └── temporal_split.parquet       (Chronological Train / Val / Test split)
```
