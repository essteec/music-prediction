Here is the exact, comprehensive breakdown of all 4 pre-computed Top-250 kNN similarity files in `data/similarity/`, including their **underlying models, source files, dimensionality, mathematical fusion, and computation algorithms**.

---

```
                               ┌────────────────────────────────────────────────────────────┐
                               │           THE 4 PRE-COMPUTED TOP-250 SIMILARITY FILES     │
                               └─────────────────────────────┬──────────────────────────────┘
                                                             │
        ┌────────────────────────────┬───────────────────────┴───────────────────────┬────────────────────────────┐
        │                            │                                               │                            │
┌───────▼────────────────────┐ ┌─────▼──────────────────────┐ ┌──────────────────────▼──────┐ ┌───────────────────▼─────────┐
│ 1. knn_audio_top250.parquet│ │ 2. knn_lyric_top250.parquet│ │ 3. knn_mood_top250.parquet   │ │ 4. knn_combined_top250.parquet│
│   (Pure Acoustic Sound)    │ │   (Lyrical Storytelling)   │ │  (Unified Mood & Context)   │ │   (Master Multimodal Fusion) │
│          1,664-D           │ │          2,048-D           │ │             83-D             │ │            3,795-D           │
└────────────────────────────┘ └────────────────────────────┘ └──────────────────────────────┘ └───────────────────────────┘
```

---

## 1. Pure Acoustic Sound: `knn_audio_top250.parquet` (1,664-D)

* **Purpose:** Matches songs strictly by acoustic timbre, production textures, beat patterns, and instrumentation (ignoring lyrics and genre labels).
* **Output Format:** Parquet table `(10,000 × 5)` (~22.03 MB).

### Models & Feature Breakdown:
| Sub-Component | Dimension | Underlying Model & Architecture | Source Embedding File |
| :--- | :--- | :--- | :--- |
| **LAION-CLAP** | **512-D** | `HTSAT-base` audio encoder (trained on LAION-Audio-630k). Full song resampled to 48 kHz mono, chunked and mean-pooled. | `data/embeddings/audio/clap_512d.npy` |
| **MERT-v1-330M**| **1,024-D** | 330M-parameter music self-supervised transformer (16 attention heads, 24 layers). Full song resampled to 24 kHz mono, mean-pooled across all 30s sliding windows. | `data/embeddings/audio/mert_330m_embeddings_1024d.npy` |
| **Google VGGish**| **128-D** | Deep acoustic CNN trained on 8M YouTube audio clips (AudioSet). Full song resampled to 16 kHz mono, mean-pooled over 0.96s frames. | `data/embeddings/audio/vggish_embeddings_128d.npy` |

### How It Is Computed:
1. Each of the 3 source arrays is individually $L_2$-normalized across rows:
   $$\mathbf{v}_{\text{clap\_norm}} = \frac{\mathbf{v}_{\text{clap}}}{\|\mathbf{v}_{\text{clap}}\|_2}, \quad \mathbf{v}_{\text{mert\_norm}} = \frac{\mathbf{v}_{\text{mert}}}{\|\mathbf{v}_{\text{mert}}\|_2}, \quad \mathbf{v}_{\text{vgg\_norm}} = \frac{\mathbf{v}_{\text{vgg}}}{\|\mathbf{v}_{\text{vgg}}\|_2}$$
2. They are horizontally concatenated and re-normalized to form the 1,664-D acoustic vector:
   $$\mathbf{v}_{\text{audio\_1664}} = \text{Normalize}\Big(\big[\, \mathbf{v}_{\text{clap\_norm}} \;\|\; \mathbf{v}_{\text{mert\_norm}} \;\|\; \mathbf{v}_{\text{vgg\_norm}} \,\big]\Big) \in \mathbb{R}^{1664}$$
3. Cosine similarity is computed across all $10,000 \times 10,000$ pairs on GPU. Self-similarity is masked to $-\infty$, and the top 250 highest neighbors are stored per song.

---

## 2. Lyrical Storytelling: `knn_lyric_top250.parquet` (2,048-D)

* **Purpose:** Matches songs purely by narrative themes, poetic style, metaphors, and sentiment—completely language-agnostic.
* **Output Format:** Parquet table `(10,000 × 5)` (~21.64 MB).

### Models & Feature Breakdown:
| Sub-Component | Dimension | Underlying Model & Architecture | Source Embedding File |
| :--- | :--- | :--- | :--- |
| **Harrier-OSS-v1-0.6B** | **1,024-D** | Microsoft state-of-the-art multilingual LLM embedding model with a **32,768-token context window**. Encodes entire song lyrics without truncation. | `data/embeddings/lyric/harrier_embeddings_1024d.npy` |
| **Multilingual-E5-Large** | **1,024-D** | 560M-parameter multilingual retriever (InfoNCE trained on 100+ languages). Winner of empirical benchmark (nDCG@10: 0.3001). | `data/embeddings/lyric/multilingual_e5_large_1024d.npy` |

### How It Is Computed:
1. Both 1,024-D embeddings are individually $L_2$-normalized.
2. They are concatenated and masked with the instrumental indicator `has_lyrics` (from `features/lyric/language_id.parquet`):
   $$\mathbf{v}_{\text{lyric\_2048}} = \text{Normalize}\Big(\big[\, \mathbf{v}_{\text{harrier\_norm}} \;\|\; \mathbf{v}_{\text{e5\_norm}} \,\big]\Big) \times \mathbb{I}_{\text{has\_lyrics}} \in \mathbb{R}^{2048}$$
   *(If a track is an instrumental with no lyrics, its vector is safely masked to a zero vector).*
3. Cosine similarity is computed on GPU, self-masked, and the Top-250 neighbors are saved.

---

## 3. Unified Mood, Vibe & Context: `knn_mood_top250.parquet` (83-D)

* **Purpose:** Matches songs strictly by continuous emotional vibe, acoustic dynamics, vocal presence, macro/subgenre style, and historical release context.
* **Output Format:** Parquet table `(10,000 × 5)` (~22.68 MB).

### LOGO Ablation Decision:
Based on empirical Leave-One-Group-Out (LOGO) ablation testing (`docs/mood_similarity_ablation_report.md`), English-only emotion classifiers (`GoEmotions` & `NRC`, 36-D) were removed to eliminate zero-padding bias on non-English tracks. Removing emotion improved artist agreement from 15.21% to 19.11% and genre agreement to 99.83%, while achieving true English vs. non-English language parity. Genre Hybrid (50-D, 40%) and Temporal & Collab Context (10-D, 15%) were fused into a unified **83-D** representation.

### Models & Feature Breakdown:
| Sub-Component | Dimension | Weight | Underlying Models & Feature Suites | Source File |
| :--- | :--- | :--- | :--- | :--- |
| **Genre Hybrid Vector** | **50-D** | **40%** ($\sqrt{0.40}$) | 17-D Main Genre + 17-D Subgenre Rollup + 16-D Latent SVD space. | `data/embeddings/metadata/genre_hybrid_50d.npy` |
| **Spotify Audio Descriptors** | **11-D** | **30%** ($\sqrt{0.30}$) | Continuous acoustic descriptors (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `mode`, `loudness_scaled`, `tempo_scaled`, `time_signature_scaled`). | `data/embeddings/metadata/spotify_audio_11d.npy` |
| **Temporal & Collab Context** | **10-D** | **15%** ($\sqrt{0.15}$) | Release decade one-hots, collaboration flag, artist followers log, duration. | `data/embeddings/metadata/temporal_collab_10d.npy` |
| **Vocal & DSP Dynamics** | **12-D** | **15%** ($\sqrt{0.15}$) | **Silero VAD** deep vocal presence + **Librosa DSP dynamics** (integrated LUFS, crest factor, onset rate, spectral contrast). | `data/embeddings/metadata/vocal_dsp_12d.npy` |

### How It Is Computed:
1. The 4 sub-blocks are individually $L_2$-normalized.
2. They are weighted by their square-root variance allocations and concatenated into the unified 83-D vector:
   $$\mathbf{v}_{\text{mood\_83}} = \text{Normalize}\Big(\big[\; \sqrt{0.40}\,\mathbf{v}_{\text{genre\_50}} \;\|\; \sqrt{0.30}\,\mathbf{v}_{\text{spotify\_11}} \;\|\; \sqrt{0.15}\,\mathbf{v}_{\text{temporal\_10}} \;\|\; \sqrt{0.15}\,\mathbf{v}_{\text{vocal\_12}} \;\big]\Big) \in \mathbb{R}^{83}$$
3. Top-250 nearest neighbors are extracted via cosine similarity on GPU.

---

## 4. Master Multimodal Fusion: `knn_combined_top250.parquet` (3,795-D)

* **Purpose:** The primary, general-purpose recommendation engine balancing sound, lyric storytelling, emotional vibe, genre taxonomy, and historical context.
* **Output Format:** Parquet table `(10,000 × 5)` (~22.30 MB).

### The 6 Fused Representation Pillars (Weighted 73% Neural / 27% Context):
| Pillar | Dimension | Weight | Underlying Sources |
| :--- | :--- | :--- | :--- |
| **1. Fused Neural Audio** | **1,664-D** | **38%** ($\sqrt{0.38}$) | CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D) |
| **2. Fused Neural Lyric** | **2,048-D** | **35%** ($\sqrt{0.35}$) | Harrier-0.6B (1024-D) + Multilingual-E5 (1024-D) |
| **3. Genre Hybrid Vector** | **50-D** | **11%** ($\sqrt{0.11}$) | `data/embeddings/metadata/genre_hybrid_50d.npy` (17-D Main + 17-D Sub Rollup + 16-D Latent SVD) |
| **4. Spotify Audio & Vibe**| **11-D** | **8%** ($\sqrt{0.08}$) | `data/embeddings/metadata/spotify_audio_11d.npy` |
| **5. Temporal & Collab Context**| **10-D** | **4%** ($\sqrt{0.04}$) | `data/embeddings/metadata/temporal_collab_10d.npy` |
| **6. Vocal & DSP Dynamics**| **12-D** | **4%** ($\sqrt{0.04}$) | `data/embeddings/metadata/vocal_dsp_12d.npy` |

### Mathematical Fusion:
$$\mathbf{v}_{\text{combined}} = \text{Normalize}\Big(\big[\; \sqrt{0.38}\,\mathbf{v}_{\text{audio\_1664}} \;\|\; \sqrt{0.35}\,\mathbf{v}_{\text{lyric\_2048}} \;\|\; \sqrt{0.11}\,\mathbf{v}_{\text{genre\_50}} \;\|\; \sqrt{0.08}\,\mathbf{v}_{\text{spotify\_11}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{temporal\_10}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{vocal\_12}} \;\big]\Big) \in \mathbb{R}^{3795}$$

*(Each of the 6 pillar blocks is individually normalized to unit $L_2$ norm before weighted concatenation, guaranteeing optimal balance between high-dimensional neural representations and low-dimensional context signals).*

---

## 5. Summary Matrix of all 4 Files

| Parquet File | Total Dims | Primary Models Used | Key Source Files | Size |
| :--- | :--- | :--- | :--- | :--- |
| **`knn_audio_top250.parquet`** | **1,664-D** | CLAP + MERT-330M + VGGish | `clap_512d.npy`, `mert_330m_embeddings_1024d.npy`, `vggish_embeddings_128d.npy` | ~22.03 MB |
| **`knn_lyric_top250.parquet`** | **2,048-D** | Harrier-0.6B + Multilingual-E5 | `harrier_embeddings_1024d.npy`, `multilingual_e5_large_1024d.npy` | ~21.64 MB |
| **`knn_mood_top250.parquet`** | **83-D** | Genre Hybrid (40%) + Spotify (30%) + Temporal (15%) + Vocal DSP (15%) | `genre_hybrid_50d.npy`, `spotify_audio_11d.npy`, `temporal_collab_10d.npy`, `vocal_dsp_12d.npy` | ~22.68 MB |
| **`knn_combined_top250.parquet`**| **3,795-D** | 73% Neural (Audio 38% + Lyric 35%) + 27% Context (Genre 11% + Spotify 8% + Temporal 4% + Vocal 4%) | `audio_fused` (1664), `lyric_fused` (2048), `genre_hybrid_50d.npy`, `spotify_audio_11d.npy`, `temporal_collab_10d.npy`, `vocal_dsp_12d.npy` | ~22.30 MB |

### Internal Parquet Table Schema (Identical across all 4 files):
1. **`row_idx`** (`int32`): 0-indexed row position (0 to 9,999) corresponding to `songs.parquet`.
2. **`track_id`** (`string`): Target track's unique Spotify URI ID.
3. **`top250_neighbor_indices`** (`list[int32]`): Array of 250 row indices of nearest neighbors sorted by descending similarity.
4. **`top250_neighbor_track_ids`** (`list[string]`): Array of 250 Spotify track IDs for the neighbors.
5. **`top250_similarities`** (`list[float32]`): Array of 250 cosine similarity scores (scaled between $0.0$ and $1.0$).

Neither **`bge-m3`** nor **`panns`** is used in `knn_combined_top250.parquet`, exactly like in `knn_audio_top250.parquet` and `knn_lyric_top250.parquet`.

---

### The Exact Reasoning:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODALITY COMPOSITION & EXCLUSIONS                                  │
├───────────────────────┬──────────────────────────────────┬───────────────────────────────────────┤
│ Modality Pillar       │ Included in Similarity Graphs    │ Excluded Based on Ablation Results    │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ 🎵 Acoustic Audio     │ • LAION-CLAP (512-D)             │ ❌ PANNs Cnn14 (2,048-D)              │
│                       │ • MERT-v1-330M (1,024-D)         │    (Ablation proved it added noise &  │
│                       │ • Google VGGish (128-D)          │    degraded artist consistency)       │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ 📝 Lyric Narrative    │ • Harrier-0.6B (1,024-D)         │ ❌ BGE-M3 (1,024-D)                   │
│                       │ • Multilingual-E5-Large (1,024-D)│    (Redundant; Harrier 32k context +  │
│                       │                                  │    E5 delivered superior retrieval)   │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ 🎭 Mood & Context     │ • Genre Hybrid (50-D - 40%)      │ ❌ GoEmotions & NRC (36-D)            │
│                       │ • Spotify Audio (11-D - 30%)     │    (Dropped due to severe zero-padding│
│                       │ • Temporal & Collab (10-D - 15%) │    disparity on non-English tracks)   │
│                       │ • Vocal & DSP Dynamics (12-D-15%)│                                       │
└───────────────────────┴──────────────────────────────────┴───────────────────────────────────────┘
```

Because `knn_combined_top250.parquet` is built directly from the optimal fused audio block (`audio_fused`: 1,664-D) and optimal fused lyric block (`lyric_fused`: 2,048-D), plus the 4 context blocks with calibrated weights:

$$\mathbf{v}_{\text{combined}} = \text{Normalize}\Big(\big[\; \sqrt{0.38}\,\mathbf{v}_{\text{audio\_1664}} \;\|\; \sqrt{0.35}\,\mathbf{v}_{\text{lyric\_2048}} \;\|\; \sqrt{0.11}\,\mathbf{v}_{\text{genre\_50}} \;\|\; \sqrt{0.08}\,\mathbf{v}_{\text{spotify\_11}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{temporal\_10}} \;\|\; \sqrt{0.04}\,\mathbf{v}_{\text{vocal\_12}} \;\big]\Big) \in \mathbb{R}^{3795}$$

**Neither PANNs, BGE-M3, nor GoEmotions enters the master combined or mood graphs.**

---

### Why are PANNs, BGE-M3, and Emotion Features still in the dataset?
They are packaged under `embeddings/audio/`, `embeddings/lyric/`, and `features/` as **standalone researcher assets**:
* **`panns_tags_527d.npy` & `panns_embeddings_2048d.npy`**: Useful for researchers building audio classification models (detecting speech, applause, drums, environmental sounds from AudioSet).
* **`bge_m3_1024d.npy`**: Provides a dense 8,192-token embedding from BAAI for NLP researchers wanting to benchmark alternative retrieval models.
* **`emotion_sentiment_36d.npy` & `go_emotions.parquet`**: Preserved for NLP researchers studying English lyric emotion distributions.

For all 4 official similarity graphs, the pipeline uses only the **empirically validated, highest-performing ensemble**.