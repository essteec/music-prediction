Here is the exact, comprehensive breakdown of all 4 pre-computed Top-100 kNN similarity files in `data/similarity/`, including their **underlying models, source files, dimensionality, mathematical fusion, and computation algorithms**.

---

```
                               ┌────────────────────────────────────────────────────────────┐
                               │           THE 4 PRE-COMPUTED TOP-100 SIMILARITY FILES      │
                               └─────────────────────────────┬──────────────────────────────┘
                                                             │
        ┌────────────────────────────┬───────────────────────┴───────────────────────┬────────────────────────────┐
        │                            │                                               │                            │
┌───────▼────────────────────┐ ┌─────▼──────────────────────┐ ┌──────────────────────▼──────┐ ┌───────────────────▼─────────┐
│ 1. knn_audio_top100.parquet│ │ 2. knn_lyric_top100.parquet│ │ 3. knn_mood_top100.parquet   │ │ 4. knn_combined_top100.parquet│
│   (Pure Acoustic Sound)    │ │   (Lyrical Storytelling)   │ │   (Dedicated Mood & Vibe)    │ │   (Master Multimodal Fusion) │
│          1,664-D           │ │          2,048-D           │ │             59-D             │ │            3,795-D           │
└────────────────────────────┘ └────────────────────────────┘ └──────────────────────────────┘ └───────────────────────────┘
```

---

## 1. Pure Acoustic Sound: `knn_audio_top100.parquet` (1,664-D)

* **Purpose:** Matches songs strictly by acoustic timbre, production textures, beat patterns, and instrumentation (ignoring lyrics and genre labels).
* **Output Format:** Parquet table `(10,000 × 5)` (~9.32 MB).

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
3. Cosine similarity is computed across all $10,000 \times 10,000$ pairs on GPU. Self-similarity is masked to $-\infty$, and the top 100 highest neighbors are stored per song.

---

## 2. Lyrical Storytelling: `knn_lyric_top100.parquet` (2,048-D)

* **Purpose:** Matches songs purely by narrative themes, poetic style, metaphors, and sentiment—completely language-agnostic.
* **Output Format:** Parquet table `(10,000 × 5)` (~9.19 MB).

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
3. Cosine similarity is computed on GPU, self-masked, and the Top-100 neighbors are saved.

---

## 3. Dedicated Mood & Vibe: `knn_mood_top100.parquet` (59-D)

* **Purpose:** Matches songs strictly by emotional energy, happiness vs. sadness (valence), emotional granularity, and vocal presence.
* **Output Format:** Parquet table `(10,000 × 5)` (~9.49 MB).

### Models & Feature Breakdown:
| Sub-Component | Dimension | Underlying Models & Feature Suites | Source File |
| :--- | :--- | :--- | :--- |
| **Spotify Audio Descriptors** | **11-D** | High-level acoustic indicators (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `mode`, `loudness`, `tempo`, `time_signature`). | `data/embeddings/metadata/spotify_audio_11d.npy` (from `metadata/songs.parquet`) |
| **GoEmotions + NRC Emotion** | **36-D** | **28 RoBERTa GoEmotions probabilities** (`admiration`, `joy`, `sadness`, `anger`, `fear`, etc.) + **8 NRC EmoLex lexicon densities**. | `data/embeddings/metadata/emotion_sentiment_36d.npy` (from `go_emotions.parquet` + `lyric_stats.parquet`) |
| **Vocal & DSP Dynamics** | **12-D** | **Silero VAD** deep vocal detector (`vocal_ratio`, `has_vocals`) + **Librosa DSP dynamics** (LUFS, crest factor, onset rate, spectral centroid, stereo width). | `data/embeddings/metadata/vocal_dsp_12d.npy` (from `vad.parquet` + `dsp_librosa.parquet`) |

### How It Is Computed:
1. The 3 sub-blocks are individually $L_2$-normalized.
2. They are concatenated and re-normalized into a compact 59-D vector:
   $$\mathbf{v}_{\text{mood\_59}} = \text{Normalize}\Big(\big[\, \mathbf{v}_{\text{spotify\_11}} \;\|\; \mathbf{v}_{\text{emotion\_36}} \;\|\; \mathbf{v}_{\text{vocal\_dsp\_12}} \,\big]\Big) \in \mathbb{R}^{59}$$
3. Top-100 nearest neighbors are extracted via cosine similarity on GPU.

---

## 4. Master Multimodal Fusion: `knn_combined_top100.parquet` (3,795-D)

* **Purpose:** The primary, general-purpose recommendation engine balancing sound, lyric storytelling, emotional vibe, genre taxonomy, and historical context.
* **Output Format:** Parquet table `(10,000 × 5)` (~9.49 MB).

### The 6 Fused Representation Pillars (Total: 3,795-D):
| Pillar | Dimension | Underlying Sources |
| :--- | :--- | :--- |
| **1. Fused Neural Audio** | **1,664-D** | CLAP (512-D) + MERT-330M (1024-D) + VGGish (128-D) |
| **2. Fused Neural Lyric** | **2,048-D** | Harrier-0.6B (1024-D) + Multilingual-E5 (1024-D) |
| **3. Spotify Audio & Vibe**| **11-D** | `data/embeddings/metadata/spotify_audio_11d.npy` |
| **4. Vocal & DSP Dynamics**| **12-D** | `data/embeddings/metadata/vocal_dsp_12d.npy` |
| **5. Genre Hybrid Vector** | **50-D** | `data/embeddings/metadata/genre_hybrid_50d.npy`<br>• 17-D Song Macro Multi-Hot<br>• 17-D Subgenre Rollup Recipe via `genres.parquet`<br>• 16-D Latent SVD Subgenre Space (via `TruncatedSVD(16)`) |
| **6. Temporal & Collab Context**| **10-D** | `data/embeddings/metadata/temporal_collab_10d.npy`<br>• Release decade one-hots, collaboration flag, artist followers log, duration |

### Mathematical Fusion:
$$\mathbf{v}_{\text{combined}} = \text{Normalize}\Big(\big[\; \mathbf{v}_{\text{audio\_1664}} \;\|\; \mathbf{v}_{\text{lyric\_2048}} \;\|\; \mathbf{v}_{\text{spotify\_11}} \;\|\; \mathbf{v}_{\text{vocal\_12}} \;\|\; \mathbf{v}_{\text{genre\_50}} \;\|\; \mathbf{v}_{\text{temporal\_10}} \;\big]\Big) \in \mathbb{R}^{3795}$$

*(Each of the 6 pillar blocks is individually normalized to unit $L_2$ norm before concatenation, ensuring that high-dimensional blocks like lyrics or audio do not overpower low-dimensional signals like genre or tempo).*

---

## 5. Summary Matrix of all 4 Files

| Parquet File | Total Dims | Primary Models Used | Key Source Files |
| :--- | :--- | :--- | :--- |
| **`knn_audio_top100.parquet`** | **1,664-D** | CLAP + MERT-330M + VGGish | `clap_512d.npy`, `mert_330m_embeddings_1024d.npy`, `vggish_embeddings_128d.npy` |
| **`knn_lyric_top100.parquet`** | **2,048-D** | Harrier-0.6B + Multilingual-E5 | `harrier_embeddings_1024d.npy`, `multilingual_e5_large_1024d.npy` |
| **`knn_mood_top100.parquet`** | **59-D** | Spotify Features + RoBERTa GoEmotions + NRC EmoLex + Silero VAD + Librosa | `spotify_audio_11d.npy`, `emotion_sentiment_36d.npy`, `vocal_dsp_12d.npy` |
| **`knn_combined_top100.parquet`**| **3,795-D** | Full Multimodal Ensemble (Audio + Lyrics + Spotify + VAD + Librosa + Genre Hybrid + Temporal) | `audio_fused` (1664), `lyric_fused` (2048), `spotify_audio_11d.npy`, `vocal_dsp_12d.npy`, `genre_hybrid_50d.npy`, `temporal_collab_10d.npy` |

### Internal Parquet Table Schema (Identical across all 4 files):
1. **`row_idx`** (`int32`): 0-indexed row position (0 to 9,999) corresponding to `songs.parquet`.
2. **`track_id`** (`string`): Target track's unique Spotify URI ID.
3. **`top100_neighbor_indices`** (`list[int32]`): Array of 100 row indices of nearest neighbors sorted by descending similarity.
4. **`top100_neighbor_track_ids`** (`list[string]`): Array of 100 Spotify track IDs for the neighbors.
5. **`top100_similarities`** (`list[float32]`): Array of 100 cosine similarity scores (scaled between $0.0$ and $1.0$).

Neither **`bge-m3`** nor **`panns`** is used in `knn_combined_top100.parquet`, exactly like in `knn_audio_top100.parquet` and `knn_lyric_top100.parquet`.

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
└───────────────────────┴──────────────────────────────────┴───────────────────────────────────────┘
```

Because `knn_combined_top100.parquet` is built directly from the optimal fused audio block (`audio_fused`: 1,664-D) and optimal fused lyric block (`lyric_fused`: 2,048-D):

$$\mathbf{v}_{\text{combined}} = \text{Normalize}\Big(\big[\; \mathbf{v}_{\text{audio\_1664}} \;\|\; \mathbf{v}_{\text{lyric\_2048}} \;\|\; \mathbf{v}_{\text{spotify\_11}} \;\|\; \mathbf{v}_{\text{vocal\_12}} \;\|\; \mathbf{v}_{\text{genre\_50}} \;\|\; \mathbf{v}_{\text{temporal\_10}} \;\big]\Big) \in \mathbb{R}^{3795}$$

**Neither PANNs nor BGE-M3 enters the master combined graph.**

---

### Why are PANNs and BGE-M3 still in the dataset?
They are packaged under `embeddings/audio/` and `embeddings/lyric/` as **standalone researcher assets**:
* **`panns_tags_527d.npy` & `panns_embeddings_2048d.npy`**: Useful for researchers building audio classification models (detecting speech, applause, drums, environmental sounds from AudioSet).
* **`bge_m3_1024d.npy`**: Provides a dense 8,192-token embedding from BAAI for NLP researchers wanting to benchmark alternative retrieval models.

For all 4 official similarity graphs, your pipeline uses only the **empirically validated, highest-performing ensemble**.