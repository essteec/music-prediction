# Spotify Top-10,000 Music Feature Dataset — Methodology & Extraction Guide

A research-grade dataset containing **multi-modal embeddings, acoustic descriptors, emotion scores, and pre-computed similarity graphs** for 10,000 popular Spotify tracks.

---

## 1. Audio Features & Embeddings

All audio features were extracted directly from 10,000 local Opus audio files decoded via `librosa` / `ffmpeg` over 100% full duration:

| Feature / Model | Dimensions / Output | Methodology & Tooling |
|---|---|---|
| **LAION-CLAP** | `(10000, 512)` float32 | Zero-shot text–audio cross-modal model (`HTSAT-base`). Audio resampled to 48 kHz and encoded to L2-normalized vectors. |
| **MERT-v1-330M** | `(10000, 1024)` float32 | Music-specific self-supervised transformer (330M params). Full-song mean-pooled representations. |
| **Google VGGish** | `(10000, 128)` float32 | Deep acoustic feature extraction using VGGish architecture trained on AudioSet. |
| **PANNs Cnn14** | `(10000, 2048)` float32 | Deep acoustic pattern extraction using CNN-14 architecture. |
| **Librosa 88-DSP Suite** | 91 columns (Parquet) | Classical MIR descriptors: MFCCs 1–20 (mean/std), spectral centroid/contrast, chroma pitch histograms, tonnetz, onset rate, and integrated LUFS loudness. |
| **Silero VAD** | 5 columns (Parquet) | Deep neural Vocal Activity Detection measuring vocal presence ratio and duration. |

---

## 2. Lyric Embeddings & NLP Descriptors

All lyrics were cleaned (removing Genius contributor tags, section headers `[Chorus]`, and whitespace normalization):

| Feature / Model | Dimensions / Output | Methodology & Tooling |
|---|---|---|
| **Harrier-OSS-v1-0.6B** | `(10000, 1024)` float32 | Microsoft state-of-the-art multilingual embedding model with 32k context window. |
| **Multilingual-E5-Large** | `(10000, 1024)` float32 | Benchmark-winning multilingual retriever (1024-D). |
| **RoBERTa GoEmotions** | 31 columns (Parquet) | Sequence classification on English tracks extracting 28 fine-grained emotion probabilities (`admiration`, `joy`, `love`, `sadness`, `anger`, etc.). |
| **Language & Script ID** | 35 columns (Parquet) | FastText (`lid.176`) + Devanagari script detection + Romanized Hindi (Hinglish) lexicons + 24 language indicators. |
| **Lexical Statistics** | 30 columns (Parquet) | TTR, Hapax ratio, Flesch reading ease, stanza counts, VADER sentiment, and NRC EmoLex emotions. |

---

## 3. Pre-Computed Similarity Graphs & 2D Manifolds (`similarity/`)

| Deliverable | File Path | Description |
|---|---|---|
| **Acoustic kNN Graph** | `knn_audio_top100.parquet` | Pre-computed Top-100 nearest neighbors via cosine indexing on optimal fused CLAP + MERT-330M + VGGish embeddings (1664-D). |
| **Lyric kNN Graph** | `knn_lyric_top100.parquet` | Pre-computed Top-100 nearest neighbors via cosine indexing on optimal fused Harrier-0.6B + Multilingual-E5 embeddings (2048-D). |
| **Mood & Vibe kNN Graph**| `knn_mood_top100.parquet` | Pre-computed Top-100 nearest neighbors on Spotify Audio (11-D) + GoEmotions/NRC (36-D) + Vocal DSP (12-D) (59-D). |
| **Master Multimodal Graph** | `knn_combined_top100.parquet` | Pre-computed Top-100 nearest neighbors on fused Audio + Lyric + Spotify + Vocal DSP + Genre Hybrid + Temporal representations (3795-D). |
| **2D UMAP Projections** | `umap_2d_*.parquet` | 2D projection coordinates for Audio, Lyric, Mood, and Combined Multimodal representations. *(Qualitative visual projections for exploratory mapping).* |
| **Cross-Validation** | `artist_grouped_5fold.parquet` | 5-fold GroupKFold by `artist_id` for leakage-free evaluation. |

---

## 4. Quickstart Code Snippet

```python
import pandas as pd
import numpy as np

# 1. Load Primary Metadata
songs = pd.read_parquet('metadata/songs.parquet')

# 2. Instant Similarity Query across 4 Dedicated Facets
knn_audio = pd.read_parquet('similarity/knn_audio_top100.parquet')
knn_lyric = pd.read_parquet('similarity/knn_lyric_top100.parquet')
knn_mood  = pd.read_parquet('similarity/knn_mood_top100.parquet')
knn_comb  = pd.read_parquet('similarity/knn_combined_top100.parquet')

query_idx = 0  # e.g., Lady Gaga - Bad Romance
print(f"Top 5 Master Recommendations for '{songs.iloc[query_idx]['track_name']}':")
for nb_idx, sim in zip(knn_comb.iloc[query_idx]['top100_neighbor_indices'][:5], knn_comb.iloc[query_idx]['top100_similarities'][:5]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Sim: {sim:.3f})")
```
