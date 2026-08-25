# Spotify Top-10,000 Music Feature Dataset — Methodology & Extraction Guide

A research-grade dataset containing **multi-modal embeddings, acoustic descriptors, emotion scores, and pre-computed similarity graphs** for 10,000 popular Spotify tracks.

---

## 🎧 1. Audio Features & Embeddings

All audio features were extracted directly from 10,000 local Opus audio files decoded via `librosa` / `ffmpeg`:

| Feature / Model | Dimensions / Output | Methodology & Tooling |
|---|---|---|
| **LAION-CLAP** | `(10000, 512)` float32 | Zero-shot text–audio cross-modal model (`HTSAT-base`). Audio resampled to 48 kHz and encoded to L2-normalized vectors. |
| **MERT-v1-95M** | `(10000, 768)` float32 | Music-specific self-supervised transformer. 24 kHz audio chunked in 30s segments with mean-pooled hidden states. |
| **PANNs Cnn14** | `(10000, 2048)` float32 | Deep acoustic feature extraction using CNN-14 architecture trained on AudioSet. |
| **PANNs AudioSet Tags** | `(10000, 527)` float32 | Sigmoid output layer predicting probabilities across 527 AudioSet sound & music classes. |
| **Librosa 88-DSP Suite** | 91 columns (Parquet) | Classical MIR descriptors: MFCCs 1–20 (mean/std), spectral centroid/flatness/contrast/rolloff, chroma pitch histograms, tonnetz, onset rate, tempo, and EBU R128 integrated LUFS loudness. |
| **Silero VAD** | 5 columns (Parquet) | Deep neural Vocal Activity Detection measuring speech/singing duration and vocal presence ratio. |

---

## 📝 2. Lyric Embeddings & NLP Descriptors

All lyrics were cleaned (removing Genius contributor tags, section headers `[Chorus]`, and whitespace normalization):

| Feature / Model | Dimensions / Output | Methodology & Tooling |
|---|---|---|
| **Multilingual-E5-Large** | `(10000, 1024)` float32 | Benchmark-winning multilingual retriever (nDCG@10: 0.3001). Cleaned lyrics encoded with `passage:` prefix on GPU. |
| **BGE-M3** | `(10000, 1024)` float32 | Dense cross-lingual model supporting full 8,192-token context length without lyric truncation. |
| **RoBERTa GoEmotions** | 31 columns (Parquet) | Sequence classification on English tracks extracting 28 fine-grained emotion probabilities (`admiration`, `joy`, `love`, `sadness`, `anger`, etc.). |
| **BERTopic Clustering** | 3 columns (Parquet) | 32 thematic lyric topics generated using BGE-M3 representations with c-TF-IDF keyword extraction. |
| **Language & Script ID** | 35 columns (Parquet) | FastText (`lid.176`) + Devanagari script detection + Romanized Hindi (Hinglish) lexicons + 24 language indicators. |
| **Lexical Statistics** | 30 columns (Parquet) | TTR, Root-TTR, MTLD, HD-D, Hapax legomena ratio, Flesch reading ease, VADER sentiment, and NRC EmoLex emotions. |

---

## 🔗 3. Similarity Graphs, Projections & Splits

| Deliverable | File Path | Description |
|---|---|---|
| **Audio kNN Graph** | `knn_audio_top50.parquet` | Pre-computed Top-50 nearest neighbors via FAISS cosine indexing on concatenated CLAP + MERT embeddings. |
| **Lyric kNN Graph** | `knn_lyric_top50.parquet` | Pre-computed Top-50 nearest neighbors via FAISS cosine indexing on Multilingual-E5 + BGE-M3 embeddings. |
| **2D Map Projections** | `umap_2d_*.parquet` | PCA-initialized t-SNE 2D coordinates normalized to `[-100, 100]` for WebGL song map visualization (Audio, Lyric, and Combined). |
| **Artist-Grouped Splits** | `artist_grouped_5fold.parquet` | Strict 5-fold cross-validation split grouped by `artist_id` to prevent data leakage. |
| **Temporal Split** | `temporal_split.parquet` | Chronological split (Train: ≤2022, Val: 2023, Test: 2024–2025) for real-world future generalization testing. |

---

## ⚡ 4. Quick Start in Python

```python
import pandas as pd
import numpy as np

# 1. Load song metadata & extracted acoustic features
songs = pd.read_parquet('metadata/songs.parquet')
audio_dsp = pd.read_parquet('features/audio/dsp_librosa.parquet')
emotions = pd.read_parquet('features/lyric/go_emotions.parquet')

# 2. Load neural embeddings
clap = np.load('embeddings/audio/clap_512d.npy')          # (10000, 512)
lyrics_e5 = np.load('embeddings/lyrics/multilingual_e5_large_1024d.npy') # (10000, 1024)

# 3. Use 5-fold zero-leakage cross-validation
splits = pd.read_parquet('splits/artist_grouped_5fold.parquet')
train_idx = np.where(splits['fold'] != 0)[0]
test_idx = np.where(splits['fold'] == 0)[0]
```
