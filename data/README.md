# Spotify Top-10,000 Music Feature Dataset

A research-grade feature dataset extracted from 10,000 popular Spotify tracks (July 2025 corpus), providing multi-modal embeddings, structured acoustic descriptors, fine-grained lyric features, and pre-computed similarity indexes for Music Information Retrieval (MIR), recommendation systems, and machine learning research.

---

## 🌟 What's Included

### 1. Audio Embeddings (`embeddings/audio/`)
- **LAION-CLAP (512-D):** Zero-shot cross-modal representations linking audio content directly to natural language descriptions.
- **MERT-v1-95M (768-D):** Self-supervised transformer audio representations capturing acoustic nuances and musical hierarchy.
- **PANNs Cnn14 (2048-D) & PANNs AudioSet Tags (527-D):** Deep acoustic features and probability distributions over 527 AudioSet classes.
- **VGGish (128-D) & Mel-Stats (512-D):** Audio classification embeddings and spectral frequency statistics.

### 2. Multilingual Lyric Embeddings (`embeddings/lyrics/`)
- **Multilingual-E5-Large (1024-D):** High-precision cross-lingual retrieval winner (nDCG@10: 0.3001 on non-English lyrics).
- **BGE-M3 (1024-D):** Full-context 8,192-token dense representations supporting 100+ languages without document truncation.

### 3. Structured Classical MIR Features (`features/audio/`)
- **Librosa 88-Descriptor Suite (`dsp_librosa.parquet`):** Standardized rhythm, tempo, timbral MFCCs 1–20, spectral roll-off/flatness/contrast, chroma pitch histograms, tonnetz vectors, and BS.1770 integrated LUFS loudness.
- **Silero Vocal Activity Detection (`vad.parquet`):** Vocal duration and vocal presence ratio per track.

### 4. Interpretable Lyric & Emotion Features (`features/lyric/`)
- **RoBERTa GoEmotions (`go_emotions.parquet`):** 28 fine-grained emotion probability scores.
- **BERTopic Clusters (`bertopic_topics.parquet`):** 32 thematic lyric clusters with c-TF-IDF keyword labels.
- **Language & Script Taxonomy (`language_id.parquet`):** 35 flags detecting Devanagari Hindi, Romanized Hindi (Hinglish), Indonesian, Japanese, Korean, Chinese, Spanish, Portuguese, Turkish, French, German, Italian, Russian, Arabic, and Scandinavian languages.
- **Lexical Richness & Sentiment (`lyric_stats.parquet`):** TTR, Root TTR, MTLD, HD-D, Hapax legomena ratio, Flesch reading ease, VADER sentiment, and NRC EmoLex scores.

### 5. Similarity Graphs & Web Visualizations (`similarity/`)
- **Pre-computed Top-50 Nearest Neighbors (`knn_audio_top50.parquet`, `knn_lyric_top50.parquet`):** Instant multi-modal retrieval lookups.
- **2D Map Projections (`umap_2d_audio.parquet`, `umap_2d_lyric.parquet`, `umap_2d_combined.parquet`):** Standardized coordinates for WebGL interactive song visualization.

### 6. Evaluation Protocols (`splits/`)
- **Artist-Grouped 5-Fold (`artist_grouped_5fold.parquet`):** Strict zero-leakage cross-validation splits.
- **Temporal Chronological Split (`temporal_split.parquet`):** Real-world future generalizability evaluation (Train: ≤2022, Val: 2023, Test: 2024–2025).

---

## 🚀 Quickstart Usage

```python
import pandas as pd
import numpy as np

# 1. Load Track Metadata and Audio Features
songs = pd.read_parquet('metadata/songs.parquet')
dsp_features = pd.read_parquet('features/audio/dsp_librosa.parquet')
emotions = pd.read_parquet('features/lyric/go_emotions.parquet')

# 2. Load Neural Embeddings
clap_embeddings = np.load('embeddings/audio/clap_512d.npy')           # (10000, 512)
lyric_embeddings = np.load('embeddings/lyrics/bge_m3_1024d.npy')       # (10000, 1024)

# 3. Fast Zero-Leakage Cross Validation
splits = pd.read_parquet('splits/artist_grouped_5fold.parquet')
train_mask = splits['fold'] != 0
test_mask = splits['fold'] == 0

X_train = np.hstack([clap_embeddings[train_mask], dsp_features.iloc[train_mask, 2:].values])
y_train = songs.loc[train_mask, 'popularity'].values
```

---

## ⚖️ Citation & License
- Derived feature representations are released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)** for academic, research, and product development applications.
