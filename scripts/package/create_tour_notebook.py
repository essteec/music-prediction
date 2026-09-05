"""
Generates notebooks/01_quickstart_and_dataset_tour.ipynb
Quickstart guide and comprehensive tour across all multimodal features, embeddings, and Top-250 similarity graphs.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""#  Spotify Top-10,000 Songs: Dataset Tour & Quickstart
### A Multimodal Dataset for Hit Song Prediction, Audio Embedding Benchmarks, and Music Recommendation

Welcome! This notebook provides a fast, comprehensive walkthrough of the **Spotify Top-10,000 Songs Dataset**.
We demonstrate how to load and query each component:

1. **Core Metadata & Metrics** (`metadata/songs.parquet`)
2. **Engineered Feature Tables** (`features/audio/`, `features/lyric/`, `features/metadata/`)
3. **Full-Song Deep Audio Embeddings** (`embeddings/audio/` — CLAP, MERT-330M, PANNs, VGGish, Mel Stats)
4. **Multilingual Lyric Embeddings** (`embeddings/lyric/` — Harrier-OSS-v1-0.6B, Multilingual E5-Large)
5. **Instant Pre-computed Top-250 Similarity Graphs** (`similarity/knn_*_top250.parquet` — Audio, Lyric, Mood, Combined)
6. **4-Facet 2D Latent Space Visualizations** (`similarity/umap_2d_*.parquet`)
7. **Zero-Leakage ML Evaluation Splits** (`splits/`)
"""))

# Section 1: Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Environment
Run this cell to set up paths. Works seamlessly both on Kaggle and in local clone environments.
"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Robust path resolution
if Path('data/metadata/songs.parquet').exists():
    DATA_DIR = Path('data')
elif Path('../data/metadata/songs.parquet').exists():
    DATA_DIR = Path('../data')
elif Path('/kaggle/input/spotify-10k-music-features').exists():
    DATA_DIR = Path('/kaggle/input/spotify-10k-music-features')
else:
    raise FileNotFoundError("Dataset path not found. Please ensure data directory exists.")

print(f"Data root resolved to: {DATA_DIR.resolve()}")
"""))

# Section 2: Metadata & Engineered Features
cells.append(nbf.v4.new_markdown_cell("""## 2. Core Metadata & Engineered Feature Tables
- `metadata/songs.parquet`: 10,000 tracks with Spotify audio features, artist genres, release dates, and lyrics.
- `features/audio/dsp_librosa.parquet`: 91 DSP features (LUFS, onset strength, spectral contrast, MFCCs).
- `features/audio/vad.parquet`: Voice Activity Detection (vocal ratio & duration).
- `features/lyric/go_emotions.parquet`: 28 fine-grained emotional probabilities.
- `features/lyric/lyric_stats.parquet`: Lexical diversity (TTR, hapax ratio, reading ease, VADER & NRC sentiment).
- `features/metadata/derived.parquet`: Temporal decade one-hots, artist collaboration flags, and follower metrics.
"""))
cells.append(nbf.v4.new_code_cell("""songs = pd.read_parquet(DATA_DIR / 'metadata' / 'songs.parquet')
derived = pd.read_parquet(DATA_DIR / 'features' / 'metadata' / 'derived.parquet')
vad = pd.read_parquet(DATA_DIR / 'features' / 'audio' / 'vad.parquet')

print(f"Loaded {len(songs):,} songs across {len(songs.columns)} columns.")
display(songs[['rank', 'track_name', 'artist_names', 'popularity', 'danceability', 'energy', 'valence', 'main_genres']].head(3))
"""))

# Section 3: Audio Embeddings
cells.append(nbf.v4.new_markdown_cell("""## 3. Full-Song Acoustic Embeddings (`embeddings/audio/`)
All audio representations were extracted over **100% full-song audio durations**:
- `clap_512d.npy`: LAION-CLAP zero-shot acoustic-text contrastive embeddings (512-D).
- `mert_330m_embeddings_1024d.npy`: Mean-pooled MERT-v1-330M musical transformer embeddings (1024-D).
- `panns_embeddings_2048d.npy`: PANNs CNN14 audio pattern embeddings (2048-D).
- `vggish_embeddings_128d.npy`: Google VGGish acoustic texture embeddings (128-D).
- `mel_stats_embeddings_512d.npy`: Spectral distribution statistics (512-D).
"""))
cells.append(nbf.v4.new_code_cell("""clap = np.load(DATA_DIR / 'embeddings' / 'audio' / 'clap_512d.npy')
mert = np.load(DATA_DIR / 'embeddings' / 'audio' / 'mert_330m_embeddings_1024d.npy')
vgg  = np.load(DATA_DIR / 'embeddings' / 'audio' / 'vggish_embeddings_128d.npy')

print(f"CLAP Embedding shape:     {clap.shape}")
print(f"MERT-330M shape:          {mert.shape}")
print(f"VGGish shape:             {vgg.shape}")
"""))

# Section 4: Multilingual Lyric Embeddings
cells.append(nbf.v4.new_markdown_cell("""## 4. Multilingual Lyric Representations (`embeddings/lyric/`)
- `harrier_embeddings_1024d.npy`: Microsoft Harrier-OSS-v1-0.6B state-of-the-art multilingual embedding model (1024-D).
- `multilingual_e5_large_1024d.npy`: Multilingual E5-Large sentence embeddings (1024-D).
- `features/lyric/language_id.parquet`: Primary/secondary language classification across 26 languages.
"""))
cells.append(nbf.v4.new_code_cell("""harrier_lyrics = np.load(DATA_DIR / 'embeddings' / 'lyric' / 'harrier_embeddings_1024d.npy')
e5_lyrics = np.load(DATA_DIR / 'embeddings' / 'lyric' / 'multilingual_e5_large_1024d.npy')
lang_id = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'language_id.parquet')

print(f"Harrier Embedding shape:  {harrier_lyrics.shape}")
print(f"E5-Large Embedding shape: {e5_lyrics.shape}")

print("\\nTop 8 Languages in Dataset:")
print(lang_id['primary_language'].value_counts().head(8))
"""))

# Section 5: Top-250 Similarity Graphs
cells.append(nbf.v4.new_markdown_cell("""## 5. Instant Similarity Search with Pre-computed Top-250 kNN Graphs (`similarity/`)
We load the pre-computed Top-250 similarity matrices for sub-millisecond query lookups:
- `knn_audio_top250.parquet`: Fused acoustic similarity (`CLAP + MERT-330M + VGGish` = 1664-D).
- `knn_lyric_top250.parquet`: Fused lyrical storytelling similarity (`Harrier + E5-Large` = 2048-D).
- `knn_mood_top250.parquet`: Unified mood, vibe & context similarity (`Genre 40% + Spotify 30% + Temporal 15% + Vocal 15%` = 83-D).
- `knn_combined_top250.parquet`: Master multimodal similarity (`73% Neural (Audio 38% + Lyric 35%) + 27% Context (Genre 11% + Spotify 8% + Temporal 4% + Vocal 4%)` = 3795-D).
"""))
cells.append(nbf.v4.new_code_cell("""# Load all 4 similarity graphs
knn_audio = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_audio_top250.parquet')
knn_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_lyric_top250.parquet')
knn_mood  = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_mood_top250.parquet')
knn_comb  = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_combined_top250.parquet')

query_idx = 10
query_title = songs.iloc[query_idx]['track_name']
query_artist = songs.iloc[query_idx]['artist_names']

print(f"Target Song [{query_idx}]: '{query_title}' by {query_artist}\\n")

print("1. Acoustically Most Similar (knn_audio_top250):")
for nb_idx, sim in zip(knn_audio.iloc[query_idx]['top250_neighbor_indices'][:3], knn_audio.iloc[query_idx]['top250_similarities'][:3]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Sim: {sim:.3f})")

print("\\n 2. Lyrically Most Similar (knn_lyric_top250):")
for nb_idx, sim in zip(knn_lyric.iloc[query_idx]['top250_neighbor_indices'][:3], knn_lyric.iloc[query_idx]['top250_similarities'][:3]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Sim: {sim:.3f})")

print("\\n 3. Unified Mood & Context Most Similar (knn_mood_top250):")
for nb_idx, sim in zip(knn_mood.iloc[query_idx]['top250_neighbor_indices'][:3], knn_mood.iloc[query_idx]['top250_similarities'][:3]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Sim: {sim:.3f})")

print("\\n 4. Master Multimodal Recommendations (knn_combined_top250):")
for nb_idx, sim in zip(knn_comb.iloc[query_idx]['top250_neighbor_indices'][:3], knn_comb.iloc[query_idx]['top250_similarities'][:3]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Sim: {sim:.3f})")
"""))

# Section 6: Side-by-Side 2D UMAP Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 6. Multi-Modal 2D Visualizations (`similarity/umap_2d_*.parquet`)
Side-by-side 2D semantic maps comparing **Acoustic Audio Space**, **Semantic Lyric Space**, **Mood & Emotion Space**, and **Master Multimodal Space**.

> **Interpretation Note**: The 2D coordinates in `similarity/umap_2d_*.parquet` are qualitative non-linear dimensionality reduction projections intended for visual exploration and clustering inspection. For quantitative metric distance or true mathematical similarity, always use the high-dimensional embeddings or the Top-250 kNN graph tables.
"""))
cells.append(nbf.v4.new_code_cell("""# Load all 4 UMAP projections
umap_audio = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_audio.parquet')
umap_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_lyric.parquet')
umap_mood  = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_mood.parquet')
umap_comb  = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_combined.parquet')

# Color mappings
lang_colors = lang_id['primary_language'].map({
    'en': 'crimson', 'es': 'gold', 'pt': 'darkorange', 'hi': 'darksalmon',
    'id': 'mediumseagreen', 'ko': 'mediumpurple', 'tr': 'turquoise', 'ja': 'hotpink',
    'fr': 'royalblue', 'none': 'gray'
}).fillna('lightgray')

genre_color_map = {
    'pop': 'dodgerblue', 'rap': 'crimson', 'hip hop': 'darkred',
    'rock': 'darkorange', 'latin': 'gold', 'dance pop': 'deepskyblue', 'edm': 'mediumpurple'
}
audio_colors = songs['main_genres'].map(lambda g: genre_color_map.get(g, 'lightgray'))

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# 1. Audio UMAP
axes[0].scatter(umap_audio['proj_x'], umap_audio['proj_y'], c=audio_colors, alpha=0.3, s=5)
axes[0].set_title("1. Acoustic Audio Space (CLAP+MERT+VGGISH)", fontsize=10, fontweight='bold')
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[0].grid(True, alpha=0.2)

# 2. Lyric UMAP
axes[1].scatter(umap_lyric['proj_x'], umap_lyric['proj_y'], c=lang_colors, alpha=0.3, s=5)
axes[1].set_title("2. Semantic Lyric Space (Harrier+E5)", fontsize=10, fontweight='bold')
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")
axes[1].grid(True, alpha=0.2)

# 3. Mood UMAP
axes[2].scatter(umap_mood['proj_x'], umap_mood['proj_y'], c=songs['valence'], cmap='coolwarm', alpha=0.35, s=5)
axes[2].set_title("3. Unified Mood & Context Space (83-D)", fontsize=10, fontweight='bold')
axes[2].set_xlabel("UMAP 1")
axes[2].set_ylabel("UMAP 2")
axes[2].grid(True, alpha=0.2)

# 4. Master Combined UMAP
axes[3].scatter(umap_comb['proj_x'], umap_comb['proj_y'], c=songs['energy'], cmap='plasma', alpha=0.35, s=5)
axes[3].set_title("4. Master Multimodal Space (3795-D)", fontsize=10, fontweight='bold')
axes[3].set_xlabel("UMAP 1")
axes[3].set_ylabel("UMAP 2")
axes[3].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
"""))

# Section 7: Machine Learning Splits
cells.append(nbf.v4.new_markdown_cell("""## 7. Machine Learning Splits (`splits/`)
- `artist_grouped_5fold.parquet`: GroupKFold by `artist_id` guaranteeing zero artist leakage across train and test folds.
- `temporal_split.parquet`: Chronological train / val / test evaluation.
"""))
cells.append(nbf.v4.new_code_cell("""art_split = pd.read_parquet(DATA_DIR / 'splits' / 'artist_grouped_5fold.parquet')
temp_split = pd.read_parquet(DATA_DIR / 'splits' / 'temporal_split.parquet')

print("Artist Grouped 5-Fold distribution:")
print(art_split['fold'].value_counts().sort_index())

print("\\nTemporal Train / Val / Test distribution:")
print(temp_split['split'].value_counts())
"""))

nb['cells'] = cells

out_file = Path("notebooks/01_quickstart_and_dataset_tour.ipynb")
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
