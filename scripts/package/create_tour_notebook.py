"""
Generates notebooks/01_quickstart_and_dataset_tour.ipynb
Walkthrough covering every file group in the dataset with simple, practical examples.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🎵 Spotify Top-10,000 Music Dataset: Complete Tour & Quickstart
Welcome to the comprehensive walkthrough of the **Spotify Top-10,000 Music Feature Dataset**!

This dataset provides **multi-modal embeddings, acoustic descriptors, emotion scores, and pre-computed similarity graphs** across 10,000 popular Spotify tracks.

---

### 📌 Universal Alignment Rule:
Every file in this dataset shares the exact same row alignment: **Row `i` in any `.parquet` or `.npy` file corresponds to Track `i` in `metadata/songs.parquet` / `track_ids.npy`**.
"""))

# Section 1: Loading Metadata & Track Alignment
cells.append(nbf.v4.new_markdown_cell("""## 1. Environment Setup & Master Metadata (`metadata/`)"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Robust path resolution for Local Repo or Kaggle environment
if Path('data/metadata/songs.parquet').exists():
    DATA_DIR = Path('data')
elif Path('../data/metadata/songs.parquet').exists():
    DATA_DIR = Path('../data')
elif Path('/kaggle/input/spotify-10k-music-features').exists():
    DATA_DIR = Path('/kaggle/input/spotify-10k-music-features')
else:
    raise FileNotFoundError("Could not locate dataset directory.")

print(f"Using dataset path: {DATA_DIR.resolve()}")

# Load master track metadata
songs = pd.read_parquet(DATA_DIR / 'metadata' / 'songs.parquet')
artists = pd.read_parquet(DATA_DIR / 'metadata' / 'artists.parquet')
genres = pd.read_parquet(DATA_DIR / 'metadata' / 'genres.parquet')
track_ids = np.load(DATA_DIR / 'track_ids.npy')

print(f"Total songs: {len(songs):,}")
print(f"Total unique artists: {len(artists):,}")
print(f"Total genre tags: {len(genres):,}")

# Preview top popular tracks
songs[['track_name', 'artist_names', 'popularity', 'release_date', 'main_genres']].head()
"""))

# Section 2: Audio DSP Features & Vocal Detection
cells.append(nbf.v4.new_markdown_cell("""## 2. Classical Audio Features & Vocal Detection (`features/audio/`)
- `dsp_librosa.parquet`: 88 classical MIR descriptors (timbre, rhythm, harmony, loudness, stereo).
- `vad.parquet`: Silero Vocal Activity Detection (vocal ratio & duration).
"""))
cells.append(nbf.v4.new_code_cell("""# Load Audio DSP & VAD features
dsp = pd.read_parquet(DATA_DIR / 'features' / 'audio' / 'dsp_librosa.parquet')
vad = pd.read_parquet(DATA_DIR / 'features' / 'audio' / 'vad.parquet')

print(f"DSP feature table shape: {dsp.shape}")
print(f"VAD feature table shape: {vad.shape}")

# Inspect sample loudness (LUFS) and vocal ratio
audio_summary = pd.DataFrame({
    'track_name': songs['track_name'],
    'artist': songs['artist_names'],
    'tempo_bpm': dsp['tempo_librosa'],
    'loudness_lufs': dsp['lufs_integrated'],
    'vocal_ratio': vad['vocal_ratio'],
    'has_vocals': vad['has_vocals']
})
audio_summary.head()
"""))

# Section 3: Deep Neural Audio Embeddings & Tag Probabilities
cells.append(nbf.v4.new_markdown_cell("""## 3. Deep Audio Embeddings & AudioSet Tag Probabilities (`embeddings/audio/`)
- `clap_512d.npy`: LAION-CLAP zero-shot text-audio embeddings (512-D).
- `mert_embeddings_768d.npy`: MERT-v1-95M self-supervised music transformer representations (768-D).
- `panns_tags_527d.npy`: Probabilities across 527 AudioSet classes.
"""))
cells.append(nbf.v4.new_code_cell("""# Load audio embeddings & PANNs tag probabilities
clap = np.load(DATA_DIR / 'embeddings' / 'audio' / 'clap_512d.npy')
mert = np.load(DATA_DIR / 'embeddings' / 'audio' / 'mert_embeddings_768d.npy')
panns_tags = np.load(DATA_DIR / 'embeddings' / 'audio' / 'panns_tags_527d.npy')

with open(DATA_DIR / 'embeddings' / 'audio' / 'panns_tags_labels.json') as f:
    audioset_labels = json.load(f)

print(f"CLAP shape: {clap.shape} (dtype: {clap.dtype})")
print(f"MERT shape: {mert.shape} (dtype: {mert.dtype})")
print(f"PANNs tags shape: {panns_tags.shape}")

# Look at top predicted sound/instrument classes for Track 0
track_idx = 0
top_tag_indices = np.argsort(panns_tags[track_idx])[-5:][::-1]
print(f"\\nTop AudioSet Tags for '{songs.iloc[track_idx]['track_name']}':")
for idx in top_tag_indices:
    print(f"  - {audioset_labels[idx]}: {panns_tags[track_idx, idx]:.3f}")
"""))

# Section 4: Multilingual Lyric Embeddings & NLP Features
cells.append(nbf.v4.new_markdown_cell("""## 4. Multilingual Lyric Embeddings & NLP Descriptors (`features/lyric/`, `embeddings/lyrics/`)
- `multilingual_e5_large_1024d.npy` & `bge_m3_1024d.npy`: High-precision multilingual lyric representations.
- `language_id.parquet`: 35 language & script detection flags.
- `go_emotions.parquet`: 28 fine-grained emotion probability scores.
- `bertopic_topics.parquet`: 32 thematic lyric clusters.
"""))
cells.append(nbf.v4.new_code_cell("""# Load lyric embeddings & NLP features
e5_lyrics = np.load(DATA_DIR / 'embeddings' / 'lyrics' / 'multilingual_e5_large_1024d.npy')
bge_lyrics = np.load(DATA_DIR / 'embeddings' / 'lyrics' / 'bge_m3_1024d.npy')
lang_id = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'language_id.parquet')
emotions = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'go_emotions.parquet')
topics = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'bertopic_topics.parquet')

with open(DATA_DIR / 'features' / 'lyric' / 'bertopic_topic_labels.json') as f:
    topic_labels = json.load(f)

print("Language distribution top 8:")
print(lang_id['primary_language'].value_counts().head(8))

# Show top emotions for a track
top_emotion_cols = emotions.columns[3:]
sample_emotions = emotions.iloc[0][top_emotion_cols].astype(float)
print(f"\\nTop emotions for '{songs.iloc[0]['track_name']}':")
print(sample_emotions.sort_values(ascending=False).head(3))
"""))

# Section 5: Similarity Search
cells.append(nbf.v4.new_markdown_cell("""## 5. Instant Similarity Search with Pre-computed kNN Graphs (`similarity/`)
- `knn_audio_top50.parquet`: Pre-computed Top-50 acoustic neighbors.
- `knn_lyric_top50.parquet`: Pre-computed Top-50 lyrical neighbors.
"""))
cells.append(nbf.v4.new_code_cell("""# Load similarity graphs
knn_audio = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_audio_top50.parquet')
knn_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_lyric_top50.parquet')

# Query a track (e.g. Track 10)
query_idx = 10
query_title = songs.iloc[query_idx]['track_name']
query_artist = songs.iloc[query_idx]['artist_names']

print(f"Target Song: '{query_title}' by {query_artist}")

print("\\n🎵 Top 3 Acoustically Most Similar Songs:")
for neighbor_idx, sim in zip(knn_audio.iloc[query_idx]['top50_neighbor_indices'][:3], knn_audio.iloc[query_idx]['top50_similarities'][:3]):
    print(f"  - '{songs.iloc[neighbor_idx]['track_name']}' by {songs.iloc[neighbor_idx]['artist_names']} (Similarity: {sim:.3f})")

print("\\n📝 Top 3 Lyrically Most Similar Songs:")
for neighbor_idx, sim in zip(knn_lyric.iloc[query_idx]['top50_neighbor_indices'][:3], knn_lyric.iloc[query_idx]['top50_similarities'][:3]):
    print(f"  - '{songs.iloc[neighbor_idx]['track_name']}' by {songs.iloc[neighbor_idx]['artist_names']} (Similarity: {sim:.3f})")
"""))

# Section 6: 2D Song Map Visualization
cells.append(nbf.v4.new_markdown_cell("""## 6. 2D Map Visualization (`similarity/umap_2d_*.parquet`)
Pre-computed 2D coordinates ready for scatter plots and WebGL interactive maps.
"""))
cells.append(nbf.v4.new_code_cell("""# Load 2D coordinates
umap_audio = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_audio.parquet')
umap_lyric = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_lyric.parquet')

# Plot Audio 2D space
plt.figure(figsize=(8, 5))
plt.scatter(umap_audio['proj_x'], umap_audio['proj_y'], alpha=0.25, s=8, c='royalblue')
plt.title("2D Acoustic Space Map (10,000 Songs)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, alpha=0.2)
plt.show()
"""))

# Section 7: Zero-Leakage Evaluation Splits
cells.append(nbf.v4.new_markdown_cell("""## 7. Machine Learning Splits (`splits/`)
- `artist_grouped_5fold.parquet`: GroupKFold by `artist_id` guaranteeing zero artist leakage across train and test folds.
- `temporal_split.parquet`: Chronological train / val / test evaluation.
"""))
cells.append(nbf.v4.new_code_cell("""splits = pd.read_parquet(DATA_DIR / 'splits' / 'artist_grouped_5fold.parquet')
temp_split = pd.read_parquet(DATA_DIR / 'splits' / 'temporal_split.parquet')

print("GroupKFold Distribution across 5 folds:")
print(splits['fold'].value_counts().sort_index())

print("\\nTemporal Split Distribution:")
print(temp_split['split'].value_counts())
"""))

nb['cells'] = cells

out_dir = Path("notebooks")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "01_quickstart_and_dataset_tour.ipynb"

with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
