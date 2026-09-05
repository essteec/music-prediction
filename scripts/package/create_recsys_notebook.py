"""
Generates notebooks/03_content_based_recommendation_engine.ipynb
Advanced Multimodal Music Recommendation Engine with Audio, Lyric, Mood, Genre, and Temporal Fusion.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""#  Multi-Modal Content-Based Music Recommendation Engine
### Fusing Full-Song Acoustic Embeddings, Multilingual Lyrics, Mood & Vibe Descriptors, and Contextual Guards

This notebook implements an industrial-grade **multi-modal content-based recommendation engine** for 10,000 Spotify songs. We demonstrate:

1. **Sub-Millisecond Pre-computed Top-250 Lookups:** Querying [`knn_audio_top250.parquet`](file:///home/esstee/projects/top10k/music-prediction/data/similarity/knn_audio_top250.parquet), [`knn_lyric_top250.parquet`](file:///home/esstee/projects/top10k/music-prediction/data/similarity/knn_lyric_top250.parquet), [`knn_mood_top250.parquet`](file:///home/esstee/projects/top10k/music-prediction/data/similarity/knn_mood_top250.parquet), and [`knn_combined_top250.parquet`](file:///home/esstee/projects/top10k/music-prediction/data/similarity/knn_combined_top250.parquet).
2. **Dynamic Multi-Modal Interactive Engine:** Blending acoustic timbre, lyrical narrative, unified mood & context (83-D), genre style, and temporal era with customizable weights.
3. **Contextual Constraints & Diversity Guards:** Enforcing same-language, same-decade filtering, or artist diversity penalties.
4. **Visual Playlist Journey on 2D Latent Manifold:** Visualizing smooth transition paths across the 2D multimodal latent space.
"""))

# Section 1: Setup & Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Data Loading"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Robust path resolution
if Path('data/metadata/songs.parquet').exists():
    DATA_DIR = Path('data')
elif Path('../data/metadata/songs.parquet').exists():
    DATA_DIR = Path('../data')
elif Path('/kaggle/input/spotify-10k-music-features').exists():
    DATA_DIR = Path('/kaggle/input/spotify-10k-music-features')
else:
    raise FileNotFoundError("Dataset path not found.")

songs = pd.read_parquet(DATA_DIR / 'metadata' / 'songs.parquet')
lang_id = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'language_id.parquet')
derived = pd.read_parquet(DATA_DIR / 'features' / 'metadata' / 'derived.parquet')

# Load L2-Normalized Feature Arrays
def l2_norm(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

# 1. Acoustic Representation (1664-D: CLAP + MERT + VGGish)
clap = np.load(DATA_DIR / 'embeddings' / 'audio' / 'clap_512d.npy')
mert = np.load(DATA_DIR / 'embeddings' / 'audio' / 'mert_330m_embeddings_1024d.npy')
vgg  = np.load(DATA_DIR / 'embeddings' / 'audio' / 'vggish_embeddings_128d.npy')
audio_norm = l2_norm(np.concatenate([l2_norm(clap), l2_norm(mert), l2_norm(vgg)], axis=1))

# 2. Lyric Representation (2048-D: Harrier-0.6B + E5-Large)
harrier = np.load(DATA_DIR / 'embeddings' / 'lyric' / 'harrier_embeddings_1024d.npy')
e5_lyric = np.load(DATA_DIR / 'embeddings' / 'lyric' / 'multilingual_e5_large_1024d.npy')
has_lyrics = (np.linalg.norm(harrier, axis=1, keepdims=True) > 1e-6).astype(np.float32)
lyric_norm = l2_norm(np.concatenate([l2_norm(harrier), l2_norm(e5_lyric)], axis=1)) * has_lyrics

# 3. Unified Mood & Context Representation (83-D: Genre 40% + Spotify 30% + Temporal 15% + Vocal DSP 15%)
# Emotion 36-D omitted per LOGO ablation due to non-English zero-padding disparity
spotify_11d = np.load(DATA_DIR / 'embeddings' / 'metadata' / 'spotify_audio_11d.npy')
vocal_12d   = np.load(DATA_DIR / 'embeddings' / 'metadata' / 'vocal_dsp_12d.npy')
genre_50d   = np.load(DATA_DIR / 'embeddings' / 'metadata' / 'genre_hybrid_50d.npy')
temporal_10d = np.load(DATA_DIR / 'embeddings' / 'metadata' / 'temporal_collab_10d.npy')

mood_blocks = [
    np.sqrt(0.40) * l2_norm(genre_50d),
    np.sqrt(0.30) * l2_norm(spotify_11d),
    np.sqrt(0.15) * l2_norm(temporal_10d),
    np.sqrt(0.15) * l2_norm(vocal_12d)
]
mood_norm = l2_norm(np.concatenate(mood_blocks, axis=1))

# 4. Genre Style Representation (50-D: 17-D Main + 17-D Sub Rollup + 16-D Latent SVD)
genre_norm = l2_norm(genre_50d)

# 5. Temporal & Collaboration Context (10-D)
temporal_norm = l2_norm(temporal_10d)

print(f"Loaded representations for {len(songs):,} songs:")
print(f"  • Audio Embedding Dimension:          {audio_norm.shape[1]}-D")
print(f"  • Lyric Embedding Dimension:          {lyric_norm.shape[1]}-D")
print(f"  • Unified Mood & Context Dimension:   {mood_norm.shape[1]}-D")
print(f"  • Genre Hybrid Dimension:             {genre_norm.shape[1]}-D")
print(f"  • Temporal & Context Dimension:       {temporal_norm.shape[1]}-D")
"""))

# Section 2: Fast-Path Top-250 Pre-computed Graph Query
cells.append(nbf.v4.new_markdown_cell("""## 2. Fast-Path: Querying Pre-Computed Top-250 Matrices
For latency-critical applications, query the static Top-250 Parquet graphs directly ($O(1)$ sub-millisecond lookup).
"""))
cells.append(nbf.v4.new_code_cell("""# Load pre-computed graph files
knn_combined = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_combined_top250.parquet')
knn_mood = pd.read_parquet(DATA_DIR / 'similarity' / 'knn_mood_top250.parquet')

seed_idx = 42
seed_title = songs.iloc[seed_idx]['track_name']
seed_artist = songs.iloc[seed_idx]['artist_names']

print(f"Seed Track [{seed_idx}]: '{seed_title}' by {seed_artist}\\n")

print("Top 3 Instant Recommendations from Pre-computed Combined Top-250 Graph:")
for nb_idx, sim in zip(knn_combined.iloc[seed_idx]['top250_neighbor_indices'][:3], knn_combined.iloc[seed_idx]['top250_similarities'][:3]):
    print(f"  - '{songs.iloc[nb_idx]['track_name']}' by {songs.iloc[nb_idx]['artist_names']} (Similarity: {sim:.3f})")
"""))

# Section 3: Dynamic Multi-Modal Recommendation Engine
cells.append(nbf.v4.new_markdown_cell("""## 3. Dynamic Multi-Modal Recommendation Function
Computes weighted cosine similarities across all 5 representation pillars with contextual guardrails:

$$\\text{Sim}_{\\text{Total}} = w_a \\cdot S_{\\text{audio}} + w_l \\cdot S_{\\text{lyric}} + w_m \\cdot S_{\\text{mood}} + w_g \\cdot S_{\\text{genre}} + w_t \\cdot S_{\\text{temporal}}$$
"""))
cells.append(nbf.v4.new_code_cell("""def recommend_songs(
    seed_idx: int,
    top_k: int = 5,
    audio_weight: float = 0.38,
    lyric_weight: float = 0.35,
    mood_weight: float = 0.15,
    genre_weight: float = 0.08,
    temporal_weight: float = 0.04,
    same_language_only: bool = False,
    same_decade_only: bool = False,
    penalize_same_artist: bool = True
):
    \"\"\"
    Recommends top_k songs based on weighted multimodal similarity.
    
    Instrumental Fallback Behavior:
    If the seed track has no lyrics (has_lyrics == 0), s_lyric falls back to s_audio.
    This intentionally re-allocates the lyric weight to acoustic similarity, preventing
    instrumental songs from being penalized by zero-vector dot products.
    \"\"\"
    s_audio    = audio_norm @ audio_norm[seed_idx]
    
    # Instrumental fallback: if seed has no lyrics, route lyric weight to audio
    if has_lyrics[seed_idx, 0] > 0:
        s_lyric = lyric_norm @ lyric_norm[seed_idx]
    else:
        s_lyric = s_audio  # Fallback to acoustic similarity
        
    s_mood     = mood_norm @ mood_norm[seed_idx]
    s_genre    = genre_norm @ genre_norm[seed_idx]
    s_temporal = temporal_norm @ temporal_norm[seed_idx]
    
    total_sim = (
        audio_weight * s_audio +
        lyric_weight * s_lyric +
        mood_weight  * s_mood +
        genre_weight * s_genre +
        temporal_weight * s_temporal
    )
    
    # Contextual Guard 1: Language filter
    if same_language_only:
        seed_lang = lang_id.iloc[seed_idx]['primary_language']
        total_sim[lang_id['primary_language'] != seed_lang] = -1e9
        
    # Contextual Guard 2: Decade filter
    if same_decade_only:
        seed_decade = derived.iloc[seed_idx]['release_decade']
        total_sim[derived['release_decade'] != seed_decade] = -1e9
        
    # Contextual Guard 3: Artist diversity penalty
    if penalize_same_artist:
        seed_artists = set(str(songs.iloc[seed_idx]['artist_names']).lower().split(','))
        for idx in range(len(songs)):
            if idx == seed_idx:
                continue
            cand_artists = set(str(songs.iloc[idx]['artist_names']).lower().split(','))
            if bool(seed_artists & cand_artists):
                total_sim[idx] *= 0.85  # Apply 15% penalty to foster musical discovery
                
    # Mask self
    total_sim[seed_idx] = -1e9
    
    top_indices = np.argsort(total_sim)[-top_k:][::-1]
    
    results = songs.iloc[top_indices][['track_name', 'artist_names', 'popularity', 'main_genres', 'release_date']].copy()
    results['similarity_score'] = np.round(total_sim[top_indices], 4)
    results['audio_sim'] = np.round(s_audio[top_indices], 4)
    results['mood_sim'] = np.round(s_mood[top_indices], 4)
    results['genre_sim'] = np.round(s_genre[top_indices], 4)
    
    return results

# Test recommendation on Song 0
seed_idx = 0
print(f"--- Recommendations for Seed Track: '{songs.iloc[seed_idx]['track_name']}' by {songs.iloc[seed_idx]['artist_names']} ---")
display(recommend_songs(seed_idx, top_k=5))
"""))

# Section 4: Comparative Exploration
cells.append(nbf.v4.new_markdown_cell("""## 4. Comparing Sonic Feel vs. Lyrical Story vs. Unified Mood & Context
Notice how shifting the pillar weights changes the recommendations:
- **Sonic Vibe Focus ($w_a=0.8$):** Matches beat, tempo, instruments, and production texture.
- **Lyrical Narrative Focus ($w_l=0.8$):** Matches poetic themes, metaphors, and storytelling regardless of tempo.
- **Unified Mood & Context Focus ($w_m=0.8$):** Matches continuous vibe, vocal presence, genre identity, and temporal era.
"""))
cells.append(nbf.v4.new_code_cell("""seed_idx = 15
seed_title = songs.iloc[seed_idx]['track_name']
seed_artist = songs.iloc[seed_idx]['artist_names']
print(f"Seed Track [{seed_idx}]: '{seed_title}' by {seed_artist}\\n")

print("1. Pure Sonic Vibe Focus (audio_weight = 0.8):")
display(recommend_songs(seed_idx, top_k=3, audio_weight=0.8, lyric_weight=0.05, mood_weight=0.05, genre_weight=0.10)[['track_name', 'artist_names', 'main_genres', 'similarity_score']])

print("\\n 2. Pure Lyrical Narrative Focus (lyric_weight = 0.8):")
display(recommend_songs(seed_idx, top_k=3, audio_weight=0.05, lyric_weight=0.8, mood_weight=0.05, genre_weight=0.10)[['track_name', 'artist_names', 'main_genres', 'similarity_score']])

print("\\n 3. Pure Unified Mood & Context Focus (mood_weight = 0.8):")
display(recommend_songs(seed_idx, top_k=3, audio_weight=0.05, lyric_weight=0.05, mood_weight=0.8, genre_weight=0.10)[['track_name', 'artist_names', 'main_genres', 'similarity_score']])
"""))

# Section 5: Visual Playlist Walk on 2D UMAP Space
cells.append(nbf.v4.new_markdown_cell("""## 5. Visualizing a Playlist Journey on 2D Multimodal Space (`umap_2d_combined.parquet`)
Generate a smooth 5-track playlist trajectory and plot its journey path across the 2D multimodal latent space.

> **Interpretation Note**: The 2D coordinates in `similarity/umap_2d_combined.parquet` are qualitative projections for visualizing transitions and clustering.
"""))
cells.append(nbf.v4.new_code_cell("""umap_comb = pd.read_parquet(DATA_DIR / 'similarity' / 'umap_2d_combined.parquet')

def generate_playlist(start_seed_idx: int, playlist_length: int = 5):
    current = start_seed_idx
    visited = {current}
    playlist_indices = [current]
    
    for _ in range(playlist_length - 1):
        recs = recommend_songs(current, top_k=15, penalize_same_artist=True)
        for candidate_idx in recs.index:
            if candidate_idx not in visited:
                playlist_indices.append(candidate_idx)
                visited.add(candidate_idx)
                current = candidate_idx
                break
                
    return playlist_indices

playlist = generate_playlist(start_seed_idx=25, playlist_length=5)

plt.figure(figsize=(10, 6.5))
plt.scatter(umap_comb['proj_x'], umap_comb['proj_y'], c='lightgray', alpha=0.2, s=6, label='Global 10k Song Library')

px = umap_comb.iloc[playlist]['proj_x'].values
py = umap_comb.iloc[playlist]['proj_y'].values

plt.plot(px, py, color='crimson', linestyle='--', linewidth=2, zorder=4)
plt.scatter(px, py, color='red', s=80, edgecolors='black', zorder=5, label='Playlist Trajectory')

for step, (idx, x, y) in enumerate(zip(playlist, px, py)):
    title = songs.iloc[idx]['track_name'][:20]
    artist = songs.iloc[idx]['artist_names'][:15]
    plt.annotate(
        f"{step+1}. {title}\\n({artist})",
        (x, y),
        xytext=(x + 2, y + 2),
        fontsize=8,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.85),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
        zorder=6
    )

plt.title("5-Track Playlist Trajectory on 2D Multimodal Latent Space", fontsize=12, fontweight='bold')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
"""))

nb['cells'] = cells

out_file = Path("notebooks/03_content_based_recommendation_engine.ipynb")
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
