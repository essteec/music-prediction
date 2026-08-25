"""
Generates notebooks/03_content_based_recommendation_engine.ipynb
Simple Content-Based Multi-Modal Recommendation & Playlist Generator (Option B).
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🎧 Simple Content-Based Multi-Modal Music Recommender
### Blending Audio Acoustics & Lyric Themes to Recommend Similar Songs

This notebook builds a simple and practical **content-based recommendation engine** that combines:
1. **Audio Similarity** (using LAION-CLAP zero-shot acoustic representations)
2. **Lyric Similarity** (using Multilingual-E5-Large semantic representations)
3. **Smart Duplicate Filtering** (removing near-identical remixes/versions of the same artist)
"""))

# Section 1: Environment & Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Data Loading"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np

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
dsp = pd.read_parquet(DATA_DIR / 'features' / 'audio' / 'dsp_librosa.parquet')
emotions = pd.read_parquet(DATA_DIR / 'features' / 'lyric' / 'go_emotions.parquet')

# Load normalized embeddings
clap = np.load(DATA_DIR / 'embeddings' / 'audio' / 'clap_512d.npy')
lyrics_e5 = np.load(DATA_DIR / 'embeddings' / 'lyrics' / 'multilingual_e5_large_1024d.npy')

# L2 Normalize vectors for fast cosine similarity via dot product
clap_norm = clap / np.maximum(np.linalg.norm(clap, axis=1, keepdims=True), 1e-8)
lyrics_norm = lyrics_e5 / np.maximum(np.linalg.norm(lyrics_e5, axis=1, keepdims=True), 1e-8)

print(f"Loaded metadata and normalized embeddings for {len(songs):,} songs.")
"""))

# Section 2: Building Simple Recommender Function
cells.append(nbf.v4.new_markdown_cell("""## 2. Multi-Modal Recommendation Function
We compute cosine similarity across both audio and lyrics, allowing a configurable weighting parameter:
$$\\text{Score} = \\alpha \\cdot \\text{Sim}_{\\text{audio}} + (1 - \\alpha) \\cdot \\text{Sim}_{\\text{lyric}}$$

We also apply a simple filter to exclude the seed song itself and redundant versions/remixes from the same primary artist.
"""))
cells.append(nbf.v4.new_code_cell("""def recommend_songs(seed_idx: int, top_k: int = 5, audio_weight: float = 0.6):
    \"\"\"
    Recommends top_k songs based on combined audio and lyric cosine similarity.
    \"\"\"
    seed_song = songs.iloc[seed_idx]
    seed_artist = str(seed_song['artist_names']).lower().split(',')[0].strip()
    seed_title = str(seed_song['track_name']).lower()
    
    # 1. Compute cosine similarities across full 10k corpus
    sim_audio = np.dot(clap_norm, clap_norm[seed_idx])
    sim_lyric = np.dot(lyrics_norm, lyrics_norm[seed_idx])
    
    # 2. Blend scores
    combined_scores = (audio_weight * sim_audio) + ((1.0 - audio_weight) * sim_lyric)
    
    # 3. Sort indices
    ranked_indices = np.argsort(combined_scores)[::-1]
    
    # 4. Filter duplicates (same title or same primary artist with variant title)
    recommendations = []
    for idx in ranked_indices:
        if idx == seed_idx:
            continue
        
        cand_song = songs.iloc[idx]
        cand_artist = str(cand_song['artist_names']).lower().split(',')[0].strip()
        cand_title = str(cand_song['track_name']).lower()
        
        # Simple duplicate check
        if cand_artist == seed_artist and (seed_title in cand_title or cand_title in seed_title):
            continue
            
        recommendations.append({
            'row_idx': idx,
            'track_name': cand_song['track_name'],
            'artist': cand_song['artist_names'],
            'genre': cand_song['main_genres'],
            'audio_sim': round(float(sim_audio[idx]), 3),
            'lyric_sim': round(float(sim_lyric[idx]), 3),
            'combined_score': round(float(combined_scores[idx]), 3)
        })
        
        if len(recommendations) >= top_k:
            break
            
    return pd.DataFrame(recommendations)
"""))

# Section 3: Running Recommendations on Sample Tracks
cells.append(nbf.v4.new_markdown_cell("""## 3. Testing Recommendations on Sample Tracks"""))
cells.append(nbf.v4.new_code_cell("""# Example 1: Seed Track 15
seed_idx = 15
seed_info = songs.iloc[seed_idx]
print(f"🎵 SEED TRACK: '{seed_info['track_name']}' by {seed_info['artist_names']} (Genre: {seed_info['main_genres']})")
print("-" * 80)

# Get top 5 recommendations (60% audio vibe, 40% lyric theme)
recs = recommend_songs(seed_idx=seed_idx, top_k=5, audio_weight=0.6)
recs[['track_name', 'artist', 'genre', 'audio_sim', 'lyric_sim', 'combined_score']]
"""))

# Section 4: Testing Audio-Heavy vs Lyric-Heavy Blending
cells.append(nbf.v4.new_markdown_cell("""## 4. Comparing Audio-Focused vs Lyric-Focused Recommendations
Adjusting the `audio_weight` allows users to prioritize musical acoustic similarity versus lyrical storytelling.
"""))
cells.append(nbf.v4.new_code_cell("""# Audio-focused (85% audio, 15% lyric)
print("🔊 AUDIO-FOCUSED (Acoustic Match):")
recs_audio = recommend_songs(seed_idx=seed_idx, top_k=3, audio_weight=0.85)
display(recs_audio[['track_name', 'artist', 'genre', 'audio_sim', 'combined_score']])

# Lyric-focused (15% audio, 85% lyric)
print("\\n📖 LYRIC-FOCUSED (Thematic Match):")
recs_lyric = recommend_songs(seed_idx=seed_idx, top_k=3, audio_weight=0.15)
display(recs_lyric[['track_name', 'artist', 'genre', 'lyric_sim', 'combined_score']])
"""))

# Section 5: Generating a Smooth 5-Track Playlist
cells.append(nbf.v4.new_markdown_cell("""## 5. Mini Playlist Sequence Generator
Building a smooth sequential playlist where each track transitions seamlessly into the next.
"""))
cells.append(nbf.v4.new_code_cell("""def generate_playlist(start_seed_idx: int, playlist_length: int = 5):
    playlist = [start_seed_idx]
    visited = {start_seed_idx}
    
    current_idx = start_seed_idx
    for step in range(playlist_length - 1):
        candidates = recommend_songs(seed_idx=current_idx, top_k=15, audio_weight=0.7)
        # Pick top candidate not already in playlist
        next_idx = None
        for _, row in candidates.iterrows():
            cand_id = int(row['row_idx'])
            if cand_id not in visited:
                next_idx = cand_id
                break
        if next_idx is None:
            break
        playlist.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
        
    playlist_df = songs.iloc[playlist][['track_name', 'artist_names', 'main_genres', 'release_date']].copy().reset_index(drop=True)
    playlist_df.index = [f"Track {i+1}" for i in range(len(playlist_df))]
    return playlist_df

print("🎶 Generated Smooth Mini-Playlist:")
generate_playlist(start_seed_idx=25, playlist_length=5)
"""))

nb['cells'] = cells

out_file = Path("notebooks/03_content_based_recommendation_engine.ipynb")
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook: {out_file}")
