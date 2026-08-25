"""
Metadata Derived Features Extraction Script.
Derives structural, temporal, and artist collaboration features from songs.csv and artists.csv.
Outputs: data/features/metadata/derived.parquet
"""

import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
OUTPUT_DIR = DATA_DIR / "features" / "metadata"
OUTPUT_FILE = OUTPUT_DIR / "derived.parquet"

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def parse_year(val):
    if not isinstance(val, str) or not val.strip():
        return np.nan
    m = re.search(r'(\d{4})', val)
    return int(m.group(1)) if m else np.nan

def parse_genre_count(val):
    if not isinstance(val, str) or not val.strip():
        return 0
    try:
        if val.startswith('[') and val.endswith(']'):
            parsed = ast.literal_eval(val)
            return len(parsed) if isinstance(parsed, list) else 1
    except Exception:
        pass
    return len([g for g in val.split(',') if g.strip()])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    years = [parse_year(str(d)) for d in df['release_date']]
    years_arr = np.array(years, dtype=np.float32)
    decades = (np.floor(years_arr / 10.0) * 10).astype(np.float32)

    # Artist count & collaboration flag
    def count_artists(a_str):
        if not isinstance(a_str, str) or not a_str.strip():
            return 1
        return len([a for a in re.split(r'[,|&]', a_str) if a.strip()])

    n_artists_arr = np.array([count_artists(a) for a in df['artist_names']], dtype=np.int32)
    is_collab = n_artists_arr > 1

    # Genre counts
    genre_counts = np.array([parse_genre_count(g) for g in df['artist_genres']], dtype=np.int32)

    # Log artist followers
    followers = df['total_artist_followers'].fillna(0).values.astype(np.float32)
    log_followers = np.log1p(np.maximum(0, followers))

    # Duration in minutes
    duration_ms = df['duration_ms'].fillna(0).values.astype(np.float32)
    duration_min = duration_ms / 60000.0

    # Key + Mode representation (e.g., 'C Major', 'A Minor')
    key_mode_strs = []
    for k, m in zip(df['key'], df['mode']):
        try:
            k_int = int(k)
            m_int = int(m)
            if 0 <= k_int < 12:
                mode_str = "Major" if m_int == 1 else "Minor"
                key_mode_strs.append(f"{PITCH_CLASSES[k_int]} {mode_str}")
            else:
                key_mode_strs.append("Unknown")
        except Exception:
            key_mode_strs.append("Unknown")

    # Tempo categories
    tempo_val = df['tempo'].fillna(120).values
    tempo_category = np.where(tempo_val < 95, 'slow', np.where(tempo_val > 130, 'fast', 'medium'))

    # Has lyrics flag
    has_lyrics = df['lyrics'].notna() & (df['lyrics'].str.strip() != '')

    derived_df = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'release_year': years_arr,
        'release_decade': decades,
        'is_2020s': decades == 2020,
        'is_2010s': decades == 2010,
        'is_2000s': decades == 2000,
        'is_pre_2000': (decades < 2000) & (~np.isnan(decades)),
        'n_artists': n_artists_arr,
        'is_collaboration': is_collab,
        'n_artist_genres': genre_counts,
        'log_total_artist_followers': log_followers,
        'duration_min': np.round(duration_min, 2),
        'key_mode_name': key_mode_strs,
        'tempo_category': tempo_category,
        'is_explicit': df['explicit'].fillna(False).astype(bool).values,
        'has_lyrics': has_lyrics.values
    })

    derived_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Metadata Derived features to: {OUTPUT_FILE}")
    print(f"Shape: {derived_df.shape}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
