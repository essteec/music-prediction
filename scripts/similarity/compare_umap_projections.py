"""
Compare Top-100 Combined kNN: Old (Unweighted Equal Split) vs. New (Weighted 73% Neural / 27% Context).

Usage:
  python scripts/similarity/compare_umap_projections.py --row_idx 42
  python scripts/similarity/compare_umap_projections.py --row_idx 0 -n 20
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
SIM_DIR = DATA_DIR / "similarity"
OLD_KNN_PATH = SIM_DIR / "knn_combined_top100 copy.parquet"
NEW_KNN_PATH = SIM_DIR / "knn_combined_top250.parquet"
if not NEW_KNN_PATH.exists():
    NEW_KNN_PATH = SIM_DIR / "knn_combined_top100.parquet"

def truncate(text: str, max_len: int) -> str:
    s = str(text) if pd.notna(text) else ""
    return s if len(s) <= max_len else s[:max_len - 3] + "..."

def main():
    parser = argparse.ArgumentParser(description="Compare Old vs. New Top-250 Combined kNN for a seed track.")
    parser.add_argument("--row_idx", type=int, required=True, help="Row index of the seed track (0 to 9999).")
    parser.add_argument("-n", "--top_n", type=int, default=100, help="Number of neighbors to display (1 to 250, default: 100).")
    args = parser.parse_args()

    seed_idx = args.row_idx
    if seed_idx < 0 or seed_idx >= 10000:
        raise ValueError(f"row_idx must be between 0 and 9999, got {seed_idx}")
    top_n = max(1, min(250, args.top_n))

    if not OLD_KNN_PATH.exists():
        raise FileNotFoundError(f"Baseline file not found: {OLD_KNN_PATH}")
    if not NEW_KNN_PATH.exists():
        raise FileNotFoundError(f"Updated file not found: {NEW_KNN_PATH}")

    # Load data
    songs = pd.read_csv(SONGS_CSV)
    df_old = pd.read_parquet(OLD_KNN_PATH)
    df_new = pd.read_parquet(NEW_KNN_PATH)

    seed_song = songs.iloc[seed_idx]
    title = seed_song['track_name']
    artist = seed_song['artist_names']
    genre = seed_song['main_genres']
    year = str(seed_song['release_date'])[:4]

    old_row = df_old.iloc[seed_idx]
    new_row = df_new.iloc[seed_idx]

    old_col_idx = 'top100_neighbor_indices' if 'top100_neighbor_indices' in old_row else 'top250_neighbor_indices'
    old_col_sim = 'top100_similarities' if 'top100_similarities' in old_row else 'top250_similarities'

    new_col_idx = 'top250_neighbor_indices' if 'top250_neighbor_indices' in new_row else 'top100_neighbor_indices'
    new_col_sim = 'top250_similarities' if 'top250_similarities' in new_row else 'top100_similarities'

    old_top_idx = list(old_row[old_col_idx])[:top_n]
    old_sims = list(old_row[old_col_sim])[:top_n]

    new_top_idx = list(new_row[new_col_idx])[:top_n]
    new_sims = list(new_row[new_col_sim])[:top_n]

    overlap = set(old_top_idx) & set(new_top_idx)
    overlap_pct = (len(overlap) / top_n) * 100.0

    # Header display
    print("\n" + "=" * 125)
    print(f"TOP-100 COMBINED KNN COMPARISON (Old Unweighted vs. New 73% Neural / 27% Context)")
    print("=" * 125)
    print(f"Seed Track [{seed_idx}]: \"{title}\" by {artist}")
    print(f"Genre: {genre} | Release Year: {year}")
    print(f"Top-{top_n} Shared Neighbors Overlap: {len(overlap)}/{top_n} ({overlap_pct:.1f}%)")
    print("=" * 125)

    header = f"{'#':<4} | {'OLD COMBINED (Equal Unweighted Split)':<54} | {'Sim':<6} | {'NEW COMBINED (73% Neural / 27% Context)':<54} | {'Sim':<6}"
    print(header)
    print("-" * 125)

    for r in range(top_n):
        # Old neighbor
        idx_o = old_top_idx[r]
        s_o = songs.iloc[idx_o]
        tag_o = " *" if idx_o in overlap else ""
        text_o = f"{s_o['track_name']} - {s_o['artist_names']} [{s_o['main_genres']}]{tag_o}"
        
        # New neighbor
        idx_n = new_top_idx[r]
        s_n = songs.iloc[idx_n]
        tag_n = " *" if idx_n in overlap else ""
        text_n = f"{s_n['track_name']} - {s_n['artist_names']} [{s_n['main_genres']}]{tag_n}"

        print(f"{r+1:<4} | {truncate(text_o, 54):<54} | {old_sims[r]:>6.3f} | {truncate(text_n, 54):<54} | {new_sims[r]:>6.3f}")

    print("-" * 125)
    print("(*) Shared neighbor in both Top-N sets.")
    print("=" * 125 + "\n")

if __name__ == "__main__":
    main()
