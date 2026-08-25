"""
Artist-Grouped and Temporal Split Generator.
Creates:
1. data/splits/artist_grouped_5fold.parquet -> 5-fold GroupKFold by artist_id (zero artist leakage)
2. data/splits/temporal_split.parquet -> Train (pre-2023), Val (2023), Test (2024-2025)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
SPLITS_DIR = DATA_DIR / "splits"

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs metadata from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    # 1. 5-Fold GroupKFold by artist_id
    print("Generating 5-Fold GroupKFold by artist_id (zero artist leakage)...")
    # Primary artist id
    primary_artist = df['artist_ids'].fillna("unknown").apply(lambda x: str(x).split(',')[0].strip())
    
    gkf = GroupKFold(n_splits=5)
    fold_assignments = np.zeros(n_songs, dtype=np.int32)
    
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=primary_artist)):
        fold_assignments[val_idx] = fold

    df_folds = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'artist_id': primary_artist.values,
        'fold': fold_assignments
    })
    
    folds_file = SPLITS_DIR / "artist_grouped_5fold.parquet"
    df_folds.to_parquet(folds_file, index=False)
    print(f"Saved GroupKFold splits to: {folds_file}")
    print("Fold counts:\n", df_folds['fold'].value_counts().sort_index())

    # 2. Temporal Split (Release Date)
    print("\nGenerating Temporal Train/Val/Test Split...")
    years = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(2020).astype(int)
    
    split_col = np.where(years >= 2024, 'test', np.where(years == 2023, 'val', 'train'))
    
    df_temp = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'release_year': years.values,
        'split': split_col
    })
    
    temp_file = SPLITS_DIR / "temporal_split.parquet"
    df_temp.to_parquet(temp_file, index=False)
    print(f"Saved Temporal splits to: {temp_file}")
    print("Temporal split counts:\n", df_temp['split'].value_counts())

if __name__ == "__main__":
    main()
