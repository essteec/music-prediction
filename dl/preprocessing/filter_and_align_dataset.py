"""
Dataset Filtering Script
Filters songs.csv to only include perfectly extracted, clean songs.
1. Finds intersection of IDs in all 4 audio embedding files.
2. Removes any IDs present in the failed extractions summary.
3. Overwrites songs.csv (with backup).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("DATASET PRUNING (THE SURGERY)")
    print("=" * 80)
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    data_dir = REPO_ROOT / 'data'
    processed_dir = data_dir / 'processed'
    embeddings_dir = data_dir / 'embeddings'
    audio_emb_dir = embeddings_dir / 'audio'
    
    # 1. Load the 4 audio ID arrays
    audio_models = ['vggish_embeddings_128d', 'mel_stats_embeddings_512d', 'mert_embeddings_768d', 'panns_embeddings_2048d']
    
    id_sets = []
    for model in audio_models:
        ids_path = audio_emb_dir / f"{model}_ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing ID file: {ids_path}")
        ids_array = np.load(ids_path)
        id_set = set([str(x) for x in ids_array])
        id_sets.append(id_set)
        print(f"Loaded {len(id_set):,} IDs from {model}")
    
    # 2. Find intersection
    intersection_ids = set.intersection(*id_sets)
    print(f"\nIntersection size (successful across all 4 models): {len(intersection_ids):,}")
    
    # 3. Exclude failures
    failed_csv_path = embeddings_dir / 'failed_extractions_summary.csv'
    if not failed_csv_path.exists():
        print("WARN: failed_extractions_summary.csv not found. No explicit failures removed.")
        failed_ids = set()
    else:
        failed_df = pd.read_csv(failed_csv_path)
        failed_ids = set(failed_df['spotify_id'].astype(str).tolist())
        print(f"Loaded {len(failed_ids):,} failed IDs from summary.")
    
    clean_ids = intersection_ids - failed_ids
    print(f"Clean IDs after removing failures: {len(clean_ids):,}")
    
    # 4. Process songs.csv
    songs_path = processed_dir / 'songs.csv'
    backup_path = processed_dir / 'songs_backup.csv'
    
    if not songs_path.exists():
        raise FileNotFoundError(f"Missing main dataset: {songs_path}")
        
    print(f"\nLoading {songs_path}...")
    df = pd.read_csv(songs_path)
    print(f"Original songs.csv size: {len(df):,}")
    
    # Backup
    if not backup_path.exists():
        print(f"Backing up original songs.csv to {backup_path}")
        df.to_csv(backup_path, index=False)
    else:
        print(f"Backup already exists at {backup_path}, skipping backup.")
        
    # Filter
    print("Filtering...")
    # Make sure we convert ids to string to match correctly
    df['id_str'] = df['id'].astype(str)
    filtered_df = df[df['id_str'].isin(clean_ids)].copy()
    filtered_df = filtered_df.drop(columns=['id_str'])
    
    print(f"Filtered songs.csv size: {len(filtered_df):,}")
    
    # 5. Save
    print(f"Overwriting {songs_path}...")
    filtered_df.to_csv(songs_path, index=False)
    
    print("\n✅ Surgery complete. The dataset is now ready for splitting.")
    print("Next steps:")
    print("1. python ml/preprocessing/data_splitting.py")
    print("2. python ml/preprocessing/run_preprocessing.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
