"""
Filter NPZ Script
Prunes the raw audio embedding .npz files perfectly to match the master songs.csv.
Ensures every .npz file has exactly the same rows (e.g., 491,632) in the exact same order as songs.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc

def main():
    print("=" * 80)
    print("FILTERING RAW NPZ FILES FOR KAGGLE PUBLICATION")
    print("=" * 80)
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    processed_dir = REPO_ROOT / 'data' / 'processed'
    
    songs_path = processed_dir / 'songs.csv'
    if not songs_path.exists():
        print(f"ERROR: {songs_path} not found.")
        return
        
    print(f"Loading master IDs from {songs_path}...")
    df = pd.read_csv(songs_path, usecols=['id'])
    master_ids = df['id'].astype(str).values
    n_master = len(master_ids)
    print(f"Master songs.csv size: {n_master:,}")
    
    models = {
        'vggish': 'vggish_embeddings',
        'mel_stats': 'mel_stats_embeddings',
        'mert': 'mert_embeddings',
        'panns': 'panns_embeddings'
    }
    
    for model_name, prefix in models.items():
        print("\n" + "-" * 80)
        print(f"Processing: {prefix}.npz")
        print("-" * 80)
        
        npz_path = processed_dir / f"{prefix}.npz"
        if not npz_path.exists():
            print(f"  WARN: {npz_path} not found. Skipping.")
            continue
            
        print("  Loading raw .npz...")
        with np.load(npz_path) as data:
            raw_ids = data['id'].astype(str)
            raw_features = data['features']
            
        print(f"  Raw shape: {raw_features.shape}")
        dim = raw_features.shape[1]
        
        # Create ID to index mapping
        print("  Mapping IDs...")
        id_to_idx = {id_str: idx for idx, id_str in enumerate(raw_ids)}
        
        # Create perfectly aligned array
        aligned_features = np.zeros((n_master, dim), dtype=np.float32)
        aligned_ids = np.empty(n_master, dtype=raw_ids.dtype)
        
        missing_count = 0
        for i, m_id in enumerate(master_ids):
            aligned_ids[i] = m_id
            if m_id in id_to_idx:
                aligned_features[i] = raw_features[id_to_idx[m_id]]
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"  ❌ WARNING: {missing_count} IDs missing from raw npz. Padded with zeros.")
        else:
            print(f"  ✅ Perfect alignment achieved.")
            
        print(f"  Saving filtered {prefix}.npz (Shape: {aligned_features.shape})...")
        np.savez_compressed(
            npz_path, 
            id=aligned_ids, 
            features=aligned_features
        )
        
        print(f"  Saved successfully.")
        
        # Cleanup
        del raw_ids, raw_features, aligned_features, aligned_ids, id_to_idx
        gc.collect()

    print("\n" + "=" * 80)
    print("✅ NPZ FILTERING COMPLETE")
    print("All .npz files are now exactly 1-to-1 perfectly aligned with songs.csv.")
    print("=" * 80)

if __name__ == "__main__":
    main()
