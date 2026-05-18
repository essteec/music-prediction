"""
Audio Embeddings Alignment Script
Aligns the raw audio embedding .npy files exactly with the new artist-aware splits.
Reads train.csv, val.csv, test.csv and creates X_{split}_{model}.npy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc

def main():
    print("=" * 80)
    print("AUDIO EMBEDDINGS RE-ALIGNMENT")
    print("=" * 80)
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    processed_dir = REPO_ROOT / 'data' / 'processed'
    audio_emb_dir = REPO_ROOT / 'data' / 'embeddings' / 'audio'
    
    splits = ['train', 'val', 'test']
    split_ids = {}
    
    # 1. Load the IDs for each split
    for split in splits:
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Did you run data_splitting.py?")
            return
            
        print(f"Loading IDs for {split} split...")
        df = pd.read_csv(csv_path, usecols=['id'])
        split_ids[split] = df['id'].astype(str).values
        print(f"  {split} size: {len(split_ids[split]):,}")
        
    models = {
        'vggish': 'vggish_embeddings_128d',
        'mel_stats': 'mel_stats_embeddings_512d',
        'mert': 'mert_embeddings_768d',
        'panns': 'panns_embeddings_2048d'
    }
    
    # 2. Process each model
    for model_name, prefix in models.items():
        print("\n" + "-" * 80)
        print(f"Processing model: {model_name.upper()}")
        print("-" * 80)
        
        data_path = audio_emb_dir / f"{prefix}.npy"
        ids_path = audio_emb_dir / f"{prefix}_ids.npy"
        
        if not data_path.exists() or not ids_path.exists():
            print(f"  WARN: Missing files for {model_name}. Skipping.")
            continue
            
        print("  Loading raw embeddings...")
        raw_ids = np.load(ids_path).astype(str)
        
        # Load data using mmap to save memory
        raw_data = np.load(data_path, mmap_mode='r')
        dim = raw_data.shape[1]
        
        print(f"  Raw shape: {raw_data.shape}")
        
        # Create ID to index mapping for fast lookup
        id_to_idx = {id_str: idx for idx, id_str in enumerate(raw_ids)}
        
        for split in splits:
            ids = split_ids[split]
            n_samples = len(ids)
            print(f"\n  Aligning {split} split ({n_samples:,} samples)...")
            
            # Create completely aligned array in memory
            aligned_data = np.zeros((n_samples, dim), dtype=np.float32)
            
            missing_count = 0
            for i, split_id in enumerate(ids):
                if split_id in id_to_idx:
                    aligned_data[i] = raw_data[id_to_idx[split_id]]
                else:
                    missing_count += 1
            
            if missing_count > 0:
                print(f"    WARNING: {missing_count} IDs were missing. Filled with zeros.")
            else:
                print(f"    Perfect alignment achieved.")
                
            out_file = audio_emb_dir / f"X_{split}_{model_name}.npy"
            print(f"    Saving to {out_file}...")
            np.save(out_file, aligned_data)
            
        # Free memory before next model
        del raw_data, raw_ids, id_to_idx
        gc.collect()

    print("\n" + "=" * 80)
    print("✅ ALIGNMENT COMPLETE")
    print("The audio embeddings are now perfectly mapped 1-to-1 with the splits.")
    print("=" * 80)

if __name__ == "__main__":
    main()
