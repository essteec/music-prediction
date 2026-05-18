import numpy as np
from pathlib import Path
import gc

def pack_to_npz():
    print("=" * 80)
    print("PACKING EMBEDDINGS TO NPZ FOR KAGGLE")
    print("=" * 80)
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    audio_emb_dir = REPO_ROOT / 'data' / 'embeddings' / 'audio'
    processed_dir = REPO_ROOT / 'data' / 'processed'
    
    models = {
        'vggish': 'vggish_embeddings_128d',
        'mel_stats': 'mel_stats_embeddings_512d',
        'mert': 'mert_embeddings_768d',
        'panns': 'panns_embeddings_2048d'
    }

    for model_name, prefix in models.items():
        print(f"\nProcessing {model_name}...")
        data_path = audio_emb_dir / f"{prefix}.npy"
        ids_path = audio_emb_dir / f"{prefix}_ids.npy"
        out_path = processed_dir / f"{model_name}_embeddings.npz"

        if not data_path.exists() or not ids_path.exists():
            print(f"  WARN: Missing files for {model_name}. Skipping.")
            continue

        print(f"  Loading {data_path.name}...")
        # Load data. For large files we just read and stream it to save if possible, 
        # but np.savez needs arrays in memory or file handles.
        features = np.load(data_path)
        
        print(f"  Loading {ids_path.name}...")
        ids = np.load(ids_path).astype(str)

        print(f"  Saving to {out_path.name}...")
        # Compress to save Kaggle storage quota
        np.savez_compressed(out_path, id=ids, features=features)
        
        print(f"  Saved {features.shape} shape features.")
        
        del features, ids
        gc.collect()

    print("\n" + "=" * 80)
    print("✅ PACKING COMPLETE")
    print("Files saved in data/processed/. Ready for Kaggle upload alongside songs.csv.")
    print("Usage: data = np.load('model_embeddings.npz'); ids = data['id']; feats = data['features']")
    print("=" * 80)

if __name__ == "__main__":
    pack_to_npz()
