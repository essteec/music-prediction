"""
Kaggle Dataset Manifest & SHA-256 Checksum Generator.
Scans all extracted features, embeddings, splits, and similarity graphs.
Outputs:
- data/manifests/extraction_manifest.json
- data/manifests/checksums.json
- data/track_ids.npy (master 10k ID array)
"""

import os
import glob
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
MANIFESTS_DIR = DATA_DIR / "manifests"

def get_file_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SONGS_CSV)
    
    # 1. Master Track IDs
    track_ids = df['track_id'].values
    np.save(DATA_DIR / "track_ids.npy", track_ids)
    print(f"Saved master track_ids.npy ({len(track_ids)} IDs)")

    manifest = {
        'dataset_name': 'spotify-10k-music-features',
        'version': '1.0.0',
        'n_tracks': len(df),
        'created_at': pd.Timestamp.now().isoformat(),
        'files': []
    }
    
    checksums = {}

    all_files = []
    for sub in ['metadata', 'features', 'embeddings', 'similarity', 'splits', 'processed']:
        all_files.extend(glob.glob(str(DATA_DIR / sub / "**/*.*"), recursive=True))

    for fp_str in sorted(all_files):
        fp = Path(fp_str)
        if fp.suffix not in ('.parquet', '.npy', '.npz', '.json', '.csv') or 'checkpoint' in fp.name or 'pilot' in fp.parts:
            continue

        rel_path = str(fp.relative_to(DATA_DIR))
        size_bytes = fp.stat().st_size
        sha256 = get_file_sha256(str(fp))
        checksums[rel_path] = sha256

        file_meta = {
            'relative_path': rel_path,
            'size_kb': round(size_bytes / 1024, 2),
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'sha256': sha256
        }

        if fp.suffix == '.parquet':
            try:
                df_temp = pd.read_parquet(fp)
                file_meta['shape'] = list(df_temp.shape)
                file_meta['columns'] = df_temp.columns.tolist()
            except Exception:
                pass
        elif fp.suffix == '.npy':
            try:
                arr = np.load(fp)
                file_meta['shape'] = list(arr.shape)
                file_meta['dtype'] = str(arr.dtype)
            except Exception:
                pass
        elif fp.suffix == '.npz':
            try:
                d = np.load(fp)
                file_meta['keys'] = list(d.keys())
                file_meta['shapes'] = {k: list(d[k].shape) for k in d.keys()}
            except Exception:
                pass

        manifest['files'].append(file_meta)

    with open(MANIFESTS_DIR / "extraction_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {MANIFESTS_DIR / 'extraction_manifest.json'}")

    with open(MANIFESTS_DIR / "checksums.json", 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"Saved checksums: {MANIFESTS_DIR / 'checksums.json'}")
    print(f"Total packaged files tracked: {len(manifest['files'])}")

if __name__ == "__main__":
    main()
