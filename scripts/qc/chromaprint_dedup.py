"""
Chromaprint Acoustic Fingerprinting and Duplicate Detection Script.
Generates Chromaprint fingerprints for all 10k tracks and identifies duplicate/near-duplicate tracks.
Outputs: data/features/qc/chromaprint_fingerprints.parquet
"""

import os
import json
import hashlib
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "processed" / "songs.csv"
AUDIO_DIR = DATA_DIR / "audio" / "pilot"
OUTPUT_DIR = DATA_DIR / "features" / "qc"
OUTPUT_FILE = OUTPUT_DIR / "chromaprint_fingerprints.parquet"

def get_chromaprint(filepath: str):
    if not os.path.exists(filepath):
        return None
    cmd = ['fpcalc', '-json', '-length', '120', filepath]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            fp_str = data.get('fingerprint', '')
            if fp_str:
                fp_hash = hashlib.sha256(fp_str.encode('utf-8')).hexdigest()[:16]
                return fp_str, fp_hash
        return None
    except Exception:
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading songs from {SONGS_CSV}...")
    df = pd.read_csv(SONGS_CSV)
    n_songs = len(df)

    file_paths = [str(AUDIO_DIR / f"{i:06d}_opus.webm") for i in range(n_songs)]

    print("Computing Chromaprint fingerprints in parallel (ProcessPoolExecutor)...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(get_chromaprint, file_paths, chunksize=50), total=n_songs))

    fingerprints = []
    hashes = []
    has_fp = []

    for r in results:
        if r is not None:
            fingerprints.append(r[0])
            hashes.append(r[1])
            has_fp.append(True)
        else:
            fingerprints.append("")
            hashes.append("")
            has_fp.append(False)

    df_out = pd.DataFrame({
        'row_idx': np.arange(n_songs, dtype=np.int32),
        'track_id': df['track_id'].values,
        'has_fingerprint': has_fp,
        'fingerprint_hash': hashes,
        'chromaprint': fingerprints
    })

    # Duplicate analysis based on identical fingerprint hash
    valid_hashes = [h for h in hashes if h]
    hash_counts = pd.Series(hashes).value_index if hasattr(pd.Series(hashes), 'value_index') else pd.Series([h for h in hashes if h]).value_counts()
    dup_hashes = set(hash_counts[hash_counts > 1].index)

    df_out['is_duplicate'] = df_out['fingerprint_hash'].isin(dup_hashes)

    df_out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved Chromaprint fingerprints parquet to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")

    total_valid = sum(has_fp)
    total_dups = df_out['is_duplicate'].sum()
    print(f"Valid fingerprints: {total_valid} / {n_songs} ({total_valid/n_songs*100:.1f}%)")
    print(f"Duplicate tracks found by exact hash: {total_dups} songs in {len(dup_hashes)} clusters")

if __name__ == "__main__":
    main()
