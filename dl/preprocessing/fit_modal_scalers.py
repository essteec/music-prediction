"""
Fit and save per-modality scalers for the multi-modal DL pipeline.

Run once from project root:
    python dl/preprocessing/fit_modal_scalers.py

IMPORTANT: Uses GLOBAL mean/std normalization, NOT per-column StandardScaler.
StandardScaler destroys embedding geometry because:
  1. Per-column scaling rescales each dimension independently, breaking
     cosine similarity and vector norms that embeddings rely on.
  2. Near-zero-std columns (PANNs has 861 dead columns with std=0,
     MPNet has 4 with std≈0) cause division-by-zero → inf/NaN values.

Instead we compute a single (global_mean, global_std) per modality and
apply uniform scaling: x_scaled = (x - global_mean) / global_std.
This preserves relative distances and directions within the embedding space.

Saves one scaler dict per modality to ml/features/scalers/:
    modal_scaler_{name}.pkl  →  {'mean': float, 'std': float}

The MultiModalDataset applies these at load time.
"""

import pickle
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
FEAT_DIR   = REPO_ROOT / 'ml' / 'features'
SCALER_DIR = FEAT_DIR / 'scalers'
SCALER_DIR.mkdir(parents=True, exist_ok=True)

# Modality name → train .npy filename stem
# NOTE: 'metadata' is intentionally excluded.
# X_train_audio.npy, X_train_text_stats.npy, X_train_sentiment.npy are all
# already properly normalised by ml/preprocessing/run_preprocessing.py:
#   - audio:      PowerTransformer(Yeo-Johnson) + StandardScaler + OneHotEncoder
#   - text_stats: Log1p + StandardScaler
#   - sentiment:  StandardScaler
# We only need to scale the 5 raw neural embedding branches.
MODALITIES = {
    'mpnet':     'X_train_mpnet',
    'vggish':    'X_train_vggish',
    'mert':      'X_train_mert',
    'panns':     'X_train_panns',
    'mel_stats': 'X_train_mel_stats',
}


def main():
    print("=" * 60)
    print("Fitting per-modality GLOBAL scalers")
    print("(metadata excluded — already scaled by run_preprocessing.py)")
    print("=" * 60)
    print(f"Feature dir:  {FEAT_DIR}")
    print(f"Scaler dir:   {SCALER_DIR}")
    print()

    for name, stem in MODALITIES.items():
        print(f"  [{name}]", end='  ')
        arr = np.load(FEAT_DIR / f'{stem}.npy', mmap_mode='r')

        # Compute GLOBAL mean and std across all values (not per-column)
        # Use float64 for numerical stability on large arrays
        global_mean = float(np.mean(arr, dtype=np.float64))
        global_std  = float(np.std(arr, dtype=np.float64))

        # Safety: never divide by near-zero std
        if global_std < 1e-6:
            print(f"⚠ WARNING: global_std={global_std:.2e}, setting to 1.0")
            global_std = 1.0

        scaler_dict = {'mean': global_mean, 'std': global_std}
        path = SCALER_DIR / f'modal_scaler_{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(scaler_dict, f)

        # Verify: compute stats after scaling on a sample
        sample = arr[:5000].astype(np.float64)
        scaled_sample = (sample - global_mean) / global_std
        print(f"shape={arr.shape}  "
              f"raw(mean={global_mean:+.4f}, std={global_std:.4f})  →  "
              f"scaled(mean={scaled_sample.mean():.4f}, std={scaled_sample.std():.4f}, "
              f"max={np.abs(scaled_sample).max():.2f})  "
              f"→ saved {path.name}")

    print()
    print("✓ All scalers saved (global mean/std, geometry-preserving).")


if __name__ == '__main__':
    main()
