"""
Data loading utilities for PyTorch training.
Loads preprocessed .npy files from ml/features/ directory.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import random


def set_worker_seed(worker_id):
    """Set seed for DataLoader workers for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MusicDataset(Dataset):
    """PyTorch Dataset for music prediction."""
    
    def __init__(self, split='train', feature_dir='ml/features', target='all'):
        """
        Args:
            split: 'train', 'val', or 'test'
            feature_dir: Directory containing .npy files
            target: 'all' (returns all 4 targets) or specific target name
        """
        self.split = split
        self.feature_dir = feature_dir
        self.target = target
        
        # Load features (concatenate all 4 types)
        print(f"Loading {split} features...")
        X_audio = np.load(f'{feature_dir}/X_{split}_audio.npy')
        X_text = np.load(f'{feature_dir}/X_{split}_text_stats.npy')
        X_sentiment = np.load(f'{feature_dir}/X_{split}_sentiment.npy')
        X_embeddings = np.load(f'{feature_dir}/X_{split}_embeddings.npy')
        
        # Concatenate: 23 + 5 + 2 + 384 = 414 features
        self.X = np.concatenate([X_audio, X_text, X_sentiment, X_embeddings], axis=1)
        print(f"  Features shape: {self.X.shape}")
        
        # Load targets
        if target == 'all':
            # Stack all 4 targets: [valence, energy, danceability, popularity]
            y_valence = np.load(f'{feature_dir}/y_{split}_valence.npy')
            y_energy = np.load(f'{feature_dir}/y_{split}_energy.npy')
            y_dance = np.load(f'{feature_dir}/y_{split}_danceability.npy')
            y_pop = np.load(f'{feature_dir}/y_{split}_popularity.npy')
            self.y = np.stack([y_valence, y_energy, y_dance, y_pop], axis=1)
        else:
            self.y = np.load(f'{feature_dir}/y_{split}_{target}.npy').reshape(-1, 1)
        
        print(f"  Targets shape: {self.y.shape}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


def load_data(batch_size=256, feature_dir='ml/features', target='all', 
              normalize=True, num_workers=4):
    """
    Load train/val/test datasets with normalization.
    
    Args:
        batch_size: Batch size for DataLoader
        feature_dir: Directory containing .npy files
        target: 'all' or specific target name
        normalize: Whether to apply StandardScaler
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Load datasets
    train_dataset = MusicDataset('train', feature_dir, target)
    val_dataset = MusicDataset('val', feature_dir, target)
    test_dataset = MusicDataset('test', feature_dir, target)
    
    # Normalize features (fit on train, transform all)
    scaler = None
    if normalize:
        print("\nNormalizing features with StandardScaler...")
        scaler = StandardScaler()
        train_dataset.X = scaler.fit_transform(train_dataset.X)
        val_dataset.X = scaler.transform(val_dataset.X)
        test_dataset.X = scaler.transform(test_dataset.X)
        print("  ✓ Normalization complete")
    
    # Create DataLoaders (with reproducibility)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader, scaler


class MultiModalDataset(Dataset):
    """
    Dataset for multi-branch fusion models (Phase 4+).

    Returns each modality as a separate tensor so per-modality encoders
    receive only their own data — no cross-modal mixing at load time.

    Modality order (must match model forward signature):
        0: metadata   — concat(audio 23d, text_stats 5d, sentiment 2d) = 30d
        1: mpnet      — MPNet lyrics embeddings                          768d
        2: vggish     — VGGish audio embeddings                          128d
        3: mert       — MERT-v1-95M audio embeddings                     768d
        4: panns      — PANNs Cnn14 audio embeddings                    2048d
        5: mel_stats  — Mel spectrogram statistics                        512d
        6: targets    — [valence, energy, danceability, popularity]

    Zero-padding convention: songs whose audio extraction failed are stored
    as all-zeros in the aligned npy arrays. The gating networks in Exp B/C
    detect and mask these out automatically.

    Memory: large embedding files are memory-mapped (mmap_mode='r').
    Each __getitem__ copies one row — safe for multiprocessing workers.

    Args:
        split:      'train', 'val', or 'test'.
        feat_dir:   Directory containing the .npy feature files.
        scaler_dir: Optional path to directory containing per-modality
                    StandardScaler PKL files (generated by
                    dl/preprocessing/fit_modal_scalers.py). When provided,
                    each modality is standardised before being returned.
                    If None, raw values are returned (original behaviour).
    """

    def __init__(self, split: str, feat_dir: str = 'ml/features',
                 scaler_dir: str = None):
        print(f"  Loading {split} split (multi-modal)...")

        # ── Load scalers if provided ───────────────────────────────────────────
        self.scalers = {}
        if scaler_dir is not None:
            import pickle
            from pathlib import Path
            sd = Path(scaler_dir)
            for name in ['metadata', 'mpnet', 'vggish', 'mert', 'panns', 'mel_stats']:
                pkl = sd / f'modal_scaler_{name}.pkl'
                if pkl.exists():
                    with open(pkl, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
            print(f"    Scalers loaded: {list(self.scalers.keys())}")

        # ── Metadata branch (small — load fully into RAM) ──────────────────────
        audio     = np.load(f'{feat_dir}/X_{split}_audio.npy').astype(np.float32)
        text      = np.load(f'{feat_dir}/X_{split}_text_stats.npy').astype(np.float32)
        sentiment = np.load(f'{feat_dir}/X_{split}_sentiment.npy').astype(np.float32)
        meta_raw  = np.concatenate([audio, text, sentiment], axis=1)  # (N, 30)
        self.metadata = self._scale('metadata', meta_raw)

        # ── Embedding branches (large — memory-mapped then optionally scaled) ──
        # Note: scalers require an in-memory array, so we load fully when scaling.
        self.mpnet     = self._load_emb(f'{feat_dir}/X_{split}_mpnet.npy',     'mpnet')
        self.vggish    = self._load_emb(f'{feat_dir}/X_{split}_vggish.npy',    'vggish')
        self.mert      = self._load_emb(f'{feat_dir}/X_{split}_mert.npy',      'mert')
        self.panns     = self._load_emb(f'{feat_dir}/X_{split}_panns.npy',     'panns')
        self.mel_stats = self._load_emb(f'{feat_dir}/X_{split}_mel_stats.npy', 'mel_stats')

        # ── Targets ────────────────────────────────────────────────────────────
        self.targets = np.stack([
            np.load(f'{feat_dir}/y_{split}_valence.npy'),
            np.load(f'{feat_dir}/y_{split}_energy.npy'),
            np.load(f'{feat_dir}/y_{split}_danceability.npy'),
            np.load(f'{feat_dir}/y_{split}_popularity.npy'),
        ], axis=1).astype(np.float32)  # (N, 4)

        self.n = len(self.metadata)
        scaled_str = " [scaled]" if self.scalers else ""
        print(f"    Samples: {self.n:,}  |  metadata: {self.metadata.shape[1]}d  "
              f"|  mpnet: {self.mpnet.shape[1] if hasattr(self.mpnet, 'shape') else '?'}d  "
              f"|  mert: {self.mert.shape[1] if hasattr(self.mert, 'shape') else '?'}d  "
              f"|  panns: {self.panns.shape[1] if hasattr(self.panns, 'shape') else '?'}d{scaled_str}")

    def _scale(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Apply scaler if available, return float32 array.

        Supports two scaler formats:
          - dict {'mean': float, 'std': float} → global normalization
            (geometry-preserving, used from Exp D onwards)
          - sklearn StandardScaler → per-column normalization
            (legacy, not recommended for embeddings)
        """
        if name in self.scalers:
            scaler = self.scalers[name]
            if isinstance(scaler, dict):
                # Global mean/std — preserves embedding geometry
                arr = ((arr - scaler['mean']) / scaler['std']).astype(np.float32)
            else:
                # Legacy sklearn StandardScaler
                arr = scaler.transform(arr).astype(np.float32)
        return arr

    def _load_emb(self, path: str, name: str) -> np.ndarray:
        """Load embedding .npy file. If scaler present, loads fully into RAM
        for transform. Otherwise memory-maps for lower RAM usage."""
        if name in self.scalers:
            arr = np.load(path).astype(np.float32)
            # Preserve zero-padding mask BEFORE scaling (Bug #2 fix).
            # Absent modalities (audio extraction failed) are stored as all-zeros.
            # Scaling transforms zeros to non-zero, breaking the presence mask
            # used by gating models (GatedFusionMLP, TaskGatedFusionMLP,
            # AttentionTaskGatedFusionMLP).
            zero_rows = arr.abs().sum(axis=1) == 0
            arr = self._scale(name, arr)
            arr[zero_rows] = 0.0
            return arr
        return np.load(path, mmap_mode='r')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # np.array() on a mmap slice creates an in-memory copy — required for
        # safe use across multiprocessing DataLoader workers.
        def _get(arr):
            if isinstance(arr, np.memmap):
                return torch.from_numpy(np.array(arr[idx], dtype=np.float32))
            return torch.from_numpy(arr[idx].copy())

        return (
            _get(self.metadata),
            _get(self.mpnet),
            _get(self.vggish),
            _get(self.mert),
            _get(self.panns),
            _get(self.mel_stats),
            torch.from_numpy(self.targets[idx].copy()),
        )


def load_multimodal_data(batch_size: int = 256, feat_dir: str = 'ml/features',
                         num_workers: int = 4, scaler_dir: str = None):
    """
    Create train/val/test DataLoaders for multi-modal fusion models.

    Args:
        batch_size:  Samples per batch.
        feat_dir:    Directory with .npy feature files.
        num_workers: DataLoader worker processes.
        scaler_dir:  Optional path to per-modality StandardScaler PKLs.
                     Pass 'ml/features/scalers' to enable scaling (Exp D+).

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\nLoading multi-modal datasets:")
    train_ds = MultiModalDataset('train', feat_dir, scaler_dir)
    val_ds   = MultiModalDataset('val',   feat_dir, scaler_dir)
    test_ds  = MultiModalDataset('test',  feat_dir, scaler_dir)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=set_worker_seed, generator=g,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=set_worker_seed,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=set_worker_seed,
        persistent_workers=(num_workers > 0),
    )

    print(f"\n✓ Multi-modal DataLoaders ready:")
    print(f"  Train: {len(train_loader):,} batches ({len(train_ds):,} samples)")
    print(f"  Val:   {len(val_loader):,} batches ({len(val_ds):,} samples)")
    print(f"  Test:  {len(test_loader):,} batches ({len(test_ds):,} samples)")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    print("Testing data loaders...")
    train_loader, val_loader, test_loader, scaler = load_data(batch_size=64)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X: {X_batch.shape}")
    print(f"  y: {y_batch.shape}")
    print("\n✓ Data loading test successful!")

    print("\nTesting multi-modal data loaders...")
    mm_train, mm_val, mm_test = load_multimodal_data(batch_size=64, num_workers=0)
    batch = next(iter(mm_train))
    names = ['metadata', 'mpnet', 'vggish', 'mert', 'panns', 'mel_stats', 'targets']
    for name, t in zip(names, batch):
        print(f"  {name}: {t.shape}")
    print("\n✓ Multi-modal data loading test successful!")
