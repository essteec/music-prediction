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


if __name__ == '__main__':
    # Test data loading
    print("Testing data loaders...")
    train_loader, val_loader, test_loader, scaler = load_data(batch_size=64)
    
    # Get one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X: {X_batch.shape}")
    print(f"  y: {y_batch.shape}")
    print("\n✓ Data loading test successful!")
