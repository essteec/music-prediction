"""Text Statistics Processing Module

Extracts and scales text statistical features from lyrics.
Features:
- word_count
- unique_word_count
- unique_ratio
- avg_word_length
- char_count

Can be run standalone or as part of the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from pipeline_utils import (
    FEATURES_DIR,
    PROCESSED_DIR,
    check_if_step_needed,
    mark_step_complete,
)

TEXT_STAT_COLUMNS = [
    "word_count",
    "unique_word_count",
    "unique_ratio",
    "avg_word_length",
    "char_count",
]


def _extract_text_stats(lyrics: str) -> Dict[str, float]:
    """Extract basic statistical features from lyrics text."""
    if not isinstance(lyrics, str) or not lyrics.strip():
        return dict.fromkeys(TEXT_STAT_COLUMNS, 0.0)
    
    text = lyrics.strip()
    words = text.split()
    unique = {w.lower() for w in words}
    
    return {
        "word_count": len(words),
        "unique_word_count": len(unique),
        "unique_ratio": len(unique) / max(len(words), 1),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "char_count": len(text),
    }


def _load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def _compute_stats_for_split(lyrics_series: pd.Series) -> pd.DataFrame:
    """Compute text statistics for a split."""
    records = [_extract_text_stats(text) for text in lyrics_series]
    return pd.DataFrame(records, columns=TEXT_STAT_COLUMNS)


def process_text_statistics(verbose: bool = True) -> Dict[str, np.ndarray]:
    """Process text statistics with intelligent caching.
    
    Args:
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with keys 'X_train_text_stats', 'X_val_text_stats', 'X_test_text_stats'
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define input and output files
    input_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
    ]
    
    output_files = [
        FEATURES_DIR / "X_train_text_stats.npy",
        FEATURES_DIR / "X_val_text_stats.npy",
        FEATURES_DIR / "X_test_text_stats.npy",
        FEATURES_DIR / "text_stats_scaler.pkl",
    ]
    
    # Check if processing is needed
    if not check_if_step_needed("text_statistics", input_files, output_files):
        if verbose:
            print("✅ Text statistics are up-to-date, skipping processing")
        return {
            "X_train_text_stats": np.load(FEATURES_DIR / "X_train_text_stats.npy"),
            "X_val_text_stats": np.load(FEATURES_DIR / "X_val_text_stats.npy"),
            "X_test_text_stats": np.load(FEATURES_DIR / "X_test_text_stats.npy"),
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING TEXT STATISTICS")
        print("=" * 80)
    
    # Load data
    splits = _load_splits()
    
    if verbose:
        print(f"Train: {len(splits['train']):,} songs")
        print(f"Val:   {len(splits['val']):,} songs")
        print(f"Test:  {len(splits['test']):,} songs")
    
    # Extract statistics
    stats = {}
    for name, df in splits.items():
        if verbose:
            print(f"Extracting text statistics from {name} set...")
        stats[name] = _compute_stats_for_split(df["lyrics"])
    
    # Apply log transformation to count features
    log_cols = ["word_count", "unique_word_count", "char_count"]
    scaler = StandardScaler()
    
    # Transform training set and fit scaler
    stats_train_transformed = stats["train"].copy()
    stats_train_transformed[log_cols] = np.log1p(stats_train_transformed[log_cols])
    scaler.fit(stats_train_transformed)
    X_train = scaler.transform(stats_train_transformed)
    
    # Transform val and test
    stats_val_transformed = stats["val"].copy()
    stats_val_transformed[log_cols] = np.log1p(stats_val_transformed[log_cols])
    X_val = scaler.transform(stats_val_transformed)
    
    stats_test_transformed = stats["test"].copy()
    stats_test_transformed[log_cols] = np.log1p(stats_test_transformed[log_cols])
    X_test = scaler.transform(stats_test_transformed)
    
    if verbose:
        print(f"\nFeature matrix shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
    
    # Save arrays
    np.save(FEATURES_DIR / "X_train_text_stats.npy", X_train)
    np.save(FEATURES_DIR / "X_val_text_stats.npy", X_val)
    np.save(FEATURES_DIR / "X_test_text_stats.npy", X_test)
    
    # Save scaler
    joblib.dump(scaler, FEATURES_DIR / "text_stats_scaler.pkl")
    
    # Mark step complete
    metadata = {
        "n_features": X_train.shape[1],
        "features": TEXT_STAT_COLUMNS,
    }
    mark_step_complete("text_statistics", input_files, output_files, metadata)
    
    if verbose:
        print(f"✅ Text statistics saved to {FEATURES_DIR}")
    
    return {
        "X_train_text_stats": X_train,
        "X_val_text_stats": X_val,
        "X_test_text_stats": X_test,
    }


if __name__ == "__main__":
    process_text_statistics(verbose=True)
