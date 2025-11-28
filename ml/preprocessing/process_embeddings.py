"""Lyric Embeddings Processing Module

Extracts semantic embeddings from lyrics using sentence-transformers.
Model: all-MiniLM-L6-v2 (English-optimized, 384 dimensions)

Key Features:
- Semantic understanding beyond word frequency
- Compact 384-dimensional representation
- Batch processing for efficiency
- Disk caching to avoid recomputation

Performance:
- ~30-60 minutes for full dataset (700k songs)
- Embeddings cached for instant reuse

Can be run standalone or as part of the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from pipeline_utils import (
    FEATURES_DIR,
    PROCESSED_DIR,
    check_if_step_needed,
    mark_step_complete,
)


def _load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def _compute_embeddings_for_split(
    lyrics_series: pd.Series,
    model,
    batch_size: int = 64,
    verbose: bool = True
) -> np.ndarray:
    """Compute embeddings for a split with batch processing.
    
    Args:
        lyrics_series: Series of lyrics strings
        model: SentenceTransformer model
        batch_size: Number of lyrics to process at once
        verbose: Whether to show progress bar
        
    Returns:
        Array of shape (n_songs, 384)
    """
    # Prepare lyrics (handle missing/empty)
    lyrics_list = []
    for lyric in lyrics_series:
        if not isinstance(lyric, str) or not lyric.strip():
            lyrics_list.append("")  # Empty string will get zero embedding
        else:
            lyrics_list.append(lyric.strip())
    
    # Compute embeddings in batches
    if verbose:
        desc = f"Computing embeddings ({len(lyrics_list):,} songs)"
        iterator = tqdm(
            range(0, len(lyrics_list), batch_size),
            desc=desc,
            unit="batch"
        )
    else:
        iterator = range(0, len(lyrics_list), batch_size)
    
    embeddings_list = []
    for i in iterator:
        batch = lyrics_list[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings_list.append(batch_embeddings)
    
    return np.vstack(embeddings_list)


def process_embeddings(
    verbose: bool = True,
    batch_size: int = 64,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, np.ndarray]:
    """Process lyric embeddings with intelligent caching.
    
    This function computes semantic embeddings for lyrics using the
    all-MiniLM-L6-v2 model. Embeddings are cached to disk to avoid
    recomputation (which takes 30-60 minutes for the full dataset).
    
    Args:
        verbose: Whether to print progress information
        batch_size: Number of lyrics to process at once
        model_name: SentenceTransformer model identifier
        
    Returns:
        Dictionary with keys 'X_train_embeddings', 'X_val_embeddings', 'X_test_embeddings'
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define input and output files
    input_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
    ]
    
    output_files = [
        FEATURES_DIR / "X_train_embeddings.npy",
        FEATURES_DIR / "X_val_embeddings.npy",
        FEATURES_DIR / "X_test_embeddings.npy",
    ]
    
    # Check if processing is needed
    if not check_if_step_needed("embeddings", input_files, output_files):
        if verbose:
            print("✅ Embeddings are up-to-date, loading from cache")
        return {
            "X_train_embeddings": np.load(FEATURES_DIR / "X_train_embeddings.npy"),
            "X_val_embeddings": np.load(FEATURES_DIR / "X_val_embeddings.npy"),
            "X_test_embeddings": np.load(FEATURES_DIR / "X_test_embeddings.npy"),
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING LYRIC EMBEDDINGS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Batch size: {batch_size}")
        print("⏱️  This will take 30-60 minutes for full dataset (computed ONCE)")
        print()
    
    # Load sentence transformer model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
    
    if verbose:
        print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Load data
    splits = _load_splits()
    
    if verbose:
        print(f"\nDataset sizes:")
        print(f"  Train: {len(splits['train']):,} songs")
        print(f"  Val:   {len(splits['val']):,} songs")
        print(f"  Test:  {len(splits['test']):,} songs")
        print()
    
    # Compute embeddings for each split
    embeddings = {}
    for name, df in splits.items():
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Processing {name} set")
            print(f"{'─' * 80}")
        
        embeddings[name] = _compute_embeddings_for_split(
            df["lyrics"],
            model,
            batch_size=batch_size,
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ {name} embeddings: {embeddings[name].shape}")
    
    # Extract arrays
    X_train = embeddings["train"]
    X_val = embeddings["val"]
    X_test = embeddings["test"]
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Final embedding shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
        print(f"{'=' * 80}\n")
    
    # Save arrays
    if verbose:
        print("Saving embeddings to disk...")
    
    np.save(FEATURES_DIR / "X_train_embeddings.npy", X_train)
    np.save(FEATURES_DIR / "X_val_embeddings.npy", X_val)
    np.save(FEATURES_DIR / "X_test_embeddings.npy", X_test)
    
    # Mark step complete
    metadata = {
        "n_features": X_train.shape[1],
        "model_name": model_name,
        "batch_size": batch_size,
    }
    mark_step_complete("embeddings", input_files, output_files, metadata)
    
    if verbose:
        print(f"✅ Embeddings cached to {FEATURES_DIR}")
        print(f"   Future runs will load instantly from cache!")
    
    return {
        "X_train_embeddings": X_train,
        "X_val_embeddings": X_val,
        "X_test_embeddings": X_test,
    }


if __name__ == "__main__":
    # Standalone execution
    import sys
    
    verbose = "--quiet" not in sys.argv
    batch_size = 64
    
    # Parse batch size if provided
    for arg in sys.argv:
        if arg.startswith("--batch-size="):
            batch_size = int(arg.split("=")[1])
    
    result = process_embeddings(verbose=verbose, batch_size=batch_size)
    
    if verbose:
        print("\n" + "=" * 80)
        print("EMBEDDINGS PROCESSING COMPLETE")
        print("=" * 80)
        for key, arr in result.items():
            print(f"  {key}: {arr.shape}")
