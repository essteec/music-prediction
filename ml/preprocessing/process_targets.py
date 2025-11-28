"""Target Variable Processing Module

Processes target variables (valence, energy, danceability, popularity).
Applies log1p transformation to popularity to handle extreme right skew (EDA finding).

Can be run standalone or as part of the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pipeline_utils import (
    FEATURES_DIR,
    PROCESSED_DIR,
    check_if_step_needed,
    mark_step_complete,
)

TARGETS = ["valence", "energy", "danceability", "popularity"]


def _load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def process_targets(verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """Process target variables with intelligent caching.
    
    Args:
        verbose: Whether to print progress information
        
    Returns:
        Nested dictionary: {target_name: {split_name: array}}
        Example: {"valence": {"train": array, "val": array, "test": array}}
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define input and output files
    input_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
    ]
    
    output_files = []
    for target in TARGETS:
        for split in ["train", "val", "test"]:
            output_files.append(FEATURES_DIR / f"y_{split}_{target}.npy")
    
    # Check if processing is needed
    if not check_if_step_needed("targets", input_files, output_files):
        if verbose:
            print("✅ Target variables are up-to-date, skipping processing")
        
        # Load existing targets
        result = {}
        for target in TARGETS:
            result[target] = {
                "train": np.load(FEATURES_DIR / f"y_train_{target}.npy"),
                "val": np.load(FEATURES_DIR / f"y_val_{target}.npy"),
                "test": np.load(FEATURES_DIR / f"y_test_{target}.npy"),
            }
        return result
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING TARGET VARIABLES")
        print("=" * 80)
    
    # Load data
    splits = _load_splits()
    
    if verbose:
        print(f"Train: {len(splits['train']):,} songs")
        print(f"Val:   {len(splits['val']):,} songs")
        print(f"Test:  {len(splits['test']):,} songs")
    
    # Extract and transform targets
    result = {}
    for target in TARGETS:
        result[target] = {}
        
        for split_name, df in splits.items():
            array = df[target].to_numpy(dtype=np.float32)
            
            # Apply log transformation to popularity
            # EDA Finding: Popularity is heavily right-skewed (mean=16.9, median=13.0)
            # Log1p transformation helps normalize the distribution
            if target == "popularity":
                array = np.log1p(array)
            
            result[target][split_name] = array
            
            # Save individual target file
            np.save(FEATURES_DIR / f"y_{split_name}_{target}.npy", array)
        
        if verbose:
            print(f"\n{target}:")
            print(f"  Train mean: {result[target]['train'].mean():.3f}")
            print(f"  Val mean:   {result[target]['val'].mean():.3f}")
            print(f"  Test mean:  {result[target]['test'].mean():.3f}")
    
    # Mark step complete
    metadata = {
        "targets": TARGETS,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }
    mark_step_complete("targets", input_files, output_files, metadata)
    
    if verbose:
        print(f"\n✅ Target variables saved to {FEATURES_DIR}")
    
    return result


if __name__ == "__main__":
    process_targets(verbose=True)
