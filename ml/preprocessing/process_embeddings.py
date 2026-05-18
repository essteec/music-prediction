"""Lyric Embeddings Processing Module

Extracts semantic embeddings from lyrics using sentence-transformers.
Models: 
- all-MiniLM-L6-v2 (English-optimized, 384 dimensions)
- all-mpnet-base-v2 (Microsoft, 768 dimensions)

Key Features:
- Semantic understanding beyond word frequency
- Compact representations
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
import torch

import numpy as np
import pandas as pd
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
    """Compute embeddings for a split with batch processing."""
    lyrics_list = []
    for lyric in lyrics_series:
        if not isinstance(lyric, str) or not lyric.strip():
            lyrics_list.append("")  
        else:
            text = lyric.strip()
            if len(text) > 3000:
                text = text[:3000]
            lyrics_list.append(text)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    embeddings = model.encode(
        lyrics_list,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return embeddings

def process_embeddings(
    verbose: bool = True,
    batch_size: int = 64,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Process lyric embeddings with intelligent caching.
    Extracts both MiniLM and MPNet embeddings.
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    models_to_run = {
        'embeddings': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'prefix': 'embeddings',
            'batch': batch_size
        },
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'prefix': 'mpnet',
            'batch': 32 
        }
    }
    
    results = {}
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )

    for step_key, config in models_to_run.items():
        prefix = config['prefix']
        
        input_files = [
            PROCESSED_DIR / "train.csv",
            PROCESSED_DIR / "val.csv",
            PROCESSED_DIR / "test.csv",
        ]
        
        output_files = [
            FEATURES_DIR / f"X_train_{prefix}.npy",
            FEATURES_DIR / f"X_val_{prefix}.npy",
            FEATURES_DIR / f"X_test_{prefix}.npy",
        ]
        
        if not check_if_step_needed(step_key, input_files, output_files):
            if verbose:
                print(f"✅ {config['name']} embeddings are up-to-date, loading from cache")
            results[step_key] = {
                f"X_train_{prefix}": np.load(FEATURES_DIR / f"X_train_{prefix}.npy"),
                f"X_val_{prefix}": np.load(FEATURES_DIR / f"X_val_{prefix}.npy"),
                f"X_test_{prefix}": np.load(FEATURES_DIR / f"X_test_{prefix}.npy"),
            }
            continue
            
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROCESSING LYRIC EMBEDDINGS: {config['name']}")
            print("=" * 80)
        
        print(f"Loading model: {config['name']}...")
        model = SentenceTransformer(config['name'])
        model.max_seq_length = 512
        
        splits = _load_splits()
        
        model_results = {}
        for name, df in splits.items():
            if verbose:
                print(f"Processing {name} set")
            
            emb = _compute_embeddings_for_split(
                df["lyrics"],
                model,
                batch_size=config['batch'],
                verbose=verbose
            )
            model_results[name] = emb
            
        X_train = model_results["train"]
        X_val = model_results["val"]
        X_test = model_results["test"]
        
        if verbose:
            print("Saving embeddings to disk...")
        
        np.save(FEATURES_DIR / f"X_train_{prefix}.npy", X_train)
        np.save(FEATURES_DIR / f"X_val_{prefix}.npy", X_val)
        np.save(FEATURES_DIR / f"X_test_{prefix}.npy", X_test)
        
        metadata = {
            "n_features": X_train.shape[1],
            "model_name": config['name'],
            "batch_size": config['batch'],
        }
        mark_step_complete(step_key, input_files, output_files, metadata)
        
        results[step_key] = {
            f"X_train_{prefix}": X_train,
            f"X_val_{prefix}": X_val,
            f"X_test_{prefix}": X_test,
        }
        
        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return results

if __name__ == "__main__":
    import sys
    verbose = "--quiet" not in sys.argv
    process_embeddings(verbose=verbose)
