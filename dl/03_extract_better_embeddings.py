"""Phase 1: Extract Better Lyric Embeddings

Extracts 768-d embeddings using all-mpnet-base-v2 (Microsoft, proven SOTA).

Replaces Phase 0's frozen MiniLM-L6-v2 (384-d) with better model.
Expected: Improved Valence prediction (semantic understanding).

Output:
    data/embeddings/mpnet_lyrics_768d_{split}.npy
    
Usage:
    python dl/03_extract_better_embeddings.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure PyTorch memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dl.utils.reproducibility import set_all_seeds

# Paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)


def load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def compute_embeddings_for_split(
    lyrics_series: pd.Series,
    model: SentenceTransformer,
    batch_size: int = 32,
    verbose: bool = True,
    max_length: int = 512,  # Truncate very long lyrics
) -> np.ndarray:
    """Compute 768-d embeddings for a split.
    
    Args:
        lyrics_series: Series of lyrics strings
        model: SentenceTransformer model
        batch_size: Number of lyrics to process at once (handled by model.encode internally)
        verbose: Whether to show progress bar
        max_length: Maximum sequence length (truncate longer lyrics)
        
    Returns:
        Array of shape (n_songs, 768)
    """
    # Prepare lyrics (handle missing/empty)
    lyrics_list = []
    for lyric in lyrics_series:
        if not isinstance(lyric, str) or not lyric.strip():
            lyrics_list.append("")  # Empty string will get zero embedding
        else:
            # Truncate very long lyrics (approx 512 tokens = 2500-3000 chars)
            text = lyric.strip()
            if len(text) > 3000:  # Rough char limit to stay under 512 tokens
                text = text[:3000]
            lyrics_list.append(text)
    
    # Clear CUDA cache before encoding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Let model.encode handle batching internally - much more efficient
    embeddings = model.encode(
        lyrics_list,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for better semantic similarity
    )
    
    # Clear cache after encoding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return embeddings


def extract_embeddings_for_model(
    model_name: str,
    output_prefix: str,
    batch_size: int = 32,
    trust_remote_code: bool = False,
) -> None:
    """Extract embeddings for all splits using specified model.
    
    Args:
        model_name: SentenceTransformer model identifier
        output_prefix: Prefix for output files (e.g., 'mpnet', 'gte')
        batch_size: Batch size for encoding
        trust_remote_code: Whether to trust remote code (needed for GTE)
    """
    print(f"\n{'='*60}")
    print(f"Extracting {output_prefix.upper()} Embeddings")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    
    # CRITICAL FIX: Set max_seq_length to prevent CUDA index out of bounds
    # MPNet is 512, GTE can be 8192 but 512 is safer and faster for lyrics
    model.max_seq_length = 512 
    
    print(f"✓ Model loaded (embedding_dim={model.get_sentence_embedding_dimension()}, max_seq_length={model.max_seq_length})")
    
    # Load splits
    print("\nLoading data splits...")
    splits = load_splits()
    print(f"✓ Train: {len(splits['train']):,} songs")
    print(f"✓ Val:   {len(splits['val']):,} songs")
    print(f"✓ Test:  {len(splits['test']):,} songs")
    
    # Extract embeddings for each split
    for split_name, df in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Check if already exists
        output_file = EMBEDDINGS_DIR / f"{output_prefix}_lyrics_768d_{split_name}.npy"
        if output_file.exists():
            print(f"⚠️  {output_file.name} already exists, skipping...")
            continue
        
        # Extract embeddings
        embeddings = compute_embeddings_for_split(
            df["lyrics"],
            model,
            batch_size=batch_size,
            verbose=True,
        )
        
        # Save to disk
        np.save(output_file, embeddings)
        
        # Verify
        expected_shape = (len(df), 768)
        actual_shape = embeddings.shape
        assert actual_shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )
        
        print(f"✓ Saved: {output_file.name}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {embeddings.nbytes / 1024**2:.1f} MB")


def main():
    """Extract MPNet embeddings for all splits."""
    # Set seed for reproducibility
    set_all_seeds(42)
    
    print("=" * 60)
    print("Phase 1: Better Lyric Embeddings Extraction")
    print("=" * 60)
    print("\nExtracting 768-d embeddings using all-mpnet-base-v2 (Microsoft)")
    print("This will replace MiniLM-L6-v2 (384-d) from Phase 0")
    
    # Extract MPNet embeddings (proven workhorse)
    extract_embeddings_for_model(
        model_name="sentence-transformers/all-mpnet-base-v2",
        output_prefix="mpnet",
        batch_size=32,  # Conservative for 6GB VRAM
    )
    
    print("\n" + "=" * 60)
    print("✓ Embedding Extraction Complete!")
    print("=" * 60)
    print("\nOutput files:")
    for split in ["train", "val", "test"]:
        filepath = EMBEDDINGS_DIR / f"mpnet_lyrics_768d_{split}.npy"
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024**2
            print(f"  {filepath.name}: {size_mb:.1f} MB")
    
    print("\nNext steps:")
    print("  1. Train MLP with MPNet embeddings (798 features total)")
    print("  2. Compare to Phase 0 baseline (414 features, MiniLM)")
    print("  3. Analyze improvement on Valence prediction")


if __name__ == "__main__":
    main()
