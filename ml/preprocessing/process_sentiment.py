"""Sentiment Processing Module

Extracts and scales sentiment features from lyrics using TextBlob.
Features:
- sentiment_polarity
- sentiment_subjectivity

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

SENTIMENT_COLUMNS = ["sentiment_polarity", "sentiment_subjectivity"]


def _extract_sentiment(lyrics: str) -> Dict[str, float]:
    """Extract sentiment features using TextBlob."""
    if not isinstance(lyrics, str) or not lyrics.strip():
        return {
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
        }
    
    try:
        from textblob import TextBlob
        blob = TextBlob(lyrics)
        return {
            "sentiment_polarity": blob.sentiment.polarity,
            "sentiment_subjectivity": blob.sentiment.subjectivity,
        }
    except Exception:
        return {
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
        }


def _load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def _compute_sentiment_for_split(lyrics_series: pd.Series, verbose: bool = False) -> pd.DataFrame:
    """Compute sentiment for a split."""
    records = []
    if verbose:
        from tqdm import tqdm
        lyrics_series = tqdm(lyrics_series, desc="Extracting sentiment")
    
    for text in lyrics_series:
        records.append(_extract_sentiment(text))
    
    return pd.DataFrame(records, columns=SENTIMENT_COLUMNS)


def process_sentiment(verbose: bool = True) -> Dict[str, np.ndarray]:
    """Process sentiment features with intelligent caching.
    
    Args:
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with keys 'X_train_sentiment', 'X_val_sentiment', 'X_test_sentiment'
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define input and output files
    input_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
    ]
    
    output_files = [
        FEATURES_DIR / "X_train_sentiment.npy",
        FEATURES_DIR / "X_val_sentiment.npy",
        FEATURES_DIR / "X_test_sentiment.npy",
        FEATURES_DIR / "sentiment_scaler.pkl",
    ]
    
    # Check if processing is needed
    if not check_if_step_needed("sentiment", input_files, output_files):
        if verbose:
            print("✅ Sentiment features are up-to-date, skipping processing")
        return {
            "X_train_sentiment": np.load(FEATURES_DIR / "X_train_sentiment.npy"),
            "X_val_sentiment": np.load(FEATURES_DIR / "X_val_sentiment.npy"),
            "X_test_sentiment": np.load(FEATURES_DIR / "X_test_sentiment.npy"),
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING SENTIMENT FEATURES")
        print("=" * 80)
        print("Using TextBlob (this may take a few minutes)...")
    
    # Load data
    splits = _load_splits()
    
    if verbose:
        print(f"Train: {len(splits['train']):,} songs")
        print(f"Val:   {len(splits['val']):,} songs")
        print(f"Test:  {len(splits['test']):,} songs")
    
    # Extract sentiment
    sentiment = {}
    for name, df in splits.items():
        if verbose:
            print(f"\nExtracting sentiment from {name} set...")
        sentiment[name] = _compute_sentiment_for_split(df["lyrics"], verbose=verbose)
    
    # Scale sentiment features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(sentiment["train"])
    X_val = scaler.transform(sentiment["val"])
    X_test = scaler.transform(sentiment["test"])
    
    if verbose:
        print(f"\nFeature matrix shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
    
    # Save arrays
    np.save(FEATURES_DIR / "X_train_sentiment.npy", X_train)
    np.save(FEATURES_DIR / "X_val_sentiment.npy", X_val)
    np.save(FEATURES_DIR / "X_test_sentiment.npy", X_test)
    
    # Save scaler
    joblib.dump(scaler, FEATURES_DIR / "sentiment_scaler.pkl")
    
    # Mark step complete
    metadata = {
        "n_features": X_train.shape[1],
        "features": SENTIMENT_COLUMNS,
    }
    mark_step_complete("sentiment", input_files, output_files, metadata)
    
    if verbose:
        print(f"✅ Sentiment features saved to {FEATURES_DIR}")
    
    return {
        "X_train_sentiment": X_train,
        "X_val_sentiment": X_val,
        "X_test_sentiment": X_test,
    }


if __name__ == "__main__":
    process_sentiment(verbose=True)
