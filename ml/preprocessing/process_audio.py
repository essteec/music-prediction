"""Audio Feature Processing Module

Processes audio features with intelligent caching.
Handles:
- Cyclical encoding for key
- Mixed scaling (normalized, standard scaled, categorical)
- One-hot encoding for genre
- Missing value handling

Can be run standalone or as part of the preprocessing pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_utils import (
    FEATURES_DIR,
    PROCESSED_DIR,
    check_if_step_needed,
    mark_step_complete,
)

# Feature group definitions
NORMALIZED_FEATURES = ["acousticness", "instrumentalness", "liveness", "speechiness"]
SCALE_FEATURES = ["loudness", "tempo", "duration_ms", "year"]
CATEGORICAL_FEATURES = ["mode"]
CYCLICAL_FEATURES = ["key"]
GENRE_FEATURE = ["genre"]


def _load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits."""
    return {
        "train": pd.read_csv(PROCESSED_DIR / "train.csv"),
        "val": pd.read_csv(PROCESSED_DIR / "val.csv"),
        "test": pd.read_csv(PROCESSED_DIR / "test.csv"),
    }


def _encode_key_cyclical(df: pd.DataFrame) -> None:
    """Apply cyclical sin/cos encoding to musical key."""
    key = pd.to_numeric(df["key"], errors="coerce")
    radians = 2 * np.pi * key / 12.0
    df["key_sin"] = np.sin(radians)
    df["key_cos"] = np.cos(radians)
    # Handle missing/unknown keys (-1) as origin
    df.loc[key.isna() | (key == -1), ["key_sin", "key_cos"]] = 0.0


def _handle_missing_values(splits: Dict[str, pd.DataFrame]) -> None:
    """Fill missing values using training set statistics."""
    train = splits["train"]
    
    numeric_features = NORMALIZED_FEATURES + SCALE_FEATURES + CATEGORICAL_FEATURES
    for feat in numeric_features:
        if train[feat].isnull().any():
            median_val = train[feat].median()
            for df in splits.values():
                df[feat].fillna(median_val, inplace=True)
    
    # Genre mode
    if train["genre"].isnull().any():
        mode_val = train["genre"].mode()[0]
        for df in splits.values():
            df["genre"].fillna(mode_val, inplace=True)


def _build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    is_train: bool = False,
) -> np.ndarray:
    """Build final feature matrix from all components."""
    # Normalized features (as-is)
    X_normalized = df[NORMALIZED_FEATURES].to_numpy(copy=False)
    
    # Scaled features
    if is_train:
        X_scaled = scaler.fit_transform(df[SCALE_FEATURES])
    else:
        X_scaled = scaler.transform(df[SCALE_FEATURES])
    
    # Categorical
    X_categorical = df[CATEGORICAL_FEATURES].to_numpy(copy=False)
    
    # Cyclical
    X_cyclical = df[["key_sin", "key_cos"]].to_numpy(copy=False)
    
    # Genre one-hot
    if is_train:
        X_genre = encoder.fit_transform(df[GENRE_FEATURE])
    else:
        X_genre = encoder.transform(df[GENRE_FEATURE])
    
    return np.hstack([X_normalized, X_scaled, X_categorical, X_cyclical, X_genre])


def process_audio_features(verbose: bool = True) -> Dict[str, np.ndarray]:
    """Process audio features with intelligent caching.
    
    Args:
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with keys 'X_train_audio', 'X_val_audio', 'X_test_audio'
    """
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define input and output files
    input_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
    ]
    
    output_files = [
        FEATURES_DIR / "X_train_audio.npy",
        FEATURES_DIR / "X_val_audio.npy",
        FEATURES_DIR / "X_test_audio.npy",
        FEATURES_DIR / "audio_scaler.pkl",
        FEATURES_DIR / "genre_encoder.pkl",
    ]
    
    # Check if processing is needed
    if not check_if_step_needed("audio_features", input_files, output_files):
        if verbose:
            print("✅ Audio features are up-to-date, skipping processing")
        return {
            "X_train_audio": np.load(FEATURES_DIR / "X_train_audio.npy"),
            "X_val_audio": np.load(FEATURES_DIR / "X_val_audio.npy"),
            "X_test_audio": np.load(FEATURES_DIR / "X_test_audio.npy"),
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING AUDIO FEATURES")
        print("=" * 80)
    
    # Load data
    splits = _load_splits()
    
    if verbose:
        print(f"Train: {len(splits['train']):,} songs")
        print(f"Val:   {len(splits['val']):,} songs")
        print(f"Test:  {len(splits['test']):,} songs")
    
    # Convert key and mode to numeric
    for df in splits.values():
        df["key"] = pd.to_numeric(df["key"], errors="coerce")
        df["mode"] = pd.to_numeric(df["mode"], errors="coerce")
    
    # Apply cyclical encoding to key
    for df in splits.values():
        _encode_key_cyclical(df)
    
    # Handle missing values
    _handle_missing_values(splits)
    
    # Create scalers/encoders
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    # Build feature matrices
    X_train = _build_feature_matrix(splits["train"], scaler, encoder, is_train=True)
    X_val = _build_feature_matrix(splits["val"], scaler, encoder, is_train=False)
    X_test = _build_feature_matrix(splits["test"], scaler, encoder, is_train=False)
    
    if verbose:
        print(f"\nFeature matrix shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
    
    # Save arrays
    np.save(FEATURES_DIR / "X_train_audio.npy", X_train)
    np.save(FEATURES_DIR / "X_val_audio.npy", X_val)
    np.save(FEATURES_DIR / "X_test_audio.npy", X_test)
    
    # Save transformers
    joblib.dump(scaler, FEATURES_DIR / "audio_scaler.pkl")
    joblib.dump(encoder, FEATURES_DIR / "genre_encoder.pkl")
    
    # Save feature names
    feature_names = (
        NORMALIZED_FEATURES
        + SCALE_FEATURES
        + CATEGORICAL_FEATURES
        + ["key_sin", "key_cos"]
        + [f"genre_{cat}" for cat in encoder.categories_[0]]
    )
    
    with open(FEATURES_DIR / "audio_feature_names.txt", "w") as f:
        f.write(f"# Audio features ({len(feature_names)} total)\n")
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Mark step complete
    metadata = {
        "n_features": X_train.shape[1],
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
    }
    mark_step_complete("audio_features", input_files, output_files, metadata)
    
    if verbose:
        print(f"✅ Audio features saved to {FEATURES_DIR}")
    
    return {
        "X_train_audio": X_train,
        "X_val_audio": X_val,
        "X_test_audio": X_test,
    }


if __name__ == "__main__":
    process_audio_features(verbose=True)
