"""
Audio Feature Preparation - Mixed Scaling Strategy
Prepares and scales audio features for modeling

Uses intelligent feature grouping:
- Already normalized [0,1] features: keep as is
- Continuous features with different scales: StandardScaler
- Categorical/binary features: keep as is
- Genre: one-hot encoding

Can be run standalone or as part of the preprocessing pipeline.
Skips processing if outputs are up-to-date.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from pathlib import Path
from typing import Dict, Tuple

from pipeline_utils import (
    check_if_step_needed,
    mark_step_complete,
    FEATURES_DIR,
    PROCESSED_DIR,
)

# Define feature groups by scaling strategy
# Already normalized [0,1] - keep as is (no scaling needed)
NORMALIZED_FEATURES = [
    'acousticness',
    'instrumentalness', 
    'liveness',
    'speechiness'
]

# Need scaling (different ranges)
SCALE_FEATURES = [
    'loudness',      # [-60, 0] dB
    'tempo',         # [20, 300] BPM
    'duration_ms',   # [1000, 3600000] ms
    'year'           # [1900, 2025] - temporal feature
]

# Categorical/binary - keep as is
CATEGORICAL_FEATURES = [
    'mode'   # [0, 1] - binary (major/minor)
]

# Cyclical features - will be encoded with sin/cos
CYCLICAL_FEATURES = [
    'key'    # [-1, 11] - pitch classes (cyclical, not linear!)
]

# Genre - will be one-hot encoded
GENRE_FEATURE = [
    'genre'
]

# All audio features combined (for reference)
ALL_AUDIO_FEATURES = (NORMALIZED_FEATURES + SCALE_FEATURES + 
                      CATEGORICAL_FEATURES + CYCLICAL_FEATURES + GENRE_FEATURE)


def process_audio_features(verbose: bool = True) -> Dict[str, np.ndarray]:
    """Process audio features with intelligent caching.
    
    Returns:
        Dictionary with keys like 'X_train_audio', 'X_val_audio', etc.
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
        FEATURES_DIR / "audio_feature_names.txt",
    ]
    
    # Check if processing is needed
    if not check_if_step_needed("audio_features", input_files, output_files):
        if verbose:
            print("✅ Audio features are up-to-date, skipping processing")
        # Load and return existing features
        return {
            "X_train_audio": np.load(FEATURES_DIR / "X_train_audio.npy"),
            "X_val_audio": np.load(FEATURES_DIR / "X_val_audio.npy"),
            "X_test_audio": np.load(FEATURES_DIR / "X_test_audio.npy"),
        }
    
    if verbose:
        print("=" * 80)
        print("AUDIO FEATURE PREPARATION - MIXED SCALING")
        print("=" * 80)
        print(f"\nFeature Groups:")
        print(f"  Normalized [0,1] (keep as-is): {len(NORMALIZED_FEATURES)} features")
        print(f"    → {', '.join(NORMALIZED_FEATURES)}")
        print(f"  Need Scaling (StandardScaler): {len(SCALE_FEATURES)} features")
        print(f"    → {', '.join(SCALE_FEATURES)}")
        print(f"  Categorical (keep as-is): {len(CATEGORICAL_FEATURES)} features")
        print(f"    → {', '.join(CATEGORICAL_FEATURES)}")
        print(f"  Cyclical (sin/cos encoding): {len(CYCLICAL_FEATURES)} features")
        print(f"    → {', '.join(CYCLICAL_FEATURES)} → produces 2 features (sin, cos)")
        print(f"  Genre (one-hot encode): {len(GENRE_FEATURE)} feature")
        print(f"    → {', '.join(GENRE_FEATURE)}")

# Load splits
print("\nLoading data splits...")
df_train = pd.read_csv(processed_dir / 'train.csv')
df_val = pd.read_csv(processed_dir / 'val.csv')
df_test = pd.read_csv(processed_dir / 'test.csv')

print(f"Train: {len(df_train):,} songs")
print(f"Val:   {len(df_val):,} songs")
print(f"Test:  {len(df_test):,} songs")

# Convert key and mode to numeric (they might be stored as strings)
print("\nConverting key and mode to numeric...")
for df in [df_train, df_val, df_test]:
    df['key'] = pd.to_numeric(df['key'], errors='coerce')
    df['mode'] = pd.to_numeric(df['mode'], errors='coerce')
print("✅ Conversion complete")

# Apply cyclical encoding to key (musical keys are cyclical, not linear!)
print("\n" + "-" * 80)
print("CYCLICAL ENCODING FOR KEY")
print("-" * 80)
print("Musical keys are cyclical: C(0) → C#(1) → ... → B(11) → C(0)")
print("Using sin/cos encoding to preserve cyclical nature")
print("Treating key=-1 (no key detected) as neutral point (0, 0)")

for df in [df_train, df_val, df_test]:
    # Temporarily replace -1 with NaN for transformation
    df['key_for_transform'] = df['key'].replace(-1, np.nan)
    
    # Apply cyclical encoding: sin and cos of 2π * key / 12
    df['key_sin'] = np.sin(2 * np.pi * df['key_for_transform'] / 12)
    df['key_cos'] = np.cos(2 * np.pi * df['key_for_transform'] / 12)
    
    # Set -1 (no key) to origin (0, 0)
    df[['key_sin', 'key_cos']] = df[['key_sin', 'key_cos']].fillna(0)
    
    # Drop temporary column
    df.drop('key_for_transform', axis=1, inplace=True)

print("✅ Key cyclically encoded as (key_sin, key_cos)")
print("   Example: C(0) → (0.0, 1.0), F#(6) → (0.0, -1.0), -1 → (0.0, 0.0)")

# Check for missing values BEFORE feature extraction
print("\n" + "-" * 80)
print("MISSING VALUES CHECK")
print("-" * 80)

train_missing = df_train[ALL_AUDIO_FEATURES].isnull().sum()
if train_missing.sum() > 0:
    print("⚠️  Missing values found in training set:")
    print(train_missing[train_missing > 0])
    print("\nFilling missing values with column median/mode...")
    
    for df in [df_train, df_val, df_test]:
        # Numeric features: use median
        numeric_features = NORMALIZED_FEATURES + SCALE_FEATURES + CATEGORICAL_FEATURES
        for feat in numeric_features:
            if df[feat].isnull().any():
                median_val = df_train[feat].median()
                df[feat].fillna(median_val, inplace=True)
        
        # Cyclical features: already handled (NaN → 0, 0)
        # key_sin and key_cos already filled in cyclical encoding step
        
        # Genre: use mode (most common) - only if genre is in features
        if GENRE_FEATURE and 'genre' in df.columns:
            if df['genre'].isnull().any():
                mode_val = df_train['genre'].mode()[0]
                df['genre'].fillna(mode_val, inplace=True)
    
    print("✅ Missing values filled")
else:
    print("✅ No missing values in training set")

# Extract and prepare features with mixed scaling
print("\n" + "-" * 80)
print("MIXED SCALING STRATEGY")
print("-" * 80)

# 1. Extract normalized features (keep as-is)
X_train_normalized = df_train[NORMALIZED_FEATURES].values
X_val_normalized = df_val[NORMALIZED_FEATURES].values
X_test_normalized = df_test[NORMALIZED_FEATURES].values
print(f"✅ Normalized features extracted: {X_train_normalized.shape}")

# 2. Extract and scale continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[SCALE_FEATURES])
X_val_scaled = scaler.transform(df_val[SCALE_FEATURES])
X_test_scaled = scaler.transform(df_test[SCALE_FEATURES])
print(f"✅ Scaled features (StandardScaler): {X_train_scaled.shape}")

# Show scaling statistics
print("\n   Scaling statistics (from training set):")
print(f"   {'Feature':<20s} {'Mean':<12s} {'Std':<12s}")
print("   " + "-" * 50)
for feat, mean, std in zip(SCALE_FEATURES, scaler.mean_, scaler.scale_):
    print(f"   {feat:<20s} {mean:<12.4f} {std:<12.4f}")

# 3. Extract categorical features (keep as-is)
X_train_categorical = df_train[CATEGORICAL_FEATURES].values
X_val_categorical = df_val[CATEGORICAL_FEATURES].values
X_test_categorical = df_test[CATEGORICAL_FEATURES].values
print(f"\n✅ Categorical features extracted: {X_train_categorical.shape}")

# 4. Extract cyclical features (already encoded as sin/cos)
cyclical_feature_names = ['key_sin', 'key_cos']
X_train_cyclical = df_train[cyclical_feature_names].values
X_val_cyclical = df_val[cyclical_feature_names].values
X_test_cyclical = df_test[cyclical_feature_names].values
print(f"✅ Cyclical features extracted: {X_train_cyclical.shape}")

# 5. One-hot encode genre (only if genre is in features)
if GENRE_FEATURE:
    print("\nOne-hot encoding genre...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_genre = encoder.fit_transform(df_train[GENRE_FEATURE])
    X_val_genre = encoder.transform(df_val[GENRE_FEATURE])
    X_test_genre = encoder.transform(df_test[GENRE_FEATURE])
    print(f"✅ Genre one-hot encoded: {X_train_genre.shape}")
    print(f"   Categories: {encoder.categories_[0].tolist()}")
else:
    print("\n⚠️  Genre feature skipped (commented out)")
    X_train_genre = np.empty((X_train_normalized.shape[0], 0))
    X_val_genre = np.empty((X_val_normalized.shape[0], 0))
    X_test_genre = np.empty((X_test_normalized.shape[0], 0))
    encoder = None

# 6. Concatenate all features
X_train_final = np.hstack([
    X_train_normalized,    # [0,1] features unchanged
    X_train_scaled,        # Scaled continuous features
    X_train_categorical,   # Binary features (mode)
    X_train_cyclical,      # Cyclical features (key_sin, key_cos)
    X_train_genre          # One-hot encoded genre
])

X_val_final = np.hstack([
    X_val_normalized,
    X_val_scaled,
    X_val_categorical,
    X_val_cyclical,
    X_val_genre
])

X_test_final = np.hstack([
    X_test_normalized,
    X_test_scaled,
    X_test_categorical,
    X_test_cyclical,
    X_test_genre
])

print("\n" + "-" * 80)
print("FINAL FEATURE MATRIX")
print("-" * 80)
print(f"Train shape: {X_train_final.shape}")
print(f"Val shape:   {X_val_final.shape}")
print(f"Test shape:  {X_test_final.shape}")
print(f"\nFeature breakdown:")
print(f"  Normalized [0,1]:     {len(NORMALIZED_FEATURES)} features")
print(f"  Scaled (continuous):  {len(SCALE_FEATURES)} features")
print(f"  Categorical:          {len(CATEGORICAL_FEATURES)} features")
print(f"  Cyclical (sin/cos):   {len(cyclical_feature_names)} features")
if GENRE_FEATURE:
    print(f"  Genre (one-hot):      {X_train_genre.shape[1]} features")
else:
    print(f"  Genre (one-hot):      0 features (skipped)")
print(f"  TOTAL:                {X_train_final.shape[1]} features")

# Extract targets for all 4 variables
print("\n" + "-" * 80)
print("EXTRACTING TARGETS")
print("-" * 80)

targets = ['valence', 'energy', 'danceability', 'popularity']

y_train = {}
y_val = {}
y_test = {}

for target in targets:
    y_train[target] = df_train[target].values
    y_val[target] = df_val[target].values
    y_test[target] = df_test[target].values
    
    print(f"\n{target.capitalize()}:")
    print(f"  Train: {y_train[target].shape[0]:,} samples, mean={y_train[target].mean():.3f}")
    print(f"  Val:   {y_val[target].shape[0]:,} samples, mean={y_val[target].mean():.3f}")
    print(f"  Test:  {y_test[target].shape[0]:,} samples, mean={y_test[target].mean():.3f}")

# Save prepared features
print("\n" + "-" * 80)
print("SAVING PREPARED FEATURES")
print("-" * 80)

# Save final feature matrices
np.save(features_dir / 'X_train_audio.npy', X_train_final)
np.save(features_dir / 'X_val_audio.npy', X_val_final)
np.save(features_dir / 'X_test_audio.npy', X_test_final)

# Save targets
for target in targets:
    np.save(features_dir / f'y_train_{target}.npy', y_train[target])
    np.save(features_dir / f'y_val_{target}.npy', y_val[target])
    np.save(features_dir / f'y_test_{target}.npy', y_test[target])

# Save scaler and encoder for future use
joblib.dump(scaler, features_dir / 'audio_scaler.pkl')
if encoder is not None:
    joblib.dump(encoder, features_dir / 'genre_encoder.pkl')
    print(f"✅ Genre encoder saved to: {features_dir}/genre_encoder.pkl")
else:
    print(f"⚠️  Genre encoder not saved (genre feature skipped)")

print(f"✅ Feature matrices saved to: {features_dir}/X_*_audio.npy")
print(f"✅ Targets saved to: {features_dir}/y_*_*.npy")
print(f"✅ Scaler saved to: {features_dir}/audio_scaler.pkl")

# Save feature names and metadata for reference
feature_info = {
    'normalized_features': NORMALIZED_FEATURES,
    'scaled_features': SCALE_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'cyclical_features': cyclical_feature_names,
    'genre_categories': encoder.categories_[0].tolist() if encoder else [],
    'total_features': X_train_final.shape[1]
}

# Create detailed feature name list
feature_names = (
    NORMALIZED_FEATURES +           # 4 features
    SCALE_FEATURES +                # 3-4 features (depending on year)
    CATEGORICAL_FEATURES +          # 1 feature (mode)
    cyclical_feature_names          # 2 features (key_sin, key_cos)
)

# Add genre features if present
if encoder is not None:
    feature_names += [f'genre_{cat}' for cat in encoder.categories_[0]]

with open(features_dir / 'audio_feature_names.txt', 'w') as f:
    f.write("# Audio Feature Names (Mixed Scaling Strategy)\n")
    f.write(f"# Total: {len(feature_names)} features\n\n")
    f.write("# Normalized [0,1] features (unchanged):\n")
    for feat in NORMALIZED_FEATURES:
        f.write(f"{feat}\n")
    f.write("\n# Scaled features (StandardScaler):\n")
    for feat in SCALE_FEATURES:
        f.write(f"{feat}\n")
    f.write("\n# Categorical features (unchanged):\n")
    for feat in CATEGORICAL_FEATURES:
        f.write(f"{feat}\n")
    f.write("\n# Cyclical features (sin/cos encoding):\n")
    for feat in cyclical_feature_names:
        f.write(f"{feat}\n")
    if encoder is not None:
        f.write("\n# Genre (one-hot encoded):\n")
        for cat in encoder.categories_[0]:
            f.write(f"genre_{cat}\n")
    else:
        f.write("\n# Genre: skipped (commented out)\n")

print(f"✅ Feature names saved to: {features_dir}/audio_feature_names.txt")

# Save metadata as JSON for easy loading
import json
with open(features_dir / 'feature_metadata.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✅ Feature metadata saved to: {features_dir}/feature_metadata.json")

print("\n" + "=" * 80)
print("FEATURE PREPARATION COMPLETE!")
print("=" * 80)
print("\n✅ Ready for baseline model training!")
