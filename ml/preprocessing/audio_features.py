"""
Audio Feature Preparation
Prepares and scales audio features for modeling

Only uses audio features (no text/lyrics initially)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

print("=" * 80)
print("AUDIO FEATURE PREPARATION")
print("=" * 80)

# Define audio features to use
AUDIO_FEATURES = [
    'acousticness',
    'instrumentalness', 
    'liveness',
    'speechiness',
    'loudness',
    'tempo',
    'duration_ms',
    'key',
    'mode'
]

# Paths
processed_dir = Path('../../dataset/processed')
features_dir = Path('../features')
features_dir.mkdir(exist_ok=True, parents=True)

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

# Extract audio features
print("\n" + "-" * 80)
print("EXTRACTING AUDIO FEATURES")
print("-" * 80)

X_train = df_train[AUDIO_FEATURES].values
X_val = df_val[AUDIO_FEATURES].values
X_test = df_test[AUDIO_FEATURES].values

print(f"\nFeatures extracted:")
print(f"  Train shape: {X_train.shape}")
print(f"  Val shape:   {X_val.shape}")
print(f"  Test shape:  {X_test.shape}")

print(f"\nUsing {len(AUDIO_FEATURES)} audio features:")
for i, feat in enumerate(AUDIO_FEATURES, 1):
    print(f"  {i}. {feat}")

# Check for missing values
print("\n" + "-" * 80)
print("MISSING VALUES CHECK")
print("-" * 80)

train_missing = pd.DataFrame(X_train, columns=AUDIO_FEATURES).isnull().sum()
if train_missing.sum() > 0:
    print("⚠️  Missing values found in training set:")
    print(train_missing[train_missing > 0])
    print("\nFilling missing values with column median...")
    
    # Calculate medians from training data
    X_train_df = pd.DataFrame(X_train, columns=AUDIO_FEATURES)
    X_val_df = pd.DataFrame(X_val, columns=AUDIO_FEATURES)
    X_test_df = pd.DataFrame(X_test, columns=AUDIO_FEATURES)
    
    # Fill with median from training set
    medians = X_train_df.median()
    X_train = X_train_df.fillna(medians).values
    X_val = X_val_df.fillna(medians).values
    X_test = X_test_df.fillna(medians).values
    
    print("✅ Missing values filled with training median")
else:
    print("✅ No missing values in training set")

# Scale features using training statistics
print("\n" + "-" * 80)
print("SCALING FEATURES")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")
print(f"   Fit on training set, applied to all sets")

# Show scaling statistics
print("\nScaling statistics (from training set):")
print(f"{'Feature':<20s} {'Mean':<12s} {'Std':<12s}")
print("-" * 80)
for feat, mean, std in zip(AUDIO_FEATURES, scaler.mean_, scaler.scale_):
    print(f"{feat:<20s} {mean:<12.4f} {std:<12.4f}")

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

# Save scaled features
np.save(features_dir / 'X_train_audio.npy', X_train_scaled)
np.save(features_dir / 'X_val_audio.npy', X_val_scaled)
np.save(features_dir / 'X_test_audio.npy', X_test_scaled)

# Save targets
for target in targets:
    np.save(features_dir / f'y_train_{target}.npy', y_train[target])
    np.save(features_dir / f'y_val_{target}.npy', y_val[target])
    np.save(features_dir / f'y_test_{target}.npy', y_test[target])

# Save scaler for future use
joblib.dump(scaler, features_dir / 'audio_scaler.pkl')

print(f"✅ Scaled features saved to: {features_dir}/X_*_audio.npy")
print(f"✅ Targets saved to: {features_dir}/y_*_*.npy")
print(f"✅ Scaler saved to: {features_dir}/audio_scaler.pkl")

# Save feature names for reference
with open(features_dir / 'audio_feature_names.txt', 'w') as f:
    for feat in AUDIO_FEATURES:
        f.write(f"{feat}\n")

print(f"✅ Feature names saved to: {features_dir}/audio_feature_names.txt")

print("\n" + "=" * 80)
print("FEATURE PREPARATION COMPLETE!")
print("=" * 80)
print("\n✅ Ready for baseline model training!")
