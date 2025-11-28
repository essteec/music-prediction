"""
Data Splitting Script - Artist-Aware Splits
Creates train/val/test splits grouped by artist to prevent data leakage

CRITICAL: Uses GroupShuffleSplit to ensure same artist doesn't appear in multiple sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

print("=" * 80)
print("ARTIST-AWARE DATA SPLITTING")
print("=" * 80)

# Paths (using REPO_ROOT for consistency)
REPO_ROOT = Path(__file__).resolve().parents[2]
data_path = REPO_ROOT / 'data' / 'processed' / 'english_ml_ready.csv'
output_dir = REPO_ROOT / 'data' / 'processed'
output_dir.mkdir(exist_ok=True)

# Load data
print(f"\nLoading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Total songs: {len(df):,}")
print(f"Unique artists: {df['artists'].nunique():,}")

# Create artist groups
# Use 'artists' column as the grouping variable
groups = df['artists']

print("\n" + "-" * 80)
print("CREATING SPLITS (70% train, 15% val, 15% test)")
print("-" * 80)

# First split: train vs temp (val + test)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=groups))

df_train = df.iloc[train_idx].copy()
df_temp = df.iloc[temp_idx].copy()

print(f"\nFirst split complete:")
print(f"  Train: {len(df_train):,} songs ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Temp:  {len(df_temp):,} songs ({len(df_temp)/len(df)*100:.1f}%)")

# Second split: val vs test (split temp 50/50)
groups_temp = df_temp['artists']
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=groups_temp))

df_val = df_temp.iloc[val_idx].copy()
df_test = df_temp.iloc[test_idx].copy()

print(f"\nSecond split complete:")
print(f"  Val:   {len(df_val):,} songs ({len(df_val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(df_test):,} songs ({len(df_test)/len(df)*100:.1f}%)")

# Verify no artist overlap
train_artists = set(df_train['artists'].unique())
val_artists = set(df_val['artists'].unique())
test_artists = set(df_test['artists'].unique())

overlap_train_val = train_artists & val_artists
overlap_train_test = train_artists & test_artists
overlap_val_test = val_artists & test_artists

print("\n" + "-" * 80)
print("VERIFICATION: Artist Overlap Check")
print("-" * 80)
print(f"Train artists: {len(train_artists):,}")
print(f"Val artists:   {len(val_artists):,}")
print(f"Test artists:  {len(test_artists):,}")
print(f"\nOverlap between:")
print(f"  Train & Val:  {len(overlap_train_val)} artists ✅ (should be 0)")
print(f"  Train & Test: {len(overlap_train_test)} artists ✅ (should be 0)")
print(f"  Val & Test:   {len(overlap_val_test)} artists ✅ (should be 0)")

if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
    print("\n⚠️ WARNING: Artist overlap detected! Data leakage risk!")
else:
    print("\n✅ SUCCESS: No artist overlap - splits are clean!")

# Target distribution check
print("\n" + "-" * 80)
print("TARGET DISTRIBUTION CHECK")
print("-" * 80)

targets = ['valence', 'energy', 'danceability', 'popularity']
print(f"\n{'Target':<15s} {'Train Mean':<12s} {'Val Mean':<12s} {'Test Mean':<12s}")
print("-" * 80)

for target in targets:
    train_mean = df_train[target].mean()
    val_mean = df_val[target].mean()
    test_mean = df_test[target].mean()
    print(f"{target:<15s} {train_mean:<12.4f} {val_mean:<12.4f} {test_mean:<12.4f}")

print("\nNote: Means should be similar across splits")

# Save splits
print("\n" + "-" * 80)
print("SAVING SPLITS")
print("-" * 80)

train_path = output_dir / 'train.csv'
val_path = output_dir / 'val.csv'
test_path = output_dir / 'test.csv'

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"✅ Train saved to: {train_path}")
print(f"✅ Val saved to:   {val_path}")
print(f"✅ Test saved to:  {test_path}")

# Final summary
print("\n" + "=" * 80)
print("SPLIT COMPLETE!")
print("=" * 80)
print(f"\nFinal dataset sizes:")
print(f"  Train: {len(df_train):,} songs ({len(df_train)/len(df)*100:.1f}%) - {len(train_artists):,} artists")
print(f"  Val:   {len(df_val):,} songs ({len(df_val)/len(df)*100:.1f}%) - {len(val_artists):,} artists")
print(f"  Test:  {len(df_test):,} songs ({len(df_test)/len(df)*100:.1f}%) - {len(test_artists):,} artists")
print(f"  Total: {len(df):,} songs - {df['artists'].nunique():,} artists")

print("\n✅ Ready for baseline model training!")
print("=" * 80)
