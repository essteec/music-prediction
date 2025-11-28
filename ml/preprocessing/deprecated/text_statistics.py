"""
Text Statistics Extraction
Extracts basic statistical features from song lyrics

Features extracted:
- word_count: Total number of words
- unique_word_count: Number of unique words
- unique_ratio: Vocabulary diversity (unique/total)
- avg_word_length: Average length of words
- char_count: Total character count
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("TEXT STATISTICS EXTRACTION")
print("=" * 80)

def extract_text_statistics(lyrics):
    """
    Extract basic statistical features from lyrics
    
    Args:
        lyrics: String containing song lyrics
    
    Returns:
        Dictionary with 5 text statistics
    """
    if pd.isna(lyrics) or not isinstance(lyrics, str) or len(lyrics.strip()) == 0:
        # Return zeros for missing/empty lyrics
        return {
            'word_count': 0,
            'unique_word_count': 0,
            'unique_ratio': 0.0,
            'avg_word_length': 0.0,
            'char_count': 0
        }
    
    # Basic processing
    words = lyrics.split()
    unique_words = set(word.lower() for word in words)
    
    # Calculate statistics
    word_count = len(words)
    unique_count = len(unique_words)
    unique_ratio = unique_count / max(word_count, 1)  # Avoid division by zero
    avg_word_len = np.mean([len(word) for word in words]) if words else 0.0
    char_count = len(lyrics)
    
    return {
        'word_count': word_count,
        'unique_word_count': unique_count,
        'unique_ratio': unique_ratio,
        'avg_word_length': avg_word_len,
        'char_count': char_count
    }

# Paths
processed_dir = Path('../../data/processed')
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

# Check for lyrics column
if 'lyrics' not in df_train.columns:
    raise ValueError("'lyrics' column not found in dataset!")

# Check lyrics availability
print("\n" + "-" * 80)
print("LYRICS AVAILABILITY CHECK")
print("-" * 80)

for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    missing = df['lyrics'].isna().sum()
    empty = (df['lyrics'].str.len() == 0).sum()
    available = len(df) - missing - empty
    print(f"{name:6s}: {available:,} / {len(df):,} songs have lyrics "
          f"({100 * available / len(df):.1f}%)")
    if missing > 0:
        print(f"         {missing:,} missing, {empty:,} empty")

# Extract text statistics
print("\n" + "-" * 80)
print("EXTRACTING TEXT STATISTICS")
print("-" * 80)

print("\nProcessing training set...")
train_stats = []
for lyrics in tqdm(df_train['lyrics'], desc='Train'):
    train_stats.append(extract_text_statistics(lyrics))
df_train_stats = pd.DataFrame(train_stats)

print("Processing validation set...")
val_stats = []
for lyrics in tqdm(df_val['lyrics'], desc='Val'):
    val_stats.append(extract_text_statistics(lyrics))
df_val_stats = pd.DataFrame(val_stats)

print("Processing test set...")
test_stats = []
for lyrics in tqdm(df_test['lyrics'], desc='Test'):
    test_stats.append(extract_text_statistics(lyrics))
df_test_stats = pd.DataFrame(test_stats)

# Show statistics summary
print("\n" + "-" * 80)
print("TEXT STATISTICS SUMMARY (Training Set)")
print("-" * 80)
print(df_train_stats.describe())

# Convert to numpy arrays
X_train_text_stats = df_train_stats.values
X_val_text_stats = df_val_stats.values
X_test_text_stats = df_test_stats.values

print("\n" + "-" * 80)
print("FEATURE MATRICES")
print("-" * 80)
print(f"Train shape: {X_train_text_stats.shape}")
print(f"Val shape:   {X_val_text_stats.shape}")
print(f"Test shape:  {X_test_text_stats.shape}")

# Save features
print("\n" + "-" * 80)
print("SAVING TEXT STATISTICS")
print("-" * 80)

np.save(features_dir / 'X_train_text_stats.npy', X_train_text_stats)
np.save(features_dir / 'X_val_text_stats.npy', X_val_text_stats)
np.save(features_dir / 'X_test_text_stats.npy', X_test_text_stats)

print(f"✅ Saved to: {features_dir}/X_*_text_stats.npy")

# Save feature names
feature_names = list(df_train_stats.columns)
with open(features_dir / 'text_stats_feature_names.txt', 'w') as f:
    f.write("# Text Statistics Feature Names\n")
    f.write(f"# Total: {len(feature_names)} features\n\n")
    for feat in feature_names:
        f.write(f"{feat}\n")

print(f"✅ Feature names saved to: {features_dir}/text_stats_feature_names.txt")

print("\n" + "=" * 80)
print("TEXT STATISTICS EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\n✅ Extracted {len(feature_names)} text statistics features")
print(f"✅ Ready for sentiment extraction (next step)")
