"""
Sentiment Feature Extraction
Extracts sentiment features from song lyrics using TextBlob

Features extracted:
- sentiment_polarity: Emotional tone from -1 (negative) to +1 (positive)
- sentiment_subjectivity: Opinion level from 0 (objective) to 1 (subjective)

Note: TextBlob is optimized for English text
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob

print("=" * 80)
print("SENTIMENT FEATURE EXTRACTION")
print("=" * 80)

def extract_sentiment(lyrics):
    """
    Extract sentiment features using TextBlob
    
    Args:
        lyrics: String containing song lyrics
    
    Returns:
        Dictionary with sentiment polarity and subjectivity
    """
    if pd.isna(lyrics) or not isinstance(lyrics, str) or len(lyrics.strip()) == 0:
        # Return neutral for missing/empty lyrics
        return {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0
        }
    
    try:
        blob = TextBlob(lyrics)
        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity
        }
    except Exception as e:
        # If TextBlob fails, return neutral
        return {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0
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

# Extract sentiment features
print("\n" + "-" * 80)
print("EXTRACTING SENTIMENT FEATURES")
print("-" * 80)
print("Using TextBlob (optimized for English)")
print("This may take 10-20 minutes for large datasets...")

print("\nProcessing training set...")
train_sentiment = []
for lyrics in tqdm(df_train['lyrics'], desc='Train'):
    train_sentiment.append(extract_sentiment(lyrics))
df_train_sentiment = pd.DataFrame(train_sentiment)

print("Processing validation set...")
val_sentiment = []
for lyrics in tqdm(df_val['lyrics'], desc='Val'):
    val_sentiment.append(extract_sentiment(lyrics))
df_val_sentiment = pd.DataFrame(val_sentiment)

print("Processing test set...")
test_sentiment = []
for lyrics in tqdm(df_test['lyrics'], desc='Test'):
    test_sentiment.append(extract_sentiment(lyrics))
df_test_sentiment = pd.DataFrame(test_sentiment)

# Show sentiment summary
print("\n" + "-" * 80)
print("SENTIMENT STATISTICS SUMMARY (Training Set)")
print("-" * 80)
print(df_train_sentiment.describe())

print("\n" + "-" * 80)
print("SENTIMENT DISTRIBUTION")
print("-" * 80)

# Categorize polarity
def categorize_polarity(polarity):
    if polarity < -0.1:
        return 'negative'
    elif polarity > 0.1:
        return 'positive'
    else:
        return 'neutral'

polarity_dist = df_train_sentiment['sentiment_polarity'].apply(categorize_polarity).value_counts()
print("\nPolarity distribution (training set):")
for category, count in polarity_dist.items():
    print(f"  {category:8s}: {count:,} songs ({100 * count / len(df_train_sentiment):.1f}%)")

# Convert to numpy arrays
X_train_sentiment = df_train_sentiment.values
X_val_sentiment = df_val_sentiment.values
X_test_sentiment = df_test_sentiment.values

print("\n" + "-" * 80)
print("FEATURE MATRICES")
print("-" * 80)
print(f"Train shape: {X_train_sentiment.shape}")
print(f"Val shape:   {X_val_sentiment.shape}")
print(f"Test shape:  {X_test_sentiment.shape}")

# Save features
print("\n" + "-" * 80)
print("SAVING SENTIMENT FEATURES")
print("-" * 80)

np.save(features_dir / 'X_train_sentiment.npy', X_train_sentiment)
np.save(features_dir / 'X_val_sentiment.npy', X_val_sentiment)
np.save(features_dir / 'X_test_sentiment.npy', X_test_sentiment)

print(f"✅ Saved to: {features_dir}/X_*_sentiment.npy")

# Save feature names
feature_names = list(df_train_sentiment.columns)
with open(features_dir / 'sentiment_feature_names.txt', 'w') as f:
    f.write("# Sentiment Feature Names\n")
    f.write(f"# Total: {len(feature_names)} features\n")
    f.write("# Model: TextBlob (English-optimized)\n\n")
    for feat in feature_names:
        f.write(f"{feat}\n")

print(f"✅ Feature names saved to: {features_dir}/sentiment_feature_names.txt")

print("\n" + "=" * 80)
print("SENTIMENT EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\n✅ Extracted {len(feature_names)} sentiment features")
print(f"✅ Ready for model training with combined features")
