# ML Pipeline Roadmap

**Iterative ML development guide for music valence prediction.**

⚠️ **Key Principle**: This is an iterative loop, not a waterfall. Each phase builds incrementally on validated baselines.

## 📋 Table of Contents

1. [Phase 1: Minimal Clean Dataset](#phase-1-minimal-clean-dataset)
2. [Phase 2: Audio-Only Baselines](#phase-2-audio-only-baselines)
3. [Phase 3: Lightweight Text Features](#phase-3-lightweight-text-features)
4. [Phase 4: Embedding-Based Text Features](#phase-4-embedding-based-text-features)
5. [Phase 5: Genre & Metadata Embeddings](#phase-5-genre--metadata-embeddings)
6. [Phase 6: Final Model & Error Analysis](#phase-6-final-model--error-analysis)

## 🔄 Development Loop

```
EDA → Feature → Baseline → Evaluate → Analyze → Fix → Iterate
```

**Do NOT complete all feature engineering before testing models.**

---

## Phase 1: Minimal Clean Dataset

### 1.1 Data Loading & Validation

**Goal**: Prepare a minimal, valid dataset for modeling

**Critical Tasks**:
- [ ] Load CSV with correct encoding
- [ ] Document dataset size (rows, columns)
- [ ] Remove invalid entries:
  - Valence outside [0, 1]
  - Negative durations
  - Empty lyrics
  - Duplicate `track_id`
- [ ] **Group-aware split by artist** (prevent data leakage)

**Code Example**:
```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load data
df = pd.read_csv('dataset/songs_with_attributes_and_lyrics.csv')

# Basic validation
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Remove invalid valence
df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]

# Remove empty lyrics
df = df[df['lyrics'].notna()]
df = df[df['lyrics'].str.len() > 0]

# Remove duplicates
df = df.drop_duplicates(subset=['id'])
```

**Deliverable**: Script `ml/preprocessing/data_cleaning.py`

### 1.2 Artist-Aware Data Split

**⚠️ CRITICAL**: Use artist-level grouping to prevent data leakage

**Why**: Songs by the same artist share style, which inflates performance if split randomly.

```python
from sklearn.model_selection import GroupShuffleSplit

def create_artist_aware_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create train/val/test splits grouped by artist_id
    Prevents data leakage from artist style
    """
    # First split: train vs temp (val+test)
    gss = GroupShuffleSplit(n_splits=1, test_size=(test_size + val_size), random_state=random_state)
    train_idx, temp_idx = next(gss.split(df, groups=df['artist_id']))
    
    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]
    
    # Second split: val vs test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['artist_id']))
    
    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]
    
    print(f"Train: {len(df_train)} songs, {df_train['artist_id'].nunique()} artists")
    print(f"Val: {len(df_val)} songs, {df_val['artist_id'].nunique()} artists")
    print(f"Test: {len(df_test)} songs, {df_test['artist_id'].nunique()} artists")
    
    return df_train, df_val, df_test
```

**Deliverable**: Script `ml/preprocessing/data_splitting.py`

### 1.3 Exploratory Data Analysis (EDA)

**Goal**: Understand distributions and relationships

**Key Visualizations**:
- [ ] Target variable (valence) distribution
- [ ] Audio feature distributions
- [ ] Correlation heatmap (audio features only)
- [ ] Valence by genre (boxplot)
- [ ] Lyrics length distribution
- [ ] Year distribution

**Key Questions**:
- Is valence balanced or skewed?
- Which audio features correlate with valence?
- Are there genre-specific valence patterns?
- Are certain genres consistently high/low valence?

**Deliverable**: Notebook `notebooks/01_data_profiling.ipynb`

---

## Phase 2: Audio-Only Baselines

### 2.1 Audio Feature Preparation

**Goal**: Prepare only audio features for initial modeling

**Audio Features to Use**:
- `energy`, `loudness`, `speechiness`, `acousticness`
- `instrumentalness`, `liveness`, `tempo`, `duration_ms`
- `danceability`, `mode`, `key`

**Feature Scaling**:
```python
from sklearn.preprocessing import StandardScaler

def prepare_audio_features(df_train, df_val, df_test):
    """Scale audio features using training statistics"""
    audio_features = [
        'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'tempo', 'duration_ms',
        'danceability', 'mode', 'key'
    ]
    
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(df_train[audio_features])
    X_val = scaler.transform(df_val[audio_features])
    X_test = scaler.transform(df_test[audio_features])
    
    return X_train, X_val, X_test, scaler
```

**No Polynomial Features**: Start simple. Only add if testing shows improvement.

**Deliverable**: Script `ml/preprocessing/audio_features.py`

### 2.2 Baseline Model Sequence

**Fixed Order** (do not skip):

1. **Mean Predictor** (sanity check)
2. **Linear Regression** (interpretable baseline)
3. **Ridge Regression** (regularized baseline)
4. **XGBoost** (strong tree baseline)

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

# 1. Mean baseline
y_pred_mean = np.full_like(y_val, y_train.mean())
print(f"Mean Baseline RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_mean)):.4f}")

# 2. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_lr)):.4f}")
print(f"Linear Regression R²: {r2_score(y_val, y_pred_lr):.4f}")

# 3. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)
print(f"Ridge RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_ridge)):.4f}")
print(f"Ridge R²: {r2_score(y_val, y_pred_ridge):.4f}")

# 4. XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_xgb)):.4f}")
print(f"XGBoost R²: {r2_score(y_val, y_pred_xgb):.4f}")
```

**Expected Performance (Audio-Only)**:
- Mean Baseline: RMSE ~0.25
- Linear Regression: RMSE 0.18-0.22
- Ridge: RMSE 0.17-0.21
- XGBoost: RMSE 0.15-0.19

**Deliverable**: Script `ml/models/audio_baseline.py`

### 2.3 Feature Importance Analysis

```python
# Audio feature importance from XGBoost
feature_importance = pd.DataFrame({
    'feature': audio_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
```

**Expected Top Features**:
1. Energy (positive correlation with valence)
2. Danceability
3. Acousticness (negative correlation)
4. Mode (major vs minor key)

**Deliverable**: Part of evaluation

---

## Phase 3: Lightweight Text Features

### 3.1 Statistical Lyric Features

**Goal**: Add cheap, interpretable text features

**Features to Extract** (fast, no ML needed):
```python
def extract_text_statistics(lyrics):
    """Extract basic lyric statistics"""
    words = lyrics.split()
    unique_words = set(word.lower() for word in words)
    
    return {
        'word_count': len(words),
        'unique_word_count': len(unique_words),
        'unique_ratio': len(unique_words) / max(len(words), 1),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'char_count': len(lyrics)
    }

# Apply to all lyrics
text_stats = df['lyrics'].apply(extract_text_statistics)
df_text_stats = pd.DataFrame(text_stats.tolist())
```

**Deliverable**: Script `ml/preprocessing/text_statistics.py`

### 3.2 Sentiment Extraction

**⚠️ Use TextBlob for English Songs**

**Model**: `TextBlob`

**Why**: 
- Simple and effective for English text
- Fast processing
- Provides polarity and subjectivity scores

```python
from textblob import TextBlob
import numpy as np

def extract_sentiment(text):
    """Extract sentiment scores using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Range: -1 to 1
        subjectivity = blob.sentiment.subjectivity  # Range: 0 to 1
        
        return {
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity
        }
    except:
        return {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0
        }

# Apply to lyrics
sentiment_features = df['lyrics'].apply(extract_sentiment)
df_sentiment = pd.DataFrame(sentiment_features.tolist())
```

**Deliverable**: Script `ml/preprocessing/sentiment_features.py`

### 3.3 Retrain Models with Text Features

**Combine**: Audio + Text Statistics + Sentiment

```python
# Prepare combined features
X_train_combined = np.hstack([X_train_audio, X_train_text_stats, X_train_sentiment])
X_val_combined = np.hstack([X_val_audio, X_val_text_stats, X_val_sentiment])

# Retrain XGBoost
xgb_text = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
xgb_text.fit(X_train_combined, y_train)

y_pred_xgb_text = xgb_text.predict(X_val_combined)
print(f"XGBoost + Text RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_xgb_text)):.4f}")
```

**Evaluate Improvement**:
- Compare RMSE: Audio-only vs Audio+Text
- Check if sentiment features have high importance
- If improvement < 0.01 RMSE, text features may not be useful

**Deliverable**: Update to model training scripts

---

## Phase 4: Embedding-Based Text Features

### 4.1 Compute & Cache Lyric Embeddings

**⚠️ CRITICAL**: Compute embeddings ONCE and save to disk

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (English optimized)

**Why**:
- Optimized for English text
- Compact (384 dimensions)
- Fast inference
- Better than TF-IDF for semantic meaning

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

def compute_lyric_embeddings(lyrics_list, model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Compute embeddings for all lyrics and save to disk
    
    ⚠️ Run this ONCE and cache results
    """
    model = SentenceTransformer(model_name)
    
    # Compute in batches to avoid memory issues
    embeddings = model.encode(
        lyrics_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings

# Compute for all splits
print("Computing embeddings for training set...")
train_embeddings = compute_lyric_embeddings(df_train['lyrics'].tolist())

print("Computing embeddings for validation set...")
val_embeddings = compute_lyric_embeddings(df_val['lyrics'].tolist())

print("Computing embeddings for test set...")
test_embeddings = compute_lyric_embeddings(df_test['lyrics'].tolist())

# Save to disk
joblib.dump(train_embeddings, 'dataset/processed/train_embeddings.pkl')
joblib.dump(val_embeddings, 'dataset/processed/val_embeddings.pkl')
joblib.dump(test_embeddings, 'dataset/processed/test_embeddings.pkl')

print(f"Embeddings shape: {train_embeddings.shape}")  # (n_samples, 384)
```

**Load Cached Embeddings**:
```python
# In future runs, just load
train_embeddings = joblib.load('dataset/processed/train_embeddings.pkl')
val_embeddings = joblib.load('dataset/processed/val_embeddings.pkl')
test_embeddings = joblib.load('dataset/processed/test_embeddings.pkl')
```

**Deliverable**: Script `ml/preprocessing/embeddings.py`

### 4.2 Train Models with Embeddings

**Combine**: Audio + Text Stats + Sentiment + Embeddings

```python
# Combine all features
X_train_full = np.hstack([X_train_audio, X_train_text_stats, X_train_sentiment, train_embeddings])
X_val_full = np.hstack([X_val_audio, X_val_text_stats, X_val_sentiment, val_embeddings])

# Train XGBoost
xgb_full = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_full.fit(X_train_full, y_train)

# Train LightGBM (faster for high-dimensional data)
import lightgbm as lgb

lgb_full = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
lgb_full.fit(X_train_full, y_train)

# Evaluate both
y_pred_xgb_full = xgb_full.predict(X_val_full)
y_pred_lgb_full = lgb_full.predict(X_val_full)

print(f"XGBoost + Embeddings RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_xgb_full)):.4f}")
print(f"LightGBM + Embeddings RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_lgb_full)):.4f}")
```

**Why Skip TF-IDF**:
- TF-IDF creates sparse, high-dimensional features (even with max_features=1000)
- Embeddings are dense, semantic, and compact (384-d)
- For 700k songs, TF-IDF is memory-intensive and slow
- Use TF-IDF only for small-scale benchmarking if curious

**Deliverable**: Script `ml/models/embedding_models.py`

---

## Phase 5: Genre & Metadata Embeddings

### 5.1 Genre Encoding Strategy

**Problem**: One-hot encoding creates high dimensionality if many genres

**Solution**: Use embeddings or target encoding

**Option 1: Target Encoding** (simple, effective)
```python
def target_encode_genre(df_train, df_val, df_test):
    """Encode genre by mean valence"""
    genre_means = df_train.groupby('genre')['valence'].mean()
    
    df_train['genre_encoded'] = df_train['genre'].map(genre_means)
    df_val['genre_encoded'] = df_val['genre'].map(genre_means)
    df_test['genre_encoded'] = df_test['genre'].map(genre_means)
    
    # Fill unknown genres with global mean
    global_mean = df_train['valence'].mean()
    df_train['genre_encoded'].fillna(global_mean, inplace=True)
    df_val['genre_encoded'].fillna(global_mean, inplace=True)
    df_test['genre_encoded'].fillna(global_mean, inplace=True)
    
    return df_train, df_val, df_test
```

**Option 2: Learned Embeddings** (for neural models only)
```python
from sklearn.preprocessing import LabelEncoder

# Encode genres as integers
le = LabelEncoder()
genre_encoded = le.fit_transform(df['genre'])

# Use in embedding layer (PyTorch example)
import torch.nn as nn

class GenreEmbedding(nn.Module):
    def __init__(self, num_genres, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_genres, embedding_dim)
    
    def forward(self, genre_ids):
        return self.embedding(genre_ids)
```

**Deliverable**: Script `ml/preprocessing/metadata_features.py`

### 5.2 Year Normalization

```python
def normalize_year(df):
    """Scale year to reasonable range"""
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    return df
```

### 5.3 Retrain with All Features

**Final Feature Set**:
- Audio features (scaled)
- Text statistics
- Sentiment scores
- Lyric embeddings (384-d)
- Genre encoding
- Year (normalized)
- Explicit flag

```python
# Combine all
X_train_final = np.hstack([
    X_train_audio,
    X_train_text_stats,
    X_train_sentiment,
    train_embeddings,
    df_train[['genre_encoded', 'year_normalized', 'explicit']].values
])

# Train final model
final_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    random_state=42
)
final_model.fit(X_train_final, y_train)
```

**Deliverable**: Script `ml/models/final_model.py`

---

## Phase 6: Final Model & Error Analysis

### 6.1 Test Set Evaluation

**⚠️ ONLY evaluate on test set ONCE**

```python
# Prepare test features
X_test_final = np.hstack([
    X_test_audio,
    X_test_text_stats,
    X_test_sentiment,
    test_embeddings,
    df_test[['genre_encoded', 'year_normalized', 'explicit']].values
])

# Final prediction
y_pred_final = final_model.predict(X_test_final)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
mae = mean_absolute_error(y_test, y_pred_final)
r2 = r2_score(y_test, y_pred_final)

print(f"Final Test RMSE: {rmse:.4f}")
print(f"Final Test MAE: {mae:.4f}")
print(f"Final Test R²: {r2:.4f}")
```

**Deliverable**: Script `ml/evaluation/final_evaluation.py`

### 6.2 Error Analysis by Segment

**Segment Errors by**:
1. Genre
2. Artist
3. Valence range (low/mid/high)

```python
def analyze_errors_by_segment(df_test, y_test, y_pred):
    """Analyze errors by different segments"""
    df_test['error'] = np.abs(y_test - y_pred)
    df_test['prediction'] = y_pred
    
    # Error by genre
    error_by_genre = df_test.groupby('genre')['error'].agg(['mean', 'std', 'count'])
    print("\nError by Genre:")
    print(error_by_genre.sort_values('mean', ascending=False))
    
    # Error by valence range
    df_test['valence_bin'] = pd.cut(y_test, bins=[0, 0.33, 0.67, 1.0], labels=['low', 'mid', 'high'])
    error_by_valence = df_test.groupby('valence_bin')['error'].agg(['mean', 'std', 'count'])
    print("\nError by Valence Range:")
    print(error_by_valence)
    
    # Worst predictions
    worst_10 = df_test.nlargest(10, 'error')[['name', 'artist', 'genre', 'valence', 'prediction', 'error']]
    print("\nWorst 10 Predictions:")
    print(worst_10)
    
    return error_by_genre, error_by_valence

# Run analysis
analyze_errors_by_segment(df_test.copy(), y_test, y_pred_final)
```

**Deliverable**: Script `ml/evaluation/error_analysis.py`

### 6.3 Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 20 features
    print("\nTop 20 Features:")
    print(importance_df.head(20))
    
    # Feature groups
    audio_importance = importance_df[importance_df['feature'].isin(audio_features)]['importance'].sum()
    text_importance = importance_df[importance_df['feature'].str.contains('text_|sentiment_|embed_')]['importance'].sum()
    meta_importance = importance_df[importance_df['feature'].isin(['genre_encoded', 'year_normalized', 'explicit'])]['importance'].sum()
    
    print(f"\nFeature Group Importance:")
    print(f"Audio: {audio_importance:.2%}")
    print(f"Text: {text_importance:.2%}")
    print(f"Metadata: {meta_importance:.2%}")
    
    return importance_df

# Create feature names
feature_names = (
    audio_features + 
    list(text_stat_features) + 
    list(sentiment_features) + 
    [f'embed_{i}' for i in range(384)] +
    ['genre_encoded', 'year_normalized', 'explicit']
)

analyze_feature_importance(final_model, feature_names)
```

**Deliverable**: Part of evaluation

### 6.4 Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(y_test, y_pred, save_dir='results/figures'):
    """Create all evaluation plots"""
    
    # 1. Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Actual Valence')
    plt.ylabel('Predicted Valence')
    plt.title('Predicted vs Actual Valence')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predicted_vs_actual.png', dpi=300)
    plt.close()
    
    # 2. Error distribution
    errors = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png', dpi=300)
    plt.close()
    
    # 3. Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, errors, alpha=0.3, s=10)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Valence')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/residuals.png', dpi=300)
    plt.close()

create_visualizations(y_test, y_pred_final)
```

**Deliverable**: Script `ml/evaluation/visualization.py`

---

## 🎯 Model Comparison Priority

**Fixed Comparison Order** (do not deviate):

1. **Audio-Only Baseline**: Linear → Ridge → XGBoost
2. **+ Lightweight Text**: XGBoost with stats + sentiment
3. **+ Embeddings**: XGBoost + LightGBM with embeddings
4. **+ Metadata**: Final model with genre/year
5. **MLP (Optional)**: Only if embeddings exist and you have time

**When to Stop**:
- If improvement plateaus (<0.01 RMSE gain)
- If validation performance degrades
- If computational cost becomes prohibitive

---

## 📊 Expected Results

### Performance Targets

| Model Configuration | Expected RMSE | Expected R² |
|---------------------|---------------|-------------|
| Mean Baseline | ~0.25 | 0.00 |
| Audio-Only (Ridge) | 0.17-0.21 | 0.20-0.35 |
| Audio-Only (XGBoost) | 0.15-0.19 | 0.35-0.50 |
| + Lightweight Text | 0.13-0.17 | 0.45-0.60 |
| + Embeddings | 0.11-0.15 | 0.55-0.70 |
| + Metadata (Final) | 0.10-0.14 | 0.60-0.75 |

**Note**: Valence is inherently subjective. R² > 0.70 would be exceptional.

### Feature Importance Hypothesis

**Expected Top Features**:
1. `sentiment_polarity` (strong positive correlation)
2. `energy` (positive)
3. `danceability` (positive)
4. `mode` (major key = higher valence)
5. `acousticness` (negative)
6. Embedding dimensions capturing emotional language
7. `genre_encoded` (genre-specific patterns)

---

## ✅ Critical Checklist

### Data Integrity
- [ ] Artist-level group split used (NO random split)
- [ ] Language detected for all lyrics
- [ ] No test data leakage

### Feature Engineering
- [ ] Embeddings computed ONCE and cached
- [ ] Multilingual sentiment model used (NOT TextBlob)
- [ ] Features scaled using training statistics only

### Model Development
- [ ] Iterate: train → evaluate → improve → repeat
- [ ] Validate on validation set, test on test set ONCE
- [ ] Document performance at each iteration

### Error Analysis
- [ ] Segment errors by genre, artist
- [ ] Identify systematic failure modes
- [ ] Understand model limitations

### Reproducibility
- [ ] Random seeds set everywhere
- [ ] Data versions tracked
- [ ] Model artifacts saved
- [ ] Experiment configs documented

---

## 🚨 Common Pitfalls to Avoid

1. ❌ **Random train/test split** → Use artist-aware grouping
2. ❌ **Computing embeddings multiple times** → Cache to disk
3. ❌ **TF-IDF as primary text representation** → Use embeddings
4. ❌ **Testing all models before iteration** → Build incrementally
5. ❌ **Ignoring compute constraints** → Plan memory/time budgets
6. ❌ **No error segmentation** → Analyze by genre
7. ❌ **Overfitting to validation set** → Touch test set ONCE

---

## 🔧 Compute Constraints Planning

### Memory Budget
- **Embeddings**: ~700k songs × 384 dims × 4 bytes ≈ 1 GB (manageable)
- **TF-IDF**: ~700k songs × 1000 features × 4 bytes ≈ 2.8 GB (sparse, but avoid)
- **Recommendation**: Use embeddings, cache to disk

### Time Budget
- **Embedding computation**: ~30-60 min for English songs (one-time)
- **Sentiment extraction**: ~10-20 min with TextBlob (one-time, batch process)
- **Model training**: 
  - XGBoost: 5-15 min per run
  - LightGBM: 2-8 min per run
  - MLP: 10-30 min per run

### Storage Budget
- **Raw data**: ~500 MB
- **Processed features**: ~1-2 GB
- **Model artifacts**: ~100 MB per model
- **Total**: <5 GB (reasonable)

---

## 🎓 Thesis Integration

### Methodology Chapter
Document:
- Artist-aware splitting rationale
- Feature engineering decisions
- Model selection criteria

### Experiments Chapter
Report:
- Iterative development process
- Performance at each phase
- Hyperparameter tuning results
- Computational costs

### Results Chapter
Present:
- Model comparison table
- Feature importance analysis
- Error analysis by segment
- Prediction visualizations

### Discussion Chapter
Analyze:
- Why embeddings outperform TF-IDF
- Genre effects on valence prediction
- Limitations and future work

---

## 🔄 Iteration Strategy

### Loop 1: Audio Baseline
- Prepare audio features
- Train Linear, Ridge, XGBoost
- Evaluate on validation set
- **Decision Point**: Is performance reasonable? (RMSE < 0.20)

### Loop 2: Add Lightweight Text
- Extract text statistics
- Compute multilingual sentiment
- Retrain XGBoost
- **Decision Point**: Did text improve performance? (ΔRMSE > 0.01)

### Loop 3: Add Embeddings
- Compute and cache embeddings
- Train XGBoost + LightGBM
- Evaluate improvement
- **Decision Point**: Are embeddings worth the cost? (ΔRMSE > 0.02)

### Loop 4: Add Metadata
- Encode genre and year
- Train final model
- Perform error analysis
- **Decision Point**: Ready for test set?

### Loop 5: Final Evaluation
- Evaluate on test set ONCE
- Create visualizations
- Document findings
- **Done**: Proceed to thesis writing

---

**Next Steps**: Start with Phase 1. Focus on getting a clean dataset with proper artist-aware splits before any modeling.

