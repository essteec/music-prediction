# ML Pipeline Roadmap

Complete guide for implementing the machine learning pipeline for music valence prediction.

## 📋 Table of Contents

1. [Phase 1: Data Preparation](#phase-1-data-preparation)
2. [Phase 2: Feature Engineering](#phase-2-feature-engineering)
3. [Phase 3: Baseline Models](#phase-3-baseline-models)
4. [Phase 4: Advanced Models](#phase-4-advanced-models)
5. [Phase 5: Evaluation & Comparison](#phase-5-evaluation--comparison)
6. [Phase 6: Analysis & Insights](#phase-6-analysis--insights)

---

## Phase 1: Data Preparation

### 1.1 Data Loading & Inspection

**Goal**: Understand the dataset thoroughly

**Tasks**:
- [ ] Load full CSV efficiently (use chunking if needed)
- [ ] Document dataset size (rows, columns)
- [ ] Check data types
- [ ] Identify missing values
- [ ] Check for duplicates

**Code Example**:
```python
import pandas as pd

# Load data
df = pd.read_csv('dataset/songs_with_attributes_and_lyrics.csv')

# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Data types:\n{df.dtypes}")
```

**Deliverable**: Jupyter notebook `01_data_inspection.ipynb`

### 1.2 Exploratory Data Analysis (EDA)

**Goal**: Understand distributions and relationships

**Visualizations to Create**:
- [ ] Target variable (valence) distribution
- [ ] Feature distributions (histograms)
- [ ] Correlation heatmap (audio features)
- [ ] Valence by genre (boxplot)
- [ ] Valence over time (by year)
- [ ] Lyrics length distribution
- [ ] Feature relationships (pairplot)

**Key Questions**:
- Is valence balanced or skewed?
- Are there outliers?
- Which features correlate with valence?
- Are there missing patterns (e.g., all songs from certain year missing genre)?

**Deliverable**: Jupyter notebook `02_eda.ipynb` with visualizations

### 1.3 Data Cleaning

**Goal**: Prepare clean dataset for modeling

**Tasks**:
- [ ] Handle missing values:
  - Drop rows if too many missing features
  - Impute if few missing (median for numerical, mode for categorical)
- [ ] Remove duplicates (same track_id)
- [ ] Remove invalid entries:
  - Valence outside [0, 1]
  - Negative durations
  - Empty lyrics
- [ ] Outlier treatment:
  - Document extreme values
  - Decide: keep, cap, or remove
- [ ] Text cleaning:
  - Handle encoding issues
  - Remove non-lyrical text (e.g., "[Chorus]", "[Verse 1]")

**Code Structure**:
```python
def clean_dataset(df):
    """Clean and validate dataset"""
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['id'])
    
    # Remove invalid valence
    df_clean = df_clean[(df_clean['valence'] >= 0) & (df_clean['valence'] <= 1)]
    
    # Handle missing lyrics
    df_clean = df_clean[df_clean['lyrics'].notna()]
    df_clean = df_clean[df_clean['lyrics'].str.len() > 0]
    
    # ... more cleaning
    
    return df_clean
```

**Deliverable**: Script `dataset/scripts/data_cleaning.py`

---

## Phase 2: Feature Engineering

### 2.1 Audio Feature Engineering

**Goal**: Prepare audio features for modeling

**Tasks**:
- [ ] Feature scaling:
  - StandardScaler for features with different ranges
  - Keep valence as-is (already 0-1)
- [ ] Feature selection:
  - Remove highly correlated features (correlation > 0.95)
  - Consider PCA if needed (optional)
- [ ] Feature interactions:
  - Try combinations: energy × tempo, acousticness × instrumentalness
  - Polynomial features (degree 2) for non-linear relationships
- [ ] Feature documentation:
  - Create data dictionary

**Code Example**:
```python
from sklearn.preprocessing import StandardScaler

def scale_audio_features(df):
    """Scale audio features"""
    audio_features = ['energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 
                     'liveness', 'tempo', 'duration_ms']
    
    scaler = StandardScaler()
    df[audio_features] = scaler.fit_transform(df[audio_features])
    
    return df, scaler
```

**Deliverable**: Script `ml/preprocessing/audio_features.py`

### 2.2 Text Feature Engineering

**Goal**: Extract meaningful features from lyrics

**Three Approaches to Implement**:

#### Approach 1: TF-IDF (Baseline)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(lyrics_series, max_features=1000):
    """Extract TF-IDF features from lyrics"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=5,  # Ignore words appearing in <5 songs
        max_df=0.7,  # Ignore words appearing in >70% songs
        ngram_range=(1, 2)  # Unigrams and bigrams
    )
    
    tfidf_matrix = vectorizer.fit_transform(lyrics_series)
    return tfidf_matrix, vectorizer
```

#### Approach 2: Sentiment Features
```python
from textblob import TextBlob

def extract_sentiment_features(lyrics):
    """Extract sentiment scores from lyrics"""
    blob = TextBlob(lyrics)
    
    return {
        'sentiment_polarity': blob.sentiment.polarity,  # -1 to 1
        'sentiment_subjectivity': blob.sentiment.subjectivity,  # 0 to 1
        'word_count': len(lyrics.split()),
        'unique_words': len(set(lyrics.lower().split())),
        'avg_word_length': np.mean([len(word) for word in lyrics.split()])
    }
```

#### Approach 3: Word Embeddings (Advanced - Optional)
```python
import gensim.downloader as api
import numpy as np

# Load pre-trained embeddings
word2vec_model = api.load('word2vec-google-news-300')

def get_embedding_vector(lyrics, model):
    """Average word vectors for lyrics"""
    words = lyrics.lower().split()
    vectors = [model[word] for word in words if word in model]
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)
```

**Text Preprocessing**:
```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_lyrics(lyrics):
    """Clean and preprocess lyrics"""
    # Lowercase
    lyrics = lyrics.lower()
    
    # Remove section markers [Chorus], [Verse], etc.
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
    # Remove special characters (keep spaces and letters)
    lyrics = re.sub(r'[^a-z\s]', '', lyrics)
    
    # Remove extra whitespace
    lyrics = ' '.join(lyrics.split())
    
    return lyrics
```

**Deliverable**: Script `ml/preprocessing/text_features.py`

### 2.3 Metadata Feature Engineering

**Tasks**:
- [ ] Genre encoding:
  - One-hot encode if few genres (<20)
  - Label encode + embedding if many genres
  - Or: Target encode (mean valence per genre)
- [ ] Year normalization:
  - Scale or use decade bins
- [ ] Explicit flag: already binary

**Code Example**:
```python
def encode_genre(df):
    """One-hot encode genre"""
    return pd.get_dummies(df, columns=['genre'], prefix='genre')
```

**Deliverable**: Part of `ml/preprocessing/feature_engineering.py`

### 2.4 Train/Val/Test Split

**Goal**: Create proper data splits for evaluation

**Strategy**:
- 70% Train
- 15% Validation (for hyperparameter tuning)
- 15% Test (for final evaluation, touch only once)

**Important**:
- Set random seed for reproducibility
- Consider stratified split if treating as classification
- Ensure splits are time-aware if using temporal features

```python
from sklearn.model_selection import train_test_split

def create_splits(X, y, random_state=42):
    """Create train/val/test splits"""
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Second split: 15% val, 15% test (50-50 of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

**Deliverable**: Script `ml/preprocessing/data_splitting.py`

---

## Phase 3: Baseline Models

### 3.1 Simple Baselines

**Goal**: Establish minimum performance bars

**Models**:
1. **Mean Predictor**: Predict average valence for all songs
2. **Median Predictor**: Predict median valence
3. **Linear Regression** (audio features only)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Mean baseline
y_pred_mean = np.full_like(y_test, y_train.mean())
rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
print(f"Mean Baseline RMSE: {rmse_mean:.4f}")

# Simple linear regression
lr = LinearRegression()
lr.fit(X_train_audio, y_train)
y_pred_lr = lr.predict(X_test_audio)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}")
```

**Deliverable**: Script `ml/models/baseline.py`

### 3.2 Feature Importance Analysis

**Goal**: Understand which features matter

```python
# For linear models: coefficient magnitudes
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

# For tree models: built-in importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## Phase 4: Advanced Models

### 4.1 Regularized Linear Models

**Models**:
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (L1 + L2)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge with cross-validation for alpha
ridge = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
```

**Deliverable**: Script `ml/models/linear_models.py`

### 4.2 Tree-Based Models

**Models to Implement**:

#### Random Forest
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
```

#### XGBoost
```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
```

#### LightGBM
```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42
)

lgb_model.fit(X_train, y_train)
```

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

**Deliverable**: Script `ml/models/tree_models.py`

### 4.3 Neural Networks (Optional)

**Simple Feedforward Network**:
```python
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15
)

mlp.fit(X_train, y_train)
```

**Deliverable**: Script `ml/models/neural_models.py`

---

## Phase 5: Evaluation & Comparison

### 5.1 Metrics Calculation

**Standard Metrics**:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred, model_name):
    """Calculate all metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results = {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MSE': mse
    }
    
    return results
```

**Deliverable**: Script `ml/evaluation/metrics.py`

### 5.2 Cross-Validation

```python
from sklearn.model_selection import cross_val_score

def cross_validate_model(model, X, y, cv=5):
    """Perform k-fold cross-validation"""
    # Negative MSE (sklearn convention)
    mse_scores = -cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='neg_mean_squared_error'
    )
    
    rmse_scores = np.sqrt(mse_scores)
    
    print(f"CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    return rmse_scores
```

### 5.3 Model Comparison

**Create Comparison Table**:
```python
import pandas as pd

results_df = pd.DataFrame([
    evaluate_model(y_test, y_pred_lr, 'Linear Regression'),
    evaluate_model(y_test, y_pred_ridge, 'Ridge'),
    evaluate_model(y_test, y_pred_rf, 'Random Forest'),
    evaluate_model(y_test, y_pred_xgb, 'XGBoost'),
])

results_df = results_df.sort_values('RMSE')
print(results_df)
```

**Deliverable**: Part of evaluation framework

---

## Phase 6: Analysis & Insights

### 6.1 Visualization

**Predicted vs Actual**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred, model_name):
    """Scatter plot of predictions vs actual"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Perfect prediction line
    plt.xlabel('Actual Valence')
    plt.ylabel('Predicted Valence')
    plt.title(f'{model_name}: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name}_predictions.png')
    plt.close()
```

**Feature Importance**:
```python
def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top N important features"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png')
    plt.close()
```

**Deliverable**: Script `ml/evaluation/visualization.py`

### 6.2 Error Analysis

**Analyze Prediction Errors**:
```python
def analyze_errors(y_true, y_pred, df_test):
    """Analyze where model fails"""
    errors = np.abs(y_true - y_pred)
    df_test['error'] = errors
    
    # Worst predictions
    worst_predictions = df_test.nlargest(10, 'error')
    
    # Error by genre
    error_by_genre = df_test.groupby('genre')['error'].mean().sort_values(ascending=False)
    
    return worst_predictions, error_by_genre
```

### 6.3 Statistical Testing

**Compare Models Statistically**:
```python
from scipy import stats

def compare_models(errors1, errors2):
    """Paired t-test for model comparison"""
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    if p_value < 0.05:
        print(f"Models are significantly different (p={p_value:.4f})")
    else:
        print(f"No significant difference (p={p_value:.4f})")
```

---

## 🎯 Experiment Orchestration

### Main Experiment Runner

```python
# ml/experiments/run_experiment.py

import yaml
import joblib
from datetime import datetime

def run_experiment(config_path):
    """Run complete ML experiment"""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)
    
    # Train models
    models = {}
    results = []
    
    for model_config in config['models']:
        print(f"\nTraining {model_config['name']}...")
        
        model = create_model(model_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_config['name'])
        results.append(metrics)
        
        # Save model
        model_path = f"results/models/{model_config['name']}_{datetime.now():%Y%m%d}.pkl"
        joblib.dump(model, model_path)
        
        models[model_config['name']] = model
    
    # Compare results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/metrics/experiment_results.csv', index=False)
    
    print("\n=== Results ===")
    print(results_df)
    
    return models, results_df
```

### Config File Example

```yaml
# ml/experiments/configs/baseline.yaml

experiment_name: "Baseline Audio Only"
random_seed: 42

data:
  features: ["audio"]  # audio, lyrics, metadata, all
  target: "valence"
  
models:
  - name: "LinearRegression"
    type: "linear"
    params: {}
  
  - name: "Ridge"
    type: "linear"
    params:
      alpha: 1.0
  
  - name: "RandomForest"
    type: "tree"
    params:
      n_estimators: 200
      max_depth: 15
      random_state: 42
```

---

## 📊 Expected Results

### Baseline Performance Expectations

Based on similar research:

| Model | Expected RMSE | Expected R² |
|-------|---------------|-------------|
| Mean Baseline | ~0.25 | 0.00 |
| Linear Regression | 0.18-0.22 | 0.15-0.30 |
| Ridge/Lasso | 0.17-0.21 | 0.20-0.35 |
| Random Forest | 0.15-0.19 | 0.35-0.50 |
| XGBoost | 0.14-0.18 | 0.40-0.55 |

**Note**: These are estimates. Your results may vary!

### Feature Importance Hypothesis

**Expected Top Features for Valence**:
1. Energy (positive correlation)
2. Danceability (positive)
3. Acousticness (negative)
4. Sentiment polarity from lyrics (strong positive)
5. Mode (major vs minor)

---

## ✅ Success Checklist

### For Each Model:
- [ ] Train on training set
- [ ] Tune hyperparameters on validation set
- [ ] Evaluate on test set (only once!)
- [ ] Perform cross-validation
- [ ] Save model artifact
- [ ] Record all metrics
- [ ] Generate visualizations
- [ ] Document findings

### For Overall Project:
- [ ] At least 3 models compared
- [ ] Clear winner identified
- [ ] Feature importance analyzed
- [ ] Error patterns understood
- [ ] Results reproducible
- [ ] Code documented
- [ ] Figures saved for thesis

---

## 🎓 Thesis Integration

### Sections This Work Supports

**Methodology**:
- Data preprocessing steps
- Feature engineering approaches
- Model selection rationale
- Evaluation metrics choice

**Experiments**:
- Model configurations
- Hyperparameter settings
- Training procedures

**Results**:
- Performance comparison table
- Prediction plots
- Feature importance charts
- Error analysis

**Discussion**:
- Why certain models worked better
- Role of lyrics vs audio features
- Limitations discovered
- Future improvements

---

**Next Steps**: Start with Phase 1 (Data Preparation) and work through sequentially. Don't skip EDA - it provides crucial insights!

