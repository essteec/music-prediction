# EDA Findings and Data Preparation Actions

**Date**: November 26, 2025  
**Dataset**: `data/processed/english_ml_ready.csv`  
**Total Songs**: 550,860  
**Analysis**: Complete EDA executed in `notebooks/01_exploratory_data_analysis.ipynb`

---

## 📊 EXECUTIVE SUMMARY

### Data Quality Status
✅ **EXCELLENT** overall data quality:
- All target variables: 100% complete (no missing values)
- All audio features: 100% complete
- Lyrics: 99.83% complete (549,906 songs)
- Only issues: 60% missing album names (not needed for modeling)

### Critical Issues Identified
1. 🚨 **Severe distribution imbalances** requiring transformation
2. 🚨 **Artist-aware splitting mandatory** (data leakage risk)
3. 🚨 **Genre and temporal imbalances** need handling
4. ⚠️ **Feature scaling critical** for multiple features

---

## 🎯 TARGET VARIABLE ANALYSIS

### 1. Valence (Mood: 0=Sad, 1=Happy)
**Distribution**: 
- Mean: 0.465, Median: 0.447, Std: 0.249
- **Slightly left-skewed** (more sad songs)
- Range: [0.000, 0.998]

**Issues**:
- ⚠️ Not perfectly normal (slight negative skew)
- Temporal trend: Declining over time (music getting sadder)

**Actions**:
- ✅ StandardScaler sufficient for most algorithms
- Consider quantile transformation for neural nets
- Year feature will help (strong temporal trend)

**Top Correlations**:
- Danceability: +0.49 (strong positive)
- Duration: -0.21 (negative - shorter songs happier)
- Instrumentalness: -0.19 (vocal songs happier)

### 2. Energy (Intensity)
**Distribution**:
- Mean: 0.671, Median: 0.716, Std: 0.246
- **Right-skewed** (bias toward high energy)
- Range: [0.000, 1.000]

**Issues**:
- ⚠️ Skewed distribution
- Strong temporal trend: Increasing over time

**Actions**:
- ✅ StandardScaler sufficient
- Year feature critical (strong upward trend)
- Genre feature important (Rock/Electronic high, Classical/Jazz low)

**Top Correlations**:
- **Loudness: +0.78** (VERY STRONG - almost redundant!)
- Acousticness: -0.75 (strong negative)
- Tempo: +0.22 (moderate positive)

**⚠️ MULTICOLLINEARITY WARNING**: Energy ↔ Loudness correlation is 0.78. Consider:
- Using Ridge/Lasso regularization
- Or removing loudness feature
- Or using PCA for dimensionality reduction

### 3. Danceability
**Distribution**:
- Mean: 0.527, Median: 0.530, Std: 0.173
- **Nearly perfect normal distribution** ✅
- Range: [0.046, 0.988]

**Issues**:
- ✅ Best-distributed target
- Weak temporal trend

**Actions**:
- ✅ StandardScaler sufficient
- No transformation needed
- Genre feature important (Hip-Hop high, Classical low)

**Top Correlations**:
- Valence: +0.49 (happy songs are danceable)
- Instrumentalness: -0.19 (vocal songs more danceable)
- Tempo: -0.17 (surprisingly negative!)

### 4. Popularity
**Distribution**:
- Mean: 16.9, Median: 13.0, Std: 17.0
- **HEAVILY right-skewed** 🚨
- Range: [0, 91]
- Most songs at 0 popularity

**Issues**:
- 🚨 **Extreme skew** - not suitable for standard regression
- 🚨 **Massive outliers** (60-91 range)
- 🚨 **Very weak correlations** with all audio features (<0.15)
- Popularity is driven by external factors (marketing, artist fame)

**Actions** (CRITICAL):
```python
# Option 1: Log transformation (RECOMMENDED)
y_pop_transformed = np.log1p(df['popularity'])  # log(1 + x)

# Option 2: RobustScaler (less sensitive to outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
y_pop_scaled = scaler.fit_transform(df['popularity'].values.reshape(-1, 1))

# Option 3: Quantile transformation
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
y_pop_quantile = qt.fit_transform(df['popularity'].values.reshape(-1, 1))

# Option 4: Convert to classification
# Binary: popular (>median) vs unpopular (<=median)
df['is_popular'] = (df['popularity'] > df['popularity'].median()).astype(int)
```

**Top Correlations**:
- ALL < 0.15 (essentially unpredictable from audio alone)
- External features needed: artist popularity, marketing spend, release timing

**Expected Performance**: Very poor (R² < 0.20) - external factors dominate

---

## 🎵 AUDIO FEATURES ANALYSIS

### Features Requiring Power Transformation (Highly Skewed)

**1. Acousticness** 🚨
- Extremely right-skewed (most songs at 0)
- Action: Apply Yeo-Johnson or log transformation

**2. Instrumentalness** 🚨
- Extremely right-skewed (most songs at 0)
- Action: Apply Yeo-Johnson or log transformation

**3. Speechiness** 🚨
- Extremely right-skewed (most songs at 0)
- Action: Apply Yeo-Johnson or log transformation

```python
from sklearn.preprocessing import PowerTransformer

# Apply to highly skewed features
skewed_features = ['acousticness', 'instrumentalness', 'speechiness']
power_transformer = PowerTransformer(method='yeo-johnson')
df[skewed_features] = power_transformer.fit_transform(df[skewed_features])
```

### Features Requiring Standard Scaling

**4. Liveness**
- Slightly right-skewed but not extreme
- Action: StandardScaler

**5. Loudness** ✅
- Near-normal distribution
- Action: StandardScaler
- Range: [-50, 0] dB

**6. Tempo** ✅
- Near-normal distribution
- Action: StandardScaler
- Range: [40, 220] BPM

**7. Duration_ms**
- Slightly right-skewed
- Action: StandardScaler
- Consider log transformation if issues arise

### Categorical Features

**8. Key** (0-11)
- Already properly encoded ✅
- Discrete distribution across all keys
- **Recommendation**: Use **cyclical encoding**

```python
# Cyclical encoding for key (music wraps around: B→C)
df['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
df['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)
```

**9. Mode** (0=Minor, 1=Major)
- Binary feature ✅
- Already properly encoded
- ~65% Major keys (natural imbalance, acceptable)
- Action: Use as-is

---

## 🎸 GENRE ANALYSIS

### Distribution
- **10 genres total** (good - not too many)
- **Severe imbalance**:
  - Rock: 197,195 (35.8%) 🚨
  - Pop: 72,592 (13.2%)
  - Electronic: 69,775 (12.7%)
  - Classical: 12,180 (2.2%)

### Issues
1. 🚨 **Rock is 16x more common than Classical**
2. ⚠️ Models will be biased toward Rock patterns
3. ⚠️ One-hot encoding creates 10 features (manageable but consider alternatives)

### Genre Impact on Targets
**Strong predictors for**:
- ✅ Energy (Rock/Electronic high, Classical/Jazz low)
- ✅ Danceability (Hip-Hop high, Classical low)

**Weak predictors for**:
- ⚠️ Valence (similar across genres, 0.39-0.59 range)
- ⚠️ Popularity (all genres 13-24 range)

### Actions

**Option 1: Stratified Sampling** (RECOMMENDED)
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, 
    stratify=df['genre'],  # Maintain genre proportions
    random_state=42
)
```

**Option 2: One-Hot Encoding** (Simple)
```python
genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
# Creates 10 binary columns
```

**Option 3: Target Encoding** (Compact)
```python
# Encode genre by mean valence (or other target)
genre_means = df_train.groupby('genre')['valence'].mean()
df['genre_encoded'] = df['genre'].map(genre_means)
# Creates 1 numeric column
```

**Recommendation**: Use **one-hot encoding** initially (10 features is manageable), then experiment with target encoding if dimensionality becomes an issue.

---

## 📅 TEMPORAL ANALYSIS (Year)

### Distribution
- Range: 1900-2025
- **Severe recency bias**: 82% of songs from 2000-2025 🚨
- Peak: 2017 (68,000 songs)
- Pre-1950: <1,000 songs/year (very sparse)

### Temporal Trends
**Strong trends**:
- ✅ Energy: Increasing over time (0.2 → 0.9)
- ✅ Valence: Decreasing over time (0.7 → 0.42) - music getting sadder!

**Weak trends**:
- ⚠️ Danceability: Slight fluctuation
- ⚠️ Popularity: No clear trend (external factors)

### Issues
1. 🚨 Raw year values (1900-2025) will distort models
2. ⚠️ Pre-1950 data is unreliable (too sparse)
3. ⚠️ Recency bias will favor modern music patterns

### Actions

**Option 1: Normalize Year** (RECOMMENDED)
```python
df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
# Maps 1900→0.0, 2025→1.0
```

**Option 2: Years Since Baseline**
```python
df['years_since_1950'] = df['year'] - 1950
# Makes older music negative, modern music positive
```

**Option 3: Decade Binning**
```python
df['decade'] = (df['year'] // 10) * 10
# Groups into decades: 1900, 1910, ..., 2020
```

**Option 4: Filter Old Data**
```python
df_modern = df[df['year'] >= 1950]
# Removes sparse historical data
```

**Recommendation**: Use **normalized year** [0, 1] to prevent raw values from dominating the model.

---

## 🎤 ARTIST ANALYSIS

### Distribution
- **78,676 unique artists** 🚨
- **Severe long tail**:
  - 40,530 artists (51.5%) have only 1 song
  - 20,077 artists (25.5%) have 2-5 songs
  - 11,417 artists (14.5%) have >10 songs

### Top Artists
- "Various Artists": 2,453 songs
- Grateful Dead: 1,307 songs
- Aretha Franklin: 1,067 songs

### 🚨 CRITICAL ISSUE: Data Leakage Risk

**Problem**: If same artist appears in both train and test sets:
- Model learns artist-specific patterns
- Artificially inflates performance
- Test metrics are invalid

**Solution**: **Artist-Aware Splitting** (MANDATORY)

```python
from sklearn.model_selection import GroupShuffleSplit

# CORRECT: Split by artist to prevent leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['artists']))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Verify zero overlap
train_artists = set(df.iloc[train_idx]['artists'])
test_artists = set(df.iloc[test_idx]['artists'])
assert len(train_artists & test_artists) == 0, "Artist leakage detected!"
```

**❌ WRONG** (DO NOT USE):
```python
# Random split allows artist overlap
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

---

## 📝 LYRICS ANALYSIS

### Coverage
- ✅ **99.83% have valid lyrics** (549,906 songs)
- Only 954 songs (0.17%) without lyrics
- Excellent for text feature extraction

### Length Statistics (Sample of 10,000)
- Mean word count: 268 words
- Median: 221 words
- Range: 8 - 3,465 words
- **Right-skewed distribution** (long tail of very long lyrics)

### Issues
1. ⚠️ High variance (8 to 3,465 words)
2. ⚠️ Outliers (3000+ words, likely rap/hip-hop)
3. 954 songs without lyrics (0.17%)

### Actions

**1. Handle Missing Lyrics**
```python
# Option 1: Drop songs without lyrics (RECOMMENDED)
df = df[df['lyrics'].notna() & (df['lyrics'].str.len() > 0)]
# Loss: 954 songs (0.17%) - negligible

# Option 2: Impute with empty string
df['lyrics'].fillna('', inplace=True)
```

**2. Extract Text Statistics** ✅ (Already Implemented)
```python
def extract_text_statistics(lyrics):
    words = lyrics.split()
    unique_words = set(word.lower() for word in words)
    
    return {
        'word_count': len(words),
        'unique_word_count': len(unique_words),
        'unique_ratio': len(unique_words) / max(len(words), 1),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'char_count': len(lyrics)
    }
```

**3. Scale Word Count**
```python
# Log transformation for highly skewed word count
df['word_count_log'] = np.log1p(df['word_count'])

# Or robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['word_count_scaled'] = scaler.fit_transform(df['word_count'].values.reshape(-1, 1))
```

---

## 🔗 MULTICOLLINEARITY ISSUES

### High Correlations Between Features

**Energy ↔ Loudness: 0.78** 🚨
- Very high correlation (near-redundant)
- Energy is almost a proxy for loudness

**Energy ↔ Acousticness: -0.75** 🚨
- High negative correlation
- Acoustic songs are low energy

### Actions

**Option 1: Ridge/Lasso Regularization** (RECOMMENDED)
```python
from sklearn.linear_model import Ridge, Lasso

# Ridge handles multicollinearity by shrinking coefficients
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

**Option 2: Remove Redundant Features**
```python
# Drop loudness (energy is more interpretable)
features_to_use = [f for f in features if f != 'loudness']
```

**Option 3: PCA for Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X)
```

**Recommendation**: Use **Ridge regression** initially (handles multicollinearity automatically), then experiment with feature removal if needed.

---

## ✅ COMPLETE DATA PREPARATION CHECKLIST

### Phase 1: Data Cleaning
- [x] Load data ✅
- [ ] **Remove 954 songs without lyrics** (0.17% loss)
- [ ] **Verify no duplicates** (by track ID)
- [ ] **Check for data type consistency**
- [ ] **Document final dataset size**

### Phase 2: Feature Engineering - Audio Features

**Highly Skewed Features** (Apply Power Transformation):
- [ ] Acousticness → PowerTransformer (Yeo-Johnson)
- [ ] Instrumentalness → PowerTransformer (Yeo-Johnson)
- [ ] Speechiness → PowerTransformer (Yeo-Johnson)

**Normal/Near-Normal Features** (Apply Standard Scaling):
- [ ] Loudness → StandardScaler
- [ ] Tempo → StandardScaler
- [ ] Duration_ms → StandardScaler
- [ ] Liveness → StandardScaler

**Categorical Features**:
- [ ] Key → Cyclical encoding (sin/cos)
- [ ] Mode → Use as-is (already binary)

### Phase 3: Feature Engineering - Metadata

**Genre**:
- [ ] One-hot encoding (creates 10 features)
- [ ] OR Target encoding (creates 1 feature)
- [ ] Ensure stratified sampling in train/test split

**Year**:
- [ ] Normalize to [0, 1] range
- [ ] OR use years_since_1950
- [ ] Consider filtering pre-1950 data

### Phase 4: Feature Engineering - Text Features

**Text Statistics** (Already Implemented ✅):
- [x] word_count
- [x] unique_word_count
- [x] unique_ratio
- [x] avg_word_length
- [x] char_count

**Sentiment** (Already Implemented ✅):
- [x] sentiment_polarity (TextBlob)
- [x] sentiment_subjectivity (TextBlob)

**Scaling Text Features**:
- [ ] Log transform word_count (highly skewed)
- [ ] StandardScaler for all text statistics

### Phase 5: Target Variable Preparation

**Valence**:
- [ ] StandardScaler OR QuantileTransformer

**Energy**:
- [ ] StandardScaler OR QuantileTransformer

**Danceability**:
- [ ] StandardScaler (already near-normal)

**Popularity** 🚨:
- [ ] **Log transformation: np.log1p(popularity)**
- [ ] OR RobustScaler
- [ ] OR QuantileTransformer
- [ ] OR convert to binary classification

### Phase 6: Train/Val/Test Splitting 🚨 CRITICAL

**Artist-Aware Splitting** (MANDATORY):
```python
from sklearn.model_selection import GroupShuffleSplit

# First split: train vs temp (val+test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['artists']))

df_train = df.iloc[train_idx]
df_temp = df.iloc[temp_idx]

# Second split: val vs test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['artists']))

df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

# Verify zero artist overlap
train_artists = set(df_train['artists'])
val_artists = set(df_val['artists'])
test_artists = set(df_test['artists'])

assert len(train_artists & val_artists) == 0
assert len(train_artists & test_artists) == 0
assert len(val_artists & test_artists) == 0
```

**Stratified Sampling**:
- [ ] Maintain genre proportions in all splits
- [ ] Optional: Maintain year distribution

### Phase 7: Feature Scaling Pipeline

**Critical Rule**: Fit on training data ONLY, transform val/test
```python
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Fit on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all splits
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for deployment
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

### Phase 8: Validation

**Data Integrity Checks**:
- [ ] Verify no NaN values in features
- [ ] Verify no NaN values in targets
- [ ] Verify all features in correct range
- [ ] Verify zero artist overlap between splits
- [ ] Verify genre distribution is similar across splits
- [ ] Verify year distribution is similar across splits

**Correlation Checks**:
- [ ] Verify no perfect correlations (r=1.0)
- [ ] Document high correlations (r>0.7)
- [ ] Decide on handling multicollinearity

---

## 🚀 RECOMMENDED IMPLEMENTATION ORDER

### Week 1: Core Data Preparation
1. **Drop songs without lyrics** (954 songs, 0.17%)
2. **Implement artist-aware splitting** (70/15/15)
3. **Apply power transformation** to skewed features
4. **Apply standard scaling** to normal features
5. **Implement cyclical encoding** for key
6. **Normalize year** to [0, 1]
7. **One-hot encode genre** (10 features)

### Week 2: Target Preparation & Baseline
8. **Apply log transformation to popularity**
9. **Save all preprocessed features** to disk
10. **Train baseline models** (Mean, Linear, Ridge, XGBoost)
11. **Evaluate baseline performance**
12. **Document results**

### Week 3: Advanced Features (If Needed)
13. **Add embeddings** (if text features show promise)
14. **Experiment with target encoding** for genre
15. **Test different popularity transformations**
16. **Hyperparameter tuning**

---

## 📊 EXPECTED PERFORMANCE BENCHMARKS

Based on correlations and distributions:

| Target | Expected R² | Difficulty | Key Features |
|--------|-------------|------------|--------------|
| **Energy** | 0.60-0.80 | Easy | Loudness (+0.78), Acousticness (-0.75), Year |
| **Danceability** | 0.40-0.60 | Moderate | Valence (+0.49), Genre (Hip-Hop high), Tempo |
| **Valence** | 0.30-0.50 | Moderate | Danceability (+0.49), Text features, Genre |
| **Popularity** | 0.10-0.30 | Hard | External factors dominate, weak audio correlations |

---

## 🎓 IS CURRENT EDA SUFFICIENT?

### ✅ What We Have Covered
- Complete target variable analysis
- Full audio feature distributions
- Correlation analysis
- Genre patterns
- Temporal trends
- Lyrics statistics
- Artist distribution
- Data quality assessment

### ⚠️ What's Missing (For Advanced EDA)

**1. Interaction Effects**
- Genre × Year interactions
- Audio feature pairs (e.g., tempo × energy)
- Text length × valence

**2. Advanced Visualizations**
- Pair plots for top correlated features
- 3D scatter plots (e.g., valence vs energy vs danceability)
- Time series decomposition (trend/seasonality)
- Genre-specific correlation matrices

**3. Statistical Tests**
- ANOVA for genre differences
- T-tests for temporal changes
- Normality tests (Shapiro-Wilk)
- Homogeneity of variance tests

**4. Outlier Analysis**
- Detailed investigation of extreme values
- Clustering of outliers
- Impact assessment

**5. Text Analysis**
- Word clouds by genre
- Most common words by valence range
- Sentiment distribution by genre
- N-gram analysis

### 🎯 RECOMMENDATION

**Current EDA is SUFFICIENT** for:
✅ Understanding data quality  
✅ Identifying preprocessing needs  
✅ Planning feature engineering  
✅ Setting baseline expectations  
✅ Starting model training  

**Advanced EDA is OPTIONAL** for:
- Deep thesis analysis (if required)
- Publication-quality visualizations
- Discovering subtle patterns
- Validating assumptions

**NEXT STEPS**:
1. ✅ **IMPLEMENT DATA PREPARATION** (highest priority)
2. ✅ **TRAIN BASELINE MODELS** (validate preprocessing)
3. ⏸️ Consider advanced EDA only if:
   - Thesis advisor requests it
   - Models underperform unexpectedly
   - Need publication-quality figures

---

## 📝 FINAL CHECKLIST FOR DATA PREPARATION

### Before Model Training
- [ ] All skewed features transformed
- [ ] All features scaled
- [ ] Categorical features encoded
- [ ] Artist-aware splits created
- [ ] Zero artist overlap verified
- [ ] All transformations saved (for deployment)
- [ ] Data validation complete
- [ ] No NaN values in final dataset
- [ ] Feature matrix shapes correct
- [ ] Target distributions documented

### Documentation
- [ ] Data dictionary updated
- [ ] Preprocessing pipeline documented
- [ ] Transformation decisions justified
- [ ] Expected performance benchmarks set
- [ ] Validation strategy defined

---

**STATUS**: Ready to proceed with data preparation and baseline modeling ✅

**CRITICAL ACTIONS**:
1. 🚨 Implement artist-aware splitting (prevent data leakage)
2. 🚨 Transform popularity (log or robust scaling)
3. 🚨 Power transform skewed features (acousticness, instrumentalness, speechiness)
4. ✅ Apply standard scaling to all features
5. ✅ Cyclical encoding for key
6. ✅ Normalize year to [0, 1]

**NEXT PHASE**: Execute data preparation pipeline
