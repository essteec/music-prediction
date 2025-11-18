# Critical Corrections to ML Pipeline

**Date**: November 12, 2025  
**Status**: ROADMAP UPDATED - Corrected from waterfall to iterative approach

## 🚨 Critical Issues Fixed

### 1. **Artist-Level Data Leakage** ✅ FIXED
**Problem**: Original roadmap used random train/test split  
**Impact**: Songs by same artist in both sets → inflated performance  
**Solution**: Use `GroupShuffleSplit` by `artist_id` (MANDATORY)

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['artist_id']))
```

### 2. **Text Sentiment Analysis** ✅ FIXED
**Problem**: Need efficient sentiment extraction for English lyrics  
**Impact**: Sentiment features are critical for valence prediction  
**Solution**: Use `TextBlob` for English-only dataset

**Dependencies**:
```bash
pip install textblob
```

### 3. **TF-IDF Computational Cost** ✅ FIXED
**Problem**: 700k songs × 1000 TF-IDF features = memory-intensive  
**Impact**: Slow training, sparse representations  
**Solution**: Use dense embeddings as primary text representation

**Primary Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Compact: 384 dimensions (vs 1000+ for TF-IDF)
- Semantic: Captures meaning, not just word frequency
- Optimized for English text

**Dependencies**:
```bash
pip install sentence-transformers
```

### 4. **Embedding Recomputation** ✅ FIXED
**Problem**: Computing embeddings takes hours for 700k songs  
**Impact**: Wasted time if recomputed each experiment  
**Solution**: Cache embeddings to disk (MANDATORY)

```python
import joblib

# Compute ONCE
embeddings = model.encode(lyrics_list, batch_size=64)
joblib.dump(embeddings, 'dataset/processed/train_embeddings.pkl')

# Reuse in all future experiments
embeddings = joblib.load('dataset/processed/train_embeddings.pkl')
```

### 5. **Sequential Waterfall Approach** ✅ FIXED
**Problem**: Original roadmap implied complete feature engineering before modeling  
**Impact**: Wasted time on features that may not improve performance  
**Solution**: Iterative development loop

**New Loop Structure**:
```
Phase 1: Audio-only → Train → Evaluate → Document baseline
Phase 2: + Lightweight text → Retrain → Compare improvement
Phase 3: + Embeddings → Retrain → Compare improvement
Phase 4: + Metadata → Final model
Phase 5: Test set evaluation ONCE → Error analysis
```

### 6. **English-Only Dataset** ✅ UPDATED
**Decision**: Using filtered English-only dataset  
**Impact**: Simplified NLP pipeline, faster processing  
**Solution**: Use English-optimized models and remove multilingual overhead

**Benefits**:
- Faster sentiment analysis with TextBlob
- English-optimized embeddings (all-MiniLM-L6-v2)
- No language detection needed
- Simplified error analysis

---

## 📊 Updated Phase Structure

### Phase 1: Minimal Clean Dataset
- Load & validate data
- Remove duplicates & invalid entries
- **Artist-aware data split** (GroupShuffleSplit)
- EDA (audio features + genre distribution)

### Phase 2: Audio-Only Baselines
- Scale audio features (no polynomial features yet)
- Train: Mean → Linear → Ridge → XGBoost
- Establish performance floor (RMSE ~0.15-0.20)
- Document feature importance

### Phase 3: Lightweight Text Features
- Extract: word count, unique ratio, avg word length
- **Sentiment analysis** (TextBlob for English)
- Retrain XGBoost
- Evaluate: Did text improve? (ΔRMSE > 0.01?)

### Phase 4: Embedding-Based Text Features
- **Compute embeddings ONCE** (MiniLM-384d)
- **Cache to disk** (joblib)
- Train: XGBoost + LightGBM
- Evaluate: Are embeddings worth it? (ΔRMSE > 0.02?)

### Phase 5: Genre & Metadata
- **Target encoding** for genre (NOT one-hot)
- Normalize year
- Final model training

### Phase 6: Final Evaluation & Analysis
- Test set evaluation **ONCE**
- **Error segmentation**: by genre, artist, valence range
- Feature importance analysis
- Visualizations for thesis

---

## 🔧 Updated Dependencies

### Required (Not Optional)
```txt
# Core ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0

# NLP (English-optimized)
sentence-transformers>=2.2.0
textblob>=0.17.0
torch>=2.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib  # Caching
tqdm    # Progress bars
pyyaml  # Configs
```

### Removed (Not Needed)
- ❌ `transformers` (not needed for English-only dataset)
- ❌ `langdetect` (not needed for English-only dataset)
- ❌ `gensim` (word2vec not needed with sentence-transformers)

### Optional (Benchmarking Only)
- `nltk` (tokenization if needed)
- `spacy` (if advanced preprocessing needed)

---

## ⚠️ Critical Rules

### 1. **NEVER use random train/test split**
✅ **Always** use `GroupShuffleSplit` by `artist_id`

### 2. **Use TextBlob for English sentiment**
✅ **Use** TextBlob for fast, effective English sentiment analysis

### 3. **NEVER compute embeddings multiple times**
✅ **Always** cache with `joblib.dump()` and reuse with `joblib.load()`

### 4. **NEVER use TF-IDF as primary text representation**
✅ **Use** dense embeddings (MiniLM-384d) instead
✅ TF-IDF only for small-scale benchmarking if curious

### 5. **English-only dataset simplification**
✅ **Use** English-optimized models for better performance

### 6. **NEVER complete all features before testing models**
✅ **Always** iterate: build → train → evaluate → improve

### 7. **NEVER evaluate test set multiple times**
✅ **Touch test set ONCE** at the very end

---

## 📈 Expected Performance (Updated)

### Audio-Only Baseline
- Mean Predictor: RMSE ~0.25
- Ridge: RMSE 0.17-0.21, R² 0.20-0.35
- XGBoost: RMSE 0.15-0.19, R² 0.35-0.50

### + Lightweight Text
- XGBoost: RMSE 0.13-0.17, R² 0.45-0.60

### + Embeddings
- XGBoost/LightGBM: RMSE 0.11-0.15, R² 0.55-0.70

### + Metadata (Final)
- LightGBM: RMSE 0.10-0.14, R² 0.60-0.75

**Note**: R² > 0.70 would be exceptional for valence (inherently subjective)

---

## 🎯 Key Takeaways

1. **Iterate, don't waterfall**: Build incrementally, validate continuously
2. **English-optimized models**: Use all-MiniLM-L6-v2 and TextBlob for better performance
3. **Cache expensive operations**: Embeddings take time to compute
4. **Prevent data leakage**: Artist-aware splits are mandatory
5. **Dense > Sparse**: Embeddings beat TF-IDF at this scale
6. **Test once**: Touch test set only at the very end

---

## 📝 Updated Files

### Documentation
- ✅ `memory-bank/ML_ROADMAP.md` - Completely rewritten
- ✅ `memory-bank/techContext.md` - Updated NLP stack
- ✅ `memory-bank/progress.md` - Updated pipeline phases
- ✅ `memory-bank/CRITICAL_CORRECTIONS.md` - This file

### Next Steps
1. Install dependencies: `sentence-transformers`, `textblob`
2. Update data splitting script to use GroupShuffleSplit (already done)
3. Create embedding cache directory: `dataset/processed/`
4. Extract text statistics and sentiment features
5. Follow updated roadmap phases iteratively

---

**Status**: Ready to proceed with corrected methodology ✅
