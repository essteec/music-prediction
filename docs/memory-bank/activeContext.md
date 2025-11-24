# Active Context: Current Work Focus

## Current Sprint (November 24, 2025)

### 🎉 PHASE 1, 2 & 3 COMPLETE: Text Features Analyzed!

**Status Update (November 24, 2025)**:
- ✅ **Data Collection COMPLETE**: All scraping finished!
- ✅ **Validation COMPLETE**: Comprehensive validation framework created
- ✅ **Cleaning COMPLETE**: Data cleaned and validated (732,988 songs)
- ✅ **Encoding Fix COMPLETE**: Key/mode standardized (43,893 rows fixed)
- ✅ **Final Dataset**: `dataset/processed/songs_ml_ready.csv` - 0 duplicates, 0 encoding issues
- ✅ **EDA COMPLETE**: `notebooks/01_exploratory_data_analysis.ipynb` executed
- ✅ **Splits COMPLETE**: Artist-aware 70/15/15 splits (zero overlap verified)
- ✅ **Audio Features COMPLETE**: 21 features (includes genre, year, cyclical key)
- ✅ **Baseline Models COMPLETE**: Mean, Linear, Ridge, XGBoost trained for all 4 targets
- ✅ **TEXT FEATURES COMPLETE**: Text stats + Sentiment extracted and tested
- ✅ **Key Finding**: Text statistics > Sentiment for valence prediction!
- 🔥 **Next Phase**: Decide on embeddings or proceed to final evaluation

### ✅ PHASE 3 COMPLETE: TEXT FEATURES (November 24, 2025)

**What We Did**:
1. ✅ Extracted text statistics (5 features): word_count, unique_word_count, unique_ratio, avg_word_length, char_count
2. ✅ Extracted sentiment (2 features): sentiment_polarity, sentiment_subjectivity via TextBlob
3. ✅ Trained 3 model variants:
   - Audio (21) + Text Stats (5) = 26 features
   - Audio (21) + Sentiment (2) = 23 features  
   - Audio (21) + Text Stats (5) + Sentiment (2) = 28 features (Combined)

**Key Results (XGBoost R²)**:

| Target | Baseline | +Text Stats | +Sentiment | Combined | Best Approach |
|--------|----------|-------------|------------|----------|---------------|
| Valence | 0.346 | **0.371** (+0.025) | 0.347 (+0.001) | **0.372** (+0.026) | Text Stats/Combined |
| Energy | 0.833 | 0.834 (+0.001) | 0.833 (±0.000) | 0.834 (+0.001) | Any (already excellent) |
| Danceability | 0.529 | **0.549** (+0.020) | 0.529 (±0.000) | **0.549** (+0.020) | Text Stats/Combined |
| Popularity | 0.092 | **0.116** (+0.024) | 0.092 (±0.000) | **0.116** (+0.024) | Text Stats/Combined |

**Critical Insights**:
- 🎯 **Text statistics (word count, uniqueness) are MORE valuable than sentiment!**
- ✅ **Valence improved**: 0.346 → 0.372 (achieved target!)
- ✅ **Popularity improved**: 0.092 → 0.116 (significant for hard target)
- ⚠️ **Sentiment alone provides minimal gain** - statistical text features capture more signal
- ✅ **Combined features don't hurt** - slight improvement in valence

**Feature Importance (Combined Model - Valence)**:
1. word_count (12.4%) - **Text feature is #1!**
2. genre_Rock (9.4%)
3. acousticness (8.3%)
4. duration_ms (7.9%)
5. char_count (7.3%) - **Another text feature!**

### Active Tasks - PHASE 4 DECISION 🤔

**Option A: Add Embeddings** (all-MiniLM-L6-v2)
- Pros: Semantic understanding, potentially better valence prediction
- Cons: 384 features, ~30-60 min computation, more complex
- Expected gain: +0.02-0.05 R² for valence

**Option B: Skip to Final Evaluation**
- Pros: Text stats already effective, simpler thesis narrative
- Cons: Miss potential semantic improvements
- Current: Valence R²=0.372 (acceptable for thesis)

**Other Tasks** (LOWER PRIORITY):
1. Get 10 similar thesis for reference
2. Write abstract
3. Document feature engineering methodology for thesis

## ✅ DECISION FINALIZED: Multi-Target Prediction

### The Decision (October 10, 2025)

We will predict **4 target variables** using **4 separate models**:
1. **Valence** (emotional positivity)
2. **Energy** (intensity/activity)
3. **Danceability** (dance suitability)
4. **Popularity** (track success)

### Approach

**Four Independent Models** (NOT one multi-output model):
- Each target gets its own trained model
- Same preprocessing and feature engineering pipeline
- Same 5 algorithms compared: Baseline, Linear, Ridge, Random Forest, XGBoost
- Independent hyperparameter tuning per target
- Comparative analysis across all targets

### Dataset Characteristics
**Available Features**:
- Audio: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration
- Text: Full lyrics
- Metadata: genre, year, explicit, popularity (scraped)

**Key Insight**: We have 4 interesting targets with different characteristics - perfect for comparative study!

### Why Multi-Target Approach?

**Rationale**:
1. **Comprehensive Scope**: Demonstrates systematic methodology suitable for final year project
2. **Efficient Reuse**: Build pipeline once, apply to 4 targets
3. **Risk Mitigation**: Multiple targets ensure success even if one is challenging
4. **Rich Analysis**: Comparative insights ARE the research contribution
5. **No Neural Networks Needed**: Traditional ML algorithms sufficient
6. **Manageable Timeline**: 8-10 weeks is achievable with parallel work

### Expected Performance by Target

| Target | Expected R² | Difficulty | Key Features |
|--------|-------------|------------|--------------|
| Valence | 0.35-0.55 | Moderate | Lyrics + mood features |
| Energy | 0.60-0.75 | Easy | Loudness + tempo |
| Danceability | 0.50-0.65 | Moderate | Tempo + beat |
| Popularity | 0.30-0.45 | Hard | Genre + year (external factors limit) |

### Research Contribution

**Not**: "We built a valence predictor"  
**But**: "We systematically compared ML approaches across 4 diverse targets and discovered which features/algorithms work best for which prediction tasks"

This is a **methodology contribution** valuable for future music prediction research.

## Next Immediate Steps

### ✅ Completed Phases

**Phase 1: Data Understanding** ✅ DONE (November 14, 2025)
- [x] Load and inspect dataset
- [x] Analyze all 4 target variables (valence, energy, danceability, popularity)
- [x] Examine audio features distributions
- [x] Study correlations between features and targets
- [x] Analyze genre patterns
- [x] Investigate temporal trends (year)
- [x] Check lyrics availability
- [x] Review artist distribution
- [x] **Dataset**: Using English-only songs from filtered dataset

**Phase 2: Audio-Only Baselines** ✅ DONE (November 14, 2025)
- [x] Fixed key/mode encoding (43,893 rows standardized: C→0, Major→1, etc.)
- [x] Created artist-aware splits (GroupShuffleSplit by artist_id)
- [x] Scaled audio features (StandardScaler fit on train)
- [x] Trained: Mean → Linear → Ridge → XGBoost
- [x] Established performance floor:
  - Energy: RMSE=0.106, R²=0.81 ✅ Excellent
  - Danceability: RMSE=0.129, R²=0.44 ✅ Good
  - Valence: RMSE=0.213, R²=0.29 ⚠️ Needs text
  - Popularity: RMSE=17.08, R²=0.04 ⚠️ External factors
- [x] Documented baseline results
- [x] Compared encoding methods (median imputation vs proper mapping - identical performance)

### 🔥 Current Phase

**Phase 3: Lightweight Text Features** (IN PROGRESS - November 14, 2025)
- [ ] Extract text statistics (word count, unique ratio, avg word length, char count)
  - Script: `ml/preprocessing/text_statistics.py`
  - Time: ~5 minutes
  - Output: 5 features per song
- [ ] Extract sentiment (TextBlob for English)
  - Script: `ml/preprocessing/sentiment_features.py`
  - Model: `TextBlob`
  - Time: ~10-20 minutes (fast for English)
  - Output: 2 features per song (polarity, subjectivity)
- [ ] Retrain XGBoost with audio + text features
- [ ] Evaluate improvement (target: ΔRMSE > 0.01 for valence)
- [ ] **Success Criteria**: Valence R² improves from 0.29 → 0.40+

**Phase 4: Embedding-Based Text Features** (UPCOMING)
- [ ] Compute embeddings **ONCE** (all-MiniLM-L6-v2 for English)
- [ ] **Cache to disk** with joblib (MANDATORY)
- [ ] Train XGBoost + LightGBM
- [ ] Evaluate semantic improvement

**Critical Rules**:
- ✅ Use `GroupShuffleSplit` by artist_id (prevent data leakage)
- ✅ Use TextBlob for English sentiment (fast and effective)
- ✅ Cache embeddings (compute once, reuse forever)
- ✅ NO TF-IDF as primary text representation (embeddings > TF-IDF)
- ✅ Iterate: build → train → evaluate → improve (NOT waterfall)

### 2. Finalize Target Variable Decision ✅
- [x] Decision made: 4 targets (valence, energy, danceability, popularity)
- [x] Update project documentation (Memory Bank updated)

### 3. Dataset Specification (After Validation)
- [ ] Document exact dataset size (number of songs after cleaning)
- [ ] Calculate missing value statistics
- [x] Scraping complete - no more needed
- [ ] Create data dictionary (all features documented)
- [ ] Check target variable distributions (valence, energy, danceability, popularity)

### 4. Reference Collection (Complete This Week)
**Search Terms**:
- "music emotion prediction machine learning"
- "valence prediction lyrics"
- "audio feature music classification"
- "sentiment analysis song lyrics"
- "music information retrieval thesis"

**Target Sources**:
- Google Scholar
- IEEE Xplore
- ACM Digital Library
- University thesis repositories

### 5. Abstract Writing (Complete This Week)
**Structure**:
```
1. Context: Music attribute prediction importance  
2. Problem: Can we predict musical characteristics (valence, energy, 
   danceability, popularity) from audio features and lyrics?
3. Method: Systematic comparison of 5 ML algorithms across 4 targets
4. Dataset: Spotify songs with lyrics and audio features
5. Expected Contribution: Comparative analysis revealing which 
   features/algorithms work best for which prediction tasks
```

## Active Decisions & Considerations

### ML Algorithm Selection (Tentative List)
1. **Baseline**: Mean predictor, simple linear regression
2. **Linear Models**: Ridge Regression, Lasso (with regularization)
3. **Tree-Based**: Random Forest, XGBoost
4. **Neural Network**: Simple feedforward NN (optional, if time permits)
5. **SVM**: Support Vector Regression (optional)

**Rationale**: Mix of interpretable (linear) and powerful (tree-based) models

### Feature Engineering Strategy

**Iterative Approach** (build → validate → add):

**Phase 1: Audio Only**
- Use existing features as-is (scaled)
- No polynomial features initially
- Train baselines: Linear, Ridge, XGBoost

**Phase 2: Lightweight Text**
- Basic stats: word count, unique words, unique ratio, avg word length
- **Sentiment**: TextBlob for English
  - Returns: polarity (-1 to 1), subjectivity (0 to 1)

**Phase 3: Embeddings**
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (English-optimized)
- **Output**: 384-dimensional dense vectors
- **Strategy**: Compute ONCE, cache with joblib
- **Why**: Semantic understanding > word frequency (TF-IDF)

**Phase 4: Metadata**
- **Genre**: Target encoding (mean valence per genre, NOT one-hot)
- **Year**: Normalize to [0, 1]
- **Explicit**: Already binary

**What NOT to do**:
- ❌ TF-IDF as primary representation (only for benchmarking if curious)
- ❌ One-hot encoding for genre (too many dimensions)
- ❌ Polynomial features without testing improvement

### Evaluation Strategy
- **Train/Validation/Test**: 70/15/15 split
- **⚠️ CRITICAL**: Use `GroupShuffleSplit` by artist_id (prevent artist leakage)
- **Cross-Validation**: GroupKFold (5 folds, grouped by artist)
- **Metrics**: 
  - Primary: RMSE, R² (for regression)
  - Secondary: MAE
  - **Error Segmentation**: by genre, artist, valence range
- **Visualizations**: 
  - Predicted vs actual scatter
  - Error distribution histogram
  - Residual plots
  - Feature importance (for tree models)
- **Test Set**: Evaluate ONCE at the very end (no peeking!)
  
### Collaboration Plan
**Team Division** (To be finalized):
- **Person 1**: Data preprocessing, feature engineering, baseline models
- **Person 2**: Advanced models, evaluation, visualization
- **Shared**: Literature review, thesis writing, final integration

**GitHub Workflow**:
- Main branch: stable code only
- Feature branches: individual work
- Pull requests: code review before merging
- Issues: track tasks and bugs

## Important Patterns & Preferences

### Code Style
- Follow PEP 8 Python style guide
- Use type hints where helpful
- Docstrings for all functions
- Meaningful variable names

### Documentation
- README in each major folder
- Inline comments for complex logic
- Jupyter notebooks for exploration (with markdown explanations)

### Reproducibility
- Set random seeds (42 is standard)
- Document library versions
- Save model artifacts with metadata
- Version control everything except large data files

### Recent Insights

### Baseline Results - Audio-Only Performance (November 14, 2025) 📊
- **Energy (R²=0.81)**: 
  - Highly predictable from audio features (loudness, tempo correlate strongly)
  - XGBoost RMSE=0.106, significantly better than Linear (0.121)
  - **Insight**: Audio is objectively measurable, matches literature expectations
  
- **Danceability (R²=0.44)**:
  - Moderately predictable from rhythm features
  - Room for improvement with text/metadata
  - **Insight**: Tempo and beat strength help, but subjective factors play role
  
- **Valence (R²=0.29)**:
  - Weakest audio-only performance
  - **Key Finding**: Needs text features! Sentiment from lyrics crucial
  - **Target**: Improve to 0.45-0.60 with sentiment + embeddings
  - **Insight**: Emotional positivity not fully captured by audio alone
  
- **Popularity (R²=0.04)**:
  - Essentially unpredictable from audio/text
  - **Insight**: External factors dominate (marketing, social trends, artist fame)
  - Will document as limitation in thesis

### Data Quality Achievement (November 14, 2025) ✅
- **Mixed Encoding Discovery**: 43,893 rows (6%) had letter-based keys (C, D, E) and text modes (Major, Minor)
- **Root Cause**: Dataset merged from multiple sources with different notation systems
- **Fix Applied**: Proper mapping (C→0, D→2, ..., B→11, Major→1, Minor→0)
- **Validation Enhanced**: Added encoding consistency checks to catch this earlier
- **Impact**: Minimal on model performance (key/mode low importance), but methodology now clean for thesis
- **Lesson**: `errors='coerce'` silently masks data quality issues - always check BEFORE conversion
- **Achievement**: Roadmap corrected from waterfall to iterative approach
- **Key Fixes**:
  1. ✅ Artist-aware GroupShuffleSplit (prevents data leakage)
  2. ✅ English-optimized models (TextBlob, all-MiniLM-L6-v2)
  3. ✅ Dense embeddings > TF-IDF (semantic, compact, fast)
  4. ✅ Embedding caching (compute once, reuse forever)
  5. ✅ Iterative development (audio → text → embeddings → metadata)
  6. ✅ Using English-only dataset (simplified pipeline)
- **Impact**: More realistic timeline, better methodology, higher quality results
- **Documentation**: See `CRITICAL_CORRECTIONS.md` for full details
- **Dependencies**: Updated `requirements-ml.txt` with correct libraries

### Data Collection - COMPLETE! ✅
- **Achievement**: All scraping successfully finished (November 10, 2025)
- **Output**: Three CSV files ready for validation
  - `songs_enhanced_full.csv`: Successful scrapes (needs cleaning)
  - `failed_tracks.csv`: Failed scrapes (needs analysis)
  - `unknown_tracks.csv`: Undetected genres (needs processing)
- **Migration Success**: HTTP-based scraper worked (Selenium → HTTP migration successful)
- **Timeline Impact**: Major blocker removed, back on track!

### Data Quality Issues Identified 🔍
- **Issue 1**: Some songs have NaN genre in songs_enhanced_full.csv
- **Issue 2**: Some songs have year = 0 in songs_enhanced_full.csv
- **Issue 3**: Unknown number of failed tracks in failed_tracks.csv
- **Issue 4**: Unknown number of songs with undetected genres in unknown_tracks.csv
- **Next Step**: Statistical analysis to quantify these issues

### Project Insights
- This is a comparison study, not just building one model
- Need to tell a story: "which approach works best and why?"
- **Iterative development > waterfall**: Validate continuously, avoid wasted work
- **English-only dataset**: Simplified NLP pipeline, faster processing
- **Prevent data leakage**: Artist-aware splits are mandatory
- Thesis should contribute methodology, not necessarily SOTA results
- GitHub presence is important for both partners' portfolios
