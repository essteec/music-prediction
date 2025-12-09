# Active Context: Current Work Focus

## Current Sprint (December 9, 2025)

### 🎉 EXPERIMENT 1 COMPLETE - FINAL TEST RESULTS OBTAINED!

**Status Update (December 9, 2025)**:
- ✅ **Data Collection COMPLETE**: All scraping finished!
- ✅ **Validation COMPLETE**: Comprehensive validation framework created
- ✅ **Cleaning COMPLETE**: Data cleaned and validated (732,988 songs)
- ✅ **Encoding Fix COMPLETE**: Key/mode standardized (43,893 rows fixed)
- ✅ **Final Dataset**: `dataset/processed/songs_ml_ready.csv` - 0 duplicates, 0 encoding issues
- ✅ **EDA COMPLETE**: `notebooks/01_exploratory_data_analysis.ipynb` + `03_feature_files_eda.ipynb`
- ✅ **Splits COMPLETE**: Artist-aware 70/15/15 splits (zero overlap verified)
- ✅ **Audio Features COMPLETE**: 21 features (includes genre, year, cyclical key)
- ✅ **Baseline Models COMPLETE**: Mean, Linear, Ridge, XGBoost trained for all 4 targets
- ✅ **TEXT FEATURES COMPLETE**: Text stats + Sentiment extracted and tested
- ✅ **EMBEDDINGS COMPLETE**: all-MiniLM-L6-v2 (384-dim semantic vectors) extracted and cached!
- ✅ **Full Feature Models COMPLETE**: Trained with all 412 features (audio + text + embeddings)
- ✅ **ENHANCED MODELS COMPLETE**: 14+ algorithms trained with default + tuned variants!
- ✅ **Enhanced Models Analysis COMPLETE**: Created and executed analysis notebooks
- ✅ **Feature Importance Analysis COMPLETE**: Extracted importance from 12 selected models
- ✅ **TEST EVALUATION COMPLETE**: Final test results obtained (December 9, 2025)!
- 🔥 **Next Phase**: Error analysis OR start Experiment 2 (artist features)
- ⏸️ **Artist Data Scripts**: fetch_artist_data.py and add_follower_counts.py available but NOT YET RUN

### 📓 New Notebooks Created (December 5-7, 2025)

1. **`notebooks/04_enhanced_models_analysis.ipynb`** ✅ EXECUTED
   - Comprehensive analysis of 28+ enhanced models across 4 targets
   - R² heatmap, Top 5 models bar charts, Training time analysis
   - Default vs Tuned comparison with improvement quantification
   - Best models identified per target
   
2. **`notebooks/04-2_enhanced_models_analysis.ipynb`** ✅ EXECUTED
   - Extended analysis variant
   
3. **`notebooks/05_feature_importance_analysis.ipynb`** ✅ EXECUTED
   - Feature importance extraction from 12 selected models
   - Models analyzed: CatBoost, LightGBM, XGBoost, ExtraTrees, MLPRegressor, RandomForest (default + tuned)
   - Top features visualization per target
   - Feature group importance analysis
   - Model comparison heatmaps

4. **`ml/models/test_evaluation_final.py`** 🔜 READY TO RUN
   - Final test set evaluation script
   - Will evaluate 12 selected models on held-out test set
   - ONE-TIME execution for final thesis results

### 📊 Key Findings from Analysis (December 5-9, 2025)

**Best Models by Target (Validation R²)**:
| Target | Best Model | R² Score | RMSE |
|--------|-----------|----------|------|
| Valence | XGBoost_tuned | 0.4659 | 0.1818 |
| Energy | XGBoost_tuned | 0.8487 | 0.0952 |
| Danceability | XGBoost_tuned | 0.6092 | 0.1070 |
| Popularity | CatBoost | 0.0783 | 1.4303 |

**🎯 FINAL TEST SET RESULTS (December 9, 2025)** - THESIS NUMBERS:
| Target | Best Model | Test R² | Test RMSE | Test MAE |
|--------|-----------|---------|-----------|----------|
| **Energy** | XGBoost_tuned | **0.8468** | 0.0945 | 0.0705 |
| **Danceability** | XGBoost_tuned | **0.6185** | 0.1061 | 0.0838 |
| **Valence** | XGBoost_tuned | **0.4742** | 0.1805 | 0.1443 |
| **Popularity** | CatBoost | **0.0696** | 1.4139 | 1.2308 |

**Test Set Statistics**:
- Total test samples: 82,274 songs
- Models evaluated: 12 (48 total model-target combinations)
- Best overall model: XGBoost_tuned (3/4 targets)
- Average R² across all models/targets: 0.4683

**Selected Models for Final Evaluation (12 total)**:
- CatBoost, CatBoost_tuned
- LightGBM, LightGBM_tuned  
- XGBoost, XGBoost_tuned
- ExtraTrees, ExtraTrees_tuned
- MLPRegressor, MLPRegressor_tuned
- RandomForest, RandomForest_tuned

**Tuning Impact Insights**:
- KNeighbors: +0.081 R² improvement (largest gain)
- RandomForest: -0.087 R² (tuning hurt performance)
- Most models: marginal improvement from tuning

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

### ✅ PHASE 4 COMPLETE: EMBEDDINGS (November 28, 2025)

**What We Did**:
1. ✅ Installed sentence-transformers library
2. ✅ Created `process_embeddings.py` module with intelligent caching
3. ✅ Extracted 384-dimensional semantic embeddings using all-MiniLM-L6-v2
4. ✅ Integrated embeddings into preprocessing pipeline
5. ✅ Created model training scripts:
   - `embedding_models.py`: Audio + Embeddings (405 features)
   - `full_features_models.py`: Audio + Text Stats + Sentiment + Embeddings (412 features)
6. ✅ Created feature files EDA notebook (`03_feature_files_eda.ipynb`)
7. ✅ Trained and evaluated full feature models

**Implementation Details**:
- Model: sentence-transformers/all-MiniLM-L6-v2 (English-optimized)
- Output: 384-dimensional dense vectors per song
- Caching: Embeddings saved to .npy files (instant reload)
- Processing time: ~30-60 minutes (one-time computation)
- Total features now: 412 (21 audio + 5 text stats + 2 sentiment + 384 embeddings)

**Results** (awaiting full analysis):
- Models trained with full feature set
- Ready for comparison with baseline and text-only models

### ✅ PHASE 5 COMPLETE: ENHANCED MODELS (December 5, 2025)

**What We Did**:
1. ✅ Trained comprehensive algorithm comparison with 14+ models
2. ✅ Models trained: Mean, LinearRegression, Ridge (tuned), Lasso (tuned), SGDRegressor (tuned), DecisionTree (tuned), RandomForest (tuned), ExtraTrees (tuned), AdaBoost (tuned), XGBoost (tuned), CatBoost (tuned), LightGBM (tuned), KNeighbors (tuned), LinearSVR (tuned), MLPRegressor (tuned)
3. ✅ All 412 features used (audio + text stats + sentiment + embeddings)
4. ✅ Trained for all 4 targets (valence, energy, danceability, popularity)
5. ✅ Checkpoint system with incremental saving implemented
6. ✅ Results saved to: `results/metrics/enhanced_results_summary_20251205_123928.csv`
7. ✅ Models saved to: `ml/models/saved/enhanced/`

**Implementation Details**:
- Script: `ml/models/enhanced_models.py`
- Total models: 28+ (14 models × 4 targets, with default + tuned variants)
- Tuned hyperparameters optimized for 386k samples and 412 features
- Early stopping to prevent overfitting
- Parallel processing where supported (n_jobs=-1)

### Active Tasks - PHASE 6: TWO-EXPERIMENT APPROACH 🔬

**Strategy Decision (December 5, 2025)**: PATH B - Two Independent Experiments

We will conduct TWO complete experiments to compare baseline vs artist-enhanced features:

## 🔵 EXPERIMENT 1: Baseline (No Artist Features) ✅ COMPLETE
**Features**: 412 (audio + text + sentiment + embeddings)
**Status**: ✅ ALL TASKS COMPLETE - Final test results obtained!

**Tasks**:
1. [x] Analyze enhanced_results_summary_20251205_123928.csv ✅
2. [x] Compare all model variants (default vs tuned) ✅
3. [x] Select best 1-2 models per target based on validation ✅
4. [x] Feature importance analysis for selected models ✅
5. [x] Run test evaluation ONCE on selected models ✅ **COMPLETED December 9, 2025**
6. [ ] Error analysis and visualization ← **OPTIONAL NEXT STEP**
7. [ ] Archive results to `experiment1_no_artist/` folder ← **OPTIONAL**

**Notebooks Created for Analysis**:
- `04_enhanced_models_analysis.ipynb` - Model comparison & visualization ✅
- `04-2_enhanced_models_analysis.ipynb` - Extended analysis ✅
- `05_feature_importance_analysis.ipynb` - Feature importance extraction ✅

**Test Evaluation Results** (December 9, 2025):
- Script executed: `ml/models/test_evaluation_final.py` ✅
- Results saved: `results/metrics/test_evaluation_final_20251209_144747.csv`
- Best models: `results/metrics/best_models_test_20251209_144747.csv`
- Comparison: `results/metrics/test_vs_validation_comparison_20251209_144747.csv`
- Visualizations: 3 figures in `results/figures/`

## 🟢 EXPERIMENT 2: With Artist Features (NEW)
**Features**: 415+ (all previous + artist followers/popularity/count)
**Status**: NOT STARTED - Will begin after Experiment 1 complete

**Tasks**:
1. [ ] Fetch artist data from Spotify API (`fetch_artist_data.py`)
2. [ ] Add follower counts to dataset (`add_follower_counts.py`)
3. [ ] Update preprocessing to include artist features
4. [ ] Retrain ALL 28+ models from scratch with new features
5. [ ] Analyze NEW validation results
6. [ ] Select best 1-2 models per target from NEW experiment
7. [ ] Feature importance analysis (focus on artist feature contribution)
8. [ ] Run test evaluation ONCE on NEW selected models
9. [ ] Compare Experiment 1 vs Experiment 2 results
10. [ ] Archive results to `experiment2_with_artist/` folder

### 📋 Artist Data Scripts

**Scripts Ready**:
- `scripts/scraping/fetch_artist_data.py` - Fetches artist metadata from Spotify API
  - Gets: spotify_id, name, popularity, followers, genres
  - Checkpoint/resume support for long-running fetch
  - Rate limiting (2 req/s to avoid 429 errors)
- `scripts/scraping/add_follower_counts.py` - Adds follower counts to songs dataset
  - Calculates: total_artist_followers, artist_count, avg_artist_popularity

**Purpose**:
- Conduct ablation study: Does artist fame improve predictions?
- Expected impact: Likely improves POPULARITY, minimal effect on valence/energy/danceability
- Research contribution: Quantify content vs context features

**Status**: Ready to execute after Experiment 1 analysis complete

### 📁 Project Structure - Two Experiments

```
ml/models/saved/
├── experiment1_no_artist/          ← Archive after completion
│   ├── XGBoost_tuned_valence.pkl
│   ├── RandomForest_tuned_energy.pkl
│   └── ... (28+ models)
│
└── experiment2_with_artist/        ← New models with artist features
    ├── XGBoost_tuned_valence.pkl
    ├── RandomForest_tuned_energy.pkl
    └── ... (28+ models)

results/metrics/
├── experiment1_no_artist/
│   ├── enhanced_results_summary.csv
│   ├── test_results_final.csv      ← Test evaluation (once)
│   └── feature_importance.csv
│
└── experiment2_with_artist/
    ├── enhanced_results_summary.csv
    ├── test_results_final.csv      ← New test evaluation (once)
    ├── feature_importance.csv
    └── comparison_with_exp1.csv
```

**Documentation Tasks** (HIGH PRIORITY):
1. [ ] Get 10 similar theses for literature review
2. [ ] Write thesis abstract (mention two-experiment approach)
3. [ ] Document feature engineering methodology
4. [ ] Create result comparison tables (Exp1 vs Exp2)
5. [ ] Document ablation study findings (artist feature contribution)

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
