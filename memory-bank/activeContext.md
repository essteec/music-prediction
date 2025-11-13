# Active Context: Current Work Focus

## Current Sprint (November 12, 2025)

### 🎉 PHASE 1 COMPLETE: Data Cleaning Done!

**Status Update (November 12, 2025)**:
- ✅ **Data Collection COMPLETE**: All scraping finished!
- ✅ **Validation COMPLETE**: Comprehensive validation framework created
- ✅ **Cleaning COMPLETE**: Data cleaned and validated
- ✅ **Final Dataset**: `dataset/processed/songs_ml_ready.csv` (732,988 songs)
- ✅ **Quality Verified**: 0 duplicates, 100% target completeness, all ranges valid
- ✅ **EDA Notebook Created**: `notebooks/01_exploratory_data_analysis.ipynb`
- 🔥 **Next Phase**: Run EDA, create train/test splits, build baselines

### Active Tasks - UPDATED PRIORITIES
1. **Data Validation & Cleaning** 🔥 URGENT
   - **All 20 columns** analyzed and specified (see `ml/COLUMN_SPECIFICATIONS.md`)
   - **4 Target Variables**: valence 🎯, energy 🎯, danceability 🎯, popularity 🎯
   - **Known Issues**: 
     - genre: NaN values ⚠️
     - year: values = 0 ⚠️
     - Range violations: need checking across all numeric columns
   - **Roadmap**: See `ml/DATA_VALIDATION_ROADMAP.md` for detailed 7-day plan
   - **Progress**: See `ml/PHASE1_CHECKLIST.md` for tracking
   - **Column Specs**: See `ml/COLUMN_SPECIFICATIONS.md` for complete validation rules
   - **Validation Script**: `scripts/comprehensive_validation.py` ready to run
   - **Next**: Run validation script to get exact statistics
   
2. **Get 10 similar thesis for reference** (UNBLOCKED - can proceed!)
   - Need to collect academic references
   - Target: Theses on music prediction, lyric analysis, audio feature ML
   
3. **Write abstract** (UNBLOCKED - can proceed!)
   - Multi-target approach confirmed (valence, energy, danceability, popularity)
   - Needs: Clear problem statement and methodology outline

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

### 1. Data Validation & Cleaning (THIS WEEK - TOP PRIORITY) 🔥

⚠️ **CRITICAL UPDATE (November 12, 2025)**: ML Roadmap has been corrected from waterfall to iterative approach. See `CRITICAL_CORRECTIONS.md` for full details.

**Phase 1: Minimal Clean Dataset**
- [ ] Load and inspect dataset
- [ ] **Add language detection** (multilingual corpus)
- [ ] Count and analyze NaN genre values
- [ ] Count and analyze year = 0 values
- [ ] Remove duplicates and invalid entries
- [ ] **CRITICAL**: Create artist-aware train/val/test splits (GroupShuffleSplit)
- [ ] Document cleaning decisions

**Phase 2: Audio-Only Baselines**
- [ ] Scale audio features (StandardScaler)
- [ ] Train: Mean → Linear → Ridge → XGBoost
- [ ] Establish performance floor (RMSE ~0.15-0.20)
- [ ] Document baseline results

**Phase 3: Lightweight Text Features**
- [ ] Extract: word count, unique ratio, avg word length
- [ ] **Multilingual sentiment** (cardiffnlp/twitter-xlm-roberta-base-sentiment)
- [ ] ⚠️ **NOT TextBlob** (English-only, weak signals)
- [ ] Retrain XGBoost, evaluate improvement

**Phase 4: Embedding-Based Text Features**
- [ ] Compute embeddings **ONCE** (paraphrase-multilingual-MiniLM-L12-v2)
- [ ] **Cache to disk** with joblib (MANDATORY)
- [ ] Train XGBoost + LightGBM
- [ ] Evaluate semantic improvement

**Critical Rules**:
- ✅ Use `GroupShuffleSplit` by artist_id (prevent data leakage)
- ✅ Use multilingual sentiment model (NOT TextBlob)
- ✅ Cache embeddings (compute once, reuse forever)
- ✅ NO TF-IDF as primary text representation (embeddings > TF-IDF)
- ✅ Iterate: build → train → evaluate → improve (NOT waterfall)

### 2. Finalize Target Variable Decision ✅
- [x] Decision made: 4 targets (valence, energy, danceability, popularity)
- [ ] Discuss with team partner
- [ ] Confirm with advisor if needed
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
- **Multilingual sentiment**: cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Returns: negative, neutral, positive probabilities + polarity (-1 to 1)
  - Supports: 50+ languages

**Phase 3: Embeddings**
- **Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Output**: 384-dimensional dense vectors
- **Strategy**: Compute ONCE, cache with joblib
- **Why**: Semantic understanding > word frequency (TF-IDF)

**Phase 4: Metadata**
- **Genre**: Target encoding (mean valence per genre, NOT one-hot)
- **Year**: Normalize to [0, 1]
- **Explicit**: Already binary

**What NOT to do**:
- ❌ TF-IDF as primary representation (only for benchmarking if curious)
- ❌ TextBlob for sentiment (English-only)
- ❌ One-hot encoding for genre (too many dimensions)
- ❌ Polynomial features without testing improvement

### Evaluation Strategy
- **Train/Validation/Test**: 70/15/15 split
- **⚠️ CRITICAL**: Use `GroupShuffleSplit` by artist_id (prevent artist leakage)
- **Cross-Validation**: GroupKFold (5 folds, grouped by artist)
- **Metrics**: 
  - Primary: RMSE, R² (for regression)
  - Secondary: MAE
  - **Error Segmentation**: by language, genre, artist, valence range
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

## Recent Insights

### ML Pipeline - CRITICAL CORRECTIONS APPLIED (November 12, 2025) 🚨
- **Achievement**: Roadmap corrected from waterfall to iterative approach
- **Key Fixes**:
  1. ✅ Artist-aware GroupShuffleSplit (prevents data leakage)
  2. ✅ Multilingual sentiment (XLM-RoBERTa, NOT TextBlob)
  3. ✅ Dense embeddings > TF-IDF (semantic, compact, fast)
  4. ✅ Embedding caching (compute once, reuse forever)
  5. ✅ Iterative development (audio → text → embeddings → metadata)
  6. ✅ Language detection (multilingual corpus support)
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
- **Multilingual support is critical**: Dataset spans 50+ languages
- **Prevent data leakage**: Artist-aware splits are mandatory
- Thesis should contribute methodology, not necessarily SOTA results
- GitHub presence is important for both partners' portfolios
