# Active Context: Current Work Focus

## Current Sprint (November 10, 2025)

### 🎉 MAJOR MILESTONE: Data Collection Complete!

**Status Update (November 10, 2025)**:
- ✅ **Scraping COMPLETE**: All data has been successfully scraped!
- 📊 **Output Files**: 
  - `songs_enhanced_full.csv` - All successful scrapes
  - `failed_tracks.csv` - All failed scrapes  
  - `unknown_tracks.csv` - Successful scrapes with undetected genres
- � **Next Challenge**: Data validation and cleaning needed
- ✅ **Timeline**: Back on track - major blocker resolved!

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
**songs_enhanced_full.csv**:
- [ ] Load and inspect dataset
- [ ] Count and analyze NaN genre values
- [ ] Count and analyze year = 0 values
- [ ] Decide on cleaning strategy (drop, impute, or manual fix)
- [ ] Apply cleaning transformations
- [ ] Validate data quality

**failed_tracks.csv**:
- [ ] Analyze failure reasons
- [ ] Count total failures
- [ ] Decide: retry failed tracks or accept data loss
- [ ] Document failure patterns

**unknown_tracks.csv**:
- [ ] Count songs with unknown genres
- [ ] Explore alternative genre detection methods
- [ ] Decide: manual mapping, drop, or use as "Unknown" category
- [ ] Integrate into main dataset if valuable

**Final Output**:
- [ ] Merge validated data into single clean CSV
- [ ] Document final dataset statistics
- [ ] Create data dictionary

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

**For Lyrics**:
- Sentiment scores (TextBlob/VADER)
- TF-IDF vectors (top 500-1000 features)
- Word embeddings (average Word2Vec vectors)
- Basic stats: word count, unique words, average word length

**For Audio**:
- Use existing features as-is
- Consider interaction terms (energy × tempo, etc.)
- Normalize/scale appropriately

**For Metadata**:
- One-hot encode genre
- Normalize year
- Binary flag for explicit

### Evaluation Strategy
- **Train/Validation/Test**: 70/15/15 or 80/10/10 split
- **Cross-Validation**: 5-fold CV on training set
- **Metrics**: 
  - Primary: RMSE, R² (for regression)
  - Secondary: MAE, explained variance
  - Visualize: Scatter plots of predicted vs actual
  
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
- Thesis should contribute methodology, not necessarily SOTA results
- GitHub presence is important for both partners' portfolios
