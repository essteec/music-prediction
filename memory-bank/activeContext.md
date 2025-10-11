# Active Context: Current Work Focus

## Current Sprint (Week Oct 7-14, 2025)

### 🔥 CRITICAL PIVOT: Scraping Performance Crisis

**Status Update (October 11, 2025)**:
- ⚠️ **Selenium scraping is FAILING**: Only 3,529 songs scraped in 12 hours (12 sec/song)
- 📊 **Remaining**: 951,768 songs = 136 days at current rate
- 🚨 **Problem**: Thesis deadline makes this approach IMPOSSIBLE
- ✅ **Decision**: Migrating to HTTP-based scraping (10-50x faster)

### Active Tasks - UPDATED PRIORITIES
1. **Data Collection - URGENT REFACTOR** 🔥
   - **Current**: Selenium-based scraper (TOO SLOW - 12 sec/song)
   - **Target**: HTTP-based scraper (1-3 sec/song)
   - **Progress**: 3,529 songs successfully scraped, 23 failed (PRESERVE THIS DATA)
   - **Next**: Reverse-engineer HTTP requests, rebuild scraper without browser overhead
   - **Strategy**: Linear scraping (no concurrency), manual stop after 10K-50K songs
   
2. **Get 10 similar thesis for reference** (Postponed until scraping resolved)
   - Need to collect academic references
   - Target: Theses on music prediction, lyric analysis, audio feature ML
   
3. **Write abstract** (Postponed until scraping resolved)
   - Waiting on: Scraping completion
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

### 1. Finalize Target Variable Decision ✅
- [x] Decision made: 4 targets (valence, energy, danceability, popularity)
- [ ] Discuss with team partner
- [ ] Confirm with advisor if needed
- [x] Update project documentation (Memory Bank updated)

### 2. Dataset Specification (Complete This Week)
- [ ] Document exact dataset size (number of songs)
- [ ] Calculate missing value statistics
- [ ] Determine if more scraping is needed
- [ ] Create data dictionary (all features documented)
- [ ] Check valence distribution (balanced? skewed?)

### 3. Reference Collection (Complete This Week)
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

### 4. Abstract Writing (Complete This Week)
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

### Dataset Insights
- Dataset is large (955,320 songs, 1.5GB CSV) - need efficient processing
- Lyrics are available - enables NLP approaches
- Audio features are comprehensive - good foundation
- **Current scraping status**: 3,529 songs enhanced, 23 failed (0.37% of total)
- **Scraping bottleneck**: Selenium approach taking 136 days total - UNACCEPTABLE

### Technical Insights - Scraping Crisis Lessons
- **Selenium is TOO SLOW**: 12 seconds per song due to browser overhead
  - Full page load, JavaScript execution, CSS rendering all unnecessary
  - Browser automation adds 10-50x overhead vs direct HTTP requests
- **Root causes identified**:
  1. Browser startup/teardown overhead
  2. Full page rendering (images, CSS, JavaScript)
  3. Dynamic genre mapping spawning separate browser instances
  4. Wait times for elements to appear (5+ seconds per page)
- **Genre mapper bug**: Created separate Chrome instances that froze/crashed
  - Fixed by reusing same browser session
  - But still too slow due to browser overhead
- **Solution path**: Migrate to HTTP requests + BeautifulSoup (no browser)
  - Expected: 1-3 sec/song (10-50x speedup)
  - Same HTML parsing logic, just faster transport
  - Zero budget (no proxies unless rate limited)
  
### Critical Decisions Made
- **No concurrency**: Linear scraping to avoid complexity and rate limiting
- **Manual stop control**: Can interrupt after 10K-50K songs (sufficient for thesis)
- **Preserve existing data**: 3,529 scraped + 23 failed must not be lost
- **Phase approach**: 
  1. ✅ Critical analysis complete
  2. 🔄 Planning phase (waiting for HTTP request examples)
  3. ⏳ Coding phase (rebuild with requests library)
  4. ⏳ Testing and full scraping run

### Project Insights
- This is a comparison study, not just building one model
- Need to tell a story: "which approach works best and why?"
- Thesis should contribute methodology, not necessarily SOTA results
- GitHub presence is important for both partners' portfolios
