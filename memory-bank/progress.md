# Progress: What's Done and What's Next

## ✅ Completed

### Dataset Phase
- [x] **Data Source Identified**: Spotify songs with attributes and lyrics (955,320 songs)
- [x] **Base Dataset Acquired**: CSV with audio features, lyrics, and metadata (1.5GB)
- [x] **Scraper Development - V1 (Selenium)**: Built initial Chosic.com scraper
  - Scrapes: popularity, genre, release year, explicit flag
  - Uses: Selenium + BeautifulSoup
  - Features: Headless mode, rate limiting, browser recovery, NaN validation
  - **Status**: ✅ COMPLETED - Successfully scraped all data
- [x] **Scraper Development - V2 (HTTP)**: ✅ COMPLETED - HTTP-based scraper implemented
  - **Result**: Successfully scraped entire dataset
  - **Output**: Three CSV files for validation and cleanup
- [x] **Genre Mapping**: Dynamic genre normalization with safe browser reuse
- [x] **Data Collection Complete**: ✅ ALL SCRAPING FINISHED (November 10, 2025)
  - **songs_enhanced_full.csv**: All successful scrapes (needs validation)
  - **failed_tracks.csv**: All failed scrapes from various causes
  - **unknown_tracks.csv**: Successful scrapes with undetected genres

### Project Structure
- [x] **Basic Folder Structure**: dataset/, ml/, thesis/, timeline/ created
- [x] **Version Control**: Git initialized
- [x] **Documentation**: Scraper README created
- [x] **Memory Bank**: Complete knowledge base established

### Planning
- [x] **Project Scope Defined**: ML comparison thesis on music prediction
- [x] **Timeline Started**: Week 1 tasks identified (Oct 7-14, 2025)

## 🚧 In Progress

### Current Phase: Data Validation & Cleaning (November 2025)
- [x] **Data Validation**: ✅ PHASE 1 COMPLETE! (November 12, 2025)
  - **Roadmap Created**: ✅ `ml/DATA_VALIDATION_ROADMAP.md` (comprehensive 7-day plan)
  - **Checklist Created**: ✅ `ml/PHASE1_CHECKLIST.md` (progress tracking)
  - **Column Specifications**: ✅ `ml/COLUMN_SPECIFICATIONS.md` (all 20 columns analyzed)
  - **Validation Script**: ✅ `scripts/comprehensive_validation.py` (automated checks)
  - **Outlier Investigation**: ✅ `scripts/investigate_outliers.py` (detailed analysis)
  - **Cleaning Script**: ✅ `scripts/clean_data.py` (executed successfully)
  - **Final Dataset**: ✅ `dataset/processed/songs_ml_ready.csv` (732,988 clean songs)
  - **Results**:
    - Removed 277 songs with tempo = 0 BPM
    - Clipped/removed 406 songs with loudness outliers (>0 dB)
    - Removed 3 songs with invalid year
    - **Final**: 732,988 rows (99.96% retention - only 406 rows lost!)
    - **Quality**: ALL range violations fixed, 0 duplicates, 100% target completeness
- [ ] **Dataset Specification**: Document final dataset characteristics
  - Size, features, distributions, missing values
  - Create data dictionary
- [ ] **Reference Collection**: Gather 10 similar theses/papers
  - Focus on music prediction, lyric analysis, audio feature ML
- [ ] **Abstract Writing**: First draft of thesis abstract
  - Multi-target approach (valence, energy, danceability, popularity)

### Critical Decisions
- [x] **Target Variable Selection**: ✅ FINALIZED October 10, 2025
  - **Decision**: Predict all 4 targets (valence, energy, danceability, popularity)
  - **Approach**: 4 separate models using same ML pipeline
  - **Rationale**: Comprehensive comparative analysis, risk mitigation, rich insights
  - Awaiting team partner confirmation and advisor approval

## 📋 To Do (Upcoming)

### Project Setup (High Priority)
- [ ] **GitHub Repository Setup**
  - Create well-structured repo
  - Add README, .gitignore, LICENSE
  - Setup collaboration workflow (branching strategy)
  - Invite team partner

- [ ] **Environment Setup**
  - Create virtual environment
  - Consolidate requirements.txt (root level)
  - Document setup instructions
  - Test reproducibility

- [ ] **Project Structure Enhancement**
  - Create subdirectories in ml/
  - Add README files to each folder
  - Setup notebook directory
  - Create config/ for experiment parameters

### Data Pipeline (Current Phase - UNBLOCKED!)
- [x] **Data Collection**: ✅ COMPLETE - All scraping finished!
- [x] **Data Validation & Cleaning** ✅ COMPLETE (November 12, 2025)
  - [x] Comprehensive validation framework created
  - [x] All 20 columns analyzed and validated
  - [x] Outliers investigated (loudness, tempo)
  - [x] Cleaning executed (removed 406 rows total)
  - [x] Check for duplicates across all files (0 duplicates found!)
  - [x] Final clean dataset created: `dataset/processed/songs_ml_ready.csv` (732,988 songs)
  - [x] Validation confirmed: ALL targets 100% complete, ALL range violations fixed
  - [ ] **⚠️ CRITICAL**: Detect language distribution for multilingual handling
  - [ ] **⚠️ CRITICAL**: Create artist-aware train/val/test splits
  - [ ] Document cleaning decisions in final report

- [ ] **Exploratory Data Analysis (EDA)**
  - Jupyter notebook with visualizations
  - Feature distributions (audio features)
  - Correlation analysis (audio features ONLY initially)
  - Target variable analysis (valence distribution)
  - **Language distribution analysis** (multilingual corpus)
  - Identify outliers

- [ ] **Feature Engineering - ITERATIVE APPROACH**
  - **Phase 1**: Audio features only (scaled, no polynomial)
  - **Phase 2**: Lightweight text features (word count, unique ratio, avg word length)
  - **Phase 3**: Multilingual sentiment (XLM-RoBERTa, NOT TextBlob)
  - **Phase 4**: Lyric embeddings (compute ONCE, cache to disk)
  - **Phase 5**: Genre & metadata (target encoding, NOT one-hot)
  - ⚠️ **NO TF-IDF** as primary representation (only for benchmarking)

### ML Pipeline (Core Work - ITERATIVE APPROACH)
- [ ] **Phase 1: Audio-Only Baselines**
  - Mean predictor (sanity check)
  - Linear regression
  - Ridge regression
  - XGBoost
  - **Goal**: Establish performance floor (RMSE ~0.15-0.20)

- [ ] **Phase 2: + Lightweight Text Features**
  - Add: word count, unique ratio, avg word length
  - Add: Multilingual sentiment (XLM-RoBERTa)
  - Retrain XGBoost
  - **Goal**: Evaluate text improvement (ΔRMSE > 0.01?)

- [ ] **Phase 3: + Embeddings**
  - Compute embeddings ONCE (paraphrase-multilingual-MiniLM-L12-v2)
  - Cache to disk with joblib
  - Train XGBoost + LightGBM
  - **Goal**: Semantic text understanding (ΔRMSE > 0.02?)

- [ ] **Phase 4: + Metadata**
  - Genre (target encoding)
  - Year (normalized)
  - Final model training
  - **Goal**: Complete feature set

- [ ] **Phase 5: Final Evaluation & Analysis**
  - Test set evaluation (ONCE)
  - Error analysis by language/genre/artist
  - Feature importance analysis
  - Visualizations for thesis

- [ ] **Evaluation Framework**
  - Artist-aware cross-validation (GroupKFold)
  - Metrics: RMSE, MAE, R²
  - Error segmentation (language, genre, valence range)
  - Statistical significance testing

### Documentation & Thesis
- [ ] **Code Documentation**
  - Docstrings for all functions
  - Inline comments
  - README updates

- [ ] **Thesis Writing**
  - Abstract ✅ (this week)
  - Introduction
  - Literature Review
  - Methodology
  - Experiments & Results
  - Discussion
  - Conclusion
  - References

- [ ] **Reproducibility**
  - Requirements.txt complete
  - Setup instructions tested
  - Notebook checkpoints saved
  - Model artifacts versioned

### Final Deliverables
- [ ] **GitHub Repository**
  - Clean, well-organized code
  - Comprehensive README
  - Example usage
  - Results summary

- [ ] **Thesis Document**
  - Complete academic paper
  - Figures and tables
  - Proper citations
  - Formatted per university guidelines

- [ ] **Presentation**
  - Slides for defense
  - Demo (optional)
  - Key findings highlighted

## 🐛 Known Issues

### Dataset - RESOLVED ✅
- **Previous Issue**: songs_enhanced_full.csv contained invalid data
  - **Problem 1**: NaN genre values
  - **Problem 2**: Invalid year (0 values)
  - **Problem 3**: Loudness outliers (>0 dB)
  - **Problem 4**: Tempo outliers (0 BPM)
  - **Solution**: ✅ Comprehensive validation and cleaning completed
  - **Result**: 732,988 clean songs in `dataset/processed/songs_ml_ready.csv`
  - **Quality**: 0 duplicates, 100% target completeness, ALL range violations fixed
  - **Status**: ✅ COMPLETE (November 12, 2025)

- **Issue**: Main CSV file >50MB (cannot open directly in VS Code)
  - **Impact**: Need alternative tools for inspection
  - **Solution**: ✅ Use pandas chunking, command-line tools, or Jupyter (working)

### ML Pipeline - CRITICAL METHODOLOGY ISSUES 🚨
- **Issue**: Artist-level data leakage risk
  - **Problem**: Random train/test split allows same artist in both sets
  - **Impact**: Inflates performance due to shared style
  - **Solution**: ✅ Use GroupShuffleSplit by artist_id (MANDATORY)
  
- **Issue**: Multilingual text handling
  - **Problem**: TextBlob is English-only, dataset is multilingual
  - **Impact**: Sentiment features will be incorrect/missing for non-English
  - **Solution**: ✅ Use `cardiffnlp/twitter-xlm-roberta-base-sentiment` (MANDATORY)
  
- **Issue**: TF-IDF computational cost
  - **Problem**: 700k songs × 1000 features = memory-intensive, slow training
  - **Impact**: Training bottleneck, sparse representations
  - **Solution**: ✅ Use dense embeddings (paraphrase-multilingual-MiniLM-L12-v2)
  
- **Issue**: Embedding recomputation waste
  - **Problem**: Computing embeddings for 700k songs takes hours
  - **Impact**: Wasted time if recomputed each experiment
  - **Solution**: ✅ Cache embeddings to disk with joblib (MANDATORY)
  
- **Issue**: Sequential waterfall approach
  - **Problem**: Original roadmap implied complete feature engineering before modeling
  - **Impact**: Wasted time on features that may not improve performance
  - **Solution**: ✅ Iterative loops - build incrementally, validate continuously

### Scraper - RESOLVED ✅
- **Previous Issue**: Selenium-based scraper too slow (12 sec/song = 136 days)
- **Solution**: ✅ HTTP-based scraper implemented and completed
- **Status**: ✅ ALL DATA COLLECTION COMPLETE (November 10, 2025)

### Collaboration
- **Issue**: GitHub not yet setup
  - **Impact**: Cannot work simultaneously on code
  - **Solution**: Priority task for next phase

- **Issue**: Task division not finalized
  - **Impact**: Potential duplicated or missed work
  - **Solution**: Create task assignment document

## 📊 Project Status Overview

### Overall Progress: ~10% (adjusted due to scraping crisis)

**Phase Breakdown**:
- 🚧 Data Collection: 20% (0.37% scraped, rebuilding scraper for speed)
- � Data Preprocessing: 0% (blocked by data collection)
- 📋 Feature Engineering: 0% (blocked by data collection)
- 📋 Model Training: 0% (blocked by data collection)
- 📋 Evaluation: 0% (blocked by data collection)
- 🚧 Thesis Writing: 5% (planning phase, paused until scraping resolved)

### Timeline Health: ⚠️ AT RISK - Critical Blocker
- **Blocker**: Scraping performance crisis (136 days at current rate)
- **Impact**: All downstream work is blocked (EDA, ML, thesis)
- **Mitigation**: Emergency migration to HTTP-based scraping
- **Recovery Plan**: 
  1. Complete HTTP scraper migration (1-2 days)
  2. Scrape 10K-50K songs (1-3 days)
  3. Resume normal timeline with sufficient data
- **Contingency**: If HTTP scraping also fails, pivot to sampling strategy

## 🎯 Next Milestone

**Milestone 1: Complete Data Validation & Preparation** ✅ ACHIEVED! (November 12, 2025)
- ✅ All data collected (songs_enhanced_full.csv, failed_tracks.csv, unknown_tracks.csv)
- ✅ Validated and cleaned songs_enhanced_full.csv
- ✅ Created final clean dataset: `dataset/processed/songs_ml_ready.csv` (732,988 songs)
- ✅ Quality verified: 0 duplicates, 100% target completeness, ALL ranges valid
- [ ] Document final dataset characteristics
- [ ] Create data dictionary

**Milestone 2: Complete EDA and Baseline (Target: Next 2 Weeks)**
- [ ] EDA complete with insights
- [ ] Features engineered and ready
- [ ] Train/test splits created
- [ ] First baseline model trained for all 4 targets

**Success Criteria**:
- Can load and process full dataset efficiently
- Have visualizations showing data characteristics
- Baseline RMSE/R² established for valence, energy, danceability, popularity
- Code is in GitHub with proper documentation

## 📈 Evolution of Decisions

### Target Variable Decision
- **Initial**: Uncertain between valence, danceability, or popularity
- **Intermediate**: Considered single-target (valence) approach
- **Final Decision**: Multi-target approach - all 4 targets (valence, energy, danceability, popularity)
- **Reasoning**: Comprehensive scope, rich comparative analysis, efficient pipeline reuse, risk mitigation
- **Status**: ✅ Decision finalized, documented in Memory Bank

### Scope Evolution
- **Initial Thought**: Single model, single target
- **Evolution**: Comparative study of multiple algorithms
- **Final Scope**: 4 targets × 5 algorithms = 20 experiments
- **Reasoning**: Comprehensive comparison IS the research contribution
- **Key Insight**: Not just "we predicted X", but "we discovered which approaches work best for which tasks"

### Collaboration Approach
- **Initial**: Unclear division
- **Current Plan**: Person-based or phase-based division with GitHub workflow
- **Next**: Formalize task assignments
