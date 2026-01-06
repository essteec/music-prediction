# 🎵 Music Attribute Prediction using Machine Learning

**A comprehensive machine learning thesis comparing algorithms for predicting musical attributes from audio features, lyrics, and artist metadata.**

[![Status](https://img.shields.io/badge/Status-Complete-success)]() [![Python](https://img.shields.io/badge/Python-3.8+-blue)]() [![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)]()

## 📊 Project Overview

This thesis presents a **dual-experiment machine learning study** comparing 28+ algorithms for predicting four musical attributes (valence, energy, danceability, popularity) from multimodal features. Using 550,622 English songs with complete lyrics, Spotify audio features, and artist metadata, we demonstrate:

- **71% improvement** in popularity prediction with artist features (R²: 0.0783 → 0.1342)
- **Conservative RFE methodology** maintaining 95%+ performance with 34-394 features (vs 414 full)
- **Gradient boosting dominance** across all targets (CatBoost, LightGBM, XGBoost)
- **Feature efficiency analysis** comparing full-feature vs reduced-feature approaches

### Project Details
- **Type**: Final year thesis project (Complete)
- **Completion Date**: January 2026
- **Dataset**: 550,622 songs (English-only, artist-aware splits)
- **Total Models**: 72 trained and evaluated (28+ algorithms × 2 experiments)
- **Features**: 414 (23 audio + 5 text + 2 sentiment + 384 embeddings)

### Target Variables & Final Results

We predict **4 independent targets** using **separate models per target**:

| Target | Range | Test R² | Best Model | Interpretation |
|--------|-------|---------|------------|----------------|
| **Energy** | 0-1 | **0.85** | CatBoost_tuned | Excellent - intensity well-captured by audio |
| **Danceability** | 0-1 | **0.62** | LightGBM_tuned | Good - rhythm patterns predictable |
| **Valence** | 0-1 | **0.47** | CatBoost_tuned | Moderate - emotional positivity challenging |
| **Popularity** | 0-100 | **0.13** | CatBoost_tuned | Limited - external factors dominate |

**Key Insight**: Intrinsic audio/lyric features excel at predicting **what a song is** (energy, danceability) but struggle with **how successful it becomes** (popularity).

## 🎯 Research Questions

1. How well can different musical attributes be predicted from audio features and lyrics?
2. Which ML algorithms perform best for each target variable?
3. What is the relative importance of audio vs. text features for different targets?
4. Do semantic embeddings improve predictions beyond basic text statistics?
5. How do different feature engineering approaches compare?

## 📁 RepositorContributions

### Key Findings

1. **Artist Context Impact** (Experiment 1 vs 2)
   - Popularity: +71% improvement (R²: 0.078 → 0.134)
   - Valence/Energy/Danceability: Minimal impact (<2%)
   - **Conclusion**: Artist fame predicts success, not song quality

2. **Feature Selection Efficiency** (RFE Analysis)
   - Maintained 95%+ performance with 8-95% fewer features
   - Conservative methodology: 10 features/iteration, 1% R² threshold
   - **Optimal features**: Valence=24, Energy=39
```
│   ├── raw/                     # Original datasets (955K+ songs)
│   ├── processed/               # Clean splits with artist features
│   │   ├── songs.csv           # 550,622 songs (final dataset)
│   │   ├── artists.csv         # 154,247 artists (Spotify metadata)
│   │   ├── train.csv           # 374,997 songs (70%)
│   │   ├── val.csv             # 89,171 songs (15%)
│   │   └── test.csv            # 86,454 songs (15%)
│   ├── scraped/                 # Web scraping results
│   └── external/                # External datasets
│
├── ml/                          # Complete ML pipeline
│   ├── preprocessing/           # Feature engineering
│   │   ├── run_preprocessing.py       # Main orchestrator
│   │   ├── process_audio.py          # Audio features (23 with artist)
│   │   ├── process_text_stats.py     # Text statistics (5)
│   │   ├── process_sentiment.py      # Sentiment (2)
│   │   ├── process_embeddings.py     # Lyric embeddings (384)
│   │   └── data_splitting.py         # Artist-aware splitting
│   ├── models/                  # Training & evaluation scripts
│   │   ├── enhanced_models.py         # 28+ algorithm comparison
│   │   ├── feature_selection_rfe.py   # Recursive feature elimination
│   │   ├── retrain_rfe_best_iterations.py  # RFE validation
│   │   ├── test_evaluation_final.py   # Test set evaluation
│   │   └── saved/                     # Model artifacts
│   └── features/                # Preprocessed features (.npy files)
│
├── notebooks/                   # Analysis notebooks (7 total)
│
├── results/                     # All outputs
│   ├── metrics/                 # Performance CSVs
│   ├── figures/                 # Publication-ready visualizations
│   ├── data-processing/         # Data validation & cleaning
│   │   ├── comprehensive_validation.py  # Data quality checks
│   │   ├── clean_data.py               # Outlier handling
│   │   └── investigate_outliers.py     # Anomaly analysis
│   └── scraping/                # Data collection
│       ├── fetch_artist_data.py        # Spotify API scraping
│       └── add_follower_counts.py      # Artist feature integration
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

**Key Directories**:
- `ml/features/` - All preprocessed features (23 .npy files)
- `results/metrics/` - CSV files with all model performance metrics
- `results/figures/` - 21 publication-ready visualizations (300 DPI)
- `notebooks/` - 7 comprehensive analysis notebooks └── literature-review/       # Paper summaries
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── QUICK_REFERENCE.md           # Command cheat sheet
├── QUQuick Start

### Prerequisites
- Python 3.8+
- ~6GB disk space (dataset + features + models)
- ~8GB RAM for training
- GPU optional (CPU training works fine)

### Installation

```bash
# Clone repository
git clone https://github.com/essteec/music-prediction.git
cd bitirme

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

4. **Download required NLTK data** (for sentiment)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Quick Start Pipeline

**1. Feature Preprocessing** (run once, ~30-60 min for embeddings)
```bExploring Results (All Models Pre-trained)

**View Analysis Notebooks**:
```bash
jupyter notebook

# Open any of the 7 analysis notebooks:
# - 01_exploratory_data_analysis.ipynb    # Dataset overview
# - 04_enhanced_models_analysis.ipynb     # Model comparison
# - 07_test_evaluation_analysis.ipynb     # Final test results
```

**Key Results Files**:
```bash
# Final test results (72 model evaluations)
cat results/metrics/test_evaluation_final_20260101_213938.csv

# RFE optimal features per target
ls results/metrics/experiment2_with_artist/rfe/rfe_optimal_features_*.csv

# All publication figures (21 total)
ls results/figures/
```

### Training Models (Optional - Already Complete)

**Full Pipeline** (if reproducing from scratch):
```bash
# 1. Preprocess features (~1-2 hours first run)
cd ml/preprocessing
python run_preprocessing.py --steps all

# 2. Train enhanced models (~4-6 hours)
cd ../models
python enhanced_models.py

# 3. Run RFE feature selection (~1-2 days)
python feature_selection_rfe.py

# 4. Evaluate on test set (once)
python test_evaluation_final.py

### Overview
- **Total Songs**: 732,988 (English-only, filtered from 1.2M)
- **Full Lyrics**: 100% coverage
- **Source**: Combined Spotify + lyrics datasets
- **Splits**: Artist-aware (zero artist overlap)
  - Train: 386,399 songs (70%)
  - Validation: 82,187 songs (15%)
  - Test: 82,274 songs (15%)

### Features (412 Total)

**Audio Features (2550,622 (English-only with complete metadata)
- **Artists**: 154,247 unique artists (Spotify API metadata)
- **Full Lyrics**: 100% coverage
- **Source**: Combined Spotify + lyrics + artist data
- **Splits**: Artist-aware (zero artist overlap)
  - Train: 374,997 songs (68.1%)
  - Validation: 89,171 songs (16.2%)
  - Test: 86,454 songs (15.7%)

### Features (414 Total)

**Audio Features (23)** - Spotify API + derived:
- Core: `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `duration_ms`
- Musical: `key` (cyclical sin/cos), `mode`, `time_signature`
- Metadata: `genre` (10 genres one-hot), `year` (normalized)
- **Artist** (Experiment 2): `log_total_artist_followers`, `avg_artist_popularity`

**Text Statistics (5)** - Lyric analysis:
- `word_count`, `unique_word_count`, `unique_ratio`, `avg_word_length`, `char_count`

**Sentiment (2)** - TextBlob analysis:
- `polarity` (-1 to +1), `subjectivity` (0-1)

**Embeddings (384)** - Semantic vectors:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense semantic representation of full lyrics
- English-optimized, pre-trained on 1B+ sentences

### Experiments

| Experiment | Features | Focus | Key Result |
|------------|----------|-------|------------|
| **Experiment 1** | 412 (no artist) | Baseline | Energy R²=0.847 |
| **Experiment 2** | 414 (+ artist) | Artist impact | Popularity +71% improvement |

### Data Quality Assurance
- ✅ **100% completeness**: No missing values in targets or features
- ✅ **Zero artist overlap**: Strict GroupShuffleSplit validation
- ✅ **Outlier handling**: 191 outliers investigated and cleaned (0.03% of data)
- ✅ **Validation reports**: 7 comprehensive quality
**Phase 2: Text Statistics** ✅ (November 15, 2025)
- 5 basic lyric statistics
- Word counts, lexical diversity, character counts
- Cached in `.npy` format for fast loading

**Phase 3: Sentiment Analysis** ✅ (November 20, 2025)
- 2 TextBlob features (polarity, subjectivity)
- Full lyrics sentiment extraction
- Integrated with pipeline

**Phase 4: Lyric Embeddings** ✅ (November 28, 2025)
- 384-dimensional semantic vectors
- Model: sentence-transformers/all-MiniLM-L6-v2
- Intelligent caching (~30-60 min first run, instant reload)
- Batch processing with progress tracking

**Total Features Available**: 412 (21 audio + 5 text stats + 2 sentiment + 384 embeddings)

### Model Comparison Study

**AlDual-Experiment Methodology

**Experiment 1: Baseline (412 features, no artist data)**
- Establishes content-only prediction ceiling
- Models: 28+ algorithms (14 × default+tuned)
- Best: XGBoost/CatBoost across all targets

**Experiment 2: Artist-Enhanced (414 features, + artist metadata)**
- Tests artist context impact
- Same 28+ algorithms for fair comparison
- Result: 71% popularity improvement, minimal impact on others

### Feature Selection: Recursive Feature Elimination (RFE)

**Methodology**:
- **Authority**: CatBoost_tuned (1000 iterations, lr=0.05, depth=10)
- **Strategy**: Conservative elimination (10 features/iteration)
- **Stopping**: 1% R² drop from baseline OR minimum 20 features
- **Validation**: 6 models retrained at optimal iterations

**Results per Target**:
| Target | Optimal Features | R² Loss | Efficiency Gain |
|--------|------------------|---------|-----------------|
| Valence | 24 (5.8% of 414) | <1% | 94.2% reduction |
| Energy | 394 (95.2%) | <1% | 4.8% reduction |
| Danceability | 38 (9.2%) | <1% | 90.8% reduction |
| Popularity | 34 (8.2%) | <1% | 91.8% reduction |

### Comprehensive Algorithm Comparison

**28+ Models Trained** (per experiment):
- **Gradient Boosting**: XGBoost, CatBoost, LightGBM, AdaBoost
- **Ensembles**: RandomForest, ExtraTrees
- **Linear**: Ridge, Lasso, LinearRegression, SGDRegressor, LinearSVR
- **Neural**: MLPRegressor
- **Instance-based**: KNeighbors
- **Tree**: DecisionTree
- **Baseline**: Mean predictor

**Each model**: Default + Tuned variant = 28+ total configurations

### Final Test Results (Experiment 2 - Enhanced)

| Target | Best Model | Test R² | Test RMSE | Features Used |
|--------|-----------|---------|-----------|---------------|
| **Energy** | CatBoost_tuned | **0.8503** | 0.0941 | 414 (full) |
| **Danceability** | LightGBM_tuned | **0.6189** | 0.1059 | 414 (full) |
| **Valence** | CatBoost_tuned | **0.4718** | 0.1807 | 414 (full) |
| **Popularity** | CatBoost_tuned | **0.1342** | 1.3646 | 414 (full) |

**Test Set**: 86,454 held-out songs (never seen during training/validation)
scikit-learn>=1.3.0
xgboost>=2.0.0Framework

**Metrics**:
- **R² Score**: Variance explained (0-1, higher better)
- **RMSE**: Root Mean Square Error (lower better, target-scaled)
- **MAE**: Mean Absolute Error (interpretable, same units as target)

**Validation Strategy**:
- **Artist-aware splits**: Zero artist overlap between train/val/test
- **GroupShuffleSplit**: Prevents data leakage from artist style
- **Single test evaluation**: Test set used once for final results only
- **Checkpoint system**: Incremental model saving during training

**Analysis Depth**:
- **Performance**: R²/RMSE heatmaps, ranking tables, model family comparison
- **Features**: Importance extraction, RFE efficiency, group contribution
- **Errors**: Residual analysis, genre patterns, temporal trends, range bias
- **Efficiency**: Training time vs performance, feature count vs accuracy
```

See `requirements.txt` for full dependency list.
& Computation
pandas==2.1.4
numpy==1.26.2

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.3
catboost==1.2.2
lightgbm==4.1.0

# NLP & Embeddings
textblob==0.17.1
sentence-transformers==2.2.2
torch==2.1.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Utilities
joblib==1.3.2
tqdm==4.66.1
requests==2.31.0  # Spotify API
```

**Full list**: See [requirements.txt](requirements.txt) (32 dependencies)

### System Requirements
- **RAM**: 8GB+ recommended (training neural networks)
- **Disk**: 6GB (1.5GB data + 2GB features + 1GB models + 1.5GB results)
- **CPU**: 16+ cores recommended (parallel training)
- **GPU**: Optional (CPU training works, 1-2 days for full pipeline)

### File Formats & Storage
- **Data**: CSV (pandas DataFrame)
- **Features**: NumPy arrays (`.npy`, memory-mapped for large files)
- **Models**: Joblib pickle (`.pkl`, sklearn/xgboost/catboost/lightgbm)
- **Results**:Timeline

**Status**: ✅ **COMPLETE** (October 2025 - January 2026)

### Completed Milestones

- ✅ **Data Collection** (Oct-Nov 2025): 955K songs scraped, 550K validated
- ✅ **Data Cleaning** (Nov 2025): Outlier handling, encoding fixes, 100% quality
- ✅ **EDA & Validation** (Nov 2025): 7 comprehensive analysis notebooks
- ✅ **Feature Engineering** (Nov-Dec 2025): 414 features across 4 modalities
- ✅ **Baseline Models** (Nov 2025): 4 algorithms × 4 targets = 16 baselines
- ✅ **Enhanced Models** (Dec 2025): 28+ algorithms × 4 targets = 112+ models
- ✅ **Experiment 2 - Artist** (Dec 2025): Re-trained all models with artist data
- ✅ **RFE Feature Selection** (Jan 2026): Conservative elimination methodology
- ✅ **Test Evaluation** (Jan 2026): 72 models evaluated on held-out test set
- ✅ **Error Analysis** (Jan 2026): Genre/decade/range patterns identified
- ✅ **Thesis Writing** (Jan 2026): Complete thesis document with all sections

### Key Dates
- **Oct 7, 2025**: Project start
- **Nov 10, 2025**: Data collection complete
- **Dec 5, 2025**: Enhanced models trained
- **Dec 31, 2025**: Experiment 2 complete
- **Jan 1, 2026**: RFE + Test evaluation complete
- **Jan 7, 2026**: Error analysis + Thesis complete

### Documentation Links

- **Progress Tracking**: [docs/memory-bank/progress.md](docs/memory-bank/progress.md)
- **Current Status**: [docs/memory-bank/activeContext.md](docs/memory-bank/activeContext.md)
- **ML Pipeline**: [docs/memory-bank/ML_ROADMAP.md](docs/memory-bank/ML_ROADMAP.md)
- **Thesis**: [thesis/thesis.md](thesis/thesis
- Inherently difficult task (external factors)
- Limited predictive power from intrinsic features
- Demonstrates model limitations realistically

### Feature Importance Insights

1. **Audio features dominate** for energy and danceability
2. **Text features critical** for valence prediction (+47% improvement)
3. **Sentiment polarity** strongly correlated with valence
4. **Embeddings potential**: May further improve valence and popularity

### Model Comparison

- **XGBoost**: Best overall performance across all targets
- **Ridge**: Competitive for linear relationships
- **Linear**: Good baseline, struggles with complex patterns
- **Mean**: Essential reference point

## 1. Artist Context vs Content Features

**Experiment Comparison**:
| Target | Exp 1 (No Artist) | Exp 2 (+ Artist) | Improvement |
|--------|-------------------|------------------|-------------|
| Energy | 0.8468 | 0.8503 | +0.4% |
| Danceability | 0.6185 | 0.6189 | +0.1% |
| Valence | 0.4742 | 0.4718 | -0.5% |
| Popularity | **0.0783** | **0.1342** | **+71%** ⭐ |

**Conclusion**: Artist fame predicts **success** (popularity), not **quality** (intrinsic attributes).

### 2. Feature Selection Efficiency (RFE)

**Dramatic Feature Reduction**:
- **Valence**: 414 → 24 features (94% reduction, <1% R² loss)
- **Danceability**: 414 → 38 features (91% reduction, <1% R² loss)
- **Popularity**: 414 → 34 features (92% reduction, <1% R² loss)
- **Energy**: 414 → 394 features (5% reduction, <1% R² loss)

**Insight**: Most targets need <10% of features; Energy uniquely requires broad feature coverage.

### 3. Algorithm Family Performance

**Gradient Boosting Dominance**:
- **Top 3 models**: CatBoost, LightGBM, XGBoost (all gradient boosting)
- **Average superiority**: 15-20% better R² than linear models
- **Hyperparameter tuning**: 2-5% average improvement (except RandomForest: -8.7%)

### 4. Error Patterns

**Genre Bias**:
- Rock: Consistently overestimated valence
- Pop: Consistently underestimated popularity

**Temporal Trends**:
- Modern songs (2020s): Harder to predict (+10-15% RMSE)
- Classic songs (1960s-1980s): More stable predictions

**Range Bias**:
- Extreme values (very high/low): 2-3× higher errors
- Regression to mean effect in all models

### 5. Practical Implications

- ✅ **Energy/Danceability**: Highly predictable from audio (R² > 0.6)
- ⚠️ **Valence**: Moderate predictability, requires multimodal features (R² ≈ 0.47)
- ❌ **Popularity**: Largely unpredictable from intrinsic features (R² ≈ 0.13)

**Recommendation**: Use intrinsic features for **content understanding**, external data for **success prediction**.
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions
- Keep preprocessinributions

### Novel Contributions

1. **Dual-Experiment Ablation Study**
   - First systematic comparison of content-only vs content+artist models
   - Quantifies artist fame impact: 71% improvement for popularity, negligible for others
   - Demonstrates fundamental difference between intrinsic and extrinsic prediction

2. **Conservative RFE Methodology**
   - Three critical bug fixes for scientifically rigorous feature elimination
   - Dual stopping criteria: 1% R² threshold + minimum 20 features
   - Separate optimal subsets per target (not one-size-fits-all)

3. **Comprehensive Algorithm Comparison**
   - 72 total model evaluations (28+ algorithms × 2 experiments + RFE)
   - Evidence of gradient boosting superiority across all music targets
   🔬 Reproducibility & Replication

### Dataset Access
Due to size constraints (1.5GB+), raw data not included in repository.

**To replicate**:
1. Original dataset: [Kaggle Spotify Songs](https://www.kaggle.com/) (search "spotify songs lyrics")
2. Artist data: Fetch via [Spotify Web API](https://developer.spotify.com/documentation/web-api)
3. Or use preprocessed features: `ml/features/*.npy` (if shared separately)

### Replicating Results

**Full Pipeline** (from raw data):
```bash
# 1. Data cleaning (~30 min)
python scripts/data-processing/comprehensive_validation.py
python scripts/data-processing/clean_data.py

# 2. Artist data collection (~2-3 hours, requires Spotify API key)
python scripts/scraping/fetch_artist_data.py
python scripts/scraping/add_follower_counts.py

# 3. Feature preprocessing (~1-2 hours)
python ml/preprocessing/run_preprocessing.py --steps all

# 4. Model training (~6-8 hours)
python ml/models/enhanced_models.py

# 5. RFE + Te

- **GitHub**: [@essteec](https://github.com/essteec)
- **Project**: Final Year Thesis (Complete)
- **Institution**: [University Name]
- **Completion**: January 2026

## 🙏 Acknowledgments

**Data Sources**:
- Spotify Web API (audio features + artist metadata)
- Public lyrics datasets (955K+ songs)

**Tools & Libraries**:
- **ML**: scikit-learn, XGBoost, CatBoost, LightGBM
- **NLP**: sentence-transformers (all-MiniLM-L6-v2), TextBlob
- **Visualization**: matplotlib, seaborn
- **Computation**: NumPy, pandas

**Research Community**:
- Music Information Retrieval (MIR) researchers
- Open-source ML/NLP contributors

---

## 📊 Quick Stats Summary

| Metric | Value |
|--------|-------|
| **Songs** | 550,622 |
| **Artists** | 154,247 |
| **Features** | 414 |
| **Models Trained** | 72 (28+ algorithms × 2 experiments + RFE) |
| **Test Samples** | 86,454 (held-out, never seen) |
| **Best Performance** | Energy R²=0.85, Danceability R²=0.62 |
| **Analysis Notebooks** | 7 (fully executed, publication-ready) |
| **Figures Generated** | 21 (300 DPI, academic styling) |
| **Project Duration** | 3 months (Oct 2025 - Jan 2026) |
| **Status** | ✅ **COMPLETE** |

---

**Last Updated**: January 7, 2026  
**Status**: ✅ Complete - All ML work and thesis writing finished  
**Version**: v2.0 - Dual experiments, RFE, comprehensive evaluation
  type={Bachelor's Thesis}
}
```eview.md) for 10+ paper analysis and [thesis/LITERATURE_REVIEW_SUMMARY.md](thesis/LITERATURE_REVIEW_SUMMARY.md)
- **Dataset Sources**: Spotify Web API, lyrics datasets
- **Tools**: scikit-learn, XGBoost, sentence-transformers
- **Inspiration**: Music emotion recognition research community
- **Thanks**: Open source ML and NLP libraries

---

**Last Updated**: November 28, 2025  
**Status**: Phase 5 - Final Evaluation  
**Version**: v1.0 - All features implemented, baseline results established
