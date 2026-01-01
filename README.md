# 🎵 Music Attribute Prediction using ML (this file is OUTDATED!)

A comprehensive machine learning thesis comparing algorithms for predicting musical attributes from audio features and lyrics. Multi-target prediction study with 732K+ English songs.

## 📊 Project Overview

This project systematically compares machine learning algorithms for predicting four distinct musical attributes from audio features and lyric text. Using a dataset of 732,988 English songs with full lyrics and Spotify audio features, we explore which features and algorithms work best for different prediction tasks.

### Team
- **Type**: Final year thesis project
- **Approach**: Systematic algorithm comparison
- **Dataset**: 732,988 songs (English-only, artist-aware splits)
- **Focus**: Multi-target prediction with comprehensive feature engineering

### Target Variables (Multi-Target Approach)

We predict **4 independent targets** using **4 separate models**:

1. **Valence** (0-1): Emotional positivity
   - Strong lyrical connection, NLP showcase
   - Current R²: 0.372 (with text features)
   
2. **Energy** (0-1): Intensity/activity  
   - Strong audio feature connection
   - Current R²: 0.834 (excellent performance)
   
3. **Danceability** (0-1): Dance suitability
   - Rhythm and tempo focused
   - Current R²: 0.549 (good performance)
   
4. **Popularity** (0-100): Track success
   - Complex, external factors involved
   - Current R²: 0.116 (inherently challenging)

**Rationale**: This multi-target approach provides comprehensive comparative analysis, demonstrates which features work for which tasks, and mitigates risk through multiple successful predictions.

## 🎯 Research Questions

1. How well can different musical attributes be predicted from audio features and lyrics?
2. Which ML algorithms perform best for each target variable?
3. What is the relative importance of audio vs. text features for different targets?
4. Do semantic embeddings improve predictions beyond basic text statistics?
5. How do different feature engineering approaches compare?

## 📁 Repository Structure

```
bitirme/
├── data/                         # All datasets and processing
│   ├── raw/                     # Original datasets
│   │   ├── songs_enhanced_full.csv              # 732,988 songs with full metadata
│   │   ├── songs_with_attributes_and_lyrics.csv # Initial dataset
│   │   └── songs_with_lyrics_and_timestamps.csv # Spotify data
│   ├── processed/               # Ready-to-use splits
│   │   ├── train.csv           # 386,399 songs (70%)
│   │   ├── val.csv             # 82,187 songs (15%)
│   │   ├── test.csv            # 82,274 songs (15%)
│   │   └── english_ml_ready.csv # Full English dataset
│   ├── scraped/                 # Web scraping results
│   │   ├── genre_mappings.csv  # Genre normalization
│   │   └── songs_enhanced_full.csv
│   └── external/                # External datasets
│
├── ml/                          # Complete ML pipeline
│   ├── preprocessing/           # Feature engineering (Phase 1-4)
│   │   ├── run_preprocessing.py       # Main orchestrator
│   │   ├── process_audio.py          # Audio features (21)
│   │   ├── process_text_stats.py     # Text statistics (5)
│   │   ├── process_sentiment.py      # Sentiment (2)
│   │   ├── process_embeddings.py     # Lyric embeddings (384)
│   │   ├── data_splitting.py         # Artist-aware splitting
│   │   ├── pipeline_utils.py         # Caching & utilities
│   │   ├── EMBEDDINGS_README.md      # Embeddings documentation
│   │   └── __init__.py
│   ├── models/                  # Training scripts
│   │   ├── baseline_models.py         # Audio-only (21 features)
│   │   ├── text_stats_models.py       # Audio + text stats (26)
│   │   ├── sentiment_models.py        # Audio + sentiment (28)
│   │   ├── combined_text_models.py    # Audio + text + sentiment (33)
│   │   ├── embedding_models.py        # Audio + embeddings (405)
│   │   ├── full_features_models.py    # All features (412)
│   │   ├── compare_text_approaches.py # Text comparison
│   │   └── saved/                     # Saved model artifacts
│   ├── features/                # Preprocessed features (.npy files)
│   │   ├── X_train_audio.npy          # 386,399 × 21
│   │   ├── X_train_text_stats.npy     # 386,399 × 5
│   │   ├── X_train_sentiment.npy      # 386,399 × 2
│   │   ├── X_train_embeddings.npy     # 386,399 × 384
│   │   ├── y_train_*.npy              # 4 target arrays
│   │   ├── *_scaler.pkl               # Feature scalers
│   │   └── preprocessing_metadata.json # Feature names & config
│   ├── evaluation/              # Metrics and plotting
│   └── notebooks/               # Exploratory analysis
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_advanced_eda.ipynb
│   └── 03_feature_files_eda.ipynb
│
├── scripts/                     # Utility scripts
│   ├── data-processing/         # Data preparation
│   └── scraping/                # Web scraping tools
│
├── results/                     # Model outputs
│   ├── metrics/                 # Performance CSVs and reports
│   ├── figures/                 # Visualizations
│   └── models/                  # Saved models
│
├── docs/                        # Documentation
│   ├── memory-bank/             # Project knowledge base
│   │   ├── projectbrief.md     # Core project definition
│   │   ├── productContext.md   # Why and how
│   │   ├── activeContext.md    # Current focus
│   │   ├── systemPatterns.md   # Architecture
│   │   ├── techContext.md      # Technologies
│   │   ├── progress.md         # Status tracking
│   │   └── ML_ROADMAP.md       # ML pipeline roadmap
│   ├── reports/                 # Validation reports
│   └── timeline.MD              # Project timeline
│
├── thesis/                      # Academic writing
│   ├── thesis.md                # Main thesis document
│   ├── lit-review.md            # Literature review
│   └── literature-review/       # Paper summaries
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── QUICK_REFERENCE.md           # Command cheat sheet
├── QUICKSTART.md                # Getting started guide
├── CONTRIBUTING.md              # Contribution guidelines
└── DECISION_SUMMARY.md          # Design decisions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- ~4GB disk space for dataset and features
- ~2GB RAM for preprocessing
- Virtual environment (`.venv` already configured)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/essteec/music-prediction.git
cd bitirme
```

2. **Activate virtual environment**
```bash
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data** (for sentiment)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Quick Start Pipeline

**1. Feature Preprocessing** (run once, ~30-60 min for embeddings)
```bash
cd ml/preprocessing

# Process all features (recommended)
python run_preprocessing.py --steps all

# Or run individual steps
python run_preprocessing.py --steps audio
python run_preprocessing.py --steps text_stats
python run_preprocessing.py --steps sentiment
python run_preprocessing.py --steps embeddings  # Takes 30-60 min first time
```

**2. Train Models**
```bash
cd ml/models

# Baseline (audio-only, 21 features)
python baseline_models.py

# Text statistics (audio + text stats, 26 features)
python text_stats_models.py

# Full feature set (412 features: audio + text + sentiment + embeddings)
python full_features_models.py
```

**3. View Results**
```bash
# Metrics saved to results/metrics/
cat results/metrics/*_results.csv

# Models saved to ml/models/saved/
ls ml/models/saved/
```

**4. Explore Data**
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in notebooks/ directory
# - 01_exploratory_data_analysis.ipynb
# - 02_advanced_eda.ipynb
# - 03_feature_files_eda.ipynb
```

## 📊 Dataset

### Overview
- **Total Songs**: 732,988 (English-only, filtered from 1.2M)
- **Full Lyrics**: 100% coverage
- **Source**: Combined Spotify + lyrics datasets
- **Splits**: Artist-aware (zero artist overlap)
  - Train: 386,399 songs (70%)
  - Validation: 82,187 songs (15%)
  - Test: 82,274 songs (15%)

### Features (412 Total)

**Audio Features (21)** - from Spotify API:
- `acousticness`: Acoustic vs. electric (0-1)
- `danceability`: Dance suitability (0-1)
- `duration_ms`: Track length (milliseconds)
- `energy`: Intensity measure (0-1) ← **Target**
- `instrumentalness`: Vocal presence (0-1)
- `key`: Musical key (0-11)
- `liveness`: Live performance indicator (0-1)
- `loudness`: Overall volume (dB)
- `mode`: Major (1) or Minor (0)
- `speechiness`: Spoken word presence (0-1)
- `tempo`: Beats per minute
- `time_signature`: Beats per bar (3-7)
- `valence`: Musical positivity (0-1) ← **Target**
- Plus 8 derived audio features

**Text Statistics (5)** - from lyrics:
- `word_count`: Total words in lyrics
- `unique_word_count`: Vocabulary size
- `unique_ratio`: Lexical diversity
- `avg_word_length`: Average word length
- `char_count`: Total character count

**Sentiment (2)** - via TextBlob:
- `polarity`: Positive/negative sentiment (-1 to 1)
- `subjectivity`: Subjective vs. objective (0-1)

**Embeddings (384)** - semantic vectors:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense semantic representation of full lyrics
- 384-dimensional English-optimized vectors

### Target Variables (4 Independent Predictions)

1. **Valence** (0-1): Emotional positivity/happiness
   - Best text connection, NLP showcase
   - Current best: R² = 0.372
   
2. **Energy** (0-1): Intensity/activity level
   - Strong audio feature performance
   - Current best: R² = 0.834
   
3. **Danceability** (0-1): Dance suitability
   - Rhythm-focused, balanced difficulty
   - Current best: R² = 0.549
   
4. **Popularity** (0-100): Track success metric
   - Challenging due to external factors
   - Current best: R² = 0.116

### Data Quality
- **Missing Values**: None (100% complete)
- **Artist Overlap**: Zero (strict artist-based splits)
- **Language**: English only (NLTK stopwords filter)
- **Validation**: Comprehensive reports in `docs/reports/`

## 🤖 Machine Learning Pipeline

### Feature Engineering Phases (COMPLETE ✅)

**Phase 1: Audio Features** ✅ (November 11, 2025)
- 21 Spotify audio features
- StandardScaler normalization
- Validation: Zero missing, no infinite values

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

**Algorithms Tested** (per target):
1. **Mean Baseline**: Average target value
2. **Linear Regression**: Simple linear model
3. **Ridge Regression**: L2 regularized linear model
4. **XGBoost Regressor**: Gradient boosted trees

**Feature Combinations**:
1. Audio-only (21 features) - `baseline_models.py`
2. Audio + Text Stats (26) - `text_stats_models.py`
3. Audio + Sentiment (28) - `sentiment_models.py`
4. Audio + Text + Sentiment (33) - `combined_text_models.py`
5. Audio + Embeddings (405) - `embedding_models.py`
6. Full Features (412) - `full_features_models.py`

### Current Best Results (Validation Set)

| Target | Best Model | Features | R² Score | RMSE |
|--------|-----------|----------|----------|------|
| **Energy** | XGBoost | Audio + Text Stats (26) | 0.834 | 0.081 |
| **Danceability** | XGBoost | Audio + Text Stats (26) | 0.549 | 0.104 |
| **Valence** | XGBoost | Audio + Text Stats (26) | 0.372 | 0.154 |
| **Popularity** | XGBoost | Audio + Text Stats (26) | 0.116 | 15.41 |

*Note: Embeddings results pending full evaluation*

### Evaluation Metrics

**Primary Metrics**:
- **R² Score**: Proportion of variance explained (0-1, higher better)
- **RMSE**: Root Mean Square Error (lower better)

**Interpretation**:
- R² > 0.7: Excellent prediction
- R² 0.5-0.7: Good prediction
- R² 0.3-0.5: Moderate prediction (valence)
- R² < 0.3: Challenging prediction (popularity)

**Validation Strategy**:
- Artist-aware splits (zero artist overlap)
- 70/15/15 train/val/test split
- Test set held out for final evaluation

## 🛠️ Technology Stack

### Core Dependencies
```python
# Data Processing
pandas>=1.5.0
numpy>=1.24.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# NLP & Embeddings
textblob>=0.17.0
nltk>=3.8.0
sentence-transformers>=2.2.0  # Phase 4: Lyric embeddings
torch>=2.0.0                  # Required by sentence-transformers

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0
pyyaml>=6.0
```

See `requirements.txt` for full dependency list.

### File Formats
- **Data**: CSV (pandas)
- **Features**: NumPy arrays (`.npy`) with pickle scalers (`.pkl`)
- **Models**: joblib serialization
- **Results**: CSV metrics, PNG/PDF figures

### Computing Requirements
- **RAM**: ~2GB for preprocessing, ~4GB for full feature models
- **Disk**: ~4GB (dataset + features + models)
- **Time**: 30-60 min for embeddings (first run), <5 min for other features

## 📈 Project Status

**Current Phase**: Phase 5 - Final Evaluation & Analysis (November 28, 2025)

### Completed Phases ✅

- ✅ **Phase 1**: Dataset Collection & Cleaning (October 2025)
  - 732,988 English songs with full lyrics
  - Artist-aware splits implemented
  
- ✅ **Phase 2**: Exploratory Data Analysis (October 2025)
  - Comprehensive EDA notebooks
  - Data quality validation
  
- ✅ **Phase 3**: Feature Engineering (November 2025)
  - Audio features (21)
  - Text statistics (5)
  - Sentiment analysis (2)
  - Lyric embeddings (384)
  
- ✅ **Phase 4**: Baseline Model Training (November 2025)
  - 6 feature combinations tested
  - 4 algorithms compared
  - Best results: Energy R²=0.834

### Current Work 🔄

**Phase 5: Final Evaluation** (In Progress)
- [ ] Analyze embeddings model performance vs. baselines
- [ ] Feature importance analysis (XGBoost)
- [ ] Final model selection per target
- [ ] Test set evaluation (one-time final validation)
- [ ] Error analysis and visualizations

**Phase 6: Thesis Writing** (HIGH PRIORITY)
- [ ] Literature review (10 similar theses)
- [ ] Abstract writing
- [ ] Methodology documentation
- [ ] Results and discussion sections
- [ ] Conclusion and future work

### Quick Links

- **Detailed Status**: [docs/memory-bank/progress.md](docs/memory-bank/progress.md)
- **Current Context**: [docs/memory-bank/activeContext.md](docs/memory-bank/activeContext.md)
- **ML Roadmap**: [docs/memory-bank/ML_ROADMAP.md](docs/memory-bank/ML_ROADMAP.md)
- **Quick Commands**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 📚 Key Findings & Insights

### Performance by Target

**Energy (R² = 0.834)** ⭐ Excellent
- Strong audio feature correlation (tempo, loudness)
- XGBoost significantly outperforms linear models
- Minimal benefit from text features

**Danceability (R² = 0.549)** ✅ Good
- Rhythm and tempo are key predictors
- Moderate improvement with text features
- Balanced prediction difficulty

**Valence (R² = 0.372)** 📊 Moderate
- Most challenging audio-based prediction
- **Text features crucial** (0.252 → 0.372 R²)
- Strong connection to lyrical sentiment
- Prime candidate for embeddings improvement

**Popularity (R² = 0.116)** ⚠️ Challenging
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

## 🎓 Academic Context

### Research Contribution
This thesis contributes a **systematic comparison** of ML algorithms and feature engineering approaches for multi-target music attribute prediction, with emphasis on:
- Artist-aware validation methodology (prevents data leakage)
- Comprehensive feature engineering pipeline (412 features)
- Multi-target approach (demonstrates task-specific feature importance)
- Reproducible preprocessing with intelligent caching

### Related Work
- Music emotion recognition from lyrics
- Audio feature-based classification
- Multi-modal music analysis
- Sentiment analysis in music

See [thesis/lit-review.md](thesis/lit-review.md) for detailed literature review.

## 🤝 Contributing

This is a thesis project with collaborative development. See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow guidelines.

### Development Workflow
1. Create feature branch from `main`
2. Make changes with clear commit messages
3. Update relevant documentation
4. Push branch and create Pull Request
5. Merge after review

### Branch Naming Conventions
- `feature/description`: New features or experiments
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `analysis/description`: Data analysis or EDA
- `model/description`: Model development

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions
- Keep preprocessing modular and cached

## 📝 License

Academic use only - Final year thesis project.

## 📧 Contact & Links

- **Repository**: [github.com/essteec/music-prediction](https://github.com/essteec/music-prediction)
- **Owner**: essteec
- **Project Type**: Final Year Thesis
- **Institution**: [Your University]
- **Advisor**: [Advisor Name]

## 🙏 Acknowledgments

- **Dataset Sources**: Spotify Web API, lyrics datasets
- **Tools**: scikit-learn, XGBoost, sentence-transformers
- **Inspiration**: Music emotion recognition research community
- **Thanks**: Open source ML and NLP libraries

---

**Last Updated**: November 28, 2025  
**Status**: Phase 5 - Final Evaluation  
**Version**: v1.0 - All features implemented, baseline results established
