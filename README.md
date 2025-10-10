# 🎵 ML Music Prediction Thesis

A machine learning comparison study for predicting musical attributes from lyrics and audio features. Final year thesis project.

## 📊 Project Overview

This project explores whether and how song lyrics combined with audio features can predict musical characteristics, specifically **valence** (emotional positivity). We compare multiple machine learning algorithms to determine which approaches work best for this multi-modal prediction task.

### Team
- Two students (collaborative project)
- Final year thesis requirement
- Focus: Algorithm comparison and methodology

### Target Prediction
**Primary**: Valence (0-1 scale, emotional positivity)
- Strong connection to lyrical sentiment
- Interesting NLP + audio feature fusion
- Clear thesis narrative

**Optional Secondary**: Danceability (for additional experiments)

## 🎯 Research Questions

1. Can lyrics effectively predict musical valence when combined with audio features?
2. Which ML algorithms perform best for this prediction task?
3. How important are lyrical features compared to audio features?
4. What patterns emerge in the relationship between words and emotional tone?

## 📁 Repository Structure

```
bitirme/
├── dataset/                      # Data collection & preprocessing
│   ├── raw/                     # Original datasets (not in git)
│   ├── processed/               # Cleaned and engineered features
│   ├── scripts/
│   │   ├── chosic_scraper.py   # Web scraper for missing metadata
│   │   ├── genre_mapper.py     # Genre standardization
│   │   ├── data_cleaning.py    # Data validation and cleaning
│   │   └── feature_engineering.py  # Feature extraction
│   ├── notebooks/               # EDA notebooks
│   └── README.md
│
├── ml/                          # Machine learning pipeline
│   ├── preprocessing/           # Data preparation
│   │   ├── text_features.py    # Lyrics feature extraction
│   │   ├── audio_features.py   # Audio feature engineering
│   │   └── data_splitting.py   # Train/test/validation splits
│   ├── models/                  # Model implementations
│   │   ├── baseline.py         # Simple baselines
│   │   ├── linear_models.py    # Regression models
│   │   ├── tree_models.py      # RF, XGBoost, LightGBM
│   │   └── neural_models.py    # Optional: Neural networks
│   ├── evaluation/              # Evaluation and metrics
│   │   ├── metrics.py          # Performance calculations
│   │   └── visualization.py    # Results plotting
│   ├── experiments/             # Experiment orchestration
│   │   ├── run_experiment.py   # Main experiment runner
│   │   └── configs/            # Experiment configurations
│   ├── notebooks/               # Training notebooks
│   └── README.md
│
├── thesis/                      # Academic documentation
│   ├── references/             # Papers and citations
│   ├── figures/                # Generated plots and diagrams
│   ├── sections/               # Thesis chapters/sections
│   └── README.md
│
├── timeline/                    # Project management
│   ├── week-01.MD              # Weekly task tracking
│   └── ...
│
├── memory-bank/                # Project knowledge base
│   ├── projectbrief.md         # Core project definition
│   ├── productContext.md       # Why and how
│   ├── activeContext.md        # Current focus
│   ├── systemPatterns.md       # Architecture
│   ├── techContext.md          # Technologies
│   └── progress.md             # Status tracking
│
├── results/                    # Experiment results (created during runs)
│   ├── models/                 # Saved model artifacts
│   ├── metrics/                # Performance metrics
│   └── figures/                # Generated visualizations
│
├── .gitignore                  # Git ignore patterns
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd bitirme
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (if using NLP features)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Quick Start

1. **Data Preparation**
```bash
cd dataset/scripts
python data_cleaning.py
python feature_engineering.py
```

2. **Run Baseline Experiment**
```bash
cd ml/experiments
python run_experiment.py --config configs/baseline.yaml
```

3. **View Results**
```bash
# Results will be in results/ directory
# Check results/metrics/ for performance numbers
# Check results/figures/ for visualizations
```

## 📊 Dataset

### Source
- **Base**: Spotify songs dataset
- **Features**: Audio attributes from Spotify API
- **Lyrics**: Collected lyrics data
- **Enhanced**: Additional metadata scraped from Chosic.com

### Features

**Audio Features** (from Spotify):
- `danceability`: How suitable for dancing (0-1)
- `energy`: Intensity measure (0-1)
- `valence`: Musical positivity (0-1) ← **Our target**
- `tempo`: BPM
- `loudness`: Overall volume (dB)
- `speechiness`, `acousticness`, `instrumentalness`, `liveness`
- `key`, `mode`, `duration_ms`

**Text Features** (extracted from lyrics):
- TF-IDF vectors
- Sentiment scores
- Lyrical statistics (word count, unique words, etc.)
- Optional: Word embeddings

**Metadata**:
- Genre (scraped + normalized)
- Release year
- Popularity score
- Explicit flag

### Data Size
- Songs: TBD (documenting this week)
- CSV Size: >50MB
- Features: ~15-20 audio + text-derived features

## 🤖 Machine Learning Pipeline

### Algorithms Compared

1. **Baseline**
   - Mean predictor
   - Simple linear regression

2. **Linear Models**
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)

3. **Tree-Based Models**
   - Random Forest Regressor
   - XGBoost
   - LightGBM (optional)

4. **Advanced** (optional)
   - Neural Networks
   - Support Vector Regression

### Evaluation Metrics

**Primary**:
- RMSE (Root Mean Square Error)
- R² Score (Coefficient of Determination)

**Secondary**:
- MAE (Mean Absolute Error)
- Explained Variance

**Visualization**:
- Predicted vs Actual scatter plots
- Feature importance charts
- Learning curves

### Experiment Workflow

```
Raw Data → Cleaning → Feature Engineering → Train/Test Split →
Model Training → Cross-Validation → Evaluation → Comparison Analysis
```

## 📈 Progress Tracking

See [timeline/](timeline/) for weekly task tracking and [memory-bank/progress.md](memory-bank/progress.md) for detailed status.

**Current Phase**: Data preparation and planning
**Next Milestone**: Complete EDA and baseline models

## 🤝 Contributing

This is a two-person collaborative project. See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow guidelines.

### Workflow
1. Create feature branch from `main`
2. Make changes and commit
3. Push branch and create Pull Request
4. Code review by partner
5. Merge to main after approval

### Branch Naming
- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `experiment/description`: ML experiments

## 📚 References

See [thesis/references/](thesis/references/) for collected papers and theses.

**Key Topics**:
- Music emotion prediction
- Lyric sentiment analysis
- Audio feature-based classification
- Multi-modal learning

## 📝 License

[Specify license - e.g., MIT, GPL, or Academic Use Only]

## 📧 Contact

- Student 1: [Name] - [Email/GitHub]
- Student 2: [Name] - [Email/GitHub]
- Advisor: [Name] - [Email]

## 🙏 Acknowledgments

- Dataset providers (Spotify, lyrics sources)
- Chosic.com for genre metadata
- Open source ML libraries (scikit-learn, XGBoost, etc.)

---

**Last Updated**: October 10, 2025
**Status**: Planning & Data Preparation Phase
