# System Patterns: ML Music Prediction Architecture

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT STRUCTURE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  data/             →  Data collection & preprocessing       │
│  ml/               →  Model training & evaluation           │
│  thesis/           →  Academic documentation                │
│  docs/             →  Project management                    │
│  memory-bank/      →  Project knowledge base                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Pipeline Architecture

### Phase 1: Data Collection & Enrichment
```
Original CSV (Spotify data)
    ↓
Chosic Scraper (Selenium + BeautifulSoup)
    ↓
Enhanced CSV (+ popularity, genre, year, explicit)
    ↓
Data Validation & Cleaning
```

**Current Implementation**:
- `chosic_scraper.py`: Main scraper using Selenium
- `fetch_genre_page.py`: Genre page fetcher
- `genre_mapper.py`: Genre normalization
- Rate limiting to avoid server overload
- Sample-first approach before full processing

### Phase 2: Feature Engineering
```
Raw Features → Feature Extraction → Feature Matrix
    ↓
├─ Audio Features (already in dataset)
│  └─ danceability, energy, valence, tempo, etc.
├─ Text Features (to be engineered)
│  └─ TF-IDF, word embeddings, sentiment scores
└─ Metadata Features
   └─ genre encoding, year normalization
```

### Phase 3: Model Training & Evaluation
```
Feature Matrix
    ↓
Train/Test Split (stratified if classification)
    ↓
├─ Model 1: Linear Regression/Logistic
├─ Model 2: Random Forest
├─ Model 3: Gradient Boosting (XGBoost/LightGBM)
├─ Model 4: Neural Network (optional)
└─ Model 5: SVM (optional)
    ↓
Cross-Validation
    ↓
Performance Metrics
    ↓
Comparison Analysis
```

## Key Technical Decisions

### Data Processing
- **Pandas** for CSV manipulation
- **BeautifulSoup + Selenium** for web scraping
- Large CSV files (>50MB) - need efficient processing strategies

### ML Framework Options
- **Scikit-learn**: Core ML algorithms, preprocessing, metrics
- **XGBoost/LightGBM**: Advanced tree-based models
- **TensorFlow/PyTorch**: If including deep learning
- **NLTK/spaCy**: NLP preprocessing for lyrics

### Feature Engineering for Lyrics (English-only)
- **Text Statistics**: Word count, unique words, character count (✅ Phase 2 complete)
- **Sentiment Analysis**: TextBlob for polarity and subjectivity (✅ Phase 3 complete)
- **Word Embeddings**: all-MiniLM-L6-v2 (384-d) (✅ Phase 4 COMPLETE - November 28, 2025)
- **Linguistic Features**: Vocabulary richness, average word length (✅ Phase 2 complete)

**Total Features Available**: 412
- Audio: 21 features (genre, year, cyclical key, audio features)
- Text Stats: 5 features (word count, uniqueness, etc.)
- Sentiment: 2 features (polarity, subjectivity)
- Embeddings: 384 features (semantic vectors)

## Component Relationships

### Dataset Module
**Purpose**: Data acquisition and preprocessing
**Key Files**:
- `chosic_scraper.py`: Web scraping logic
- `genre_mapper.py`: Genre standardization
- Data validation scripts (to be created)
- Feature engineering scripts (to be created)

### ML Module
**Purpose**: Model development and comparison
**Planned Structure**:
```
ml/
├── preprocessing/
│   ├── text_features.py      # Lyrics feature extraction
│   ├── audio_features.py     # Audio feature engineering
│   └── data_splitting.py     # Train/test/val splits
├── models/
│   ├── baseline.py           # Simple baseline models
│   ├── linear_models.py      # Linear/logistic regression
│   ├── tree_models.py        # RF, XGBoost, LightGBM
│   └── neural_models.py      # Optional deep learning
├── evaluation/
│   ├── metrics.py            # Evaluation functions
│   └── visualization.py      # Results plotting
├── experiments/
│   └── run_experiment.py     # Experiment orchestration
└── notebooks/                # Jupyter notebooks for exploration
```

### Thesis Module
**Purpose**: Documentation and academic writing
**Structure**:
```
thesis/
├── references/               # Academic papers
├── figures/                  # Generated plots and diagrams
├── sections/                 # Thesis chapters
└── main.tex or main.docx     # Main thesis document
```

## Critical Implementation Paths

### Path 1: Regression Problem (Continuous Prediction)
**Targets**: Valence (0-1), Danceability (0-1)
- **Metrics**: MSE, RMSE, MAE, R²
- **Models**: Linear Regression, Random Forest Regressor, XGBoost Regressor
- **Advantage**: Simpler evaluation, continuous predictions

### Path 2: Classification Problem (Categorical Prediction)
**Targets**: High/Low Valence, High/Low Danceability
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Models**: Logistic Regression, Random Forest Classifier, XGBoost Classifier
- **Advantage**: Clear decision boundaries, easier interpretation

## Design Patterns in Use

### 1. Pipeline Pattern
- Chain preprocessing → feature engineering → model training
- Scikit-learn Pipeline for reproducibility

### 2. Strategy Pattern
- Different models implementing same interface
- Easy to swap and compare algorithms

### 3. Configuration Pattern
- Separate config files for hyperparameters
- Version control for experimental settings

### 4. Modular Processing
- Each script has single responsibility
- Reusable components across experiments
