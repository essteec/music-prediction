# Tech Context: Technologies and Tools

## Core Technologies

### Python (Primary Language)
- **Version**: 3.13 (based on `__pycache__` evidence)
- **Why**: De facto standard for ML, excellent libraries, team familiarity

### Data Processing Stack
- **pandas**: CSV manipulation, data cleaning, feature engineering
- **numpy**: Numerical operations, array processing
- **beautifulsoup4**: HTML parsing for web scraping
- **selenium**: Browser automation for dynamic content scraping

### Machine Learning Stack

#### Core Framework
- **scikit-learn**: Essential for:
  - Preprocessing (StandardScaler, LabelEncoder, TF-IDF)
  - Model selection (train_test_split, cross_val_score)
  - Algorithms (LinearRegression, RandomForest, SVM)
  - Metrics (r2_score, mean_squared_error, accuracy_score)

#### Advanced Models
- **XGBoost**: Gradient boosting, often best performance
- **LightGBM**: Fast gradient boosting, good for large datasets
- **CatBoost**: Handles categorical features well (for genre encoding)

#### NLP/Text Processing
- **TextBlob**: English sentiment analysis (PRIMARY)
  - Provides polarity (-1 to +1) and subjectivity (0 to 1)
  - Fast, lightweight, effective for English-only dataset
- **sentence-transformers**: English embeddings (OPTIONAL for Phase 4)
  - Model: `all-MiniLM-L6-v2` (384-d, English-optimized)
  - Fast, semantic, compact for semantic understanding
- **nltk** or **spaCy**: Text preprocessing, tokenization
- **scikit-learn TfidfVectorizer**: Only for small-scale benchmarking (NOT primary)

#### Optional Deep Learning (Not Used)
- **TensorFlow** or **PyTorch**: Skipped for traditional ML focus
- **Keras**: Not used - traditional ML sufficient for thesis

### Data Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots (optional)
- **pandas plotting**: Quick data exploration

### Development Tools

#### Version Control
- **Git**: Already initialized (`.git/` present)
- **GitHub**: Planned for collaboration and portfolio

#### Environment Management
- **venv** or **conda**: Virtual environment (recommended to create)
- **requirements.txt**: Currently exists in dataset folder

#### Notebooks
- **Jupyter**: For exploratory data analysis and experimentation
- **JupyterLab**: Enhanced notebook interface (optional)

#### Code Quality
- **black**: Code formatting (recommended)
- **flake8** or **pylint**: Linting (recommended)
- **pytest**: Testing framework (recommended for critical functions)

## Current Dependencies

From `dataset/requirements.txt`:
```
pandas
beautifulsoup4
selenium
requests  # Added for HTTP-based scraper
```

## Dataset Files (November 24, 2025)

### ML-Ready Dataset ✅ COMPLETE
1. **data/processed/english_ml_ready.csv** (PRIMARY INPUT)
   - Clean, English-only dataset for ML pipeline
   - **Size**: 732,988 songs
   - **Splits**: 386,399 train / 82,187 val / 82,274 test (artist-aware)
   - **Features**: All audio features + lyrics + metadata (genre, year, explicit)
   - **Status**: ✅ READY - Current production dataset
   - **Validation**: Complete - no NaN values, valid ranges, deduplicated

### Scraped Data Files (Archive)
2. **data/scraped/songs_enhanced_full.csv**
   - All successful scrapes from Chosic
   - Contains: popularity, genre, year, explicit flag
   - **Status**: Archive - superseded by english_ml_ready.csv
   
3. **data/scraped/failed_tracks.csv**
   - Failed scraping attempts
   - **Status**: Archive - for reference only
   
4. **data/scraped/genre_mappings.csv**
   - Genre normalization mappings
   - **Status**: Reference file

### Original Data Files (Archive)
5. **data/raw/songs_with_attributes_and_lyrics.csv**
   - Original base dataset (955,320 songs, 1.5GB)
   - Audio features + lyrics
   - **Status**: Archive - source for english_ml_ready.csv

## Recommended Additional Dependencies

### For ML Pipeline
```
# Core ML
scikit-learn>=1.3.0
xgboost>=2.0.0

# NLP - English-only
textblob>=0.17.0  # Sentiment analysis (PRIMARY)
sentence-transformers>=2.2.0  # Optional for Phase 4 embeddings

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm  # Progress bars
joblib  # Model persistence & caching

# Jupyter
jupyter
ipykernel
```

## Development Setup

### Recommended Structure
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# For Jupyter kernel
python -m ipykernel install --user --name=bitirme
```

### Configuration Files to Create
1. **`.gitignore`**: Exclude large files, cache, credentials
2. **`requirements.txt`**: Root-level dependencies
3. **`config.yaml`**: ML experiment configurations
4. **`setup.py`**: Package installation (if creating reusable modules)

## Technical Constraints

### Data Size
- **Current Issue**: `songs_with_attributes_and_lyrics.csv` > 50MB
- **Implications**: 
  - Cannot be viewed directly in VS Code
  - Need efficient chunk processing for large operations
  - Git LFS may be needed for version control
  - Consider data sampling for quick experiments

### Memory Management
- **Strategy**: 
  - Compute embeddings ONCE, cache to disk with `joblib`
  - Use pandas chunking for initial CSV processing
  - Artist-aware GroupShuffleSplit prevents data leakage
- **Dataset Size**: 732,988 songs (English-only, cleaned)
- **Embedding Cache** (Optional Phase 4):
  ```python
  # Compute once
  embeddings = model.encode(lyrics_list, batch_size=64)
  joblib.dump(embeddings, 'ml/features/train_embeddings.pkl')
  
  # Reuse in future runs
  embeddings = joblib.load('ml/features/train_embeddings.pkl')
  ```
- **Avoid**: TF-IDF on 700k songs (memory-intensive, sparse)

### Browser Automation
- **ChromeDriver**: Needs to be in PATH or specified explicitly
- **Headless Mode**: Important for server/automated runs
- **Rate Limiting**: Essential to avoid being blocked

## Compute Requirements

### For Model Training
- **CPU**: Sufficient for scikit-learn, XGBoost, LightGBM
- **RAM**: 8GB+ recommended for full dataset processing
- **GPU**: Optional, only if using deep learning (TensorFlow/PyTorch)

### For Development
- **Storage**: ~5-10GB for datasets, models, and environments
- **Network**: Needed for scraping (if continuing data collection)

## Tool Usage Patterns

### Data Pipeline
```python
# Standard import pattern
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Model Training
```python
# Common workflow with CRITICAL data split rule
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ⚠️ CRITICAL: Artist-aware split (prevents data leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['artist_id']))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
```

### Experiment Tracking
- **Manual**: CSV files with results
- **Advanced** (optional): MLflow, Weights & Biases
- **Minimal**: JSON files with hyperparameters and results

## Platform
- **OS**: Linux (based on environment info)
- **Shell**: zsh
- **IDE**: VS Code (implied by context)
