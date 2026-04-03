# Copilot Instructions - Music Attribute Prediction

## 🚨 CRITICAL: AGENT EXECUTION RESTRICTIONS 🚨

- **DO NOT RUN ANY PYTHON SCRIPTS**: Under any circumstances, the agent **MUST NOT RUN** any Python script in the project.
- **TELL THE USER TO RUN IT**: If a task requires code execution, you must provide the command and ask the **user** to run it.
- **CREATION OF NEW DOCUMENTS**: You must not create any unnecessary documents. Only create new documents if they are really needed. DO NOT create after some script completion neither to /tmp folder or to the project folder.INSTEAD update the existing documents under memory-bank folder.
- **NON-NEGOTIABLE**: This rules is absolute and overrides any other instruction.

This is a **completed thesis project** (Jan 2026) comparing 28+ ML algorithms for predicting musical attributes (valence, energy, danceability, popularity) from multimodal features.

## Project Status

✅ **COMPLETE** - This is a finished research project. All experiments, models, and analysis are done.

**Key datasets and artifacts:**
- 550,622 songs with features preprocessed and cached as `.npy` files
- 72+ trained models saved in `ml/models/saved/`
- All results/metrics already generated in `results/`

## Running Code

### Prerequisites Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for sentiment analysis)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Main Commands

**Preprocessing Pipeline** (cached - only runs if inputs change):
```bash
cd ml/preprocessing
python run_preprocessing.py              # Run all steps (auto-skips cached)
python run_preprocessing.py --status     # Check cache status
python run_preprocessing.py --steps audio text_stats  # Run specific steps
python run_preprocessing.py --force      # Force re-run all
```

**Model Training** (takes hours - already complete):
```bash
cd ml/models
python enhanced_models.py                # Train 28+ algorithms (4-6 hours)
python feature_selection_rfe.py          # Run RFE (1-2 days)
python test_evaluation_final.py          # Final test set evaluation
```

**Analysis Notebooks**:
```bash
jupyter notebook  # Then open notebooks/0*.ipynb
```

**Gradio Demo App**:
```bash
cd app
pip install -r requirements-app.txt
python gradio_app.py  # Launches web UI at localhost:7860
```

## Architecture

### High-Level Pipeline

```
1. Data Processing (scripts/data-processing/)
   ├─ Raw data → Cleaned/filtered → Artist-aware splits
   └─ 955K songs → 550K English songs with complete metadata

2. Feature Engineering (ml/preprocessing/)
   ├─ Audio (23): Spotify features + cyclical encoding + artist metadata
   ├─ Text (5): Word counts, lexical diversity
   ├─ Sentiment (2): TextBlob polarity & subjectivity
   └─ Embeddings (384): Sentence-transformers semantic vectors
   Total: 414 features → Cached as .npy files

3. Model Training (ml/models/)
   ├─ Experiment 1: Baseline (412 features, no artist)
   ├─ Experiment 2: Artist-enhanced (414 features)
   └─ RFE: Conservative feature elimination (10 features/iter, 1% R² threshold)

4. Analysis (notebooks/)
   └─ 7 comprehensive notebooks covering EDA → Final test results
```

### Dual-Experiment Design

**Experiment 1** (baseline): 412 features (no artist metadata)
**Experiment 2** (artist-enhanced): 414 features (+ artist followers & popularity)

**Key Finding**: Artist context improves popularity prediction by 71% (R²: 0.078 → 0.134) but has minimal impact on intrinsic attributes like energy/valence.

### Directory Organization

- `ml/features/`: Preprocessed `.npy` arrays (23 files: train/val/test × feature types)
- `ml/models/saved/`: Trained model artifacts organized by experiment
  - `experiment1_no_artist/`: Baseline models
  - `experiment2_with_artist/`: Artist-enhanced models
  - `experiment2_with_artist/rfe/`: Feature selection models
  - `experiment2_with_artist/rfe_best/`: Optimal RFE models for deployment
- `results/metrics/`: CSV files with all performance metrics
- `results/figures/`: Publication-ready visualizations (21 figures, 300 DPI)

## Key Conventions

### Feature Processing Patterns

**EDA-Driven Transformations** (applied consistently):
- **Power Transform** (Yeo-Johnson): `acousticness`, `instrumentalness`, `speechiness` (highly right-skewed)
- **Log1p Transform**: `popularity` (target), word counts, `total_artist_followers`
- **Cyclical Encoding**: `key` → `key_sin`, `key_cos` (music theory: B wraps to C)
- **StandardScaler**: `loudness`, `tempo`, `duration_ms`, `year`
- **One-hot**: `genre` (10 categories)

**Feature Groups** (used throughout codebase):
```python
FEATURE_GROUPS = {
    'audio': 0-22,      # 23 features (21 audio + 2 artist)
    'text': 23-27,      # 5 text statistics
    'sentiment': 28-29, # 2 TextBlob features
    'embeddings': 30-413 # 384 sentence-transformer dimensions
}
```

### Artist-Aware Data Splitting

**Critical**: Always use `GroupShuffleSplit` with artist IDs to prevent data leakage.

```python
# Correct: Zero artist overlap between splits
GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
splitter.split(X, y, groups=df['artist_id'])

# Wrong: Random split allows same artist in train/test
train_test_split(X, y, test_size=0.15)
```

**Why**: Artists have distinct styles. Random splits leak artist patterns from train → test, inflating performance metrics.

### Model Training Patterns

**Checkpoint System**: All training scripts use incremental checkpoints to resume after interruptions:
```python
checkpoint_file = checkpoint_dir / 'enhanced_training_checkpoint.json'
# Saves after each model completes
# Can resume mid-training without re-running completed models
```

**Target Variables**: Train **separate models** for each of 4 targets (not multi-output):
- `valence` (0-1): Emotional positivity
- `energy` (0-1): Intensity/activity level
- `danceability` (0-1): Rhythm suitability for dancing
- `popularity` (0-100, log-transformed): Mainstream success

**Standard Metrics**: Report `R²`, `RMSE`, `MAE` for all models (computed on validation set during training, test set only once at the end).

### RFE Methodology

**Authority Model**: `CatBoost_tuned` (1000 iterations, lr=0.05, depth=10)
**Strategy**: Conservative elimination
- Remove 10 features per iteration (not 10%)
- Stop when R² drops >1% OR <20 features remain
- Validate by retraining 6 different models at optimal iteration

**File Naming Convention**:
```
rfe_results_valence_iter23.csv           # RFE iteration results
optimal_features_valence_iter23.csv      # Feature indices to keep
CatBoost_tuned_valence_iter23.pkl        # Model trained with optimal features
```

### Caching and Performance

**Intelligent Caching**: Preprocessing steps cache outputs and skip if inputs unchanged:
- Cache files: `ml/features/.cache/*.json`
- Cache keys: MD5 hash of input data + transformation parameters
- Force refresh: `--force` flag or delete cache files

**Memory-Mapped Arrays**: Large embeddings use `np.load(..., mmap_mode='r')` to avoid loading 384×550K floats into RAM.

## Data Locations

**Processed Data**: `data/processed/`
- `songs.csv` - Main dataset (550,622 songs)
- `train.csv`, `val.csv`, `test.csv` - Artist-aware splits (68.1% / 16.2% / 15.7%)

**Features**: `ml/features/` (.npy arrays)
- Naming: `X_{split}_{feature_type}.npy` and `y_{split}_{target}.npy`
- Example: `X_train_audio.npy`, `y_val_valence.npy`

**Models**: Organized by experiment in `ml/models/saved/`

**Results**: `results/metrics/` (CSV) and `results/figures/` (PNG, 300 DPI)

## Key Files

**Main entry points**:
- `ml/preprocessing/run_preprocessing.py` - Feature engineering orchestrator
- `ml/models/enhanced_models.py` - Train 28+ algorithm comparison
- `ml/models/feature_selection_rfe.py` - Recursive feature elimination
- `ml/models/test_evaluation_final.py` - Final test set evaluation

**Individual preprocessing modules** (`ml/preprocessing/`):
- `process_audio.py` - Audio + artist features
- `process_text_stats.py` - Lyric statistics
- `process_sentiment.py` - TextBlob sentiment
- `process_embeddings.py` - Sentence-transformer vectors
- `process_targets.py` - Target variable transformations

**Analysis notebooks** (`notebooks/`):
- `01_exploratory_data_analysis.ipynb` - Dataset overview
- `04_enhanced_models_analysis.ipynb` - Model comparison (Exp 2)
- `05_feature_importance_analysis.ipynb` - Feature contribution
- `06_error_analysis.ipynb` - Residual patterns
- `07_test_evaluation_analysis.ipynb` - Final test results

## Notes

- **No automated tests**: This is a research project with manual validation via notebooks
- **No CI/CD**: Single-researcher thesis project
- **Linting tools available** (but not enforced): `black`, `flake8` in requirements.txt
- **Thesis document**: Complete writeup in `thesis/thesis.md` (not PDF yet)
- **Gradio app**: Demo UI for predictions on new audio files (uses RFE-optimized models)

## Common Tasks

**Add a new feature type**:
1. Create `ml/preprocessing/process_newfeature.py` with cache support
2. Add to `run_preprocessing.py` STEP_FUNCTIONS
3. Save as `X_{split}_newfeature.npy` in `ml/features/`
4. Update feature loading in model scripts to concatenate new features

**Retrain a specific model**:
```bash
cd ml/models
# Models are trained in enhanced_models.py line-by-line
# Edit the TARGET variable at top and comment out unwanted algorithms
python enhanced_models.py
```

**Reproduce full pipeline** (from scratch):
```bash
# ~4-8 hours total
cd ml/preprocessing && python run_preprocessing.py --force
cd ../models && python enhanced_models.py
# RFE takes 1-2 days - skip unless needed
```
