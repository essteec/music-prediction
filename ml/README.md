# ML Module

Machine learning pipeline for music valence prediction.

## Structure

```
ml/
├── preprocessing/    # Data preparation
├── models/          # Model implementations
├── evaluation/      # Metrics and visualization
├── experiments/     # Experiment orchestration
└── notebooks/       # Training notebooks
```

## Modules

### Preprocessing (`preprocessing/`)

**`text_features.py`**
- Lyrics preprocessing (cleaning, tokenization)
- TF-IDF vectorization
- Sentiment analysis extraction
- Word embeddings (optional)

**`audio_features.py`**
- Feature scaling
- Feature selection
- Interaction features
- Dimensionality reduction (optional)

**`data_splitting.py`**
- Train/validation/test splits
- Cross-validation setup
- Stratification if needed

### Models (`models/`)

**`baseline.py`**
- Mean/median predictors
- Simple linear regression
- Baseline metrics

**`linear_models.py`**
- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet

**`tree_models.py`**
- Random Forest
- XGBoost
- LightGBM
- Hyperparameter tuning

**`neural_models.py`** (Optional)
- Feedforward neural networks
- Custom architectures

### Evaluation (`evaluation/`)

**`metrics.py`**
- RMSE, MAE, R² calculation
- Cross-validation utilities
- Statistical comparison

**`visualization.py`**
- Prediction scatter plots
- Feature importance charts
- Learning curves
- Error analysis plots

### Experiments (`experiments/`)

**`run_experiment.py`**
- Main experiment orchestration
- Config-based training
- Results logging
- Model artifact saving

**`configs/`**
- YAML configuration files
- Different experiment setups
- Hyperparameter specifications

## Quick Start

### 1. Prepare Data
```python
from ml.preprocessing.data_splitting import create_splits
from ml.preprocessing.text_features import extract_tfidf_features
from ml.preprocessing.audio_features import scale_audio_features

# Load cleaned data
df = pd.read_csv('dataset/processed/songs_cleaned.csv')

# Prepare features
X, y = prepare_features(df)
X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)
```

### 2. Train Baseline Model
```python
from ml.models.baseline import train_baseline
from ml.evaluation.metrics import evaluate_model

# Train
model = train_baseline(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = evaluate_model(y_test, y_pred, 'Baseline')
print(metrics)
```

### 3. Run Full Experiment
```bash
cd ml/experiments
python run_experiment.py --config configs/baseline.yaml
```

### 4. Compare Models
```python
from ml.evaluation.metrics import compare_models
from ml.evaluation.visualization import plot_comparison

# Run all models
results = run_all_models(X_train, X_test, y_train, y_test)

# Compare
comparison_df = compare_models(results)
plot_comparison(comparison_df)
```

## Workflow

```
Preprocessed Data
    ↓
Split into Train/Val/Test
    ↓
Feature Engineering
    ↓
Train Multiple Models
    ↓
Hyperparameter Tuning (on validation set)
    ↓
Final Evaluation (on test set)
    ↓
Comparison & Analysis
```

## Model Pipeline

Each model follows this pattern:

```python
# Train
model = Model(params)
model.fit(X_train, y_train)

# Validate
y_val_pred = model.predict(X_val)
val_metrics = evaluate(y_val, y_val_pred)

# Tune hyperparameters based on validation

# Final test
y_test_pred = model.predict(X_test)
test_metrics = evaluate(y_test, y_test_pred)

# Save
joblib.dump(model, f'results/models/{model_name}.pkl')
```

## Experiment Configs

Example configuration:

```yaml
experiment_name: "Audio + Lyrics Full"
random_seed: 42

data:
  features: ["audio", "lyrics", "metadata"]
  target: "valence"
  
preprocessing:
  text:
    method: "tfidf"
    max_features: 1000
  audio:
    scaling: "standard"
    
models:
  - name: "Ridge"
    type: "linear"
    params:
      alpha: 1.0
      
  - name: "RandomForest"
    type: "tree"
    params:
      n_estimators: 200
      max_depth: 15
      random_state: 42
```

## Evaluation Metrics

### Primary Metrics
- **RMSE**: Root Mean Squared Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)

### Secondary Metrics
- **MAE**: Mean Absolute Error
- **Explained Variance**: Proportion of variance explained

### Visualization
- Predicted vs Actual scatter plots
- Feature importance bar charts
- Error distribution histograms
- Learning curves (train vs validation)

## Best Practices

1. **Set Random Seeds**: Ensure reproducibility
   ```python
   random_state = 42
   np.random.seed(random_state)
   ```

2. **Save Everything**: Models, scalers, vectorizers
   ```python
   joblib.dump(model, 'model.pkl')
   joblib.dump(scaler, 'scaler.pkl')
   ```

3. **Track Experiments**: Log all hyperparameters and results
   ```python
   results = {
       'model': 'RandomForest',
       'params': model.get_params(),
       'rmse': rmse,
       'r2': r2,
       'timestamp': datetime.now()
   }
   ```

4. **Cross-Validate**: Use k-fold CV for robust estimates
   ```python
   scores = cross_val_score(model, X_train, y_train, cv=5)
   ```

5. **Touch Test Set Once**: Avoid overfitting to test data
   - Use validation set for all tuning
   - Evaluate on test set only at the end

## Expected Performance

Based on similar research:

| Model | Expected RMSE | Expected R² |
|-------|---------------|-------------|
| Mean Baseline | ~0.25 | 0.00 |
| Linear | 0.18-0.22 | 0.15-0.30 |
| Ridge/Lasso | 0.17-0.21 | 0.20-0.35 |
| Random Forest | 0.15-0.19 | 0.35-0.50 |
| XGBoost | 0.14-0.18 | 0.40-0.55 |

## Next Steps

1. ⏳ Implement preprocessing modules
2. ⏳ Create baseline models
3. ⏳ Implement evaluation framework
4. ⏳ Train advanced models
5. ⏳ Hyperparameter tuning
6. ⏳ Final comparison and analysis

## References

See [memory-bank/ML_ROADMAP.md](../memory-bank/ML_ROADMAP.md) for detailed implementation guide.
