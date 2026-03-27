# Recursive Feature Elimination (RFE) - Complete Documentation

**Date**: January 1, 2026  
**Experiment**: Experiment 2 (414 features with artist data)  
**Status**: Complete - Not included in thesis, preserved for future reference

---

## Overview

RFE was implemented as a feature selection strategy to identify optimal feature subsets for each target variable. The approach uses CatBoost as the authority model to iteratively remove least important features.

---

## Configuration

### Parameters
```python
FEATURES_PER_ITERATION = 10   # Conservative: remove 10 features per step
R2_DROP_THRESHOLD = 0.01      # Stop if R² drops > 1% from BASELINE
MIN_FEATURES = 20             # Safety minimum (don't go below this)
```

### Authority Model: CatBoost_tuned
```python
CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=10,
    l2_leaf_reg=8,
    subsample=0.8,
    bootstrap_type='Bernoulli',
    random_state=42,
    verbose=False,
    early_stopping_rounds=50,
    thread_count=-1,
    grow_policy='Lossguide',
    max_leaves=64
)
```

### Feature Groups
| Group | Count | Indices | Description |
|-------|-------|---------|-------------|
| Audio | 23 | 0-22 | Audio features + artist data |
| Text | 5 | 23-27 | Text statistics |
| Sentiment | 2 | 28-29 | Polarity, subjectivity |
| Embeddings | 384 | 30-413 | MiniLM-L6-v2 semantic vectors |
| **Total** | **414** | | |

---

## Methodology

### Algorithm
```
1. Start with all 414 features
2. Train CatBoost_tuned, record baseline R²
3. Loop:
   a. Get feature importances from model
   b. Remove 10 least important features
   c. Retrain model
   d. Calculate R² drop from BASELINE (not previous iteration!)
   e. If drop > 1%: STOP, restore previous features
   f. Track best R² iteration
4. Use features from best R² iteration (not stopping point)
```

### Critical Bug Fixes Applied

**Bug 1: Feature Restoration (CRITICAL)**
- **Problem**: Original code restored ALL features instead of previous subset
- **Before**: `optimal_features = [f for f in range(X_train.shape[1]) if f not in features_to_remove]`
- **After**: `previous_features = current_features.copy()` then `optimal_features = previous_features`
- **Impact**: Without fix, RFE would have invalid feature sets

**Bug 2: Performance Drift (CRITICAL)**
- **Problem**: Comparing only to previous iteration allows cumulative drift
- **Before**: `if previous_r2 - current_r2 > R2_DROP_THRESHOLD`
- **After**: `if baseline_r2 - current_r2 > R2_DROP_THRESHOLD`
- **Impact**: Ensures total degradation never exceeds 1%, not just per-step

**Bug 3: Insufficient Logging**
- **Problem**: Only tracking one type of R² drop
- **Solution**: Track both:
  - `r2_drop_from_baseline`: Total drop from iteration 0
  - `r2_drop_iteration`: Drop from previous iteration only
- **Impact**: Better debugging and visualization

---

## Results

### Best Iterations per Target
| Target | Best Iteration | Features | R² (Optimal) | R² (Baseline) |
|--------|----------------|----------|--------------|---------------|
| Valence | 23 | 184 | 0.4169 | 0.4149 |
| Energy | 38 | 34 | 0.8485 | 0.8473 |
| Danceability | 34 | 74 | 0.6123 | 0.6095 |
| Popularity | 2 | 394 | 0.1359 | 0.1342 |

### Key Observations

**Valence (184 features)**
- Removed 230 features (55% reduction)
- Mostly embedding dimensions removed
- All audio features retained
- Best R² slightly BETTER than baseline (0.4169 vs 0.4149)

**Energy (34 features!)**
- Aggressive reduction: 380 features removed (92%)
- R² actually IMPROVED: 0.8485 vs 0.8473
- Indicates most features are noise for energy prediction
- Core predictors: loudness, tempo, acousticness

**Danceability (74 features)**
- Moderate reduction: 340 removed (82%)
- Slight improvement: 0.6123 vs 0.6095
- Tempo and rhythm features most important

**Popularity (394 features)**
- Minimal reduction: only 20 features removed (5%)
- Stopped early at iteration 2
- Indicates all feature types contribute to popularity
- R² still low (~0.13) - external factors dominate

---

## Feature Group Analysis

### Features Removed First (Least Important)
- **Embeddings**: Dominated early removals (high-dimensional noise)
- **Sentiment**: Subjectivity often removed before polarity
- **Audio**: Genre flags often removed before core audio features
- **Key/Mode**: Cyclical key encodings often removed early

### Features Retained Longest (Most Important)
- **Loudness**: Critical for energy prediction
- **Tempo**: Critical for danceability
- **Artist features**: Important for popularity
- **Polarity**: Important for valence
- **Word count**: Basic but effective text feature

---

## Scripts

### Main RFE Script
`ml/models/feature_selection_rfe.py`
- Performs RFE for all 4 targets
- Checkpoint/resume system for long runs
- Generates iteration logs and visualizations

### Retraining Script
`ml/models/retrain_rfe_best_iterations.py`
- Loads optimal features from best iterations
- Retrains 6 models per target:
  - XGBoost_tuned
  - CatBoost (default)
  - CatBoost_tuned
  - LightGBM_tuned
  - MLPRegressor (default)
  - MLPRegressor_tuned

---

## Output Files

### Results Directory
`results/metrics/experiment2_with_artist/rfe/`

| File | Description |
|------|-------------|
| `rfe_iterations_20260101_023946.txt` | Master log with all iterations |
| `rfe_iterations_{target}_{timestamp}.csv` | Per-target iteration logs |
| `optimal_features_{target}_{timestamp}.csv` | Final feature lists |
| `checkpoints/rfe_checkpoint.json` | Resume checkpoint |

### Models Directory
`ml/models/saved/experiment2_with_artist/rfe/`
- Intermediate models saved during RFE
- Best iteration models

`ml/models/saved/experiment2_with_artist/rfe_best/`
- Retrained models at optimal iterations

### Visualizations
`results/figures/rfe/`
- 4-panel plots per target:
  - R² vs Feature Count
  - R² Drop per Iteration (dual bars: total + iterative)
  - Feature Group Evolution
  - RMSE Comparison

---

## Code Snippets

### Loading Optimal Features
```python
import pandas as pd

# Load optimal features for valence
optimal_df = pd.read_csv('results/metrics/experiment2_with_artist/rfe/optimal_features_valence_20260101_*.csv')
optimal_indices = optimal_df['feature_index'].tolist()
optimal_names = optimal_df['feature_name'].tolist()

# Apply to data
X_train_reduced = X_train_full[:, optimal_indices]
X_val_reduced = X_val_full[:, optimal_indices]
```

### Analyzing Feature Groups
```python
def analyze_feature_groups(feature_indices):
    FEATURE_GROUPS = {
        'audio': list(range(0, 23)),
        'text': list(range(23, 28)),
        'sentiment': list(range(28, 30)),
        'embeddings': list(range(30, 414))
    }
    return {
        group: sum(1 for i in feature_indices if i in indices)
        for group, indices in FEATURE_GROUPS.items()
    }
```

### Stopping Condition Logic
```python
# CRITICAL: Compare to BASELINE, not previous iteration
r2_drop_from_baseline = baseline_r2 - current_r2

if r2_drop_from_baseline > R2_DROP_THRESHOLD:
    # Stop and restore PREVIOUS features (not all features!)
    optimal_features = previous_features.copy()
    break
```

---

## Lessons Learned

### Why RFE Was Valuable
1. **Feature efficiency**: Energy achieved 92% feature reduction with improved R²
2. **Insight generation**: Revealed which feature groups matter per target
3. **Model simplicity**: Fewer features = faster inference, less overfitting risk

### Why It Wasn't in Thesis
1. **Complexity**: Hard to explain dual experiment (full vs RFE) in limited pages
2. **Marginal gains**: R² improvements were small (0.1-0.2%)
3. **Focus**: Thesis emphasized algorithm comparison, not feature selection

### Future Use
- RFE methodology is scientifically sound and thesis-defense ready
- Can be used for Semester 2 deep learning feature selection
- Could be applied to audio spectrograms or BERT embeddings

---

## Reproducibility

### To Reproduce Results
```bash
cd ml/models
python feature_selection_rfe.py
# Takes ~1-2 days for all 4 targets

# Then retrain at best iterations
python retrain_rfe_best_iterations.py
```

### Environment
- Python 3.10+
- CatBoost >= 1.2.0
- XGBoost >= 2.0.0
- LightGBM >= 4.0.0
- scikit-learn >= 1.3.0

---

## Summary Table

| Metric | Valence | Energy | Danceability | Popularity |
|--------|---------|--------|--------------|------------|
| Best Iteration | 23 | 38 | 34 | 2 |
| Original Features | 414 | 414 | 414 | 414 |
| Optimal Features | 184 | 34 | 74 | 394 |
| Reduction % | 55% | 92% | 82% | 5% |
| Baseline R² | 0.4149 | 0.8473 | 0.6095 | 0.1342 |
| Optimal R² | 0.4169 | 0.8485 | 0.6123 | 0.1359 |
| R² Change | +0.0020 | +0.0012 | +0.0028 | +0.0017 |
