# Plan 2: Focused Hyperparameter Optimization (30 Trials Each)

## Objective

The validation baseline comparison showed clear winners per target, but the margins are small enough that tuning could shift the ranking. Run a focused 30-trial Optuna HPO on the three strongest models to see whether hyperparameter optimization changes the thesis conclusions.

## Recent Validation Baseline

| Target | CatBoost | AttentionDL | XGBoost |
|---|---:|---:|---:|
| Valence | 0.7131 | **0.7178** | 0.6728 |
| Energy | **0.9224** | 0.9101 | 0.9073 |
| Danceability | **0.8027** | 0.7897 | 0.7693 |
| Popularity | **0.1487** | 0.1075 | 0.1478 |
| Average | **0.6467** | 0.6313 | 0.6243 |

The key question for HPO: **Can tuning change the ranking?** Specifically:
- Can XGBoost close the gap with CatBoost (Energy/Danceability)?
- Can DL break away from CatBoost on Valence?
- Can CatBoost push Energy past 0.93?

## Methodology

1. **No test set allowed.** Train on `train`, evaluate on `val`. Test set stays untouched.
2. **Metric:** Maximise validation R² (per-target for ML, average for DL).
3. **Budget:** 30 trials per model class (90 total). This is enough to see whether HPO matters, without 4 days of compute.

---

## Stage 1: CatBoost HPO (30 trials)

**Model:** `catboost.CatBoostRegressor`

**Search space:**
- `learning_rate`: Log-uniform [0.005, 0.2]
- `depth`: Int [4, 10]
- `l2_leaf_reg`: Log-uniform [1e-3, 10.0]
- `subsample`: Uniform [0.6, 1.0]
- `colsample_bylevel`: Uniform [0.6, 1.0]

**Per-target:** Run 4 separate studies (one per target) since CatBoost is trained per-target.

**Output:** `results/hpo/catboost_best_params.json`

---

## Stage 2: XGBoost HPO (30 trials)

**Model:** `xgboost.XGBRegressor`

**Search space:**
- `learning_rate`: Log-uniform [0.005, 0.2]
- `max_depth`: Int [4, 10]
- `reg_lambda`: Log-uniform [1e-3, 10.0]
- `subsample`: Uniform [0.6, 1.0]
- `colsample_bytree`: Uniform [0.6, 1.0]

**Per-target:** Run 4 separate studies (one per target).

**Output:** `results/hpo/xgboost_best_params.json`

---

## Stage 3: DL HPO (30 trials)

**Model:** `AttentionTaskGatedFusionMLP` (from `dl/utils/thesis_models.py`)

**Objective:** Maximise **average validation R²** across all 4 targets (multi-task).

**Search space:**
- `lr`: Log-uniform [5e-5, 1e-3]
- `weight_decay`: Log-uniform [1e-4, 1e-1]
- `dropout_enc`: Uniform [0.1, 0.5]
- `dropout_fusion`: Uniform [0.2, 0.6]
- `batch_size`: Categorical [256, 512]

**Pruning:** Use Optuna MedianPruner. Kill trials if avg val R² < 0.55 at epoch 20.

**Output:** `results/hpo/dl_best_params.json`

---

## Stage 4: Final Evaluation (One Shot)

After HPO, run exactly one final test evaluation:

```bash
python ml/models/thesis_ml_models.py --eval-split test
python dl/14_thesis_architecture_comparison.py --eval-split test --checkpoint-dir models/checkpoints/thesis
```

Build the final thesis table from test results.

---

## Thesis Integration

If HPO does not change the ranking (expected outcome):
> "Hyperparameter optimization confirmed the baseline ranking: CatBoost remains the strongest model across Energy, Danceability, and Popularity. DL's marginal advantage on Valence persisted after tuning. We conclude that CatBoost's ordered boosting is the most effective paradigm for this multimodal music prediction task."

If HPO does change the ranking:
> "After hyperparameter optimization, [X model] improved by [Y R²] on [target], shifting the winner. This highlights the importance of tuning in comparative ML research."
