# Plan 2: Focused Finalist Hyperparameter Optimization

## Objective

Use validation-only hyperparameter optimization as a robustness check before final thesis evaluation.

The goal is not to keep searching until one model wins. The goal is to answer a focused thesis question:

> After selecting the strongest baseline models, does reasonable hyperparameter optimization change the ML vs DL conclusion?

Current validation results show CatBoost as the strongest overall ML model and `AttentionTaskGatedFusionMLP` as the strongest DL model. Therefore HPO should focus on these finalists, not every possible model.

---

## Current Validation Baseline

| Target | CatBoost | AttentionDL | XGBoost |
|---|---:|---:|---:|
| Valence | 0.7131 | **0.7178** | 0.6728 |
| Energy | **0.9224** | 0.9101 | 0.9073 |
| Danceability | **0.8027** | 0.7897 | 0.7693 |
| Popularity | **0.1487** | 0.1075 | 0.1478 |
| Average | **0.6467** | 0.6313 | 0.6243 |

Interpretation:

- CatBoost is the strongest model overall.
- DL only slightly leads CatBoost on Valence (+0.005), which should be treated as a near-tie unless confirmed after tuning/final test.
- XGBoost is no longer the main ML finalist because CatBoost dominates it on validation average R².

---

## Why Not Tune Everything?

The earlier 30-trial plan was too broad.

It said 30 trials per model class, but CatBoost and XGBoost are trained separately per target. That means:

- CatBoost: 30 trials x 4 targets = 120 trials
- XGBoost: 30 trials x 4 targets = 120 trials
- DL: 30 trials
- Total = 270 trials

That is too expensive for the current stage and risks turning the thesis into leaderboard chasing.

Best practice here is finalist-only HPO:

1. Tune the strongest ML model: CatBoost.
2. Tune the strongest DL model: `AttentionTaskGatedFusionMLP`.
3. Skip XGBoost unless there is extra time and it is clearly marked as appendix/secondary analysis.

---

## Methodology Rules

1. Use validation split only for HPO.
2. Do not touch test during HPO.
3. Keep the search budget small and declared in the thesis.
4. Select final hyperparameters by validation R².
5. Use test exactly once for final thesis reporting.
6. For final test, ML and DL should both train on `train+val` before evaluating on `test`.

---

## Stage 1: CatBoost HPO

**Model:** `catboost.CatBoostRegressor`

**Why tune it:** CatBoost is the strongest validation model overall and wins Energy, Danceability, and Popularity.

**Budget:** 10-15 trials per target.

Recommended default: 12 trials per target = 48 total CatBoost trials.

**Metric:** Per-target validation R².

**Search space (current optimized script):**

- Fixed:
- `iterations=1500` (with early stopping)
- `od_type='Iter'`, `od_wait=300`, `use_best_model=True`
- `border_count=254`, `eval_metric='R2'`
- Tuned:
- `learning_rate`: log-uniform [1e-4, 0.3]
- `depth`: int [6, 10]
- `l2_leaf_reg`: log-uniform [1e-3, 10.0]
- `random_strength`: log-uniform [1e-3, 10.0]
- `bagging_temperature`: uniform [0.0, 1.0]
- `leaf_estimation_iterations`: int [1, 10]

**Pruning:**

- Optuna `CatBoostPruningCallback('R2')` (aligned with study direction `maximize`).

**Output:**

```text
results/hpo/catboost_best_params.json
results/hpo/catboost_hpo_val_<target>_<timestamp>.csv
```

`catboost_best_params.json` should include, per target:

- `best_val_r2`
- `best_iteration_on_val`
- `recommended_iterations_for_retrain` (`best_iteration + 1`)
- `tuned_params`
- `fixed_params`
- `full_params_for_refit`

---

## Stage 2: Attention DL HPO

**Model:** `AttentionTaskGatedFusionMLP`

**Why tune it:** It is the best DL architecture on validation and the only DL model that slightly leads CatBoost on Valence.

**Budget:** 15-20 trials total.

Recommended default: 20 trials with pruning.

**Metric:** Average validation R² across all 4 targets.

**Search space:**

- `lr`: log-uniform [5e-5, 8e-4]
- `weight_decay`: log-uniform [1e-4, 5e-2]
- `dropout_enc`: uniform [0.1, 0.4]
- `dropout_fusion`: uniform [0.2, 0.6]
- `batch_size`: categorical [256, 512]

**Pruning:** Use Optuna MedianPruner or early stop weak trials after a minimum warmup period.

**Output:**

```text
results/hpo/attention_dl_best_params.json
results/hpo/attention_dl_hpo_val_<timestamp>.csv
```

`attention_dl_best_params.json` should include:

- `model_name`
- `best_params`
- `best_value` (avg val R²)
- `best_epoch` (validation-selected training budget)
- `best_val_metrics` (per-target R²/RMSE/MAE)
- training protocol metadata (loss weights, patience, pruning config, scaler flag)

---

## Stage 3: Optional XGBoost HPO

Skip this by default.

Only run XGBoost HPO if CatBoost and DL HPO finish early and there is still time.

Reason:

- XGBoost is not the best ML model anymore.
- Tuning XGBoost mostly answers an internal ML question, not the main ML vs DL thesis question.
- If included, put it in an appendix or secondary analysis, not the main conclusion.

Suggested budget if run:

- 8-10 trials per target.

---

## Stage 4: Final Evaluation

Final evaluation should happen only after HPO is complete and final hyperparameters are chosen.

Important fairness rule:

- `ml/models/thesis_ml_models.py --eval-split test` trains ML models on `train+val` and evaluates on `test`.
- `dl/14_thesis_architecture_comparison.py --eval-split test` currently loads checkpoints trained on `train` only and evaluates on `test`.

Before final DL test reporting, update the DL final-test path so tuned params are applied only to the intended architecture (Attention), while other architectures keep their own defaults. Then retrain on `train+val` for a fixed validation-selected training budget (`best_epoch`) and evaluate once on `test`.

Do not use test for early stopping, checkpoint selection, or hyperparameter decisions.

---

## Final Commands After HPO And DL Final-Test Fix

Run ML final test:

```bash
python ml/models/thesis_ml_models.py --eval-split test --models CatBoost
```

Run DL final test after the script supports train+val final training:

```bash
python dl/14_thesis_architecture_comparison.py --eval-split test --checkpoint-dir models/checkpoints/thesis_final
```

If tuned scripts are created separately, use the tuned-final scripts instead, but keep the same rule: train on `train+val`, evaluate once on `test`.

---

## Thesis Integration

If HPO does not change the ranking:

> Focused hyperparameter optimization confirmed the validation ranking. CatBoost remained the strongest overall model, while the best DL architecture remained competitive mainly for Valence. This suggests that ordered boosting is especially effective for the mixed tabular and embedding-based feature space, while multimodal DL provides its clearest benefit for emotion-related prediction.

If HPO changes the ranking:

> After focused hyperparameter optimization, [model] improved by [delta R²] on [target], changing the winner for [target]. This demonstrates that hyperparameter tuning materially affects comparative conclusions and should be reported as part of the model selection process.

Recommended thesis section title:

```text
Focused Hyperparameter Optimization of Finalist Models
```
