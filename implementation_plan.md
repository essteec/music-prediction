# Plan: Thesis-Ready ML vs DL Comparison

## Current Situation

The recent results changed the project direction. The corrected Ultimate ML baseline is much stronger than the earlier broken run, and the DL experiments A-H already show the key scientific result:

- DL clearly improves Valence.
- DL ties Energy and Danceability against the corrected Ultimate XGBoost baseline.
- DL still trails ML on Popularity.

The next step is not more architecture chasing. The next step is to consolidate results into a clean, comparable thesis workflow.

---

## Corrected Baseline To Compare Against

Valid corrected Ultimate ML result:

`results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`

| Target | Ultimate XGBoost R2 |
|---|---:|
| Valence | 0.6728 |
| Energy | 0.9073 |
| Danceability | 0.7693 |
| Popularity | 0.1478 |
| Average | 0.6243 |

Invalid debugging result:

`results/metrics/ultimate_test/ultimate_results_20260516_184227.csv`

Reason: affected by the NPZ split-index bug. Do not use it as a true result.

---

## Best DL Results So Far

| Target | Best DL R2 | Source | vs Ultimate XGBoost |
|---|---:|---|---:|
| Valence | 0.7181 | Exp H | +0.0453 |
| Energy | 0.9069 | Exp H | -0.0004 |
| Danceability | 0.7699 | Exp F | +0.0006 |
| Popularity | 0.1133 | Exp C/D | -0.0345 |

Best single DL model for thesis-friendly reporting is currently Exp F:

`results/dl_metrics/exp_f_feat_eng_20260517_231126.csv`

| Target | Exp F R2 |
|---|---:|
| Valence | 0.7176 |
| Energy | 0.9067 |
| Danceability | 0.7699 |
| Popularity | 0.0645 |
| Average | 0.6147 |

Interpretation:

- Exp F is strong and explainable: task-gated multimodal fusion plus small metadata interactions and R2-based checkpointing.
- Exp H is useful diagnostically, but not the preferred thesis model because uncertainty weighting hurts Popularity and Danceability.
- Popularity should be discussed honestly as a target where metadata/external effects dominate and tree models remain stronger.

---

## Methodology Rules From Here

1. Use validation split for architecture/model selection.
2. Use test split only for final thesis reporting.
3. Every result row must include `split`.
4. Result filenames must include the evaluated split: `*_val_*.csv` or `*_test_*.csv`.
5. Do not mix validation and test numbers in the same final thesis table.
6. Keep metrics consistent: R2, RMSE, MAE per target.
7. Keep architecture count small enough to explain in the thesis.

---

## Stage 1: Clean ML Baseline Evaluation

### Status

`ml/models/ultimate_models.py` is now split-explicit:

```bash
python ml/models/ultimate_models.py --eval-split val
python ml/models/ultimate_models.py --eval-split test
```

It writes to:

```text
results/metrics/ultimate_val/ultimate_results_val_<timestamp>.csv
results/metrics/ultimate_test/ultimate_results_test_<timestamp>.csv
```

### Next Action

Run validation first if you want a clean replacement for the historically misnamed `ultimate_test` validation run:

```bash
python ml/models/ultimate_models.py --eval-split val
```

Only run test when final models are selected:

```bash
python ml/models/ultimate_models.py --eval-split test
```

---

## Stage 2: DL Version Of Enhanced Models

Create a single comparison script equivalent in spirit to `ml/models/enhanced_models.py`, but smaller and thesis-focused.

Recommended file:

`dl/14_thesis_architecture_comparison.py`

### Architectures To Compare

| Architecture | Purpose | Complexity |
|---|---|---|
| `FlatAllMLP` | Baseline neural model on all concatenated features | Low |
| `MultiModalFusionMLP` | Tests whether modality-specific encoders help | Medium |
| `TaskGatedFusionMLP` | Tests target-specific modality weighting | Medium |
| `AttentionTaskGatedFusionMLP` | One advanced architecture with cross-modal interaction | High |

Optional only if needed:

| Architecture | Include If |
|---|---|
| `WideTaskGatedFusionMLP` | You need a capacity ablation and can explain it clearly |

Do not include every A-H experiment in the thesis comparison table. A-H can be summarized as development history, but the main table should use the clean architecture set above.

### Required Script Behavior

The script should support:

```bash
python dl/14_thesis_architecture_comparison.py --eval-split val
python dl/14_thesis_architecture_comparison.py --eval-split test --checkpoint-dir models/checkpoints/thesis
```

Validation mode should:

1. Train each architecture on train split.
2. Select best checkpoint by average validation R2.
3. Save model checkpoints.
4. Write one comparison CSV.

Test mode should:

1. Load selected checkpoints.
2. Evaluate on test split.
3. Write final test CSV.
4. Not change architecture/hyperparameter choices.

### Output Files

Validation:

```text
results/dl_metrics/thesis_architecture_comparison_val_<timestamp>.csv
models/checkpoints/thesis/<architecture>_best.pt
```

Final test:

```text
results/dl_metrics/final_dl_test_<timestamp>.csv
```

---

## Stage 3: Minimal New Code Needed

### New File: `dl/utils/thesis_models.py`

Purpose: collect the final thesis architectures in one place.

Should include:

1. `FlatAllMLP`
2. imports/re-exports for existing `MultiModalFusionMLP`
3. imports/re-exports for existing `TaskGatedFusionMLP`
4. imports/re-exports for existing `AttentionTaskGatedFusionMLP`

Rationale:

- Existing architecture code is spread across experiment files.
- A thesis comparison script should not depend on old A-H scripts directly.
- This creates a clean source of truth for thesis architectures.

### New File: `dl/14_thesis_architecture_comparison.py`

Purpose: train/evaluate the small architecture set with consistent logging.

Should reuse:

- `dl/utils/data_loaders.py`
- `dl/utils/metrics.py`
- `dl/utils/fusion.py`
- `dl/utils/fusion_attention.py`

Required CSV columns:

```text
timestamp,split,experiment,model,target,r2,rmse,mae,epoch,selection_metric,notes
```

---

## Stage 4: Final Thesis Comparison Table

Once final ML and DL test results are available, produce one table like this:

| Target | Best ML Model | ML R2 | Best DL Model | DL R2 | Winner | Interpretation |
|---|---|---:|---|---:|---|---|
| Valence | Ultimate XGBoost | TBD | Task/Attention Fusion | TBD | TBD | Lyrics + multimodal DL helps emotion |
| Energy | Ultimate XGBoost | TBD | Task/Attention Fusion | TBD | TBD | Audio features dominate |
| Danceability | Ultimate XGBoost | TBD | Task/Attention Fusion | TBD | TBD | Audio/rhythm features dominate |
| Popularity | Ultimate XGBoost | TBD | Best DL | TBD | likely ML | External/contextual factors dominate |

This is the table the thesis should optimize for, not a giant dump of every experiment.

---

## What Not To Do Now

### Do Not Make DL Ensemble The Main Path

Reason:

- Harder to explain.
- Weak architecture insight.
- Easy to look like leaderboard optimization rather than research methodology.

### Do Not Make DL+ML Ensemble The Main Path

Reason:

- It may improve headline scores, but it blurs the ML vs DL comparison.
- It is better as an appendix/future-work result after the clean thesis comparison exists.

### Do Not Keep Adding Slight Variants

Reason:

- The architecture story is already clear.
- More variants will make the thesis harder to write and less convincing.

---

## Recommended Thesis Story

Use this narrative:

1. Start with strong classical ML baselines over engineered and learned features.
2. Show that simple DL baselines are weaker on tabular data.
3. Add multimodal DL fusion to handle high-dimensional text/audio embeddings better.
4. Show DL substantially improves Valence and matches Energy/Danceability.
5. Explain why Popularity remains difficult and why ML still wins there.

This is more honest and stronger than claiming DL universally beats ML.

---

## Immediate Commands For User Execution

Clean validation Ultimate ML baseline:

```bash
python ml/models/ultimate_models.py --eval-split val
```

After final model selection only:

```bash
python ml/models/ultimate_models.py --eval-split test
python dl/14_thesis_architecture_comparison.py --eval-split test
```

Do not run final test until the thesis architecture comparison is finalized.
