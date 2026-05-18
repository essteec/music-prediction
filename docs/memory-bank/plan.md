# Plan: Thesis-Ready Result Consolidation

## Goal

Create a clean, reproducible comparison between corrected Ultimate ML results and a compact set of DL architectures for thesis reporting.

The old goal of only aligning Kaggle artifacts is complete. The current priority is methodology clarity: validation for selection, test for final reporting, and no confusing result naming.

---

## Completed Alignment Work

- `songs.csv` and audio embedding NPZ files were aligned to the same row order.
- `data_splitting.py` was fixed to use global indices for validation/test NPZ slicing.
- Feature arrays in `ml/features/` were regenerated after the fix.
- Audio model scripts and `ultimate_models.py` were used to validate the fixed pipeline.

---

## Corrected ML Result Interpretation

Bad debugging result:

`results/metrics/ultimate_test/ultimate_results_20260516_184227.csv`

Reason: affected by NPZ local/global split-index bug.

Valid corrected result:

`results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`

Corrected Ultimate XGBoost R2:

| Target | R2 |
|---|---:|
| Valence | 0.6728 |
| Energy | 0.9073 |
| Danceability | 0.7693 |
| Popularity | 0.1478 |

---

## Remaining Methodology Work

1. Re-check `ml/models/ultimate_models.py` and ensure it is syntactically valid.
2. Make `ultimate_models.py` explicit about evaluated split: `--eval-split val|test`.
3. Ensure output path includes split name, for example:
   - `results/metrics/ultimate_val/ultimate_results_val_<timestamp>.csv`
   - `results/metrics/ultimate_test/ultimate_results_test_<timestamp>.csv`
4. Treat older `*_test` folders as potentially validation results unless the script loaded `X_test_*`.
5. Build DL comparison results using the same split discipline.

---

## Thesis DL Comparison Plan

Create one comparison script analogous to `ml/models/enhanced_models.py`, but much smaller and cleaner.

Recommended script:

`dl/14_thesis_architecture_comparison.py`

Architectures:

1. `FlatAllMLP`
2. `MultiModalFusionMLP`
3. `TaskGatedFusionMLP`
4. `AttentionTaskGatedFusionMLP`

Optional only if needed:

5. `WideTaskGatedFusionMLP`

Outputs:

- `results/dl_metrics/thesis_architecture_comparison_val_<timestamp>.csv`
- `models/checkpoints/thesis_<architecture>_best.pt`

Final test output after selection:

- `results/dl_metrics/final_dl_test_<timestamp>.csv`

---

## Verification Plan

1. Confirm every result row includes `split`.
2. Confirm validation and test files are separate.
3. Confirm all models use the same train/val/test data artifacts.
4. Confirm per-target R2/RMSE/MAE are reported.
5. Confirm final thesis table compares selected models only, not every exploratory A-H run.
