# Critical Corrections and Non-Negotiables

## Methodology Rules
1. Use artist-aware splits only (group by `artist_id`).
2. Keep test set untouched until final evaluation.
3. Cache expensive artifacts (embeddings/features/checkpoints); do not recompute unnecessarily.
4. Compare new experiments against a fixed baseline, not moving baselines.
5. Name result files by the actual evaluated split (`val` vs `test`). Do not store validation results in `*_test` files for thesis reporting.
6. Use validation results for model/architecture selection; use test results only for final thesis numbers.

## Deep Learning Rules
1. Set deterministic seeds for every run.
2. Save best checkpoint by the declared selection metric and record epoch. Prefer validation R2 for thesis comparison when R2 is the primary reported metric.
3. Use early stopping.
4. Log per-target R2, RMSE, MAE for train/val/test.

## Known Pitfalls Already Fixed
1. Output activation mismatch with transformed targets.
2. Non-deterministic training due to missing seed setup.
3. Multi-target imbalance (target scaling/weighting issue).
4. PyTorch checkpoint loading compatibility (`weights_only` behavior).
5. **NPZ SPLIT INDEX BUG (May 2026):** `data_splitting.py` used `val_idx`/`test_idx` (local to `df_temp`) directly on the full NPZ `features` array. 70% of val/test audio embedding rows were actually TRAINING data from different songs. This caused every audio-embedding model to produce R²<0 on val/test. Fixed by converting to global indices: `val_idx_global = temp_idx[val_idx]`. Regenerated all 4 audio NPY splits. Verified: Ridge(VGGish→energy) went from R²=-0.77 to R²=+0.80 on val.
6. **ULTIMATE RESULTS CORRECTION (May 2026):** `results/metrics/ultimate_test/ultimate_results_20260516_184227.csv` is invalid/debugging-only because it was affected by the NPZ split-index bug. The corrected Ultimate ML baseline is `results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`: XGBoost R2 = Valence 0.6728, Energy 0.9073, Danceability 0.7693, Popularity 0.1478.
7. **SPLIT NAMING PITFALL:** Some ML scripts historically wrote to `*_test` folders while loading `X_val_*` and `y_val_*`. Treat those as validation/selection results unless the script explicitly loads `test` split data.

## Operating Constraint
- Do not execute project Python scripts from the agent.
- Provide commands for user execution when runs are needed.
