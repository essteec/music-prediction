# Final Evaluated Models (June 2026)

This document lists the paths and performance metrics for the best-performing models from the final thesis evaluation phase. All models listed here were trained on the combined `train+val` split (417,059 samples) and evaluated once on the `test` split (74,573 samples).

## 🏆 Project Champion: CatBoost (Tuned)

CatBoost with per-target hyperparameter optimization is the strongest model overall.

- **Average R² (Test):** 0.6413
- **Result Artifact:** `results/metrics/thesis_ml_test/thesis_ml_results_test_20260601_160912.csv`

### Model Checkpoints
| Target | Path | R² (Test) |
|---|---|---|
| **Valence** | `ml/models/saved/thesis_ml_test/CatBoost_tuned_valence.pkl` | 0.7220 |
| **Energy** | `ml/models/saved/thesis_ml_test/CatBoost_tuned_energy.pkl` | 0.9212 |
| **Danceability** | `ml/models/saved/thesis_ml_test/CatBoost_tuned_danceability.pkl` | 0.7903 |
| **Popularity** | `ml/models/saved/thesis_ml_test/CatBoost_tuned_popularity.pkl` | 0.1316 |

---

## 🧠 Deep Learning Champion: AttentionTaskGatedFusionMLP (Tuned)

The best-performing neural architecture, utilizing cross-modal attention and per-target gating.

- **Average R² (Test):** 0.6264
- **Result Artifact:** `results/dl_metrics/final_dl_test_20260602_211313.csv`

### Model Checkpoint
- **Path:** `models/checkpoints/thesis_final_tuned/AttentionTaskGatedFusionMLP_retrained.pt`

### Performance Breakdown
| Target | R² (Test) | RMSE | MAE |
|---|---|---|---|
| **Valence** | 0.7214 | 0.1318 | 0.1012 |
| **Energy** | 0.9050 | 0.0738 | 0.0541 |
| **Danceability** | 0.7700 | 0.0817 | 0.0626 |
| **Popularity** | 0.1092 | 1.4093 | 1.1860 |

---

## 📊 Reference Evaluation Sets

Full comparison tables for the final test runs:

1. **Tuned ML (Final):** `results/metrics/thesis_ml_test/thesis_ml_results_test_20260601_160912.csv`
2. **Untuned ML Baseline:** `results/metrics/thesis_ml_test/thesis_ml_results_test_20260528_230741.csv`
3. **Tuned DL (Final):** `results/dl_metrics/final_dl_test_20260602_211313.csv`
4. **Untuned DL Comparison:** `results/dl_metrics/final_dl_test_20260601_212018.csv`
