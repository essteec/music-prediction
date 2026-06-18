# Deep Learning Roadmap

## Goal

Prepare a clean comparison between the strongest corrected classical ML baselines and a small set of interpretable DL architectures.

The corrected Ultimate ML baseline is very strong. The project should present a defensible comparison rather than an overfit architecture hunt.

---

## Corrected Baseline

Valid corrected Ultimate ML result:

`results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`

| Target | Ultimate XGBoost R2 |
|---|---:|
| Valence | 0.6728 |
| Energy | 0.9073 |
| Danceability | 0.7693 |
| Popularity | 0.1478 |
| Average | 0.6243 |

Invalid debugging artifact:

`results/metrics/ultimate_test/ultimate_results_20260516_184227.csv`

That run was affected by the NPZ split-index bug and should not be used as a true baseline.

---

## DL Result Reality After A-H

Best observed DL by target:

| Target | Best DL R2 | Source | Interpretation |
|---|---:|---:|---|---|
| Valence | 0.7181 | Exp H | DL clearly beats Ultimate XGBoost |
| Energy | 0.9069 | Exp H | Essentially tied with Ultimate XGBoost |
| Danceability | 0.7699 | Exp F | Essentially tied with Ultimate XGBoost |
| Popularity | 0.1133 | Exp C/D | DL trails Ultimate XGBoost |

---

## Completed Stages

### Stage 1: Methodology Cleanup

1. ML scripts are split-explicit: `--eval-split val|test`.
2. Output filenames include the actual evaluated split.
3. Validation and test results are separated.
4. Older `*_test` folders treated as potentially validation results.

### Stage 2: DL Comparison Script

Completed: `dl/14_thesis_architecture_comparison.py`

Architectures compared:

- `FlatAllMLP` — Simple neural baseline on concatenated all features
- `MultiModalFusionMLP` — Per-modality encoders with fusion
- `TaskGatedFusionMLP` — Target-specific modality weighting
- `AttentionTaskGatedFusionMLP` — Cross-modal interaction
- `TaskGatedFusionMLP_FeatEng` — Best-optimized feat eng variant

### Stage 3: Validation Selection

Completed: `results/dl_metrics/thesis_val/` with per-architecture val R² scores.

Selection criteria:
1. Primary: average validation R².
2. Secondary: per-target R², especially Valence/Energy/Danceability.

### Stage 4: Final Test Evaluation

Completed:

- `results/metrics/thesis_ml_test/` — CatBoost_tuned result
- `results/dl_metrics/final_dl_test_<timestamp>.csv` — DL result

### Stage 6: Comprehensive Notebooks And Figures (Complete)

- Comprehensive notebooks implemented: `notebooks/20_*` through `notebooks/27_*`
- Publication-quality figures generated under `results/figures/thesis/`

---

## Experiments To Deprioritize

DL checkpoint ensemble and DL+ML ensemble are not the main path.

Reason:

- Ensembles are harder to explain.
- They do not directly answer which architecture/modality helped.
- They risk optimizing leaderboard-style performance.

They may be considered only if needed.

---

## Non-Negotiables

1. Artist-aware splits only.
2. Validation for model selection.
3. Test only for final reporting.
4. Fixed metric schema: R2, RMSE, MAE per target.
5. Deterministic seeds for DL.
6. Cache checkpoints and results.
7. Do not mix validation and test results in one table.
