# Deep Learning Roadmap - Thesis Consolidation

## Goal

Prepare a clean thesis comparison between the strongest corrected classical ML baselines and a small set of interpretable DL architectures.

The goal is no longer to keep adding experiments until DL beats every target. The corrected Ultimate ML baseline is already very strong, and the thesis should present a defensible comparison rather than an overfit architecture hunt.

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
|---|---:|---|---|
| Valence | 0.7181 | Exp H | DL clearly beats Ultimate XGBoost |
| Energy | 0.9069 | Exp H | Essentially tied with Ultimate XGBoost |
| Danceability | 0.7699 | Exp F | Essentially tied with Ultimate XGBoost |
| Popularity | 0.1133 | Exp C/D | DL trails Ultimate XGBoost |

Best single model for thesis-friendly DL reporting is currently Exp F:

`dl/11_feature_engineering.py`

It is preferable to Exp H for interpretation because Exp H's uncertainty loss downweights difficult targets and hurts Popularity.

---

## Roadmap From Here

### Stage 1: Methodology Cleanup

1. Make ML scripts split-explicit: `--eval-split val|test`.
2. Make output filenames include the actual evaluated split.
3. Keep validation and test results separated.
4. Document that older `*_test` folders may contain validation results if the script loaded `X_val_*`.

### Stage 2: Thesis DL Comparison Script

Create a single DL comparison script analogous to `ml/models/enhanced_models.py`.

Recommended file:

`dl/14_thesis_architecture_comparison.py`

It should compare only 3-5 clear architectures:

| Architecture | Purpose |
|---|---|
| `FlatAllMLP` | Simple neural baseline on concatenated all features |
| `MultiModalFusionMLP` | Tests whether per-modality encoders help |
| `TaskGatedFusionMLP` | Tests target-specific modality weighting |
| `AttentionTaskGatedFusionMLP` | One advanced model with cross-modal interaction |

Optional only if time permits:

| Architecture | Purpose |
|---|---|
| `WideTaskGatedFusionMLP` | Capacity ablation; include only if it stays interpretable |

### Stage 3: Validation Selection

Run the comparison script on validation only.

Output:

`results/dl_metrics/thesis_architecture_comparison_val_<timestamp>.csv`

Selection criteria:

1. Primary: average validation R2.
2. Secondary: per-target R2, especially Valence/Energy/Danceability.
3. Tertiary: simplicity and explainability for thesis writing.

### Stage 4: Final Test Evaluation

After ML and DL candidates are selected and focused HPO completes, run final test evaluation once.

Output examples:

- `results/metrics/final_ml_test_<timestamp>.csv`
- `results/dl_metrics/final_dl_test_<timestamp>.csv`
- `results/metrics/final_thesis_comparison_<timestamp>.csv`

### Stage 5: Thesis Tables And Discussion

Final thesis should include:

1. Corrected Ultimate ML baseline table.
2. DL architecture comparison table.
3. Best ML vs best DL per target.
4. Modality discussion: lyrics/text strongest for Valence, audio strongest for Energy/Danceability, Popularity remains metadata/external-factor dominated.

### Stage 6: Thesis Notebooks And Figures (Complete)

- Thesis notebook refresh implemented: `notebooks/10_*` through `notebooks/15_*`.
- Publication-quality figures generated under `results/figures/thesis/`.
- Reference plan: `implementation_plan_3.md`.

---

## Experiments To Deprioritize

DL checkpoint ensemble and DL+ML ensemble should not be the main path.

Reason:

- Ensembles are harder to explain in a thesis.
- They do not directly answer which architecture/modality helped.
- They risk optimizing leaderboard-style performance instead of producing a clean methodological comparison.

They may be appendix/future-work experiments only.

---

## Non-Negotiables

1. Artist-aware splits only.
2. Validation for model selection.
3. Test only for final reporting.
4. Fixed metric schema: R2, RMSE, MAE per target.
5. Deterministic seeds for DL.
6. Cache checkpoints and results.
7. Do not mix validation and test results in one thesis table.
