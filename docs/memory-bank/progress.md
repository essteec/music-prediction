# Progress: Music Prediction Project

## Current Phase

Thesis consolidation after Phase 4 deep learning experiments.

**Status**: Dataset alignment and audio-embedding ML baselines are corrected. DL experiments A-H are complete. The project should now stop broad experimentation and move toward clean, thesis-ready comparison tables.

---

## Completed Work

### Semester 1 ML Baseline

- Built the original 550K-song thesis pipeline with artist-aware train/val/test splits.
- Trained broad classical ML baselines with engineered audio, text, sentiment, and MiniLM features.
- Strongest original models were gradient boosting methods: CatBoost, XGBoost, LightGBM.
- Published thesis/Kaggle artifacts from the first project stage.

### Phase 0: PyTorch MLP Baseline

- Created deterministic PyTorch training pipeline.
- Added multi-task regression over Valence, Energy, Danceability, and Popularity.
- Fixed major DL issues: output activation mismatch, missing seeds, target imbalance, PyTorch checkpoint loading behavior.
- Established that a simple MLP on tabular features is not enough to beat gradient boosting.

### Phase 1: MPNet Text Embeddings

- Extracted MPNet lyric embeddings for all splits.
- Trained MLP with 798 features: metadata + MPNet.
- Improved over the initial MLP, but text-only improvement was not enough to close the ML gap.
- Decision: skip costly BERT fine-tuning and focus on audio/multimodal fusion.

### Audio Acquisition And Embedding Pipeline

- Built YouTube audio acquisition pipeline.
- Built/tested VGGish, Mel Stats, MERT, and PANNs embedding extraction.
- Produced aligned feature arrays in `ml/features/`:
  - `X_*_vggish.npy`
  - `X_*_mel_stats.npy`
  - `X_*_mert.npy`
  - `X_*_panns.npy`
  - `X_*_mpnet.npy`

### Dataset Alignment And NPZ Bug Fix

- Pruned/cleaned the dataset to aligned rows after successful extractions.
- Updated splitting so CSV metadata and NPZ-derived features stay synchronized.
- Fixed the critical NPZ local/global index bug in `data_splitting.py`.
- Regenerated audio NPY splits after the fix.

### Classical ML On Audio Embeddings

Scripts/results:

- `ml/models/vggish_models.py` -> `results/metrics/vggish_test/vggish_results_20260517_114704.csv`
- `ml/models/mert_models.py` -> `results/metrics/mert_test/mert_results_20260517_114009.csv`
- `ml/models/mel_stats_models.py` -> `results/metrics/mel_stats_test/mel_stats_results_20260517_115825.csv`
- `ml/models/panns_models.py` -> `results/metrics/panns_test/panns_results_20260517_125020.csv`
- `ml/models/ultimate_models.py` -> `results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`

Important correction:

- `ultimate_results_20260516_184227.csv` was the broken run caused by the NPZ split-index bug.
- `ultimate_results_20260517_144042.csv` is the corrected result and is the valid Ultimate ML baseline.

Corrected Ultimate XGBoost baseline:

| Target | R2 |
|---|---:|
| Valence | 0.6728 |
| Energy | 0.9073 |
| Danceability | 0.7693 |
| Popularity | 0.1478 |
| Average | 0.6243 |

This result is one of the strongest baselines in the project. It is the reason the later DL experiments had a high bar.

### Phase 4 DL Experiments A-H

Completed scripts:

- `dl/06_multimodal_fusion.py` - Exp A, multi-branch fusion.
- `dl/07_gated_multimodal.py` - Exp B, global gated fusion.
- `dl/08_task_gated_multimodal.py` - Exp C, per-target gated fusion.
- `dl/09_training_recipe.py` - Exp D, scheduler/scaling recipe.
- `dl/10_wider_deeper.py` - Exp E, wider/deeper model.
- `dl/11_feature_engineering.py` - Exp F, engineered metadata + R2 checkpointing.
- `dl/12_cross_modal_attention.py` - Exp G, cross-modal attention.
- `dl/13_loss_tuning.py` - Exp H, Huber + uncertainty weighting.

Best observed DL results across A-H (compared to XGBoost baseline):

| Target | Best DL R2 | Source | Ultimate XGBoost R2 | Status |
|---|---:|---:|---|---|
| Valence | 0.7181 | Exp H | 0.6728 | DL wins |
| Energy | 0.9069 | Exp H | 0.9073 | Tie / tiny ML lead |
| Danceability | 0.7699 | Exp F | 0.7693 | Tie / tiny DL lead |
| Popularity | 0.1133 | Exp C/D | 0.1478 | ML wins |

Note: these compare DL against XGBoost (0.624 avg). The thesis_ml_models.py run later showed CatBoost achieves 0.647 avg, which changes the ranking significantly (see "Thesis Validation Runs" section below).

Best single DL model for average performance:

`results/dl_metrics/exp_f_feat_eng_20260517_231126.csv`

| Target | R2 |
|---|---:|
| Valence | 0.7176 |
| Energy | 0.9067 |
| Danceability | 0.7699 |
| Popularity | 0.0645 |
| Average | 0.6147 |

### Thesis Validation Runs

Both `ml/models/thesis_ml_models.py` and `dl/14_thesis_architecture_comparison.py` completed on validation.

Results at:
- `results/metrics/thesis_ml_val/thesis_ml_results_val_20260520_011851.csv`
- `results/dl_metrics/thesis_val/thesis_architecture_comparison_val_20260525_175338.csv`

**Critical finding: CatBoost (avg 0.647) beat the best DL model (AttentionTaskGatedFusionMLP, avg 0.631) on 3/4 targets.**

| Target | CatBoost R² | Best DL R² | Winner |
|---|---:|---:|---|
| Valence | 0.7131 | 0.7178 | Tie (DL +0.005) |
| Energy | **0.9224** | 0.9101 | CatBoost |
| Danceability | **0.8027** | 0.7897 | CatBoost |
| Popularity | **0.1487** | 0.1075 | CatBoost |

This changes the thesis narrative. The earlier "DL wins Valence" claim was based on comparing against XGBoost (0.673). Against CatBoost (0.713), DL only ties. CatBoost's ordered boosting handles the mixed tabular+embedding 4254-feature space better than either XGBoost or multimodal DL.

### Hyperparameter Optimization Completed

Focused HPO completed on validation split:

- CatBoost: 12 trials per target via `ml/models/hpo_catboost.py` (Optuna + CatBoostPruningCallback).
- AttentionTaskGatedFusionMLP: 20 trials via `dl/15_hpo_attention_dl.py` (Optuna + MedianPruner).

Results saved to `results/hpo/`:

- `results/hpo/catboost_best_params.json` — per-target tuned params
- `results/hpo/attention_dl_best_params.json` — tuned params (avg val R² = 0.633)

### Final Test Evaluation Completed (June 2026)

All models trained on `train+val` (417,059 samples), evaluated once on `test` (74,573 samples).

4 runs executed:

1. All 8 ML families (default) → `results/metrics/thesis_ml_test/thesis_ml_results_test_20260528_230741.csv`
2. CatBoost (tuned) → `results/metrics/thesis_ml_test/thesis_ml_results_test_20260601_160912.csv`
3. All 5 DL architectures (default, 15 epochs) → `results/dl_metrics/final_dl_test_20260601_212018.csv`
4. AttentionDL (tuned, 11 epochs) → `results/dl_metrics/final_dl_test_20260601_224941.csv`

Final head-to-head (test):

| Target | CatBoost_tuned R² | AttentionDL_tuned R² | Winner |
|---|---:|---:|---|
| Valence | 0.7220 | 0.7214 | Tie |
| Energy | **0.9212** | 0.9050 | CatBoost |
| Danceability | **0.7903** | 0.7700 | CatBoost |
| Popularity | **0.1316** | 0.1092 | CatBoost |
| **Average** | **0.6413** | **0.6264** | **CatBoost** |

HPO effect: CatBoost +0.004 avg, AttentionDL +0.005 avg. Ranking unchanged.

Bug fix applied: `thesis_ml_models.py` — disabled `use_best_model` and removed `od_type`/`od_wait` for CatBoost retrain on `train+val` (no eval_set available).

### Thesis Infrastructure (DL)

- `ml/models/thesis_ml_models.py`: split-explicit ML baseline on the 4254-feature DL-equivalent input set. Trains 8 model families (Mean, Ridge, XGBoost, LightGBM, CatBoost, MLPRegressor, ExtraTrees, RandomForest) with checkpointing, timing, and tuned-params support.
- `dl/utils/thesis_models.py`: centralized source of truth for thesis DL architectures. Provides `FlatAllMLP`, re-exports `MultiModalFusionMLP`, `TaskGatedFusionMLP`, `AttentionTaskGatedFusionMLP`, and the `engineer_metadata()` helper for the feat eng variant.
- `dl/14_thesis_architecture_comparison.py`: clean DL comparison pipeline. Supports `--eval-split val` (train + select by val R²) and `--eval-split test --retrain` (retrain on train+val + evaluate on test). Supports `--tuned-params` for architecture-specific HPO params. Runs 5 architectures with consistent hyperparams.

### Thesis Notebook Refresh And Figures (Complete)

- New thesis notebooks created and executed: `notebooks/10_*` through `notebooks/15_*`.
- Publication-quality figures generated under `results/figures/thesis/`.
- Plan reference: `implementation_plan_3.md`.

---

## Methodology Caveat

Some ML result folders use names like `ultimate_test`, but the scripts historically loaded `X_val_*` and `y_val_*`. These should be treated as validation/selection results unless the script is made split-explicit and rerun on `test`.

Going forward:

1. Validation split is for model/architecture selection.
2. Test split is for final thesis reporting only.
3. Result filenames must include the evaluated split explicitly.
4. Final tables should not mix validation and test numbers.

---

## Current Next Step

Final test evaluation is **COMPLETE**. The project is now in the thesis-writing phase.

Next actions:

1. Write the thesis comparison chapter using the final test tables.
2. Generate publication-quality figures (bar charts, radar plots, architecture diagrams).
3. Optional: statistical significance tests (paired bootstrap) on per-sample predictions.
