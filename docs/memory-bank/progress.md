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

Best observed DL results across A-H:

| Target | Best DL R2 | Source | Ultimate XGBoost R2 | Status |
|---|---:|---|---:|---|
| Valence | 0.7181 | Exp H | 0.6728 | DL wins |
| Energy | 0.9069 | Exp H | 0.9073 | Tie / tiny ML lead |
| Danceability | 0.7699 | Exp F | 0.7693 | Tie / tiny DL lead |
| Popularity | 0.1133 | Exp C/D | 0.1478 | ML wins |

Best single DL model for average performance:

`results/dl_metrics/exp_f_feat_eng_20260517_231126.csv`

| Target | R2 |
|---|---:|
| Valence | 0.7176 |
| Energy | 0.9067 |
| Danceability | 0.7699 |
| Popularity | 0.0645 |
| Average | 0.6147 |

Key finding:

- Exp F is the strongest single DL model for Valence/Energy/Danceability.
- Exp H is not a clean overall improvement because uncertainty weighting hurts Popularity and Danceability.
- Popularity remains better handled by classical ML, likely because it depends more on artist/contextual metadata and external factors than learned audio/text representations.

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

Build a thesis-ready DL comparison script similar in spirit to `ml/models/enhanced_models.py`, but with a small number of interpretable neural architectures:

1. Flat all-feature MLP.
2. Multi-branch fusion MLP.
3. Task-gated fusion MLP.
4. Attention task-gated fusion MLP.

The goal is not more overkill architecture search. The goal is clean, reproducible, comparable results for the thesis.
