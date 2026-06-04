# Music Attribute Prediction

This project predicts four music attributes from multimodal features:
valence, energy, danceability, and popularity.

## Status

- Final test evaluation complete (train+val retrain, single test run)
- Thesis notebooks and publication-quality figures complete
- Current focus: thesis writing and packaging

## Final Test Results (June 2026)

All models trained on train+val (417,059 samples) and evaluated once on test
(74,573 samples), using the aligned 4,254-feature multimodal representation.

Best ML: CatBoost_tuned

| Target | R2 |
|---|---:|
| Valence | 0.7220 |
| Energy | 0.9212 |
| Danceability | 0.7903 |
| Popularity | 0.1316 |
| Average | 0.6413 |

Best DL: AttentionTaskGatedFusionMLP_tuned

| Target | R2 |
|---|---:|
| Valence | 0.7214 |
| Energy | 0.9050 |
| Danceability | 0.7700 |
| Popularity | 0.1092 |
| Average | 0.6264 |

Key conclusion: CatBoost_tuned is strongest overall.

## Feature Set (4,254 features)

Final multimodal representation used for ML and DL:

- Audio: 23
- Text stats: 5
- Sentiment: 2
- MPNet embeddings: 768
- VGGish embeddings: 128
- MERT embeddings: 768
- PANNs embeddings: 2048
- Mel stats: 512

Feature order is defined in `ml/models/thesis_ml_models.py`.

## Thesis Notebook Suite

Notebooks:

- `notebooks/10_thesis_dataset_and_target_eda.ipynb`
- `notebooks/11_thesis_multimodal_feature_inventory.ipynb`
- `notebooks/12_thesis_final_ml_vs_dl_comparison.ipynb`
- `notebooks/13_thesis_architecture_and_hpo_analysis.ipynb`
- `notebooks/14_thesis_modality_and_interpretability_analysis.ipynb`
- `notebooks/15_thesis_error_analysis_and_significance.ipynb`

## Methodology Notes

- Artist-aware splits only.
- Validation split used for selection and HPO.
- Test split used once for final reporting.
- All results are split-explicit and include per-target R2/RMSE/MAE.

## Key Scripts

- ML baseline and final evaluation: `ml/models/thesis_ml_models.py`
- DL architecture comparison: `dl/14_thesis_architecture_comparison.py`
- HPO scripts: `ml/models/hpo_catboost.py`, `dl/15_hpo_attention_dl.py`

## Repository Highlights

- `docs/memory-bank/` contains the current project state and methodology
- `results/metrics/` and `results/dl_metrics/` contain final CSV results
- `models/checkpoints/` contains DL checkpoints
- `results/hpo/` contains tuned parameter JSONs
