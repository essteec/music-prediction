# Music Attribute Prediction

This project predicts four music attributes from multimodal features:
valence, energy, danceability, and popularity.
Download [Kaggle dataset](https://www.kaggle.com/datasets/serkantysz/490k-spotify-song-audio-embeddings-and-metadata/data) and put under 'data/processed/'.

## Status

- Final test evaluation complete (train+val retrain, single test run)
- Notebooks and publication-quality figures complete
- Current focus: Reporting and packaging

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

## Comprehensive Report Notebook Suite (20_* Series)

Deep exploratory notebooks analyzing all 8 ML families, 5 DL architectures, 4 targets, and 8 modalities — not just the two tuned finalists:

- `notebooks/20_thesis_comprehensive_dataset_eda.ipynb`
- `notebooks/21_thesis_comprehensive_feature_inventory.ipynb`
- `notebooks/22_thesis_comprehensive_ml_baselines.ipynb`
- `notebooks/23_thesis_comprehensive_dl_architecture.ipynb`
- `notebooks/24_thesis_comprehensive_hpo_analysis.ipynb`
- `notebooks/25_thesis_comprehensive_feature_modality_importance.ipynb`
- `notebooks/26_thesis_comprehensive_error_analysis.ipynb`
- `notebooks/27_thesis_comprehensive_final_comparison.ipynb`

A concise report narrative suite (`10_*` through `15_*`) also exists but is superseded by the 20_* series for exploratory depth.

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
