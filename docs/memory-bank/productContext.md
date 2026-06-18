# Product Context

## What This Project Is

A research codebase for predicting four music attributes from multimodal features:

- Valence
- Energy
- Danceability
- Popularity

## Current Objective

Maintain clean, reproducible pipelines from data preprocessing through live inference. The app serves the final tuned models and must use preprocessing identical to training. Any future training-side changes must be mirrored in the app.

Comprehensive analysis notebooks and publication-quality figures are complete (see `notebooks/20_*` through `notebooks/27_*` and `results/figures/thesis/`).

## Baseline Reality

- Corrected Ultimate XGBoost is a very strong baseline after the NPZ split-index fix.
- DL improves Valence and ties Energy/Danceability, but does not beat ML on Popularity.
- Popularity likely depends on artist/contextual/external variables more than raw audio/text representation learning.

## Key Findings

1. CatBoost (avg R² 0.641) is the strongest model overall, winning 3/4 targets.
2. Multimodal DL ties CatBoost on Valence, suggesting cross-modal attention helps emotion prediction.
3. DL architecture progression (Flat → Fusion → Gated → Attention) adds measurable value.
4. Popularity remains difficult (best R² 0.132) — better modeled by classical ML over metadata-rich features.
