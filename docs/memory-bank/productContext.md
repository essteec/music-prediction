# Product Context

## What This Project Is

A thesis-oriented codebase for predicting four music attributes from multimodal features:

- Valence
- Energy
- Danceability
- Popularity

## Current Objective

Convert the recent ML and DL experiments into clean, comparable, thesis-ready results. The priority is methodology quality and interpretability, not more unchecked experiment growth.

## Baseline Reality

- Corrected Ultimate XGBoost is a very strong baseline after the NPZ split-index fix.
- DL now clearly improves Valence and ties Energy/Danceability, but does not beat ML on Popularity.
- Popularity likely depends on artist/contextual/external variables more than raw audio/text representation learning.

## Success Definition Now

1. Produce a clean validation comparison across a small number of DL architectures.
2. Produce a final test comparison between selected ML and selected DL models.
3. Keep all results split-explicit and methodologically defensible.
4. Explain modality contributions clearly for thesis writing.

## Thesis Narrative

The strongest story is not "DL beats ML everywhere." The stronger and more honest story is:

- Multimodal DL meaningfully improves emotion-related prediction, especially Valence.
- Learned audio/text representations can match strong XGBoost baselines on Energy and Danceability.
- Popularity remains difficult and is still better modeled by classical ML over metadata-rich features.
- Architecture matters, but clear modality-aware architectures are preferable to opaque ensemble chasing for thesis work.
