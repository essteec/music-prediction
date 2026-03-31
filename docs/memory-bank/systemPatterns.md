# System Patterns

## Data and Modeling Pattern
1. Load cached arrays from `ml/features/` (audio, text stats, sentiment, targets).
2. Load external lyric embeddings from `data/embeddings/`.
3. Concatenate features into one matrix per split.
4. Train multi-output regression model for 4 targets.
5. Evaluate per target and per split using R2/RMSE/MAE.

## Directory Pattern
- Training scripts: `dl/`
- Shared helpers: `dl/utils/`
- Checkpoints: `models/checkpoints/`
- Metrics outputs: `results/dl_metrics/`
- Narrative state/docs: `docs/memory-bank/`

## Experiment Pattern
1. Baseline lock.
2. Single-factor change.
3. Same metrics schema and naming.
4. Comparison table with explicit deltas.

## Reliability Pattern
1. Deterministic seed setup.
2. Early stopping.
3. Save best validation checkpoint.
4. Keep run metadata with timestamps.
