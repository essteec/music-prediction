# Critical Corrections and Non-Negotiables

## Methodology Rules
1. Use artist-aware splits only (group by `artist_id`).
2. Keep test set untouched until final evaluation.
3. Cache expensive artifacts (embeddings/features/checkpoints); do not recompute unnecessarily.
4. Compare new experiments against a fixed baseline, not moving baselines.

## Deep Learning Rules
1. Set deterministic seeds for every run.
2. Save best checkpoint by validation loss and record epoch.
3. Use early stopping.
4. Log per-target R2, RMSE, MAE for train/val/test.

## Known Pitfalls Already Fixed
1. Output activation mismatch with transformed targets.
2. Non-deterministic training due to missing seed setup.
3. Multi-target imbalance (target scaling/weighting issue).
4. PyTorch checkpoint loading compatibility (`weights_only` behavior).

## Operating Constraint
- Do not execute project Python scripts from the agent.
- Provide commands for user execution when runs are needed.
