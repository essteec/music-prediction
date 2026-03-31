# RFE Methodology (Archived Reference)

## Status
Complete. Preserved for reference; not the active optimization track.

## Purpose
Identify compact feature subsets per target using CatBoost-based recursive elimination.

## Key Setup
- Start: 414 features
- Remove per iteration: 10
- Stop guard: if R2 drop from baseline > 1%
- Safety floor: 20 features

## Critical Implementation Rules
1. Compare each iteration to the original baseline R2 (not only previous iteration).
2. On stop, restore the previous valid feature set.
3. Track both per-iteration and total baseline drop.

## Best Historical Iterations
- Valence: 184 features, R2 ~0.4169
- Energy: 34 features, R2 ~0.8485
- Danceability: 74 features, R2 ~0.6123
- Popularity: 394 features, R2 ~0.1359

## Practical Takeaway
RFE provided interpretability and efficiency wins, but current Semester 2 priority is DL architecture experimentation rather than additional classic feature pruning.
