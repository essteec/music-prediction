# Lyric Similarity Fusion: Leave-One-Group-Out Ablation Study

## Executive Summary

This study evaluates the individual contribution of each multilingual lyric embedding model (`Harrier-OSS-v1-0.6B`, `Multilingual E5-Large`, `BGE-M3`) within the fused Top-100 kNN graph for the 10,000 Spotify tracks dataset.

### Ablation Results Table

| Configuration | Remaining Dim | Overlap @10 | Overlap @50 | Overlap @100 | Genre Agr @10 | Genre Δ | Artist Agr @10 | Artist Δ | Impact Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Full Baseline)** | 3072 | 100.0% | 100.0% | 100.0% | 78.38% | `0.00%` | 8.12% | `0.00%` | **Reference (Full 3-Model Ensemble)** |
| **Without Harrier-0.6B** | 2048 | 62.8% | 67.7% | 69.5% | 76.40% | `-1.98%` | 7.08% | `-1.04%` | **Essential Signal (Keep - High Quality Drop)** |
| **Without E5-Large** | 2048 | 85.5% | 88.6% | 89.7% | 78.06% | `-0.31%` | 7.87% | `-0.26%` | **Marginal / Neutral Contribution** |
| **Without BGE-M3** | 2048 | 59.0% | 65.0% | 68.2% | 79.35% | `+0.97%` | 8.15% | `+0.03%` | **Distinct but Harmful / Noisy (Drop Candidate)** |

## Metric Definitions

1. **Neighbor Overlap @ K (Jaccard Rank Overlap):** Percentage of Top-K nearest neighbors shared with the full 3-model baseline. Lower overlap indicates the removed model provides a **distinct semantic representation**.
2. **Genre Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors that share at least one genre with the query song. Negative Δ indicates removing the model degrades genre consistency.
3. **Artist Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors by the same artist/collaborator.
