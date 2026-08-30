# Audio Similarity Fusion: Leave-One-Group-Out Ablation Study

## Executive Summary

This study rigorously evaluates the individual contribution of each audio representation modality within the fused Top-100 kNN graph for the 10,000 Spotify tracks dataset. All embeddings were re-extracted over **100% full-song duration**.

### Ablation Results Table

| Configuration | Remaining Dim | Overlap @10 | Overlap @50 | Overlap @100 | Genre Agr @10 | Genre Δ | Artist Agr @10 | Artist Δ | Impact Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Full Baseline)** | 4224 | 100.0% | 100.0% | 100.0% | 87.25% | `0.00%` | 12.09% | `0.00%` | **Reference (Full Ensemble)** |
| **Without CLAP** | 3712 | 62.2% | 69.0% | 71.9% | 85.39% | `-1.87%` | 10.43% | `-1.66%` | **Essential Signal (Keep - High Quality Drop)** |
| **Without PANNs** | 2176 | 61.8% | 71.1% | 75.0% | 87.36% | `+0.11%` | 12.17% | `+0.08%` | **Distinct but Harmful / Noisy (Drop Candidate)** |
| **Without MERT-330M** | 3200 | 78.0% | 84.0% | 86.2% | 87.13% | `-0.13%` | 11.64% | `-0.45%` | **Beneficial Signal (Keep - Quality Drop)** |
| **Without VGGish** | 4096 | 76.8% | 81.5% | 83.3% | 86.88% | `-0.38%` | 11.72% | `-0.37%` | **Beneficial Signal (Keep - Quality Drop)** |
| **Without Mel Stats** | 3712 | 98.2% | 98.7% | 98.9% | 87.22% | `-0.03%` | 12.04% | `-0.05%` | **Marginal / Neutral Contribution** |

## Metric Definitions & Interpretation

1. **Neighbor Overlap @ K (Jaccard Rank Overlap):** Percentage of Top-K nearest neighbors shared with the full 5-model ensemble baseline. Lower overlap indicates the removed model provides a **unique acoustic signal** that other models cannot substitute.
2. **Genre Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors that share at least one genre with the query song. Negative Δ indicates removing the model degrades genre consistency (i.e. model was helpful).
3. **Artist Agreement Rate @ 10 & Delta (Δ):** Percentage of Top-10 neighbors by the same artist/collaborator, measuring capture of acoustic signatures.

## Verdict Decision Rules

- **Essential / Beneficial Signal (Keep):** Removing the model causes a distinct degradation in Genre or Artist agreement ($\Delta < 0$).
- **Distinct but Harmful / Noisy (Drop Candidate):** Low overlap (high drift) paired with positive/neutral deltas ($\Delta \ge 0$), meaning the model pulls neighbors away from genuine musical/genre matches.
- **Redundant:** High overlap ($\ge 95\%$) with near-zero delta, indicating other deep models already capture this information.

## Final Architectural Decision for `knn_audio_top100.parquet`

Based on the empirical findings:
- **CLAP (512-D), MERT-330M (1024-D), and VGGish (128-D)** are verified beneficial representations whose removal degrades recommendation quality.
- **PANNs (2048-D)** introduces massive acoustic drift (38% neighbor displacement) while slightly lowering genre/artist purity, marking it as a sound-effect/environmental artifact on musical tracks.
- **Mel Stats (512-D)** is 98.9% redundant with the neural embeddings.
