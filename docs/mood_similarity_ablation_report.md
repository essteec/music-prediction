# Mood & Context Similarity Facet: Leave-One-Group-Out (LOGO) Ablation Study

## Executive Summary

This benchmark evaluates unifying **Mood (Spotify 11D + Vocal 12D + Emotion 36D)** with **Context (Genre Hybrid 50D + Temporal 10D)** into a single cohesive Context & Vibe facet.

### 1. LOGO Ablation Results Table (Overall 10,000 Tracks)

| Configuration | Remaining Dim | Overlap @10 | Overlap @50 | Vibe MAE @10 | Vibe Δ | Artist Agr @10 | Artist Δ | Genre Agr @10 | Impact Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full 5-Block Baseline (119-D)** | 119 | 100.0% | 100.0% | 0.1583 | `+0.0000` | 15.21% | `+0.00%` | 99.57% | **Reference (Full 5-Block)** |
| **Without Emotion & Sentiment (83-D)** | 83 | 41.1% | 53.1% | 0.1478 | `-0.0106` | 19.11% | `+3.90%` | 99.83% | **Distinct but Divergent / Noisy (Drop)** |
| **Without Genre Hybrid (69-D)** | 69 | 29.6% | 34.1% | 0.1547 | `-0.0036` | 4.94% | `-10.27%` | 75.06% | **Essential Signal (Keep - High Value)** |
| **Without Temporal & Collab (109-D)** | 109 | 47.0% | 53.8% | 0.1545 | `-0.0038` | 13.18% | `-2.03%` | 99.80% | **Essential Signal (Keep - High Value)** |
| **Without Spotify Audio (108-D)** | 108 | 69.4% | 77.8% | 0.1711 | `+0.0127` | 15.58% | `+0.37%` | 99.66% | **Essential Signal (Keep - High Value)** |
| **Without Vocal & DSP (107-D)** | 107 | 74.7% | 81.6% | 0.1647 | `+0.0064` | 14.72% | `-0.49%` | 99.63% | **Essential Signal (Keep - High Value)** |
| **Current Official Mood (59-D)** | 59 | 11.9% | 16.4% | 0.1403 | `-0.0181` | 3.03% | `-12.18%` | 71.97% | **Essential Signal (Keep - High Value)** |
| **Spotify 11-D Alone** | 11 | 2.1% | 4.0% | 0.0676 | `-0.0907` | 1.92% | `-13.29%` | 63.37% | **Essential Signal (Keep - High Value)** |
| **Genre Hybrid 50-D Alone** | 50 | 14.6% | 25.3% | 0.1832 | `+0.0249` | 31.67% | `+16.46%` | 100.00% | **Essential Signal (Keep - High Value)** |
| **Temporal 10-D Alone** | 10 | 4.5% | 8.8% | 0.2016 | `+0.0433` | 7.77% | `-7.44%` | 64.32% | **Essential Signal (Keep - High Value)** |
| **Vocal DSP 12-D Alone** | 12 | 2.6% | 4.8% | 0.1396 | `-0.0187` | 2.77% | `-12.44%` | 67.99% | **Essential Signal (Keep - High Value)** |
| **Emotion 36-D Alone** | 36 | 4.0% | 7.2% | 0.2147 | `+0.0564` | 1.38% | `-13.84%` | 61.89% | **Essential Signal (Keep - High Value)** |

### 2. Language Parity (English vs. Non-English/Instrumental)

| Configuration | English Vibe MAE | Non-English Vibe MAE | English Artist Agr | Non-English Artist Agr | English Genre Agr | Non-English Genre Agr |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full 5-Block Baseline (119-D)** | 0.1633 | 0.1502 | 15.27% | 15.12% | 99.48% | 99.71% |
| **Without Emotion & Sentiment (83-D)** | 0.1518 | 0.1411 | 19.00% | 19.29% | 99.82% | 99.86% |
| **Current Official Mood (59-D)** | 0.1456 | 0.1315 | 2.29% | 4.26% | 66.50% | 80.97% |
| **Emotion 36-D Alone** | 0.2147 | 0.2146 | 1.41% | 1.32% | 58.03% | 68.23% |

## 3. Key Observations

- **Without Emotion & Sentiment (83-D)**: Drops noisy zero-padding, improving both language parity and artist agreement.
- **Genre Hybrid (50-D)**: Essential anchor for stylistic consistency across both English and global tracks.
- **Spotify Audio (11-D)**: Core driver of continuous vibe, valence, and energy.
