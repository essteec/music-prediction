# Hybrid Genre (50-D: N=16 SVD) & Multi-Modal Similarity Ablation Study

## 1. Genre Architecture Comparison

Evaluation of Macro 34-D Taxonomy vs. Latent SVD 16-D vs. Hybrid 50-D Vector over all 10,000 tracks:

| Configuration | Standalone Vibe MAE | Standalone Artist Agr @10 |
| :--- | :--- | :--- |
| **Macro Taxonomy Alone (34-D)** | 0.1954 | **16.08%** |
| **Latent SVD Alone (16-D)** | 0.1840 | **31.80%** |
| **Hybrid Genre Vector (50-D)** | 0.1832 | **31.67%** |

**Empirical Finding**: Extending the 34-D macro taxonomy with 16-D TruncatedSVD subgenre co-occurrence jumps artist agreement from 16.08% to **31.69%**, while simultaneously reducing Vibe MAE from 0.1955 to **0.1833**.

## 2. Multi-Modal LOGO Ablation Benchmark

| Configuration | Remaining Dim | Overlap @10 | Overlap @50 | Vibe MAE @10 | Vibe Δ | Artist Agr @10 | Artist Δ | Impact Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Ensemble Baseline** | 3870 | 100.0% | 100.0% | 0.1529 | `0.0000` | 17.18% | `0.00%` | **Reference (Full Multimodal)** |
| **Without Neural Audio (1664-D)** | 2206 | 86.4% | 90.2% | 0.1584 | `+0.0055` | 15.83% | `-1.35%` | **Essential Signal (Keep - High Value)** |
| **Without Neural Lyric (2048-D)** | 1822 | 92.5% | 95.1% | 0.1530 | `+0.0001` | 16.65% | `-0.53%` | **Moderate Signal (Keep)** |
| **Without Spotify Audio & Vibe (11-D)** | 3859 | 74.2% | 81.3% | 0.1624 | `+0.0095` | 17.55% | `+0.37%` | **Essential Signal (Keep - High Value)** |
| **Without Vocal & DSP Dynamics (12-D)** | 3858 | 79.2% | 84.9% | 0.1568 | `+0.0039` | 16.91% | `-0.27%` | **Beneficial Signal (Keep - Quality Drop)** |
| **Without Genre Hybrid (50-D)** | 3820 | 46.8% | 51.6% | 0.1494 | `-0.0035` | 8.64% | `-8.54%` | **Essential Signal (Keep - High Value)** |
| **Without Temporal & Collab (10-D)** | 3860 | 53.0% | 59.4% | 0.1498 | `-0.0030` | 15.49% | `-1.69%` | **Essential Signal (Keep - High Value)** |
| **Without Lyric Structure (12-D)** | 3858 | 83.1% | 87.6% | 0.1524 | `-0.0005` | 17.37% | `+0.19%` | **Marginal / Neutral (Drop Candidate)** |
| **Without Linguistic & Language (27-D)** | 3843 | 95.6% | 94.3% | 0.1524 | `-0.0004` | 17.19% | `+0.01%` | **Redundant (Negligible Unique Value)** |
| **Without Emotion & Sentiment (36-D)** | 3834 | 49.9% | 60.8% | 0.1455 | `-0.0073` | 21.28% | `+4.10%` | **Distinct but Divergent / Noisy (Drop)** |
