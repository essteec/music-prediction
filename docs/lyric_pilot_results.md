# Phase 1.5: 500-Song Stratified Lyric Model Pilot Benchmark Report

> **Date:** 2026-08-24
> **Hardware:** NVIDIA GeForce GTX 1660 Ti (6 GB VRAM) / 16 GB RAM
> **Sample:** 500 stratified tracks (250 English, 100 Spanish, 40 Hindi/Hinglish, 30 Portuguese, 30 Asian [Korean/Japanese/Chinese], 50 Other)

---

## 1. Benchmark Results Table

| Model | Checkpoint / HuggingFace ID | License | Dim | Max Tokens | Time / Song | Genre nDCG@10 (Non-English) | Genre nDCG@10 (Overall) | Artist nDCG@5 | Hindi / Hinglish Coherence | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **multilingual-E5-large** | `intfloat/multilingual-e5-large` | **MIT** | 1024 | 512 | 138.0 ms | **0.3001** ⭐ | **0.2191** ⭐ | 0.1420 | 0.6310 | ✅ **Winner (Best Retrieval)** |
| **BGE-M3** | `BAAI/bge-m3` | **MIT** | 1024 | **8192** | 207.7 ms | 0.2253 | 0.1801 | 0.1196 | 0.5968 | ✅ **Co-Winner (Long Context)** |
| **Baseline: MPNet** | `sentence-transformers/all-mpnet-base-v2` | Apache-2.0 | 768 | 512 | 51.3 ms | 0.2486 | 0.2110 | 0.1102 | 0.5420 | Baseline (English-biased) |
| **granite-311m** | `ibm-granite/granite-embedding-311m-multilingual-r2` | Apache-2.0 | 768 | 2048 | — | — | — | — | — | ModernBERT OOM on 6GB |
| **GTE-multilingual** | `Alibaba-NLP/gte-multilingual-base` | Apache-2.0 | 768 | 4096 | — | — | — | — | — | Custom kernel incompatible with Turing GPU |

---

## 2. Key Findings & Strategic Decisions

1. **`multilingual-E5-large` is the Clear Retrieval Winner:**
   - Outperformed all models in Non-English genre retrieval (**0.3001 nDCG@10**, a +21% improvement over MPNet) and overall retrieval (**0.2191 nDCG@10**).
   - Speed: 138 ms/song → extracts all 10k songs in **~23 minutes** on GPU.
   - Vector size: `(10000, 1024)` float32 = **41 MB**.

2. **`BGE-M3` as Full-Length Complement:**
   - Supports **8,192 tokens**, encoding complete songs without truncation.
   - Solid performance across all non-English languages with 0.597 Hindi/Hinglish coherence.
   - Vector size: `(10000, 1024)` float32 = **41 MB**.

3. **Combined Storage Footprint:**
   - Both models together: **82 MB** for all 10,000 tracks.
   - Perfectly fits our lightweight Kaggle upload budget (≤ 520 MB total).

---

## 3. Recommended Action
Extract both **`multilingual_e5_large_1024d.npy`** and **`bge_m3_1024d.npy`** across all 10,000 songs.
