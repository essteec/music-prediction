# Extended Implementation Plan
## 10k Song Feature Extraction — Kaggle Dataset Quality Focus

> **Based on:** [`docs/report.md`](./report.md) + user decision responses (2026-08-24)
> **Primary Goal:** Maximize Kaggle dataset quality — focused on feature extraction and packaging
> **Hardware:** GTX 1660 Ti · 6 GB VRAM · 16 GB RAM · 34 GB audio corpus
> **Storage budget:** ≤ 70 GB total disk (2× audio size). Upload budget: minimize aggressively.

---

## Decision Log (User Choices)

| # | Decision Point | Choice |
|---|---|---|
| D1 | Lyric embedding model | Compare **BGE-M3 + GTE-multilingual-base + multilingual-E5-large** + MTEB shortlist candidates on 500-song pilot |
| D2 | DSP feature scope | **Extended ~180–220 dims** — full Essentia MusicExtractor + librosa |
| D3 | MuQ neural audio | **Yes — Tier 1** alongside CLAP and Essentia |
| D4 | MERT-v1-330M | **Tier 2 pilot** — 200 songs, scale if Δ nDCG@10 > 0.02 vs 95M |
| D5 | Demucs separation | **Skip entirely** — CLAP+Essentia covers acoustic information |
| D6 | Web product priority | **All four MVPs simultaneously** (Song Map, Similar Songs, Song DNA, NL Vibe Search) |
| D7 | Kaggle identifiers | **Spotify song_id + artist_id** as primary keys (songs.csv, artists.csv, genres.csv already prepared) |
| D8 | Lyric benchmark | **Small scale self-annotate** — 100 queries × 10 pairs = 1,000 pairs with 1–2 friends |
| D9 | LLM lyric annotation | **Skip** — BERTopic + NRC EmoLex is sufficient |
| D10 | Project focus | **Maximize Kaggle dataset quality** — extraction + packaging first |

---

## Storage Budget Plan

> **Hard constraint: ≤ 70 GB total disk usage.** Audio files = 34 GB. Remaining budget: ~36 GB for all features.

| Asset | Format | Est. Size | Keep Local? | Upload to Kaggle? |
|---|---|---|---|---|
| Audio files (Opus) | .opus | 34 GB | ✅ Local | ❌ No |
| **Embeddings (all .npy float32)** | | | | |
| MERT-v1-95M (768-D) | .npy | 30 MB | ✅ | ✅ |
| PANNs Cnn14 (2048-D) | .npy | 82 MB | ✅ | ✅ |
| PANNs 527-tag probs | .npy | 21 MB | ✅ | ✅ |
| VGGish (128-D) | .npy | 5 MB | ✅ | ✅ |
| Mel statistics (512-D) | .npy | 20 MB | ✅ | ✅ |
| LAION-CLAP (512-D) | .npy | 20 MB | ✅ | ✅ |
| MuQ-base (768-D) | .npy | 30 MB | ✅ | ✅ |
| Essentia Discogs-EffNet (1280-D) | .npy | 51 MB | ✅ | ✅ |
| Essentia Discogs 400-tag logits | .npy | 16 MB | ✅ | ✅ |
| BGE-M3 lyrics (1024-D) | .npy | 41 MB | ✅ | ✅ |
| GTE-multilingual (768-D) | .npy | 30 MB | ✅ | Pilot winner only |
| E5-large lyrics (1024-D) | .npy | 41 MB | ✅ | Pilot winner only |
| MTEB winner (dim varies) | .npy | 15–60 MB | ✅ | If selected |
| MPNet (768-D, existing) | .npy | 30 MB | ✅ | ✅ (baseline) |
| MiniLM (384-D, existing) | .npy | 15 MB | ✅ | ✅ (baseline) |
| **Scalar features (.parquet)** | | | | |
| DSP suite (~200 dims) | .parquet | ~8 MB | ✅ | ✅ |
| Essentia MusicExtractor (~200) | .parquet | ~8 MB | ✅ | ✅ |
| Lyric statistics + NRC + VADER | .parquet | ~3 MB | ✅ | ✅ |
| GoEmotions (English) | .parquet | ~1 MB | ✅ | ✅ |
| BERTopic topic IDs + probs | .parquet | ~1 MB | ✅ | ✅ |
| Language ID + QC flags | .parquet | ~1 MB | ✅ | ✅ |
| Metadata (songs/artists/genres.csv) | .parquet | ~5 MB | ✅ | ✅ |
| UMAP 2D coords | .parquet | <1 MB | ✅ | ✅ |
| kNN top-50 audio | .parquet | ~20 MB | ✅ | ✅ |
| kNN top-50 lyrics | .parquet | ~20 MB | ✅ | ✅ |
| **Total estimate** | | **~520 MB upload** | — | **~520 MB Kaggle** |

> ⚠️ **Warn triggers:** If any single .npy file exceeds 100 MB OR total embeddings exceed 500 MB, flag before uploading. MERT-330M embeddings (1024-D = 41 MB) are fine. Avoid any embeddings > 2048-D unless strictly required.

---

## MTEB Lyric Model Shortlist (From Your Benchmark File)

Based on filtering: `Open Weights=true`, `<3500 MB RAM`, `<2.5B params`, `ST Compatible=true`, `Retrieval > 60`, permissive license:

| MTEB Rank | Model | License | Mem MB | Dim | Max Tokens | Retrieval Score |
|---|---|---|---|---|---|---|
| 10 | `microsoft/harrier-oss-v1-0.6b` | MIT | 1,137 | 1024 | 32,768 | 70.8 |
| 71 | `ibm-granite/granite-embedding-311m-multilingual-r2` | Apache 2.0 | 594 | 768 | 8,192 | 65.2 |
| 18 | `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | 1,136 | 1024 | 32,768 | 64.7 |
| 22 | `BidirLM/BidirLM-1.7B-Embedding` | Apache 2.0 | 3,282 | 2048 | 8,192 | 62.2 |
| 26 | `BidirLM/BidirLM-1B-Embedding` | Apache 2.0 | 1,907 | 1152 | 32,768 | 61.6 |

> **Pilot strategy (§ Phase 1.5 below):** Run top 3 affordable MTEB candidates + BGE-M3 + GTE + E5-large on 500-song subset. Keep best 1–2 for full 10k extraction. Skip CC-BY-NC models (jina v5) for Kaggle.

**Recommended MTEB additions to pilot:** `harrier-oss-v1-0.6b` (MIT, 70.8 retrieval, 1024-D, long-context) and `Qwen3-Embedding-0.6B` (Apache 2.0, 64.7, 1024-D) — both fit well within 6 GB VRAM.

---

## Phase 0: Data Hygiene & Zero-Cost Features
**Duration: 1–2 days · CPU only · ~0 MB output**

### 0.1 Audio Quality Control
```bash
# Duration check vs Spotify metadata
python scripts/qc/audio_duration_check.py
# Output: features/qc/audio_qc.parquet — columns: duration_delta_s, silence_ratio, clipping_ratio, has_audio
```

### 0.2 Chromaprint Fingerprinting (Deduplication)
```bash
pip install pyacoustid
python scripts/qc/chromaprint_dedup.py
# Output: features/qc/fingerprints.parquet — columns: fingerprint_hash, is_duplicate_flag, duplicate_of_track_id
```

### 0.3 Lyric Cleaning Pipeline
```python
import re, unicodedata

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[.*?\]', '', text)          # Remove [Verse 1], [Chorus] etc.
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
# Note: save cleaned text LOCALLY only — never upload raw lyrics to Kaggle
```

### 0.4 Language Identification
```bash
pip install fasttext-wheel
python scripts/lyric/language_id.py
# Output: features/lyric/language_id.parquet — columns: lang_code, lang_confidence, is_english, is_multilingual
```

### 0.5 Statistics Pooling of Existing Embeddings ⭐ (Highest ROI action)
```python
import numpy as np

for name, path in [('mert', 'mert_v1_95m_768d.npy'), ('panns', 'panns_cnn14_2048d.npy'),
                   ('vggish', 'vggish_128d.npy'), ('mel', 'mel_statistics_512d.npy')]:
    emb = np.load(f'embeddings/audio/{path}')    # (10000, D)
    pooled = np.concatenate([
        emb.mean(axis=1, keepdims=True) if emb.ndim == 3 else emb,
        # If frame-level tensors available; otherwise enrichment below
    ], axis=-1)
    # For existing song-level embeddings: compute cross-embedding statistics
    # Save as-is — mean pooling already done; add std/max when re-extracting

# Re-extract MERT with multi-stat pooling:
# mean + std + max over all temporal frames → 3 × 768 = 2304-D
# Then PCA to 768-D (fit on train only) → same storage, richer info
```
> **Note:** Full benefit of mean+std+max requires re-extracting from frame-level outputs, not song-level .npy files. Flag this for re-extraction in Phase 1 alongside MuQ.

### 0.6 Lyric Structure & Lexical Features
```bash
pip install lexicalrichness textstat yake pronouncing NRCLex vaderSentiment
python scripts/lyric/lexical_features.py
# Output: features/lyric/lyric_stats.parquet
# Columns: line_count, avg_line_length, stanza_count, chorus_count_est, repetition_ratio,
#          unique_ratio, ttr, mtld, hd_d, hapax_ratio, flesch_ease,
#          nrc_anger, nrc_fear, nrc_joy, nrc_sadness, nrc_surprise, nrc_disgust,
#          nrc_trust, nrc_anticipation, nrc_positive, nrc_negative,
#          vader_compound, vader_pos, vader_neg, vader_neu,
#          top5_keywords (YAKE, stored as JSON string)
```

### 0.7 Metadata-Derived Features
```python
# From songs.csv / artists.csv — zero compute
# release_decade, log_followers, genre_count, is_explicit, has_lyrics, lang_code
# Output: features/metadata/derived.parquet  (~1 MB)
```

**Phase 0 Gate:** QC report shows: 0 unresolved NaN shapes, <5% audio duration mismatches >10s, duplicate list reviewed, all .npy shapes confirmed.

---

## Phase 1: Core High-ROI Audio Feature Extraction
**Duration: 3–5 days · GPU + CPU**

### 1.1 LAION-CLAP Audio Embeddings (512-D)
```bash
pip install laion-clap
python scripts/audio/extract_clap.py \
  --checkpoint laion/clap-htsat-unfused \
  --audio_dir data/audio/ \
  --output embeddings/audio/clap_htsat_512d.npy \
  --batch_size 4 --fp16 --chunk_sec 10
```
- VRAM: ~1.8 GB · Runtime: ~3.5–8 h · Output: (10000, 512) float32 = **20 MB**

> **Note on checkpoint:** Use `laion/clap-htsat-unfused` (Apache 2.0). Do NOT use `htsat-fused` or `music_audioset_epoch_15` — their exact license status is ambiguous. The `-unfused` variant is explicitly Apache 2.0.

### 1.2 PANNs 527-Class AudioSet Tag Probabilities
```python
# Re-run your existing PANNs pipeline but save the sigmoid output (527 probs)
# instead of/alongside the 2048-D embedding layer output
# Output: embeddings/audio/panns_tags_527d.npy — (10000, 527) float32 = 21 MB
# Column names: embeddings/audio/panns_tags_labels.json  (AudioSet 527 label names)
```

### 1.3 MuQ-base Neural Audio Embeddings (768-D) [NEW — Tier 1]
```bash
git clone https://github.com/tencent-ailab/MuQ
cd MuQ && pip install -e .
python scripts/audio/extract_muq.py \
  --model OpenMuQ/MuQ-base \
  --output embeddings/audio/muq_base_768d.npy \
  --batch_size 8 --fp16
```
- VRAM: ~2–3 GB · Runtime: ~1.4–3 h · Output: (10000, 768) float32 = **30 MB**
- After extraction: check `pearsonr(muq_emb.mean(1), mert_emb.mean(1))` — if r > 0.90 across tasks, consider dropping.

### 1.4 Extended DSP + Essentia MusicExtractor (~180–220 dims) [D2 Choice]

This is the single most important interpretable feature addition.

#### 1.4a librosa DSP suite (~100 dims)
```bash
pip install librosa==0.10 pyloudnorm madmom
python scripts/audio/extract_dsp_librosa.py  # overnight on CPU
```
| Group | Features | Dims |
|---|---|---|
| Rhythm | BPM, beat_conf, beat_interval_mean/std/cv, onset_rate, onset_strength_mean/std/max, tempo_stability, syncopation_proxy | 12 |
| Timbre | MFCC 1–20 mean/std (40), spectral centroid/bandwidth/rolloff/flatness/contrast mean/std (12), ZCR mean/std, spectral flux mean/std | 58 |
| Harmony | Chroma-CENS hist 12, chroma entropy/var, Tonnetz 6×mean/std, key (0–11), scale (0/1), key_strength, chord_entropy | 30 |
| Energy | RMS mean/std/max/q10/q90, LUFS_integrated, loudness_range, dynamic_complexity_proxy, crest_factor, low_energy_fraction | 12 |
| Sub-band | Sub-bass/bass/low-mid/high-mid/treble energy ratios | 5 |
| Structure | n_sections, section_dur_mean/std, novelty_peak_count, repetition_score, self_sim_mean, intro_dur_s, outro_dur_s | 7 |
| Stereo | stereo_width_mean, ms_ratio, lr_correlation, silence_ratio, clipping_ratio | 5 |

#### 1.4b Essentia MusicExtractor + TF Taggers (~120 dims)
```bash
pip install essentia-tensorflow  # includes TF model support
python scripts/audio/extract_essentia.py
```
| Model | Output | Dims | Time |
|---|---|---|---|
| MusicExtractor | key/scale/BPM/LUFS/onset/dissonance/HPCP/etc. | ~80 scalars | ~8–12 h CPU |
| Discogs-EffNet embedding | style embedding vector | 1280-D → **PCA to 128-D** (save both) | ~7–12 min |
| Discogs-EffNet logits | 400 genre/style probabilities | 400-D | same run |
| MTG-Jamendo mood_happy/sad/aggressive/relaxed | mood probabilities | 4 × ~8 prob outputs | ~45 min |
| MTG-Jamendo instrument taggers | instrument presence probs | ~20 probs | ~30 min |
| voice_instrumental classifier | is_vocal probability | 1 | ~15 min |

> ⚠️ **Storage warning:** Essentia Discogs-EffNet 1280-D = 51 MB. This is fine. But do NOT store raw 400-tag logits as float32 (16 MB) + 1280-D (51 MB) + PCA-128 (5 MB) all redundantly. **Store:** 1280-D raw (for downstream PCA if needed) + 400-tag probs as parquet (int8 or float16 → ~5 MB). PCA-128 computed at packaging time from train split only.

#### Combined DSP output
```
features/audio/dsp_librosa.parquet     (~8 MB, ~100 named columns)
features/audio/essentia_mir.parquet    (~8 MB, ~80 named columns)
embeddings/audio/essentia_effnet_1280d.npy   (51 MB)
features/audio/essentia_tags.parquet   (~5 MB, 400+instrument+mood probs as float16)
```

### 1.5 Silero VAD — Vocal Activity Ratio
```bash
python scripts/audio/extract_silero_vad.py
# Output: features/audio/vad.parquet  — columns: vocal_ratio, speech_ratio, silent_ratio  (~200 KB)
```
Runtime: ~1.5 h CPU · No VRAM needed.

**Phase 1 Gate (run before Phase 2):**
```python
# Run this ablation check on 1,000-song subset:
# 1. CLAP retrieval vs MERT: Spearman ρ < 0.7 → complementary ✅
# 2. MuQ vs MERT: Pearson r per dimension — if r < 0.85 on ≥50% dims → keep MuQ ✅
# 3. Essentia DSP adds R² > 0.02 on energy/danceability vs metadata-only baseline ✅
# If any gate fails, document and skip that feature for Kaggle upload
```

---

## Phase 1.5: Lyric Model Pilot (500 Songs)
**Duration: 1 day · GPU**

Compare these models on the same 500-song subset before committing to full 10k extraction:

| Model | HF ID | License | Dim | Max Tok | Memory | Why Include |
|---|---|---|---|---|---|---|
| **BGE-M3** | BAAI/bge-m3 | MIT | 1024 | 8192 | ~1.9 GB | Consensus #1 choice |
| **GTE-multilingual-base** | Alibaba-NLP/gte-multilingual-base | Apache 2.0 | 768 | 8192 | ~1.2 GB | Fast, long-context |
| **multilingual-E5-large** | intfloat/multilingual-e5-large | MIT | 1024 | 512 | ~2.2 GB | Strong MTEB scores |
| **harrier-oss-v1-0.6b** | microsoft/harrier-oss-v1-0.6b | MIT | 1024 | 32768 | ~1.1 GB | MTEB rank 10, very high retrieval |
| **Qwen3-Embedding-0.6B** | Qwen/Qwen3-Embedding-0.6B | Apache 2.0 | 1024 | 32768 | ~1.1 GB | MTEB rank 18, Apache license |
| **granite-311m-multilingual** | ibm-granite/granite-embedding-311m-multilingual-r2 | Apache 2.0 | 768 | 8192 | ~594 MB | Smallest feasible, Apache 2.0 |

```bash
pip install FlagEmbedding sentence-transformers
python scripts/lyric/pilot_lyric_comparison.py \
  --models bgem3 gte_multilingual e5_large harrier qwen3_0.6b granite_311m \
  --sample_size 500 \
  --metrics recall@10 ndcg@10 spearman_r2 \
  --output docs/lyric_pilot_results.md
```

**Pilot evaluation (proxy labels — no annotation needed yet):**
- Same-artist cosine sim ranking (should rank same-artist pairs higher)
- Same-genre vs random Recall@10
- English vs non-English retrieval separately (use fasttext labels)

**Decision after pilot:** Keep top 2 models by nDCG@10 for full 10k extraction. If a MTEB model ties or beats BGE-M3 on non-English lyrics specifically, include it as a secondary embedding.

---

## Phase 2: Lyric Feature Extraction (Full 10k)
**Duration: 2–3 days · GPU + CPU**

### 2.1 Winner Lyric Embedding(s) from Pilot
```bash
# Extract top 2 models from pilot on all 10k songs
python scripts/lyric/extract_lyric_embed.py \
  --model BAAI/bge-m3 \
  --output embeddings/lyrics/bge_m3_1024d.npy \
  --batch_size 8 --fp16
# Repeat for second place model
```
Runtime per model: ~6–50 min · Output per model: 41 MB (1024-D) or 30 MB (768-D)

### 2.2 GoEmotions RoBERTa — Sectional Emotion Arc (English only)
```bash
python scripts/lyric/extract_go_emotions.py \
  --lang_filter english \
  --output features/lyric/go_emotions.parquet
# Applies to ~6,500–7,000 English-detected songs only
# 28 emotion probabilities, averaged across song sections
# Output: ~1 MB parquet
```

### 2.3 BERTopic Topic Modeling
```bash
pip install bertopic
python scripts/lyric/extract_bertopic.py \
  --embeddings embeddings/lyrics/bge_m3_1024d.npy \
  --n_topics 32 \
  --output features/lyric/bertopic_topics.parquet
# Columns: topic_id, topic_label, top_words (JSON), topic_prob
# Output: ~1 MB parquet
```

### 2.4 MEmoLon Valence/Arousal/Dominance (Multilingual)
```bash
pip install MEmoLon
python scripts/lyric/extract_memelon.py \
  --output features/lyric/memelon_vad.parquet
# 3 dims (V/A/D) per song, multilingual, averaged word-level → song-level
# Output: ~300 KB parquet
```

### 2.5 Lyric Structure + NRC + Lexical (if not done in Phase 0)
Already covered in Phase 0.6. Verify completeness.

---

## Phase 3: Lyric Similarity Benchmark Construction (Self-Annotate)
**Duration: ~1–2 weekends · No compute**

### 3.1 Sample Design
```python
# 100 query songs, stratified by language × genre
# 10 candidates per query = 1,000 pairs total
# Candidate tiers:
#   - 2 same-artist (known positive control)
#   - 2 top-5 BGE-M3 nearest (different artist)
#   - 2 same genre / different BGE-M3 rank
#   - 2 same theme (BERTopic) / different genre
#   - 2 uniform random (hard negative)
python scripts/eval/build_lyric_benchmark.py \
  --n_queries 100 --n_candidates 10 \
  --output evaluation/lyric_similarity_benchmark.json
```

### 3.2 Annotation Protocol
- Tool: Simple spreadsheet (Google Sheets) or lightweight web form
- Rating per pair: **1–5 scale** on 4 dimensions:
  1. Thematic/topic overlap
  2. Emotional tone alignment
  3. Narrative style similarity
  4. Overall lyrical substitutability
- 3 annotators: you + 2 friends (target: Fleiss κ > 0.65)
- Estimated time: ~4–6 hours per annotator for 1,000 pairs

### 3.3 Benchmark Evaluation
```python
# After annotation, evaluate each lyric model:
python scripts/eval/eval_lyric_retrieval.py \
  --benchmark evaluation/lyric_similarity_benchmark.json \
  --embeddings embeddings/lyrics/ \
  --output docs/lyric_benchmark_results.md
# Metrics: nDCG@10, MRR, MAP@10, Recall@10/50, Spearman ρ vs human scores
```

---

## Phase 4: MERT-330M Pilot (200 Songs) [Tier 2]
**Duration: 0.5 days · GPU**

```bash
python scripts/audio/extract_mert_330m_pilot.py \
  --model m-a-p/MERT-v1-330M \
  --sample_size 200 \
  --fp16 --chunk_sec 30 \
  --batch_size 1 \
  --output embeddings/audio/pilot/mert_330m_pilot_200.npy
# VRAM: ~3.2 GB · Runtime: ~1.5–4 h for 200 songs
```

**Gate check:**
```python
from scipy.stats import pearsonr
import numpy as np
m95 = np.load('embeddings/audio/mert_v1_95m_768d.npy')[:200]
m330 = np.load('embeddings/audio/pilot/mert_330m_pilot_200.npy')

# Measure complementarity via retrieval nDCG@10 on validation subset
# Also: correlation between the two models
r_vals = [pearsonr(m95[i], m330[i, :768])[0] for i in range(200)]
print(f"Mean dim-wise Pearson r: {np.mean(r_vals):.3f}")
# If mean r > 0.92 AND retrieval improvement < 0.02 nDCG → SKIP MERT-330M full extraction
# If Δ nDCG@10 >= 0.02 → extract full 10k (~15–22 h)
```
Full extraction if gate passes: (10000, 1024) float32 = **41 MB** — acceptable.

---

## Phase 5: Ablation & Feature Selection
**Duration: 2–3 days**

Run the full ablation matrix to decide what to include in the Kaggle dataset:

```python
# Mandatory split: GroupKFold(n_splits=5) on artist_id — NO EXCEPTIONS
from sklearn.model_selection import GroupKFold
import pandas as pd

songs = pd.read_csv('data/processed/songs.csv')
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(songs, groups=songs['artist_id']))
```

| Ablation ID | Feature Set | Dims | Purpose |
|---|---|---|---|
| B0 | Spotify metadata only | ~13 | Hard baseline |
| B1 | Current audio (MERT+PANNs+VGGish+mel) | ~3,469 | Current stack |
| B2 | B1 + MPNet + MiniLM + lyric stats | ~4,628 | Current full pipeline |
| A_pool | B1 with mean+std (re-extracted) | ~6,938 | Pooling ablation |
| A_clap | B2 + CLAP 512 | ~5,140 | CLAP marginal value |
| A_muq | B2 + MuQ 768 | ~5,396 | MuQ marginal value |
| A_dsp | B2 + DSP suite ~200 | ~4,828 | Handcrafted MIR |
| A_essentia | B2 + Essentia MIR ~200 + tags | ~4,828 | Essentia marginal |
| L_new | B1 + new lyric embed + NRC + emotions | ~5,200 | Upgraded lyric stack |
| FULL | All above combined | ~7,000–9,000 | Max combination |
| Drop-one | FULL minus each modality | — | Per-modality importance |

```bash
python scripts/eval/run_ablation.py \
  --targets valence energy danceability genre \
  --models ridge catboost lightgbm \
  --n_bootstrap 1000 \
  --output docs/ablation_results.md
```

**Go/no-go per feature group (promote to Kaggle if):**
- Δ nDCG@10 ≥ 0.03 on retrieval OR Δ R² ≥ 0.02 on prediction
- p < 0.05 in 1,000-iteration bootstrap CI
- p-value survives artist-aware AND temporal cross-validation

---

## Phase 6: Kaggle Dataset Packaging
**Duration: 2–3 days**

### 6.1 Final Directory Structure
```
spotify-10k-music-features/
├── README.md                    ← Comprehensive dataset card
├── LICENSE                      ← CC-BY-4.0 for derived features
├── CHANGELOG.md
├── DATA_DICTIONARY.md           ← Every column documented with dim, dtype, model, license
├── PROVENANCE.md                ← Sources, extraction dates, HF commit hashes
│
├── metadata/
│   ├── songs.parquet            ← 10k rows, NO raw lyrics  [Spotify IDs kept with disclaimer]
│   ├── artists.parquet          ← Artist-level metadata
│   └── genres.parquet           ← Genre taxonomy
│
├── features/
│   ├── audio/
│   │   ├── dsp_librosa.parquet  ← ~100 librosa DSP features
│   │   ├── essentia_mir.parquet ← ~80 Essentia MusicExtractor features
│   │   ├── essentia_tags.parquet← 400 Discogs style + mood + instrument probs (float16)
│   │   └── vad.parquet          ← Vocal activity ratio
│   └── lyric/
│       ├── lyric_stats.parquet  ← Lexical richness, repetition, NRC EmoLex, VADER, YAKE
│       ├── go_emotions.parquet  ← 28-emotion probs (English only)
│       ├── bertopic_topics.parquet ← 32 topic IDs + probs
│       ├── memelon_vad.parquet  ← Valence/Arousal/Dominance (multilingual)
│       └── language_id.parquet  ← fasttext language labels
│
├── embeddings/
│   ├── audio/
│   │   ├── vggish_128d.npy      ← (10000, 128) float32   [5 MB]
│   │   ├── mert_v1_95m_768d.npy ← (10000, 768) float32  [30 MB]
│   │   ├── panns_cnn14_2048d.npy← (10000, 2048) float32 [82 MB] ⚠️ largest
│   │   ├── panns_tags_527d.npy  ← (10000, 527) float32  [21 MB]
│   │   ├── mel_stats_512d.npy   ← (10000, 512) float32  [20 MB]
│   │   ├── clap_512d.npy        ← (10000, 512) float32  [20 MB]
│   │   ├── muq_base_768d.npy    ← (10000, 768) float32  [30 MB]
│   │   └── essentia_effnet_1280d.npy ← (10000, 1280) float32 [51 MB]
│   └── lyric/
│       ├── bge_m3_1024d.npy     ← (10000, 1024) float32 [41 MB]  ← winner
│       ├── [pilot_winner_2].npy ← Second best from pilot  [varies]
│       ├── mpnet_768d.npy       ← (10000, 768) float32  [30 MB] existing baseline
│       └── minilm_384d.npy      ← (10000, 384) float32  [15 MB] existing baseline
│
├── similarity/
│   ├── knn_audio_top50.parquet  ← Top-50 audio neighbors per song [~20 MB]
│   ├── knn_lyric_top50.parquet  ← Top-50 lyric neighbors per song [~20 MB]
│   ├── umap_2d_audio.parquet    ← 2D UMAP of combined audio embeds [<1 MB]
│   └── umap_2d_lyric.parquet    ← 2D UMAP of lyric embeds [<1 MB]
│
├── splits/
│   ├── artist_grouped_5fold.parquet ← GroupKFold splits (MANDATORY for eval)
│   └── temporal_split.parquet       ← Chronological train/val/test
│
├── evaluation/
│   └── lyric_similarity_benchmark.json ← 1,000 annotated pairs [Phase 3]
│
├── manifests/
│   ├── extraction_manifest.json ← Per-file: model, HF commit hash, date, shape, dtype
│   └── checksums.json           ← SHA-256 per .npy and .parquet file
│
└── track_ids.npy                ← Master ID alignment array [400 KB]
```

> ⚠️ **Storage check before upload:**
> - PANNs Cnn14 2048-D = 82 MB — largest single file, still fine
> - Total estimate: ~520 MB — well under budget ✅
> - If MERT-330M pilot passes: +41 MB → ~560 MB still fine ✅
> - If both lyric pilot winners included: max +41 MB → ~600 MB ✅

### 6.2 Packaging Scripts
```bash
# Generate manifests
python scripts/package/generate_manifest.py
python scripts/package/generate_checksums.py

# Build kNN indexes (not uploaded, used for web product)
python scripts/package/build_knn.py \
  --audio_embeds embeddings/audio/clap_512d.npy embeddings/audio/mert_v1_95m_768d.npy \
  --lyric_embeds embeddings/lyrics/bge_m3_1024d.npy \
  --k 50

# Compute UMAP (fit on full dataset, store coords only)
python scripts/package/compute_umap.py

# Generate split files
python scripts/package/generate_splits.py \
  --songs metadata/songs.parquet \
  --group_col artist_id \
  --n_folds 5
```

### 6.3 What NOT to Upload (Hard Stops)
| Asset | Reason |
|---|---|
| Raw lyrics text column | Copyright — literary works |
| Audio .opus files | YouTube / copyright |
| Audio stems | Derived from copyrighted audio |
| Essentia AGPL code | Code license, not data |
| Chromaprint fingerprint hashes | Keep local for QC only |
| LLM-generated lyric summaries | Reproducibility + copyright ambiguity |
| Any .npy > 100 MB without prior warning | Upload budget |

### 6.4 Kaggle Dataset Card Template
```markdown
# Spotify Top-10k Music Feature Dataset

A research-grade feature dataset extracted from 10,000 popular Spotify tracks (July 2025),
providing multi-modal embeddings, structured audio descriptors, and interpretable lyric features
for music information retrieval, similarity modeling, and machine learning research.

## What's Included
- **Audio embeddings:** VGGish (128-D), MERT-v1-95M (768-D), PANNs Cnn14 (2048-D),
  LAION-CLAP (512-D), MuQ-base (768-D), PANNs AudioSet tags (527-D), Mel statistics (512-D),
  Essentia Discogs-EffNet (1280-D)
- **Lyric embeddings:** BGE-M3 (1024-D), [pilot winner 2], MPNet baseline (768-D), MiniLM (384-D)
- **Structured audio features:** ~200 DSP descriptors (librosa + Essentia MusicExtractor)
- **Interpretable lyric features:** NRC EmoLex, VADER, lexical richness, BERTopic topics,
  GoEmotions arc (English), MEmoLon VAD (multilingual)
- **Similarity index:** Pre-computed top-50 audio + lyric neighbors, UMAP 2D coords
- **Evaluation:** Artist-grouped 5-fold splits, 1,000-pair lyric similarity benchmark

## What's NOT Included
- Raw lyrics (copyright), audio files (YouTube ToS), audio stems

## Identifiers
- Primary key: `spotify_track_id`, `spotify_artist_id` [research use — Spotify API derived]
- Cross-reference: ISRC, MusicBrainz Recording ID where available

## Citation
...
```

---

## Phase 7: Web Product MVP (4 Features Simultaneously)
**Duration: 3–4 weeks · Next.js + FastAPI**

### Architecture
```
Browser (Next.js / React)
  ↕ REST JSON / WebSocket
FastAPI Backend (Python 3.11)
  ├── Pre-loaded: FAISS IndexFlatIP (10k → instant, <100ms)
  │     • audio index: CLAP + MERT (multi-index, RRF merge)
  │     • lyric index: BGE-M3
  │     • combined index: concat(PCA(audio, 128), PCA(lyric, 128))
  ├── Pre-loaded: kNN JSON (top-50 per song, served from dict)
  ├── Pre-loaded: Parquet metadata (pandas in memory, ~20 MB)
  ├── CLAP text encoder (for NL vibe search — ~1.8 GB, loaded lazily)
  └── UMAP coords + feature data (served as static JSON at startup)
```

### Feature 1: Interactive 2D Song Map
- **Stack:** regl-scatterplot (WebGL) for 10k points
- **Data:** Pre-computed UMAP 2D coords (`umap_2d_combined.parquet`) → baked into frontend JSON at build time
- **Interaction:** Hover → song card; Click → Song DNA page; Lasso → filter panel
- **Color by:** Genre / Mood (Essentia) / Language / Release decade / BERTopic topic
- **API endpoint:** None needed — UMAP coords are static; queries use HNSW, not UMAP

### Feature 2: Multi-Modal Similar Songs
```python
@app.get("/similar/{track_id}")
async def similar_songs(track_id: str, 
                         audio_weight: float = 0.6, 
                         lyric_weight: float = 0.4,
                         n: int = 10):
    # Load pre-computed top-50 from JSON (instant)
    audio_neighbors = knn_audio[track_id][:50]
    lyric_neighbors = knn_lyric[track_id][:50]
    # Reciprocal Rank Fusion (RRF)
    merged = rrf_merge(audio_neighbors, lyric_neighbors, 
                       weights=[audio_weight, lyric_weight])
    return merged[:n]
```

### Feature 3: Song DNA Page
- **Components:** Radar chart (energy/valence/danceability/tempo_normalized/speechiness)
- **Data:** Spotify descriptors + Essentia MIR + VAD from parquet
- **Tags:** Top-5 PANNs AudioSet tags (confidence-filtered) + Essentia Discogs top-3 styles
- **Neighbors:** 3 panels — Audio (CLAP), Lyric (BGE-M3), Combined (RRF)
- **Mood:** GoEmotions top-3 emotions (English) + Essentia mood bar

### Feature 4: Natural Language Vibe Search
```python
import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt('laion/clap-htsat-unfused')

@app.get("/search")
async def nl_search(query: str, n: int = 10, filters: dict = {}):
    # Encode text query using CLAP text encoder
    query_emb = model.get_text_embedding([query])   # (1, 512)
    # Search against pre-built CLAP audio FAISS index
    D, I = faiss_clap_index.search(query_emb, n * 3)
    # Apply metadata filters (genre, language, era)
    results = apply_filters(I, D, filters)[:n]
    return results
```

### Phase 7 Tech Stack
```bash
# Backend
pip install fastapi uvicorn faiss-cpu pandas pyarrow laion-clap torch

# Frontend
npx create-next-app@latest music-explorer --typescript --tailwind
npm install @regljs/regl @regl-worldwind/regl-scatterplot d3 recharts

# Optional: regl-scatterplot for the map
npm install regl-scatterplot
```

---

## Master Timeline

```
Week 1
  ├── Phase 0 complete: QC, lang-ID, chromaprint, lyric clean, stats pooling, lexical
  └── Phase 1 start: LAION-CLAP extraction (overnight GPU)

Week 2
  ├── Phase 1 complete: PANNs 527, MuQ, DSP suite (overnight CPU), Essentia, Silero VAD
  └── Phase 1.5: Lyric model pilot (500 songs × 6 models, 1 day)

Week 3
  ├── Phase 2: Full 10k lyric extraction (winner × 2), GoEmotions, BERTopic, MEmoLon
  └── Phase 4: MERT-330M pilot (200 songs, gate check)

Week 4
  ├── Phase 3: Build lyric benchmark (sample + annotation tool setup)
  ├── Phase 5: Run full ablation matrix, drop low-value features
  └── Phase 6 start: Package Kaggle dataset, generate manifests

Week 5
  ├── Phase 6 complete: Kaggle upload, dataset card, README
  └── Phase 7 start: FastAPI backend skeleton + FAISS indexes

Week 6–8
  └── Phase 7: Web product MVP — all 4 features simultaneously
```

---

## Quick-Start Install

```bash
# Phase 0
pip install librosa==0.10 pyloudnorm pyacoustid fasttext-wheel \
            NRCLex vaderSentiment yake lexicalrichness textstat \
            pronouncing pandas pyarrow numpy

# Phase 1
pip install laion-clap
pip install essentia-tensorflow     # or: pip install essentia
pip install madmom                  # for beat tracking
# MuQ: git clone https://github.com/tencent-ailab/MuQ && pip install -e ./MuQ

# Phase 2
pip install FlagEmbedding sentence-transformers bertopic

# Phase 5 (ablation)
pip install scikit-learn catboost lightgbm xgboost shap faiss-cpu umap-learn

# Phase 7 (web backend)
pip install fastapi uvicorn hnswlib
```

---

## Reminders & Hard Rules

> **NEVER** upload raw lyrics, audio files, or audio stems to Kaggle.

> **ALWAYS** use `GroupKFold(n_splits=5)` on `artist_id` — random splits are invalid for this dataset.

> **ALWAYS** fit StandardScaler, PCA, and UMAP on the training split only. Never on full dataset before splitting.

> **NEVER** use 2D UMAP coordinates for nearest-neighbor retrieval. Use FAISS/HNSW on full-dimensional embeddings.

> **NEVER** include `popularity`, `total_artist_followers`, or `rank` as input features when predicting popularity.

> **WARN** if any single .npy file exceeds 100 MB or total upload exceeds 600 MB before proceeding.

> **PIN** all HF model commit hashes in `manifests/extraction_manifest.json` for reproducibility.
