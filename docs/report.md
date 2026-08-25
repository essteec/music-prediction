# Master Synthesis Report: 10k Song Feature Expansion & MIR Project Development

> **Synthesized from:** 8 LLM responses — Brainstorm_{ChatGPT, Claude, DeepSeek, Gemini} + Research_{ChatGPT, Claude, DeepSeek, Gemini}
> **Date:** 2026-08-24 · **Hardware Target:** GTX 1660 Ti 6 GB VRAM / 16 GB RAM

---

## 1. Executive Summary & Cross-LLM Consensus Matrix

### 1.1 Unified Technical Direction

All 8 responses converge on a core thesis: **your existing 4,256-D feature bank is already strong, and the highest ROI additions are not more generic audio SSL embeddings, but rather (a) structured/interpretable MIR descriptors, (b) a cross-modal audio–text model (CLAP), (c) multilingual lyric embeddings replacing truncated English-only ones, and (d) smarter temporal pooling of existing embeddings.** There is strong consensus on avoiding Jukebox, wav2vec/HuBERT for music, and running Demucs on the full 10k without a pilot. Moderate divergence exists on whether MERT-330M fits in 6 GB VRAM (Gemini/ChatGPT: yes with fp16; Claude-Research: no — OOM), and on which single lyric model to prioritize (BGE-M3 vs multilingual-E5 vs Nomic vs GTE).

### 1.2 Consensus vs. Divergence Table

| Proposed Area | Strong Consensus (5+ LLMs) | Moderate (2–4) | Unique Outlier (1) | Key Disagreements |
|---|---|---|---|---|
| **LAION-CLAP audio embeddings** | ✅ All 8 — must-try, cross-modal, 512-D, Apache 2.0 | — | — | Minor: which checkpoint |
| **Structured DSP/MIR features** | ✅ All 8 — #1 gap; CPU-only, free, interpretable | — | — | Recommended dims vary (50–390) |
| **BGE-M3 multilingual lyric embedding** | ✅ 7/8 — MIT, 1024-D, 8192 tokens | — | — | Claude-Research prefers Nomic v1.5 first |
| **Statistics pooling (mean+std+max)** | ✅ 7/8 — trivial cost, moderate-high gain | — | — | None |
| **PANNs 527-class tag vector** | ✅ 6/8 — overlooked interpretable layer | — | — | None |
| **Essentia Discogs-EffNet / mood taggers** | 5/8 — music-specific tags, CPU-fast | — | — | Gemini emphatic; some omit |
| **MuQ (300M)** | 3/8 — MARBLE SOTA (ChatGPT-B, DeepSeek-R) | — | ChatGPT-Brainstorm most enthusiastic | Others don't mention MuQ |
| **MERT-v1-330M** | 4/8 recommend; 3/8 defer | — | — | Claude-Research: OOM (8–12 GB needed) |
| **BEATs** | 4/8 recommend; 3/8 defer (redundant PANNs) | — | — | Divergence on priority |
| **Demucs stem separation** | ✅ All 8 — pilot 100–200 songs only | — | — | Runtime estimates diverge (35–300 h) |
| **Jukebox** | ✅ All 8 — REJECT; OOM, archived, non-commercial | — | — | None |
| **Artist-aware splits (GroupKFold)** | ✅ All 8 — mandatory | — | — | None |
| **No raw lyrics on Kaggle** | ✅ All 8 — CRITICAL legal risk | — | — | None |
| **No universal lyric benchmark** | ✅ All 8 — LyricSIM is closest; build in-domain | — | — | None |

### 1.3 Top 5 "Do Immediately"

| Rank | Action | Effort | Consensus |
|---|---|---|---|
| **1** | Statistics pooling (mean+std+max) of existing MERT/PANNs/VGGish frames | ~30 min, 0 VRAM | 7/8 |
| **2** | Compact DSP/MIR vector via librosa (~120–150 dims) | ~20–30 h CPU | 8/8 |
| **3** | PANNs 527-class AudioSet tag probabilities | ~25–40 min GPU | 6/8 |
| **4** | Lyric cleaning + fasttext language ID pipeline | ~30 min CPU | 8/8 |
| **5** | NRC EmoLex + stylistic/poetic features | ~2 h CPU | 6/8 |

### 1.4 Top 5 "Do Not Build / Defer"

| Rank | Item | Reason | Consensus |
|---|---|---|---|
| **1** | Jukebox representations | 5B params, 16+ GB VRAM OOM, non-commercial, archived | 8/8 REJECT |
| **2** | Full Demucs on all 10k songs | 35–300 h GPU; pilot 100–200 songs first | 8/8 pilot-only |
| **3** | wav2vec 2.0 / HuBERT | Speech SSL; redundant with MERT | 7/8 reject |
| **4** | Generative LLM annotation at 10k scale | 28–83 h CPU; hallucination risk; reproducibility issues | 7/8 defer Tier 3 |
| **5** | Concatenating everything into one giant vector | With 10k songs, high-D correlated dims hurt supervised models | 5/8 warn explicitly |

---

## 2. Audio Modeling & Extraction Inventory

### 2.1 Master Audio Comparison Table

| Model / Feature Family | Checkpoint | Dim | Signal Encoded | Redundancy | VRAM | Runtime 10k | License | Recommendation |
|---|---|---|---|---|---|---|---|---|
| **LAION-CLAP** | `laion/clap-htsat-unfused` or `larger_clap_music` | 512 | Cross-modal audio–text; zero-shot tag | **Low** — unique cross-modal | ~1.8 GB | 3.5–8 h | Apache-2.0 | ✅ Tier 1 Must |
| **Essentia Discogs-EffNet** | `discogs-effnet-bs64-1.pb` | 1280+400 | Music genre/style (Discogs taxonomy) | **Low-Medium** | <0.1 GB | 7–12 min | CC-BY-NC-SA | ✅ Tier 1 Must |
| **Essentia MTG-Jamendo taggers** | mood/instr models | ~96 | Perceived mood/instrument presence | **Low** | CPU | 3–5 min | CC-BY-NC-SA | ✅ Tier 1 Must |
| **PANNs 527-class sigmoid** | Re-run existing PANNs | 527 | AudioSet tag probs; interpretable | **Zero** — complements 2048-D | ~1.4 GB | 25–40 min | MIT | ✅ Tier 1 Must |
| **MuQ-base** | `OpenMuQ/MuQ-base` | 768 | Mel-RVQ SSL; MARBLE SOTA | **Medium** — diff pretraining | ~2–3 GB | 1.4–3 h | MIT-like | ✅ Tier 1 Try |
| **BEATs** | `beats_iter3_plus_AS2M_finetuned` | 768 | Audio SSL + acoustic tokenizers | **Medium-High** — overlaps PANNs | ~1.4 GB | 25 min–1.5 h | MIT | ⚠️ Tier 2 Pilot |
| **MERT-v1-330M** | `m-a-p/MERT-v1-330M` | 1024 | Music SSL, deeper than 95M | **High** — same family | ~3.2 GB fp16 | 1.25–22 h | CC-BY-NC-SA | ⚠️ Tier 2 Ablate |
| **EnCodecMAE-base** | `lpepino/encodecmae-base` | 768 | MAE over EnCodec tokens; timbral | **Medium** | ~0.5 GB | ~50 min | MIT | ⚠️ Tier 2 Try |
| **MuQ-MuLan-large** | `OpenMuQ/MuQ-MuLan-large` | 512 | Music-text joint; zero-shot | **Low** — cross-modal | ~4–5.8 GB | 20–56 h | CC-BY-NC-4.0 | ⚠️ Pilot Only |
| **CLaMP3 SAAS** | sanderwood/clamp3 | 768 | Universal MIR; multilingual | **Medium** | ~4–5.8 GB | 22–56 h | MIT code | ⚠️ Pilot Only |
| **Dasheng-base** | `mispeech/dasheng-base` | 768 | General audio SSL; HEAR benchmark | **Medium-High** | ~1–2 GB | 50 min–1.5 h | Apache-2.0 | Optional |
| **MusicFM** | ICASSP 2024 | ~768 | Music foundation; 30s context | **Medium** | ~1 GB | 10–15 h | Verify | Optional |
| **AudioMAE ViT-B** | facebookresearch/AudioMAE | 768 | General audio MAE | **Very High** — redundant PANNs | ~1.5 GB | ~5.5 h | Apache-2.0 | Skip |
| **Music2Vec** | `m-a-p/music2vec-v1` | 768 | Same lab as MERT; diff SSL | **High** — redundant | ~1.5 GB | ~6 h | CC-BY-NC-4.0 | Skip |
| **YAMNet logits** | `google/yamnet/1` | 521 | AudioSet tags; MobileNet | **High** — overlaps PANNs | CPU | ~1.5 h | Apache-2.0 | Optional |
| **Jukebox** | openai/jukebox | 4800 | Generative VQ-VAE | Medium-unique | ❌ 14+ GB | >200 h | Non-commercial | ❌ REJECT |
| **Demucs htdemucs** | facebookresearch/demucs | 4 stems | Source separation | Unique (isolation) | ~3.1 GB | 35–83 h | MIT | ⚠️ Pilot 100-200 |
| **Basic Pitch** | spotify/basic-pitch | ~13 stats | Polyphonic note transcription | **Low** | CPU (ONNX) | ~8 h | Apache-2.0 | ⚠️ Tier 2 |
| **Silero VAD** | snakers4/silero-vad | ~3 | Vocal activity ratio | **None** — QC | CPU 2MB | ~1.5 h | MIT | ✅ Tier 1 |
| **Chromaprint** | fpcalc + pyacoustid | fingerprint | Audio identity; dedup | **None** — identity | CPU | ~8–30 min | LGPL-2.1 | ✅ Tier 0 |

### 2.2 Compact DSP/MIR Feature Suite (~120–180 dims)

**Consensus: HIGHEST PRIORITY (8/8 LLMs). This is the single biggest gap in your current representation.**

| Group | Features | Dims | Library | CPU Time (10k) | Task Value |
|---|---|---|---|---|---|
| **Rhythm** | BPM, beat confidence, beat interval mean/std/CV, onset rate, onset strength stats, onset density, tempo stability, syncopation proxy | 12–15 | librosa, madmom | ~2–6 h | Dance ★★★★★ |
| **Timbre** | MFCC 1–20 mean/std, spectral centroid/bandwidth/rolloff/flatness/contrast mean/std, ZCR, spectral flux | 40–80 | librosa | ~3–5 h | Genre ★★★★★ |
| **Harmony** | Chroma-CENS hist (12), chroma entropy/var, Tonnetz (6 stats), HPCP, key, scale, key strength, chord entropy, harmonic change rate | 30–40 | librosa, essentia | ~4–6 h | Mood ★★★★ |
| **Energy/Dynamics** | RMS mean/std/max/q10/q90, LUFS, loudness range, dynamic complexity, low-energy fraction, sub-bass ratio, crest factor, HPSS H/P ratio | 15–20 | librosa, pyloudnorm | ~2–4 h | Energy ★★★★★ |
| **Structure** | Number of sections, section dur mean/std, novelty peaks, repetition score, self-similarity mean, intro/outro duration | 8–10 | librosa.segment | ~2–5 h | Genre ★★★★ |
| **Stereo/Production** | Stereo width, M/S ratio, L/R correlation, silence %, clipping ratio | 6–8 | manual STFT | ~1 h | Production style ★★★ |
| **Frequency bands** | Sub-bass/bass/low-mid/high-mid/treble energy ratios | 5–6 | librosa | ~1 h | Dance ★★★★ |

**Total: ~120–180 dims, ~20–30 h CPU overnight. Pure ISC/BSD licensed.**

> **Note on Madmom:** Provides superior beat/downbeat tracking vs librosa (DBN, meter confidence, syncopation). License: BSD 2-Clause code, CC-BY-NC-SA pretrained models — fine for extraction, cannot redistribute model files.

### 2.3 Better Temporal Pooling (Existing Embeddings)

**Consensus: #1 cheapest high-value action — do before adding any new models (7/8 LLMs)**

| Pooling Method | Dims Added | Cost | Verdict |
|---|---|---|---|
| Mean (current) | baseline | 0 | Already have |
| **Mean + Std** | +D | Trivial (NumPy) | **Must-Try** |
| **Mean + Std + Max** | +2D | Trivial | **Must-Try** |
| Mean + Std + q10 + q90 | +3D | Trivial | Try |
| Beat-synchronous mean | +D | Low (madmom) | Try |
| Intro/middle/outro segments | 3×D per model | Moderate (re-extract) | Pilot first |
| Attention-weighted | +D | Low training | After baseline |

> **Critical:** Current MERT-95M = mean over first 30s only. For a 3.5-min song, this misses chorus, bridge, outro entirely. Multi-window extraction is high-priority pilot.

### 2.4 Source Separation & Vocal Analysis

**Consensus (8/8 LLMs): Pilot 100–200 songs ONLY. Do NOT run full 10k first.**

Do NOT store stems (~1.6 TB). Instead derive:
- Vocal/instrumental RMS energy ratio
- Stem-specific CLAP/MERT embeddings per stem (optional)
- Vocal pitch statistics (median F0, range, std) via CREPE post-Demucs

**Gate:** Stem features must improve retrieval nDCG@10 by ≥ +0.03 OR R² by ≥ +0.02 on 100-song test. Otherwise reject full extraction.

---

## 3. Lyrics & NLP Representation Inventory

### 3.1 Master Lyric Comparison Table

| Technique / Model | Repo | Dims | Multilingual | Context | Marginal Value | Source LLMs |
|---|---|---|---|---|---|---|
| **BGE-M3** | BAAI/bge-m3 | 1024 dense+sparse | ✅ 100+ | 8192 tokens | **High** — full lyrics, multilingual | 7/8 |
| **multilingual-E5-large** | intfloat/multilingual-e5-large | 1024 | ✅ 94 langs | 512 tokens | High — needs chunking | 4/8 |
| **GTE-multilingual-base** | Alibaba-NLP/gte-multilingual-base | 768 | ✅ 75+ | **8192 tokens** | High — long-context, Apache 2.0 | 3/8 |
| **Jina-embeddings-v3** | jinaai/jina-embeddings-v3 | 1024 | ✅ 89 | Long-context | High quality but **CC-BY-NC-4.0** | 3/8 |
| **Nomic Embed v1.5** | nomic-ai/nomic-embed-text-v1.5 | 768 (Matryoshka) | ⚠️ English-focused | 8192 tokens | High for English | 2/8 |
| **LaBSE** | sentence-transformers/LaBSE | 768 | ✅ 109 | 512 tokens | Medium-High; Apache 2.0 | 3/8 |
| **multilingual-E5-base** | intfloat/multilingual-e5-base | 768 | ✅ 100+ | 512 tokens | Medium — lighter E5 | 2/8 |
| **CLaMP3 text branch** | sanderwood/clamp3 | 768 | ✅ 95 | — | High — music-aware cross-modal | 1/8 |
| **all-mpnet-base-v2** (existing) | sentence-transformers | 768 | ❌ English | 512 tokens | Baseline — English, truncates | Existing |
| **all-MiniLM-L6-v2** (existing) | sentence-transformers | 384 | ❌ English | 512 | Baseline | Existing |
| **NRC EmoLex** | NRCLex (pip) | 10 | ✅ 108 | word-level | High for mood | 6/8 |
| **MEmoLon VAD** | JULIELab/MEmoLon | 3 | ✅ 91 | word-level | High — Valence/Arousal/Dominance | 2/8 |
| **VADER** | vaderSentiment | 4 | ❌ English | sentence | Low-Medium — English only | 5/8 |
| **GoEmotions RoBERTa** | SamLowe/roberta-base-go_emotions | 28 | ❌ English | 512 | High for English — 28 emotions | 3/8 |
| **BERTopic** | BERTopic + embeddings | 16–32 probs | ✅ via embeds | N/A | Medium — automated themes | 5/8 |
| **fasttext lang-ID** | fasttext lid.176.bin | 1+conf | ✅ 176 | N/A | High — prerequisite | 5/8 |
| **Lexical richness** | lexicalrichness | 4–8 | ✅ | N/A | High — vocab sophistication | 4/8 |
| **Stylistic/poetic features** | pronouncing + custom | 10–20 | ⚠️ English rhyme | N/A | Medium — interpretable | 6/8 |
| **YAKE keywords** | yake | 5–10 | ✅ multilingual | N/A | Medium-Low — topic keywords | 4/8 |
| **Local LLM annotation** | Qwen2/Llama Q4 | structured tags | ✅ | — | High quality but slow | 4/8 |

### 3.2 Lyric Model Decision Guide

**BGE-M3 is the consensus first choice (7/8 LLMs)**

| Priority | Model | Why | Chunking? | Notes |
|---|---|---|---|---|
| **1st** | BGE-M3 | MIT, 8192 tokens (no truncation), multilingual, retrieval-optimized | No | pip install FlagEmbedding |
| **2nd** | GTE-multilingual-base | Apache 2.0, 8192 tokens, 768-D, lighter | No | Excellent speed/quality tradeoff |
| **3rd** | multilingual-E5-large | MIT, 1024-D, 94 langs, strong MTEB | Yes (512 tok) | Chunk lyrics, mean-pool chunks |
| **Avoid for Kaggle** | jina-embeddings-v3 | CC-BY-NC-4.0 restricts redistribution | No | Good model, wrong license |
| **English backup** | Nomic v1.5 | Apache 2.0, Matryoshka dims, 8192 | No | English subset analysis |

### 3.3 Interpretable Lyric Features (CPU-only, minutes)

| Feature | Dims | Tool | Multilingual |
|---|---|---|---|
| Language detection | 1+conf | fasttext | ✅ 176 langs |
| Lyric structure (chorus count, verse count, structural entropy) | 5 | regex | ✅ |
| Repetition score (repeated line fraction) | 3–4 | difflib | ✅ |
| Line stats (count, avg length, std) | 4–6 | pure Python | ✅ |
| Lexical richness (TTR, MTLD, HD-D, Hapax ratio) | 4–8 | lexicalrichness | ✅ |
| Readability scores | 4–6 | textstat | ⚠️ English best |
| Rhyme density | 2–4 | pronouncing (CMUdict) | ⚠️ English |
| Pronoun ratios (1st/2nd/3rd person) | 4–6 | spaCy / regex | ⚠️ English NER |
| Profanity/explicit density | 2 | wordlists | ✅ |
| YAKE top-5 keywords | 5–10 | yake | ✅ multilingual |
| NRC EmoLex 8 emotions + 2 sentiment | 10 | NRCLex | ✅ 108 langs |
| MEmoLon Valence/Arousal/Dominance | 3 | MEmoLon | ✅ 91 langs |
| VADER compound+components | 4 | vaderSentiment | English |

### 3.4 Lyric Preprocessing Pipeline (Critical Prerequisite)

```python
import re, unicodedata

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    # Remove [Verse 1], [Chorus], [Bridge] etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove scraping artifacts
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
```

**Missing lyrics (203 songs):** Zero vector for embeddings; `has_lyrics=False` flag; NaN for stats. Do not impute.

---

## 4. Metadata, Derived & Graph Features

| Feature | Derivation | Dims | Leakage Risk |
|---|---|---|---|
| Release decade | year → 10-year bin | 1 | None |
| Log-transformed followers | log(followers + 1) | 1 | ⚠️ Exclude when predicting popularity |
| Language (from fasttext) | fasttext lid | 1 | None — critical for stratification |
| Artist genre complexity | len(artist_genres list) | 1 | None |
| Song duplicates/near-dups | Chromaprint + cosine sim | 1 flag | None |
| Cross-modal mismatch | |audio_energy - lyric_NRC_positive| | 1–3 | None |
| Audio duration delta | |opus_duration - spotify_duration_ms/1000| | 1 | None — QC |
| Artist collaboration degree | co-occurrence graph centrality | 1–3 | None |

> **Target leakage boundaries:** NEVER use `popularity`, `total_artist_followers`, `avg_artist_popularity`, `rank` as input when predicting popularity. Spotify audio features are OK as inputs for predicting *other* targets (mood, genre), but never themselves.

---

## 5. Web Product Concepts & Discovery Tools

### 5.1 Ranked Web Feature Portfolio

| Rank | Concept | User Problem | Key Inputs | 10k Now | Notes |
|---|---|---|---|---|---|
| **1 ★** | **Interactive Song Map (2D Galaxy)** | Visual exploration | All embeddings → UMAP 2D | ✅ Easy | WebGL scatter (regl-scatterplot) |
| **2 ★** | **Multi-Modal Similar Songs** | "Find songs like X" | CLAP + MERT + BGE-M3 | ✅ Easy | Per-modality top-5 + combined |
| **3 ★** | **Song DNA Page** | Understand any song's fingerprint | All features + neighbors | ✅ Easy | Radar chart + modality neighbor lists |
| **4 ★** | **Natural Language Vibe Search** | "Melancholic Turkish acoustic songs" | CLAP text encoder → ANN | ✅ Medium | Text → CLAP → HNSW retrieval |
| **5 ★** | **Semantic Lyric Quote Finder** | "What song has this lyric theme?" | BGE-M3 lyric embeddings | ✅ Easy | Text → BGE-M3 → ANN retrieval |
| 6 | **Controllable Sliders** | "Like X but calmer" | All + Essentia descriptors | ✅ Medium | α_audio / α_lyric / α_rhythm sliders |
| 7 | **Audio-Lyric Mood Mismatch** | "Happy music + dark lyrics" | Essentia mood + GoEmotions | ✅ Easy | Pre-computed mismatch index |
| 8 | **"Why Similar?" Explainer** | Understand recommendations | All features + SHAP | ✅ Medium | Feature-diff + SHAP bars |
| 9 | **Playlist Flow Generator** | Smooth transitions | kNN graph + BPM/energy | ✅ Medium | Graph traversal with diversity |
| 10 | **Era / Time-Travel Explorer** | Music evolution over decades | Metadata + embeddings | ✅ Easy | Timeline scroll by release year |
| 11 | **Lyric Theme Map** | Explore by lyrical theme | BERTopic + BGE-M3 | ✅ Easy | Topic sunburst; click → songs |
| 12 | **Hidden Gems** | Discover underrated songs | Popularity + similarity | ✅ Easy | Low-popularity near high-popularity |
| 13 | **Cover / Near-Duplicate Explorer** | Find covers, remasters | Chromaprint + chroma cosine | ✅ Easy | Fingerprint cluster view |
| 14 | **Playlist Coherence Diagnostics** | Analyze playlist tightness | All embeddings | ✅ Medium | Mean pairwise similarity histogram |
| 15 | **A/B Similarity Evaluation UI** | Crowdsource lyric judgments | Any embeddings | ✅ Easy | Triplet annotation → CSV |

### 5.2 Critical Architecture Rules (All 8 LLMs Agree)

> **NEVER use 2D UMAP coordinates for retrieval.** UMAP distorts high-dimensional distances. Two songs near in 2D may be dissimilar in high-D. Use HNSW/FAISS for all ANN queries. UMAP coords are for canvas rendering only (WebGL scatter).

### 5.3 Recommended Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Backend | FastAPI (Python 3.11) | Native ML ecosystem; async; easy FAISS |
| ANN (10k) | FAISS IndexFlatIP | Exact cosine, <100ms at 10k |
| ANN (100k+) | FAISS HNSW / HNSWlib | M=16, efConstruction=200, cosine |
| Frontend | Next.js (React) or SvelteKit | Next.js: larger ecosystem; Svelte: lighter |
| Viz | regl-scatterplot (WebGL) | 100k+ points at 60 FPS with hover/click |
| Charts | D3.js or Recharts | Radar, histograms, network graphs |
| Database | SQLite + Parquet | Simple; avoid PostgreSQL until needed |
| Hosting | Vercel (frontend) + Hetzner VPS (API) | Low cost for hobby project |

### 5.4 Data Compaction Strategy

| Strategy | 10k Size | 100k Size | Notes |
|---|---|---|---|
| Full float32 embeddings | ~160–400 MB | ~1.6–4 GB | Master storage; load at startup |
| float16 embeddings | ~80–200 MB | ~800 MB | Serve layer; minimal quality loss |
| PCA 64–256 dims | ~2.5–10 MB | ~25–100 MB | Fit on training split only |
| Precomputed top-50 kNN JSON | ~8–25 MB | ~80–250 MB | Instant lookup, no runtime ANN |
| UMAP 2D coords | ~240 KB | ~2.4 MB | Parquet alongside metadata |

---

## 6. System Architecture, Storage & Kaggle Packaging

### 6.1 Recommended Directory Layout

```
music-10k-features-v1/
├── README.md                           # Dataset card, model cards, extraction details
├── LICENSE                             # CC-BY-4.0 for derived features
├── CHANGELOG.md
├── DATA_DICTIONARY.md                  # Column/feature documentation
├── PROVENANCE.md                       # Sources, dates, model versions, HF hashes
├── metadata/
│   ├── songs.parquet                   # NO raw lyrics, NO audio files
│   ├── artists.parquet
│   └── genre_taxonomy.json
├── features/
│   ├── audio_descriptors.parquet       # Essentia MIR, madmom, librosa (~150 dims)
│   ├── lyric_statistics.parquet        # Lexical richness, repetition, VADER, NRC
│   ├── lyric_emotions.parquet          # GoEmotions, MEmoLon, NRC aggregated
│   ├── lyric_topics.parquet            # BERTopic topic IDs + probabilities
│   ├── language_id.parquet             # fasttext labels + confidence
│   └── quality_control.parquet         # Duration delta, silence, clipping flags
├── embeddings/
│   ├── audio/
│   │   ├── mert_v1_95m_768d.npy        # (10000, 768) float32
│   │   ├── panns_cnn14_2048d.npy       # (10000, 2048) float32
│   │   ├── panns_tags_527d.npy         # (10000, 527) float32 — AudioSet sigmoid
│   │   ├── vggish_128d.npy             # (10000, 128) float32
│   │   ├── mel_statistics_512d.npy     # (10000, 512) float32
│   │   ├── clap_htsat_512d.npy         # (10000, 512) float32 — LAION-CLAP
│   │   ├── essentia_discogs_1280d.npy  # (10000, 1280) float32
│   │   └── dsp_structured_150d.npy     # (10000, ~150) float32 — compact MIR
│   └── lyrics/
│       ├── bge_m3_1024d.npy            # (10000, 1024) float32
│       ├── mpnet_base_v2_768d.npy      # (10000, 768) float32 — baseline
│       └── minilm_v2_384d.npy          # (10000, 384) float32 — baseline
├── similarity/
│   ├── knn_audio_top50.parquet         # Pre-computed top-50 audio neighbors
│   ├── knn_lyrics_top50.parquet        # Pre-computed top-50 lyric neighbors
│   └── umap_2d_combined.parquet        # UMAP 2D coords (visualization only)
├── manifests/
│   ├── extraction_manifest.json        # Per-feature: model, HF commit, date, shape, dtype, checksum
│   ├── embedding_checksums.json        # SHA-256 per .npy
│   └── feature_statistics.json
├── splits/
│   ├── artist_grouped_5fold.parquet    # GroupKFold(5) on artist_id — mandatory
│   └── temporal_split.parquet          # Chronological train/val/test
├── evaluation/
│   └── lyric_similarity_benchmark.json # 500 annotated pairs (if collected)
└── track_ids.npy                       # Master ID alignment — all .npy share this order
```

### 6.2 Redistribution Boundaries

| Asset | Kaggle-Safe? | Risk | Recommendation |
|---|---|---|---|
| Raw lyrics text | ❌ NO | **Critical** — copyright | Never distribute. Features only. |
| Audio files (.webm/.opus) | ❌ NO | **Critical** — YouTube/copyright | Keep local. Never distribute. |
| Audio stems (Demucs output) | ❌ NO | Same copyright | Keep local; stem-derived features only |
| Spotify track_id / artist_id | ⚠️ Medium | Spotify ToS | Use ISRC + MusicBrainz IDs as primary keys |
| ISRC codes | ✅ Safe | Public identifiers | Include as cross-reference |
| Derived audio embeddings | ✅ Generally safe | Check each model license | CLAP/BGE-M3/librosa: OK; MERT/Essentia: CC-BY-NC-SA (research only) |
| Spotify audio features | ⚠️ Medium | API ToS | Include with "research only" disclaimer |

> **Most important legal rule (8/8 LLMs):** NEVER include raw lyric text in any public dataset. Distribute computed embeddings, statistics, and scalars only.

---

## 7. In-Domain Evaluation & Ablation Framework

### 7.1 Lyric Benchmark Landscape

**Direct answer: There is NO universally accepted benchmark for lyric embeddings (8/8 LLMs agree)**

| Benchmark | Task | Language | Size | Availability |
|---|---|---|---|---|
| **LyricSIM** (2023) | Semantic similarity | Spanish | 676 pairs | CC-BY-SA — best direct benchmark, Spanish only |
| **MARBLE** (2023) | Audio representation (18 tasks) | N/A | 12 datasets | Open — for audio models, NO lyric tasks |
| **MTEB / MMTEB** | General text embedding | Multilingual | Comprehensive | Open — indirect proxy |
| **MoodyLyrics** | 4-quadrant emotion | English | ~2,500 songs | Research-only |
| **musiXmatch (MSD BoW)** | Topic/genre | English | 237k | Discontinued — BoW only |
| **DALI v2** | Time-aligned lyrics | Mostly English | ~7,900 | CC-BY-NC-SA |
| **4MuLA** | Genre/emotion/similarity | EN/PT/ES | ~48k | Academic |
| **PMEmo / DEAM** | Arousal/valence regression | Instrumental | 794/1802 | CC-BY-NC 4.0 |
| **CMI-Bench** (ISMIR 2025) | Music instruction following | Multilingual | Comprehensive | Releasing Sept 2025 |
| **PoetryMTEB** | Poetry/lyric retrieval | Multilingual | Comprehensive | Releasing Nov 2025 |

### 7.2 In-Domain Evaluation Protocol

**Build your own lyric benchmark from the 10k dataset (6/8 LLMs recommend)**

**Annotation Design:**
- **100–300 query songs** stratified by language, genre, lyric length
- **10–20 candidate pairs per query:**
  - 2 highly similar (same-artist known positive — control)
  - 2 moderately similar (same genre, different artist)
  - 2 same genre / different theme
  - 2 same theme / different genre
  - 2 uniform random (hard negatives)
- **3 annotators per pair** (blind to artist names)
- **5-point Likert on 4 dimensions:** Thematic/Topic overlap · Emotional valence & tone · Narrative perspective & style · Overall lyrical substitutability
- **Agreement target:** Fleiss κ > 0.65; ICC(2,1) > 0.70

**Retrieval metrics:**
- Primary: nDCG@10 (handles graded relevance)
- Secondary: MRR, MAP@10, Recall@10/50
- Hard constraint: same-artist songs excluded from candidate pool during evaluation

### 7.3 Ablation Matrix

**Mandatory split: `GroupKFold(n_splits=5)` on `artist_id`**

| Ablation ID | Feature Set | Dims | Purpose |
|---|---|---|---|
| **B0** | Spotify metadata only | ~13 | Hard baseline |
| **B1** | B0 + current audio (MERT+PANNs+VGGish+mel) | ~3,469 | Current audio stack |
| **B2** | B1 + MPNet + MiniLM + 5 stats + 2 TextBlob | ~4,628 | **Current full pipeline** |
| **A_pooling** | B1 with mean+std pooling | ×2 base dims | Pooling ablation — do first |
| **A_clap** | B2 + CLAP 512 | +512 | CLAP marginal value |
| **A_essentia** | B2 + Essentia MIR (~200) | +200 | Structured MIR value |
| **A_dsp** | B2 + compact DSP ~150 | +150 | Handcrafted MIR value |
| **L_bgem3** | B1 + BGE-M3 + NRC + stylistic | ~1,200 | Upgraded lyric stack |
| **FULL** | B1 + CLAP + Essentia + BGE-M3 + DSP + emotions | ~6,000–8,000 | All proposed additions |
| **Drop-one** | FULL minus each modality | — | Per-modality contribution |

**Go/no-go gate:** Promote feature if Δ nDCG@10 ≥ 0.03 OR Δ R² ≥ 0.02 AND improvement survives artist-aware splits AND p < 0.05 (1000-iteration bootstrap CI).

**Models:** Ridge (baseline) + CatBoostRegressor + LightGBM/XGBoost. Feature scaling: StandardScaler fit on train only. PCA per feature group: 95% variance, fit on train only.

**Prediction targets:** valence, energy, danceability (direct prediction); popularity (understand only — document selection bias). Leakage: never include target column in feature matrix.

---

## 8. Compute Budget, Hardware Profiling & Risk Register

### 8.1 Feasibility Matrix (GTX 1660 Ti 6 GB VRAM)

**Feasibility tiers:**
- `✅ Feasible – Immediate` → <4 hours, <4 GB VRAM
- `⚠️ Feasible – Chunked/Batched` → 4–15 hours, batch 1–4, fp16
- `⚠️ Pilot Only` → >15 hours or tight 6 GB; run on subset first
- `❌ Infeasible / Reject` → OOM or redundant

| Task | VRAM | Runtime 10k | Verdict |
|---|---|---|---|
| Statistics pooling (existing) | 0 | <30 min | ✅ Feasible – Immediate |
| Lyric clean + lang-ID + NRC | 0 | <30 min | ✅ Feasible – Immediate |
| DSP suite (librosa + essentia) | 0 | 20–30 h CPU | ✅ Feasible – Immediate |
| Chromaprint dedup | 0 | 8–30 min | ✅ Feasible – Immediate |
| LAION-CLAP | ~1.8 GB | 3.5–8 h | ✅ Feasible – Immediate |
| PANNs 527-tag (re-run) | ~1.4 GB | 25–40 min | ✅ Feasible – Immediate |
| Essentia Discogs-EffNet | <0.1 GB | 7–12 min | ✅ Feasible – Immediate |
| Essentia MTG mood/instr | CPU | 3–5 min | ✅ Feasible – Immediate |
| BGE-M3 lyrics (batch 8, fp16) | ~1.9 GB | 6–50 min | ✅ Feasible – Immediate |
| multilingual-E5-large | ~2.2 GB | 17–50 min | ✅ Feasible – Immediate |
| GTE-multilingual-base | ~1.2 GB | 8–15 min | ✅ Feasible – Immediate |
| GoEmotions RoBERTa (English) | ~0.5 GB | 8–30 min | ✅ Feasible – Immediate |
| BERTopic | CPU/GPU | ~15 min total | ✅ Feasible – Immediate |
| Silero VAD | CPU | ~1.5 h | ✅ Feasible – Immediate |
| Basic Pitch (ONNX) | CPU | ~8 h | ✅ Feasible – Immediate |
| MuQ-base (95M) | ~2–3 GB | 1.4–3 h | ✅ Feasible – Immediate |
| BEATs (iter3+AS2M) | ~1.4 GB | 25 min–1.5 h | ✅ Feasible – Immediate |
| EnCodecMAE-base | ~0.5 GB | ~50 min | ✅ Feasible – Immediate |
| MERT-v1-330M (fp16, 30s chunk) | ~3.2 GB | 1.25–22 h | ⚠️ Feasible – Chunked |
| MuQ-MuLan-large | ~4–5.8 GB | 20–56 h | ⚠️ Pilot Only |
| CLaMP3 SAAS | ~4–5.8 GB | 22–56 h | ⚠️ Pilot Only |
| Qwen2-1.5B Q4 (LLM annot.) | 0 GPU | ~28 h CPU | ⚠️ Pilot Only |
| Llama 3.1 8B Q4_K_M (LLM) | 0 GPU | ~83 h CPU | ⚠️ Pilot Only |
| Demucs htdemucs | ~3.1 GB | 35–83 h | ⚠️ Pilot Only |
| Jukebox | 14+ GB | >200 h | ❌ Infeasible / Reject |
| AudioMAE | ~1.5 GB | ~5.5 h | Low priority — skip |
| wav2vec 2.0 / HuBERT | ~1.5 GB | ~6 h | Skip — redundant |

### 8.2 Risk Register

| Risk ID | Category | Threat | Severity | Likelihood | Mitigation |
|---|---|---|---|---|---|
| **R-01** | Legal | Raw lyrics or audio distributed → DMCA | Critical | High | Features/embeddings only; zero raw text/audio |
| **R-02** | Legal | Spotify ToS violation | High | Medium | ISRC + MusicBrainz IDs; research disclaimer |
| **R-03** | Legal | CC-BY-NC model redistribution | High | Medium | Track per model; use Apache 2.0/MIT for Kaggle |
| **R-04** | Leakage | `popularity` as input → R² ≈ 1.0 | Critical | High | Programmatic blacklist; immutable feature groups |
| **R-05** | Leakage | Artist in train and test | Critical | High | GroupKFold on artist_id; Chromaprint dedup |
| **R-06** | Hardware | MERT-330M/CLAP CUDA OOM | High | High | fp16; chunk 10s; batch=1; empty_cache() |
| **R-07** | ML | English-only models on non-English lyrics | Medium | High | fasttext lang-ID; route to multilingual models |
| **R-08** | Data | YouTube ≠ studio version (live/remaster) | Medium | High | Duration delta flag; Chromaprint vs expected |
| **R-09** | Data | Feature matrix misalignment | Critical | Medium | Immutable track_ids.npy; runtime assertion checks |
| **R-10** | ML | Embedding version drift | Medium | Low | Pin HF commit hashes; store checksums |
| **R-11** | Pipeline | Corrupt audio / NaN/Inf outputs | Medium | Medium | Resumable checkpoints; try/except + validation |
| **R-12** | Evaluation | Same-genre trivially separates — overoptimistic | High | High | Per-genre retrieval; cross-genre evaluation |
| **R-13** | Data | Popularity corpus bias — not representative | Info | Certain | Document explicitly; no generalization claims |
| **R-14** | ML | LLM hallucination in lyric annotations | Medium | High | Schema validation; 10% human audit; store as "predicted" |
| **R-15** | Storage | Demucs stems ~1.6 TB | Medium | High | Never store full stems; derive stats only |
| **R-16** | Architecture | UMAP coords used for ANN retrieval | High | Medium | Explicit rule: HNSW/FAISS for all queries |
| **R-17** | Legal | Essentia AGPL-3.0 code redistribution | Low | Low | Use as library only; AGPL affects code, not data |

---

## 9. Master Decision Menu & Tiered Implementation Roadmap

### Tier 0: Data Hygiene & Free Features
**(1–2 days, CPU-only, zero VRAM)**

| # | Task | Tool | Time | Always? |
|---|---|---|---|---|
| 0.1 | Lyric cleaning pipeline | regex + unicodedata | 5 min | ✅ Yes |
| 0.2 | Language identification | fasttext lid.176.bin | 2 min | ✅ Yes |
| 0.3 | Chromaprint fingerprinting + dedup | fpcalc + pyacoustid | 30 min | ✅ Yes |
| 0.4 | Audio duration QC vs Spotify | ffprobe / librosa | 10 min | ✅ Yes |
| 0.5 | Silence/clipping detection | librosa RMS | 1 h | ✅ Yes |
| 0.6 | Lyric structure features | regex + pure Python | 15 min | ✅ Yes |
| 0.7 | Lexical richness suite | lexicalrichness | 5 min | ✅ Yes |
| 0.8 | NRC EmoLex + VADER | NRCLex, vaderSentiment | 5 min | ✅ Yes |
| 0.9 | **Statistics pooling (mean+std+max)** | NumPy (existing embeddings) | 30 min | ✅ Yes |
| 0.10 | YAKE keywords | yake | 10 min | ✅ Yes |
| 0.11 | Metadata-derived features | pandas | 30 min | ✅ Yes |

**Gate:** QC report — flag duplicates, duration mismatches >10s, silence >10%, ≥203 missing lyrics.

```bash
pip install pyloudnorm NRCLex vaderSentiment yake pronouncing lexicalrichness textstat pyacoustid langdetect
```

---

### Tier 1: Core High-ROI Additions
**(Weeks 1–3, GPU/CPU, ~20–60 h total)**

| # | Task | Model | Runtime | Go/No-Go |
|---|---|---|---|---|
| 1.1 | **Compact DSP feature suite (~150 dims)** | librosa + pyloudnorm + madmom | ~20–30 h CPU | Always |
| 1.2 | **LAION-CLAP audio embeddings** | `laion/clap-htsat-unfused` | ~3.5–8 h GPU | Always |
| 1.3 | **PANNs 527 AudioSet tag probs** | Re-run PANNs sigmoid | ~25–40 min GPU | Always |
| 1.4 | **Essentia Discogs-EffNet + mood taggers** | essentia TF models | ~1–2 h CPU | Always |
| 1.5 | **BGE-M3 multilingual lyric embeddings** | BAAI/bge-m3 | ~6–50 min GPU | Always |
| 1.6 | **GoEmotions RoBERTa** (English-detected) | SamLowe/roberta-base-go_emotions | ~8–30 min GPU | Always |
| 1.7 | **BERTopic topic modeling** | BERTopic + BGE-M3 embeds | ~15 min GPU | Always |
| 1.8 | **Silero VAD vocal activity** | silero-vad | ~1.5 h CPU | Always |
| 1.9 | **GTE-multilingual-base OR E5-large** | Compare vs BGE-M3 | ~8–50 min GPU | Compare |

**Gate after Tier 1:**
- CLAP vs MERT retrieval: Spearman ρ < 0.7 → complementary confirmed
- BGE-M3 Recall@10 > MPNet by > 5% on English subset
- Essentia DSP adds R² > +0.02 on energy/danceability

```bash
pip install laion-clap FlagEmbedding sentence-transformers bertopic faiss-cpu umap-learn
pip install essentia  # or essentia-tensorflow for TF models
```

---

### Tier 2: Controlled Pilots
**(Weeks 4–8, 100–500 song trials with clear go/no-go gates)**

| Task | Pilot | Pilot Time | Full Time | Success Gate |
|---|---|---|---|---|
| **MERT-v1-330M extraction** | 200 songs | ~1–2 h GPU | ~15–22 h | Δ nDCG@10 > +0.02 over MERT-95M |
| **MuQ (300M) extraction** | 200 songs | ~1–2 h GPU | ~8–15 h | Pearson r < 0.85 with MERT → new signal |
| **Demucs stem features** | 100–200 songs | ~1–5 h GPU | ~35–83 h | Stem features improve emotion R² by > +0.03 |
| **Multi-window MERT temporal pooling** | 200 songs | ~2 h GPU | ~30 h re-extract | Recall@10 improves > +5% vs mean-pool |
| **Basic Pitch note statistics** | 500 songs | ~1–2 h CPU | ~8 h | r < 0.50 with Spotify danceability/acousticness |
| **LLM lyric annotation (Qwen2-1.5B)** | 200 songs | ~1 h CPU | ~28 h | >80% valid JSON; improves BERTopic coherence |
| **BGE-M3 chunk+section pooling** | 500 songs | ~15 min GPU | — | nDCG@10 > full-doc by > +3% |

---

### Tier 3: Deferred / Cloud GPU
**(Long-term or requires cloud compute)**

| Task | Prerequisite | Why Deferred |
|---|---|---|
| Full Demucs + CREPE on 10k | Successful pilot; cloud GPU | 35–83 h GPU; high storage |
| CLaMP3 SAAS (full 10k) | Verify checkpoint; MuQ pilot results | 4–5.8 GB edge case |
| MuQ-MuLan-large | After MuQ-base pilot | 700M, tight 6 GB |
| Domain LoRA fine-tuning | Validated eval set (500+ annotated pairs) | 10k may be insufficient |
| LLM annotation at 10k scale | Pilot validation | 28–83 h CPU; hallucination |
| E5-mistral-7B | Cloud GPU | 7B params; 14+ GB RAM |
| Jukebox | **Never** | OOM, non-commercial, archived |
| AudioMAE | Low priority | Redundant with PANNs/VGGish |
| wav2vec 2.0 / HuBERT | **Skip** | Speech SSL; redundant with MERT |

---

### Implementation Timeline Estimate

| Phase | Tasks | Wall-Clock Time |
|---|---|---|
| Tier 0 | QC + free features + pooling | 1–2 days |
| Tier 1 | DSP, CLAP, PANNs-527, Essentia, BGE-M3, GoEmotions, BERTopic | 3–5 days |
| Tier 1 Ablation | Feature comparison experiments | 1–2 days |
| Tier 2 Pilots | MERT-330M, MuQ, Demucs pilot, Basic Pitch pilot | 1–2 weeks |
| Tier 2 Full (if pilots pass) | Scale passing pilots to 10k | 1–2 weeks |
| Packaging | Parquet + NPY + manifests + splits + README | 2–3 days |

---

## Appendix: Key References

| Resource | License | Primary Link |
|---|---|---|
| MERT (Li et al., 2023) | CC-BY-NC-SA | arXiv:2306.00107 · HF: m-a-p/MERT-v1-95M |
| LAION-CLAP (Wu et al., 2023) | Apache-2.0 | arXiv:2211.06687 · github.com/LAION-AI/CLAP |
| MuQ (Zhu et al., 2025) | MIT-like | github.com/tencent-ailab/MuQ |
| BGE-M3 (Chen et al., 2024) | MIT | arXiv:2402.03216 · HF: BAAI/bge-m3 |
| Essentia + Discogs-EffNet | AGPL-3.0 + CC-BY-NC-SA | essentia.upf.edu/models.html |
| Madmom (Böck et al., 2016) | BSD 2-Clause (CC-BY-NC-SA models) | github.com/CPJKU/madmom |
| Demucs v4 (Défossez, 2021) | MIT | github.com/facebookresearch/demucs |
| BEATs (Chen et al., 2023) | MIT | github.com/microsoft/unilm/tree/master/beats |
| Basic Pitch (Bittner et al., 2022) | Apache-2.0 | github.com/spotify/basic-pitch |
| multilingual-E5 | MIT | HF: intfloat/multilingual-e5-large |
| GTE-multilingual-base | Apache-2.0 | HF: Alibaba-NLP/gte-multilingual-base |
| Nomic Embed v1.5 | Apache-2.0 | arXiv:2402.01613 · HF: nomic-ai/nomic-embed-text-v1.5 |
| Jina-embeddings-v3 | **CC-BY-NC-4.0** | HF: jinaai/jina-embeddings-v3 |
| MARBLE benchmark | Varies | arXiv:2306.10102 |
| LyricSIM benchmark | CC-BY-SA 4.0 | SEPLN Journal 2023 |
| FAISS | MIT | github.com/facebookresearch/faiss |
| librosa 0.10+ | ISC | librosa.org |
| UMAP | BSD-3 | arXiv:1802.03426 |
| NRC EmoLex | Research free / paid commercial | nrc.canada.ca |
| GoEmotions (2020) | Apache-2.0 | arXiv:2005.00547 · HF: SamLowe/roberta-base-go_emotions |
| BERTopic (2022) | MIT | github.com/MaartenGr/BERTopic |
| Silero VAD | MIT | github.com/snakers4/silero-vad |
| fasttext lang-ID | CC-BY-SA 3.0 | fasttext.cc |
| CLaMP3 | MIT code | github.com/sanderwood/clamp3 |
