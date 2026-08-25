# MIR Feature Extraction & Music Platform — Research Brainstorm Report

**Date:** August 2026 · **Hardware:** GTX 1660 Ti 6 GB / 16 GB RAM · **Dataset:** 10,000 top-streamed songs (Spotify July 2025)

---

## SECTION 1 — Audio Feature Extraction Ideas

### 1.1 Pretrained Audio/Music Embedding Models (Beyond What You Have)

You already have VGGish (128-d), MERT-v1-95M (768-d), PANNs Cnn14 (2048-d), and Mel stats (512-d) = 3,456-d of neural audio features. Here are verified additions ranked by value÷effort:

#### 🏆 Tier 1 — Must-Try (highest value, confirmed feasible)

| # | Model | Params | Dims | VRAM (fp16) | Input SR | License | Source |
|---|-------|--------|------|-------------|----------|---------|--------|
| 1 | **MERT-v1-330M** | 330M | 1024 | ~1.3 GB | 24 kHz | CC-BY-NC-4.0 | [HuggingFace](https://huggingface.co/m-a-p/MERT-v1-330M) |
| 2 | **LAION-CLAP** (htsat-fused) | 154M | 512 | ~0.8 GB | 48 kHz | Apache 2.0 | [HuggingFace](https://huggingface.co/laion/clap-htsat-fused) |
| 3 | **BEATs** (iter3+ AS2M) | ~90M | 768 | ~0.5 GB | 16 kHz | MIT | [GitHub](https://github.com/microsoft/unilm/tree/master/beats) |

**Why these 3?**
- **MERT-330M** is the bigger sibling of your 95M — same API, same pipeline, just swap model ID. 24 transformer layers vs 12, 1024-d vs 768-d. On MARBLE benchmark it's consistently top-3 for genre, mood, key, and beat tasks. The CC-BY-NC license is fine for research/Kaggle but blocks commercial use.
- **LAION-CLAP** gives you *cross-modal* audio↔text embeddings. You can compute cosine similarity between a song's audio embedding and any text query ("upbeat Latin dance") — this is killer for the website's search and exploration features. It's also the only model here that directly handles 48 kHz audio (matching your Opus files). Apache 2.0 = fully permissive.
- **BEATs** is Microsoft's AudioSet SOTA — iterative self-supervised with discrete token prediction. It captures semantic audio events (instruments, environment, mood) differently from MERT's music-specific training. MIT license = maximally permissive.

#### 🥈 Tier 2 — Strong Value, Worth Trying

| # | Model | Params | Dims | VRAM (fp16) | License | Source |
|---|-------|--------|------|-------------|---------|--------|
| 4 | **Music2Vec** | ~95M | 768 | ~0.5 GB | MIT | [HuggingFace](https://huggingface.co/m-a-p/music2vec-v1) |
| 5 | **EnCodecMAE** (base) | ~90M | 768 | ~0.5 GB | MIT | [HuggingFace](https://huggingface.co/lpepino/encodecmae-base) |
| 6 | **data2vec-audio** (base) | ~95M | 768 | ~0.5 GB | MIT | [HuggingFace](https://huggingface.co/facebook/data2vec-audio-base) |
| 7 | **OpenL3** (music, mel256) | ~4.7M | 512 | ~0.1 GB (CPU OK) | MIT | [GitHub](https://github.com/marl/openl3) |

- **Music2Vec** is from the same m-a-p lab as MERT but uses a different SSL objective. Complementary to MERT — captures different aspects of music structure.
- **EnCodecMAE** uses Meta's EnCodec discrete tokens as the MAE reconstruction target — novel training signal that captures timbral/spectral nuance differently from MERT's CQT teacher.
- **data2vec-audio** is Meta's general-purpose audio SSL model. 768-d from a wav2vec2-style architecture but with a unified self-distillation objective.
- **OpenL3** is a tiny CNN trained on AudioSet with L3-Net (audio-visual correspondence). Extremely fast, CPU-friendly, good for a "cheap diverse feature" to complement the transformers.

#### 🥉 Tier 3 — Situational / Lower Priority

| # | Model | Params | Dims | VRAM | License | Notes |
|---|-------|--------|------|------|---------|-------|
| 8 | **AST** (Audio Spectrogram Transformer) | ~87M | 768 | ~0.5 GB | MIT | [GitHub](https://github.com/YuanGongND/ast) — AudioSet-pretrained ViT, overlaps with PANNs |
| 9 | **YAMNet** | ~3.2M | 1024 | CPU OK | Apache 2.0 | [TF Hub](https://tfhub.dev/google/yamnet/1) — MobileNet, TF-only, overlaps with VGGish |
| 10 | **Wav2CLIP** | ~80M | 512 | ~0.4 GB | MIT | [GitHub](https://github.com/descriptinc/lyrebird-wav2clip) — CLIP-aligned, interesting for search |
| 11 | **Whisper encoder** (small) | 244M | 768 | ~1.0 GB | MIT | [HuggingFace](https://huggingface.co/openai/whisper-small) — Speech-optimized, less musical value |

#### ❌ Infeasible on Your Hardware

| Model | Why Not | Workaround |
|-------|---------|------------|
| **Jukebox** (OpenAI) | 5B params, needs 16+ GB VRAM minimum | None practical — skip entirely |
| **E5-mistral-7b** (audio via text) | 7.1B params, 14+ GB RAM | Skip — text-only and too large |
| **MERT-v1-1B** (if released) | Would need 4+ GB VRAM min | Monitor — might work with int8 |

### 1.2 Better Pooling of Existing Embeddings (Cheap Wins)

Your current approach is **temporal mean pooling** over all frames. This discards significant information. Research consistently shows mean pooling is suboptimal for music similarity:

| Pooling Method | Dims Added | Effort | Expected Gain | Source/Evidence |
|----------------|-----------|--------|---------------|-----------------|
| **Statistics pooling** (mean + std) | 2× current | Easy | Moderate | Standard in speaker verification (ECAPA-TDNN); std captures temporal variation — a song that changes a lot vs static will differ. Doubles your dims from 3,456 to 6,912. |
| **Multi-stat pooling** (mean, std, max, min, median, q25, q75) | 7× current | Easy | Good | Used in x-vector systems. Pick mean+std+max for 3× = 10,368-d. |
| **Attention-weighted mean** | Same as base | Medium | Good | Learn a single-layer attention head over frames. Requires ~100 lines of code + a small training signal (genre labels work). Standard in speech tasks. |
| **CLS / first token** (for MERT) | Same as base | Easy | Variable | MERT's paper recommends layer-wise weighted sum. Try extracting the CLS token from layer 11-12 of MERT-330M specifically. |
| **Per-chunk statistics** | ~4× per model | Easy | Moderate | Split the 30s MERT window into 3×10s chunks, compute mean per chunk, concatenate → captures temporal progression (intro vs chorus). |

> **Recommendation**: At minimum, compute **mean + std** for all 4 extractors. This is ~2 hours of NumPy work on already-extracted per-frame data (if you saved raw frame embeddings) or a re-extraction with frame-level output. The expected lift is 5-15% on similarity tasks based on speaker/music retrieval literature.

### 1.3 Classical DSP / Handcrafted Features (CPU-only, Free)

These run on CPU with **librosa** (MIT) and/or **essentia** (AGPLv3 — note the license for commercial use). Ranked by value for your goals:

#### Top 10 Handcrafted Features (ranked)

| Rank | Feature Group | Dims | Library | Value For | Est. Time (10k) |
|------|--------------|------|---------|-----------|-----------------|
| 1 | **MFCC + Δ + ΔΔ** (20 coeffs × 3 × 4 stats) | 240 | librosa | Genre, timbre, similarity | ~3h CPU |
| 2 | **Chroma (CENS)** (12 bins × 4 stats) | 48 | librosa | Key, harmony, covers | ~2h CPU |
| 3 | **Spectral contrast** (7 bands × 4 stats) | 28 | librosa | Mood, genre, texture | ~2h CPU |
| 4 | **Tonnetz** (6 dims × 4 stats) | 24 | librosa | Harmonic content, mood | ~2h CPU |
| 5 | **Tempogram stats** (autocorrelation, 4 stats) | 8 | librosa | Dance, rhythm stability | ~3h CPU |
| 6 | **RMS energy envelope** (4 stats + dynamics) | 8 | librosa | Energy, loudness, dynamics | ~1h CPU |
| 7 | **Spectral centroid/bandwidth/rolloff/flatness** (4 × 4 stats) | 16 | librosa | Brightness, timbre | ~1h CPU |
| 8 | **Zero-crossing rate** (4 stats) | 4 | librosa | Percussiveness, noise | ~0.5h CPU |
| 9 | **Onset strength** (4 stats + onset rate) | 5 | librosa | Rhythmic density | ~2h CPU |
| 10 | **HPSS energy ratio** (harmonic/percussive) | 2 | librosa | Harmonic vs rhythmic character | ~4h CPU |

**Total: ~383 dims, ~20h CPU runtime** (can run overnight).

Additional low-cost features worth including:

| Feature | Dims | Notes |
|---------|------|-------|
| Loudness (LUFS via pyloudnorm) | 1 | Integrated LUFS — gold standard for loudness, correlates with energy |
| Dynamic range (LUFS range) | 1 | Compressed pop vs dynamic classical |
| Bass energy ratio (power < 250 Hz / total) | 1 | Danceability correlate |
| Stereo width (mid-side correlation) | 2 | Production quality proxy |
| Tempo stability (beat intervals std) | 1 | Live vs electronic |

### 1.4 Vocal/Singing Analysis

| Approach | Model | VRAM | Feasibility | Worth It? |
|----------|-------|------|-------------|-----------|
| **Source separation → vocal/instrumental embeddings** | htdemucs | ~4 GB | ✅ Fits with `--segment 8` | **High value** — lets you extract MERT/CLAP separately on vocals and instrumentals |
| Spleeter (2-stem) | Spleeter | ~2 GB | ✅ Easy | Good but lower quality than Demucs |
| Open-Unmix | Open-Unmix | ~0.5 GB | ✅ Very easy | Lowest quality of the three |
| Vocal activity ratio | (from separation) | — | ✅ | Cheap byproduct — % of song with vocals |

> **Verdict**: Running htdemucs to get vocal and instrumental stems, then extracting CLAP or MERT embeddings on each stem separately, would be **one of the highest-value additions** for similarity. "Songs with similar instrumentals but different vocals" is a powerful search axis. However, it's expensive: ~5 min/song × 10k = ~35 days continuous GPU. Consider running on a subset (1,000 songs) or using Spleeter for speed.

### 1.5 Harmony/Chord Extraction

| Tool | What It Does | Output | Feasibility |
|------|-------------|--------|-------------|
| **librosa.feature.chroma_cqt** → chord templates | Match chroma to chord templates | Chord histogram (24-d: 12 major + 12 minor) | Easy, CPU |
| **Chordino** (VAMP plugin) | HMM-based chord recognition | Chord sequence → histogram | Medium (needs VAMP host) |
| **madmom.features.chords** | DNN chord recognition | Chord sequence | Medium (BSD code, CC-BY-NC-SA data) |

Chord complexity features (CPU, ~5 dims): number of unique chords, chord change rate, harmonic rhythm, most common chord ratio, chord entropy.

> **Verdict**: The chroma-based histogram (24-d) + 5 chord complexity features = 29 dims, ~3h CPU. Good value for genre/mood clustering. Not a priority over the neural models.

### 1.6 Semantic/Tagging Embeddings

**Did you waste the PANNs 527-class tag vector?** Yes, partially. The 2048-d penultimate embedding is a "general" representation. The 527-d sigmoid output is an *interpretable* probability vector over AudioSet classes (guitar, crowd cheering, speech, rain, etc.). These two are complementary:

| Approach | Dims | Effort | Value |
|----------|------|--------|-------|
| **PANNs 527-class tag probabilities** | 527 | Easy (1-line code change) | High — directly usable for "what instruments/sounds are in this song?" |
| Top-K tag selection (keep ~50 most relevant music tags) | 50 | Easy | Cleaner, less noise from irrelevant tags (chainsaw, gun, etc.) |
| **CLAP zero-shot tags** (custom music tag vocabulary) | N×1 similarity scores | Medium | You define your own tag list ("electronic beat", "acoustic guitar solo", "rap verse") and get soft labels — very powerful for the website |

### 1.7 Audio Fingerprinting

| Tool | Purpose | Useful for You? |
|------|---------|----------------|
| **Chromaprint/AcoustID** | Exact-match identification → MusicBrainz IDs | ⚠️ Limited — you already have Spotify track_ids. Useful only for dedup/cross-referencing with MusicBrainz. **NOT useful for cover detection** (designed for exact match, not musical similarity). License: LGPL-2.1. |
| Dejavu | Fingerprint + match | ❌ Unmaintained (Python 2 era) |
| Panako | JVM fingerprint | ⚠️ AGPL + patent concerns |

> **Verdict**: Skip fingerprinting. Your Spotify IDs already identify tracks. Use CLAP or chroma for cover/remix detection instead.

### 1.8 Temporal Structure

| Approach | What It Does | Output | Feasibility |
|----------|-------------|--------|-------------|
| **MSAF** (Music Structure Analysis Framework) | Section segmentation (verse/chorus/bridge) | Boundary timestamps + labels | Medium — Python, uses librosa internally. CPU-only. [GitHub](https://github.com/urinieto/msaf) |
| librosa.segment.recurrence_matrix | Self-similarity matrix → novelty curve | Section boundary candidates | Easy, CPU |
| Repetition features (from self-similarity) | Count repeating sections, structural complexity | 3-5 dims | Easy |

Structure features (~5 dims): number of sections, average section length, structural entropy, chorus ratio (if detectable), intro length.

### 1.9 Creative / Niche Features

| Feature | Dims | Notes |
|---------|------|-------|
| **Loudness (LUFS/EBU R128)** | 2 | pyloudnorm library. Integrated + short-term range. Directly comparable across songs. |
| **Dynamic range** | 1 | Max LUFS - min LUFS. Correlates with "wall of sound" vs dynamic range. |
| **Bass weight** (sub-250Hz energy ratio) | 1 | Strong danceability correlate. |
| **Stereo width** (correlation coefficient) | 2 | Mean + std of L-R correlation. Production quality/style indicator. |
| **Tempo stability** | 1 | Std of inter-beat intervals / mean. Live music → high, EDM → low. |
| **Beat strength** (onset envelope autocorrelation peak) | 1 | How prominent the beat is. |
| **Frequency band energy ratios** (sub-bass / bass / mid / upper-mid / presence / brilliance) | 6 | Standard audio engineering bands. Timbral fingerprint. |

---

### 🏆 Section 1 — Ranked Shortlist Table

| Rank | Approach | Dims | Time (10k) | VRAM | License | Feasibility | Value | **Must-Try?** |
|------|----------|------|-----------|------|---------|-------------|-------|---------------|
| 1 | ⭐ **LAION-CLAP embeddings** | 512 | ~8h GPU | 0.8 GB | Apache 2.0 | Easy | ★★★★★ | **YES** |
| 2 | ⭐ **MERT-v1-330M embeddings** | 1024 | ~12h GPU | 1.3 GB | CC-BY-NC-4.0 | Easy | ★★★★★ | **YES** |
| 3 | ⭐ **DSP feature suite** (MFCC+chroma+spectral+rhythm) | ~390 | ~20h CPU | 0 | MIT | Easy | ★★★★☆ | **YES** |
| 4 | **Statistics pooling** (mean+std of existing) | 6,912 | ~4h CPU* | 0 | — | Easy | ★★★★☆ | |
| 5 | **BEATs embeddings** | 768 | ~6h GPU | 0.5 GB | MIT | Easy | ★★★★☆ | |
| 6 | **PANNs 527-tag probabilities** | 527 | ~4h GPU | 0.5 GB | MIT | Easy | ★★★☆☆ | |
| 7 | **Music2Vec embeddings** | 768 | ~6h GPU | 0.5 GB | MIT | Easy | ★★★☆☆ | |
| 8 | **OpenL3 embeddings** | 512 | ~5h CPU | 0 | MIT | Easy | ★★★☆☆ | |
| 9 | **EnCodecMAE embeddings** | 768 | ~6h GPU | 0.5 GB | MIT | Medium | ★★★☆☆ | |
| 10 | **Loudness + dynamics + bass** | 10 | ~3h CPU | 0 | MIT | Easy | ★★★☆☆ | |

*\*Statistics pooling time assumes you have saved per-frame embeddings; otherwise requires re-extraction.*

---

## SECTION 2 — Lyrics Feature Extraction Ideas

### 2.1 The Benchmark Question — Direct Answer

> **Is there a standard benchmark for lyric embeddings / lyric similarity?**

**No, there is no single universally-accepted benchmark equivalent to MTEB for lyrics.** The landscape as of mid-2026:

| Benchmark/Dataset | What It Evaluates | Status | Source |
|-------------------|-------------------|--------|--------|
| **MARBLE** | Audio representations (not lyrics) | Active, 18 tasks | [marble-bm.shef.ac.uk](https://marble-bm.shef.ac.uk) |
| **MTEB** | General text embeddings | Active, not lyrics-specific | [huggingface.co/spaces/mteb](https://huggingface.co/spaces/mteb/leaderboard) |
| **MoodyLyrics** | Lyric → mood classification (valence/arousal) | Dataset exists, no leaderboard | [Research papers](https://dl.acm.org/doi/10.1145/3340555.3353763) |
| **Music4All** | Multimodal (audio + lyrics + metadata) | Dataset exists; lyrics often omitted from public releases due to copyright | [zenodo.org](https://zenodo.org/record/6519485) |
| **LyricSIM** | Lyric semantic similarity (Spanish) | Niche, not widely adopted | Academic papers |

**Key findings:**
- **"In-the-Song" (ITS)** is NOT a recognized standard model/paper. It likely refers to intra-song segment-level representations in some niche work. **No public weights exist.**
- **SongBERT, SongTextBERT, Lyrisong** — **None of these exist as public models with downloadable weights.** They appear to be hypothetical or from unpublished work.
- **Current SOTA for lyric-to-mood**: Fine-tuned XLM-RoBERTa or multilingual E5 embeddings fed into a lightweight classifier. Cross-modal approaches like HeartCLAP (audio+text contrastive) are emerging.

**Bottom line**: Use **MTEB multilingual** rankings as a proxy. For your multilingual lyric corpus, the best approach is a top-performing multilingual sentence embedder + domain-specific evaluation on your own valence/mood/genre prediction tasks.

### 2.2 Text Embedding Models Ranked for Lyrics

Given your requirements (multilingual, 200-500 word texts, 16 GB RAM, need both similarity and prediction):

#### 🏆 Tier 1 — Recommended

| Rank | Model | Dims | Params | RAM | Multilingual | License | Source |
|------|-------|------|--------|-----|-------------|---------|--------|
| 1 | ⭐ **BGE-M3** | 1024 | 567M | ~2.3 GB | 100+ languages | MIT | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| 2 | ⭐ **multilingual-e5-large** | 1024 | 560M | ~2.2 GB | 100+ languages | MIT | [HuggingFace](https://huggingface.co/intfloat/multilingual-e5-large) |
| 3 | ⭐ **jina-embeddings-v3** | 1024 | 570M | ~2.3 GB | 89 languages | CC-BY-NC-4.0 | [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v3) |

**Why these 3?**
- **BGE-M3** is the current production champion for multilingual dense+sparse hybrid retrieval. Supports 100+ languages including all yours (Spanish, Turkish, Korean, Hindi, Portuguese). MIT license. Handles up to 8192 tokens — plenty for full lyrics.
- **multilingual-e5-large** is the strong baseline — consistently top-5 on MTEB multilingual. MIT license. Same compute class.
- **jina-embeddings-v3** has the best MTEB scores per-parameter among multilingual models, but **CC-BY-NC-4.0** limits commercial use.

All three produce 1024-d embeddings and fit comfortably in 16 GB RAM with batch inference. Expected time: ~2-4 hours for 10k lyrics on CPU, ~30 min on GPU.

#### 🥈 Tier 2 — Good Alternatives

| Model | Dims | Params | RAM | Multilingual | License | Notes |
|-------|------|--------|-----|-------------|---------|-------|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | 118M | ~0.5 GB | 50+ languages | Apache 2.0 | Ultra-fast, lower quality. Good for rapid iteration. |
| **LaBSE** | 768 | 471M | ~1.9 GB | 109 languages | Apache 2.0 | Older but broadest language coverage. |
| **multilingual-e5-base** | 768 | 278M | ~1.1 GB | 100+ languages | MIT | Lighter E5 variant, good speed/quality tradeoff. |
| **GTE-multilingual-base** (Alibaba) | 768 | ~305M | ~1.2 GB | 50+ languages | MIT | Recent, strong MTEB scores. |

#### Upgrade from your baseline

Your current **all-mpnet-base-v2** (768-d, English-focused) should be **replaced by BGE-M3 or multilingual-e5-large**. Expected improvement:
- Multilingual lyrics: **large** improvement (MPNet is English-centric, non-English lyrics are poorly embedded)
- English lyrics similarity: ~5-10% improvement (newer training, better architecture)
- Mood prediction: ~5-15% improvement (better semantic capture)

### 2.3 Lexicon/Psycholinguistic Features

| Feature Set | Dims | Multilingual? | License | Library | Value |
|-------------|------|--------------|---------|---------|-------|
| **NRC EmoLex** (8 emotions + 2 sentiment) | 10 | ✅ 108 languages (auto-translated) | Free research, paid commercial | `NRCLex` (pip) | ★★★★☆ — directly predicts valence/arousal |
| **VADER** | 4 | ❌ English only | MIT | `vaderSentiment` | ★★☆☆☆ — skip, use NRC instead |
| **LIWC** | ~90 categories | ❌ Paid only | Commercial ($) | — | ★★★☆☆ — too expensive, not open |
| **TextBlob** (you already have) | 2 | ❌ English only | MIT | `textblob` | ★★☆☆☆ — baseline, replace with NRC |
| **MEmoLon** (valence/arousal/dominance per word) | 3 | ✅ 91 languages | CC-BY-SA | [GitHub](https://github.com/JULIELab/MEmoLon) | ★★★★☆ — best multilingual emotion lexicon |

> **Action**: Replace TextBlob with **NRC EmoLex (10-d) + MEmoLon VAD aggregates (3-d)** = 13 dims of multilingual emotion features. ~30 min CPU for 10k songs.

### 2.4 Stylistic/Poetic Features

| Feature | Dims | Tool | Multilingual? | Notes |
|---------|------|------|--------------|-------|
| Rhyme density | 1 | `pronouncing` (CMUdict) | ❌ English only | Fraction of line-ending rhyme pairs |
| Line count / avg line length / line length variance | 3 | Pure Python | ✅ | Structural proxy |
| Vocabulary sophistication (type-token ratio) | 1 | Pure Python | ✅ | Already partially have (unique_ratio) |
| Repetition ratio (repeated lines / total lines) | 1 | Pure Python | ✅ | Chorus detection proxy |
| Question mark density | 1 | Pure Python | ✅ | Interrogative style |
| Exclamation density | 1 | Pure Python | ✅ | Emotional intensity |
| Pronoun ratios (I/you/we/they) | 4 | spaCy/regex | ⚠️ Needs multilingual NLP | Personal vs universal perspective |

Total: ~12 dims, ~1h CPU. Low effort, moderate value for explainability.

### 2.5 Handling Multilingual + Code-Switched Lyrics

| Strategy | Quality | Speed | Complexity |
|----------|---------|-------|------------|
| **Multilingual embedder directly** (BGE-M3, E5) | ★★★★★ | Fast | Low — recommended |
| Translate-then-embed (Google Translate → English MPNet) | ★★★☆☆ | Slow (API calls) | High — rate limits, cost |
| Character n-grams (language-agnostic) | ★★☆☆☆ | Very fast | Low — poor semantic capture |
| Language detection → per-language model | ★★★★☆ | Medium | High — complex pipeline |

> **Winner**: Use a multilingual embedder directly. BGE-M3 handles code-switching natively because it's trained on multilingual data. No translation needed.

### 2.6 LLM-Based Approaches

| Approach | Model | RAM | Feasibility | Value |
|----------|-------|-----|-------------|-------|
| **BERTopic** (topic modeling) | Any sentence-transformer + HDBSCAN | ~3 GB | ✅ CPU feasible | ★★★☆☆ — gives ~20-50 topic labels per song |
| **KeyBERT** (keyword extraction) | Sentence-transformer | ~2 GB | ✅ CPU feasible | ★★★☆☆ — top-5 keywords per song for search |
| **YAKE** (keyword extraction) | Statistical (no model) | ~0.1 GB | ✅ Very easy | ★★★☆☆ — fast, multilingual, no GPU |
| **Quantized LLM tagging** (Qwen2-1.5B or Phi-3-mini) | 1.5-3.8B Q4 | ~3-5 GB | ✅ Feasible on CPU | ★★★★☆ — generate mood/theme/imagery tags |
| **7-8B LLM tagging** (Llama 3.1 8B Q4) | 8B Q4_K_M | ~5-6 GB RAM | ⚠️ Slow but feasible | ★★★★☆ — higher quality tags |

> **Recommendation**: Use **YAKE** for quick keyword extraction (multilingual, no GPU), then **BERTopic** with BGE-M3 embeddings for topic clusters. If you want richer tags, a quantized **Qwen2-1.5B** via Ollama can generate structured mood/theme annotations at ~5-10 songs/minute on CPU.

### 2.7 Structure-Level Features from Text

Detecting verse/chorus/bridge from plain text (no audio alignment):

| Method | How | Feasibility |
|--------|-----|-------------|
| **Repeated block detection** | Find groups of lines that repeat verbatim → label as chorus | ✅ Easy, pure Python |
| **Line similarity clustering** | Embed each line with MiniLM, cluster → structural segments | Medium |
| **Blank line segmentation** | Many lyrics use blank lines to separate sections | ✅ Trivial |

Output features (~5-8 dims): estimated_chorus_count, chorus_ratio, unique_section_count, structural_entropy, avg_section_length_lines.

---

### 🏆 Section 2 — Ranked Shortlist

| Rank | Approach | Dims | Time (10k CPU) | License | Best For |
|------|----------|------|---------------|---------|----------|
| 1 | ⭐ **BGE-M3 embeddings** | 1024 | ~3h CPU / 30m GPU | MIT | Lyric similarity, mood, Kaggle |
| 2 | ⭐ **multilingual-e5-large embeddings** | 1024 | ~3h CPU / 30m GPU | MIT | Lyric similarity, multilingual |
| 3 | ⭐ **NRC EmoLex + MEmoLon features** | 13 | ~30 min CPU | Research free / CC-BY-SA | Valence/mood prediction |
| 4 | **YAKE keywords + BERTopic topics** | 5 keywords + topic_id | ~2h CPU | MIT / BSD | Kaggle enrichment, search |
| 5 | **Stylistic features** (repetition, structure, line stats) | 12 | ~1h CPU | MIT | Explainability, dataset value |
| 6 | **jina-embeddings-v3** | 1024 | ~3h CPU / 30m GPU | CC-BY-NC | Quality ceiling (non-commercial) |
| 7 | **paraphrase-multilingual-MiniLM** (fast baseline) | 384 | ~30m CPU | Apache 2.0 | Speed, lightweight apps |
| 8 | **Quantized LLM mood/theme tags** | 5-10 tags | ~15h CPU | Varies | Rich annotation, website |

---

## SECTION 3 — Website Tools Brainstorm

### 3.1 Tool Ideas (Ranked by User Value)

| Rank | Tool Name | Description | Required Data/Features | Feasibility | User Value |
|------|-----------|-------------|----------------------|-------------|------------|
| **1** | 🗺️ **Song Map** | Interactive 2D/3D UMAP scatter plot — zoom, click, color by genre/mood/era | All embeddings → UMAP 2D/3D | ★★★★★ | ★★★★★ |
| **2** | 🔍 **Similar Songs Finder** | "Find songs like X" with modality selector (audio / lyrics / both) | All embeddings + ANN index | ★★★★★ | ★★★★★ |
| **3** | 🧬 **Song DNA Page** | Per-track page: radar chart (energy/valence/dance/…), top-5 neighbors per modality, genre tags, lyrics snippet | All features | ★★★★☆ | ★★★★★ |
| **4** | 🎯 **Playlist Generator** | Seed song + feature sliders (energy, valence, tempo) → filtered ranked list | Metadata + embeddings | ★★★★☆ | ★★★★☆ |
| **5** | 💬 **Lyric Search** | Semantic search: "songs about heartbreak in the rain" → ranked lyrics | BGE-M3 lyric embeddings | ★★★★☆ | ★★★★☆ |
| 6 | 🎭 **Mood Explorer** | 2D valence × energy scatter, click to play/explore quadrants | Valence, energy values | ★★★★★ | ★★★★☆ |
| 7 | 🔬 **"Why Similar?" Explainer** | Side-by-side comparison showing feature contributions to similarity score | Feature vectors + SHAP-like attribution | ★★★☆☆ | ★★★★☆ |
| 8 | 📊 **Comparison Tool** | Side-by-side "Song DNA" radar for 2-4 songs | Audio features | ★★★★★ | ★★★☆☆ |
| 9 | 🎲 **Daily Discovery** | Random underrated song (low popularity, high feature interest) | Popularity + novelty score | ★★★★★ | ★★★☆☆ |
| 10 | 🕰️ **Era Explorer** | Timeline view (release_date axis) with genre/mood coloring | Metadata | ★★★★★ | ★★★☆☆ |
| 11 | 📻 **Embedding Radio** | Walk smoothly through embedding space — "radio" that plays song-by-song along a path | Embeddings + TSP-like path | ★★★☆☆ | ★★★★☆ |
| 12 | 🎯 **Genre Explorer** | Hierarchical genre sunburst + artist similarity network | Genre labels + artist embeddings | ★★★★☆ | ★★★☆☆ |
| 13 | 🎮 **Guess the Song** | Quiz: hear 5s audio snippet or see lyrics → guess title/artist | Audio files + lyrics | ★★★★☆ | ★★★☆☆ |
| 14 | 🌐 **Translation Tool** | Show lyrics with auto-translation for non-English songs | Lyrics + translation API | ★★☆☆☆ | ★★★☆☆ |
| 15 | 🧑‍🎤 **Artist Similarity** | "Artists like X" based on aggregated song features | Per-artist mean embeddings | ★★★★☆ | ★★★☆☆ |
| 16 | 🎵 **Audio Feature Annotator** | 30s audio player with synchronized feature timeline (energy, spectral centroid over time) | Per-frame features | ★★☆☆☆ | ★★★☆☆ |
| 17 | 💡 **Hidden Gems** | Songs with unusual feature profiles (outlier detection) | All features | ★★★★☆ | ★★☆☆☆ |
| 18 | 🎤 **Karaoke Mode** | Display lyrics in time (approximate sync from audio structure analysis) | Lyrics + structure | ★☆☆☆☆ | ★★★☆☆ |

#### Top 5 to Build First (in order)
1. **Song Map** — the "wow" feature, most visually impressive, drives engagement
2. **Similar Songs Finder** — core utility, what users will use most
3. **Song DNA Page** — per-track detail, encourages browsing
4. **Lyric Search** — unique differentiator vs other music sites
5. **Playlist Generator** — actionable output users can export

### 3.2 Data Compaction & Scaling Strategy

| Technique | What It Does | Memory Estimate (10k songs) | Scaling to 100k |
|-----------|-------------|----------------------------|-----------------|
| **PCA to 256-d** per modality | Reduce ~4000-d total to ~1000-d | ~40 MB (float32) | ~400 MB |
| **UMAP 2D/3D** | Pre-compute for visualization | ~240 KB | ~2.4 MB |
| **int8 quantization** | Compress embeddings 4× | ~10 MB for 1000-d × 10k | ~100 MB |
| **FAISS IndexFlatL2** | Exact NN search | ~160 MB (4000-d × 10k × float32) | ~1.6 GB |
| **FAISS IndexHNSW** | Approximate NN (fast) | ~163 MB | ~1.63 GB |
| **Precomputed k-NN graph** (k=50) | Sparse similarity for web serving | ~8 MB (50 neighbors × 10k × 8 bytes) | ~80 MB |

**For 10k songs, exact search (IndexFlatL2) is fine** — 160 MB fits in any browser/server RAM. HNSW or IVF only needed at 100k+.

**Recommendation**: Pre-compute the top-50 neighbors per song per modality (audio, lyrics, combined) and ship as a JSON/parquet file. The website can then do instant lookups without any ANN engine at runtime. Total: ~25 MB for 3 modalities × 50 neighbors × 10k songs.

### 3.3 Recommended Tech Stack

| Layer | Recommendation | Why |
|-------|---------------|-----|
| **Backend** | **FastAPI** (Python) | You already know Python; async; serves pre-computed data; easy FAISS integration |
| **Frontend** | **Next.js** (React) | SEO-friendly SSR, excellent DX, massive ecosystem. Alternatively: **Svelte** if you prefer lighter weight |
| **Embedding Viz** | **regl-scatterplot** + UMAP coordinates | WebGL-accelerated, handles 100k+ points with hover/click/lasso, better than scatter-gl |
| **ANN Engine** | **FAISS** (for pre-computation) → ship pre-computed k-NN JSON | No runtime ANN needed for 10k |
| **Charts** | **D3.js** or **Recharts** (React) | Radar charts, histograms, timelines |
| **Audio Player** | **Howler.js** or HTML5 `<audio>` | Standard, lightweight |
| **Hosting** | **Vercel** (frontend) + **Hugging Face Spaces** or **Railway** (API) | Free tiers sufficient for hobby project |
| **PWA** | Feasible — pre-cache pre-computed neighbor data + UMAP coords | Ship ~30 MB of static JSON, works offline |

### 3.4 ⚠️ Legal/Licensing Red Flags for Kaggle

> [!CAUTION]
> **This is the most critical section. Getting this wrong = account ban or DMCA.**

| Data Type | Can You Publish on Kaggle? | Risk Level | Recommendation |
|-----------|---------------------------|------------|----------------|
| **Spotify track_id, album_id, artist_ids** | ⚠️ **Technically violates Spotify ToS** (no database creation from API data). However, many popular Kaggle datasets do include these. | **Medium** | Include with disclaimer "for research purposes only." Many precedents exist on Kaggle. |
| **Spotify audio features** (danceability, energy, etc.) | ⚠️ **Same ToS concern** — these are API-derived | **Medium** | Same as above. These are the most valuable columns for ML researchers. |
| **Popularity scores** | ⚠️ API-derived, changes over time | **Low** | Include with a snapshot date |
| **ISRC codes** | ✅ **ISRCs are industry identifiers**, not copyrighted content | **Low** | Safe to include |
| **Full song lyrics** | ❌ **ABSOLUTELY NOT** — lyrics are copyrighted literary works. Genius/Musixmatch ToS prohibit redistribution. | **CRITICAL** | **Do NOT include full lyrics.** Alternatives: (1) publish lyrics *embeddings* only, (2) publish bag-of-words/TF-IDF sparse matrices (as musiXmatch does), (3) include only the *features* derived from lyrics (sentiment, word count, topic, etc.), (4) provide a script that users run to fetch their own lyrics. |
| **Derived audio embeddings** (MERT, CLAP, PANNs vectors) | ✅ **Safe** — these are transformed representations, not the original audio | **Low** | This is your primary Kaggle value. Ship the feature vectors. |
| **MusicBrainz IDs** | ✅ **CC0 (Public Domain)** | **None** | Include as cross-reference. Use `isrc` to look up MusicBrainz Recording IDs. |
| **Artist metadata** (followers, genres) | ⚠️ API-derived | **Medium** | Include with research disclaimer |

**Recommended Kaggle Dataset Structure:**
```
music-10k-features/
├── metadata.parquet          # rank, track_name, artist_names, release_date, duration, key, tempo, time_sig
├── spotify_features.parquet  # danceability, energy, valence, etc. (with research disclaimer)
├── audio_embeddings/
│   ├── mert_95m_768d.npy
│   ├── mert_330m_1024d.npy
│   ├── clap_512d.npy
│   ├── panns_2048d.npy
│   ├── beats_768d.npy
│   ├── vggish_128d.npy
│   ├── mel_stats_512d.npy
│   └── dsp_features_390d.npy
├── lyric_embeddings/
│   ├── bge_m3_1024d.npy
│   ├── emotion_features_13d.npy
│   └── text_stats_12d.npy
├── ids.parquet               # track_id, isrc, musicbrainz_id (for cross-referencing)
├── splits/                   # artist-aware train/val/test splits
├── README.md                 # Feature documentation, dimensions, extraction details
└── fetch_lyrics.py           # Script for users to fetch their own lyrics (not included)
```

---

## SECTION 4 — Hardware Feasibility Matrix

### Complete Table: Every Approach on GTX 1660 Ti (6 GB) / 16 GB RAM

| Approach | Dims | VRAM Est. | RAM Est. | Time/Song (1660 Ti) | Total (10k) | License | Verdict |
|----------|------|-----------|----------|---------------------|-------------|---------|---------|
| **AUDIO — Neural Models** | | | | | | | |
| MERT-v1-330M | 1024 | ~1.3 GB | ~3 GB | ~4 s | ~12 h | CC-BY-NC-4.0 | ✅ Fits |
| LAION-CLAP (htsat-fused) | 512 | ~0.8 GB | ~2 GB | ~3 s | ~8 h | Apache 2.0 | ✅ Fits |
| BEATs (iter3+AS2M) | 768 | ~0.5 GB | ~1.5 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| Music2Vec | 768 | ~0.5 GB | ~1.5 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| EnCodecMAE (base) | 768 | ~0.5 GB | ~1.5 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| data2vec-audio (base) | 768 | ~0.5 GB | ~1.5 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| OpenL3 (music, 512) | 512 | ~0.1 GB | ~0.5 GB | ~1 s | ~3 h | MIT | ✅ CPU OK |
| AST | 768 | ~0.5 GB | ~1.5 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| YAMNet | 1024 | CPU only | ~0.5 GB | ~0.5 s | ~1.5 h | Apache 2.0 | ✅ CPU OK (TF) |
| Wav2CLIP | 512 | ~0.4 GB | ~1 GB | ~2 s | ~6 h | MIT | ✅ Fits |
| Whisper-small encoder | 768 | ~1.0 GB | ~2 GB | ~3 s | ~8 h | MIT | ✅ Fits |
| Jukebox | — | 16+ GB | 32+ GB | — | — | MIT | ❌ Skip |
| **AUDIO — Source Separation** | | | | | | | |
| htdemucs (--segment 8) | Stems | ~4 GB | ~4 GB | ~30 s | ~83 h | MIT | ⚠️ Tight but works |
| Spleeter (2-stem) | Stems | ~2 GB | ~2 GB | ~10 s | ~28 h | MIT | ✅ Fits |
| Open-Unmix | Stems | ~0.5 GB | ~1 GB | ~15 s | ~42 h | MIT | ✅ Fits |
| **AUDIO — DSP/Handcrafted** | | | | | | | |
| MFCC + Δ + ΔΔ (stats) | 240 | 0 | ~2 GB | ~1 s | ~3 h | MIT (librosa) | ✅ CPU |
| Chroma CENS (stats) | 48 | 0 | ~2 GB | ~0.8 s | ~2 h | MIT (librosa) | ✅ CPU |
| Spectral contrast (stats) | 28 | 0 | ~2 GB | ~0.8 s | ~2 h | MIT (librosa) | ✅ CPU |
| Tonnetz (stats) | 24 | 0 | ~2 GB | ~0.8 s | ~2 h | MIT (librosa) | ✅ CPU |
| Tempogram stats | 8 | 0 | ~2 GB | ~1 s | ~3 h | MIT (librosa) | ✅ CPU |
| RMS / ZCR / spectral (stats) | 28 | 0 | ~2 GB | ~0.5 s | ~1.5 h | MIT (librosa) | ✅ CPU |
| LUFS loudness | 2 | 0 | ~0.5 GB | ~0.3 s | ~1 h | MIT (pyloudnorm) | ✅ CPU |
| HPSS energy ratio | 2 | 0 | ~3 GB | ~1.5 s | ~4 h | MIT (librosa) | ✅ CPU |
| Chord histogram (chroma template) | 29 | 0 | ~2 GB | ~1 s | ~3 h | MIT (librosa) | ✅ CPU |
| **AUDIO — Other** | | | | | | | |
| Statistics pooling (existing) | 2×existing | 0 | ~1 GB | ~0.1 s | ~20 min* | — | ✅ Trivial |
| PANNs 527-tag probs | 527 | ~0.5 GB | ~1.5 GB | ~1.5 s | ~4 h | MIT | ✅ Fits |
| MSAF structure | ~5 | 0 | ~2 GB | ~10 s | ~28 h | MIT | ✅ CPU (slow) |
| Chromaprint | fingerprint | 0 | ~0.2 GB | ~0.5 s | ~1.5 h | LGPL-2.1 | ✅ CPU |
| **LYRICS — Embeddings** | | | | | | | |
| BGE-M3 | 1024 | ~2.3 GB | ~3 GB | ~0.3 s | ~50 min GPU | MIT | ✅ Fits |
| multilingual-e5-large | 1024 | ~2.2 GB | ~3 GB | ~0.3 s | ~50 min GPU | MIT | ✅ Fits |
| jina-embeddings-v3 | 1024 | ~2.3 GB | ~3 GB | ~0.3 s | ~50 min GPU | CC-BY-NC-4.0 | ✅ Fits |
| paraphrase-multilingual-MiniLM | 384 | ~0.5 GB | ~1 GB | ~0.1 s | ~15 min GPU | Apache 2.0 | ✅ Fits |
| LaBSE | 768 | ~1.9 GB | ~2.5 GB | ~0.2 s | ~30 min GPU | Apache 2.0 | ✅ Fits |
| **LYRICS — Lexicon/Features** | | | | | | | |
| NRC EmoLex (8 emotions + 2 sent) | 10 | 0 | ~0.5 GB | ~0.01 s | ~2 min | Research free | ✅ CPU |
| MEmoLon (VAD per word) | 3 | 0 | ~0.5 GB | ~0.01 s | ~2 min | CC-BY-SA | ✅ CPU |
| Text statistics | 5 | 0 | ~0.1 GB | ~0.001 s | ~10 s | — | ✅ CPU |
| Stylistic features | 12 | 0 | ~0.5 GB | ~0.05 s | ~8 min | MIT | ✅ CPU |
| YAKE keywords | 5 per song | 0 | ~0.2 GB | ~0.05 s | ~8 min | MIT | ✅ CPU |
| **LYRICS — LLM** | | | | | | | |
| BERTopic (with MiniLM) | topic_id | 0 GPU | ~4 GB | ~0.5 s | ~1.5 h CPU | BSD | ✅ CPU |
| Qwen2-1.5B (Q4, tagging) | tags | 0 GPU | ~3 GB | ~10 s | ~28 h CPU | Apache 2.0 | ⚠️ Slow |
| Llama 3.1 8B (Q4_K_M) | tags | 0 GPU | ~6 GB | ~30 s | ~83 h CPU | Llama 3.1 Community | ⚠️ Very slow |

*\*Statistics pooling assumes per-frame embeddings are already saved on disk.*

---

## SECTION 5 — Feature-Set Comparison Methodology

### 5.1 Task Set

#### A. Supervised Prediction (R² / classification accuracy)

| Task | Target | Type | Expected Difficulty | Baseline (your prior) |
|------|--------|------|--------------------|-----------------------|
| Valence prediction | Spotify `valence` | Regression (R²) | Medium | R² = 0.72 |
| Energy prediction | Spotify `energy` | Regression (R²) | Easy | R² = 0.92 |
| Danceability prediction | Spotify `danceability` | Regression (R²) | Medium | R² = 0.79 |
| Popularity prediction | Spotify `popularity` | Regression (R²) | Hard | R² = 0.13 |
| Genre classification | `main_genres` | Multi-class accuracy | Medium | — |
| Explicit content | `explicit` | Binary classification (F1) | Easy | — |

#### B. Similarity/Retrieval Quality (unsupervised proxies)

| Metric | How to Compute | What It Measures |
|--------|---------------|------------------|
| **Genre cluster purity** | k-NN (k=10) → % neighbors sharing same `main_genre` | Whether similar songs are same genre |
| **Mood neighborhood agreement** | k-NN → mean absolute difference of `valence` in neighborhood | Whether similar songs have similar mood |
| **Tempo agreement** | k-NN → % neighbors within ±10 BPM | Whether rhythm similarity is captured |
| **Artist separation** | Mean intra-artist distance vs inter-artist distance ratio | Whether same-artist songs cluster together |
| **Human evaluation** (50-100 triplets) | "Which of B or C is more similar to A?" → agreement rate | Ground truth quality check |

#### C. Dataset Quality Heuristics

| Heuristic | What to Check |
|-----------|--------------|
| Pairwise distance distribution | Should be roughly Gaussian, not degenerate (all same distance) |
| Modality agreement | Do audio-similar pairs also tend to be lyrics-similar? Compute rank correlation |
| Feature correlation | PCA explained variance — do features have redundancy? |
| Outlier detection | Isolation Forest on each feature set — flag extraction errors |

### 5.2 Experimental Protocol

```
1. SPLIT (once, reuse everywhere):
   - Artist-aware stratified split: 70/15/15 train/val/test
   - NO song from the same artist appears in both train and test
   - Stratify by main_genre to ensure genre balance across splits
   
2. MODELS (fixed, no tuning per feature set):
   - CatBoost: depth=6, lr=0.05, 1000 rounds, early stopping on val
   - Simple MLP: [input → 512 → 256 → 128 → output], dropout=0.3, AdamW
   - Both trained identically across all feature ablations
   
3. ABLATION GROUPS:
   a) Audio neural only (MERT + PANNs + VGGish + CLAP + BEATs)
   b) Audio DSP only (MFCC + chroma + spectral + rhythm)
   c) Audio all (a + b)
   d) Lyrics embeddings only (BGE-M3)
   e) Lyrics features only (emotion + stylistic + stats)
   f) Lyrics all (d + e)
   g) Multimodal (c + f + metadata)
   h) Per-model ablation: drop one model at a time from (g)
   
4. METRICS: Report mean ± std over 3 random seeds
```

### 5.3 Pitfalls to Avoid

| Pitfall | Mitigation |
|---------|------------|
| **Artist leakage** | MUST use artist-aware splits. Same artist's songs share style/timbre → artificially inflates metrics |
| **Genre confounding** | Report per-genre results, not just global R². Popularity varies hugely by genre. |
| **Test set reuse** | Lock the test split. Use val for ablation decisions. Report test only once per final model. |
| **Feature scaling** | StandardScaler fit on train only, transform val/test |
| **High dimensionality** | With ~5000+ features and only 10k samples, regularization is critical. CatBoost handles this; MLP needs dropout + weight decay |
| **Correlated features** | Many neural embeddings will be correlated. PCA or feature selection before MLP may help |
| **Popularity is external** | Popularity depends on marketing, artist fame, playlist placement — not just audio/lyrics. Don't over-optimize on it; report as a sanity check. |

### 5.4 Cost-Aware Feature Ranking

| Feature Set | Extraction Cost | Expected R² Gain (valence) | Priority |
|-------------|----------------|---------------------------|----------|
| Statistics pooling (existing) | ~20 min | +0.02-0.05 | **Do first** |
| DSP suite (CPU) | ~20 h | +0.03-0.08 | **Do first** |
| CLAP embeddings | ~8 h GPU | +0.05-0.10 | **Do second** |
| MERT-330M | ~12 h GPU | +0.03-0.07 | **Do second** |
| BGE-M3 lyrics | ~1 h GPU | +0.05-0.12 (for valence) | **Do second** |
| NRC EmoLex | ~2 min | +0.02-0.05 | **Do first** |
| BEATs | ~6 h GPU | +0.02-0.05 | Do third |
| Source separation | ~35-83 h GPU | +0.03-0.08 | Only if results justify |

---

## SECTION 6 — Final Action Plan

### Phase A — This Week (CPU/Lightweight, ~2 days)

**Day 1-2: CPU Feature Extraction + Cheap Wins**

| Task | Time | What |
|------|------|------|
| Statistics pooling | 2-4h | If per-frame embeddings saved: compute mean+std for all 4 extractors → 6,912-d. If not: queue for re-extraction in Phase B |
| DSP feature suite | 20h (overnight) | Extract MFCC+Δ+ΔΔ, chroma, tonnetz, spectral contrast/centroid/bandwidth/rolloff/flatness, ZCR, RMS, tempogram, onset rate, HPSS ratio → ~390-d |
| LUFS loudness | 1h | `pyloudnorm` integrated LUFS + range → 2-d |
| Bass/stereo/dynamics | 1h | Simple spectral band ratios + correlation → ~10-d |
| NRC EmoLex | 30 min | Multilingual emotion features → 10-d |
| Text statistics | 10 min | Already have 5; add line count, repetition ratio, structure features → +12-d |
| YAKE keywords | 30 min | Top-5 keywords per song for Kaggle enrichment |

**Install (Phase A):**
```bash
pip install pyloudnorm essentia NRCLex yake pronouncing
```

### Phase B — Next 1-2 Weeks (GPU, Heavier)

| Priority | Task | Time | What |
|----------|------|------|------|
| B1 | **LAION-CLAP extraction** | ~8h GPU | 512-d audio-text embeddings. Use `laion/clap-htsat-fused` from transformers. |
| B2 | **MERT-v1-330M extraction** | ~12h GPU | 1024-d embeddings. Same pipeline as your 95M script, change model_id. |
| B3 | **BGE-M3 lyrics extraction** | ~1h GPU | 1024-d multilingual lyric embeddings. Replace MPNet. |
| B4 | **BEATs extraction** | ~6h GPU | 768-d audio embeddings. Requires loading from GitHub checkpoint. |
| B5 | **PANNs 527-tag extraction** | ~4h GPU | Re-run PANNs, save the sigmoid output layer instead of penultimate. |
| B6 | **multilingual-e5-large** | ~1h GPU | 1024-d alternative lyric embeddings for comparison |

**Install (Phase B):**
```bash
pip install laion-clap faiss-cpu umap-learn bertopic keybert
# BEATs: clone https://github.com/microsoft/unilm, install manually
# MERT-330M: pip install transformers (already have)
# BGE-M3: pip install FlagEmbedding (or use sentence-transformers)
```

**Scheduling**: Run GPU jobs sequentially overnight. CLAP (night 1) → MERT-330M (night 2) → BEATs (night 3) → PANNs tags (night 4). Lyrics models during daytime (fast).

### Phase C — Compare + Package (Week 3-4)

| Task | Time | What |
|------|------|------|
| Run ablation suite | 2-3 days | Per Section 5 protocol. CatBoost + MLP across all ablation groups. |
| Select winners | 1 day | Identify which feature sets are complementary vs redundant |
| Package for Kaggle | 2-3 days | Create parquet/npy files, write README, generate splits, add fetch_lyrics.py script |
| UMAP visualization | 1 day | Generate 2D/3D embeddings for website |
| Pre-compute k-NN | 1 day | Top-50 neighbors per modality for website |

### Top 5 Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | **OOM during MERT-330M extraction** | High | Use `--max_length 30s`, fp16, batch_size=1. Monitor with `nvidia-smi`. Your 95M script already handles this — 330M uses ~2.5× VRAM but still fits. |
| 2 | **Lyrics copyright on Kaggle** | Critical | NEVER publish full lyrics. Ship embeddings + features + fetch script only. Add clear disclaimer. |
| 3 | **Multilingual embedding quality** | Medium | Validate on a 50-song manually-annotated subset. If BGE-M3 fails on Turkish/Korean lyrics, fall back to language-specific models or translate-then-embed. |
| 4 | **Overfitting to genre confounds** | Medium | Artist-aware splits. Report per-genre metrics. Check if "valence prediction" is really just "genre prediction in disguise." |
| 5 | **Spotify ToS for dataset publication** | Medium | Many Kaggle datasets include Spotify metadata. Add "for research purposes only" disclaimer. Use MusicBrainz IDs as primary identifiers. Strip or anonymize if Spotify requests takedown. |

---

## Executive Summary

### Top 5 Audio Ideas

| Rank | Approach | Dims | Why |
|------|----------|------|-----|
| 1 | **LAION-CLAP** | 512 | Cross-modal audio↔text, enables semantic search, Apache 2.0 |
| 2 | **MERT-v1-330M** | 1024 | MARBLE SOTA, music-specific, direct upgrade of existing extractor |
| 3 | **DSP feature suite** | ~390 | Free (CPU), interpretable, covers harmony/rhythm/timbre, complements neural |
| 4 | **Statistics pooling** (mean+std) | 2× existing | Zero-cost upgrade, captures temporal dynamics |
| 5 | **BEATs** | 768 | AudioSet SOTA, semantic audio events, MIT license |

### Top 3 Lyrics Ideas

| Rank | Approach | Dims | Why |
|------|----------|------|-----|
| 1 | **BGE-M3** | 1024 | Best multilingual embedder, MIT, replaces MPNet |
| 2 | **NRC EmoLex + MEmoLon** | 13 | Multilingual emotion features, 2 minutes to extract, directly predicts valence |
| 3 | **multilingual-e5-large** | 1024 | Strong alternative, MIT, for comparison vs BGE-M3 |

### The Benchmark Answer

> There is **no standard lyric embedding benchmark** equivalent to MARBLE (audio) or MTEB (text). "SongBERT," "ITS embeddings," and "Lyrisong" **do not exist as public models**. The current best practice is to use MTEB multilingual rankings as a proxy, evaluate on your own tasks (valence/mood prediction, genre cluster purity), and use MoodyLyrics dataset for lyric-mood validation if needed. MARBLE covers only audio, not lyrics.

### Top 5 Website Tools

| Rank | Tool | Why Build First |
|------|------|----------------|
| 1 | 🗺️ **Song Map** (UMAP scatter) | Visual "wow" factor, drives engagement, showcases the dataset |
| 2 | 🔍 **Similar Songs Finder** | Core utility, highest repeat usage |
| 3 | 🧬 **Song DNA Page** | Per-track detail, radar charts, encourages browsing |
| 4 | 💬 **Lyric Semantic Search** | Unique differentiator powered by CLAP + BGE-M3 |
| 5 | 🎯 **Playlist Generator** | Actionable output with feature sliders |
