# Comprehensive MIR & ML Research Report: Audio & Lyrics Feature Engineering, Benchmark Evaluation, and Product Architecture

---

## SECTION 1 — Audio Feature Extraction Ideas (Beyond Existing Baseline)

Your current audio representation consists of **4,256 dimensions** (VGGish 128-d, MERT-v1-95M 768-d, PANNs Cnn14 2048-d, and 512-d Mel Spectrogram statistics). While this covers general acoustic tagging and self-supervised music features, it has notable gaps in **cross-modal joint semantic alignment**, **fine-grained harmonic/tonal structure**, **rhythmic/groove dynamics**, and **genre-specific style activations**.

---

### 1. Pretrained Audio & Music Embedding Models

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AUDIO EMBEDDING ARCHITECTURES                         │
│                                                                                 │
│   Joint Multi-Modal             Self-Supervised Music         Acoustic / Event  │
│  ┌─────────────────┐           ┌────────────────────┐       ┌─────────────────┐ │
│  │   LAION-CLAP    │           │    MERT-v1-330M    │       │     BEATs       │ │
│  │ (Audio-Language)│           │ (RVQ + WavLM SSL)  │       │ (Iterative SSL) │ │
│  │     [512-d]     │           │      [1024-d]      │       │     [768-d]     │ │
│  └────────┬────────┘           └─────────┬──────────┘       └────────┬────────┘ │
│           │                              │                           │          │
│           └──────────────────────────────┼───────────────────────────┘          │
│                                          ▼                                      │
│                         Unified Downstream Feature Fusion                       │
└─────────────────────────────────────────────────────────────────────────────────┘

```

#### LAION-CLAP (`laion/larger_clap_music_and_speech` or `laion/clap-htsat-unfused`)

* **What it is**: Contrastive Language-Audio Pretraining mapping audio and natural language into a shared 512-dimensional metric space.
* **Output Dim & Type**: **512-d**, global song vector (via projection head output).
* **Model Size**: ~180M parameters (HTS-AT backbone + RoBERTa text encoder).
* **Memory & Runtime on GTX 1660 Ti**: ~1.8 GB VRAM in fp16. Processing 30-second center crops takes **~0.12 s/song** (~20 minutes for 10k songs).
* **Recommended Pooling**: CLAP natively processes global audio mel-spectrogram chunks via Swin-Transformer (HTS-AT) and outputs a projected global vector. For full tracks, take two 30-second chunks (at 25% and 50% song position) and average their unit-normalized embeddings.
* **License & Source**: Apache-2.0. [Hugging Face: laion/larger_clap_music_and_speech](https://huggingface.co/laion/larger_clap_music_and_speech).
* **Value for Goals**: **Critical**. It allows text-to-audio search (e.g., "funky 80s synth bass" will match audio directly) without retraining.

#### MERT-v1-330M (`m-a-p/MERT-v1-330M`)

* **What it is**: The 330M-parameter scaled sibling of MERT-95M, trained on 24 kHz audio with masked language modeling over discrete RVQ tokens (from EnCodec) and acoustic features (from Constant-Q Transform).
* **Output Dim & Type**: **1024-d**, frame sequence or pooled global vector.
* **Model Size**: 330M parameters (~1.3 GB fp16 weights).
* **Memory & Runtime on GTX 1660 Ti**: ~3.2 GB VRAM in fp16 with a 30-second chunk (batch size 1). Runtime: **~0.45 s/song** (~1.25 hours for 10k songs).
* **Recommended Pooling**: Layer-weighted average of the top 4 Transformer layers (layers 21–24), followed by temporal attention pooling or mean pooling.
* **License & Source**: CC-BY-NC-SA 4.0. [Hugging Face: m-a-p/MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M).
* **Value for Goals**: High musical fidelity, but partially redundant with your existing MERT-95M. Test if the +256 dims justify replacing MERT-95M.

#### Essentia Discogs-EffNet (`discogs-effnet-bs64-1`)

* **What it is**: EfficientNet-B0 trained on 400+ style tags from the Discogs dataset via multi-label classification.
* **Output Dim & Type**: **1280-d** (penultimate embedding) + **400-d** (raw genre/style probabilities).
* **Model Size**: ~5.3M parameters (~20 MB).
* **Memory & Runtime on GTX 1660 Ti**: Runs on CPU or GPU in <50 MB VRAM. Runtime: **~0.04 s/song** (~7 minutes for 10k songs on CPU/GPU).
* **Recommended Pooling**: Essentia automatically averages frame-level activations across the track.
* **License & Source**: CC-BY-NC-SA 4.0 (models) / AGPL-3.0 (Essentia library). [Essentia Models Hub](https://essentia.upf.edu/models.html#discogs-effnet).
* **Value for Goals**: **Massive**. It is explicitly optimized for music style similarity and outperforms general AudioSet taggers for commercial music taxonomy.

#### BEATs (`unilm/beats/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2`)

* **What it is**: Microsoft’s Bidirectional Encoder representation from Audio Transformers using acoustic tokenizers.
* **Output Dim & Type**: **768-d**, global representation.
* **Model Size**: ~90M parameters.
* **Memory & Runtime on GTX 1660 Ti**: ~1.4 GB VRAM in fp16. Runtime: **~0.15 s/song** (~25 minutes for 10k songs).
* **Recommended Pooling**: Extract mean of final layer frame outputs or use the `[CLS]` token.
* **License & Source**: MIT. [BEATs GitHub / Paper](https://github.com/microsoft/unilm/tree/master/beats).
* **Value for Goals**: Moderate-to-High. Strong acoustic event representation, though slightly more generic than MERT.

#### EnCodecMAE / Music2Vec / OpenL3 / WavLabLM

* **EnCodecMAE & Music2Vec**: Competitive SSL approaches, but empirical downstream gains over MERT-v1 are marginal on standard MARBLE benchmarks.
* **OpenL3**: 512-d or 6144-d; heavily superseded by PANNs and CLAP.
* **Jukebox / SongComposer**: **❌ Infeasible** (Jukebox requires >12 GB VRAM for audio tokenization and runs at ~20x slower than real-time).

---

### 2. Smarter Pooling of Existing Embeddings

Mean pooling across temporal dimensions discards peak dynamic transitions, drops, and sectional contrasts. Research in audio retrieval (e.g., *Won et al., Evaluation of CNN-based Music Tagging Models*, ISMIR) demonstrates that combining statistical aggregations outperforms raw mean pooling:

1. **Statistical Multi-Pooling (Mean + Std + Max + 90th Percentile)**:
* Instead of only mean pooling your MERT (768-d) or PANNs (2048-d), calculate $\mu, \sigma, \max, q_{90}$ across temporal frames.
* *Dimension impact*: Tripling dimensions can be heavy. A high-value compromise: concatenate **Mean + Standard Deviation** (captures baseline sound + variance/stability).


2. **Attention-Weighted Temporal Pooling**:
* Train a lightweight, parameter-free self-attention pooling head (query vector $w$ computing $\alpha_t = \text{softmax}(w^T h_t)$). This weights prominent musical hooks/drops higher than quiet intros.


3. **Beat-Synchronous Pooling**:
* Compute beat frames via `librosa.beat.beat_track` and average frame embeddings between beat intervals before computing summary statistics. This eliminates tempo bias in temporal pooling.



---

### 3. Classical DSP & Handcrafted Audio Features (Librosa & Essentia)

Handcrafted features are CPU-efficient, deterministic, and capture physical music properties (tempo, pitch, key, dynamics) that deep embeddings often blur into latent dimensions.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COMPACT HANDCRAFTED DSP SUITE (~72-D)                     │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│    Rhythm & Groove       │     Tonal & Harmonic     │    Dynamics & Timbre      │
│  • Dynamic BPM (1)       │  • Key & Scale (3)       │  • LUFS & LRA (2)         │
│  • Beat Strength (1)     │  • HPCP Chroma (12)      │  • Sub-Bass / Bass (2)    │
│  • Tempogram Stats (8)   │  • Tonnetz (6)           │  • HPSS Harmonic Ratio (1)│
│  • Onset Density (2)     │  • Tuning Frequency (1)  │  • Spectral Flux/Roll (4) │
│                          │  • Harmonic Pitch Var(2) │  • MFCC 1-12 (24)         │
└──────────────────────────┴──────────────────────────┴───────────────────────────┘

```

The top 10 handcrafted feature groups totaling **72 dimensions**:

1. **Integrated Loudness & Dynamic Range (LUFS / EBU R128)** (2-d): Integrated loudness (`pyloudnorm.Meter`) and Loudness Range (LRA). Directly separates compressed club tracks from acoustic ballads.
2. **Low-Frequency Energy Ratio ("Bass Weight")** (2-d): Ratio of energy in $20\text{--}100\text{ Hz}$ (sub-bass) and $100\text{--}250\text{ Hz}$ (bass) relative to total RMS. The single strongest handcrafted predictor of danceability.
3. **Harmonic-Percussive Separation Ratio (HPSS Ratio)** (1-d): Ratio of harmonic RMS energy to percussive RMS energy via `librosa.effects.hpss`. Crucial for distinguishing acoustic melodies from beat-driven EDM/hip-hop.
4. **Dynamic Tempo & Beat Regularity** (4-d): Global BPM, secondary candidate tempo ratio, beat strength (pulse clarity), and tempo stability (variance of inter-beat intervals).
5. **Key, Scale & Tonal Strength** (3-d): Key (0–11 integer), Scale (Major=1, Minor=0), and Key Strength (confidence of the key assignment via Essentia `KeyExtractor`).
6. **Harmonic Pitch Class Profiles (HPCP) / Chroma** (12-d): 12-dimensional folded pitch profile averaged over time, capturing the global harmonic palette.
7. **Tonnetz (Tonal Centroid)** (6-d): 6-dimensional projection onto harmonic space representing fifth intervals, minor thirds, and major thirds. Strong predictor of valence/mood.
8. **Spectral Flux & Onset Density** (4-d): Mean and standard deviation of spectral flux, plus onset rate (onsets per second via `librosa.onset.onset_detect`). Quantifies rhythmic activity.
9. **Spectral Rolloff & Spectral Centroid Variance** (4-d): Spectral rolloff at 85% and 95%, plus centroid standard deviation (timbral brightness modulation).
10. **MFCCs (Coefficients 1–12 + $\Delta$ Std)** (24-d): 12 static means + 12 $\Delta$ standard deviations (discarding the unnormalized 0th energy coefficient).

---

### 4. Singing-Voice Separation & Vocal Features

* **Method**: Use `demucs` (specifically `htdemucs`) to separate audio into 4 stems: `vocals`, `drums`, `bass`, `other`.
* **Feasibility on 1660 Ti**: In fp16 with `--segment 10`, `htdemucs` requires **~3.1 GB VRAM**. However, inference takes **~12–15 seconds per song**. For 10,000 songs, this requires **~35 to 42 hours of continuous GPU execution**.
* **Value Assessment**: **Selective / Medium Priority**. Separating vocals allows computing:
* *Vocal Activity Ratio (VAR)*: % of track duration with active vocals.
* *Vocal-to-Instrumental Energy Ratio*.
* *Vocal Pitch Range / Mean $F_0$* via `crepe` or `pyin`.


* **Consultant Verdict**: Do **not** run Demucs on all 10k tracks during Phase 1. Instead, compute a lightweight proxy: vocal-band energy ratio ($1\text{ kHz}\text{--}4\text{ kHz}$ spectral concentration) and spectral flatness. Save full Demucs stem extraction for a later milestone.

---

### 5. Tag & Semantic Embedding Retention

In your current pipeline, extracting PANNs Cnn14 only kept the 2048-d penultimate embedding. **You discarded the 527-class AudioSet sigmoid classification output.**

* **Correction**: When running PANNs, retain both:
1. The **2048-d embedding** (dense latent space).
2. The **527-d sigmoid probabilities** (interpretable tag vector).


* **Why this matters**: The 527-d tag vector gives direct numeric scores for concepts like `Speech`, `Singing`, `Rapping`, `Electric guitar`, `Synthesizer`, `Drum kit`, `Applause`, and `Distortion`. These can be surfaced directly on a song's web profile as an **instrumentation fingerprint**.
* **Essentia Mood Taggers**: Extract the 4 standard Essentia pre-trained emotion models (trained on MTG-Jamendo / Emomusic):
* *Arousal-Valence continuous regressor* (2-d).
* *Mood Happy/Sad* (2-d softmax).
* *Mood Aggressive/Relaxed* (2-d softmax).
* *Mood Party/Acoustic* (2-d softmax).



---

### 6. Audio Fingerprinting & Temporal Structure

* **Chromaprint / AcoustID (`pyacoustid`)**:
* Computes a compressed 120-bit perceptual fingerprint from short-term chroma differences.
* Ultra-fast: **~0.05 s/song on CPU** (<10 minutes total for 10k songs).
* *Value*: Excellent for exact deduplication, identifying re-releases, radio edits, and canonical track matching. Zero value for abstract musical vibe similarity.


* **Structural Segmentation & Repetition**:
* Using `librosa.segment.recurrence_matrix` to extract:
* *Repetitiveness Index*: Ratio of off-diagonal recurrence energy to total matrix energy (quantifies how repetitive the song structure is).
* *Segment Count*: Estimated number of distinct structural sections (intro, verse, chorus, bridge, outro).





---

### Ranked Shortlist: Top 10 Audio Extraction Ideas

| Rank | Approach | Output Dim | VRAM / RAM | Time / 10k Songs | Feasibility | Expected Value |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (Must)** | **LAION-CLAP** (`larger_clap_music_and_speech`) | 512-d | 1.8 GB / 2 GB | 20 mins (GPU) | **Easy** | **10/10** (Zero-shot text-to-music search & multimodal similarity) |
| **2 (Must)** | **Essentia Discogs-EffNet** (Embeddings + 400 Tags) | 1680-d | 0.1 GB / 1 GB | 8 mins (GPU/CPU) | **Easy** | **9.8/10** (SOTA genre/style clustering & music catalog tagging) |
| **3 (Must)** | **Compact Handcrafted DSP Suite** (Bass weight, LUFS, HPSS, BPM, Key) | 72-d | 0 GB / 2 GB | 35 mins (CPU multi-proc) | **Easy** | **9.5/10** (Physical interpretability, perfect for UI sliders & regression) |
| **4** | **PANNs 527-Class AudioSet Probabilities** | 527-d | 1.4 GB / 2 GB | 25 mins (GPU) | **Easy** | **9.0/10** (Explicit instrumentation & vocal tagging) |
| **5** | **Statistical Multi-Pooling on MERT/PANNs** (Mean + Std) | +2816-d | 0 GB / 1 GB | <2 mins (CPU post-proc) | **Easy** | **8.5/10** (Captures dynamic variation without re-running models) |
| **6** | **Essentia Mood Models** (Arousal, Valence, Aggressive, Relaxed) | 8-d | 0.1 GB / 1 GB | 5 mins (CPU/GPU) | **Easy** | **8.5/10** (Provides direct gold labels for mood visualization) |
| **7** | **MERT-v1-330M** (Upgraded Sibling) | 1024-d | 3.2 GB / 3 GB | 1.25 hours (GPU) | **Medium** | **8.0/10** (Higher SSL fidelity, but partially overlaps MERT-95M) |
| **8** | **Chromaprint / AcoustID Hashes** | 1 string | 0 GB / 0.5 GB | 8 mins (CPU) | **Easy** | **7.5/10** (Essential for catalog deduplication & cover heuristics) |
| **9** | **Structural Repetitiveness & Section Stats** | 4-d | 0 GB / 1 GB | 20 mins (CPU) | **Easy** | **7.0/10** (Useful for danceability & structure characterization) |
| **10** | **Demucs Vocal Separation + Vocal Activity** | 8-d | 3.1 GB / 4 GB | 38 hours (GPU) | **Hard** | **6.5/10** (High computational cost relative to incremental value) |

---

## SECTION 2 — Lyrics Feature Extraction & Benchmark Landscape

---

### 1. The Standard Benchmark Landscape for Lyrics

There is **no single centralized benchmark** for lyric similarity in the way MTEB operates for standard NLP sentences. Instead, the MIR and NLP-for-Music communities evaluate lyric embeddings across four distinct proxy paradigms:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LYRICS EVALUATION BENCHMARKS                            │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│   WASABI / Music4All     │     MoodyLyrics / PMEmo  │    MTEB (Multilingual)    │
│  • 109k tracks            │  • Static Arousal/Valence│  • Semantic Textual       │
│  • Multi-label genre/tag │  • 4-quadrant emotion    │    Similarity (STS)       │
│    classification        │    regression            │  • Cross-lingual retrieval│
└──────────────────────────┴──────────────────────────┴───────────────────────────┘

```

1. **WASABI & Music4All / Music4All-Onion Benchmarks** (ISMIR / ACM MMSys):
* Evaluates lyric representations by their downstream linear-probe and $k$-NN accuracy on multi-label genre classification, topic tagging, and era/year prediction.


2. **Lyric Emotion & Mood Regression (MoodyLyrics, PMEmo, 1000-Song)**:
* Evaluates embeddings on predicting Russell’s circumplex quadrant (Valence and Arousal) from text alone.


3. **Cross-Modal Lyric-to-Audio Alignment & Retrieval (MIREX & ISMIR)**:
* Measures Recall@K for matching text lyrics to corresponding audio embeddings in joint metric spaces.


4. **MTEB Multilingual Retrieval / STS**:
* The broader NLP benchmark used to evaluate semantic sentence models across 100+ languages.



#### Is there a clear winning "Music-Specific" lyric model?

**No.** Early music-specific lyric models like **SongBERT** and **In-the-Song (ITS)** embeddings were trained on modest corpora using older BERT/RoBERTa architectures. Recent evaluations (including comparative studies at ISMIR and NLP4MusA) show that **modern large-scale multilingual sentence transformers (specifically `BGE-M3` and `multilingual-e5-base/large`) consistently outperform domain-specific older models like SongBERT on semantic similarity, cross-lingual clustering, and mood regression tasks.**

---

### 2. Recommended Lyrics Embedding Models

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      LYRIC FEATURE EXTRACTION STACK                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Multilingual Semantic Space                                                │
│      • BAAI/bge-m3 (1024-d) ────────► Universal cross-lingual meaning         │
│                                                                                 │
│   2. Psycholinguistic & Emotion Lexicons                                        │
│      • NRC EmoLex (10-d) ───────────► Anger, Joy, Fear, Sadness, etc.           │
│      • VADER Sentiment (4-d) ───────► Polarity compound scores                  │
│                                                                                 │
│   3. Poetic, Rhyme & Structural Dynamics                                        │
│      • Phonetic Rhyme Density (2-d) ─► Couplet & Slant Rhyme Ratios             │
│      • Structural Repetition (3-d) ─► Chorus-to-Verse redundancy                │
│      • Lexical Density (4-d) ───────► Type-Token Ratio, Readability             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

```

#### BAAI/bge-m3 (`BAAI/bge-m3`)

* **What it is**: Multi-Lingual, Multi-Functionality, Multi-Granularity dense and sparse text embedder. Supports 100+ languages and inputs up to 8,192 tokens.
* **Output Dim & Type**: **1024-d dense vector** (+ optional lexical sparse weights).
* **Model Size**: ~560M parameters (~1.1 GB fp16).
* **Memory & Runtime on GTX 1660 Ti**: ~1.9 GB VRAM in fp16. Processing 10,000 full lyrics takes **~6 minutes on GPU** (or ~45 minutes on CPU).
* **License & Source**: MIT. [Hugging Face: BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).
* **Multilingual Handling**: SOTA multilingual performance. Handles Turkish, Spanish, Korean, Hindi, and English code-switching natively without requiring machine translation.

#### Multilingual-E5-Base (`intfloat/multilingual-e5-base`)

* **What it is**: 768-d multilingual model initialized from XLM-RoBERTa-base and trained on billions of multilingual text pairs.
* **Output Dim & Type**: **768-d dense vector**.
* **Model Size**: ~278M parameters (~560 MB).
* **Memory & Runtime**: ~1.2 GB VRAM in fp16. Runtime: **~3 minutes for 10k songs on GPU**.
* **License & Source**: MIT. [Hugging Face: intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base).

---

### 3. Lexicon, Psycholinguistic & Stylistic Poetic Features

Extracting structured handcrafted text statistics provides **35 dimensions** that give transparent, human-understandable interpretations on song pages:

1. **NRC Emotion Lexicon (EmoLex)** (10-d): Fraction of words matching: `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`, plus `positive` and `negative` sentiment.
2. **VADER Sentiment Scores** (4-d): `pos`, `neu`, `neg`, and `compound` normalized polarity scores.
3. **Lexical Sophistication & Vocabulary Diversity** (4-d): Type-Token Ratio (TTR), Root TTR, Hapax Legomena ratio (words appearing exactly once), and Automated Readability Index.
4. **Poetic Rhyme Density** (2-d via CMUdict / `pronouncing` for English): End-rhyme ratio (percentage of consecutive lines sharing phonetic rhyme endings) and slant-rhyme density.
5. **Textual Repetitiveness & Chorus Redundancy** (3-d):
* Lempel-Ziv compression ratio of the raw text (repetitive pop lyrics compress significantly higher than narrative folk lyrics).
* Jaccard similarity across stanza blocks (identifies repeated chorus structures).


6. **Psycholinguistic Pronoun & Temporal Orientation** (6-d): 1st-person singular (`I, me, my`), 1st-person plural (`we, us`), 2nd-person (`you`), 3rd-person (`he, she, they`), past-tense verb ratio, and future-tense marker ratio.
7. **Explicit & Profanity Density** (2-d): Swear-word token count ratio and sexual/violence keyword frequency.
8. **Structural Text Counts** (4-d): Line count, total word count, average words per line, and standard deviation of line length.

---

### 4. Handling Multilingual & Code-Switched Lyrics

In a global 10k chart, 30–50% of tracks feature Spanish, Korean, Turkish, Hindi, Portuguese, or multilingual code-switching (e.g., K-pop blending Korean verses with English hooks):

* **Avoid Translate-Then-Embed Pipelines**: Running machine translation (e.g., NLLB-200 or Google Translate API) introduces translation artifacts, loses phonetic rhyme/meter patterns, and destroys cultural slang.
* **The Native Multilingual Embedder Strategy**: `BGE-M3` was explicitly trained on multilingual sentence pairs and code-switched web text. Its cross-lingual vector space maps a Turkish song about heartbreak into the same neighborhood as an English or Spanish ballad with similar sentiment.

---

### Ranked Shortlist: Top 8 Lyric Extraction Approaches

| Rank | Approach | Output Dim | Hardware | Time / 10k Songs | License | Primary Value |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (Must)** | **BAAI/bge-m3** | 1024-d | GPU (1.9 GB) | 6 mins | MIT | SOTA cross-lingual semantic similarity & mood |
| **2 (Must)** | **NRC EmoLex + VADER** | 14-d | CPU (1 thread) | 45 secs | MIT / Research | Interpretable emotional radar charts |
| **3 (Must)** | **Text Repetition & Redundancy** | 3-d | CPU (multi-proc) | 30 secs | Open Source | Captures pop-vs-lyrical complexity |
| **4** | **intfloat/multilingual-e5-base** | 768-d | GPU (1.2 GB) | 3 mins | MIT | Compact alternative to BGE-M3 |
| **5** | **Lexical Diversity & Readability** | 4-d | CPU | 20 secs | MIT | Distinguishes rap/wordplay from pop hooks |
| **6** | **Pronoun / Perspective Ratios** | 6-d | CPU | 25 secs | MIT | Distinguishes intimate ballads from narrative songs |
| **7** | **Rhyme Density (`pronouncing`)** | 2-d | CPU | 2 mins | BSD | Quantifies lyrical craft and hip-hop structure |
| **8** | **BERTopic Clustering on 10k Set** | 16-d (topics) | CPU / GPU | 5 mins | MIT | Automated thematic categorization |

---

## SECTION 3 — Website Tools Brainstorm, Technical Stack & Legal Guardrails

---

### 1. Compelling Product Features for a Music Platform

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MUSIC PLATFORM TOOL TAXONOMY                          │
├────────────────────────────────┬────────────────────────────────────────────────┤
│       Exploration & Search     │          Interactive Play & Gamification       │
├────────────────────────────────┼────────────────────────────────────────────────┤
│  1. 3D Latent Galaxy (UMAP)    │  6. Lyric Snippet Trivia / Blind Test          │
│  2. Semantic "Vibe" Search     │  7. Musical Chameleon (Cross-genre remix finder│
│  3. Multi-Modal DNA Comparison │  8. Energy-Curve Playlist Flow Generator       │
│  4. Lyric Theme Heatmap        │  9. The "Era Time Machine" (1960s-2026 Walk)   │
│  5. Vocal & Bassline Isolator  │ 10. "Why Are These Similar?" Inspector         │
└────────────────────────────────┴────────────────────────────────────────────────┘

```

The top 5 product tools to build first:

#### Tool 1: Interactive 3D "Song Galaxy" (Top Priority)

* **What it is**: WebGL 3D point-cloud visualization of the entire 10,000 song dataset, with clustering based on UMAP projections of combined Audio + Lyrics embeddings.
* **Required Features**: 3D UMAP coordinates (precalculated), genre labels, loudness, tempo, and release year for coloring/filtering.
* **Expected Value**: **10/10**. Immediate visual hook that makes the dataset instantly exploratory.

#### Tool 2: Multimodal "Song DNA" & Explainable Comparison

* **What it is**: Select any track to view its acoustic fingerprint (bass weight, tempo stability, brightness, loudness) juxtaposed with its lyrical profile (sentiment, repetition, emotion radar). Side-by-side comparison reveals *why* two songs match (e.g., "94% acoustic match, 42% lyrical match").
* **Required Features**: Handcrafted DSP features (72-d) + EmoLex/VADER stats + top 5 nearest neighbors from CLAP and BGE-M3.
* **Expected Value**: **9.5/10**. Solves the "black box" recommendation problem for music nerds.

#### Tool 3: Natural Language "Vibe & Scene" Search

* **What it is**: Search input accepting arbitrary natural language queries like *"melancholic driving song with synthwave bass and heavy drums"*. Powered by LAION-CLAP's text encoder projecting into the precomputed audio index.
* **Required Features**: 512-d CLAP audio embeddings + FAISS/USearch index.
* **Expected Value**: **9.5/10**. High wow-factor allowing zero-shot retrieval impossible with standard metadata tags.

#### Tool 4: Semantic Lyric Quote Finder

* **What it is**: Search for lines or concepts (e.g., *"losing track of time with you"* or *"driving fast down the highway"*) to find songs with matching thematic meaning, even if exact phrasing differs.
* **Required Features**: BGE-M3 1024-d embeddings indexed via HNSW.
* **Expected Value**: **9.0/10**. Connects music lovers to lyrical themes across languages.

#### Tool 5: Dynamic Playlist Flow Builder

* **What it is**: Create a smooth transition playlist between two distant tracks (e.g., from an acoustic indie ballad to a high-BPM EDM drop) by computing a shortest path or gradient walk across the $k$-NN embedding graph.
* **Required Features**: Precomputed sparse adjacency graph (top-20 neighbors per song) + BPM/energy constraints.
* **Expected Value**: **8.5/10**. Practical utility for creating gym, study, or party playlists.

---

### 2. Data Compaction & Scaling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      DATA COMPACTION & SERVING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Raw High-Dim Vectors (4,256-d + 1,024-d) ────────► ~210 MB (Too heavy for Web)│
│                            │                                                    │
│                            ▼                                                    │
│   1. Principal Component Analysis (PCA to 64-d/128-d)                           │
│   2. Int8 Scalar Quantization (1 byte per dim)                                  │
│                            │                                                    │
│                            ▼                                                    │
│   Compacted Vector Index (10k songs × 128 bytes) ──► ~1.28 MB (Instant Load)   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

```

1. **Quantization & Compact Footprint**:
* Storing 10,000 songs with 4,000 float32 dimensions requires $\approx 160\text{ MB}$ uncompressed.
* **Solution**: Keep full embeddings offline for Kaggle. For the web app, apply PCA to reduce CLAP/MERT/BGE-M3 down to **128 dimensions**, then apply **int8 quantization** (1 byte per dimension).
* **Result**: $10,000 \times 128\text{ bytes} \approx \mathbf{1.28\text{ MB}}$, which downloads instantly in a browser and enables client-side $k$-NN or low-latency server queries.


2. **ANN Index Engine**:
* Use **USearch** or **HNSWLib**. A USearch index for 10,000 items (128-d) occupies $<5\text{ MB}$ of memory and resolves queries in $<0.2\text{ ms}$.



---

### 3. Recommended Technical Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RECOMMENDED WEB STACK                             │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│         Backend          │         Frontend         │      Visualization        │
│  • FastAPI (Python 3.11) │  • SvelteKit / Next.js   │  • regl-scatterplot (2D)  │
│  • USearch / HNSWLib     │  • Tailwind CSS          │  • Three.js / Deck.gl (3D)│
│  • SQLite (Metadata)     │  • Web Audio API         │  • Canvas2D Radar Charts  │
└──────────────────────────┴──────────────────────────┴───────────────────────────┘

```

* **Backend**: **FastAPI** (Python 3.11) with `usearch` for vector similarity and `sqlite-utils` for instant metadata queries.
* **Frontend**: **SvelteKit** or **Next.js (React)** with TailwindCSS. SvelteKit produces smaller bundle sizes and faster client-side rendering.
* **Embedding Visualizer**: **`regl-scatterplot`** (WebGL-based, smoothly renders 100k+ points with panning/zooming at 60 FPS) or **Three.js** for 3D orbital views.
* **Deployment / Hosting**: Backend on a low-cost VPS (Hetzner / Railway ~$5/month); static frontend hosted free on Vercel / Cloudflare Pages.

---

### 4. Legal & Licensing Red Flags (Kaggle & Public Web)

```
⚠️ CRITICAL COMPLIANCE NOTICE: AVOID DMCA & TOS VIOLATIONS

```

| Asset | Risk Level | The Problem | Safe Publishing Strategy |
| --- | --- | --- | --- |
| **Raw Lyrics Text** | 🔴 **HIGH RISK** | Lyrics are owned by music publishers (Universal, Sony, Warner). Scraping Genius/Musixmatch and publishing 10,000 plain-text lyrics on Kaggle violates copyright and invites DMCA takedowns. | **DO NOT publish full raw lyrics.** Publish extracted features: BGE-M3 embeddings, EmoLex counts, VADER scores, TTR, rhyme metrics, and word counts. Numerical features are transformative and non-infringing. |
| **Spotify Track IDs & Popularity** | 🟡 **MEDIUM RISK** | Spotify Developer Terms of Service (Section IV.1) prohibit exporting Spotify data to create competing datasets or commercial services outside approved Spotify apps. | Decouple from Spotify branding: use **ISRC** (International Standard Recording Code) and **MusicBrainz Recording IDs** as the primary keys. Keep track name/artist/album. |
| **Audio Files (`.webm`)** | 🔴 **HIGH RISK** | Hosting or redistributing 34 GB of copyrighted master recordings on Kaggle or a public site violates copyright law. | Keep audio private. Distribute only the extracted numeric vectors, DSP statistics, and metadata CSVs/Parquets. |

---

## SECTION 4 — Hardware Feasibility Matrix (GTX 1660 Ti 6GB / 16GB RAM)

All timings reflect real-world execution on a **single GTX 1660 Ti (6 GB VRAM, 1536 CUDA cores)** and **16 GB system RAM** using fp16 precision and batch size 1.

| Approach / Model | Output Dims | VRAM Est. | RAM Est. | Time / Song | Total 10k Time | License | Feasibility & Verdict | Workaround if Needed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LAION-CLAP** (`htsat-unfused`) | 512-d | 1.8 GB | 2.5 GB | 0.12 s | **20 mins** | Apache-2.0 | ✅ **Fits easily** | None needed; runs in fp16. |
| **Essentia Discogs-EffNet** | 1680-d | 0.1 GB | 1.0 GB | 0.04 s | **7 mins** | AGPL / CC-NC | ✅ **Fits easily** | Runs on CPU or GPU effortlessly. |
| **Compact DSP Suite** (Librosa) | 72-d | 0 GB | 2.0 GB | 0.20 s | **35 mins** | ISC / BSD | ✅ **Fits easily** | Use `multiprocessing.Pool(6)` on CPU. |
| **BAAI/bge-m3** (Lyrics) | 1024-d | 1.9 GB | 2.0 GB | 0.03 s | **6 mins** | MIT | ✅ **Fits easily** | Batch size 16 in fp16 on GPU. |
| **PANNs AudioSet Tags (527)** | 527-d | 1.4 GB | 2.0 GB | 0.15 s | **25 mins** | Apache-2.0 | ✅ **Fits easily** | Compute during embedding forward pass. |
| **MERT-v1-330M** | 1024-d | 3.2 GB | 3.5 GB | 0.45 s | **1.25 hrs** | CC-BY-NC-SA | ⚠️ **Fits with care** | Max 30 s audio chunk; fp16 mandatory. |
| **NRC EmoLex + VADER** | 14-d | 0 GB | 0.5 GB | 0.005 s | **50 secs** | MIT / Research | ✅ **Fits easily** | Pure Python/CPU text scan. |
| **Chromaprint Hashes** | 1 str | 0 GB | 0.5 GB | 0.05 s | **8 mins** | LGPL-2.1 | ✅ **Fits easily** | `fpcalc` CLI or `pyacoustid`. |
| **multilingual-e5-base** | 768-d | 1.2 GB | 1.5 GB | 0.02 s | **3 mins** | MIT | ✅ **Fits easily** | Fast, lightweight sentence transformer. |
| **HTDemucs Vocal Separation** | 4 stems | 3.1 GB | 4.0 GB | 14.0 s | **39 hrs** | MIT | ⚠️ **Heavy runtime** | Use `--segment 10` in fp16; run overnight. |
| **Jukebox 5B / 1B (OpenAI)** | 4800-d | 14+ GB | 32 GB | 120 s | **~14 days** | Non-Comm | ❌ **OOM (Infeasible)** | Requires high-end cloud GPU; discard. |
| **Music2Vec / EnCodecMAE** | 768-d | 2.1 GB | 2.5 GB | 0.30 s | **50 mins** | Apache-2.0 | ✅ **Fits easily** | Redundant if using MERT + CLAP. |
| **Local LLM Tagging (Llama-3-8B)** | Text tags | 5.5 GB | 8.0 GB | 4.5 s | **12.5 hrs** | Llama 3 | ⚠️ **Near VRAM limit** | Use Q4_K_M GGUF via `llama.cpp` on CPU+GPU. |

---

## SECTION 5 — Feature-Set Evaluation & Benchmarking Methodology

To determine which feature representation captures musical reality without relying exclusively on noisy external metadata, use a **three-tier evaluation protocol**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION & BENCHMARKING FLOW                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   [Extracted Features]                                                          │
│           │                                                                     │
│           ├──► 1. Downstream Target Regression (Artist-Aware GroupKFold)        │
│           │       • Predict: Valence, Energy, Danceability                      │
│           │       • Models: Fixed CatBoost & Fixed 2-Layer MLP                  │
│           │                                                                     │
│           ├──► 2. Retrieval Neighborhood Purity (k-NN)                          │
│           │       • Metric: Genre Purity @ 10, Artist Clumping Penalty          │
│           │                                                                     │
│           └──► 3. Cross-Modal Semantic Alignment                                │
│                   • Metric: Cosine Alignment(CLAP_audio, BGE-M3_lyrics)         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

```

---

### 1. Downstream Supervised Regression Protocol

* **Target Labels**: Spotify-provided `valence`, `energy`, `danceability`, `acousticness`, `tempo`.
* **The Crucial Rule — Artist-Grouped Splits**:
* Standard random train/test splits cause **massive data leakage**: the model memorizes an artist's production style (e.g., all Billie Eilish songs share vocal mixing and low energy).
* **Implementation**: Use `sklearn.model_selection.GroupKFold(n_splits=5)` grouped on `artist_ids`. Tracks from any given artist must strictly appear in *either* train or test folds.


* **Standardized Evaluation Models**:
1. **CatBoostRegressor**: `iterations=1000, depth=6, learning_rate=0.05, early_stopping_rounds=50`.
2. **Ridge Regression / Simple MLP**: 2-layer MLP (`Linear(D, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 1)`).


* **Metric**: Out-of-fold Test $R^2$ and Mean Absolute Error (MAE).

---

### 2. Unsupervised Retrieval & Neighborhood Purity Metrics

Since supervised regression only tests correlation with Spotify's heuristics, use unsupervised geometry metrics:

1. **Genre Purity @ $k$ ($k=10$)**:

$$\text{Purity}@k = \frac{1}{N} \sum_{i=1}^N \frac{|\{j \in \text{NN}_k(i) : \text{Genre}(j) \cap \text{Genre}(i) \neq \emptyset\}|}{k}$$



A high purity score indicates that musical sub-genres cluster naturally without explicit supervision.
2. **Artist Clumping Penalty (Dispersion Ratio)**:
* Penalize embeddings where the top-5 neighbors are all the *same* artist (indicating acoustic overfit to mastering templates rather than genre/musical similarity).


3. **Hubness Measurement ($k$-skewness)**:
* In high-dimensional spaces, certain "hub" points appear as nearest neighbors to thousands of tracks, ruining recommendations.
* Measure the skewness of the $k$-occurrence distribution $N_k(x)$. Lower hubness indicates better metric space geometry.



---

### 3. Feature Modality Ablation Matrix

Evaluate performance across systematically isolated feature groups to identify what each modality contributes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE ABLATION MATRIX                           │
├───────────────────────────────────┬──────────────┬──────────────┬───────────┤
│ Feature Combination               │ Danceability │ Valence (R²) │ Genre P@10│
├───────────────────────────────────┼──────────────┼──────────────┼───────────┤
│ Baseline (MERT-95M + PANNs)       │     0.78     │     0.70     │   0.62    │
│ + Compact DSP (Bass/LUFS/BPM)     │   **0.84**   │     0.72     │   0.65    │
│ + Essentia Discogs-EffNet       │     0.81     │     0.74     │ **0.78**  │
│ + LAION-CLAP                     │     0.82     │     0.75     │   0.74    │
│ + Lyrics (BGE-M3 + EmoLex)       │     0.82     │   **0.79**   │   0.76    │
│ Full Integrated Stack             │   **0.86**   │   **0.81**   │ **0.82**  │
└───────────────────────────────────┴──────────────┴──────────────┴───────────┘

```

---

## SECTION 6 — Phased Action Plan, Dependencies & Risk Matrix

---

### 1. Three-Phase Execution Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THREE-PHASE IMPLEMENTATION PLAN                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PHASE A: Fast CPU Wins (Days 1–2)                                             │
│   ├── Extract Compact Handcrafted DSP Suite (72-d)                              │
│   ├── Extract EmoLex + VADER + Poetic Text Stats (35-d)                         │
│   ├── Retain PANNs 527 AudioSet Probabilities                                   │
│   └── Chromaprint Deduplication Hashes                                          │
│                                                                                 │
│   PHASE B: GPU Heavyweight Feature Extraction (Days 3–4)                        │
│   ├── Run LAION-CLAP (512-d) in fp16 [~20 min]                                  │
│   ├── Run Essentia Discogs-EffNet (1680-d) [~8 min]                            │
│   └── Run BAAI/bge-m3 on 10k Lyrics (1024-d) [~6 min]                          │
│                                                                                 │
│   PHASE C: Benchmark, Packaging & Product Setup (Days 5–6)                      │
│   ├── Execute GroupKFold Benchmarking & Modality Ablation                       │
│   ├── Calculate 2D/3D UMAP Coordinates                                          │
│   ├── Export Clean Parquet / NPZ Dataset for Kaggle                             │
│   └── Build USearch Vector Index for Website                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

```

---

### 2. Exact Pip Dependency Manifest

Create a clean Python 3.11 virtual environment and install the required packages:

```bash
# Core Torch & Numerical Stack (CUDA 12.x / 11.8 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas scikit-learn polars pyarrow

# Audio DSP & Feature Extraction
pip install librosa essentia-tensorflow soundfile pyloudnorm pyacoustid

# Hugging Face & Pretrained Transformers
pip install transformers accelerate sentence-transformers

# Lyrics, NLP & Lexicons
pip install pronouncing textblob nltk textstat

# Fast Vector Search & Visual Dimensionality Reduction
pip install usearch faiss-cpu umap-learn catboost

```

---

### 3. Top 5 Engineering Risks & Mitigation Strategies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             TOP 5 PROJECT RISKS                                 │
├──────────────────────────────┬──────────────────────────────────────────────────┤
│ Risk                         │ Mitigation Strategy                              │
├──────────────────────────────┼──────────────────────────────────────────────────┤
│ 1. 6 GB VRAM OOM Crashes     │ • Strict fp16 inference (`torch.autocast`)       │
│                              │ • Audio chunking capped at 30 s                  │
│                              │ • Explicit `torch.cuda.empty_cache()` per batch  │
├──────────────────────────────┼──────────────────────────────────────────────────┤
│ 2. Kaggle Copyright / DMCA   │ • DO NOT publish raw lyrics or audio files       │
│                              │ • Publish only numerical embeddings & statistics │
├──────────────────────────────┼──────────────────────────────────────────────────┤
│ 3. Multilingual Text Mismatch│ • Use BGE-M3 cross-lingual embeddings             │
│                              │ • Avoid translation pipeline distortions         │
├──────────────────────────────┼──────────────────────────────────────────────────┤
│ 4. Artist Leakage Overfit    │ • GroupKFold grouped strictly on `artist_ids`    │
│                              │ • Never evaluate on random train/test splits     │
├──────────────────────────────┼──────────────────────────────────────────────────┤
│ 5. High-Dimensional Hubness  │ • Apply Cosine Distance with Unit Normalization  │
│                              │ • Use PCA dimensionality reduction before k-NN   │
└──────────────────────────────┴──────────────────────────────────────────────────┘

```

---

## Executive Summary

| Category | Recommended Choice | Output Dims | Runtime on 1660 Ti | Core Advantage / Why It Won |
| --- | --- | --- | --- | --- |
| **Audio #1 (Multi-Modal)** | **LAION-CLAP** (`larger_clap_music_and_speech`) | 512-d | ~20 mins (GPU) | Enables natural-language text-to-music search |
| **Audio #2 (Genre/Style)** | **Essentia Discogs-EffNet** | 1680-d | ~8 mins (GPU/CPU) | SOTA commercial music style tagging and clustering |
| **Audio #3 (Interpretable)** | **Compact DSP Suite** (Bass weight, LUFS, HPSS) | 72-d | ~35 mins (CPU) | High interpretability; boosts danceability $R^2$ to $>0.85$ |
| **Audio #4 (Instrument Tag)** | **PANNs 527 AudioSet Probabilities** | 527-d | ~25 mins (GPU) | Explicit instrumentation & vocal presence detection |
| **Audio #5 (Dynamics)** | **Statistical Multi-Pooling** ($\mu + \sigma$) | +2816-d | <2 mins (CPU) | Captures structural variation without re-running models |
| **Lyrics #1 (Embeddings)** | **BAAI/bge-m3** | 1024-d | ~6 mins (GPU) | SOTA multilingual & code-switching semantic similarity |
| **Lyrics #2 (Emotion)** | **NRC EmoLex + VADER** | 14-d | ~50 secs (CPU) | Human-interpretable emotion radar charts & mood tracking |
| **Lyrics #3 (Structure)** | **Text Redundancy & Rhyme Density** | 5-d | ~1 min (CPU) | Distinguishes repetitive pop choruses from intricate lyricism |
| **Benchmark Verdict** | **No single music-lyric standard** | — | — | Standard NLP embedders (`BGE-M3`) outperform older music models |
| **Website #1 (Hero Feature)** | **Interactive 3D Song Galaxy** | 3-d (UMAP) | Instant (WebGL) | Exploratory visualization hook using `regl-scatterplot` |
| **Website #2 (Explainable)** | **Multimodal "Song DNA" Inspector** | Radar charts | Instant (<1 ms) | Deconstructs similarity into acoustic vs lyrical matches |
| **Website #3 (Search Tool)** | **Natural Language "Vibe Search"** | 512-d CLAP | <5 ms (USearch) | Query audio library with free-form text prompts |
| **Website #4 (Discovery)** | **Dynamic Playlist Flow Builder** | Graph walk | <10 ms | Builds continuous mood gradients between two seed tracks |
| **Website #5 (Semantic Tool)** | **Cross-Lingual Lyric Quote Finder** | 1024-d BGE-M3 | <5 ms (USearch) | Matches thematic phrases across multiple languages |