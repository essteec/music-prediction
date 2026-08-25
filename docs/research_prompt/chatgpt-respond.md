# Music Dataset Feature-Expansion Research Report — Part 1/1

I treated the uploaded brief as the project specification and verified the key model/library/benchmark facts against current primary or official sources. Your existing MERT/PANNs/VGGish/mel-statistics + MPNet/MiniLM baseline is **not** something I would rebuild; the central question is what adds *orthogonal information* rather than another large generic embedding. Your brief explicitly establishes those representations as existing assets. 

## 1. Executive recommendation

* **Test first: structured MIR descriptors**, especially rhythm/beat stability, onset density, chroma/HPCP, harmonic-change statistics, section/novelty structure, dynamics, and intro/outro/silence statistics. These are cheap, interpretable, and substantially more orthogonal to a global MERT/PANNs/VGGish vector than another generic SSL embedding. `librosa` directly exposes chroma, CQT, MFCC, RMS, spectral and related features; Essentia provides higher-level MIR descriptors including segmentation, voice/instrumental and dynamic-complexity descriptors. ([Librosa][1])
* **Test MusicFM**, not five more general audio SSL models. MusicFM is explicitly a music-informatics foundation model, making it a more defensible complement experiment than BEATs/HuBERT/wav2vec2. ([GitHub][2])
* **Test LAION-CLAP as a separate retrieval/cross-modal branch**, not blindly concatenate it into the 4,254-D supervised feature vector. CLAP is explicitly trained to align audio and natural-language representations and has released music/speech checkpoints. ([GitHub][3])
* **Replace the current lyric embedding pipeline experimentally with one multilingual long-context model**, with **BGE-M3 as the first candidate**. It supports 8,192 tokens, 1024-D embeddings and multilingual retrieval; its model license is MIT. ([Hugging Face][4])
* **Use section-aware/chunked lyric representations**, because truncating every lyric at 3,000 characters systematically discards later verses, bridges and outros. Your current pipeline explicitly uses that truncation. 
* **Build a small human lyric-similarity benchmark yourself.** There is no credible universal "best lyric embedding" benchmark that directly answers your question.
* **Do not prioritize BEATs, HuBERT, wav2vec 2.0 or AudioMAE initially.** They are useful research models, but your existing PANNs + MERT already cover much of the generic audio representation territory.
* **Do not prioritize Jukebox.** It is old, computationally inappropriate for this laptop, archived, and its code/weights use a noncommercial license. ([GitHub][5])
* **Do not make source separation a default preprocessing stage.** Instead, run a 100–200-song pilot and determine whether stem-derived descriptors improve retrieval/prediction enough to justify the substantial extraction cost.
* **Basic Pitch is a much more interesting cheap vocal/melodic experiment than full AMT.** It is explicitly designed as a lightweight polyphonic transcription model and is Apache-2.0 licensed. ([GitHub][6])
* **Do not publish your downloaded YouTube audio or raw lyrics to Kaggle merely because your derived vectors are publishable.** The underlying recording/lyrics rights remain separate from the feature-model license.
* **Keep audio similarity, lyric similarity and supervised prediction as separate representation problems.**
* **Do not concatenate everything.** With only 10,000 observations, thousands of highly correlated embedding dimensions can easily increase variance and make your supervised models worse.
* **Use artist-aware and duplicate-aware splits.** A random song-level split is particularly dangerous for a popularity corpus dominated by major artists.
* **Your popularity target needs special treatment.** Rank, popularity, artist followers and artist popularity can be temporally or directly related to the target; they are not legitimate predictors if they encode the target or post-target information.
* My first experimental stack would therefore be: **structured MIR → MusicFM → BGE-M3 lyrics → CLAP retrieval branch → Basic Pitch-derived melody features**.

---

# 2. Current-state audit

You have a surprisingly strong baseline already.

### Audio

| Existing representation |         D | Main information                          |
| ----------------------- | --------: | ----------------------------------------- |
| VGGish                  |       128 | general AudioSet acoustic representation  |
| MERT-v1-95M             |       768 | music-specific SSL representation         |
| PANNs Cnn14             |     2,048 | AudioSet acoustic/event representation    |
| Mel statistics          |       512 | explicit spectral distribution statistics |
| **Total**               | **3,456** |                                           |

Your MERT is especially important: MERT was designed specifically for acoustic music understanding, using musical and acoustic pseudo-labeling and reporting results across 14 music-understanding tasks. ([arXiv][7])

The MERT-v1-95M checkpoint is currently listed as **CC-BY-NC-4.0**, so the model license must be treated separately from the license of your derived feature matrix. ([Hugging Face][8])

### Lyrics

Already present:

* 5 basic statistics
* TextBlob polarity/subjectivity
* MiniLM 384-D
* MPNet 768-D
* first-3,000-character truncation

So the current lyric baseline is **1,159 dimensions + statistics**, not an empty lyric pipeline. 

The major weakness is not "you need an embedding."

It is:

> **You currently have a good English-oriented semantic sentence embedding applied to an unusually long, structured, multilingual document.**

That is a mismatch.

### Dataset limitations

Your 10,000 tracks are a **popular-song convenience corpus**, not a representative sample. Consequently:

* popularity prediction is subject to severe selection bias;
* genre distributions are heavily skewed;
* artist duplication is likely;
* artist popularity is correlated with selection;
* retrieval evaluation must not be interpreted as general music retrieval;
* a model can perform well simply because it recognizes artist/style clusters.

The brief explicitly identifies the corpus as a July-2025 popular-song corpus. 

---

# 3. Candidate matrix: audio

## Ranking

| Rank  | Candidate                        |             Output | Complementarity      | 6GB GPU       | Prediction | Retrieval | Decision          |
| ----- | -------------------------------- | -----------------: | -------------------- | ------------- | ---------- | --------- | ----------------- |
| **1** | Structured MIR                   |   ~50–300 selected | **Very high**        | Excellent     | ★★★★★      | ★★★★      | **DO**            |
| **2** | MusicFM                          |    model-dependent | High                 | Pilot         | ★★★★       | ★★★★      | **DO**            |
| **3** | CLAP                             |    model-dependent | High for cross-modal | Pilot         | ★★★        | ★★★★★     | **DO separately** |
| **4** | Basic Pitch descriptors          | compact engineered | High                 | Good          | ★★★★       | ★★★       | **Pilot**         |
| **5** | Vocal/source-derived descriptors |            compact | High                 | Moderate/poor | ★★★★       | ★★★       | **Pilot later**   |
| 6     | BEATs                            |    model-dependent | Medium/low           | Possible      | ★★★        | ★★★       | Defer             |
| 7     | AudioMAE                         |    model-dependent | Medium               | Poorer        | ★★★        | ★★★       | Defer             |
| 8     | HuBERT/wav2vec2                  |    768-ish typical | Low                  | Possible      | ★★         | ★★        | Reject initially  |
| 9     | Jukebox representations          |       huge/awkward | Unknown              | **Poor**      | ★★         | ★★★       | Reject            |
| 10    | More mel/MFCC statistics         |              small | Low                  | Excellent     | ★★         | ★★        | Only as QC        |

---

## 3.1 Structured MIR — **highest priority**

This is the biggest gap in your current representation.

Your current embeddings answer approximately:

> "What does this audio sound statistically/semantically like?"

Structured MIR features can answer:

> "What is its rhythm doing, how stable is its tempo, how does its harmony evolve, how repetitive is its structure, how dynamic is it, and where do important musical events occur?"

Those are much easier to interpret.

### Recommended feature groups

#### Rhythm

Extract:

* global BPM
* median/local BPM
* BPM mean/std
* BPM slope
* beat interval variance
* beat confidence
* onset rate
* onset strength mean/std
* onset-density quantiles
* rhythmic stability
* tempo-change count
* beat-synchronous energy statistics

#### Harmony

Extract:

* chroma mean/std
* CQT chroma
* HPCP
* estimated key
* key strength/confidence
* mode
* harmonic-change rate
* chord entropy
* tonal centroid / Tonnetz
* chroma self-similarity

`librosa` exposes CQT chroma, CENS, STFT chroma, MFCC, RMS, spectral centroid and related descriptors directly. ([Librosa][1])

#### Timbre

You already have mel statistics, so **do not dump another giant spectral matrix**.

Instead extract compact summaries:

* MFCC 1–20 mean/std
* spectral centroid mean/std
* bandwidth
* rolloff
* flatness
* contrast
* zero-crossing rate
* spectral flux

#### Dynamics

Particularly useful:

* RMS distribution
* loudness distribution
* dynamic complexity
* crest factor
* peak/RMS ratio
* silence percentage
* low-energy fraction
* intro loudness
* outro loudness

Essentia explicitly provides dynamic-complexity and related music descriptors. ([GitHub][9])

#### Structure

This is one of my strongest recommendations.

Extract:

* number of detected sections
* section duration distribution
* novelty-curve statistics
* novelty peak count
* repetition ratio
* self-similarity statistics
* intro duration
* outro duration
* first major transition time
* repeated-section count
* structural entropy

Essentia exposes segmentation-related descriptors as well. ([GitHub][9])

### Why this should beat another embedding

Your existing MERT vector is already 768-D and was trained specifically for music understanding. Adding another 768-D generic representation may produce correlated information.

A 100-D rhythm/harmony/structure vector can therefore be much more valuable.

**Recommendation: 150–250 carefully selected structured features, not 2,000 raw descriptors.**

### Hardware

CPU is entirely reasonable.

You do not need the GTX 1660 Ti for most classical MIR extraction.

For 10k four-minute tracks, exact runtime depends strongly on decoding, FFT configuration, disk speed and whether features are calculated once or repeatedly. **Do not trust an absolute runtime estimate before benchmarking 100 files.**

Pilot:

```text
100 tracks
↓
measure wall time
↓
time_per_song × 10,000
↓
record CPU/RAM/disk utilization
```

This is the first extraction I would run.

---

# 3.2 MusicFM — **highest-priority neural experiment**

MusicFM is specifically described by its authors as a **Foundation Model for Music Informatics** and was presented at ICASSP 2024. ([GitHub][2])

That makes it substantially more interesting for your project than simply adding BEATs.

### Expected information

Potentially:

* musical semantics
* timbre
* rhythm
* harmony
* musical structure
* genre/style

The critical question is **incremental information over MERT**, not whether MusicFM is "good."

### Experiment

Do not concatenate immediately.

Run:

```text
Existing audio
    ├── baseline
    ├── + MusicFM
    └── MusicFM alone
```

Then compare:

* popularity
* danceability
* energy
* valence
* genre
* audio retrieval

If:

```text
MERT + MusicFM ≈ MERT
```

for your tasks, discard it.

If:

```text
MERT + MusicFM > MERT
```

consistently across artist-aware splits, keep it.

### Licensing

The repository provides downloadable pretrained checkpoints, but you should verify the exact checkpoint's license before redistribution. Do not assume that "GitHub repository is public" means the weights are unrestricted. ([GitHub][2])

---

# 3.3 CLAP — **excellent retrieval candidate, not necessarily a prediction feature**

LAION-CLAP explicitly produces representations for **audio and text in a shared latent space**. ([GitHub][3])

This is fundamentally different from MERT.

MERT:

```text
audio → music representation
```

CLAP:

```text
audio ↔ natural-language concept
```

That makes CLAP particularly interesting for your future website.

For example:

> "female vocal, melancholic, acoustic, intimate"

can potentially be embedded into the same space as audio.

### Best use

**Do not concatenate CLAP into the 4,254-D regression matrix initially.**

Store separately:

```text
audio_clap.npy
text_clap_queries/
```

Then build:

```text
song → CLAP vector
query text → CLAP vector
```

This gives you natural-language music discovery.

The LAION repository is currently CC0-1.0 at the software/repository level, while the model's training-data provenance and released checkpoint specifics still require separate scrutiny. The authors explicitly state that their full training dataset cannot be released for copyright reasons. ([GitHub][3])

---

# 3.4 BEATs — defer

BEATs is a legitimate high-quality audio SSL system with multiple pretrained iterations and AudioSet fine-tuned variants. ([GitHub][10])

The problem is **redundancy**.

You already have:

* PANNs trained on AudioSet;
* VGGish trained around AudioSet;
* MERT trained for music.

BEATs is likely to add some acoustic information, but I would not spend your limited extraction budget testing it before structured MIR.

**Verdict: Tier 2 pilot only.**

---

# 3.5 AudioMAE — defer

AudioMAE is a masked-autoencoder audio representation with music, speech and event demonstrations and released ViT-B checkpoints. ([GitHub][11])

But it is another general audio representation.

Your marginal-value ordering should be:

```text
structured musical information
        ↓
music-specific neural representation
        ↓
cross-modal representation
        ↓
another general audio SSL model
```

not the reverse.

**Verdict: Tier 3.**

---

# 3.6 Jukebox — reject

Jukebox is fascinating scientifically, but wrong for this project.

The official repository is archived and says the code is provided as-is. ([GitHub][5])

More importantly, its license is explicitly **Noncommercial Use**, covering the code and weights. ([GitHub][12])

Its original environment also targets extremely old Python/PyTorch/CUDA versions, making integration into your current laptop environment unattractive. ([GitHub][5])

**Verdict: do not prioritize.**

---

# 3.7 Source separation

Demucs is a strong source-separation baseline; the official repository reports substantially improved SDR for later Hybrid/HT Demucs variants and the code is MIT licensed. ([GitHub][13])

But:

```text
mix
 ↓
separation
 ↓
4 stems
 ↓
4 × feature extraction
```

is expensive.

And separation is imperfect.

Therefore:

### Do NOT store stems as your primary dataset.

Instead:

```text
vocals:
    RMS
    percentage active
    spectral centroid
    pitch statistics

drums:
    onset density
    rhythmic energy

bass:
    low-frequency energy
    pitch stability

other:
    spectral/timbral statistics
```

The resulting feature vector might be only 50–150 dimensions.

### Pilot

Use 100 songs:

* 25 pop
* 25 hip-hop/rap
* 25 electronic
* 25 acoustic/rock

Measure:

```text
Δ prediction
Δ retrieval
runtime/song
OOM/failure rate
```

If stem features don't improve anything materially, discard the entire branch.

---

# 3.8 Singing/melody: Basic Pitch

This is more attractive.

Spotify's Basic Pitch is specifically designed for lightweight polyphonic note transcription and pitch-bend detection. It supports multiple instruments, although it works best on one instrument at a time. ([GitHub][6])

It is Apache-2.0 licensed. ([GitHub][14])

Don't store MIDI as your main feature.

Instead derive:

```text
note_count
note_density
median_pitch
pitch_range
pitch_std
pitch_entropy
melodic_interval_mean
melodic_interval_std
ascending_ratio
descending_ratio
repeated_note_ratio
pitch_bend_ratio
estimated_melodic_activity
```

Potentially extremely useful for:

* vocal style
* melody similarity
* song characterization
* genre prediction

But it is **not** a substitute for a singing-voice separation/vocal model.

---

# 3.9 Temporal pooling

This deserves much more attention than adding another model.

Your MERT currently uses a temporal mean over the first 30 seconds. 

That means:

> a 4-minute song is represented partly by its first 30 seconds.

This is a significant limitation.

Test:

### A

```text
first 30s mean
```

### B

```text
intro
middle
outro
```

### C

```text
0–30
30–60
60–90
...
```

### D

```text
intro + verse + chorus + bridge + outro
```

The best compromise for your dataset is likely:

```text
intro mean
early-middle mean
middle mean
late-middle mean
outro mean
```

For a 768-D MERT representation:

```text
5 × 768 = 3,840 dimensions
```

which is too large for naive concatenation.

Instead use:

```text
mean across windows
std across windows
max across windows
```

giving:

```text
3 × 768 = 2,304
```

or reduce each temporal statistic to 128–256 dimensions with PCA **fit only on training data**.

For retrieval, however, I would preserve the segment vectors separately.

---

# 4. Candidate matrix: lyrics

## My ranking

| Rank  | Method                   |           D | Multilingual     | Long lyrics  | Retrieval         | Prediction | Recommendation          |
| ----- | ------------------------ | ----------: | ---------------- | ------------ | ----------------- | ---------- | ----------------------- |
| **1** | BGE-M3                   |        1024 | **Yes**          | **8192**     | ★★★★★             | ★★★★       | **DO**                  |
| **2** | GTE-multilingual-base    |         768 | **75 languages** | **8192**     | ★★★★★             | ★★★★       | **DO**                  |
| **3** | multilingual-e5-base     |         768 | 100-ish          | 512          | ★★★★              | ★★★★       | Pilot                   |
| **4** | Jina v3                  |        1024 | **94**           | long-context | ★★★★★             | ★★★★       | License caution         |
| 5     | MPNet current            |         768 | weak             | 512-ish      | ★★★               | ★★★★       | Baseline                |
| 6     | MiniLM current           |         384 | weak             | short        | ★★                | ★★★        | Baseline                |
| 7     | Nomic v1.5               | 768/512/256 | English          | 8192         | ★★★★              | ★★★        | English-only experiment |
| 8     | lyric-domain fine-tuning |      custom | depends          | depends      | potentially ★★★★★ | ★★★★       | Later                   |

---

## 4.1 BGE-M3 — first lyric experiment

BGE-M3 provides:

* **1,024-D dense embeddings**
* **8,192-token context**
* multilingual support
* dense/sparse/ColBERT-style retrieval capabilities
* MIT license. ([Hugging Face][4])

This directly attacks your biggest existing problem:

```text
lyrics
  ↓
first 3,000 characters
  ↓
MPNet
```

Instead:

```text
full cleaned lyric
       ↓
BGE-M3
       ↓
1024-D
```

But I would actually go one step further.

### Recommended lyric representation

```text
title
+
artist-independent lyric body
+
section markers
```

Then:

```text
full lyric embedding
```

plus:

```text
section embeddings
```

For example:

```text
intro
verse
chorus
verse
chorus
bridge
chorus
outro
```

Compute:

```text
E_full

E_verse
E_chorus
E_bridge

mean(E_sections)
std(E_sections)
```

For retrieval, keep section vectors separately.

For supervised prediction, use only a compact aggregation.

---

# 4.2 GTE multilingual base — probably the best compact alternative

GTE multilingual base is currently listed as:

* **75 languages**
* **768-D**
* **8,192-token context**
* Apache-2.0. ([Hugging Face][15])

This is especially attractive for your hardware.

It is smaller than BGE-M3 while still providing long-context multilingual encoding.

I would run:

```text
BGE-M3
vs
GTE-multilingual-base
vs
current MPNet
```

on exactly the same lyric evaluation set.

Do **not** assume BGE wins because its model is larger.

---

# 4.3 multilingual-E5

The multilingual-E5 family remains a strong retrieval baseline. The currently documented large model has 1024 dimensions and a 512-token maximum, while the base model is 768-dimensional. ([Hugging Face][16])

The problem for lyrics is obvious:

```text
512 tokens
```

is substantially shorter than many complete lyrics.

Therefore:

### Use E5 only with chunking.

For example:

```text
lyrics
 ↓
512-token chunks
 ↓
embedding each chunk
 ↓
mean + max + attention-like weighted aggregation
```

It is not my first choice for your corpus.

---

# 4.4 Jina embeddings v3

Jina v3 supports **94 languages**, is a 0.6B-parameter model, and is listed as CC-BY-NC-4.0. ([Hugging Face][17])

Technically attractive.

Licensing-wise:

> **I would not make it the default Kaggle representation.**

The noncommercial model license introduces unnecessary friction if the goal is broad redistribution.

---

# 4.5 Nomic

Nomic Embed v1.5 is interesting because it supports **8192 tokens** and Matryoshka dimensionality reduction:

```text
768
512
256
128
64
```

with published MTEB performance at each size. ([Hugging Face][18])

It is Apache-2.0. ([Hugging Face][18])

But it is **English-focused**, so it should not replace your multilingual candidate.

It is a good English-only retrieval baseline.

---

# 4.6 Lyric preprocessing

This is arguably more important than choosing between BGE and GTE.

## Pipeline

```text
raw lyrics
   ↓
Unicode normalization
   ↓
source boilerplate removal
   ↓
section-header normalization
   ↓
repetition detection
   ↓
language identification
   ↓
code-switch detection
   ↓
clean lyric
   ↓
section parser
   ↓
embedding
```

### Do not simply delete repeated choruses

Repetition itself is musically meaningful.

Instead create two versions:

```text
lyrics_raw_structure
lyrics_deduplicated_semantic
```

Example:

```text
chorus × 5
```

can become:

```text
semantic_embedding = one chorus
structural_feature = chorus_repeat_count = 5
```

This is much better than deleting repetition entirely.

---

# 5. Lyrics benchmark answer

## Direct answer

**No.**

There is no broadly accepted benchmark that can legitimately answer:

> "Model X is the best lyric representation for lyrical similarity on my 10,000-song corpus."

The existing evaluation landscape is fragmented across:

* semantic similarity;
* emotion;
* theme;
* genre;
* language;
* music/audio understanding;
* multimodal audio-text understanding.

That is fundamentally different from your desired task:

> **lyrics of song A are more lyrically similar to song B than to song C.**

Multimodal music benchmarks exist, but they do not solve this exact problem.

For example, MuChoMusic is explicitly a benchmark for **multimodal music understanding**, with human-validated multiple-choice questions associated with music tracks. It is useful for general music understanding, not direct lyric-embedding selection. ([GitHub][19])

The Song Describer Dataset contains roughly 1.1k captions for 706 permissively licensed music recordings and is useful for audio-language evaluation, but again it is not a lyric-semantic-similarity benchmark. ([GitHub][20])

MTG-Jamendo is much more useful for evaluating music tagging. It contains over 55,000 tracks and 195 tags spanning genre, instrument and mood/theme. ([GitHub][21])

Therefore:

> **Build your own in-domain lyric retrieval benchmark.**

---

# 5.1 Human evaluation set

Take:

### 300 query songs

Stratify by:

* language
* genre
* popularity
* artist
* lyric length

For each query:

```text
1 query
+
10 candidate songs
```

Candidates:

* 2 highly similar
* 2 moderately similar
* 2 same genre but different theme
* 2 same theme but different genre
* 2 deliberately unrelated

This gives:

```text
300 × 10 = 3,000 pair judgments
```

But do not show annotators artist names.

---

# 5.2 Rating dimensions

Ask annotators to score 1–5:

1. **Theme similarity**
2. **Emotional similarity**
3. **Narrative similarity**
4. **Imagery/concept similarity**
5. **Overall lyrical similarity**

This is superior to one vague question because you can discover *what the embedding actually represents*.

For example:

```text
BGE:
theme 4.3
emotion 4.1
imagery 3.7
```

versus:

```text
MPNet:
theme 4.0
emotion 3.8
imagery 3.2
```

---

# 5.3 Retrieval metrics

For each representation:

```text
Recall@5
Recall@10
nDCG@10
MAP@10
MRR
```

Use nDCG as the primary metric because your judgments are graded.

---

# 5.4 Critical leakage controls

Never allow:

```text
same artist → train and test
```

for your primary retrieval benchmark.

Also control:

* same album
* duplicate lyrics
* translated versions
* remixes
* covers
* near-identical lyrics
* repeated songs
* artist collaborations

Otherwise the model can simply exploit artist identity.

---

# 6. Website/product concepts

## 1. Controllable similarity search

**Highest-value product.**

Instead of:

> Find songs similar to X.

Provide:

```text
Audio similarity       40%
Lyrics                 25%
Mood                   15%
Rhythm                 10%
Harmony                5%
Vocal                  5%
```

Then:

> "Find songs like X, but calmer and lyrically darker."

Representation:

```text
audio embedding
+
structured MIR
+
lyric embedding
+
metadata
```

At 10k:

**trivial.**

At 1M:

still practical with ANN indexes.

---

## 2. "Why are these songs similar?"

Show:

```text
Audio
████████░░  82%

Lyrics
█████████░  91%

Rhythm
███████░░░  74%

Harmony
████████░░  81%

Mood
█████████░  89%
```

This is much more defensible than:

> "AI says these songs are similar."

---

## 3. Lyric-theme explorer

Cluster lyric embeddings.

Examples:

```text
love / separation
nostalgia
self-confidence
party
religion/spirituality
social anxiety
money/status
nightlife
family
politics
```

Do not claim these clusters are objective truths.

Use:

> "semantic cluster"

not:

> "the meaning of this song."

---

## 4. Audio–lyric mismatch explorer

This could be genuinely interesting.

Calculate:

```text
audio_mood_vector
lyric_emotion_vector
```

Then:

```text
distance(audio, lyric)
```

Examples:

> Happy production + sad lyrics

> Aggressive production + vulnerable lyrics

> Calm acoustic track + euphoric lyrics

This is much more interesting scientifically than another generic song map.

---

## 5. Near-duplicate / cover explorer

Use:

* audio fingerprints
* spectral fingerprints
* embedding similarity
* metadata

Then detect:

```text
original
cover
remaster
live version
sped-up version
slowed version
duplicate download
```

This also improves dataset quality.

---

# 7. Kaggle-ready data architecture

I would **not** publish one giant CSV containing 4,000+ columns.

Use:

```text
dataset/
├── README.md
├── LICENSE
├── CITATION.cff
├── data_dictionary.csv
├── manifest.csv
│
├── metadata/
│   ├── songs.parquet
│   ├── artists.parquet
│   └── albums.parquet
│
├── features/
│   ├── audio/
│   │   ├── mert_v1_95m.npy
│   │   ├── panns_cnn14.npy
│   │   ├── vggish.npy
│   │   ├── mel_stats.npy
│   │   ├── musicfm.npy
│   │   └── structured_mir.npy
│   │
│   ├── lyrics/
│   │   ├── mpnet.npy
│   │   ├── minilm.npy
│   │   ├── bge_m3.npy
│   │   └── gte_multilingual.npy
│   │
│   └── derived/
│       ├── vocal_features.npy
│       ├── melody_features.npy
│       └── qc_features.npy
│
├── ids/
│   └── track_id.npy
│
├── manifests/
│   ├── feature_manifest.json
│   ├── extraction_manifest.json
│   └── checksums.sha256
│
└── evaluation/
    ├── splits.csv
    ├── retrieval_queries.csv
    └── annotations.csv
```

### Every feature array needs

```json
{
  "feature_name": "mert_v1_95m",
  "shape": [10000, 768],
  "dtype": "float32",
  "id_file": "ids/track_id.npy",
  "model": "m-a-p/MERT-v1-95M",
  "model_revision": "...",
  "sample_rate": 24000,
  "pooling": "mean",
  "created_at": "...",
  "sha256": "..."
}
```

---

## What I would not publish

### Raw lyrics

Unless you have explicit redistribution rights.

Your own preprocessing does not make copyrighted lyrics yours.

### Downloaded YouTube audio

Do not package it as a Kaggle dataset merely because it was technically downloadable.

Your brief explicitly says the audio is usually YouTube-source Opus/WebM. 

### Raw stems

Same provenance problem.

### Safe-to-publish candidates

Potentially:

* metadata you have redistribution rights for;
* your own numerical derived features;
* model outputs where the model license permits redistribution;
* feature manifests;
* checksums;
* evaluation code;
* preprocessing code.

But **"derived vector" does not automatically mean legally unrestricted**. Treat each model's license separately.

---

# 8. Staged 10k-song roadmap

## Tier 0 — almost free

### T0.1 Audio QC

Extract:

```text
duration
sample_rate
channels
codec
bitrate
peak
RMS
clipping ratio
silence ratio
DC offset
loudness
```

### T0.2 Duplicate detection

Use:

```text
duration
spectral fingerprint
audio hash
embedding similarity
```

### T0.3 Lyrics QC

Add:

```text
language
language_confidence
code_switch_ratio
lyric_length
section_count
repetition_ratio
missing_reason
```

**Pilot:** entire 10k.

---

# Tier 1 — highest ROI

| Priority | Feature               | Pilot |           Storage | Decision                          |
| -------- | --------------------- | ----: | ----------------: | --------------------------------- |
| **1**    | Structured MIR        |   100 |           <~50 MB | Must improve retrieval/prediction |
| **2**    | BGE-M3 lyrics         |   100 | ~0.4 MB for 1000? | Compare retrieval                 |
| **3**    | GTE multilingual      |   100 |             small | Compare against BGE               |
| **4**    | MERT temporal pooling |   100 |          moderate | Compare against first-30s         |
| **5**    | CLAP                  |   100 |          moderate | Evaluate cross-modal retrieval    |

### Important storage calculation

A float32 vector costs:

```text
dimensions × 4 bytes
```

Therefore:

```text
768 × 4 × 10,000 ≈ 30.7 MB
1024 × 4 × 10,000 ≈ 41.0 MB
2048 × 4 × 10,000 ≈ 81.9 MB
```

So storage is **not** your major problem at 10k.

Your major problem is extraction time and experimental complexity.

---

# Tier 2 — pilot only

Test:

* MusicFM
* Basic Pitch
* Demucs stem descriptors
* BEATs

Only promote one to full extraction if it passes a predefined gate.

### Example gate

Promote if:

```text
ΔnDCG@10 ≥ 0.03
```

or:

```text
ΔR² ≥ 0.02
```

**and**

improvement survives artist-aware/duplicate-aware validation.

Those thresholds are **project heuristics, not published universal standards**.

---

# Tier 3

Defer:

* Jukebox
* giant temporal embeddings
* large-scale lyric fine-tuning
* AudioMAE
* HuBERT
* wav2vec2
* elaborate source separation
* generative LLM annotation at 10k scale

---

# 9. Ablation/evaluation protocol

This is the most important part of the experiment.

## Baseline A — current audio

```text
VGGish
+
MERT
+
PANNs
+
mel stats
```

## Baseline B — current lyrics

```text
MiniLM
+
MPNet
+
5 stats
+
TextBlob
```

## Baseline C — metadata

```text
Spotify descriptors
+
artist metadata
+
genre
+
release information
```

## Candidate additions

```text
+ MIR
+ MusicFM
+ CLAP
+ BGE
+ GTE
+ temporal MERT
+ Basic Pitch
```

---

## Prediction split

Do **not** use:

```text
random train_test_split
```

as your only result.

Use:

### Split 1

Random stratified baseline.

### Split 2

Artist-group split.

### Split 3

Album-group split.

### Split 4

Temporal split.

For example:

```text
train: older releases
validation: middle period
test: newest period
```

This is particularly important for popularity.

---

# Retrieval evaluation

### Audio

Create pseudo-positive labels from:

* manually judged similarity;
* genre-compatible similarity;
* controlled metadata groups.

But manual labels should be the primary result.

### Lyrics

Human judgments are essential.

### Metrics

```text
Recall@5
Recall@10
nDCG@10
MAP@10
MRR
```

---

# Dimensionality reduction

Never:

```text
PCA.fit(all_data)
```

before splitting.

Correct:

```python
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
```

Same rule for:

* normalization parameters
* feature selection
* imputation
* learned pooling
* clustering-derived features.

---

# Confidence intervals

Use bootstrap confidence intervals for:

```text
MAE
RMSE
R²
nDCG
Recall@K
```

A model that improves:

```text
nDCG 0.421 → 0.424
```

is not automatically meaningfully better.

You need uncertainty.

---

# 10. Risk register

| Risk                            | Severity     | Mitigation                                                 |
| ------------------------------- | ------------ | ---------------------------------------------------------- |
| Copyrighted lyrics              | **Critical** | Don't redistribute raw lyrics                              |
| YouTube audio provenance        | **Critical** | Keep audio local; publish provenance/IDs where appropriate |
| Model license                   | High         | Record exact checkpoint + revision + license               |
| Artist leakage                  | **Critical** | Group splits                                               |
| Duplicate tracks                | **Critical** | fingerprint + duration + metadata QC                       |
| Popularity leakage              | **Critical** | remove target-derived/post-target variables                |
| Multilingual bias               | High         | language-stratified evaluation                             |
| Lyrics truncation               | High         | full/section-aware embeddings                              |
| Embedding version drift         | High         | pin model revision/checksum                                |
| GPU OOM                         | Medium       | batch=1, chunking, CPU fallback                            |
| Extraction failure              | Medium       | resumable manifest                                         |
| NaN/Inf                         | Medium       | row-level validation                                       |
| Storage growth                  | Low at 10k   | separate matrices                                          |
| Source separation artifacts     | Medium       | stem-quality pilot                                         |
| Predicted tags treated as truth | High         | store probability + `predicted`, never ground-truth        |
| LLM hallucinated labels         | Medium/high  | schema validation + human audit                            |
| Popularity corpus bias          | High         | explicitly describe corpus as convenience sample           |

---

# 11. Annotated bibliography

### Core music representation

* **MERT — Acoustic Music Understanding Model with Large-Scale Self-supervised Training.** The foundational MERT paper; reports evaluation over 14 music-understanding tasks and 95M/330M configurations. ([arXiv][7])
  [MERT paper](https://arxiv.org/abs/2306.00107?utm_source=chatgpt.com)

* **MERT-v1-95M model card.** Current checkpoint information, model files and CC-BY-NC-4.0 license. ([Hugging Face][8])
  [MERT-v1-95M model card](https://huggingface.co/m-a-p/MERT-v1-95M?utm_source=chatgpt.com)

* **MusicFM.** Music-specific foundation model for music informatics; ICASSP 2024. ([GitHub][2])
  [MusicFM repository](https://github.com/minzwon/musicfm?utm_source=chatgpt.com)

### General audio

* **PANNs.** AudioSet-pretrained audio neural networks; official repository reports AudioSet training and Cnn14/PANN variants. ([GitHub][22])
  [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn?utm_source=chatgpt.com)

* **BEATs.** Acoustic-tokenizer-based audio SSL with multiple pretrained iterations. ([GitHub][10])
  [BEATs repository](https://github.com/microsoft/unilm/tree/master/beats?utm_source=chatgpt.com)

* **AudioMAE.** Masked-autoencoder audio representation; ViT-B AudioSet checkpoints. ([GitHub][11])
  [AudioMAE repository](https://github.com/facebookresearch/AudioMAE?utm_source=chatgpt.com)

### Audio-text

* **LAION-CLAP.** Contrastive language-audio representation with released audio/text embeddings and music/speech checkpoints. ([GitHub][3])
  [LAION-CLAP repository](https://github.com/LAION-AI/CLAP?utm_source=chatgpt.com)

* **MuChoMusic.** Human-validated multimodal music-understanding benchmark. MIT code / CC-BY-SA-4.0 dataset. ([GitHub][19])
  [MuChoMusic repository](https://github.com/mulab-mir/muchomusic?utm_source=chatgpt.com)

* **Song Describer Dataset.** Approximately 1.1k captions for 706 permissively licensed music recordings. ([GitHub][20])
  [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset?utm_source=chatgpt.com)

### Lyrics/text

* **BGE-M3.** 1,024-D multilingual long-context model with 8,192-token context and MIT license. ([Hugging Face][4])
  [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com)

* **GTE multilingual base.** 768-D, 8,192-token, 75-language Apache-2.0 embedding model. ([Hugging Face][15])
  [GTE multilingual base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base?utm_source=chatgpt.com)

* **Multilingual-E5.** Strong multilingual retrieval baseline; the large model is 1,024-D with 512-token maximum length and MIT license. ([Hugging Face][23])
  [multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large?utm_source=chatgpt.com)

* **Jina embeddings v3.** 94-language embedding model; 0.6B parameters; CC-BY-NC-4.0. ([Hugging Face][17])
  [Jina embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3?utm_source=chatgpt.com)

* **Nomic Embed v1.5.** Apache-2.0 long-context embedding with Matryoshka dimensionality options. ([Hugging Face][18])
  [Nomic Embed v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5?utm_source=chatgpt.com)

### MIR / symbolic audio

* **librosa feature extraction.** Current documentation for chroma/CQT, MFCC, RMS, spectral descriptors and related MIR primitives. ([Librosa][1])
  [librosa feature documentation](https://librosa.org/doc/latest/feature.html?utm_source=chatgpt.com)

* **Essentia.** Music-information-retrieval library and pretrained-model ecosystem; descriptors include segmentation, voice/instrumental and dynamic complexity. ([Essentia][24])
  [Essentia documentation](https://essentia.upf.edu/documentation.html?utm_source=chatgpt.com)

* **Basic Pitch.** Lightweight polyphonic note transcription/pitch-bend model from Spotify; Apache-2.0. ([GitHub][6])
  [Basic Pitch repository](https://github.com/spotify/basic-pitch?utm_source=chatgpt.com)

### Source separation

* **Demucs.** Hybrid waveform/spectrogram source separation; official results include HT Demucs v4 and MIT code license. ([GitHub][13])
  [Demucs repository](https://github.com/facebookresearch/demucs?utm_source=chatgpt.com)

* **Spleeter.** Deezer source-separation system; MIT code license, with an explicit warning concerning copyrighted material. ([GitHub][25])
  [Spleeter repository](https://github.com/deezer/spleeter?utm_source=chatgpt.com)

### Datasets

* **MTG-Jamendo.** >55k tracks, 195 tags covering genre/instrument/mood-theme; audio is individually CC licensed and metadata is CC-BY-NC-SA; research-use restrictions apply. ([GitHub][21])
  [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset?utm_source=chatgpt.com)

* **Jukebox.** OpenAI's archived generative music model; code and weights are under a Noncommercial Use License. ([GitHub][5])
  [Jukebox repository](https://github.com/openai/jukebox?utm_source=chatgpt.com)

---

# Bottom line

If this were **my 10,000-song experiment on a GTX 1660 Ti 6 GB / 16 GB RAM**, I would **not** spend the next week downloading ten foundation models.

I would execute this exact order:

```text
                    CURRENT DATA
                         │
             ┌───────────┴───────────┐
             │                       │
         AUDIO                       LYRICS
             │                       │
     ┌───────┴───────┐       ┌───────┴────────┐
     │               │       │                │
 Structured MIR   MERT      BGE-M3       GTE-multilingual
     │               │       │                │
     └───────┬───────┘       └────────┬───────┘
             │                        │
             └──────────┬─────────────┘
                        │
                  EVALUATION
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Prediction     Audio IR      Lyric IR
          │             │             │
          └─────────────┼─────────────┘
                        │
                  only then:
                        │
             ┌──────────┼───────────┐
             │          │           │
          MusicFM      CLAP     Basic Pitch
             │          │           │
             └──────────┴───────────┘
```

The **single most important architectural change** I recommend is to stop thinking of the dataset as one giant feature vector.

Maintain three explicit representation spaces:

```text
PREDICTION SPACE
    compact + interpretable + PCA/feature-selected

AUDIO RETRIEVAL SPACE
    MERT/MusicFM/CLAP + structured MIR

LYRIC RETRIEVAL SPACE
    multilingual long-context embeddings + section structure
```

That architecture will be substantially more useful for both the thesis experiments and the eventual music-discovery website than simply growing your current 4,254-dimensional vector to 8,000–15,000 dimensions.

[1]: https://librosa.org/doc/latest/feature.html?utm_source=chatgpt.com "Feature extraction — librosa 0.11.0 documentation"
[2]: https://github.com/minzwon/musicfm?utm_source=chatgpt.com "GitHub - minzwon/musicfm · GitHub"
[3]: https://github.com/LAION-AI/CLAP?utm_source=chatgpt.com "GitHub - LAION-AI/CLAP: Contrastive Language-Audio Pretraining · GitHub"
[4]: https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com "BAAI/bge-m3 · Hugging Face"
[5]: https://github.com/openai/jukebox?utm_source=chatgpt.com "GitHub - openai/jukebox: Code for the paper \"Jukebox: A Generative Model for Music\" · GitHub"
[6]: https://github.com/spotify/basic-pitch?utm_source=chatgpt.com "GitHub - spotify/basic-pitch: A lightweight yet powerful audio-to-MIDI converter with pitch bend detection · GitHub"
[7]: https://arxiv.org/abs/2306.00107?utm_source=chatgpt.com "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training"
[8]: https://huggingface.co/m-a-p/MERT-v1-95M/tree/main?utm_source=chatgpt.com "m-a-p/MERT-v1-95M at main"
[9]: https://github.com/MTG/essentia/blob/master/doc/descriptors.yaml?utm_source=chatgpt.com "essentia/doc/descriptors.yaml at master · MTG/essentia · GitHub"
[10]: https://github.com/microsoft/unilm/blob/master/beats/README.md?utm_source=chatgpt.com "unilm/beats/README.md at master · microsoft/unilm · GitHub"
[11]: https://github.com/facebookresearch/AudioMAE?utm_source=chatgpt.com "GitHub - facebookresearch/AudioMAE: This repo hosts the code and models of \"Masked Autoencoders that Listen\". · GitHub"
[12]: https://github.com/openai/jukebox/blob/master/LICENSE?utm_source=chatgpt.com "jukebox/LICENSE at master · openai/jukebox · GitHub"
[13]: https://github.com/facebookresearch/demucs?utm_source=chatgpt.com "GitHub - facebookresearch/demucs: Code for the paper Hybrid Spectrogram and Waveform Source Separation · GitHub"
[14]: https://github.com/spotify/basic-pitch/blob/main/setup.py?utm_source=chatgpt.com "basic-pitch/setup.py at main · spotify/basic-pitch · GitHub"
[15]: https://huggingface.co/Alibaba-NLP/gte-multilingual-base/blob/main/README.md?utm_source=chatgpt.com "README.md · Alibaba-NLP/gte-multilingual-base at main"
[16]: https://huggingface.co/Teradata/multilingual-e5-base?utm_source=chatgpt.com "Teradata/multilingual-e5-base · Hugging Face"
[17]: https://huggingface.co/jinaai/jina-embeddings-v3?utm_source=chatgpt.com "jinaai/jina-embeddings-v3 · Hugging Face"
[18]: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5?utm_source=chatgpt.com "nomic-ai/nomic-embed-text-v1.5 · Hugging Face"
[19]: https://github.com/mulab-mir/muchomusic?utm_source=chatgpt.com "GitHub - mulab-mir/muchomusic: MuChoMusic is a benchmark for evaluating music understanding in multimodal audio-language models. · GitHub"
[20]: https://github.com/mulab-mir/song-describer-dataset?utm_source=chatgpt.com "GitHub - mulab-mir/song-describer-dataset: The Song Describer dataset is an evaluation dataset made of ~1.1k captions for 706 permissively licensed music recordings. · GitHub"
[21]: https://github.com/MTG/mtg-jamendo-dataset?utm_source=chatgpt.com "GitHub - MTG/mtg-jamendo-dataset: Metadata, scripts and baselines for the MTG-Jamendo dataset · GitHub"
[22]: https://github.com/qiuqiangkong/audioset_tagging_cnn?utm_source=chatgpt.com "GitHub - qiuqiangkong/audioset_tagging_cnn · GitHub"
[23]: https://huggingface.co/Teradata/multilingual-e5-large?utm_source=chatgpt.com "Teradata/multilingual-e5-large · Hugging Face"
[24]: https://essentia.upf.edu/documentation.html?utm_source=chatgpt.com "Overview — Essentia 2.1-beta6-dev documentation"
[25]: https://github.com/deezer/spleeter?utm_source=chatgpt.com "GitHub - deezer/spleeter: Deezer source separation library including pretrained models. · GitHub"
