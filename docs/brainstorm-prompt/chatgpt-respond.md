# Research-Backed MIR Feature-Extraction and Music Website Brainstorm Report

**Scope:** 10,000 top-streamed songs, July 2025 Spotify global chart, full audio available locally, multilingual lyrics, GTX 1660 Ti 6 GB VRAM + 16 GB RAM, no cloud GPU.

**Evidence cutoff:** **August 18, 2026**. Model specifications, benchmark names, licenses, and repository availability were checked against current project/model pages or primary papers. Extraction times and VRAM figures are **engineering estimates for your exact hardware**, not published benchmark measurements; assume roughly **±2×** depending on song length, CUDA/PyTorch version, thermals, and implementation.

## Executive verdict

Your existing feature bank is already unusually broad. The highest-return additions are **not another ten generic AudioSet encoders**.

The strongest research path is:

1. **MuQ ~300M** — currently the most compelling new music-specific SSL experiment for your machine.
2. **MuQ-MuLan 512-d** — unusually valuable because it gives you a music-text-aligned representation rather than another generic audio vector.
3. **CLaMP3 768-d** — especially interesting for your multilingual lyrics and cross-modal website, although its checkpoint is large and it internally depends on MERT features.
4. **A compact handcrafted MIR vector (~60–90 dims)** — cheap, interpretable and likely more useful than its dimensionality suggests for energy, harmony, tempo, structure and explainability.
5. **Better temporal pooling of the representations you already have** — probably the single cheapest experiment in the entire project.
6. For lyrics, **multilingual-e5-large, gte-multilingual-base, and a CLaMP3/MuQ-MuLan-style music-aware representation** are a much more sensible 2026 comparison than simply adding another English MPNet.
7. There is **no single de-facto universal lyric-similarity benchmark** in 2026. **LyricSIM** is the clearest dedicated benchmark for lyric semantic similarity, but it is Spanish-only and small. For audio representation evaluation, **MARBLE** is the closest community-standard benchmark. ([arXiv][1])

A crucial non-ML conclusion: **do not publicly publish the raw lyrics, Spotify-derived metadata dump, or derived Spotify-content feature dataset without resolving the licensing/contractual issue first.** Spotify's current Developer Terms explicitly prohibit using Spotify Content to train ML/AI models, and Spotify's Developer Policy says metadata must not be offered as a standalone service/product. Kaggle's Terms likewise require you to have the necessary rights for public submissions. ([Spotify for Developers][2])

---

# SECTION 1 — Audio Feature Extraction Ideas Beyond Your Existing 4,256-D Bank

## 1.1 What you already have — and what that means

Your existing representation is:

| Existing extractor            |       Dim | What it contributes                         |
| ----------------------------- | --------: | ------------------------------------------- |
| VGGish                        |       128 | General AudioSet perceptual/audio semantics |
| MERT-v1-95M                   |       768 | Music-specific SSL representation           |
| PANNs Cnn14                   |      2048 | AudioSet acoustic/event semantics           |
| Mel statistics                |       512 | Explicit spectral distribution              |
| **Total**                     | **3,456** | These four                                  |
| Plus your prior text features |   **800** | 5 stats + 2 sentiment + MPNet 768           |
| **Previous total**            | **4,256** | Your stated benchmark feature set           |

VGGish's 128-D embedding is explicitly the AudioSet embedding layer, while MERT-v1-95M is a 95M-parameter music SSL encoder producing 768-D representations. ([GitHub][3])

The major gap is therefore not "you have no embeddings." You have a lot of embeddings.

The missing axes are:

* **newer music SSL**
* **audio-text alignment**
* **multilingual cross-modal alignment**
* **temporal structure**
* **harmony**
* **beat/groove**
* **vocal/instrumental disentanglement**
* **interpretable handcrafted MIR descriptors**

---

## 1.2 Ranked pretrained audio models

### Tier A — actually worth spending GPU time on

| Rank    | Model / idea                      | Output                                       |                              Params |   VRAM est. | Time/song* | 10k total* | License                                    | Feasibility | Value                                    |
| ------- | --------------------------------- | -------------------------------------------- | ----------------------------------: | ----------: | ---------: | ---------: | ------------------------------------------ | ----------- | ---------------------------------------- |
| **1 ★** | **MuQ-large-msd-iter**            | 1024 global or temporal sequence             |                               ~300M |  2.5–4.5 GB |     5–15 s |    14–42 h | **CC-BY-NC-4.0 weights**                   | Easy/Med    | **Very high**                            |
| **2 ★** | **MuQ-MuLan**                     | **512-D aligned audio/text**                 |                               ~700M |    4–5.8 GB |     8–20 s |    22–56 h | **CC-BY-NC-4.0 weights**                   | Medium      | **Very high**                            |
| **3 ★** | **CLaMP3 SAAS**                   | 768 global / T×768                           | multi-component; checkpoint ~2.5 GB |    4–5.8 GB |     8–20 s |    22–56 h | MIT code; verify weight terms              | Medium      | **Very high for multilingual retrieval** |
| **4**   | **MERT-v1-330M**                  | 1024 temporal/global                         |                                330M | 2–4 GB fp16 |     5–15 s |    14–42 h | **CC-BY-NC-4.0**                           | Easy/Med    | High                                     |
| **5**   | **EnCodecMAE Base**               | temporal latent / pooled vector              |                  ~85M-class encoder |    1.5–3 GB |     3–10 s |     8–28 h | **MIT**                                    | Easy        | High                                     |
| **6**   | **BEATs**                         | temporal/global ~768                         |                     ~90M-class base |    1.5–3 GB |      2–6 s |     6–17 h | MIT code                                   | Easy        | Medium/high                              |
| **7**   | **Dasheng-Base**                  | **768-D**                                    |                                 86M |      1–2 GB |      2–6 s |     6–17 h | Apache-2.0                                 | Easy        | Medium/high                              |
| **8**   | **LAION-CLAP / music checkpoint** | audio embedding / audio-text latent          |             ~600MB checkpoint class |    1.5–3 GB |      2–6 s |     6–17 h | Apache-2.0 code/model card                 | Easy        | High for semantic search                 |
| **9**   | **AudioMAE**                      | latent tensor; pooled 768-ish representation |                    85.6M checkpoint |    1.5–3 GB |      2–6 s |     6–17 h | project/model-specific; inspect checkpoint | Easy        | Medium                                   |
| **10**  | **CAV-MAE audio branch**          | 768-D-ish                                    |                      Base-sized MAE |      2–4 GB |     5–12 s |    14–33 h | MIT code                                   | Medium      | Medium                                   |

*Engineering estimates, using roughly 3–4 minute songs and chunked inference where the model requires short windows.

### MuQ — your most important new audio experiment

MuQ is the strongest addition I would make to your experiment matrix.

The official repository describes MuQ as a large music foundation model using Mel-RVQ self-supervision and reports SOTA results on multiple MIR tasks. The released MuQ model is approximately 300M parameters; the hidden dimension is 1024. The official repository distinguishes model-code licensing from weight licensing: **MIT code, CC-BY-NC-4.0 weights**. ([GitHub][4])

The published MARBLE results are particularly interesting:

| Model         | GTZAN genre | GiantSteps key refined | EMO valence R² | EMO arousal R² | VocalSet |
| ------------- | ----------: | ---------------------: | -------------: | -------------: | -------: |
| MERT          |        78.6 |                   65.6 |           61.2 |           74.7 |     87.1 |
| MusicFM       |        83.8 |                   63.9 |           60.3 |           76.3 |     92.0 |
| MuQ           |    **85.5** |                   63.5 |       **60.9** |           75.4 | **95.8** |
| MuQ iterative |    **85.6** |               **65.0** |       **62.8** |           76.1 | **96.2** |

The precise benchmark setup and reported numbers are from the MuQ paper's MARBLE evaluation. ([ResearchGate][5])

That makes MuQ substantially more interesting than merely another 768-D vector.

**Recommended extraction:**

* store global `muq_1024_fp16`
* also store 10-second chunk embeddings if storage permits
* experiment with:

  * mean
  * max
  * std
  * 10th/50th/90th percentiles
  * learned attention pooling

Your 10k-song global 1024-D float16 matrix is only:

**10,000 × 1,024 × 2 ≈ 20.5 MB**

So storage is trivial.

[MuQ official repository](https://github.com/tencent-ailab/muq?utm_source=chatgpt.com)
[MuQ model card](https://huggingface.co/OpenMuQ/MuQ-MuLan-large?utm_source=chatgpt.com)

---

### MuQ-MuLan — unusually important for your website

MuQ-MuLan is not merely "another audio model."

It is a **joint music-text model**, conceptually closer to what you ultimately want for:

* "songs about heartbreak"
* "find songs like this lyric"
* lyric ↔ audio matching
* semantic search
* multilingual text/music discovery

The released MuQ-MuLan-large model is approximately **700M parameters** and its configuration specifies a **512-D joint latent**, using a 1024-D MuQ audio backbone. It explicitly supports English and Chinese text. ([Hugging Face][6])

The checkpoint is currently around **2.65 GB**, so the GTX 1660 Ti is not automatically disqualified. ([Hugging Face][7])

**But:** the model is much closer to the VRAM ceiling after audio activations. Use:

```python
model.eval()
with torch.inference_mode():
    ...
```

and process **one 10-second audio chunk at a time**.

Because the weight license is CC-BY-NC-4.0, I would use it for **research benchmarking and your prototype**, not as the unquestioned production dependency of a commercial service. ([GitHub][4])

---

### CLaMP3 — perhaps your most interesting multilingual experiment

CLaMP3 is explicitly intended for **universal MIR across unaligned modalities and unseen languages**, which maps unusually well to your dataset because your lyrics are:

* multilingual
* unaligned
* song-level
* paired to audio

Its official implementation supports audio, text and symbolic music. The recommended audio version is the **SAAS** checkpoint. Feature extraction produces **T×768** temporal features or a **768-D global vector** using average pooling. ([Hugging Face][8])

The Hugging Face weights are large: the SAAS checkpoint is approximately **2.57 GB**. ([Hugging Face][9])

CLaMP3 also explicitly uses **MERT-v1-95M features as the audio representation input**, which means it is not a completely independent acoustic family. ([GitHub][10])

That reduces its value as a pure audio ablation but **increases its value for your cross-modal website**.

**Verdict:** definitely test it, but classify it as a **cross-modal representation experiment**, not simply "audio model #8."

[CLaMP3 repository](https://github.com/sanderwood/clamp3?utm_source=chatgpt.com)

---

### MERT-v1-330M

This is the obvious sibling experiment to your existing MERT-v1-95M.

Official model information:

* **330M parameters**
* 1024 hidden dimension
* 24 kHz
* 75 Hz feature rate
* 160K hours of pretraining
* CC-BY-NC-4.0 model license

The downloadable safetensors checkpoint is about **1.26 GB**. ([Hugging Face][11])

This absolutely fits in 6 GB in fp16 inference.

The question is not feasibility; it is **incremental information**.

You should expect much more modest gains than moving from MERT to an entirely different representation family.

**Priority: 4/10 relative to MuQ.**

[MERT-v1-330M model card](https://huggingface.co/m-a-p/MERT-v1-330M?utm_source=chatgpt.com)

---

### EnCodecMAE

EnCodecMAE uses masked prediction against discrete EnCodec codec targets rather than ordinary waveform reconstruction. It has an MIT-licensed implementation/checkpoint and supports direct feature extraction. ([Hugging Face][12])

This is a particularly attractive experiment because it is:

* music-relevant
* smaller than MERT-330M
* architecturally different
* computationally manageable

**Priority: high.**

[EnCodecMAE model card](https://huggingface.co/lpepino/encodecmae-base?utm_source=chatgpt.com)

---

### BEATs

BEATs is general audio rather than music-specific, but it is architecturally interesting because it learns audio representations using acoustic-tokenizer targets. Its official implementation is MIT-licensed. ([GitHub][13])

For your dataset I would not expect BEATs to beat the best music-native encoder overall.

However, it may contribute:

* environmental/contextual sound information
* production characteristics
* general audio semantics
* robustness outside the musical ontology

**Use it as a diversity baseline.**

[BEATs repository](https://github.com/microsoft/unilm/tree/master/beats?utm_source=chatgpt.com)

---

### Dasheng-Base

Dasheng is the most attractive general-audio alternative.

The official paper/repository reports:

* 272K hours of training data
* 86M Base model
* 600M and 1.2B larger variants
* Base music score on the HEAR benchmark substantially above older AudioMAE/WavLM baselines

The released Base model is **86M parameters**, **768-dimensional**, and mean-pools to 768. ([GitHub][14])

This is practically ideal for your hardware.

**Priority: 7/10.**

[Dasheng repository](https://github.com/XiaoMi/dasheng?utm_source=chatgpt.com)
[Dasheng model card](https://huggingface.co/mispeech/dasheng-base?utm_source=chatgpt.com)

---

### LAION-CLAP / Music-CLAP family

LAION-CLAP is particularly useful for your website because it explicitly learns **audio and text representations in a shared space**. The project currently exposes music-trained checkpoints as well as general audio checkpoints. The Hugging Face HTSAT checkpoint is Apache-2.0 and around 618 MB. ([Hugging Face][15])

The Hugging Face preprocessing configuration is particularly convenient for your laptop:

* 48 kHz
* 10-second chunks
* 480,000 samples per chunk

([Hugging Face][16])

For your project, CLAP is most valuable for:

**semantic search**, not necessarily conventional MIR similarity.

You can encode text such as:

> "melancholic Turkish indie pop with dreamy female vocals"

and retrieve audio.

That is a website killer feature.

[LAION-CLAP repository](https://github.com/LAION-AI/CLAP?utm_source=chatgpt.com)
[LAION CLAP model](https://huggingface.co/laion/clap-htsat-unfused?utm_source=chatgpt.com)

---

### AudioMAE / CAV-MAE

AudioMAE is useful as a 2-D spectrogram SSL baseline. The released implementation supports music, speech and environmental sound and its common Base configuration is around 85–86M parameters. A released Hugging Face implementation exposes a 768-D latent and explicitly recommends pooling according to task. It accepts a maximum of 10 seconds per input. ([GitHub][17])

CAV-MAE is more specialized: it learns joint audio-visual representations. The official implementation reports **65.9% VGGSound accuracy** and uses Base-scale models that are practical as inference encoders on modest GPUs. ([GitHub][18])

For your **audio-only** dataset, however, CAV-MAE's audiovisual training objective is partly wasted information.

**CAV-MAE is therefore a secondary experiment, not a must-have.**

---

### OpenL3

OpenL3 is old relative to MuQ/CLaMP3, but still a very useful controlled baseline.

It offers:

* 512-D
* 6144-D
* music-specific and environmental models
* 0.1-second default hop
* music-trained models derived from AudioSet videos

([GitHub][19])

Do **not** store 6144-D OpenL3 unless you are specifically testing it.

Use 512-D.

---

### YAMNet / AudioCLIP

YAMNet is extremely cheap: the original model has ~3.2M parameters and outputs **1024-D embeddings**. ([Hugging Face][20])

That makes it effectively free compared with MERT/MuQ.

However, because you already have VGGish + PANNs, its expected information gain is low.

AudioCLIP is more interesting because it adds a CLIP-like audio/text/image alignment layer over ESResNeXt. Its repository is MIT licensed. ([GitHub][21])

Use it if you want an additional semantic/audio-text baseline, but LAION-CLAP is the more modern choice.

---

## 1.3 Models I would **not** spend your 6 GB GPU budget on

### Jukebox

Jukebox is a **generative** model, not a modern song-retrieval feature extractor.

The official repository's own measurements say:

* 1B lyrics prior: ~3.8 GB
* 5B prior: ~10.3–11.5 GB
* V100 sampling of 20 seconds: around **3 hours**

The project is also explicitly **non-commercial**. ([GitHub][22])

On your laptop this is completely misaligned with the research objective.

**Verdict: ❌ Do not run.**

### SongComposer

SongComposer is a music/lyrics **generation** model for lyric-to-melody, melody-to-lyric and song generation. The official implementation uses a language-model architecture and is not intended as a compact MIR embedding extractor. ([GitHub][23])

**Verdict: ❌ Do not add to the feature benchmark.**

### MuQ-MuLan-large training

Inference is feasible; training is not.

The paper reports training MuQ-MuLan on **32× V100 32 GB** with a batch size of 768. ([ResearchGate][5])

**Verdict: inference yes, training no.**

---

# 1.4 Better pooling of embeddings you already have

This is your **highest ROI experiment**.

For every existing temporal representation, generate:

```text
mean
std
max
min
q10
q25
q50
q75
q90
first
last
```

Then test dimensionality-controlled combinations.

### Recommended pooling hierarchy

| Pooling                   |              Cost | Recommendation                              |
| ------------------------- | ----------------: | ------------------------------------------- |
| Mean                      |           trivial | baseline                                    |
| Mean + std                |           trivial | **must try**                                |
| Mean + std + max          |           trivial | **must try**                                |
| Quantiles                 |           trivial | **must try**                                |
| First/last                |           trivial | useful only for sequence-sensitive encoders |
| Learned attention pooling | low training cost | **must try after baseline**                 |
| Beat-synchronous pooling  |          moderate | useful for rhythm/danceability              |
| Chunk-level aggregation   |          moderate | **high value**                              |
| Transformer pooling head  | training required | later                                       |
| CLS only                  | encoder-dependent | do not assume it is best                    |

The important point is that there is **no universal evidence that attention pooling dominates mean pooling for song-level MIR similarity**. Conversely, models such as CLaMP3 explicitly use average pooling for a global representation. ([GitHub][10])

Therefore:

> Don't replace mean pooling because a paper says attention is fashionable.
> **Benchmark the pooling operator itself.**

### Strong experiment

For MERT:

```text
MERT_mean              768
MERT_mean_std         1536
MERT_mean_max         1536
MERT_mean_std_q10_q90 3840
MERT_attention         768
```

Then use a **linear projection to 128 or 256 dimensions** before retrieval.

---

## 1.5 Chunk-level pooling

This is especially important because your existing MERT pipeline is effectively throwing away temporal information.

Suppose a 3.5-minute song is divided into 30-second chunks:

```text
song
 ├── chunk 1
 ├── chunk 2
 ├── ...
 └── chunk 7
```

Instead of storing only:

```text
mean(chunk_1...chunk_7)
```

store:

```text
mean
std
max
q10
q50
q90
```

over the chunk embeddings.

For 768-D MERT:

```text
7 × 768                  = 5,376 values
mean + std + quantiles   = 3–6 × 768
```

but you can compress that back to 128–256 dimensions with PCA.

This gives your website the opportunity to distinguish:

* slow intros
* high-energy choruses
* outro behavior
* multi-section songs
* tracks with extreme dynamic changes

without shipping audio.

---

# 1.6 Handcrafted DSP — do it

Despite the prestige of modern embeddings, I strongly recommend a **~70-dimensional interpretable MIR vector**.

Librosa is ISC licensed; Essentia provides a much larger MIR descriptor ecosystem but its main library is AGPLv3 and its licensing documentation distinguishes research/non-commercial and commercial use. Madmom has BSD source code but its pretrained models/data files carry additional CC-BY-NC-SA terms. ([GitHub][24])

### My recommended compact DSP vector

#### Rhythm — 12 dimensions

1. BPM
2. BPM confidence
3. beat interval mean
4. beat interval std
5. beat interval CV
6. onset strength mean
7. onset strength std
8. onset strength max
9. onset density
10. tempogram mean
11. tempogram variance
12. tempo stability

#### Timbre — 20 dimensions

13–20. MFCC 1–8 mean
21–28. MFCC 1–8 std
29. spectral centroid mean
30. centroid std
31. spectral bandwidth mean
32. bandwidth std
33. spectral rolloff mean
34. rolloff std
35. spectral contrast mean
36. spectral contrast std
37. spectral flux mean
38. spectral flux std
39. zero crossing mean
40. zero crossing std

#### Harmony — 15 dimensions

41–52. chroma class histogram
53. chroma entropy
54. chroma variance
55. tonnetz mean energy
56. tonnetz std

#### Energy / dynamics — 10 dimensions

57. RMS mean
58. RMS std
59. RMS max
60. RMS q10
61. RMS q90
62. RMS crest factor
63. low-frequency energy ratio
64. mid-frequency energy ratio
65. high-frequency energy ratio
66. dynamic-range proxy

#### Stereo / production — 8 dimensions

67. stereo width mean
68. stereo width std
69. mid/side energy ratio
70. channel correlation mean
71. channel correlation std
72. silence ratio
73. clipping ratio
74. loudness proxy

That is enough.

**Do not make a 2,000-D handcrafted Frankenstein vector.**

---

## 1.7 Highest-value handcrafted features by task

| Feature             |  Mood | Energy/danceability | Genre/similarity |
| ------------------- | ----: | ------------------: | ---------------: |
| Tempo               |   ★★★ |               ★★★★★ |             ★★★★ |
| RMS/dynamics        | ★★★★★ |               ★★★★★ |              ★★★ |
| Spectral centroid   |  ★★★★ |                ★★★★ |            ★★★★★ |
| Spectral contrast   |  ★★★★ |                 ★★★ |            ★★★★★ |
| Chroma histogram    |   ★★★ |                 ★★★ |            ★★★★★ |
| Tonnetz             |  ★★★★ |                  ★★ |            ★★★★★ |
| Onset density       |    ★★ |               ★★★★★ |             ★★★★ |
| Tempo stability     |   ★★★ |               ★★★★★ |             ★★★★ |
| Low-frequency ratio |    ★★ |               ★★★★★ |             ★★★★ |
| MFCC statistics     |  ★★★★ |                 ★★★ |            ★★★★★ |
| Dynamic-range proxy |  ★★★★ |               ★★★★★ |              ★★★ |
| Stereo width        |    ★★ |                 ★★★ |              ★★★ |

---

# 1.8 Vocal/instrumental separation

## Recommendation: yes, but as a **secondary experiment**

Demucs, Spleeter and Open-Unmix all provide practical source separation.

Spleeter supports 2-, 4- and 5-stem separation. ([GitHub][25])

Open-Unmix explicitly supports four stems:

* vocals
* drums
* bass
* other

and its open models include MIT-licensed options; however, individual pretrained models have distinct terms. ([GitHub][26])

### What I would do

Do **not** extract an entire new embedding family for every individual stem.

Instead:

```text
original audio
     │
     ├── vocals
     │      └── 256/512-D embedding
     │
     └── instrumental
            └── 256/512-D embedding
```

This tests:

> Does separating "what is sung" from "what is played" improve song similarity?

That is a much cleaner scientific question.

### Expected value

| Task                  | Expected value |
| --------------------- | -------------- |
| Lyric/mood similarity | Medium/high    |
| Instrument similarity | High           |
| Danceability          | Medium         |
| Genre                 | Medium         |
| Website "vocal style" | High           |
| Dataset size          | Excellent      |

Demucs is the highest-quality option but also the least attractive on your thermally limited 1660 Ti.

**Use Spleeter/Open-Unmix first.**

---

# 1.9 Harmony/chord extraction

Chord information is worth adding.

Chordino is a Vamp chord/chroma plugin implementing NNLS-Chroma and chord estimation; its software is GPL-2.0. ([GitHub][27])

For your project, don't store raw chord sequences as the main feature.

Instead extract:

```text
12-D chord/root histogram
major/minor ratio
chord entropy
unique-chord count
chord-change rate
dominant/Tonic frequency ratio
modal-profile correlation
```

### Result

~20 dimensions.

This will probably be weak for popularity but useful for:

* similarity
* genre
* harmony explorer
* "songs in the same harmonic world"
* explainability

---

# 1.10 PANNs: you should retain the 527 AudioSet probabilities

**Yes.**

You currently keep the 2048-D penultimate representation.

I would also store:

```text
527-D sigmoid tag vector
```

Why?

Because these two representations answer different questions:

```text
PANNs 2048
→ latent acoustic representation
```

versus

```text
PANNs 527
→ interpretable semantic predictions
```

For the website, the second is extremely valuable.

It can produce labels such as:

```text
electronic music
singing
drum
speech
female voice
guitar
crowd
...
```

instead of presenting an opaque nearest-neighbor system.

**Storage cost:** only about 5 MB in float32 for 10k songs.

---

# 1.11 Audio fingerprinting

## Yes — but keep it separate from ML features

Chromaprint/AcoustID and Panako solve an **identity** problem, not a semantic similarity problem.

Use them for:

* duplicate detection
* accidental duplicate downloads
* alternate encodings
* cover detection candidates
* exact/near-exact audio lookup

Do **not** feed fingerprints into CatBoost as normal semantic features.

Chromaprint is specifically designed for acoustic fingerprinting. Panako is another fingerprinting system intended for robust audio identification. For your website, this belongs in an `identity_index`, not `embedding_index`.

---

# 1.12 Temporal structure

This is a surprisingly valuable missing modality.

For each song estimate:

```text
number_of_sections
section_count_normalized_by_duration
repetition_score
novelty_peak_count
mean_section_duration
section_duration_std
largest_section_fraction
intro_duration
outro_duration
high_energy_section_fraction
```

You do **not** need perfect verse/chorus labels.

A novelty curve plus self-similarity matrix can give useful structural statistics.

This is particularly interesting for:

* pop
* EDM
* K-pop
* hip-hop
* soundtrack music

and your global feature mean completely hides it.

**Expected research value: high.**

---

# 1.13 Creative features worth adding

These are cheap and underrated:

### Production

* LUFS-like loudness
* true peak
* clipping ratio
* dynamic range
* silence fraction
* crest factor

### Frequency distribution

* sub-bass ratio
* bass ratio
* low-mid ratio
* high-mid ratio
* treble ratio
* spectral centroid trajectory

### Groove

* beat interval coefficient of variation
* onset-to-beat deviation
* syncopation proxy
* onset strength entropy
* tempo stability

### Stereo

* mid/side ratio
* stereo width
* left/right correlation

These descriptors are almost free relative to neural encoders.

---

# 1.14 Ranked audio shortlist

|    Rank | Idea                                       | Effort      | Value       | Feasibility     | Recommendation |
| ------: | ------------------------------------------ | ----------- | ----------- | --------------- | -------------- |
| **1 ★** | Better pooling of existing embeddings      | Tiny        | Very high   | **Easy**        | **Must try**   |
| **2 ★** | MuQ-300M                                   | Medium      | Very high   | **Easy/Medium** | **Must try**   |
| **3 ★** | Compact DSP/MIR vector                     | Low         | Very high   | **Easy**        | **Must try**   |
|       4 | MuQ-MuLan                                  | Medium/high | Very high   | Medium          | Run            |
|       5 | CLaMP3 SAAS                                | High        | Very high   | Medium          | Run            |
|       6 | PANNs 527 tag probabilities                | Tiny        | High        | Easy            | Run            |
|       7 | EnCodecMAE                                 | Medium      | High        | Easy            | Run            |
|       8 | Vocal/instrumental separation + embeddings | Medium/high | High        | Medium          | Run            |
|       9 | Temporal structure features                | Low/medium  | High        | Easy            | Run            |
|      10 | Dasheng-Base                               | Low         | Medium/high | Easy            | Good baseline  |
|      11 | LAION-CLAP                                 | Medium      | High        | Easy            | Run            |
|      12 | Chord histogram/features                   | Low         | Medium/high | Easy            | Run            |
|      13 | BEATs                                      | Medium      | Medium      | Easy            | Optional       |
|      14 | AudioMAE                                   | Medium      | Medium      | Easy            | Optional       |
|      15 | YAMNet                                     | Tiny        | Low/medium  | Very easy       | Baseline       |

---

# SECTION 2 — Lyrics Features and the Benchmark Question

# 2.1 Direct answer: is there a standard lyric-embedding benchmark?

## **No — not in the same sense that MARBLE is a standardized audio-representation benchmark.**

There is a major distinction here.

### Audio MIR

**MARBLE** is explicitly designed as a broad representation benchmark, covering multiple MIR hierarchy levels and many datasets/tasks. Its current description covers 18 tasks on 12 public datasets. ([OpenReview][28])

### Lyric similarity

For lyrics, the closest dedicated benchmark I found is:

## **LyricSIM**

It contains:

* 2,775 initially annotated Spanish song pairs
* 63 annotators
* 6-point semantic similarity scale
* 676 high-quality pairs retained for model evaluation

([arXiv][1])

The benchmark evaluates semantic similarity rather than merely matching words.

That distinction matters because song lyrics involve:

* metaphor
* emotion
* cultural context
* recurring refrains
* poetic structure
* indirect meaning

The authors explicitly argue that generic STS benchmarks do not necessarily predict lyric-specific performance. ([arXiv][1])

### LyricSIM results

The published results include:

| Model                      | Test Spearman | Test Pearson |
| -------------------------- | ------------: | -----------: |
| BERTIN                     |         85.72 |            — |
| MarIA large                |     **90.02** |            — |
| XLM-R base                 |         86.45 |            — |
| XLM-R large                |         86.74 |            — |
| Sentence Transformer XLM-R |     **88.91** |            — |
| mDeBERTa3                  |         89.15 |            — |

The paper reports the combined evaluation using Spearman/Pearson correlations. XLM-R-family models perform strongly, with MarIA-large best on the reported test configuration among those baselines. ([journal.sepln.org][29])

### Important conclusion

There is **not** a trustworthy statement such as:

> "Model X is the universally accepted SOTA lyric embedding model in 2026."

That would be false.

Instead:

> **LyricSIM is your best directly relevant public benchmark, while MARBLE is the benchmark you should use for audio representation sanity checks.**

Your own dataset should ultimately become a **multilingual lyric-similarity benchmark** through human annotation.

That could actually become publishable research.

---

# 2.2 Recommended lyric embedding benchmark set

You should test **four very different philosophies**.

|    Rank | Encoder               |          Dim | Languages    | Why                                                         |
| ------: | --------------------- | -----------: | ------------ | ----------------------------------------------------------- |
| **1 ★** | multilingual-e5-large |         1024 | 94           | Strong multilingual retrieval baseline                      |
| **2 ★** | gte-multilingual-base |          768 | 75+          | Excellent speed/size/quality tradeoff                       |
| **3 ★** | CLaMP3 text branch    |          768 | 95           | Music-aware/cross-modal                                     |
|       4 | multilingual MPNet    |          768 | 50           | Direct continuity with your prior MPNet                     |
|       5 | BGE-M3                |         1024 | multilingual | Strong retrieval-oriented alternative                       |
|       6 | LaBSE                 |          768 | multilingual | Older but useful cross-language baseline                    |
|       7 | Nomic Embed v1.5      | 768 → 64–768 | English      | Excellent compact baseline, bad primary multilingual choice |
|       8 | all-mpnet-base-v2     |          768 | English      | Your historical baseline                                    |

---

## multilingual-e5-large

The official E5 repository lists:

* multilingual-e5-small: 384
* multilingual-e5-base: 768
* multilingual-e5-large: 1024
* multilingual-e5-large-instruct: 1024

and 94-language multilingual support for the large family. ([GitHub][30])

For your data, **multilingual-e5-large is the strongest conventional NLP baseline I would run**.

It is especially useful if you want:

```text
Spanish lyric
     ↕
English lyric
     ↕
Turkish lyric
     ↕
Korean lyric
```

to remain semantically comparable.

---

## gte-multilingual-base

This is probably the best **hardware-conscious** lyric model.

Official model information:

* 305M parameters
* 768 dimensions
* up to 8192 tokens
* 75+ languages
* Apache-2.0
* long-context architecture

([Hugging Face][31])

This matters because entire song lyrics frequently exceed ordinary 512-token windows.

For your use case:

> **GTE-multilingual-base is a better first production candidate than MPNet.**

---

## Multilingual MPNet

Your previous all-MPNet model is English.

The multilingual MPNet variant supports **50 languages**, outputs 768 dimensions, uses XLM-R, and has a 128-token Sentence-Transformers maximum sequence length in its model configuration. ([Hugging Face][32])

That 128-token limit is a significant weakness for entire lyrics.

You can chunk lyrics, but then your embedding pipeline becomes:

```text
lyrics
  ↓
line/paragraph chunks
  ↓
embedding each chunk
  ↓
attention/mean pooling
```

rather than simply encoding the complete document.

Therefore:

**keep multilingual MPNet as a baseline, not your likely winner.**

---

## BGE-M3

BGE-M3 is MIT-licensed and explicitly designed for multilingual dense/sparse retrieval. ([Hugging Face][33])

I would test it because it provides a different retrieval-oriented training philosophy.

---

## Nomic Embed v1.5

Nomic Embed v1.5 is especially interesting for your website because it supports **Matryoshka embeddings**.

The same model can produce:

* 768
* 512
* 256
* 128
* 64

dimensions with graceful quality degradation. ([Hugging Face][34])

This is excellent for website scaling.

However, it is English-only. ([Hugging Face][35])

So:

> **Great compact-storage experiment; poor primary multilingual model.**

---

# 2.3 Music-aware lyric representations

There is a more interesting direction than generic NLP:

## CLaMP3

CLaMP3 explicitly addresses multilingual music information retrieval across unaligned modalities and supports text and audio retrieval. It returns a 768-D representation and provides MRR/Hit@K evaluation tooling. ([GitHub][10])

For your project this is potentially much more relevant than generic MTEB scores.

It asks:

> Can a lyric representation live in the same semantic space as music?

That is precisely the architecture you eventually want.

---

## SongComposer

Do **not** treat SongComposer as a lyric encoder.

It is a generative system for lyric/melody/song composition. ([GitHub][23])

---

# 2.4 Whole lyric vs line-level embedding

For your dataset, **do not simply truncate every lyric to 512/1024 tokens.**

Use hierarchical encoding:

```text
Song
 ├── verse 1
 ├── chorus
 ├── verse 2
 ├── chorus
 ├── bridge
 └── outro
       ↓
chunk embeddings
       ↓
attention pooling
       ↓
song embedding
```

At minimum:

```text
line groups of 4–8 lines
→ encoder
→ attention-weighted pooling
```

This preserves chorus repetition while avoiding a hard context window.

---

# 2.5 Multilingual handling

You have roughly 30–50% non-English data.

## Do not translate everything first.

Translation introduces another model and another possible semantic distortion.

Instead perform this controlled experiment:

### A

Native multilingual encoder:

```text
original language
→ multilingual-e5-large
```

### B

Translation:

```text
original language
→ English translation
→ English embedding
```

### C

Hybrid:

```text
native embedding
+
translated English embedding
```

Then compare **within-language** and **cross-language** retrieval separately.

### My expected ranking

| Approach            | Semantic quality | Cross-language | Complexity |
| ------------------- | ---------------- | -------------- | ---------- |
| multilingual native | ★★★★★            | ★★★★★          | Low        |
| translate → English | ★★★★             | ★★★★           | High       |
| native + translated | ★★★★★            | ★★★★★          | Very high  |
| character n-gram    | ★★               | ★★★            | Very low   |

Character n-grams are useful as a **lexical baseline**, not as your semantic representation.

---

# 2.6 Lyric structural features

These are excellent additions because embeddings can accidentally overvalue repeated generic phrases.

Compute:

### Repetition

* repeated-line ratio
* repeated n-gram ratio
* unique line ratio
* longest repeated sequence
* chorus repetition proxy

### Structure

* line count
* stanza count
* average lines/stanza
* stanza length variance
* line length variance
* shortest/longest line

### Lexical

* type/token ratio
* hapax ratio
* average word length
* vocabulary entropy
* profanity density
* named-entity density

### Rhyme

For languages where phonemization is reliable:

* rhyme-pair density
* end-rhyme consistency
* syllables/line
* syllable variance

Do **not** pretend these work equally well across Turkish, Hindi, Korean and Spanish. Multilingual phonology support is uneven.

---

# 2.7 Emotion/psycholinguistic features

NRC Emotion Lexicon provides eight basic emotions plus positive/negative sentiment, but its license is not an unrestricted MIT/Apache license: NRC offers research access and separate commercial licensing. ([National Research Council Canada][36])

VADER is MIT-licensed but primarily targeted at English/social-media-style sentiment. ([GitHub][37])

Therefore:

**Do not use VADER as your multilingual emotion ground truth.**

Instead keep:

```text
NRC-derived features
VADER
TextBlob
```

as separate feature families.

---

# 2.8 LLM tagging

A local quantized 7–8B model is technically feasible in 16 GB RAM, but it is not my first choice.

For each lyric, have a local LLM generate structured JSON:

```json
{
  "themes": ["breakup", "nostalgia", "loneliness"],
  "mood": ["melancholic"],
  "relationship": "romantic",
  "setting": ["night"],
  "narrative_perspective": "first_person"
}
```

### Cost

10k songs × multiple inference passes will be **orders of magnitude slower** than sentence embeddings.

### Best use

Run the LLM on:

* 1,000–2,000 songs
* benchmark whether the generated tags improve retrieval or prediction
* expand only if the gain is real

Do **not** make an LLM the foundation of the dataset.

---

# 2.9 Ranked lyric shortlist

|    Rank | Approach                                |          Dim | CPU/GPU cost | Product value | Recommendation |
| ------: | --------------------------------------- | -----------: | ------------ | ------------- | -------------- |
| **1 ★** | multilingual-e5-large                   |         1024 | Medium       | Very high     | **Must run**   |
| **2 ★** | gte-multilingual-base                   |          768 | Low/medium   | Very high     | **Must run**   |
| **3 ★** | CLaMP3 text embedding                   |          768 | Medium/high  | Very high     | **Must run**   |
|       4 | BGE-M3                                  |         1024 | Medium       | High          | Run            |
|       5 | multilingual MPNet                      |          768 | Low/medium   | High          | Run            |
|       6 | hierarchical chunk+attention embeddings | configurable | Low          | Very high     | Run            |
|       7 | lexical/psycholinguistic features       |       20–100 | Tiny         | Medium/high   | Run            |
|       8 | lyric structural/poetic features        |        20–80 | Tiny         | High          | Run            |
|       9 | LLM semantic tags                       |       20–100 | Huge         | High          | Later          |

---

# SECTION 3 — Music-Lover Website Brainstorm

## 3.1 Top website concepts

|    Rank | Tool                      | What it does                                                                         | Required modalities             | Feasibility | User value |
| ------: | ------------------------- | ------------------------------------------------------------------------------------ | ------------------------------- | ----------- | ---------- |
| **1 ★** | **Song DNA**              | One page showing audio/lyrics/harmony/rhythm profile + nearest neighbors by modality | All                             | Easy        | ★★★★★      |
| **2 ★** | **Why Is This Similar?**  | Explains similarity using tempo, energy, embeddings, genre, lyric themes             | Audio + metadata + lyrics       | Medium      | ★★★★★      |
| **3 ★** | **Semantic Music Search** | Search natural language: "dreamy late-night Turkish pop"                             | CLAP/MuQ-MuLan/CLaMP3           | Medium      | ★★★★★      |
| **4 ★** | **Personal Music Map**    | Interactive 2D/3D map where users explore nearby songs                               | audio/lyrics embeddings         | Medium      | ★★★★★      |
| **5 ★** | **Smart Radio**           | Walk through nearest-neighbor graph rather than normal genre playlists               | ANN graph                       | Easy        | ★★★★★      |
|       6 | Playlist generator        | Seed + desired energy/valence/tempo constraints                                      | embeddings + DSP                | Easy        | ★★★★       |
|       7 | Mood explorer             | Browse by mood/energy/arousal                                                        | DSP + emotion + lyrics          | Easy        | ★★★★       |
|       8 | Lyric semantic search     | Search concepts rather than exact lines                                              | lyric embeddings                | Easy        | ★★★★       |
|       9 | Song comparison           | Side-by-side "Song A vs Song B"                                                      | all modalities                  | Easy        | ★★★★       |
|      10 | Artist similarity graph   | Map artists rather than tracks                                                       | metadata + embeddings           | Easy        | ★★★★       |
|      11 | Era explorer              | Browse similarity across release year                                                | metadata + embeddings           | Easy        | ★★★★       |
|      12 | Hidden gems               | Songs with high similarity to popular songs but lower rank/popularity                | embedding + rank                | Very easy   | ★★★★       |
|      13 | Cover detector            | Search likely alternate recordings                                                   | fingerprint + acoustic features | Medium      | ★★★★       |
|      14 | Genre explorer            | Semantic exploration of genre neighborhoods                                          | PANNs + metadata + embeddings   | Easy        | ★★★        |
|      15 | Dynamic "energy curve"    | Plot energy through the song                                                         | temporal embeddings             | Medium      | ★★★        |
|      16 | Lyric mood analyzer       | Visualize thematic/emotional dimensions                                              | lyrics                          | Easy        | ★★★        |
|      17 | Song trivia               | Key, tempo, structure, unusual properties                                            | DSP                             | Easy        | ★★★        |
|      18 | Taste profile             | User uploads/chooses liked songs and gets feature-space profile                      | ANN + embeddings                | Medium      | ★★★★★      |

---

# 3.2 The five I would actually build first

## 1. Song DNA

Example:

```text
Song DNA
────────────────────────

Energy        ████████░░
Valence       ██████░░░░
Danceability  █████████░
Tempo         126 BPM
Key           F# minor

Audio neighbors
1. ...
2. ...
3. ...

Lyric neighbors
1. ...
2. ...
3. ...

Harmonic neighbors
1. ...
2. ...
3. ...

Semantic tags
electronic • female vocal • dance • synth
```

This naturally showcases your entire dataset.

---

## 2. Why Is This Similar?

This is far more differentiated than another "similar songs" button.

Example:

> Similar because both songs have:
>
> * close tempo
> * similar low-frequency energy
> * similar MERT/MuQ embedding
> * high acoustic-semantic similarity
> * similar lyric themes
> * similar harmonic profile

For ML explainability, you can use CatBoost feature importance or SHAP.

---

## 3. Semantic Music Search

Query:

> "sad atmospheric songs about missing someone, preferably not too slow"

Pipeline:

```text
query
 ↓
multilingual text encoder
 ↓
audio/text shared space
 ↓
ANN retrieval
 ↓
constraints:
   tempo < 100
   valence < threshold
 ↓
results
```

MuQ-MuLan/CLAP/CLaMP3 make this possible conceptually. ([GitHub][4])

---

## 4. Personal Music Map

Two maps:

```text
Audio Map
Lyrics Map
```

and eventually:

```text
Combined Map
```

A user can move across the space.

The important architecture is:

```text
1024D embedding
        ↓
PCA 50
        ↓
UMAP 2D
        ↓
precomputed coordinates
```

Do **not** run UMAP dynamically on every request.

---

## 5. Smart Radio

Treat your songs as a graph:

```text
Song A
 ├── Song B
 ├── Song C
 ├── Song D
 └── Song E
```

Then "radio" becomes graph traversal with controlled diversity.

Avoid:

```text
nearest → nearest → nearest → nearest
```

because it quickly collapses into one artist/genre cluster.

Instead optimize:

```text
similarity
+
diversity
+
user preferences
+
artist exclusion
```

---

# 3.3 Data compaction

Your data is small enough that you should optimize for **simplicity first**.

For 10k songs:

| Representation |    Size |
| -------------- | ------: |
| 1024-D float32 |   41 MB |
| 1024-D float16 | 20.5 MB |
| 768-D float32  | 30.7 MB |
| 768-D float16  | 15.4 MB |
| 512-D float16  | 10.2 MB |
| 128-D float16  |  2.6 MB |

For **100k songs**, multiply by 10.

### Recommendation

Store:

```text
raw master embedding → float16
production embedding → PCA 256 float16
web embedding → PCA 64/128 float16
```

Do not quantize until you've measured whether it damages retrieval.

---

# 3.4 PCA vs UMAP

### PCA

Use for:

* dimensionality reduction before ANN
* compression
* deterministic preprocessing
* inference/storage

### UMAP

Use for:

* visualization
* exploration

**Do not use UMAP coordinates as your retrieval representation.**

UMAP is a visualization/manifold tool, not the semantic retrieval space.

---

# 3.5 FAISS / HNSW / USearch

For **10k songs**, exact cosine similarity is already fast.

You don't actually need ANN.

At 10k × 768:

```text
10,000 × 768
```

is trivial.

Nevertheless, build the ANN layer now for future scaling.

### Recommendation

**FAISS** is my default.

It is MIT-licensed, actively maintained, and its current repository shows v1.14.3 released June 13, 2026. ([GitHub][38])

For 100k–1M tracks:

```text
FAISS HNSW
```

or

```text
FAISS IVF-PQ
```

becomes more interesting.

---

# 3.6 Recommended architecture

```text
                    ┌─────────────────────┐
                    │     React/Next      │
                    └──────────┬──────────┘
                               │
                         REST / JSON
                               │
                    ┌──────────▼──────────┐
                    │       FastAPI       │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼─────────────────┐
            │                  │                 │
       ┌────▼────┐        ┌────▼─────┐      ┌───▼────┐
       │ Postgres│        │  FAISS   │      │ Static │
       │ metadata│        │ ANN index│      │ assets │
       └─────────┘        └───────────┘      └────────┘
```

### Backend

**FastAPI**

You already use Python, embeddings and ML.

Do not switch to Flask unless you have a specific reason.

### Frontend

**React/Next.js**

Your existing web-development background makes this the lowest-friction path.

### Visualization

Use:

* WebGL
* regl/scatter-gl
* Three.js only if you genuinely need 3D

A 2D map is likely more useful than a gimmicky 3D one.

---

# 3.7 Legal/licensing risk — this is the most important non-technical section

## Spotify

Spotify's current Developer Terms state that you must not use Spotify Content to train an ML/AI model or ingest Spotify Content into an ML/AI model. ([Spotify for Developers][2])

Spotify's current Developer Policy also says:

* metadata must be attributed
* metadata/cover art/audio previews require links back to Spotify
* metadata must not be offered as a standalone service/product

([Spotify for Developers][39])

Spotify additionally explicitly warns that it can suspend/disable applications for policy violations. ([Spotify for Developers][40])

### Therefore

Your current scientific pipeline may be perfectly reasonable as **private research**, but the transition to:

> "I scraped Spotify metadata/audio and publish all derived features on Kaggle"

is **not automatically safe**.

The fact that a feature is mathematically derived does not automatically erase contractual restrictions attached to the source data.

---

## Lyrics

Lyrics are normally copyrighted textual works.

You should assume:

```text
raw lyrics → ❌ public Kaggle dataset
```

unless you have the relevant rights/license.

Kaggle's Terms explicitly say you must not publish/distribute/exploit content you do not own or have permission to use. ([Kaggle][41])

### Safer public dataset

Publish:

```text
song_id
artist_hash / permitted ID
release_year
your own numerical features
audio embeddings
lyric embedding
lyric statistics
genre labels
language
```

but **not the lyrics themselves**.

Then provide a script that allows a user with legally sourced lyrics to reproduce the lyric feature extraction.

---

## MusicBrainz

For future open metadata, MusicBrainz is much more attractive.

Its core database is CC0; supplementary data has CC BY-NC-SA restrictions. ([MusicBrainz][42])

Therefore:

> Use MusicBrainz identifiers wherever possible in your open dataset.

But don't assume every field in MusicBrainz is CC0; distinguish core vs supplementary data.

---

# SECTION 4 — Hardware Feasibility Matrix

**Assumption:** GTX 1660 Ti 6 GB, fp16 where supported, batch=1, inference mode, 3–4 minute typical song, chunked models processed sequentially.

| Approach                    |               Dims |              Params / size |        VRAM est. |          RAM est. |     Time/song |     10k total | License                   | Verdict              |
| --------------------------- | -----------------: | -------------------------: | ---------------: | ----------------: | ------------: | ------------: | ------------------------- | -------------------- |
| Existing VGGish             |                128 |                   ~few 10M |            <1 GB |            1–2 GB |          <2 s |          <6 h | Apache code               | ✅ already have       |
| Existing MERT-95M           |                768 |                        95M |      ~1.5–2.5 GB |            2–3 GB |      existing |             — | CC-BY-NC                  | ✅                    |
| Existing PANNs Cnn14        |               2048 |                       ~80M |            ~2 GB |            2–3 GB |      existing |             — | research/project terms    | ✅                    |
| Existing Mel stats          |                512 |                        DSP |              CPU |             <1 GB |          <1 s |          <3 h | librosa ISC               | ✅                    |
| **MERT-330M**               |               1024 |                       330M |           2–4 GB |            2–4 GB |        5–15 s |       14–42 h | CC-BY-NC                  | ✅                    |
| **MuQ**                     |               1024 |                      ~300M |       2.5–4.5 GB |            3–5 GB |        5–15 s |       14–42 h | CC-BY-NC                  | ✅ **must test**      |
| **MuQ-MuLan**               |          512 joint |                      ~700M |         4–5.8 GB |            4–7 GB |        8–20 s |       22–56 h | CC-BY-NC                  | ⚠️ careful chunking  |
| **CLaMP3**                  |                768 |         ~2.5 GB checkpoint |         4–5.8 GB |            5–8 GB |        8–20 s |       22–56 h | MIT code; inspect weights | ⚠️                   |
| EnCodecMAE                  |    temporal/global |                 ~85M-class |         1.5–3 GB |            2–3 GB |        3–10 s |        8–28 h | MIT                       | ✅                    |
| BEATs Base                  |               ~768 |                       ~90M |         1.5–3 GB |            2–3 GB |         2–6 s |        6–17 h | MIT                       | ✅                    |
| Dasheng-Base                |                768 |                        86M |           1–2 GB |            2–3 GB |         2–6 s |        6–17 h | Apache-2.0                | ✅                    |
| Dasheng-0.6B                |               1280 |                       600M |           3–5 GB |            4–6 GB |        6–15 s |       17–42 h | Apache project            | ⚠️                   |
| Dasheng-1.2B                |               1536 |                       1.2B |          5–6+ GB |            5–8 GB |       10–30 s |       28–83 h | Apache project            | ❌/⚠️ CPU workaround  |
| LAION-CLAP                  |               ~512 |               ~600 MB ckpt |         1.5–3 GB |            2–4 GB |         2–6 s |        6–17 h | Apache checkpoint         | ✅                    |
| AudioMAE                    |         768 latent |                        86M |         1.5–3 GB |            2–4 GB |         2–6 s |        6–17 h | project-specific          | ✅                    |
| CAV-MAE                     |                768 |                 Base-sized |           2–4 GB |            3–5 GB |        5–12 s |       14–33 h | MIT                       | ✅                    |
| OpenL3                      |                512 |                      small |     1–2 GB / CPU |              2 GB |        2–10 s |        6–28 h | project license           | ✅                    |
| OpenL3                      |               6144 |                      small |           1–2 GB |            2–3 GB |        2–10 s |        6–28 h | project license           | ⚠️ storage-heavy     |
| YAMNet                      |               1024 |                       3.2M |            <1 GB |             <1 GB |          <2 s |          <6 h | Apache-2.0                | ✅                    |
| AudioCLIP                   |       model latent |           ESResNeXt + CLIP |           2–3 GB |            3–4 GB |         2–5 s |        6–14 h | MIT code                  | ✅                    |
| Jukebox 1B                  |  latent/generative |                         1B |          3.8 GB+ |             >8 GB |     hours/min |      enormous | Noncommercial             | ❌                    |
| Jukebox 5B                  |  latent/generative |                         5B |     10.3–11.5 GB |            >16 GB |         hours |      enormous | Noncommercial             | ❌                    |
| SongComposer                |         generative |                   large LM |  >6 GB practical |            >16 GB |          high |   impractical | inspect model-specific    | ❌                    |
| Spleeter 2/4 stem           |         stem audio |                      small |          ~1–2 GB |            2–4 GB |        3–10 s |        8–28 h | repo/model dependent      | ✅                    |
| Open-Unmix                  |         stem audio |                   moderate |          ~1–3 GB |            3–5 GB |        5–15 s |       14–42 h | MIT/open model variants   | ✅                    |
| Demucs                      |         stem audio |                     larger |          ~3–6 GB |            4–8 GB |       20–60 s |      56–167 h | model-specific            | ⚠️                   |
| Chordino                    |           sequence |                        DSP |              CPU |             <1 GB |         1–5 s |        3–14 h | GPL-2.0                   | ✅ research           |
| Librosa DSP                 |             50–100 |                        DSP |              CPU |             <1 GB |         1–5 s |        3–14 h | ISC                       | ✅ **must test**      |
| Madmom beats                |           sequence |                     DSP/ML |              CPU |            1–2 GB |         1–8 s |        3–22 h | BSD code; model NC-SA     | ⚠️                   |
| Essentia                    |             50–200 |                     DSP/ML |              CPU |            1–2 GB |         1–6 s |        3–17 h | AGPLv3 main               | ⚠️                   |
| PANNs 527 tags              |                527 |             existing model | negligible extra |        negligible |  <0.5 s extra |        <1.5 h | same model terms          | ✅ **do**             |
| Better pooling              |      768/1024 etc. |                   no model |              CPU |             <1 GB |  milliseconds |       minutes | your code                 | ✅ **highest ROI**    |
| Attention pooling           |           same dim |                  tiny head |       negligible |             <1 GB |  milliseconds |       minutes | your code                 | ✅                    |
| Beat-synchronous pooling    |       configurable |                        DSP |              CPU |             <1 GB |         1–4 s |        3–11 h | own/librosa               | ✅                    |
| Chromaprint                 |        fingerprint |               compact hash |              CPU |             <1 GB |          <1 s |          <3 h | LGPL/project              | ✅ identity           |
| Panako                      |        fingerprint |                    compact |              CPU |             <1 GB |          <2 s |          <6 h | GPL/project               | ✅ identity           |
| Lyric multilingual-e5-large |               1024 |            ~560M-ish class |           2–4 GB |            3–5 GB |     CPU 1–5 s |        3–14 h | MIT                       | ✅                    |
| multilingual-e5-base        |                768 |                      ~278M |         1.5–3 GB |            2–4 GB |         1–4 s |        3–11 h | MIT                       | ✅                    |
| gte-multilingual-base       |                768 |                       305M |         1.5–3 GB |            2–4 GB |         1–4 s |        3–11 h | Apache-2.0                | ✅ **must test**      |
| BGE-M3                      |               1024 |             ~large encoder |           2–4 GB |            3–5 GB |         2–6 s |        6–17 h | MIT                       | ✅                    |
| multilingual MPNet          |                768 |                      ~300M |           2–3 GB |            3–4 GB |         1–4 s |        3–11 h | Apache-2.0                | ✅                    |
| all-MPNet                   |                768 |                      ~110M |            <2 GB |            2–3 GB |          <3 s |          <8 h | Apache-2.0                | ✅ baseline           |
| Nomic Embed v1.5            |             768→64 |                       137M |            <2 GB |            2–3 GB |          <3 s |          <8 h | Apache-2.0                | ✅ English            |
| LaBSE                       |                768 | large multilingual encoder |           2–3 GB |            3–4 GB |         2–6 s |        6–17 h | Apache-style model terms  | ✅                    |
| NRC emotion                 |                10+ |                    lexicon |              CPU |        negligible |        <0.1 s |       minutes | research/commercial terms | ⚠️ publish carefully |
| VADER                       |              4-ish |                    lexicon |              CPU |        negligible |        <0.1 s |       minutes | MIT                       | ✅                    |
| TextBlob                    |                  2 |                        NLP |              CPU |        negligible |        <0.1 s |       minutes | permissive                | ✅ baseline           |
| BERTopic                    | topic distribution |                    depends |           1–4 GB |            2–6 GB |  seconds/song |         hours | MIT                       | ✅ later              |
| KeyBERT                     |       keyword list |                sparse/text |          CPU/GPU |            1–3 GB |       0.5–3 s |         1–8 h | MIT                       | ✅                    |
| YAKE                        |           keywords |                     sparse |              CPU |             <1 GB |        <0.1 s |        <0.3 h | MIT                       | ✅                    |
| UMAP                        |              2/3-D |                postprocess |              CPU |            1–4 GB | dataset-level | minutes-hours | BSD                       | ✅                    |
| PCA                         |             64–512 |                postprocess |              CPU |           <1–2 GB | dataset-level |       minutes | BSD                       | ✅                    |
| FAISS                       |                ANN |                      index |              CPU | dataset-dependent |      ms/query |    negligible | MIT                       | ✅ **recommend**      |

The project/model facts underlying this matrix are documented in the official model repositories and cards cited above, including MERT, MuQ, MuQ-MuLan, CLaMP3, Dasheng, EnCodecMAE, AudioMAE, LAION-CLAP, YAMNet, VGGish, Librosa, Essentia, Madmom, Open-Unmix and FAISS. ([Hugging Face][11])

---

# SECTION 5 — Feature-Set Comparison Methodology

Your previous benchmark tells us something very important:

> A huge feature vector does **not** imply a better feature vector.

Your previous 4,254-D CatBoost result:

```text
valence       R² ≈ 0.72
energy        R² ≈ 0.92
danceability  R² ≈ 0.79
popularity    R² ≈ 0.13
```

already demonstrates diminishing returns.

Your next experiment should therefore be **ablation-first**, not feature-hoarding.

---

# 5.1 Prediction benchmark

Targets:

```text
valence
energy
danceability
popularity
```

### Required split

Use:

## **Artist-aware split**

Do not randomly split individual tracks.

Otherwise:

```text
Artist A song 1 → train
Artist A song 2 → test
```

creates leakage through:

* production
* vocalist
* genre
* label
* era
* artist-specific sound
* duplicated lyrics/themes

Use:

```text
GroupKFold(groups=artist_id)
```

or grouped train/validation/test.

---

# 5.2 Three model levels

### Model A — linear

Ridge / Elastic Net

Purpose:

> Does the feature itself contain linearly extractable signal?

### Model B — fixed MLP

One architecture across every experiment.

For example:

```text
input
 ↓
512
 ↓
256
 ↓
128
 ↓
target
```

### Model C — CatBoost

Keep your benchmark configuration fixed.

Do not retune CatBoost independently for every feature family.

That would turn:

> "feature quality"

into:

> "feature quality + tuning budget."

---

# 5.3 Modality ablation

Create:

```text
A = Spotify numeric/audio gold features
B = DSP
C = MERT
D = MuQ
E = PANNs
F = audio-text
G = lyric embeddings
H = lyric statistics
I = structure
J = artist metadata
```

Then test:

```text
A
A+B
A+C
A+D
A+E
A+F
A+G
A+H
A+I

A+B+C
A+B+D
A+G+H
A+C+D+F
A+B+C+D+G+H+I
```

---

# 5.4 Popularity must be evaluated separately

Your previous R²≈0.13 is not surprising.

Spotify popularity is not an intrinsic acoustic property.

Spotify's own current API documentation says track popularity is an algorithmic score between 0 and 100 based primarily on play counts and recency. ([Spotify for Developers][43])

Therefore popularity contains external variables such as:

* marketing
* playlist placement
* artist fame
* release timing
* social trends
* virality
* geography

Your model should therefore report:

```text
R²
MAE
RMSE
Spearman correlation
```

and compare:

### Acoustic-only

vs

### Metadata-only

vs

### Audio + lyrics + metadata

The latter is more scientifically interesting.

---

# 5.5 Retrieval benchmark

You need three levels.

## Level 1 — automatic proxy

Measure whether nearest neighbors agree on:

* main genre
* tempo range
* energy bin
* valence bin
* release era
* artist-independent style

Metrics:

```text
Precision@K
Recall@K
NDCG@K
MRR
cluster purity
```

## Level 2 — known covers

SHS100K and similar cover-song datasets are useful because songs by different artists can have strong musical identity overlap. The MIR-Datasets catalogue identifies SHS100K as around 100,000 tracks/10,000 songs. ([GitHub][44])

This is especially useful for acoustic retrieval.

## Level 3 — human evaluation

Take:

```text
100 query songs
×
5 retrieved neighbors
```

Have humans answer:

```text
How musically similar are these?
1–5
```

Do this separately for:

```text
audio
lyrics
combined
```

This becomes your strongest actual product metric.

---

# 5.6 Lyric evaluation

Use **LyricSIM** as the external benchmark and create a small internal benchmark.

Your 10k dataset can become:

```text
100 songs
→ construct 200–500 candidate pairs
→ human similarity annotation
```

Annotation criteria:

```text
theme
emotion
message
narrative situation
literal meaning
cultural context
```

These are exactly the kinds of dimensions LyricSIM uses. ([arXiv][1])

---

# 5.7 Modality agreement

This is one of the most interesting experiments you can do.

For every song pair calculate:

```text
audio_similarity
lyric_similarity
metadata_similarity
harmony_similarity
rhythm_similarity
```

Then classify:

### Type A

```text
high audio
high lyrics
```

→ truly similar songs

### Type B

```text
high audio
low lyrics
```

→ musically similar, lyrically unrelated

### Type C

```text
low audio
high lyrics
```

→ semantically similar songs with different sound

### Type D

```text
high metadata
low audio
```

→ genre labels causing false similarity

This naturally becomes a website feature.

---

# 5.8 Cost-aware feature ranking

Define:

[
ROI_i = \frac{\Delta Performance_i}{ExtractionHours_i + StoragePenalty_i + EngineeringPenalty_i}
]

A more practical version:

```text
ROI =
normalized validation improvement
/
(log10(extraction_hours + 1) + complexity penalty)
```

This prevents:

> 0.003 R² gain after 48 hours of GPU compute

from beating:

> 0.01 gain after 20 minutes.

---

# 5.9 Main methodological pitfalls

## 1. Artist leakage

Most dangerous.

## 2. Duplicate/remix leakage

Use ISRC/fingerprint/metadata to detect duplicates.

## 3. Genre confounding

If a model simply learns:

```text
EDM → high energy
```

you have not learned general musical representation.

Report performance within major genre strata.

## 4. Audio-length bias

Some models see:

```text
10 seconds
```

others:

```text
30 seconds
```

and your DSP sees the entire track.

Normalize this where possible.

## 5. Label leakage

You explicitly treat Spotify audio features as gold labels.

Correct.

Do not feed them into another feature extractor pipeline and call the result independent.

## 6. Popularity leakage

Artist followers and artist popularity are legitimate predictive variables, but you must explicitly distinguish:

```text
intrinsic-song prediction
```

from

```text ecosystem prediction
```

---

# SECTION 6 — Final Action Plan

# Phase A — this week: cheap experiments

## A1 — pooling existing embeddings

Run:

```text
mean
mean+std
mean+std+max
q10/q50/q90
attention
```

on:

```text
MERT
PANNs
VGGish
```

This should be the first experiment.

### Expected time

**Minutes to a few hours**, not days.

---

## A2 — 70-D DSP vector

Implement one reproducible extractor:

```text
librosa
scipy
numpy
```

Keep feature order in:

```text
features/dsp_v1.json
```

Example:

```json
{
  "version": "dsp_v1",
  "features": [
    "tempo",
    "tempo_confidence",
    "onset_strength_mean",
    ...
  ]
}
```

The versioning is important for Kaggle.

---

## A3 — PANNs semantic tags

Extract the 527 predictions.

Store:

```text
panns_tag_000
...
panns_tag_526
```

and retain the class dictionary separately.

---

## A4 — lyric baselines

Run:

```text
multilingual-e5-base
gte-multilingual-base
multilingual-mpnet
```

Do not start with the giant models.

For lyrics longer than the model context:

```text
chunk
→ encode
→ attention-pool
```

---

## A5 — structural lyric features

Extract:

```text
line_count
stanza_count
repetition_ratio
unique_ratio
line_length_mean
line_length_std
word_length_mean
vocab_entropy
chorus_proxy
rhyme_proxy
```

---

# Phase B — GPU experiments

Run in this order:

## B1 — MuQ

**First heavyweight model.**

Reason:

* current music-native representation
* ~300M
* 1024-D
* strong MARBLE results
* fits 6 GB with chunking

([GitHub][4])

---

## B2 — MuQ-MuLan

Second.

Because it tests the most interesting new capability:

> audio ↔ language alignment.

([Hugging Face][45])

---

## B3 — CLaMP3

Third.

Because it targets your unusual combination:

```text
multilingual lyrics
+
music audio
+
unaligned modalities
```

([GitHub][10])

---

## Optional B4 — EnCodecMAE

Use this as the main architecture-diversity baseline.

---

## Optional B5 — Dasheng

Use if you want a cheap, strong general-audio encoder.

---

# Phase C — Evaluation + publication

## C1 — Build a master experiment table

Every row:

```text
feature_set
feature_dim
model
pooling
split
target
R2
MAE
RMSE
Spearman
runtime
RAM
VRAM
license
```

---

## C2 — Choose winners separately

Do not choose one winner globally.

Choose:

```text
best audio similarity
best lyric similarity
best mood prediction
best energy prediction
best danceability prediction
best semantic search
best compact representation
best explainability
```

You may end up with:

```text
Audio similarity     = MuQ
Lyrics similarity    = E5
Cross-modal search   = MuQ-MuLan
Explainability       = DSP
Semantic tags        = PANNs 527
```

That is a **better scientific result** than claiming one model is best at everything.

---

# Exact dependency set

Start with:

```bash
pip install \
  torch \
  torchaudio \
  torchvision \
  transformers \
  sentence-transformers \
  librosa \
  scipy \
  numpy \
  pandas \
  scikit-learn \
  catboost \
  lightgbm \
  xgboost \
  umap-learn \
  faiss-cpu \
  hnswlib \
  usearch \
  vaderSentiment \
  textblob \
  bertopic \
  keybert \
  yake \
  pyacoustid
```

For the model families that provide dedicated packages:

```bash
pip install muq
pip install dasheng
pip install laion-clap
```

Use `faiss-cpu` for your laptop backend initially; you have no reason to sacrifice VRAM for ANN indexing at 10k vectors. FAISS is MIT licensed and the repository is actively maintained. ([GitHub][38])

For Chordino, Madmom and some Essentia functionality, I would prefer their native/repository installation procedures rather than blindly pinning arbitrary PyPI versions because the model/data licenses differ from the source-code licenses. ([GitHub][46])

---

# Recommended data layout for Kaggle

Do **not** make one gigantic CSV.

Use:

```text
dataset/
├── metadata.parquet
├── audio_features.parquet
├── lyric_features.parquet
├── dsp_features.parquet
├── panns_tags.parquet
├── artists.parquet
├── splits/
│   ├── train.txt
│   ├── validation.txt
│   ├── test.txt
│   └── artist_groups.json
│
├── embeddings/
│   ├── mert95_fp16.npy
│   ├── mert330_fp16.npy
│   ├── muq1024_fp16.npy
│   ├── mulan512_fp16.npy
│   ├── clamp3_768_fp16.npy
│   └── lyric_e5_1024_fp16.npy
│
└── schema/
    ├── audio_features.json
    ├── lyric_features.json
    ├── dsp_features.json
    ├── panns_tags.json
    └── LICENSES.md
```

Parquet for tabular metadata.

NPY/NPZ or memory-mapped arrays for dense embeddings.

---

# Top five risks

| Risk                                      | Severity     | Mitigation                                                                                  |
| ----------------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| **1. OOM on 6 GB GPU**                    | Critical     | batch=1; 10s/30s chunking; `inference_mode`; unload models between passes                   |
| **2. Licensing/Kaggle rights**            | **Critical** | do not publish raw lyrics or questionable Spotify-derived content until rights are resolved |
| **3. Multilingual embedding degradation** | High         | native-vs-translation benchmark; language-stratified metrics                                |
| **4. Artist/genre leakage**               | High         | artist-grouped splits + genre-stratified testing                                            |
| **5. Too many weak features**             | Medium/high  | ablation + cost-aware ROI; no feature hoarding                                              |

---

# Recommended timeline on your hardware

Assuming overnight batches and no cloud GPU:

| Phase       | Work                 | Expected wall time |
| ----------- | -------------------- | -----------------: |
| A1          | existing pooling     |             <1 day |
| A2          | DSP                  |          0.5–1 day |
| A3          | PANNs tags           |             <1 day |
| A4          | 3 lyric encoders     |           1–3 days |
| B1          | MuQ                  |          ~1–2 days |
| B2          | MuQ-MuLan            |          ~2–3 days |
| B3          | CLaMP3               |          ~2–3 days |
| B4 optional | EnCodecMAE           |          ~1–2 days |
| C           | benchmark + ablation |           2–4 days |
| Packaging   | Kaggle/web dataset   |           1–3 days |

Because your machine is thermal-limited, **run one GPU process at a time**. Sustained maximum clocks can be worse than running at a lower power target overnight.

---

# 2026 Recommended Feature Architecture

I would eventually publish/store four major representation layers:

```text
                    SONG
                     │
        ┌────────────┼────────────┐
        │            │            │
       AUDIO       LYRICS      METADATA
        │            │            │
   ┌────┼─────┐   ┌──┼────┐      │
   │    │     │   │  │    │      │
  DSP  MuQ   PANN E5 GTE CLaMP  genre/rank
   │    │     │   │  │    │
   └────┼─────┘   └──┼────┘
        │            │
        └─────┬──────┘
              │
        CROSS-MODAL SPACE
              │
      ┌───────┼────────┐
      │       │        │
   similarity search  map
      │       │        │
    radio   semantic  discovery
```

This gives you **three scientifically meaningful representations**:

### Intrinsic acoustic

```text
DSP + MuQ + MERT + PANNs
```

### Linguistic

```text
E5 + GTE + lyric structure
```

### Cross-modal

```text
MuQ-MuLan / CLaMP3 / CLAP
```

That is a far more compelling research story than:

> "I concatenated 12 embeddings and trained CatBoost."

---

# 1-PAGE EXECUTIVE SUMMARY

| Category             | Best recommendation                   | Why                                                                                                                       | Hardware        | License concern              |
| -------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------- |
| **Audio #1**         | **MuQ ~300M / 1024-D**                | Strong current music SSL results; different generation from your MERT baseline                                            | ✅ 2.5–4.5 GB    | **CC-BY-NC**                 |
| **Audio #2**         | **Compact MIR/DSP ~70-D**             | Very cheap, interpretable, complements neural vectors                                                                     | ✅ CPU           | Low with librosa             |
| **Audio #3**         | **MuQ-MuLan 512-D**                   | Audio-text semantic space; excellent website potential                                                                    | ⚠️ ~4–5.8 GB    | **CC-BY-NC**                 |
| **Audio #4**         | **CLaMP3 768-D**                      | Multilingual, cross-modal, unaligned MIR                                                                                  | ⚠️ 4–5.8 GB     | Check weights carefully      |
| **Audio #5**         | **EnCodecMAE**                        | Different SSL paradigm; compact enough for laptop                                                                         | ✅               | MIT                          |
| **Lyrics #1**        | **multilingual-e5-large 1024-D**      | Strong general multilingual retrieval baseline                                                                            | ✅ with batching | MIT                          |
| **Lyrics #2**        | **gte-multilingual-base 768-D**       | Best quality/size/context compromise                                                                                      | ✅               | Apache-2.0                   |
| **Lyrics #3**        | **CLaMP3 text 768-D**                 | Music-aware/cross-modal representation                                                                                    | ⚠️              | Check checkpoint             |
| **Benchmark answer** | **No universal lyric-SOTA benchmark** | **LyricSIM** is the clearest dedicated lyric-similarity benchmark; **MARBLE** is the major audio representation benchmark | —               | —                            |
| **Website #1**       | **Song DNA**                          | Exposes every modality in one understandable interface                                                                    | ✅               | Depends on data              |
| **Website #2**       | **Why Is This Similar?**              | Makes the ML system explainable                                                                                           | ✅               | Depends on data              |
| **Website #3**       | **Semantic Music Search**             | "Find dreamy melancholic Turkish pop"                                                                                     | ✅               | Shared-space model licensing |
| **Website #4**       | **Interactive Music Map**             | Most visually distinctive feature                                                                                         | ✅               | Mostly your derived vectors  |
| **Website #5**       | **Smart Radio / Graph Walk**          | Converts similarity graph into an actual product                                                                          | ✅               | Depends on catalog rights    |

## The most important research recommendation

Do **not** immediately add another 10 audio encoders.

Run this sequence:

```text
1. Better pooling of your existing MERT/PANNs/VGGish
2. 70-D handcrafted MIR vector
3. PANNs 527 tag outputs
4. multilingual-e5-base / large
5. gte-multilingual-base
6. MuQ
7. MuQ-MuLan
8. CLaMP3
9. human similarity benchmark
10. feature ablation
```

Then make the website from the **winning modalities**, not from a predetermined giant concatenation.

## The single most promising scientific direction

Your dataset has an unusual combination:

```text
10k real popular songs
+
full audio
+
multilingual lyrics
+
Spotify-style metadata
+
artist information
+
multiple modern audio embeddings
```

That gives you the opportunity to study a particularly useful question:

> **When are two songs similar acoustically, lyrically, or cross-modally — and when do those notions disagree?**

That is substantially more interesting than another popularity-prediction experiment.

And it maps directly onto the product:

```text
"Similar sonically"
"Similar lyrically"
"Similar emotionally"
"Similar harmonically"
"Similar overall"
"Similar, but for completely different reasons"
```

One final caution: **Spotify's current terms explicitly prohibit using Spotify Content to train ML/AI models, while their Developer Policy also restricts standalone metadata products; Kaggle requires you to have the necessary rights for public submissions.** Treat your current Spotify-derived research artifact as private until the provenance/licensing question is resolved. ([Spotify for Developers][2])

### Primary sources worth keeping with the project

[MARBLE benchmark — NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com)
[MuQ official repository](https://github.com/tencent-ailab/muq?utm_source=chatgpt.com)
[MuQ-MuLan model card](https://huggingface.co/OpenMuQ/MuQ-MuLan-large?utm_source=chatgpt.com)
[CLaMP3 official repository](https://github.com/sanderwood/clamp3?utm_source=chatgpt.com)
[MERT-v1-330M model card](https://huggingface.co/m-a-p/MERT-v1-330M?utm_source=chatgpt.com)
[EnCodecMAE model card](https://huggingface.co/lpepino/encodecmae-base?utm_source=chatgpt.com)
[Dasheng official repository](https://github.com/XiaoMi/dasheng?utm_source=chatgpt.com)
[LAION-CLAP official repository](https://github.com/LAION-AI/CLAP?utm_source=chatgpt.com)
[LyricSIM benchmark](https://arxiv.org/abs/2306.01325?utm_source=chatgpt.com)
[Multilingual E5 repository](https://github.com/microsoft/unilm/tree/master/e5?utm_source=chatgpt.com)
[GTE multilingual model](https://huggingface.co/Alibaba-NLP/gte-multilingual-base?utm_source=chatgpt.com)
[Spotify Developer Terms](https://developer.spotify.com/terms?utm_source=chatgpt.com)
[Spotify Developer Policy](https://developer.spotify.com/policy?utm_source=chatgpt.com)
[Kaggle Terms of Use](https://www.kaggle.com/terms?utm_source=chatgpt.com)
[MusicBrainz data licensing](https://musicbrainz.org/doc/About/Data_License?utm_source=chatgpt.com)

[1]: https://arxiv.org/abs/2306.01325 "LyricSIM: A novel Dataset and Benchmark for Similarity Detection in Spanish Song LyricS"
[2]: https://developer.spotify.com/terms?source=post_page---------------------------&utm_source=chatgpt.com "Spotify Developer Terms | Spotify for Developers"
[3]: https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md?utm_source=chatgpt.com "models/research/audioset/vggish/README.md at master · tensorflow/models · GitHub"
[4]: https://github.com/tencent-ailab/muq?utm_source=chatgpt.com "GitHub - tencent-ailab/MuQ: Official repository of the paper \"MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization\". · GitHub"
[5]: https://www.researchgate.net/publication/387671171_MuQ_Self-Supervised_Music_Representation_Learning_with_Mel_Residual_Vector_Quantization?utm_source=chatgpt.com "(PDF) MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization"
[6]: https://huggingface.co/OpenMuQ/MuQ-MuLan-large/blob/main/README.md?utm_source=chatgpt.com "README.md · OpenMuQ/MuQ-MuLan-large at main"
[7]: https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main?utm_source=chatgpt.com "OpenMuQ/MuQ-MuLan-large at main"
[8]: https://huggingface.co/sander-wood/clamp3/blame/main/README.md?utm_source=chatgpt.com "README.md · sander-wood/clamp3 at main"
[9]: https://huggingface.co/sander-wood/clamp3/tree/main?utm_source=chatgpt.com "sander-wood/clamp3 at main"
[10]: https://github.com/sanderwood/clamp3?utm_source=chatgpt.com "GitHub - sanderwood/clamp3: CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages [ACL 2025] · GitHub"
[11]: https://huggingface.co/m-a-p/MERT-v1-330M?utm_source=chatgpt.com "m-a-p/MERT-v1-330M · Hugging Face"
[12]: https://huggingface.co/lpepino/encodecmae-base?utm_source=chatgpt.com "lpepino/encodecmae-base · Hugging Face"
[13]: https://github.com/microsoft/unilm/blob/master/beats/README.md?utm_source=chatgpt.com "unilm/beats/README.md at master · microsoft/unilm · GitHub"
[14]: https://github.com/XiaoMi/dasheng?utm_source=chatgpt.com "GitHub - XiaoMi/dasheng: Official PyTorch code for Deep Audio-Signal Holistic Embeddings · GitHub"
[15]: https://huggingface.co/laion/clap-htsat-unfused/tree/main?utm_source=chatgpt.com "laion/clap-htsat-unfused at main"
[16]: https://huggingface.co/laion/clap-htsat-unfused/blame/main/preprocessor_config.json?utm_source=chatgpt.com "preprocessor_config.json · laion/clap-htsat-unfused at main"
[17]: https://github.com/facebookresearch/AudioMAE?utm_source=chatgpt.com "GitHub - facebookresearch/AudioMAE: This repo hosts the code and models of \"Masked Autoencoders that Listen\". · GitHub"
[18]: https://github.com/yuangongnd/cav-mae?utm_source=chatgpt.com "GitHub - YuanGongND/cav-mae: Code and Pretrained Models for ICLR 2023 Paper \"Contrastive Audio-Visual Masked Autoencoder\". · GitHub"
[19]: https://github.com/marl/openl3/blob/main/docs/tutorial.rst?utm_source=chatgpt.com "openl3/docs/tutorial.rst at main · marl/openl3 · GitHub"
[20]: https://huggingface.co/STMicroelectronics/yamnet?utm_source=chatgpt.com "STMicroelectronics/yamnet · Hugging Face"
[21]: https://github.com/AndreyGuzhov/AudioCLIP?utm_source=chatgpt.com "GitHub - AndreyGuzhov/AudioCLIP: Source code for models described in the paper \"AudioCLIP: Extending CLIP to Image, Text and Audio\" (https://arxiv.org/abs/2106.13043) · GitHub"
[22]: https://github.com/openai/jukebox?utm_source=chatgpt.com "GitHub - openai/jukebox: Code for the paper \"Jukebox: A Generative Model for Music\" · GitHub"
[23]: https://github.com/rfhits/songcomposer_mcp?utm_source=chatgpt.com "GitHub - rfhits/songcomposer_mcp: [ACL 2025 Main] SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition · GitHub"
[24]: https://github.com/librosa/librosa/blob/main/LICENSE.md?utm_source=chatgpt.com "librosa/LICENSE.md at main · librosa/librosa · GitHub"
[25]: https://github.com/deezer/spleeter/wiki/2.-Getting-started/f13ee9d2efed269a67ff955f042c3d41de04a613?utm_source=chatgpt.com "2. Getting started · deezer/spleeter Wiki · GitHub"
[26]: https://github.com/Open-Unmix-Music-Source-Separation/?utm_source=chatgpt.com "Inria (Fabian-Robert Stöter, Antoine Liutkus) · GitHub"
[27]: https://github.com/shidephen/chordino/blob/master/plugins.cpp?utm_source=chatgpt.com "chordino/plugins.cpp at master · shidephen/chordino · GitHub"
[28]: https://openreview.net/forum?id=2s7ZZUhEGS&utm_source=chatgpt.com "MARBLE: Music Audio Representation Benchmark for Universal Evaluation | OpenReview"
[29]: https://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/download/6550/3950?utm_source=chatgpt.com "Procesamiento del Lenguaje Natural, Revista nº 71, septiembre de 2023, pp. 149-163"
[30]: https://github.com/microsoft/unilm/blob/master/e5/README.md?utm_source=chatgpt.com "unilm/e5/README.md at master · microsoft/unilm · GitHub"
[31]: https://huggingface.co/Alibaba-NLP/gte-multilingual-base?utm_source=chatgpt.com "Alibaba-NLP/gte-multilingual-base · Hugging Face"
[32]: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2?utm_source=chatgpt.com "sentence-transformers/paraphrase-multilingual-mpnet-base-v2 · Hugging Face"
[33]: https://huggingface.co/BAAI/bge-m3/blob/main/config.json?utm_source=chatgpt.com "config.json · BAAI/bge-m3 at main"
[34]: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5?utm_source=chatgpt.com "nomic-ai/nomic-embed-text-v1.5 · Hugging Face"
[35]: https://huggingface.co/Teradata/nomic-embed-text-v1.5?utm_source=chatgpt.com "Teradata/nomic-embed-text-v1.5 · Hugging Face"
[36]: https://www.nrc.canada.ca/en/research-development/products-services/technical-advisory-services/sentiment-emotion-lexicons?utm_source=chatgpt.com "Sentiment and emotion lexicons - National Research Council Canada"
[37]: https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt?utm_source=chatgpt.com "vaderSentiment/LICENSE.txt at master · cjhutto/vaderSentiment · GitHub"
[38]: https://github.com/facebookresearch/faiss?utm_source=chatgpt.com "GitHub - facebookresearch/faiss: A library for efficient similarity search and clustering of dense vectors. · GitHub"
[39]: https://developer.spotify.com/policy?utm_source=chatgpt.com "Spotify Developer Policy | Spotify for Developers"
[40]: https://developer.spotify.com/compliance-tips?utm_source=chatgpt.com "Compliance Tips | Spotify for Developers"
[41]: https://www.kaggle.com/terms?utm_source=chatgpt.com "Terms of Use | Kaggle"
[42]: https://musicbrainz.org/doc/About/Data_License?utm_source=chatgpt.com "About / Data License - MusicBrainz"
[43]: https://developer.spotify.com/documentation/web-api/reference/get-the-users-currently-playing-track?additional_types=episode&utm_source=chatgpt.com "Web API Reference | Spotify for Developers"
[44]: https://github.com/ismir/mir-datasets/blob/master/outputs/mir-datasets.md?utm_source=chatgpt.com "mir-datasets/outputs/mir-datasets.md at master · ismir/mir-datasets · GitHub"
[45]: https://huggingface.co/OpenMuQ/MuQ-MuLan-large?utm_source=chatgpt.com "OpenMuQ/MuQ-MuLan-large · Hugging Face"
[46]: https://github.com/shidephen/chordino?utm_source=chatgpt.com "GitHub - shidephen/chordino · GitHub"
