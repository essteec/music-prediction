# Feature-Expansion Plan for 10k-Song Music Dataset

> Produced 2026-08-19. Based on web research, codebase audit, and literature review.
> Hardware baseline: NVIDIA GTX 1660 Ti (6 GB VRAM), 16 GB RAM, consumer laptop.

---

## 1. Executive Recommendation

### Test First (Tier 0-1)

1. **LAION-CLAP `larger_clap_music`** (512-D cross-modal embedding) — uniquely bridges audio and text in a shared space; enables zero-shot tagging and text-query retrieval. Fits 6 GB VRAM. Low redundancy with MERT/PANNs. ~4-6 h for 10k songs.
2. **Essentia `MusicExtractor` + TF/ONNX classifiers** — CPU-only, fast, provides ~200+ interpretable scalar descriptors (key, scale, BPM, loudness LUFS, mood tags, voice/instrumental probability, dynamic complexity, onset rate). Highly complementary; nothing in the current stack provides structured MIR descriptors. ~8-12 h CPU.
3. **Upgrade lyric embedding to Nomic Embed v1.5** (768-D, 8192-token context, Apache-2.0, Matryoshka dims, 137M params) — replace MiniLM; keep MPNet as second baseline. No more truncation. ~30 min GPU for 10k.
4. **Structured librosa features beyond mel stats** — MFCCs, chroma CENS, spectral contrast, tonnetz, RMS energy, ZCR, tempogram. ~30-50 h CPU (parallelizable).
5. **Interpretable lyric features** — language ID (fasttext), VADER/RoBERTa emotion, readability (textstat), lexical richness, rhyme density, POS distribution, topic modeling (BERTopic). All CPU, minutes to hours.
6. **Data audit and QC pass** — Chromaprint fingerprinting for duplicate/mismatch detection, duration cross-check, silence/clipping detection, lyric boilerplate removal, language identification. Zero model cost.

### Avoid / Defer

1. **MERT-v1-330M** — requires ~8-12 GB VRAM; will OOM on 6 GB. Marginal gain over 95M per MARBLE benchmark.
2. **Jukebox representations** — 5-12B params, needs 16 GB+ VRAM, CC-BY-NC, deprecated by MERT.
3. **Full Demucs v4 source separation (10k songs)** — feasible on 6 GB but ~220-300 h GPU time and ~1.6 TB stems. Pilot 100 songs first.
4. **wav2vec 2.0 / HuBERT base** — speech SSL models fully superseded by MERT for music. Redundant.
5. **Generative LLM lyric annotation at scale** — local 3-4B models are feasible but slow, non-reproducible, and hallucination-prone. Pilot 200 songs first.
6. **MusicFM / Music2Vec** — nearly identical architecture and training signal to MERT-v1-95M. Highly redundant.

---

## 2. Current-State Audit

### Known Assets

| Category | Asset | Shape/Size | Notes |
|---|---|---|---|
| Metadata | `songs.csv` | 10000 x 32 | Spotify descriptors, genre, artist, lyrics |
| Audio files | Opus/WebM | 10000 files | YouTube-source, ~144 kb/s, 48 kHz stereo |
| VGGish | embedding | (10000, 128) | AudioSet-pretrained, general audio |
| MERT v1 95M | embedding | (10000, 768) | Music-specific SSL; 24 kHz mono; first 30 s |
| PANNs Cnn14 | embedding | (10000, 2048) | AudioSet-pretrained; song-level |
| Mel statistics | features | (10000, 512) | 128 mel bands x 4 stats |
| MiniLM lyric emb | embedding | (N, 384) | `all-MiniLM-L6-v2`; first 3000 chars; per-split |
| MPNet lyric emb | embedding | (N, 768) | `all-mpnet-base-v2`; first 3000 chars; normalized |
| Basic lyric stats | features | (N, 5) | word count, unique words, ratio, avg length, char count |
| TextBlob sentiment | features | (N, 2) | polarity, subjectivity; English-centric |
| Thesis pipeline | 4254-D input | train/val/test | 30 metadata + 768 MERT + 128 VGGish + 768 MPNet + 2048 PANNs + 512 mel |

### Key Gaps

1. **No structured MIR descriptors** from audio (no key, chord, BPM, onset density, dynamic range, harmonic/percussive ratio, vocal/instrumental classification computed from YOUR audio files)
2. **No cross-modal embedding** connecting audio and text in a shared space
3. **Lyrics truncated at 3000 chars** — both MiniLM and MPNet max at 512 tokens
4. **No language identification** — multilingual corpus with English-only sentiment tools
5. **No emotion model beyond TextBlob** — no dimensional emotion, no multi-label emotion
6. **No audio QC / duplicate detection** — YouTube audio could be live versions, remasters, or wrong songs
7. **No temporal pooling beyond mean** — temporal structure (intro vs chorus vs bridge) is lost
8. **Lyric boilerplate not cleaned** — `[Verse]`, `[Chorus]`, tags and contributor credits likely present

---

## 3. Candidate Matrix: Audio

### Tier 1: High Value, Feasible Now

| # | Candidate | Dims | Training Data | Encodes | Redundancy | VRAM | Runtime 10k | License | Kaggle-safe | Best Use |
|---|---|---:|---|---|---|---|---|---|---|---|
| A1 | **LAION-CLAP music** | 512 | LAION-Audio-630K + music | Audio-text alignment | **Low** (cross-modal) | ~2 GB | ~4-6 h GPU | Apache-2.0/MIT | Yes | Retrieval, zero-shot tags |
| A2 | **Essentia MusicExtractor** | ~200+ | N/A (DSP) | Key, BPM, loudness, onset | **Very low** | CPU only | ~8-12 h | AGPLv3 code; data OK | Yes | Prediction, interpretability |
| A3 | **Essentia TF classifiers** | ~50 probs | MTG-Jamendo | Mood, genre, voice/instr | **Low** (tags) | CPU | ~2-4 h | CC-BY-NC-SA 4.0 | Likely yes | Tags for UI, validation |
| A4 | **Librosa extended** | ~150 | N/A (DSP) | MFCCs, chroma, tonnetz, spectral | **Medium** (mel overlap) | CPU | ~30-50 h | ISC | Yes | Prediction, interpretability |
| A5 | **Chromaprint** | 1 hash | N/A | Audio identity | **None** (QC) | CPU | ~30 min | LGPL-2.1 | Yes | QC: duplicates |
| A6 | **Silero VAD** | ~3 scalars | Speech | Vocal presence/ratio | **None** | ~2 MB | ~1-2 h | MIT | Yes | Prediction, QC |

**Sources:** [LAION-CLAP paper](https://arxiv.org/abs/2211.06687) | [HF: laion/larger_clap_music](https://huggingface.co/laion/larger_clap_music) | [Essentia](https://essentia.upf.edu/) | [Essentia models](https://essentia.upf.edu/models.html) | [librosa](https://librosa.org/) | [Silero VAD](https://github.com/snakers4/silero-vad)

### Tier 2: Pilot First

| # | Candidate | Dims | Redundancy | VRAM | Runtime 10k | License | Pilot Protocol |
|---|---|---:|---|---|---|---|---|
| A7 | BEATs | 768 | **High** with PANNs | ~2 GB | ~6-8 h | MIT | 200 songs; reject if sim ranking <5% different from PANNs |
| A8 | AST | 768 | **High** with PANNs | ~2 GB | ~4-6 h | BSD-3 | Same as BEATs |
| A9 | Demucs v4 (separation) | 4 stems | **None** (unique) | ~4 GB | ~220-300 h | MIT | 100 songs; assess stem quality for pitch tracking |
| A10 | CREPE pitch (on vocals) | ~10 | **None** | CPU | ~2 h post-stems | MIT | Depends on A9 quality |
| A11 | Multi-window pooling | 3-5x base | **Low** | N/A | Same as base | N/A | 200 songs; test retrieval vs single mean |

**Sources:** [BEATs](https://arxiv.org/abs/2212.09058) | [AST](https://arxiv.org/abs/2104.01778) | [Demucs](https://github.com/facebookresearch/demucs) | [CREPE](https://github.com/marl/crepe)

### Tier 3: Future / Requires Better Hardware

| # | Candidate | Why Deferred |
|---|---|---|
| A12 | MERT-v1-330M (1024-D) | Needs ~8-12 GB VRAM |
| A13 | Jukebox (4800-D) | 5-12B params, 16 GB+ VRAM, CC-BY-NC |
| A14 | MusicFM 330M | Same VRAM issue as MERT-330M |
| A15 | Music2Vec | Redundant with MERT-v1-95M |
| A16 | AudioMAE | Redundant with PANNs/BEATs; CC-BY-NC |
| A17 | MuLan (Google) | **Not publicly available** |
| A18 | wav2vec 2.0 / HuBERT | Speech SSL; superseded by MERT |
| A19 | EnCodec | Codec, not semantic extractor |

### Key Audio Candidate Details

**A1: LAION-CLAP** is the highest-priority new audio feature. It produces embeddings in a shared audio-text space: you can query "upbeat electronic dance track with female vocals" and retrieve songs by cosine similarity. MERT encodes music structure; PANNs encode audio events; CLAP encodes *what natural language describes about the audio*. Orthogonal signal. Input: 48 kHz audio. Output: 512-D L2-normalized vector. HTSAT-large audio encoder + RoBERTa text encoder, ~2 GB VRAM. Pilot: 100 songs, compare cosine-sim ranking vs MERT (expect Spearman rho < 0.7 = complementary).

**A2-A3: Essentia** fills the biggest gap: your pipeline has *zero* structured MIR descriptors from the audio itself. Spotify provides `key`, `tempo`, etc., but those are black-box API outputs. Essentia gives open, reproducible, interpretable descriptors. MusicExtractor outputs: tonal (key, scale, HPCP 36-D), rhythm (BPM, onset rate), loudness (LUFS, LRA, dynamic complexity), spectral (centroid, dissonance). TF classifiers add mood/genre/voice-instrumental tags. Runtime: >10x real-time per CPU core.

**A4: Librosa extended** adds MFCCs (timbre), chroma (harmony), tonnetz (harmonic network), spectral contrast (harmonic vs noise). MFCCs have moderate correlation with mel stats (MFCCs are a DCT of mel bands), but chroma, tonnetz, and spectral contrast are genuinely new. Pilot: extract for 200 songs, keep only groups with max |r| < 0.85 vs mel stats.

---

## 4. Candidate Matrix: Lyrics

### Tier 1: High Value, Feasible Now

| # | Candidate | Dims | Tool | Max Tokens | Multilingual | License | Runtime 10k | Use |
|---|---|---:|---|---:|---|---|---|---|
| L1 | **Nomic Embed v1.5** | 768 | `nomic-ai/nomic-embed-text-v1.5` | 8192 | 100+ langs | Apache-2.0 | ~20-30 min GPU | Prediction + Retrieval |
| L2 | **Language ID** | 1+conf | `fasttext lid.176.bin` | N/A | 176 langs | CC-BY-SA 3.0 | ~2 min CPU | Stratification |
| L3 | **VADER sentiment** | 4 | `vaderSentiment` | N/A | English | MIT | ~1 min CPU | Replace TextBlob |
| L4 | **GoEmotions RoBERTa** | 28 | `SamLowe/roberta-base-go_emotions` | 512 | English | MIT | ~30 min GPU | Emotion similarity |
| L5 | **textstat readability** | ~10 | `textstat` | N/A | Partial | MIT | ~2 min CPU | Prediction |
| L6 | **Lexical richness** | ~8 | `lexicalrichness` | N/A | Agnostic | MIT | ~5 min CPU | Prediction |
| L7 | **Lyric cleaning** | N/A | regex + custom | N/A | All | N/A | ~5 min CPU | Prerequisite |
| L8 | **BERTopic** | topic+prob | `BERTopic` | N/A | Via embeddings | MIT | ~10-15 min | Theme similarity |

**Sources:** [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | [GoEmotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | [BERTopic](https://github.com/MaartenGr/BERTopic) | [fasttext lang ID](https://fasttext.cc/docs/en/language-identification.html)

### Tier 2: Pilot First

| # | Candidate | Dims | Notes |
|---|---|---:|---|
| L9 | BGE-M3 | 1024 | 8192 tokens, multilingual, MIT. Test if marginal gain over Nomic |
| L10 | Jina v3 | 1024 | **CC-BY-NC 4.0** — may restrict Kaggle |
| L11 | spaCy POS/NER | ~20 | English-only quality; POS distribution, entity counts, pronoun ratios |
| L12 | Rhyme detection | ~5 | `pronouncing` CMU dict; English-only; noisy on slang |
| L13 | Repetition features | ~8 | Line repetition ratio, chorus detection, hookiness proxy |
| L14 | Detoxify toxicity | 6 | Correlates with explicit flag |

### Tier 3: Future

| # | Candidate | Why Deferred |
|---|---|---|
| L15 | LLM annotation (Phi-3-mini) | Slow, non-reproducible, hallucination risk. Pilot 200 songs first |
| L16 | Fine-tuned lyric embedding (LoRA) | 10k songs may be too few; needs positive/negative pairs |
| L17 | Lyric-audio alignment | Requires CLAP first; research-grade |
| L18 | LIWC psycholinguistics | Paid license ($99+); open alternatives less validated |

### Key Lyric Candidate Details

**L1: Nomic Embed v1.5** — 8192-token context means *no truncation* for virtually any song lyrics. Matryoshka representation allows 768-D full quality or 256-D/128-D for faster retrieval. vs MPNet: MPNet maxes at 512 tokens and requires truncating lyrics. Nomic was designed for semantic search. Preprocessing: prefix `search_document:` for indexing, `search_query:` for queries. Pilot: 500 songs, full lyrics (Nomic) vs truncated (MPNet), compare Recall@10.

**L4: GoEmotions RoBERTa** — 28 emotion probabilities (admiration, amusement, anger, joy, love, sadness, surprise, etc.). Far richer than TextBlob's 2-D. English-only; for non-English, apply only to English-detected lyrics. Chunk long lyrics into 512-token segments, mean-pool probabilities.

**L7: Lyric Cleaning Pipeline (critical prerequisite):**
```python
import re, unicodedata

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Verse], [Chorus] etc.
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Contributors?|Lyrics?\s*by|Source|Embed|You might also like|\d+Embed)',
        l.strip(), re.IGNORECASE)]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
```

**Missing lyrics (203 songs):** Zero vector for embeddings, `has_lyrics=False` flag, NaN for stats. Do not impute.

**Copyright:** Do NOT distribute raw lyrics on Kaggle. Distribute only computed features.

---

## 5. Lyrics Benchmark Answer

### Is there a benchmark that lets me choose the best lyric model directly?

**No.** There is no single, broadly accepted benchmark for lyric embedding evaluation. Reasons:

1. **Copyright prevents shared datasets.** Lyrics are copyrighted; the musiXmatch dataset contained only bag-of-words and is [no longer officially distributed](http://millionsongdataset.com/musixmatch/).
2. **Task heterogeneity.** "Best" depends on similarity, emotion, genre, topic, or cross-lingual retrieval.
3. **Language diversity.** Most NLP benchmarks are English-only; lyrics span dozens of languages.
4. **No ground-truth similarity labels.** Lyric similarity is subjective and task-dependent.

### Existing Datasets

| Dataset | Content | Size | Available | License | Languages | Limitation |
|---|---|---|---|---|---|---|
| musiXmatch | BoW lyrics (MSD) | 237k tracks | Discontinued | Research-only | English | BoW only |
| DALI v2 | Time-aligned lyrics | ~7900 songs | [GitHub](https://github.com/gabolsgabs/DALI) | CC-BY-NC-SA | Mostly English | No audio |
| WASABI | Metadata + lyrics + NLP | 2.1M songs | [Site](https://wasabi.i3s.unice.fr/) | Research-only | Multilingual | Redistribution restricted |
| MTG-Jamendo | Audio + tags | 55k tracks | [GitHub](https://mtg.github.io/mtg-jamendo-dataset/) | CC | English/instrumental | No lyrics |
| LyricSIM | Annotated pairs | 676 pairs | Contact authors | Research-only | Spanish | Very small |
| MARBLE | Audio benchmark | 8 tasks | [Site](https://marble-bm.sheffield.ac.uk/) | Varies | N/A | No lyric tasks |
| MoodyLyrics | V-A labeled lyrics | 2000 songs | Paper | Research-only | English | Small |

### In-Domain Evaluation Design

**A. Human-Judged Similarity (Gold Standard)**
- 30 query songs stratified by language, genre, lyric length
- 20 candidates per query (5 same-genre, 5 same-artist, 10 random) = 600 pairs
- 5-point Likert on: thematic, emotional, stylistic, lexical, overall similarity
- 3 annotators per pair; Krippendorff's alpha > 0.6
- 10% obvious-similar + 10% obvious-dissimilar controls

**B. Quantitative Retrieval**
- Cosine-similarity ranking, Recall@10/50, nDCG@10, MRR
- Proxy labels: same-artist, same-genre, same-language, shared BERTopic topic
- Artist-aware: no same-artist songs in both query and candidate sets
- Separate English vs non-English reporting

**C. Baselines**
- Random | TF-IDF | MiniLM (384-D) | MPNet (768-D) | Nomic (768-D) | BoW+LSA

---

## 6. Website / Product Concepts

Ranked by user value x feasibility:

| # | Concept | User Need | Key Inputs | Feasible Now |
|---|---|---|---|---|
| 1 | **Overall Similar Songs** | "Find songs like X" | Weighted audio + lyric similarity | Yes |
| 2 | **Lyrically Similar Songs** | "Find songs with similar lyrics" | Nomic lyric cosine similarity | Yes (after Nomic) |
| 3 | **Song Map** | Visual exploration | UMAP 2-D of combined embeddings | Yes |
| 4 | **Controllable Search** | "Like X, but calmer" | Multiple embeddings + Essentia descriptors + sliders | Yes (re-ranking) |
| 5 | **Mood Mismatch Explorer** | Audio happy / lyrics sad | Essentia mood + GoEmotions | Yes (after extraction) |
| 6 | **Era Map** | Music evolution over time | Release date + embeddings | Yes |
| 7 | **Theme Map** | Explore by lyrical themes | BERTopic + UMAP | Yes (after BERTopic) |
| 8 | **Duplicate Explorer** | Find covers/remasters | Chromaprint + chroma similarity | Yes (after fingerprint) |
| 9 | **Playlist Diagnostics** | Analyze playlist coherence | Any embedding space | Yes |
| 10 | **A/B Similarity Eval** | Crowdsource judgments | Web UI + embeddings | Yes |

> [!WARNING]
> **UMAP is for exploration, not similarity search.** UMAP distorts distances; nearby points in 2-D may not be truly similar in high-D space. Always use original embeddings for nearest-neighbor queries.

**Indexing:** At 10k songs, brute-force cosine similarity (<100 ms) is sufficient. At 100k+, use FAISS `IndexFlatIP` or `IndexIVFFlat`. Do not over-engineer ANN until scale demands it.

---

## 7. Kaggle-Ready Data Architecture

### Directory Layout

```
music-prediction-dataset-v1/
  README.md                          # Dataset card
  LICENSE                            # CC-BY-4.0
  CHANGELOG.md                       # Version history  
  DATA_DICTIONARY.md                 # Column documentation
  PROVENANCE.md                      # Sources, dates, tools
  metadata/
    songs.parquet                    # 10k rows, NO raw lyrics
    songs_schema.json
    genre_taxonomy.json
  features/
    spotify_audio_descriptors.parquet
    essentia_descriptors.parquet
    librosa_extended.parquet
    lyric_statistics.parquet
    lyric_emotions.parquet
    lyric_topics.parquet
    language_id.parquet
    quality_control.parquet
  embeddings/
    audio/
      vggish_128d.npy
      mert_v1_95m_768d.npy
      panns_cnn14_2048d.npy
      clap_music_512d.npy
      mel_statistics_512d.npy
    lyrics/
      nomic_embed_v1.5_768d.npy
      mpnet_base_v2_768d.npy
      minilm_v2_384d.npy
    track_ids.npy                    # Shared ID index
  similarity/
    umap_2d_combined.parquet
    umap_2d_audio.parquet
    umap_2d_lyrics.parquet
  manifests/
    extraction_manifest.json
    embedding_checksums.json
    feature_statistics.json
  notebooks/
    01_exploratory_analysis.ipynb
    02_similarity_demo.ipynb
    03_retrieval_demo.ipynb
```

### What NOT to Distribute

| Content | Reason | Alternative |
|---|---|---|
| Raw lyrics text | Copyright | Computed features only |
| Audio files | YouTube TOS | Track IDs for users to obtain independently |
| Audio stems | Derived from copyrighted audio | Computed features from stems only |
| Spotify API raw responses | Spotify TOS | Subset Spotify allows; users fetch own |

### Format Recommendations

| Content | Format | Rationale |
|---|---|---|
| Metadata + scalars | Parquet (snappy) | Columnar, typed, efficient |
| High-D embeddings | NPY (float32) | Direct NumPy load; fast |
| ID alignment | Single `track_ids.npy` | Prevents misalignment |
| Manifests | JSON | Human-readable |
| Docs | Markdown | GitHub/Kaggle renders natively |

Store embeddings **unnormalized**. Document which models output L2-normalized vectors. Users normalize at load time.

---

## 8. Staged 10k-Song Roadmap

### Tier 0: Data Audit (Week 1, ~1 day compute)

| Priority | Task | Compute | Go/No-Go |
|---|---|---|---|
| 0.1 | Lyric cleaning pipeline | CPU, ~5 min | Always do |
| 0.2 | Language ID (fasttext) | CPU, ~2 min | Always do |
| 0.3 | Chromaprint fingerprinting | CPU, ~30 min | Always do |
| 0.4 | Duration cross-check vs Spotify | CPU, ~10 min | Always do |
| 0.5 | Silence/clipping detection | CPU, ~1 h | Always do |
| 0.6 | Lyric structure features | CPU, ~5 min | Always do |
| 0.7 | VADER sentiment | CPU, ~1 min | If corr(VADER, TextBlob) < 0.85 |

**Gate:** Review QC report. Flag duplicates, duration mismatches >20%, excessive silence, missing lyrics.

### Tier 1: High-Value Additions (Weeks 2-3, ~2-3 days compute)

| Priority | Task | Compute | Storage | Go/No-Go |
|---|---|---|---|---|
| 1.1 | Essentia MusicExtractor | CPU, ~8-12 h | ~50 MB | Always do |
| 1.2 | Essentia TF classifiers | CPU, ~2-4 h | ~5 MB | Always do |
| 1.3 | Nomic Embed v1.5 lyrics | GPU, ~20-30 min | ~30 MB | Always do |
| 1.4 | GoEmotions RoBERTa | GPU, ~30 min | ~3 MB | Always do |
| 1.5 | LAION-CLAP audio | GPU, ~4-6 h | ~20 MB | Always do |
| 1.6 | textstat readability | CPU, ~2 min | ~1 MB | Always do |
| 1.7 | Lexical richness | CPU, ~5 min | ~1 MB | Always do |
| 1.8 | Silero VAD | CPU, ~1-2 h | ~200 KB | Always do |
| 1.9 | BERTopic | CPU/GPU, ~10-15 min | ~1 MB | Always do |

**Gate:** Run ablation (Section 9). Measure CLAP vs MERT/PANNs retrieval, Nomic vs MPNet retrieval, Essentia vs metadata prediction R-squared.

### Tier 2: Pilot then Scale (Weeks 4-6)

| Task | Pilot | Scale If | Full Compute |
|---|---|---|---|
| Librosa extended | 200 songs, ~1 h | Corr with mel < 0.85 for >50% features | ~30-50 h CPU |
| Demucs v4 separation | 100 songs, ~20-30 h | Stems clean enough for CREPE | ~220-300 h GPU |
| CREPE pitch tracking | 100 songs post-Demucs | Vocal features r > 0.15 with target | ~20 h CPU |
| Multi-window pooling | 200 songs | Recall@10 improves >5% | Re-extract all 10k |
| BGE-M3 lyrics | 500 songs, ~15 min | Retrieval >3% better than Nomic | ~30 min GPU |
| spaCy POS/NER | 500 songs | POS correlates with genre r > 0.2 | ~30 min CPU |
| Rhyme detection | 500 English songs | Correlates with genre/era | ~15 min CPU |
| LLM annotation | 200 songs | Coherent >80%; improves topics | ~6-14 h GPU |

### Tier 3: Future

| Task | Prerequisite | Expected Value |
|---|---|---|
| MERT-v1-330M | 12+ GB VRAM | Marginal over 95M |
| LoRA fine-tuned lyrics | Validated eval set | High for specialized similarity |
| Lyric-audio alignment | CLAP + Nomic extracted | Research-grade |
| Full-song temporal sequences | Retrieval infra | Temporal similarity queries |

---

## 9. Ablation / Evaluation Protocol

### Supervised Prediction

**Targets:** valence, energy, danceability, popularity.

**Splits:** Artist-aware stratified 70/15/15. No artist in both train and test.

**Models:** Ridge (baseline), RF, XGBoost, LightGBM. Fixed hyperparameters for ablation.

**Metrics:** R-squared (primary), MAE, RMSE. 95% CI via bootstrap (1000 iterations).

| Ablation ID | Features | Dims | Purpose |
|---|---|---:|---|
| B0 | Spotify metadata only | 13 | Baseline |
| B1 | B0 + audio embeddings | 3469 | Current audio stack |
| B2 | B1 + MPNet + MiniLM + stats + sentiment | 4628 | Current full pipeline |
| A_clap | B2 + CLAP (512) | 5140 | CLAP marginal value |
| A_essentia | B2 + Essentia (~200) | 4828 | Structured MIR value |
| L_nomic | B1 + Nomic + emotions + VADER + readability | ~4287 | Upgraded lyric stack |
| FULL | B1 + CLAP + Essentia + full lyrics | ~5700 | Everything |

**Dimensionality reduction:** PCA per feature group, 95% variance, fitted on train only.

**Leakage checks:** popularity not as input when predicting popularity; Spotify descriptors not as input when predicting themselves; no artist ID as input.

### Retrieval Evaluation

**Audio:** Cosine similarity per embedding. Proxy labels: same-genre, same-artist (evaluate separately). Metrics: Recall@10/50, nDCG@10, MRR. Exclude same-artist from primary metrics.

**Lyrics:** Cosine similarity per embedding. Proxy labels: same-topic (BERTopic), same-language, same-genre. Compare: MPNet | MiniLM | Nomic | TF-IDF.

**Combined:** Weighted sim = alpha * audio + (1-alpha) * lyric. Sweep alpha in {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}.

### Temporal Pooling Ablation

For MERT/PANNs:
1. Mean only (current)
2. Mean + std
3. Mean + std + max
4. 3-segment (intro/middle/outro) means
5. Attention-weighted mean (learned)

Adopt if >2% Recall@10 improvement.

### Record-Keeping

For every extraction: model name, version, HF commit hash, preprocessing params, pooling method, date, runtime, failure rate, output shape, dtype, NaN/Inf count, SHA-256 checksum. Store in `manifests/extraction_manifest.json`.

---

## 10. Risk Register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Lyrics copyright** | High | Do NOT distribute raw lyrics; features only |
| R2 | **YouTube audio provenance** | High | Do NOT distribute audio; features only |
| R3 | **Model weight licenses** | Medium | MERT: CC-BY-NC; Essentia TF: CC-BY-NC-SA. Prefer Apache-2.0 models. Computed features are not weights |
| R4 | **Spotify API TOS** | Medium | Review developer terms; consider BYO-data approach |
| R5 | **Target leakage** | High | Feature-target exclusion in pipeline |
| R6 | **Artist/duplicate leakage** | High | Artist-aware splits; report with/without artist overlap |
| R7 | **Multilingual bias** | Medium | Language ID; English-only tools on English lyrics only |
| R8 | **Audio mismatch** | Medium | Chromaprint + duration check + voice/instrumental flag |
| R9 | **Extraction failures** | Medium | Resumable checkpoints; validate NaN/Inf/shape |
| R10 | **Embedding version drift** | Low | Pin HF commit hashes; store checksums |
| R11 | **Hardware OOM** | Medium | Batch size 1; FP16 where stable; CPU fallback |
| R12 | **Storage growth** | Medium | Features: ~200 MB total. Stems: ~1.6 TB (avoid unless committed) |
| R13 | **Essentia AGPLv3** | Low | Affects code distribution, not data |
| R14 | **ISRC codes** | Low | Public identifiers; freely publishable |
| R15 | **Popularity corpus bias** | Info | Document: convenience sample, not representative |

---

## 11. Annotated Bibliography

### Audio Models

| Resource | License | Links |
|---|---|---|
| MERT (2023) | CC-BY-NC 4.0 | [Paper](https://arxiv.org/abs/2306.00107) / [HF 95M](https://huggingface.co/m-a-p/MERT-v1-95M) / [HF 330M](https://huggingface.co/m-a-p/MERT-v1-330M) |
| PANNs (2020) | MIT | [Paper](https://arxiv.org/abs/1912.10211) / [GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| VGGish (2017) | Apache-2.0 | [Paper](https://arxiv.org/abs/1609.09430) / [TF Hub](https://tfhub.dev/google/vggish/1) |
| LAION-CLAP (2023) | Apache-2.0/MIT | [Paper](https://arxiv.org/abs/2211.06687) / [GitHub](https://github.com/LAION-AI/CLAP) / [HF](https://huggingface.co/laion/larger_clap_music) |
| BEATs (2023) | MIT | [Paper](https://arxiv.org/abs/2212.09058) / [GitHub](https://github.com/microsoft/unilm/tree/master/beats) |
| AST (2021) | MIT | [Paper](https://arxiv.org/abs/2104.01778) / [GitHub](https://github.com/YuanGongND/ast) |
| Demucs v4 (2023) | MIT | [Paper](https://arxiv.org/abs/2211.08553) / [GitHub](https://github.com/facebookresearch/demucs) |
| Jukebox (2020) | CC-BY-NC | [Paper](https://arxiv.org/abs/2005.00341) / [GitHub](https://github.com/openai/jukebox) |
| Music2Vec (2022) | CC-BY-NC 4.0 | [Paper](https://arxiv.org/abs/2212.02508) / [HF](https://huggingface.co/m-a-p/music2vec-v1) |
| Silero VAD (2021) | MIT | [GitHub](https://github.com/snakers4/silero-vad) |
| CREPE (2018) | MIT | [Paper](https://arxiv.org/abs/1802.06182) / [GitHub](https://github.com/marl/crepe) |

### Text Embedding Models

| Resource | License | Links |
|---|---|---|
| Nomic Embed v1.5 (2024) | Apache-2.0 | [Paper](https://arxiv.org/abs/2402.01613) / [HF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| MPNet (2020) | Apache-2.0 | [HF](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| MiniLM (2020) | Apache-2.0 | [HF](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| BGE-M3 (2024) | MIT | [Paper](https://arxiv.org/abs/2402.03216) / [HF](https://huggingface.co/BAAI/bge-m3) |
| GTE-large (2024) | Apache-2.0 | [HF](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) |

### MIR Libraries

| Resource | License | Links |
|---|---|---|
| librosa 0.10+ | ISC | [Docs](https://librosa.org/) / [GitHub](https://github.com/librosa/librosa) |
| Essentia 2.1 | AGPLv3 | [Site](https://essentia.upf.edu/) / [GitHub](https://github.com/MTG/essentia) / [Models](https://essentia.upf.edu/models.html) |
| Chromaprint | LGPL-2.1 | [GitHub](https://github.com/acoustid/chromaprint) |
| pyacoustid | MIT | [PyPI](https://pypi.org/project/pyacoustid/) |

### NLP / Lyric Tools

| Resource | License | Links |
|---|---|---|
| GoEmotions (2020) | Apache-2.0 | [Paper](https://arxiv.org/abs/2005.00547) / [HF](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| VADER (2014) | MIT | [GitHub](https://github.com/cjhutto/vaderSentiment) |
| BERTopic (2022) | MIT | [Paper](https://arxiv.org/abs/2203.05794) / [GitHub](https://github.com/MaartenGr/BERTopic) |
| textstat | MIT | [PyPI](https://pypi.org/project/textstat/) |
| lexicalrichness | MIT | [PyPI](https://pypi.org/project/lexicalrichness/) |
| fasttext langdetect | CC-BY-SA 3.0 | [Docs](https://fasttext.cc/docs/en/language-identification.html) |
| spaCy | MIT | [Site](https://spacy.io/) |
| pronouncing | MIT | [GitHub](https://github.com/aparrish/pronouncing) |
| Detoxify | Apache-2.0 | [GitHub](https://github.com/unitaryai/detoxify) |

### Benchmarks and Datasets

| Resource | License | Links |
|---|---|---|
| MARBLE (2023) | Varies | [Paper](https://arxiv.org/abs/2306.10102) / [Site](https://marble-bm.sheffield.ac.uk/) |
| MTG-Jamendo (2019) | CC | [GitHub](https://github.com/MTG/mtg-jamendo-dataset) |
| DALI v2 (2019) | CC-BY-NC-SA | [GitHub](https://github.com/gabolsgabs/DALI) |
| AudioSet (2017) | CC-BY 4.0 | [Site](https://research.google.com/audioset/) |
| FMA (2017) | CC | [GitHub](https://github.com/mdeff/fma) |
| MTEB (2022) | Apache-2.0 | [Paper](https://arxiv.org/abs/2210.07316) / [HF](https://huggingface.co/spaces/mteb/leaderboard) |

### Similarity Search

| Resource | License | Links |
|---|---|---|
| FAISS (2017) | MIT | [GitHub](https://github.com/facebookresearch/faiss) |
| UMAP (2018) | BSD-3 | [Paper](https://arxiv.org/abs/1802.03426) / [GitHub](https://github.com/lmcinnes/umap) |
| Annoy | Apache-2.0 | [GitHub](https://github.com/spotify/annoy) |

> [!IMPORTANT]
> At 10k songs, brute-force cosine similarity is fast enough (<100 ms). Do not over-engineer ANN indexes until past 100k songs.

> [!NOTE]
> Never distribute raw lyrics. Always distribute computed features only. This is the single most important legal consideration for the Kaggle dataset.
