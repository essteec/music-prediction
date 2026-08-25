# ULTIMATE RESEARCH PROMPT — Music Feature Extraction & Product Brainstorm

> Copy everything below this line into your favorite LLM (Claude, ChatGPT, Gemini,
> DeepSeek, etc.). The prompt is self-contained: all context is inline, no repo access needed.

---

You are a senior music-information-retrieval (MIR) and ML research consultant. I need a
comprehensive, research-backed brainstorm report. **Use web search to verify and extend
everything** — models, benchmarks, licenses, VRAM footprints, and current SOTA as of mid-2026.
Do not rely only on your training memory; every claim that matters (model exists, dims, license,
benchmark name) must be verified with a URL source.

## MY CONTEXT (ground truth — assume everything below is accurate)

### The dataset I own
- **10,000 top-streamed songs** (global Spotify chart, July 2025), all metadata in one CSV.
- **10,000 audio files**: `data/audio/pilot/*.webm` named `000000_opus.webm` … `009999_opus.webm`.
  Each file is WebM/Matroska container, **Opus codec, 48 kHz, stereo, ~144 kb/s, 3–4 MB** (~34 GB total).
  File metadata embeds `track_id`, `album_id`, `isrc`, artist, album, date, genre (via ffprobe COMMENT tags).
- **Metadata CSV columns** (one row per song, 32 columns):
  `rank, track_name, track_id, artist_names, artist_ids, album_name, album_id, popularity, duration,
  explicit, release_date, album_type, isrc, copies, danceability, energy, key, loudness, mode,
  speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature,
  total_artist_followers, avg_artist_popularity, artist_genres, main_genres, lyrics`
  (the first 10 audio columns are Spotify-provided audio features — I treat them as labels/gold,
  not extracted features).
- **Lyrics**: `lyrics` column present for ~9,800 of 10,000 songs (203 missing). **Multilingual** —
  many songs are non-English (Spanish, Turkish, Korean, Hindi, Portuguese, …). Lyrics have no
  alignment with time; plain text only.
- **Artists CSV**: `artist_id, name, followers, popularity, genres, main_genre` (5,015 artists).

### Audio features I ALREADY extracted (these exist, do not re-suggest unless as a better variant)
All are per-song global vectors via temporal mean pooling, extracted with fp16, batch size 1:
| Extractor | Dim | Notes |
|---|---|---|
| VGGish | 128 | 16 kHz, 0.96s frames |
| MERT-v1-95M (music transformer) | 768 | 24 kHz, 30 s audio cap |
| PANNs Cnn14 (AudioSet tagging) | 2048 | 32 kHz, penultimate layer |
| Mel spectrogram statistics | 512 | 128 bands × 4 stats (mean/std/max/min) |

So: **4,256-d of audio features already exist** per song. My pipeline saves each song's vector,
supports `--resume` checkpoints, and mean-pools over time.

### Lyrics features I ALREADY used in a previous project (but NOT yet extracted for this 10k set)
- 5 text statistics (word_count, unique_word_count, unique_ratio, avg_word_length, char_count)
- 2 TextBlob sentiment features (polarity, subjectivity)
- all-mpnet-base-v2 sentence embeddings (768-d)

### My hardware (very important — everything must fit this)
- Consumer laptop, **NVIDIA GTX 1660 Ti 6 GB VRAM**, **16 GB RAM**, no ECC, thermal-limited.
- I can run long batch jobs overnight/for days; slow is OK, OOM is not.
- I have CPU-only fallback capability for small models. Python + PyTorch + librosa + transformers
  + sentence-transformers ecosystem. No cloud GPUs.

### My goals (ranked)
1. **Research phase (now)**: Find the BEST additional feature-extraction methods — audio and lyrics —
   so I can test several and pick winners. I want to compare which feature set is better for what.
2. **Publish a dataset on Kaggle**: 10k songs metadata + audio features + lyrics features,
   structured for ease of use (e.g., clean CSVs/parquets/npz, feature order documented, split files).
3. **Build a music-lover website** (longer term) with tools like: overall similar songs, most
   similar-lyrics songs, a 2D/3D "song map" (similarity embedding visualization), and more tools.
   This requires compact, scalable data (I cannot ship 34 GB of audio; I will ship features).

### Prior benchmark context from my previous project (for reference only)
On 490k songs with 4,254 features (23 audio + 5 text stats + 2 sentiment + 768 MPNet + 128 VGGish
+ 768 MERT + 2048 PANNs + 512 mel stats), best model CatBoost reached test R²: valence 0.72,
energy 0.92, danceability 0.79, popularity 0.13. So popularity is known-hard (external factors).

---

## WHAT I NEED FROM YOU

Produce a single, well-structured markdown report with the 6 sections below. **Be concrete and
ranked.** Every idea must include: name, what it is, output dimension & type (global vector vs
per-second sequence), model size/parameters, estimated VRAM/RAM, estimated extraction time per
song on MY hardware (GTX 1660 Ti 6GB) and total for 10k songs, license, source (URL), and a
feasibility rating (Easy / Medium / Hard) plus expected value for my goals. Cite sources.

---

### SECTION 1 — Audio feature extraction ideas (beyond what I have)

Brainstorm broadly across ALL families, then rank:

1. **Lightweight pretrained audio/music embedding models** I haven't tried (verify current SOTA):
   - Music-specific: Music2Vec, EnCodecMAE, CLAP-family (LAION-CLAP, Music-CLAP), MERT-v1-330M
     (bigger sibling of what I have), Jukebox (probably too big), SongComposer, MusicFM, WavLabLM,
     GR-Music, MAESTRO-based, etc.
   - General audio: BEATs, AudioMAE, CAV-MAE, SSAST, AST, OpenL3, YAMNet, TRILL, Wav2CLIP,
     AudioCLIP, SoundStream embeddings, ACE, ESResNeXt, etc.
   - For each: is it feasible on 6 GB VRAM? What pooling is recommended (mean vs attention vs CLS)?
2. **Better pooling of EXISTING embeddings** (cheap wins):
   - Attention-weighted temporal pooling, max/quantile stats over time, beat-synchronous pooling,
     first/last frame (CLS token) instead of mean, per-chunk statistics of the 30 s MERT segment.
   - Do any papers show mean-pooling is suboptimal for music similarity tasks?
3. **Classical DSP / handcrafted features** (librosa, essentia, madmom — CPU-only, free):
   - Chroma/tonnetz/harmonic features, MFCC + deltas + stats, spectral contrast/flux/rolloff/bandwidth,
     onset strength & tempogram, beat/downbeat tracking, HPSS (harmonic/percussive separation)
     feature ratios, key/scale detection, tuning frequency, "drop" detection, novelty curves,
     RMS energy envelope statistics, zero-crossing, spectral centroid over time, vocal-activity ratio.
   - What are the ~10 highest-value handcrafted features for (a) mood, (b) energy/danceability,
     (c) genre & similarity? Keep the total under ~50-100 dims and rank them.
4. **Vocal/singing analysis**: singing-voice separation (Demucs/Spleeter/Open-Unmix on 6 GB?),
   then extract vocal-only and instrumental-only embeddings; vocal activity detection; vocal range
   estimation; is this worth it for my goals?
5. **Harmony/chord extraction**: Chordino, CREMA? (no — CREMA is emotion), DeepChroma, or librosa
   chroma-based chord estimation. Chord histograms / chord complexity as features.
6. **Semantic/tagging embeddings**: PANNs already gives 527 AudioSet tag probabilities — did I
   waste them by only keeping the 2048-d embedding? Should I add the raw 527-class tag vector or
   MusicGenrenizer / genre-classifier probability vectors? Music emotion/mood taggers
   (Deezer mood detection, EmoMusic, PMEmo, music-emotion-recognition models).
7. **Audio fingerprinting / identity**: Chromaprint (AcoustID), dejavu, Panako — useful for
   deduplication, cover detection, and "same song" matching on the website. Worth it?
8. **Temporal structure**: section segmentation (chorus/verse boundaries), repetition count,
   structural complexity. Feasible without supervision? Any light models?
9. Anything creative I'm missing — e.g., noise level, sample rate artifacts, stereo width over time,
   loudness (LUFS/EBU R128), dynamic range, energy distribution across frequency bands,
   low-frequency "bass weight" features (relevant for danceability), tempo stability/beat
   consistency, quantization ("groove").

For Section 1, finish with a **ranked shortlist table**: top ~10 ideas by (value ÷ effort) for my
goals, each with feasibility on my hardware, and clearly mark the 2-3 "must-try" ones.

---

### SECTION 2 — Lyrics feature extraction ideas + benchmark question

1. **Direct answer first**: Is there a **standard benchmark** for lyric embeddings / lyric
   similarity / lyric-to-song retrieval? (I'm aware of "In-the-Song" ITS embeddings and Music4All —
   verify what the current de-facto evaluation is in 2026.) If a clear winner exists for "song
   lyrics similarity," name it and justify with benchmark numbers.
2. **Embedding families** (rank for: lyric similarity, mood, and multilingual robustness):
   - Sentence transformers: MPNet, MiniLM, BGE, GTE, E5, nomic-embed, ember, LaBSE, multilingual-E5,
     XLM-R-based embedders. Which are best for short/mid-length lyric texts? Multilingual matters!
   - Music-aware lyric models: SongBERT, SongTextBERT, ITS (In-the-Song), Lyrisong, "lyric
     embeddings trained on lyrics corpora" — verify each exists, dims, license, weights availability.
   - LLM-based (on my 16 GB RAM laptop or via free tier APIs): distill LLM embeddings, topic
     modeling with BERTopic, keyword/keyphrase extraction (KeyBERT, YAKE), LLM-generated tags
     (mood, theme, imagery) — local small LLM (e.g., 7-8B quantized) feasible? Cost/quality tradeoff.
   - Lexicon/psycholinguistic features: NRC EmoLex, VADER, ANEW, LIWC-style categories, arousal-
     valence-dominance values per line, swear word density, pronouns/temporal references.
   - Stylistic/poetic features: rhyme density (CMUdict/phonemizer), syllable counts, meter
     consistency, repetition/refrain structure, chorus detection from text, line length variance,
     vocabulary sophistication, concrete-vs-abstract word ratios, metaphor/imagery density.
3. **Handling multilingual + code-switched lyrics**: translate-then-embed vs multilingual embedders
   vs character n-grams — what actually works? Roughly 30-50% of my set is non-English.
4. **Structure-level features**: detect verse/chorus/repetition from plain text (no audio sync).
5. Finish with a **ranked shortlist**: top ~8 lyric approaches for (a) lyric-similarity product,
   (b) valence/mood prediction, (c) Kaggle dataset value — with dims, runtime on CPU, and license.

---

### SECTION 3 — Website tools brainstorm (beyond what I have)

My planned tools: overall similar songs, most similar lyrical songs, interactive song map.
Brainstorm 10-20 MORE compelling, feasible tools for a music-enthusiast site, e.g.:
- Playlist generator (seed song / feature sliders like valence-energy targets)
- "Why is this similar?" explainability (feature contributions, radar charts)
- Mood explorer (valence-valence / arousal map)
- Lyric quote search ("what song has this line?") with semantic search
- Cover/remix/sample detection via fingerprints
- Genre explorer & artist similarity
- Musical "radio" (walk on the embedding map)
- Comparison tool (side-by-side song DNA)
- Daily discovery / hidden gems below the chart
- Era/time-travel explorer (release_date axis)
- "Songs you'll like if you like X" with user taste profile
- Games (guess the song from lyrics/audio snippet)
- Karaoke/lyric display, translation tools
- Audio preview player (30 s) with feature annotations
- Song "DNA page" per track with links to 5 nearest neighbors per modality (audio/lyrics/metadata)

For each idea: name, one-line description, required data/features (which of my modalities),
feasibility, and expected user value. Rank the top 5 to build first. Also discuss:
- **Data compaction/scaling strategy**: quantization (int8), PCA/UMAP dimensionality reduction,
  FAISS/USearch/HNSW ANN indexes, precomputed sparse similarity graphs, approximate NN vs exact,
  memory estimates for 10k songs, how to scale to future 100k+ sets.
- **Recommended stack**: backend (FastAPI? Flask?), ANN engine (FAISS, USearch, Qdrant, HNSWLib),
  frontend (React/Next? Svelte? Gradio?), embedding map viz (UMAP + scatter-gl / regl / three.js),
  hosting for a hobby project, offline PWA feasibility.
- **Legal/licensing red flags** for publishing a Kaggle dataset containing Spotify metadata,
  ISRC, popularity, and Genius-style lyrics: what's allowed, what to strip, alternatives
  (e.g., MusicBrainz IDs instead of Spotify IDs? public-domain lyric sources?). Flag this clearly —
  I must not get my Kaggle account banned or face a DMCA.

---

### SECTION 4 — Hardware feasibility matrix (6 GB VRAM / 16 GB RAM)

For EVERY model/approach mentioned anywhere in your report, provide a compact row in one big table:
`Approach | Dims | VRAM est. | RAM est. | Time/song on 1660Ti | Total for 10k | License | Verdict`.
Mark anything that won't fit my hardware with ❌ and give the minimal workaround (CPU mode, chunking,
quantized weights, smaller variant).

---

### SECTION 5 — Feature-set comparison methodology

How do I actually TEST which feature approach is better, without labels I can fully trust?
- Task set: (a) predict valence/energy/danceability/popularity from features (R² on artist-aware
  splits), (b) similarity retrieval quality (proxy metrics: genre/mood cluster purity, tempo/valence
  neighborhood agreement, known-cover retrieval, human evaluation on 50-100 pairs),
  (c) dataset-quality heuristics (pairwise distance distributions, modality agreement).
- Same-model protocol: fixed CatBoost config + fixed simple MLP; artist-aware split; no leakage
  (no test re-use); ablation by modality group. What pitfalls should I avoid?
- Cost-aware ranking: expected value of running each extraction on my hardware vs expected gain.

---

### SECTION 6 — Final action plan

A concrete 3-phase plan:
1. **Phase A (this week, cheap)**: CPU/lightweight extras to run immediately (e.g., handcrafted
   DSP set, better pooling of existing embeddings, lexicon features).
2. **Phase B (GPU, heavier)**: the top 2-3 heavyweight audio models + top 2-3 lyric embedders.
3. **Phase C**: compare results per Section 5, then package the winners for Kaggle + website.

Include: rough timeline on my hardware, dependency install list (exact pip packages), and the
top 5 risks (OOM, license, multilingual quality, overfitting to genre confounds, Kaggle ToS).

---

## RULES FOR YOUR OUTPUT
1. One markdown report, all 6 sections, tables wherever possible.
2. Every model/benchmark/tool claim verified via web search with a URL.
3. Be brutally honest about feasibility on a 6 GB VRAM laptop — no "this might work on A100" ideas
   without a realistic fallback path.
4. Rank everything; no unranked laundry lists.
5. End with a 1-page executive summary: the 5 best audio ideas, the 3 best lyrics ideas, the
   answer to the benchmark question, and the top 5 website tools — in one table.
