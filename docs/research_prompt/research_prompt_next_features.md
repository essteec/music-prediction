# Research Brief Prompt: Expanding a 10k Popular-Songs Feature Dataset

Copy the prompt below verbatim into a research-capable LLM. Ask it to browse the web and cite primary sources.

---

You are a senior music-information-retrieval (MIR), NLP, and recommendation-systems researcher. Produce a deeply researched, practical feature-expansion plan for the music dataset described below. This is not a request for generic model lists: I need evidence-backed recommendations, implementation detail, benchmark evidence, cost/compute estimates, licensing cautions, and a staged decision framework.

## Research requirements

1. **Browse the web.** Prefer original papers, official model repositories/model cards, official benchmark pages, and authoritative library documentation. Use sources current through the day you answer.
2. Cite every non-obvious factual claim with a direct link. Clearly label inference/opinion versus reported benchmark evidence.
3. Do not assume cloud GPUs, APIs with paid usage, or a high-end workstation. Recommend CPU/GPU feasibility for my actual hardware before recommending a method.
4. Distinguish clearly between:
   - features useful for supervised prediction;
   - embeddings useful for song-to-song retrieval/similarity/search;
   - labels, tags, or externally obtained metadata; and
   - features that are legally safe to publish versus only safe to compute locally.
5. Avoid recommending more of the same. I already have four broad audio representations and general lyric sentence embeddings. Prioritize complementary signal and evaluate marginal value.
6. If a candidate is unsupported, unavailable, restricted, impractical on my hardware, or redundant with my current features, say so plainly and either reject it or put it in a long-term tier.

## Project context

I have a 10,000-song dataset of the most listened/popular Spotify songs as of July 2025 (with a small number of release dates just after July due to data collection). It is a convenience/popularity corpus, **not** a representative sample of all music. Current data is:

### Song-level table (`songs.csv`, 10,000 rows, 32 columns)

```text
rank, track_name, track_id, artist_names, artist_ids, album_name, album_id,
popularity, duration, explicit, release_date, album_type, isrc, copies,
danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms, time_signature,
total_artist_followers, avg_artist_popularity, artist_genres, main_genres, lyrics
```

Spotify-derived audio descriptors such as danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, and time signature are already available. Artist metadata and genre fields are also present.

Lyrics are non-empty for **9,797 / 10,000** songs; 203 are missing. Lyrics may be multilingual, noisy, incomplete, repeated, or copyright-restricted. I need explicit recommendations for data cleaning, language identification, missing-lyrics handling, and copyright-aware publishing.

### Audio corpus

- 10,000 locally stored full-song audio files, usually YouTube-source Opus/WebM.
- Example: Opus, 48 kHz, stereo, approximately 144 kb/s, around four minutes long.
- Files include basic title/artist/album/date/genre metadata.
- All 10,000 current downloads and all four current audio extractors completed successfully.
- Audio may not always be the authoritative studio recording; account for possible mismatches/live versions and recommend quality-control features/checks.

### Features already extracted

Audio embeddings are one vector per song, aligned by `track_id`:

| Feature family | Dimensions | Model/approach | Existing pooling/notes |
|---|---:|---|---|
| VGGish | 128 | VGGish | song-level vector |
| MERT v1 95M | 768 | `m-a-p/MERT-v1-95M` | 24 kHz mono; last hidden state temporal mean; first 30 seconds only |
| PANNs Cnn14 | 2,048 | AudioSet-pretrained PANNs | 32 kHz mono; song-level embedding |
| Mel spectrogram statistics | 512 | librosa | 128 mel bands × mean/std/max/min across time |

Existing lyric/text features from an older pipeline (so do **not** describe lyric feature extraction as starting from zero):

| Feature family | Dimensions | Details/limitations |
|---|---:|---|
| Basic lyric statistics | 5 | word count, unique-word count, unique ratio, average word length, character count |
| TextBlob sentiment | 2 | polarity and subjectivity; English-centric and weak baseline |
| MiniLM lyric embedding | 384 | `sentence-transformers/all-MiniLM-L6-v2`; lyrics cut to first 3,000 characters |
| MPNet lyric embedding | 768 | `sentence-transformers/all-mpnet-base-v2`; lyrics cut to first 3,000 characters; normalized |

The current 4 audio + MPNet + basic text/sentiment representation is 4,254 features in the thesis pipeline. Existing feature arrays are float32 NumPy `.npy` plus a separate ordered ID array.

### Hardware and operating constraints

- Consumer laptop: NVIDIA GTX 1660 Ti with **6 GB VRAM**, **16 GB RAM**.
- Extraction must work locally; batch size 1 or small batches and chunked audio are acceptable.
- Storage, runtime, and resilience matter. I want resumable processing, song-ID alignment, validation for NaN/Inf/shape errors, checksums/manifests, and output that can be released on Kaggle.
- Give approximate runtime and disk usage for 10,000 songs wherever possible. State assumptions and uncertainty.
- It is acceptable to propose a tiny pilot (e.g., 100–500 songs) before full extraction. Do not advise a full expensive run without a pilot and a way to judge value.

## Future product goals

1. Compare feature methods to find what best represents songs for multiple tasks.
2. Publish a well-structured, easy-to-use Kaggle dataset containing metadata and extracted features, while respecting licenses/copyright and provenance.
3. Eventually build a music-enthusiast website with:
   - overall similar songs;
   - lyrically similar songs;
   - a song map based on similarity;
   - additional genuinely useful music-discovery/analysis tools.
4. The web product is a later phase. Favor compact, scalable, interpretable representations and retrieval-ready indexes, but do not over-engineer a production system now.

## Questions to answer

### A. Audio: candidate feature families

Research and compare complementary audio approaches. Cover at least the following categories, but add better categories if justified:

1. **Music-specific foundation-model embeddings** beyond current MERT v1 95M (including newer MERT variants if relevant, MusicFM, CLAP/LAION-CLAP music suitability, Jukebox-derived representations, MuLan, or other current, reproducible music/audio models).
2. **Self-supervised general-audio/speech models** where they truly add information for music (e.g., BEATs, wav2vec 2.0, HuBERT, AudioMAE); explain likely redundancy versus PANNs/VGGish/MERT and whether they are worth testing.
3. **Structured MIR descriptors**, including rhythm/beat/tempo stability, onset density, tempo curve, meter/downbeat confidence, chroma/HPCP/key/mode/key-strength, chord/key progression, tonnetz, harmonic change detection, MFCC/spectral descriptors, timbral texture, dynamic range, loudness, silence/intro/outro, segment/section structure, repetition, and novelty curves.
4. **Source separation / vocal-instrumental features** (e.g., vocals, drums, bass, other): feasibility on 6 GB VRAM; features from stems versus merely storing stems; whether separation errors undermine the value.
5. **Singing voice and vocal-production features**: vocal presence, singing/speech/rap probability, pitch/melody contour, vibrato, vocal range, voice activity, vocal/instrumental ratio, and whether robust open models are practical for mixed mastered tracks.
6. **Audio tags/classifiers**: genre, mood, instrument, vocal, scene, acoustic/electronic, production/style tags. Identify open models and trustworthy datasets, but warn against treating predicted tags as ground truth.
7. **Temporal representations and pooling**: how to improve on one global mean vector. Compare multi-window sampling (intro/middle/outro), mean+std/max, attention/learned pooling, VLAD/Fisher-like aggregation, segment embeddings, and fixed-length sequence storage. Explain which formats are best for training versus similarity search.
8. **Cross-modal music–text representations**: embeddings designed to align audio with natural-language music descriptions; assess whether they help metadata/tag queries and song similarity.
9. **Quality-control / duplicate / mismatch features**: fingerprinting, audio duration matching, cover/live/remaster detection ideas, near-duplicate clusters, clipping/silence/loudness anomalies.

For each candidate method, provide:

- exact model/library/version and primary source;
- task/training data and what musical properties it is likely to encode;
- input preprocessing, expected output dimensions, and sensible pooling;
- expected complementarity or redundancy with **VGGish, MERT v1 95M, PANNs Cnn14, and mel statistics**;
- feasibility on GTX 1660 Ti 6 GB / 16 GB RAM, including a conservative batch size, CPU fallback, likely bottleneck, and estimated 10k runtime;
- licensing/weight license/dataset provenance and whether vectors can likely be distributed on Kaggle (flag uncertainty; this is not legal advice);
- best use: predictive modeling, audio similarity, user-facing explanation, quality control, or not recommended;
- a pilot protocol and success criterion.

### B. Lyrics: extraction, representations, and benchmarks

I especially need a rigorous answer here. Research the best practical way(s) to represent lyrics for both prediction and **lyrically similar song retrieval**.

1. **Benchmark landscape:** Is there a broadly accepted benchmark that identifies the “best” lyric representation/model directly? If not, explain why not (task, language, copyright, data-access, and evaluation differences). Identify the most relevant existing lyric/MIR benchmarks and datasets, their tasks, languages, sizes, availability/licenses, and whether they are actually usable for model selection. Cover lyric similarity/semantic relatedness, emotion, theme/topic, genre, language, and any multimodal audio–lyrics benchmarks that are relevant. Verify names and current availability instead of inventing a benchmark.
2. **Recommended evaluation plan instead of blindly choosing a benchmark:** propose a small, legally safe in-domain evaluation set for this corpus. Include at least:
   - human-judged lyrical similarity (how many query songs, candidate-pair sampling, annotator instructions, 3–5 rating dimensions, agreement metric);
   - quantitative retrieval metrics (e.g., Recall@k, nDCG@k, MAP/MRR when labels exist);
   - task-based comparisons for lyric emotion/theme/language and downstream prediction;
   - leakage prevention, multilingual stratification, artist-duplicate controls, and baselines;
   - how to combine automatic proxy labels with manual evaluation without overstating results.
3. **Text preparation:** lyric-source and licensing caveats; boilerplate removal; `[Verse]`/`[Chorus]` section parsing; repeated-chorus handling; Unicode normalization; contractions/slang/profanity; language ID; code-switching; translation versus multilingual encoders; missing/instrumental handling; truncation/chunking strategies for long lyrics. Discuss whether using only the first 3,000 characters creates bias.
4. **Dense lyric embeddings:** compare strong current open sentence/document embedding models suitable for multilingual and English lyrics. Include general retrieval embeddings (e.g., E5/BGE/GTE/Nomic/Jina-class models as appropriate), multilingual models, and any lyric/music-domain-adapted encoders with reproducible weights. Recommend compact candidates that fit the laptop, and clarify whether mean-pooling full lyrics, chunk embeddings + late pooling, section-aware pooling, or title+lyrics+metadata text works best for each objective.
5. **Fine-tuning/adaptation:** assess realistic parameter-efficient options (contrastive fine-tuning, LoRA/adapters, weak supervision) versus zero-shot embeddings. Include what positive/negative pairs can be constructed without leaking artist identity and which approach is credible with 10k songs.
6. **Interpretable lyric features:** propose robust features grouped by category, such as:
   - lexical richness/readability/word-frequency and character-level style;
   - rhyme density and rhyme scheme quality (with multilingual limitations);
   - repetition, chorus/verse structure, hookiness proxies, line-length/rhythm proxies;
   - POS/dependency/style, pronouns, tense, named entities, concreteness/imagery;
   - topics/themes, narrative point of view, semantic fields;
   - emotion dimensions/categories, sentiment trajectory, subjectivity, toxicity/profanity;
   - language, code-switching, and translation confidence;
   - cultural/genre/style indicators, with bias cautions.
   For each, give tools/models, output schema, language limitations, expected value, and cost.
7. **Generative LLM extraction:** assess whether structured LLM-derived lyric annotations (theme, mood, narrative, setting, point of view, chorus summary) are worthwhile. Address reproducibility, hallucination, pricing/local feasibility, prompt versioning, output validation, copyright restrictions, and whether generated labels should be released.
8. **Lyrics-to-audio relationships:** propose features measuring agreement/disagreement between lyric mood and audio mood/energy, alignment of text/audio embeddings, and cross-modal retrieval. Clearly distinguish promising research ideas from techniques practical for this project.

For every lyric candidate, include multilingual validity, hardware feasibility, licenses, output size, and use for (a) prediction, (b) lyrical retrieval, and (c) publishable Kaggle artifact.

### C. Metadata, graph, and derived features

Identify useful features that can be derived from the current metadata or safely added without unnecessarily depending on proprietary APIs. Consider release-era, artist/album collaboration structure, genre hierarchy, artist popularity/followers transformations, novelty relative to an artist/genre, rank/popularity caveats, explicitness, duration, temporal/country availability where legally obtainable, and audio/lyric/metadata disagreement features. Flag target leakage: popularity is an existing column and must not be used as an input when popularity is the predicted or evaluated outcome.

### D. Similarity and future website ideas

Propose useful tools beyond the three already planned. For each, specify the user need, minimum viable inputs, recommended representation(s), explainability UI, retrieval/indexing strategy, and whether it is feasible at 10k now and at 100k/1M later.

Include ideas such as (only if justified):

- controllable “find songs like X, but calmer/more lyrical/more danceable” search;
- separate audio, lyrical, mood, rhythm, harmony, vocal-style, and cultural/context similarity sliders;
- “why these songs?” explanations based on non-sensitive features;
- audio–lyric mood mismatch explorer;
- song evolution/era/genre maps;
- lyric-theme map and lyric motif discovery;
- near-duplicate/cover/remaster explorer;
- playlist coherence/diversity diagnostics;
- interactive A/B similarity evaluation and feedback collection;
- privacy, licensing, attribution, and misuse considerations.

Describe a storage/index strategy for static Kaggle data and a later website: row-level parquet/CSV metadata, feature matrix format, a manifest/data dictionary, vector normalization, PCA/quantization options, approximate nearest neighbor indexes (e.g., FAISS/hnswlib/ScaNN only where appropriate), and UMAP map creation. Explain why a 2-D UMAP map is for exploration and should not be treated as the similarity engine itself.

### E. Experiment and decision plan

Create a staged plan that protects limited compute:

1. **Tier 0:** data audit/quality checks and one or two nearly-free features.
2. **Tier 1:** the highest-value, low-risk additions.
3. **Tier 2:** pilot only, then scale if metrics justify it.
4. **Tier 3:** future/too costly/requires a stronger GPU or permission.

For each tier, provide a table with priority, candidate, new information it adds, expected compute/storage, implementation difficulty, license risk, and go/no-go criterion.

Design a fair ablation protocol. It should compare additions against these exact baselines:

- Existing four audio feature families;
- existing MPNet (768-D), MiniLM (384-D), 5 lyric statistics, and 2 TextBlob features;
- Spotify metadata/audio descriptors;
- retrieval quality for audio similarity and lyric similarity, not just supervised R².

Include duplicate/artist-aware splits, feature alignment checks, temporal pooling ablations, dimensionality-reduction rules learned only on training data, statistical confidence intervals, and a way to record all model versions, preprocessing, timing, and failure rates.

## Required output format

Produce the response in this exact order:

1. **Executive recommendation** — no more than 15 bullet points; name the 3–6 things I should test first and the 3–6 things I should avoid/defer.
2. **Current-state audit** — restate the known assets and gaps accurately; call out that basic lyric features and MiniLM/MPNet already exist.
3. **Candidate matrix: audio** — ranked table with all fields requested above.
4. **Candidate matrix: lyrics** — ranked table with all fields requested above.
5. **Lyrics benchmark answer** — direct answer to “Is there a benchmark that lets me choose the best model directly?” followed by benchmark/dataset table and an in-domain evaluation design.
6. **Website/product concepts** — ranked list, with data/representation/index needs.
7. **Kaggle-ready data architecture** — directory layout, file formats, metadata schema, manifests, naming/versioning, documentation, and what not to distribute due to copyright/license concerns.
8. **Staged 10k-song roadmap** — Tier 0–3 plan with pilot sizes, commands/pseudocode where helpful, expected runtime/storage, and decision gates.
9. **Ablation/evaluation protocol** — prediction and retrieval, with metrics and split rules.
10. **Risk register** — at minimum: copyright/lyrics, model/weight licenses, YouTube-source audio provenance, multilingual bias, target leakage, artist/duplicate leakage, extraction failures, embedding version drift, hardware OOM, and storage growth.
11. **Annotated bibliography** — direct links to all primary papers, repositories, benchmarks, model cards, and tools used. Include publication/version date and license where available.

Be decisive. If the evidence does not support a candidate’s incremental value over MERT + PANNs + VGGish + mel statistics and MPNet/MiniLM, say “do not prioritize.” If exact runtime or licensing is unknown, state that it is unknown rather than guessing.

---

## Local repository facts used in this prompt

- `data/processed/songs.csv`: 10,000 rows × 32 columns; 9,797 non-empty lyric rows.
- `data/logs/download_log_pilot.csv`: 10,000 successful, unique audio downloads.
- Existing feature arrays: VGGish `(10000, 128)`, MERT `(10000, 768)`, PANNs `(10000, 2048)`, mel statistics `(10000, 512)`, all float32 and ID-aligned.
- The historical codebase includes basic lyric statistics, TextBlob sentiment, MiniLM, and MPNet preprocessing. The prompt frames them as existing baselines so researchers do not repeat them as novel work.
