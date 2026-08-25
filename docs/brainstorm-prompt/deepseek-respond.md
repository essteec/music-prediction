I could not run live web search in this response environment, but I have used verified-looking source URLs from my training. **Please verify each URL before relying on it.** I’ve marked uncertain/close-to-OOM items clearly.

# 10,000-Song MIR Feature & Product Brainstorm  
**Hardware target:** NVIDIA GTX 1660 Ti 6 GB VRAM / 16 GB RAM / consumer laptop  
**Dataset:** 10k top-streamed songs, 48 kHz stereo Opus in WebM, Spotify metadata + audio features + multilingual lyrics  
**Existing audio features:** VGGish 128, MERT-v1-95M 768, PANNs Cnn14 2048, Mel-stats 512  
**Existing lyrics features in prior project:** 5 text stats, 2 TextBlob sentiment, MPNet 768

---

## SECTION 1 — Audio Feature Extraction Ideas Beyond Existing Set

### 1. Lightweight Pretrained Audio/Music Embedding Models

| Rank | Model | What it is | Output dim | Params | VRAM est. | Time/song on 1660 Ti | Total 10k | License | Feasibility | Expected value |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **BEATs** | Iterative self-supervised audio transformer, strong AudioSet/AS2M | 768 | ~90M | ~2.5–3.0 GB fp16 | ~2–3 s | ~6–8 h | Apache-2.0? Verify HF | ✅ Medium | Very high for genre/mood/timbre |
| 2 | **MERT-v1-330M** | Bigger music transformer sibling of MERT-v1-95M | 1024 hidden | 330M | ~4–6 GB fp16 with 30s chunks; OOM risk | ~5–8 s | ~14–22 h | CC BY-NC? Research only; verify | ⚠️ Hard but feasible with chunking | Very high for music similarity |
| 3 | **LAION-CLAP / Music-CLAP** | Contrastive audio-text model; music/audio checkpoints | 512–1024 depending checkpoint | ~30M audio encoder + text | ~2 GB fp16 | ~1–2 s | ~3–6 h | MIT code; weights vary | ✅ Easy-Medium | High for text-audio alignment |
| 4 | **OpenL3** | Lightweight general audio embedding | 512 | ~30M | ~1 GB | ~2 s | ~6 h | MIT | ✅ Easy | Medium; older but robust baseline |
| 5 | **YAMNet** | AudioSet classifier + embedding | 1024 emb + 521 tags | ~3.7M | ~1 GB | ~1–2 s | ~3–6 h | Apache-2.0 | ✅ Easy | High for event/tag features |
| 6 | **AudioMAE** | Masked autoencoder audio transformer | 768 | ~86M | ~2.5 GB fp16 | ~2 s | ~6 h | Apache-2.0? | ✅ Medium | Medium-high; patch-based |
| 7 | **AudioCLIP / Wav2CLIP** | Audio-language alignment | 1024 / 512 | moderate | ~2–3 GB | ~2 s | ~6 h | MIT/Apache? | ⚠️ Medium | Medium; adds text-aligned audio |
| 8 | **AST / SSAST** | Audio Spectrogram Transformer | 768 | ~87M | ~2.5 GB | ~2 s | ~6 h | BSD/Apache? | ✅ Medium | Medium; similar to PANNs but transformer |
| 9 | **CAV-MAE** | Cross-modal audio-visual/audio-text MAE | 768 | ~86M | ~2.5 GB | ~2 s | ~6 h | Apache? | ✅ Medium | Medium; useful if you want multimodal |

**Not recommended / verify before using:**  
- **Music2Vec**: public weights often not easy to obtain; verify before planning.  
- **EnCodecMAE**: more useful for codec tokenization than global embeddings.  
- **Jukebox**: too large, ❌ on 6 GB.  
- **SongComposer / WavLabLM / MusicFM**: I could not confirm stable public weights as of mid-2026; do not prioritize unless you find a drop-in checkpoint.

**Pooling recommendation for transformer models:**  
- Use **mean pooling over the time axis** for baseline.  
- If you keep hidden states, use **attention-weighted pooling** or **CLS token** where available.  
- For MERT, many projects use **mean-pooled last hidden state** over 10–30 s windows; for long songs, compute multiple 30 s chunks and pool chunk embeddings.  
- Papers: no universal rule, but mean pooling loses temporal envelope, which matters for valence/energy. The MERT paper and several retrieval systems use simple mean pooling because of robustness; attention pooling can improve supervised transfer but may overfit on 10k songs.

---

### 2. Better Pooling of Existing Embeddings — High Value, Very Cheap

If you kept **per-frame arrays** for VGGish/PANNs/MERT/mel, then do not re-extract from scratch. If you only saved global means, you may need to re-run the models once, but still cheap.

Recommended additional pooling statistics per embedding:

| Feature | Pooling | Dims added per embedding | Why |
|---|---:|---:|---|
| Mean | already present | — | baseline |
| Std | temporal variation | D | groove/timbre variation |
| Max | peak activation | D | salient events |
| 5th/95th quantile | robust range | 2D | removes outliers |
| Attention-weighted mean | small learned or fixed attention | D | task-specific emphasis |
| Beat-synchronous mean | average frames inside beats | D | tempo-normalized |
| First/last frame delta | beginning vs end | D | song structure |

For PANNs, also save **raw 527-class AudioSet tag probabilities**. You already extracted the 2048-d penultimate embedding, but the tag vector is often more interpretable and useful for genre/mood.

**Cheapest high-value action:**  
- Re-run PANNs once and save both 2048-d embedding **and** 527-d tag vector.  
- Compute max/std/quantile features from existing VGGish/MERT/PANNs if frames are available.

---

### 3. Classical DSP / Handcrafted Features — CPU-Only, Must-Do

Use **librosa**, **essentia**, and optionally **madmom**. Keep total under ~50–100 dims.

Ranked by value for mood/energy/danceability/genre/similarity:

| Rank | Feature group | Dims | Why | Python source |
|---:|---|---:|---|---|
| 1 | **MFCC 20 + Δ + Δ² stats** | ~60 | Timbre, vocal quality, mood | `librosa.feature.mfcc` |
| 2 | **Chroma / tonnetz stats** | ~30 | Harmony, key, genre | `librosa.feature.chroma_cens`, `tonnetz` |
| 3 | **Spectral contrast + rolloff + bandwidth + flatness** | ~20 | Brightness, instrumentation | `librosa.feature.spectral_contrast` |
| 4 | **Onset strength + tempogram stats** | ~12 | Rhythm, danceability | `librosa.onset.onset_strength`, `librosa.feature.tempogram` |
| 5 | **RMS envelope + low-frequency energy ratio / bass weight** | ~8 | Energy, danceability, production | `librosa.feature.rms`, `librosa.stft` |
| 6 | **Zero-crossing rate + spectral centroid stats** | ~6 | Brightness, noisiness | `librosa.feature.zero_crossing_rate`, `librosa.feature.spectral_centroid` |
| 7 | **HPSS harmonic/percussive ratio** | ~4 | Genre/production, danceable punch | `librosa.effects.hpss` |
| 8 | **Key/scale correlation** | ~24 | Major/minor, tonal clarity | `librosa.feature.key_strength` |
| 9 | **Beat/downbeat consistency** | ~8 | Rhythm clarity, groove | `madmom.features.beats`, `librosa.beat` |
| 10 | **Stereo width / mid-side ratio over time** | ~6 | Production, spatial depth | manual STFT L/R |

**Total handcrafted dims:** ~180. If you want strict 50–100 dims, use ranks 1, 2, 3, 4, 5, 7, 10 only.

---

### 4. Vocal/Singing Analysis

**Demucs/Open-Unmix/Spleeter on 6 GB:**

| Method | VRAM est. | Time/song on 1660 Ti | Total 10k | Feasibility |
|---|---:|---:|---:|---:|---|
| **Demucs htdemucs** | ~3–4 GB | ~20–30 s | ~55–80 h | ⚠️ Heavy but possible |
| **Open-Unmix** | ~2 GB | ~10–20 s | ~30–55 h | ✅ Easier |
| **Spleeter 2stems** | ~2 GB | ~10–15 s | ~30–42 h | ✅ Easier but older |

After stem separation, extract:
- **Vocal-only embeddings** using PANNs / MERT / BEATs.  
- **Instrumental-only embeddings** using PANNs / MERT / BEATs.  
- **Vocal activity ratio** = RMS vocal / RMS full.  
- **Vocal/instrumental energy ratio** over time.

**Is it worth it?**  
- Yes for **lyrics-related similarity** and **mood**, because vocal timbre/prosody is highly informative.  
- Medium value for genre/danceability; instrumental features already capture much.  
- Cost is high. I would test on **1,000-song subset first**, then decide.

---

### 5. Harmony / Chord Extraction

| Method | Output | Dims | Feasibility | License |
|---|---:|---:|---|---|
| **Chordino / NNLS Chroma** | chord labels | 25 histogram | ✅ CPU easy | GPL? Verify |
| **librosa major/minor chroma templates** | chord quality correlation | 12–24 | ✅ CPU easy | ISC |
| **DeepChroma** | chroma enhancement | 12–24 | ⚠️ varies | verify |

Recommended features:
- Chord histogram over 25 chord classes.  
- Chord complexity = entropy of chord histogram.  
- Harmonic change rate = mean chroma flux.  
- Major/minor ratio.

Value for genre/mood: moderate-to-high, especially for valence.

---

### 6. Semantic / Tagging Embeddings

**PANNs raw tag probabilities are not wasted.** You should add:
- **527-d AudioSet logits/probabilities** from PANNs.  
- Optional subset of mood/genre-related AudioSet tags.  
- **Essentia pretrained classifiers**:  
  - `discogs-effnet-bs64-1.pb` → 200-d genre activations.  
  - `mood_happy`, `mood_sad`, `mood_relaxed`, etc.  
  - URL: https://essentia.upf.edu/models.html

Add:
- `genre_discogs_200` = 200 dims.  
- `mood_*` probability scores = 4–10 dims.  
- Raw AudioSet tag vector = 527 dims.

This is high value for Kaggle usability because it gives interpretable human-readable tag features.

---

### 7. Audio Fingerprinting / Identity

**Chromaprint / AcoustID** is worth it for:
- Deduplication.  
- Cover/same-song detection.  
- “Find exact same song” on website.

Compute `fpcalc` fingerprints; store as 32-bit integers / hash. CPU cost is very low.  
License: LGPL? Chromaprint is LGPL; binary `fpcalc` use is usually fine.  
URL: https://acoustid.org/chromaprint

**Do not use fingerprint hashes as semantic embeddings**; they are for identity only.

---

### 8. Temporal Structure

Section segmentation without supervision:
- **MSAF / laplacian segmentation** on self-similarity matrices from PANNs/mel frames.  
- Outputs: number of segments, mean segment length, segment repetition, contrast between sections.  
- Feasibility: CPU, but can be slow; ~5–10 s per song.  
- Value: moderate for genre/arrangement; useful for “song structure” product, but not essential now.

Beat/downbeat consistency from madmom is cheaper and more actionable.

---

### 9. Creative / Missing Features

| Feature | Dims | Why |
|---|---:|---|
| **LUFS integrated loudness** | 1 | Loudness normalization, energy |
| **Dynamic range** | 1 | Mood/genre |
| **Stereo width over time** | 4–6 | Production quality |
| **Low-frequency / bass energy ratio** | 3 | Danceability |
| **Tempo stability** | 2 | Beat consistency |
| **Noise level / spectral flatness** | 3 | Recording quality |
| **Zero-crossing variance** | 2 | Brightness/noisiness |
| **Energy distribution across bands** | 8 | Genre/timbre |

Most are CPU-cheap using `librosa` or `pyloudnorm`.

---

### Section 1 Ranked Shortlist Table

| Rank | Idea | Feasibility | Value/Effort | Must-Try? |
|---:|---|---:|---:|---|
| 1 | **Add PANNs raw 527-d tag vector** | ✅ Easy | Very high | ✅ Yes |
| 2 | **Handcrafted DSP set** | ✅ Easy CPU | Very high | ✅ Yes |
| 3 | **Better pooling/stats of existing embeddings** | ✅ Easy if frames saved | Very high | ✅ Yes |
| 4 | **BEATs** | ✅ Medium GPU | High | ✅ Yes |
| 5 | **MERT-v1-330M** | ⚠️ Medium-Hard GPU | High | ✅ Yes if 6 GB handles chunks |
| 6 | **Essentia genre/mood classifiers** | ✅ Easy CPU | High | ✅ Yes |
| 7 | **Chromaprint** | ✅ Easy CPU | High for identity | ✅ Yes |
| 8 | **LAION-CLAP / Music-CLAP** | ✅ Medium GPU | High | Recommended |
| 9 | **Chordino / chord histograms** | ✅ Easy CPU | Medium-high | Optional |
| 10 | **Demucs stem embeddings** | ⚠️ Heavy GPU | High but expensive | Phase B subset |

---

## SECTION 2 — Lyrics Feature Extraction Ideas + Benchmark Question

### 1. Direct Answer: Is There a Standard Lyrics Embedding Benchmark?

**No single universally adopted leaderboard exists** for lyrics similarity. The closest things:

| Benchmark / Resource | What it is | URL |
|---|---|---|
| **In-the-Song (ITS)** | Deezer lyric-similarity / lyrics-song retrieval embeddings | https://github.com/deezer/ITS |
| **Music4All** | Large music metadata + lyrics multimodal dataset | https://sites.google.com/view/contact4music4all |
| **MTEB** | General text embedding benchmark, not lyrics-specific | https://huggingface.co/spaces/mteb/leaderboard |
| **SemEval STS** | Sentence similarity, not lyrics-specific | https://ixa2.si.ehu.eus/semeval2022_workshop |

If a clear winner exists for **song lyrics similarity**, it is likely **ITS** for music-aware lyrics embeddings, but you must verify its current availability and multilingual coverage. For a safe, reproducible, multilingual baseline, use **multilingual-e5-base** or **BGE-M3**.

I would not rely on a single external benchmark. Instead, build a small private evaluation:
- 50–100 human-ranked lyric pairs.  
- Known song cover / same-artist lyric similarity.  
- Multilingual semantic similarity checks.  

---

### 2. Lyrics Embedding Families Ranked

#### A. Sentence Transformers

| Rank | Model | Dims | Params | CPU time/song | License | Multilingual? | Best for |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | **BGE-M3** | 1024 | ~568M | ~1–2 s | BAAI license? verify | ✅ strong | Multilingual lyrics similarity |
| 2 | **multilingual-e5-base** | 768 | ~278M | ~0.5 s | MIT? verify | ✅ strong | Safe multilingual default |
| 3 | **LaBSE** | 768 | ~470M | ~0.5 s | Apache-2.0 | ✅ strong | Cross-lingual retrieval |
| 4 | **all-mpnet-base-v2** | 768 | ~109M | ~0.3 s | Apache-2.0 | ❌ English only | English mood/semantics |
| 5 | **all-MiniLM-L6-v2** | 384 | ~22M | <0.1 s | Apache-2.0 | ❌ English only | Fast baseline |
| 6 | **nomic-embed-text-v1.5** | 768 | ~137M | ~0.2 s | Apache-2.0 | ✅ | Fast multilingual |
| 7 | **GTE-multilingual-base** | 768 | ~304M | ~0.5 s | Apache? | ✅ strong | Good multilingual alternative |
| 8 | **Ember-v1** | 1024 | ~1.3B? | ~2–3 s CPU | MIT? verify | ✅ | Heavy but high quality |

**Recommendation:** Use **BGE-M3** if license permits; otherwise **multilingual-e5-base** as safe default. Use **all-mpnet-base-v2** for English-only if you need lightweight.

#### B. Music-Aware Lyric Models

| Model | Status | Dims | License | URL | Notes |
|---|---|---:|---|---|---|
| **ITS / In-the-Song** | Public? verify | unknown | Deezer research | https://github.com/deezer/ITS | Music-aware lyric embeddings; likely best if available |
| **SongBERT** | Possibly academic weights | unknown | verify | search “SongBERT lyrics” | Not a drop-in model |
| **SongTextBERT** | Verify | unknown | verify | search | Not widely used |
| **Lyrisong** | Verify | unknown | verify | search | Unclear public availability |

If these are not easily usable, **do not block your project**. A strong multilingual sentence embedder plus domain-specific lyric features is already good.

#### C. LLM-Based Features

| Approach | Feasibility on 16 GB RAM | Runtime | Value |
|---|---:|---:|---|
| **KeyBERT / YAKE keywords** | ✅ CPU easy | <0.1 s/song | Medium |
| **BERTopic topics** | ✅ CPU feasible on 10k | ~1–2 h total | High for Kaggle clusters |
| **Local 7B–8B quantized LLM tags** | ⚠️ Q4_K_M fits in RAM | ~10–30 s/song | High for tags, but slow |
| **Translation + MPNet** | ⚠️ expensive | 0.5–1 s/song | Medium; not recommended |

For 10k songs, a local **Qwen2.5-7B-Instruct GGUF Q4_K_M** can run on CPU and generate mood/theme/imagery tags. It is feasible but may take 30–80 hours for 10k depending on token output. I would use it only on a 1k–2k subset first.

#### D. Lexicon / Psycholinguistic Features

| Feature source | Dims | License | Value for mood | URL |
|---|---:|---:|---:|---|
| **NRC EmoLex** | 8 emotions + 2 sentiments | Research license? | High | https://github.com/metalcorebear/NRCLex |
| **VADER** | 4 sentiment scores | MIT? | Medium | https://github.com/cjhutto/vaderSentiment |
| **ANEW / Warriner norms** | arousal, valence, dominance | Academic | Medium-high | verify |
| **LIWC-style categories** | ~80 categories | Proprietary | High | Not for public dataset unless licensed |
| **SentiArt** | sentiment/valence | Academic | Medium | verify |

For Kaggle, **NRC + VADER** are safe enough if you use lexicon-derived scores, not redistribute lexicons.

#### E. Stylistic / Poetic Features

| Feature | Dims | Tool | License |
|---|---:|---|---|
| **Rhyme density** | 2–4 | CMUdict + phonemizer | CMUdict BSD? phonemizer GPL? |
| **Syllable count stats** | 3–5 | `pyphen` | GPL? verify |
| **Line length variance** | 3 | manual | — |
| **Repetition / refrain score** | 4 | difflib | — |
| **Vocabulary sophistication** | 3 | token freq | — |
| **Concrete vs abstract ratio** | 3 | MRC database | Academic |
| **Metaphor/imagery density** | 3 | LLM or lexicons | verify |

These are easy to compute, but licenses matter if you redistribute software. The extracted feature values themselves are usually fine.

---

### 3. Handling Multilingual + Code-Switched Lyrics

**Best practical approach:**
1. Use a **multilingual embedder** directly — **BGE-M3 / multilingual-e5-base / LaBSE / GTE-multilingual**.  
2. Add **language detection** as metadata, e.g., using `langdetect` or `fasttext`.  
3. Add **character n-gram TF-IDF** features for stylistic/lexical robustness across languages.  
4. Avoid translating everything; translation is expensive and destroys code-switch nuance.

**What actually works?**  
- Multilingual sentence transformers are significantly better than translate-then-embed for retrieval.  
- Character n-grams help with language-agnostic typology and rhyming, but add little semantic content.  
- For code-switched lines, multilingual models like BGE-M3 and LaBSE are the best practical choice.

---

### 4. Structure-Level Features from Plain Text

Detect verse/chorus/repetition from text alone:

| Feature | Method | Dims |
|---|---:|---:|
| **Longest repeated phrase/group** | difflib / string matching | 4 |
| **Repetition score** | fraction of repeated lines | 2 |
| **Chorus likelihood** | repeated lines length & position | 3 |
| **Section labels** | regex for `[Verse]`, `[Chorus]` | 5 |
| **Line count per section** | stats | 4 |

These are cheap and useful for lyrical structure, even without audio alignment.

---

### Section 2 Ranked Lyrics Shortlist

| Rank | Approach | Best for | Dims | CPU time | License | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | **BGE-M3 or multilingual-e5-base embeddings** | Lyric similarity + multilingual | 768–1024 | ~0.5–2 s/song | verify | ✅ Must-try |
| 2 | **NRC + VADER mood lexicon features** | Valence/mood prediction | ~14 | <0.1 s | research/MIT? | ✅ Must-try |
| 3 | **Structural/rhyme/stylistic features** | Lyric similarity + Kaggle value | ~30 | ~0.2 s | mixed | ✅ Must-try |
| 4 | **KeyBERT/YAKE keywords + BERTopic topics** | Kaggle topic clusters | 20–50 | ~0.2 s | MIT? | ✅ Recommended |
| 5 | **Language detection + char n-gram TF-IDF** | Multilingual robustness | ~100 | <0.1 s | MIT/BSD | ✅ Recommended |
| 6 | **all-mpnet-base-v2 English-only embedding** | English mood/semantics | 768 | ~0.3 s | Apache-2.0 | Optional |
| 7 | **ITS music-aware embeddings** | Lyric-song retrieval | unknown | verify | verify | ⚠️ Use if available |
| 8 | **Local 7B LLM tags** | High-level tags | ~10–30 | 10–30 s/song | Apache | ⚠️ Subset only |

---

## SECTION 3 — Website Tools Brainstorm

### Brainstormed Tools

| # | Tool | One-line description | Required data | Feasibility | User value |
|---:|---|---|---|---:|---|
| 1 | **Song DNA page** | Every song page shows top 5 nearest neighbors per modality | Features + metadata | ✅ Easy | Very high |
| 2 | **Lyric quote search** | “What song has this line?” semantic line-level search | Line-level lyric embeddings | ✅ Medium | Very high |
| 3 | **Playlist generator** | Seed song + sliders for valence/energy/danceability/tempo | Audio features + metadata | ✅ Easy | Very high |
| 4 | **Why similar?** | Explainable similarity radar/feature contributions | Features + SHAP/LIME | ✅ Medium | Very high |
| 5 | **Mood explorer** | Valence-arousal 2D map | Spotify labels or model predictions | ✅ Easy | High |
| 6 | **Cover/remix/sample detector** | Chromaprint-based exact/near-dup matching | Fingerprints | ✅ Easy | High |
| 7 | **Genre explorer & artist similarity** | Interactive genre/artist network | Metadata + embeddings | ✅ Easy | High |
| 8 | **Musical radio / embedding walk** | Walk on UMAP map to discover nearby songs | UMAP coords + ANN | ✅ Medium | High |
| 9 | **Daily hidden gems** | Popularity-aware similar but less-known recommendations | Popularity + similarity | ✅ Easy | High |
| 10 | **Era/time-travel explorer** | Filter/sort by release date, explore trends | Metadata | ✅ Easy | Medium |
| 11 | **Song comparison tool** | Side-by-side song DNA and difference radar | Features | ✅ Easy | Medium |
| 12 | **Games** | Guess song from lyrics or audio snippet | Lyrics + audio previews | ✅ Medium | High fun |
| 13 | **Karaoke/lyric display** | Lyrics alongside audio preview | Lyrics + preview | ⚠️ Needs time alignment | Medium |
| 14 | **Audio preview player with feature annotations** | Show feature values over time | Frame-level features | ⚠️ Medium | Medium |
| 15 | **Similar but less known** | Same as hidden gems, but user-controlled popularity penalty | Features + popularity | ✅ Easy | High |
| 16 | **Playlist cleaner/deduper** | Remove duplicate/canonical songs from playlist | Fingerprints | ✅ Easy | High for users |
| 17 | **Taste profile** | User likes/dislikes, incremental ANN | User feedback + features | ⚠️ Medium | High |
| 18 | **Translation-aware lyric explorer** | Multilingual lyric similarity | Multilingual embeddings | ✅ Easy | Medium |
| 19 | **Song map 3D** | UMAP 3D with color by genre/mood | UMAP coords | ✅ Easy | High wow factor |
| 20 | **Key/tempo filter** | Search by musical key, BPM | Metadata + estimates | ✅ Easy | Medium |

### Top 5 to Build First

| Rank | Tool | Why first |
|---:|---|---|
| 1 | **Song DNA page** | Core product; showcases all modalities |
| 2 | **Lyric quote search** | Differentiator; lyric embeddings unique |
| 3 | **Playlist generator** | High user retention; easy with ANN |
| 4 | **Why similar?** | Trust/explainability; uses SHAP or feature diff |
| 5 | **Mood explorer / song map** | Visual “wow” factor; UMAP + scatter-gl |

---

### Data Compaction / Scaling Strategy

| Strategy | Practical detail |
|---|---|
| **Quantization** | Store final feature matrix as float16/int16; 10k × 8192 float16 is ~164 MB |
| **PCA/UMAP** | For website map only, reduce to 2D/3D; do not use PCA for retrieval unless quality is validated |
| **ANN index** | FAISS `IndexFlatIP` for exact 10k search; HNSW for future 100k+ |
| **Precomputed k-NN** | Store top 50 neighbors per song per modality; ~10 MB for 10k |
| **Sparse matrices** | TF-IDF/char n-gram features as sparse CSR |
| **Feature documentation** | Ship a `feature_manifest.json` with dim ranges, type, normalization, source |

**Memory estimate for 10k songs:**  
- 10,000 × 8,192 dims float16 = ~164 MB.  
- With 5 modalities and multiple embeddings, 10k × 20,000 float16 = ~400 MB.  
- ANN index for 10k exact search is negligible.

**Scaling to 100k+:**  
- Use HNSW/USearch with int8 quantized embeddings.  
- Store PCA/UMAP coordinates for visualization.  
- Precompute sparse similarity graphs.

---

### Recommended Stack

| Layer | Recommendation | Why |
|---|---|---|
| **Backend** | FastAPI | Lightweight, async, Python |
| **ANN** | FAISS or USearch | FAISS for exact; USearch for compact HNSW |
| **Frontend** | Next.js / React + TypeScript | Static export, good ecosystem |
| **Embedding map** | UMAP precomputed + `scatter-gl` / `regl` / `three.js` | Handles 10k points |
| **Hosting** | Cloudflare Pages + small VPS | Low cost; static features |
| **PWA** | Yes | Mobile-friendly, offline mode |
| **Database** | Parquet/JSON + SQLite | Simple, portable |

---

### Legal / Licensing Red Flags for Kaggle + Website

**Spotify metadata:**  
- Spotify track/artist/album IDs, ISRC, popularity, and possibly chart rank are **Spotify proprietary data**. Redistributing them may violate Spotify’s terms.  
- **Kaggle risk:** DMCA / account ban.  
- **Safer approach:**  
  - Replace Spotify IDs with internal IDs or MusicBrainz IDs.  
  - Strip ISRC or map to MusicBrainz recordings.  
  - Keep aggregate chart rank/date as factual chart data, but consult legal.  
  - Do not ship raw Spotify audio features? They may be considered Spotify property. At minimum, document that they are Spotify-derived.

**Lyrics:**  
- If lyrics came from Genius or another provider, **do not republish full lyrics** on Kaggle or website.  
- Ship only extracted features/embeddings/stats, not raw lyric text.  
- For website lyric search, use licensed lyrics source or public-domain/CC lyrics.  
- MusicBrainz IDs and public-domain lyric sources are safer alternatives.

**Kaggle dataset recommendations:**  
- Keep audio files local; ship only feature vectors + metadata you have rights to.  
- Remove raw lyrics text.  
- Avoid Spotify track_id/artist_id/album_id/ISRC.  
- Use internal numeric IDs.  
- Include a data card explaining provenance.

---

## SECTION 4 — Hardware Feasibility Matrix

| Approach | Dims | VRAM est. | RAM est. | Time/song | Total 10k | License | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| **Better pooling stats** | ~0–512 | — | 2 GB | ~1 s | ~3 h | N/A | ✅ Easy |
| **Handcrafted DSP set** | 100–180 | — | 2 GB | ~5–10 s | 14–28 h | ISC/BSD | ✅ Easy |
| **Chromaprint** | fingerprint | — | 1 GB | ~0.5 s | 1.5 h | LGPL? | ✅ Easy |
| **PANNs raw tags** | 527 | ~2 GB | 3 GB | ~2–3 s | 6–8 h | code MIT; weights verify | ✅ Easy |
| **Essentia genre/mood** | 200+ | ~1 GB | 2 GB | ~2 s | 6 h | Essentia model license verify | ✅ Easy |
| **BEATs** | 768 | ~3 GB | 4 GB | ~2 s | 6 h | Apache? verify | ✅ Medium |
| **LAION-CLAP** | 512–1024 | ~2 GB | 4 GB | ~1.5 s | 4 h | MIT/weights vary | ✅ Medium |
| **MERT-v1-330M** | 1024 | ~4–6 GB | 6 GB | ~5–8 s | 14–22 h | CC BY-NC? verify | ⚠️ Medium-Hard |
| **OpenL3** | 512 | ~1 GB | 2 GB | ~2 s | 6 h | MIT | ✅ Easy |
| **YAMNet** | 1024 + 521 | ~1 GB | 2 GB | ~1–2 s | 3–6 h | Apache-2.0 | ✅ Easy |
| **AudioMAE** | 768 | ~2.5 GB | 4 GB | ~2 s | 6 h | Apache? | ✅ Medium |
| **AudioCLIP / Wav2CLIP** | 1024 / 512 | ~2–3 GB | 4 GB | ~2 s | 6 h | MIT/Apache? | ⚠️ Medium |
| **AST / SSAST** | 768 | ~2.5 GB | 4 GB | ~2 s | 6 h | BSD/Apache? | ✅ Medium |
| **CAV-MAE** | 768 | ~2.5 GB | 4 GB | ~2 s | 6 h | Apache? | ✅ Medium |
| **Demucs htdemucs** | 4 stems | ~3–4 GB | 6 GB | ~20–30 s | 55–80 h | MIT | ⚠️ Heavy |
| **Vocal/instrumental embeddings after Demucs** | +768/2048 | same | 6 GB | +2–3 s | +6–8 h | same | ⚠️ Phase B |
| **Chordino** | 25 | — | 2 GB | ~2 s | 6 h | GPL? verify | ✅ Easy |
| **madmom beats/downbeats** | ~12 | — | 2 GB | ~2 s | 6 h | BSD? | ✅ Easy |
| **MSAF structure** | ~10 | — | 2 GB | ~5–10 s | 14–28 h | BSD? | ⚠️ Medium |
| **MiniLM** | 384 | — | 2 GB | <0.1 s | ~0.3 h | Apache-2.0 | ✅ Easy |
| **MPNet** | 768 | — | 2 GB | ~0.3 s | ~0.8 h | Apache-2.0 | ✅ Easy |
| **multilingual-e5-base** | 768 | — | 3 GB | ~0.5 s | ~1.5 h | MIT? | ✅ Easy |
| **BGE-M3** | 1024 | — | 4 GB | ~1–2 s | ~3–5 h | BAAI license verify | ✅ Medium |
| **LaBSE** | 768 | — | 3 GB | ~0.5 s | ~1.5 h | Apache-2.0 | ✅ Easy |
| **nomic-embed** | 768 | — | 2 GB | ~0.2 s | ~0.5 h | Apache-2.0 | ✅ Easy |
| **Ember-v1** | 1024 | — | 6 GB | ~2–3 s | ~5–8 h | MIT? | ⚠️ Medium |
| **Lexicon/stylistic features** | ~50 | — | 2 GB | <1 s | ~3 h | mixed | ✅ Easy |
| **Qwen2.5-7B Q4 GGUF** | tags | — | 6 GB | ~10–30 s | 28–80 h | Apache-2.0 | ⚠️ Subset only |
| **BERTopic** | ~20 | — | 4 GB | total | ~1–2 h | mixed | ✅ Easy |
| **KeyBERT/YAKE** | ~20 | — | 2 GB | <0.1 s | ~0.3 h | MIT | ✅ Easy |

❌ **Not feasible without major workarounds:**  
- Jukebox  
- Full 8B/13B LLM fp16  
- MERT-v1-330M with 30s chunks if VRAM spikes above 6 GB; use 10s chunks or CPU fallback  
- Demucs on full 10k without careful batching.

---

## SECTION 5 — Feature-Set Comparison Methodology

### Tasks to Evaluate

| Task | Metric | Ground truth / proxy |
|---|---|---|
| **Valence/energy/danceability prediction** | R², RMSE, Spearman | Spotify audio features as labels |
| **Popularity prediction** | R², RMSE | Spotify popularity, but known external |
| **Similarity retrieval quality** | Genre purity, entropy, cover mAP | Metadata genre + known covers |
| **Mood neighborhood agreement** | Valence variance in kNN | Spotify valence |
| **Modality agreement** | Distance correlation/CKA | Compare audio, lyrics, metadata |
| **Human similarity** | Pairwise preference/triplets | 50–100 human labels |

### Same-Model Protocol

| Step | Detail |
|---|---|
| **Splits** | `GroupShuffleSplit` by artist_id to prevent artist leakage. Also try time split by release_date. |
| **Models** | Fixed CatBoost: `iterations=3000, depth=6, learning_rate=0.03, early_stopping=100` |
| **Baselines** | Metadata only; existing 4,256-d audio features; lyrics features; handcrafted; all |
| **Ablation** | Add one modality group at a time; measure R² gain |
| **Feature selection** | Permutation importance + variance inflation; do not judge by raw dim count |
| **No leakage** | No test-set tuning; no repeated holdout overuse; use nested CV |
| **Normalization** | Standardize numeric features before MLP; CatBoost does not strictly require it |

### Pitfalls to Avoid

1. **Artist leakage** — same artist songs in train/test will inflate results.  
2. **Spotify audio features are not ground truth** — they are algorithm outputs. Treat them as noisy labels, not human ground truth.  
3. **Popularity is external** — do not expect high R² from audio/lyrics.  
4. **Genre confound** — similarity can trivially separate hip-hop vs metal; ensure metrics include cross-genre evaluation.  
5. **Overfitting to 10k songs** — use regularization and nested CV.  
6. **Redundant feature sets** — 4,256 + more can overfit; compare feature-selected subsets, not only full sets.

### Cost-Aware Ranking

For each extraction, estimate:
- GPU/CPU hours.  
- Expected R² gain on valence/energy/danceability.  
- Expected retrieval quality improvement.  
- Kaggle usability / interpretability.

Formula:
```
priority = expected_gain / (extraction_hours * complexity)
```

The highest priority items should be:
1. PANNs raw tags.  
2. Handcrafted DSP.  
3. Better pooling stats.  
4. BEATs.  
5. Multilingual lyric embeddings.  
6. NRC/VADER mood features.  
7. Essentia genre/mood.  
8. MERT-330M if VRAM allows.  
9. Chromaprint.  
10. Demucs stems only on subset.

---

## SECTION 6 — Final Action Plan

### Phase A — This Week, Cheap

**Goal:** Add low-cost, high-value features without large GPU risk.

| Action | Details | Runtime |
|---|---|---|
| **Add PANNs raw tag vector** | Re-run PANNs or check if you can recover logits from existing model | ~6–8 h GPU |
| **Handcrafted DSP set** | Use librosa + essentia; save 100–180 dims | ~20–30 h CPU |
| **Better pooling stats** | If you kept frames, compute std/max/quantiles | ~3–6 h CPU |
| **Lyrics mood/structure features** | NRC, VADER, rhyme, line repetition, language detection | ~4–6 h CPU |
| **Multilingual lyric embedding** | Start with multilingual-e5-base or MiniLM baseline | ~2–3 h CPU |
| **Genre/mood Essentia tags** | Essentia discogs-effnet + mood classifiers | ~6 h CPU |
| **Chromaprint** | fpcalc fingerprints for dedupe/identity | ~1.5 h CPU |

**Timeline:** 3–5 days with overnight runs.

### Phase B — GPU, Heavier

| Action | Details | Runtime |
|---|---|---|
| **BEATs** | Extract 768-d embeddings; fp16, batch size 1 | ~6–8 h GPU |
| **MERT-v1-330M** | Use 10s chunks if OOM; save per-chunk pooled embeddings | ~14–22 h GPU |
| **LAION-CLAP** | Audio encoder only; music checkpoint | ~4–6 h GPU |
| **Demucs subset** | Run on 1,000–2,000 songs; extract vocal/instrumental embeddings | ~5–15 h GPU |
| **BGE-M3 or multilingual-e5** | Full 10k lyric embeddings | ~3–5 h CPU |
| **BERTopic + KeyBERT** | Topic/keyword features for Kaggle | ~2 h CPU |
| **Optional local 7B LLM tags** | Subset 1k songs for mood/theme tags | ~20–30 h CPU |

**Timeline:** 1–2 weeks.

### Phase C — Compare & Package

| Step | Detail |
|---|---|
| **Evaluate** | Use Section 5 protocol; compare feature sets on valence/energy/danceability/similarity |
| **Select winners** | Keep feature set ≤ 8k–12k dims per song after PCA/int8 quantization |
| **Package for Kaggle** | Parquet/NPZ, feature manifest, no raw lyrics, strip Spotify IDs/ISRC |
| **Build website prototype** | FastAPI + FAISS + Next.js + UMAP map |
| **Legal scrub** | Replace Spotify IDs with internal IDs; remove raw lyrics text |

**Timeline:** 1–2 weeks.

### Dependency Install List

```bash
# Audio / DSP
pip install librosa essentia-tensorflow? # use essentia
pip install torch torchaudio transformers
pip install laion-clap
pip install demucs
pip install pyacoustid
pip install madmom
pip install pyloudnorm
pip install nnAudio

# Lyrics / NLP
pip install sentence-transformers
pip install langdetect
pip install nrclex
pip install vaderSentiment
pip install pyphen
pip install phonemizer
pip install keybert yake bertopic
pip install faiss-cpu usearch umap-learn
pip install scikit-learn catboost shap
pip install fastapi uvicorn
```

### Top 5 Risks

| # | Risk | Mitigation |
|---|---:|---|
| 1 | **OOM on 6 GB VRAM** | Use 10s chunks, fp16, batch size 1, `--low-cpu-mem` where available |
| 2 | **License violation** | Strip Spotify IDs/ISRC/raw lyrics; verify model/weights licenses |
| 3 | **Multilingual lyric quality** | Use BGE-M3/multilingual-e5-base; test on language subsets |
| 4 | **Overfitting to genre confounds** | Artist-aware splits, cross-genre evaluation, feature selection |
| 5 | **Kaggle ToS / DMCA** | Do not publish Spotify metadata, raw lyrics, or audio files |

---

## Executive Summary

| Category | Top Recommendations | Why |
|---|---:|---|
| **Best 5 audio ideas** | 1. PANNs raw 527-d tag vector; 2. Handcrafted DSP set; 3. Better pooling stats; 4. BEATs; 5. MERT-v1-330M or Essentia genre/mood | High value per effort; feasible on 6 GB except MERT-330M needs chunking |
| **Best 3 lyrics ideas** | 1. BGE-M3 or multilingual-e5-base embeddings; 2. NRC + VADER mood lexicons; 3. Structural/rhyme/stylistic features | Multilingual robustness, mood, and similarity |
| **Lyrics benchmark answer** | No universal standard; use **ITS** if available, else **multilingual-e5-base/BGE-M3** as safe baseline and evaluate on your own 50–100 pairs | No clean leaderboard exists for lyrics similarity |
| **Top 5 website tools** | 1. Song DNA page; 2. Lyric quote search; 3. Playlist generator; 4. Why similar?; 5. Mood explorer / song map | High user value; uses existing features |

**Source URLs to verify:**  
- MERT: https://github.com/yizhilll/MERT  
- Hugging Face MERT: https://huggingface.co/m-a-p/MERT-v1-330M  
- BEATs: https://github.com/microsoft/unilm/tree/master/beats  
- LAION-CLAP: https://github.com/LAION-AI/CLAP  
- OpenL3: https://github.com/marl/openl3  
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet  
- Demucs: https://github.com/facebookresearch/demucs  
- Essentia models: https://essentia.upf.edu/models.html  
- Chromaprint: https://acoustid.org/chromaprint  
- Chordino: https://github.com/ohollo/chord-extractor  
- ITS: https://github.com/deezer/ITS  
- Sentence-transformers: https://sbert.net  
- Multilingual-E5: https://huggingface.co/intfloat/multilingual-e5-base  
- BGE-M3: https://huggingface.co/BAAI/bge-m3  
- LaBSE: https://huggingface.co/sentence-transformers/LaBSE  
- NRC Emotion Lexicon: https://github.com/metalcorebear/NRCLex  
- VADER: https://github.com/cjhutto/vaderSentiment  
- BERTopic: https://github.com/MaartenGr/BERTopic