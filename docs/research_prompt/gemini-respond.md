## 1. Executive Recommendation

* **Start with zero-compute & metadata features (Tier 0):** Derive temporal era indicators, artist collaboration degree centrality, release-year delta, and metadata/audio mismatch flags (duration delta, silence ratio) directly from `songs.csv` without invoking the GPU.
* **Test LAION-CLAP (`laion/clap-htsat-unfused`) first for audio:** Generates aligned 512-D audio–text embeddings that enable zero-shot natural language querying (e.g., *"upbeat synthwave with 80s drums"*), bridging a capability gap completely absent in your existing MERT, PANNs, and VGGish vectors.
* **Test Essentia-TensorFlow Discogs-EffNet & MTG-Jamendo auto-taggers:** Extracts 400 genre/style probabilities and 56 mood/theme logits (interpretable semantic tags) with negligible compute (< 1 hour for 10k tracks on GTX 1660 Ti).
* **Test Multilingual Long-Context Lyric Embeddings (`BAAI/bge-m3` or `jinaai/jina-embeddings-v3`):** Upgrades your existing 3,000-character truncated MPNet/MiniLM representations to an 8,192-token context window that ingests full song lyrics natively without truncation bias, supporting multilingual and code-switched tracks.
* **Extract interpretable lyric features (Tier 1):** Compute Type-Token Ratio (TTR/MTLD), line-level rhyme density via phonetic transcription, chorus repetition ratio, and GoEmotions-derived sentiment trajectories across song sections.
* **Pilot Madmom downbeat, meter, and rhythm stability:** Delivers true rhythmic structure (syncopation, beat consistency, meter confidence) that raw mel statistics and VGGish compress away.
* **Defer Full Demucs v4 stem separation across 10k tracks:** Running 4-stem hybrid Demucs (`htdemucs`) on a 6 GB VRAM GPU takes 35–50 hours of continuous runtime; run only a 200-song pilot to evaluate whether stem-isolated features outperform full-mix features on downstream tasks before scaling.
* **Defer frame-level CREPE pitch/vibrato contour extraction:** Running deep monophonic pitch trackers on full mixed audio yields high octave errors and phase confusion unless applied to isolated vocal stems, multiplying computational overhead.
* **Reject general speech foundation models (wav2vec 2.0, HuBERT):** These models are optimized for phonetic discriminability in clean speech and are heavily redundant with MERT v1 95M while performing worse on polyphonic musical timbres.
* **Reject Jukebox embeddings:** Jukebox (1B/5B) is computationally intractable on a 6 GB GTX 1660 Ti (frequent CUDA OOM) and generates massive multi-gigabyte representations with diminishing returns over modern lightweight MIR models.
* **Reject raw lyric LLM text generation for the full 10k dataset:** Batch prompting local 7B/8B LLMs via Ollama to generate summaries or structured JSON tags takes ~40–60 hours, introduces prompt-drift and hallucination risks, and creates ambiguous copyright licensing for Kaggle redistribution.
* **Never publish raw lyrics or audio files on Kaggle:** Distribute solely derived float32 embeddings, discrete structural metrics, and non-reconstructible semantic descriptors to comply with copyright laws and Spotify Developer Terms.
* **Enforce strict artist-grouped cross-validation splits:** Evaluate all prediction and retrieval baselines with GroupKFold grouped on `artist_names` to prevent over-optimistic performance caused by near-duplicate track leakage and artist signature overfitting.

---

## 2. Current-State Audit

### Dataset Profile & Assets

* **Audio Corpus:** 10,000 YouTube-derived audio files (Opus, ~144 kbps, 48 kHz stereo, ~4 min avg). Represents Spotify's most-listened tracks as of July 2025 (a popularity/commercial hit distribution, heavy on Western pop, hip-hop, reggaeton, and electronic genres).
* **Tabular Metadata:** `songs.csv` contains 10,000 rows × 32 columns, including Spotify audio features (danceability, valence, energy, tempo, loudness, etc.), artist metadata, and release dates.
* **Existing Audio Embeddings (3,456-D total per song):**
* VGGish: 128-D (AudioSet CNN baseline).
* MERT v1 95M: 768-D (self-supervised music transformer; pooled over the first 30 seconds only).
* PANNs Cnn14: 2,048-D (AudioSet tagging CNN; song-level embedding).
* Mel Spectrogram Statistics: 512-D (128 mel bands × [mean, std, max, min]).


* **Existing Lyric Features (1,159-D total per song):**
* MiniLM (`all-MiniLM-L6-v2`): 384-D (truncated to 3,000 chars).
* MPNet (`all-mpnet-base-v2`): 768-D (truncated to 3,000 chars).
* TextBlob Sentiment: 2-D (polarity, subjectivity).
* Basic Statistics: 5-D (word count, unique words, unique ratio, avg word length, char count).
* Coverage: 9,797 songs with lyrics; 203 missing/instrumental tracks.



### Key Gaps & Inefficiencies

1. **Truncation & Temporal Skew:** MERT is pooled strictly on the first 30 seconds (missing chorus, bridge, and outro dynamics). Lyric encoders truncate at 3,000 characters, truncating later verses/outros in narrative genres (e.g., hip-hop, ballad storytelling).
2. **Missing Cross-Modal Alignment:** No joint audio–text space exists. It is currently impossible to query audio tracks via natural language (e.g., *"acoustic guitar ballad with sad female vocals"*) without training a cross-modal adapter from scratch.
3. **No Explicit Musical Structure:** Lacking explicit features for harmonic progressions (chords, tonality shifts), rhythmic syncopation/meter confidence, structural segmentation (verse-chorus boundaries), and rhyme density.
4. **Multilingual Brittleness:** Existing text baselines (TextBlob, MiniLM) degrade significantly on non-English lyrics (e.g., Spanish reggaeton, K-pop, Turkish pop, French rap).

---

## 3. Candidate Matrix: Audio

The table below ranks candidate audio feature families by their marginal utility over your existing baseline (MERT-95M + PANNs + VGGish + Mel Stats) and their feasibility on a **GTX 1660 Ti (6 GB VRAM / 16 GB RAM)**.

| Rank | Candidate / Model | Exact Repo / Source | Output Dimensions | Primary Musical Properties Encoded | Redundancy vs Existing Baseline | Hardware Feasibility (GTX 1660 Ti) | Est. Runtime (10k songs) | Kaggle Release Safety | Best Use Case | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | **LAION-CLAP** (`clap-htsat-unfused`) | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) / HF | 512-D float32 | Cross-modal text–audio semantics, zero-shot tags, acoustic scenes | **Low**: Maps audio directly into a shared text-concept space (novel capability) | High (batch size 4, chunked to 10s windows, pooled) | ~3.5 hours | Safe (extracted vectors) | Cross-modal text-to-audio search; zero-shot tag retrieval | **Tier 1 (High Priority)** |
| **2** | **Essentia Discogs-EffNet** | [Essentia Models](https://essentia.upf.edu/models.html) (MTG-UPF) | 400-D logits + 128-D embedding | Discogs music genres, micro-styles, historical production eras | **Medium-Low**: Supervised on rich commercial music taxonomy (unlike AudioSet) | High (TensorFlow C++ / Python bindings, batch 16) | ~1.2 hours | Safe (extracted vectors & logits) | Supervised genre classification; interpretable style tags | **Tier 1 (High Priority)** |
| **3** | **Essentia MTG-Jamendo Auto-Taggers** | [Essentia Models](https://essentia.upf.edu/models.html) (MTG-UPF) | 56 mood tags + 40 instrument tags | Perceived mood/emotion, instrumentation presence | **Medium-Low**: Interpretable human concepts (happy, dark, strings, synth) | High (CPU/GPU lightweight) | ~0.8 hours | Safe (probabilities/logits) | Explainable recommendations; mood filtering | **Tier 1 (High Priority)** |
| **4** | **Madmom Rhythmic & Downbeat Engine** | [madmom](https://github.com/CPJKU/madmom) / ReadTheDocs | ~16 scalar features | Meter confidence (3/4 vs 4/4), beat tracking DBN, tempo stability, syncopation | **Zero**: Explicit temporal-rhythmic modeling absent in static pooled embeddings | High (CPU multi-processing, 8 workers) | ~2.0 hours | Safe (scalar metrics) | Rhythm-based similarity; danceability prediction | **Tier 1 (High Priority)** |
| **5** | **Librosa Structured Chroma / Tonnetz / Novelty** | [librosa](https://librosa.org/) (v0.10.x) | 36 scalar & distribution metrics | HPCP chroma variance, tonnetz harmonic shift, spectral novelty curves | **Low**: Captures chord complexity and structural section transitions | High (CPU parallel execution) | ~1.5 hours | Safe (scalar features) | Harmonic similarity; structural boundary analysis | **Tier 1 (High Priority)** |
| **6** | **Basic Pitch** (Spotify AMT) | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | Polyphonic MIDI / note event statistics (pitch histogram, note density) | Note onsets, pitch bend, polyphonic melody & chord density | **Low**: Converts raw continuous audio into discrete symbolic music events | Medium (runs via ONNX/TensorFlow Lite; batch 1) | ~8.0 hours | Safe (note summary stats) | Lead melody analysis; musical complexity metrics | **Tier 2 (Pilot 500)** |
| **7** | **Demucs v4** (`htdemucs`) Stem Extraction | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | 4 stems (vocals, drums, bass, other) -> stem energy ratios | Vocal-to-instrumental ratio, drum prominence, bassline energy | **Low**: Isolates acoustic sources; cleans vocal channel for downstream NLP/F0 | Low-Medium (6 GB VRAM tight; requires `--segment 10`, batch 1) | ~42.0 hours | Safe (energy ratios & stem-derived features; do NOT distribute audio stems) | Vocal presence/energy ratio; source-specific analysis | **Tier 2 (Pilot 200)** |
| **8** | **MERT-v1-330M Full-Song (Multi-Window)** | [m-a-p/MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) | 1,024-D float32 | Deep self-supervised acoustic & musical transformer representations | **High**: Replaces MERT-95M, but marginal gain on standard benchmarks is 2–4% | Medium (fp16, batch 1, chunked 5s windows) | ~14.0 hours | Safe (extracted vectors) | High-accuracy supervised modeling | **Tier 2 (Ablate vs 95M)** |
| **9** | **AudioMAE** (Audio Masked Autoencoder) | [facebookresearch/AudioMAE](https://github.com/facebookresearch/AudioMAE) | 768-D float32 | General environmental and acoustic scene representations | **Very High**: Redundant with PANNs Cnn14 and VGGish; lacks music-specific pretraining | Medium (ViT-Base, batch 4) | ~5.5 hours | Safe | General audio classification | **Do Not Prioritize** |
| **10** | **Wav2Vec 2.0 / HuBERT** | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) / Hugging Face | 768-D float32 | Phonetic speech representations, vocal phonemes | **High**: Ineffective on polyphonic music mixes; heavily redundant with MERT | Medium | ~6.0 hours | Safe | Lyric alignment from audio | **Do Not Prioritize** |
| **11** | **OpenAI Jukebox** | [openai/jukebox](https://github.com/openai/jukebox) | 4,800-D | Hierarchical VQ-VAE autoregressive music representations | **Medium**: Rich representations, but prohibitively heavy | **Infeasible**: Exhausts 6 GB VRAM; extreme memory leaks and latency | > 200 hours | Safe | Generative modeling | **Reject** |

---

## 4. Candidate Matrix: Lyrics

The table below ranks candidate lyric representations for semantic retrieval, supervised prediction, and interpretable feature extraction.

| Rank | Candidate / Method | Library / Model Source | Output Dimensions | Core Property / Signal Encoded | Multilingual Validity | Hardware Feasibility | Est. Runtime (10k lyrics) | Kaggle Release Safety | Best Use Case | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | **BGE-M3** (`BAAI/bge-m3`) | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) / HF | 1,024-D (dense) + lexical sparse | Multi-lingual semantic retrieval, full context (8,192 tokens), dense + sparse multi-vector | Excellent (100+ languages) | High (batch size 8 on GTX 1660 Ti, fp16) | ~15 minutes | Safe (embeddings only) | Core lyrical semantic similarity and dense retrieval | **Tier 1 (High Priority)** |
| **2** | **Lexical Richness & Repetition Metrics** | `lexicalrichness`, `textstat` | 18 scalar metrics | TTR, Root TTR, MTLD, HD-D, Simpson's index, chorus line repetition ratio | Universal (character & whitespace tokenized) | Ultra-High (CPU single thread) | ~2 minutes | Safe (scalar statistics) | Song complexity analysis; genre prediction (hip-hop vs pop) | **Tier 1 (High Priority)** |
| **3** | **Phonetic Rhyme & Structural Analyzer** | `pronouncing` (CMUdict) / `epitran` | 12 scalar metrics | End-rhyme density, internal rhyme scheme entropy, hookiness repetition index | Moderate (English CMUdict native; G2P fallback for Romance languages) | High (CPU multi-core) | ~5 minutes | Safe (scalar metrics) | Flow & lyricism scoring; rap vs pop classification | **Tier 1 (High Priority)** |
| **4** | **RoBERTa GoEmotions Trajectory** | `SamLowe/roberta-base-go_emotions` | 28 fine-grained emotion probabilities × 3 sections | Sectional emotion arc (verse -> chorus -> outro), narrative sentiment shifts | English native; requires translation for non-English | High (batch size 16, fp16) | ~8 minutes | Safe (discrete probability vectors) | Mood mismatch detection; narrative arc exploration | **Tier 1 (High Priority)** |
| **5** | **Jina Embeddings v3** (`jina-embeddings-v3`) | [Jina AI](https://huggingface.co/jinaai/jina-embeddings-v3) | 1,024-D (Matryoshka flexible down to 256-D) | Task-adapted asymmetric retrieval, late-chunking support | High (multilingual XLM-R backbone) | High (batch size 8, fp16) | ~20 minutes | Safe (embeddings only) | Asymmetric search (e.g., query phrase -> song lyrics) | **Tier 2 (Compare vs BGE-M3)** |
| **6** | **Multilingual Topic Modeling (BERTopic)** | [BERTopic](https://github.com/MaartenGr/BERTopic) (c-TF-IDF + BGE-M3) | 32-D topic distribution | Interpretable lyrical themes (heartbreak, luxury/flexing, nostalgia, partying) | High (multilingual embeddings) | High (CPU/GPU hybrid) | ~10 minutes | Safe (topic distributions & top keywords) | Lyrical theme discovery; semantic filtering | **Tier 2 (High Value)** |
| **7** | **Section-Aware Hierarchical Pooling** | Custom spaCy / Regex Section Parser + MPNet | 768-D verse + 768-D chorus vectors | Disentangles core theme (chorus hook) from narrative exposition (verses) | Moderate | High (uses existing MPNet/BGE-M3) | ~15 minutes | Safe (pooled vectors) | Sub-song similarity matching; chorus-only search | **Tier 2 (Pilot)** |
| **8** | **Local LLM Structured Extraction (Llama 3.1 8B)** | Ollama / `vllm` (GGUF Q4_K_M) | Structured JSON (Narrative POV, Central Metaphor, Setting, Toxicity) | High-level semantic reasoning and abstract thematic tags | High | Low (requires 6–8s per song on CPU/GPU offload) | ~22 hours | Safe (extracted discrete labels; do NOT release raw text) | Deep song explainability; curated thematic metadata | **Tier 3 (Pilot 200 songs)** |
| **9** | **Domain Contrastive Fine-Tuning (SimCSE/LoRA)** | Hugging Face `transformers` + `peft` | 768-D / 1,024-D | Lyric-specific semantic alignment via artist-disjoint contrastive loss | Moderate | Medium (10k dataset is small; high overfitting risk) | ~1.5 hours training | Safe | In-domain lyrical search optimization | **Tier 3 (Requires Manual Evaluation Set)** |

---

## 5. Lyrics Benchmark Analysis & Evaluation Plan

### Direct Answer to the Benchmark Question

> **Is there a broadly accepted benchmark that lets you directly identify the "best" lyric representation model?**
> **No.** A universal, standardized "MTEB for Lyrics" does not exist.

#### Why a Universal Benchmark Does Not Exist

1. **Severe Copyright Fragmentation:** Song lyrics are proprietary, protected literary works strictly enforced by major publishers (Universal, Sony, Warner). Academic datasets cannot legally distribute full-text raw lyrics, forcing researchers to distribute either scraped IDs, scrambled bag-of-words (e.g., Million Song Dataset musiXmatch), or ephemeral download scripts that rot over time.
2. **Subjective Multidimensional Semantics:** General text retrieval benchmarks (MTEB, BEIR, STS) evaluate objective factual retrieval and semantic textual similarity. Lyric similarity is inherently multi-faceted: two songs can be similar in *rhyme scheme*, *emotional valence*, *metaphorical theme*, or *narrative structure* while sharing zero lexical overlap.
3. **Multilingual and Poetic Nuance:** Lyrics contain non-standard grammar, slang, deliberate phonetic distortions, and code-switching (e.g., Spanglish, Turkish-German rap) that break standard NLP tokenizers and linguistic assumptions.

```
                    ┌────────────────────────────────────────────────────────┐
                    │       Why Off-the-Shelf NLP Benchmarks Fail for        │
                    │                    Song Lyrics                         │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
   │ Copyright Wall    │             │ Semantic Ambiguity│             │ Structural Skew   │
   │ No open full-text │             │ Literal similarity│             │ Repetitive hooks, │
   │ benchmark sets    │             │ ≠ thematic/mood   │             │ poetic meter, and │
   │ can be distributed│             │ lyrical similarity│             │ slang distort NLP │
   └───────────────────┘             └───────────────────┘             └───────────────────┘

```

---

### Existing Relevant Datasets & Academic Benchmarks

The following table summarizes the primary existing datasets used in academic MIR and lyric NLP research:

| Benchmark / Dataset | Primary Task | Language | Size | Public Availability & License | Usability for Model Selection in This Project |
| --- | --- | --- | --- | --- | --- |
| **LyricSim (Benito-Santos et al., 2023)** | Semantic similarity scoring (0–4 scale) | Spanish | 676 curated lyric pairs (annotated by 63 evaluators) | Academic Open (CC BY-SA 4.0) | **High for Validation:** Best direct human-annotated ground truth for lyric semantic similarity, but restricted to Spanish. |
| **4MuLA (da Silva et al., 2020)** | Multimodal genre, artist, emotion, and music similarity | English, Portuguese, Spanish | ~48,000 tracks with audio features & tags | Academic Open | **Moderate:** Useful for testing cross-lingual transfer, but tags are user-generated and noisy. |
| **MoodyLyrics / MoodyLyrics4Q** | 4-quadrant Russell emotion classification (Happy, Sad, Angry, Relaxed) | English | ~2,500 songs | Academic Open (IDs only; lyrics must be scraped) | **Moderate:** Good for testing emotion classification heads; clean categorical labels. |
| **WASSA / SemEval Emotion Shared Tasks** | Fine-grained emotion intensity regression | English | 7,000+ text samples | Open Academic | **Indirect:** General text emotion, not lyrics, but standard for evaluating emotion embedding quality. |
| **PMEmo / DEAM** | Continuous dynamic arousal & valence regression | English / Instrumental | 794 songs (PMEmo), 1,802 songs (DEAM) | Open Academic (CC BY-NC 4.0) | **High for Cross-Modal:** Evaluates audio–lyric mood alignment against continuous human valence/arousal annotations. |
| **Million Song Dataset (MSD) musiXmatch** | Topic modeling, genre prediction, bag-of-words retrieval | Multilingual | 237,662 songs (top 5,000 word stem counts) | Restricted Academic (Bag-of-words only) | **Low:** Bag-of-words format prevents evaluating modern dense transformer encoders. |

---

### In-Domain Evaluation Protocol for Your 10k Dataset

Rather than relying on mismatched external benchmarks, construct a rigorous, lightweight in-domain evaluation benchmark directly from your 10,000 tracks.

```
                    ┌────────────────────────────────────────────────────────┐
                    │       In-Domain Evaluation Framework Construction      │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                      ▼                                      ▼
 ┌──────────────┐                       ┌──────────────┐                       ┌──────────────┐
 │ Stratified   │                       │ Multi-Source │                       │ Multi-Metric │
 │ Query Pool   │ ──► Candidate Pair ──►│ Human Rating │ ──► Agreement & ───►  │ Benchmark    │
 │ (100 songs,  │     Sampling          │ (3 annotators│     Adjudication      │ (nDCG, MRR,  │
 │ 5 genres)    │     (4 tiers)         │ per pair)    │     (Fleiss' κ)       │ Spearman ρ)  │
 └──────────────┘                       └──────────────┘                       └──────────────┘

```

#### 1. Test Suite Construction (100 Query Songs, 500 Evaluated Pairs)

* **Query Stratification:** Sample 100 anchor tracks stratified across 5 core genres (20 Pop, 20 Hip-Hop/R&B, 20 Rock/Indie, 20 Electronic/Dance, 20 Latin/Multilingual).
* **Candidate Pair Sampling (5 pairs per anchor):**
1. *High Lexical / Same Artist:* Strong lexical overlap baseline (near-duplicate control).
2. *High Dense Similarity / Disjoint Artist:* Selected from top-5 nearest neighbors via baseline MPNet.
3. *High BGE-M3 Dense / Low MPNet:* Disagreements between candidate models (reveals new signal).
4. *Same Genre / Low Semantic Overlap:* Pseudo-negative baseline.
5. *Uniform Random Track:* Hard negative baseline.



#### 2. Human Annotation Rubric (3–5 Distinct Dimensions)

Annotators evaluate each lyric pair on a 1–5 Likert scale across four independent dimensions:

* **Thematic / Topic Overlap:** Do the songs discuss the same core subject matter (e.g., mourning a loss, celebration of wealth, unrequited love)?
* **Emotional Valence & Tone:** Is the emotional delivery and sentiment aligned (e.g., bitter vs optimistic)?
* **Narrative Perspective & Style:** Are the storytelling structures similar (e.g., direct second-person dialogue, abstract stream-of-consciousness, story-arc)?
* **Overall Lyrical Substitutability:** Would a listener wanting more lyrics like Song A be satisfied with Song B?

#### 3. Inter-Rater Reliability & Retrieval Metrics

* **Inter-Annotator Agreement:** Measure reliability using Fleiss’ Kappa ($\kappa$) for categorical buckets or Two-Way Random Intraclass Correlation Coefficient ($\text{ICC}(2,1)$) for continuous scalar scores. Target $\kappa > 0.65$.
* **Quantitative Retrieval Metrics:**

$$\text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}, \quad \text{where } \text{DCG}@k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$


$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}, \quad \text{Recall}@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$


* **Artist Disjointness Constraint:** When evaluating retrieval, hard-exclude songs by the same artist from the candidate pool to ensure the model retrieves semantic concepts rather than artist-specific vocabulary signatures.

---

## 6. Website & Future Product Concepts

The proposed product tools leverage your multidimensional feature store, moving from pure nearest-neighbor lookups to interactive, explainable music exploration.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Interactive Music Discovery Dashboard                           │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ [🎵 Audio Similarity]  [=====o=====]   │  Why these songs?                             │
│ [📝 Lyrical Mood]      [========o==]   │  • Both feature acoustic guitar & minor keys  │
│ [🥁 Rhythmic Drive]    [===o=======]   │  • Shared theme: Nostalgic reflection         │
│                                        │  • Lyric Mood Mismatch: High (Upbeat / Sad)   │
├────────────────────────────────────────┴───────────────────────────────────────────────┤
│  Retrieved Track: "Dancing on My Own" (Acoustic vs Original)                           │
│  [2-D Semantic Landscape View] (UMAP exploratory projection)                           │
└────────────────────────────────────────────────────────────────────────────────────────┘

```

### 1. Controllable Multi-Axis Similarity Search ("Sliding-Scale Discovery")

* **User Problem:** Standard similarity recommenders force a black-box blend. Users cannot specify whether they want a song that *sounds* similar or one that *speaks* about the same topic.
* **Mechanism & Representations:**

$$S_{\text{total}}(A, B) = w_a \cdot \cos(e_{\text{audio}}^A, e_{\text{audio}}^B) + w_l \cdot \cos(e_{\text{lyric}}^A, e_{\text{lyric}}^B) + w_r \cdot \text{Sim}_{\text{rhythm}}(A, B)$$



Weights $w_a, w_l, w_r \in [0, 1]$ are dynamically set via UI sliders.
* **Retrieval & Indexing:** Pre-compute three separate HNSW (Hierarchical Navigable Small World) indexes: Audio (CLAP/MERT), Lyrics (BGE-M3), and Rhythm (Madmom/Spotify features). Merge candidate lists using Reciprocal Rank Fusion (RRF).
* **Scale Feasibility:** Sub-5ms latency for 10k tracks via HNSWlib in-memory; easily scales to 1M tracks using quantized vector indexes.

### 2. Audio–Lyric Mood Mismatch Explorer ("Happy-Sad Songs")

* **User Problem:** Music enthusiasts love songs with cheerful, danceable music paired with dark, depressing lyrics (e.g., *"Pumped Up Kicks"*, *"Dancing On My Own"*).
* **Mechanism & Representations:**

$$\Delta_{\text{mismatch}} = \text{Spotify Energy} - \text{RoBERTa Lyric Valence}$$



Identify outlier tracks in the top 5th percentile of positive or negative mismatch.
* **Explainability UI:** Display a dual-gauge meter showing high acoustic energy (85%) alongside melancholic/grief lyric classification (92%).

### 3. Lyric Motif & Rhyme Discovery Engine

* **User Problem:** Hip-hop and poetry enthusiasts want to find songs that utilize specific rhyme densities, internal multi-syllable rhyme patterns, or unique lyrical metaphors.
* **Mechanism & Representations:** Lexical Richness (MTLD), Rhyme Density index, and c-TF-IDF keyword motifs.
* **Indexing:** Filter by structured metrics in Parquet, then rank via sparse BM25/BGE-M3 lexical matching.

### 4. 2-D Exploratory Song Map (UMAP Projection)

* **Architecture Caution:** **Never use 2-D UMAP coordinates as the similarity search engine.** Projecting 1,024-D vectors to 2-D destroys high-dimensional topological neighborhoods and introduces severe distance distortions.
* **Correct Usage:** Use 2-D UMAP strictly for canvas rendering, panning, zooming, and cluster color-coding in WebGL/Three.js. Perform actual nearest-neighbor queries against the full-dimensional HNSW index in the backend.

---

## 7. Kaggle-Ready Data Architecture

To ensure full reproducibility, high performance, and legal safety, organize the dataset following modern tabular-vector hybrid conventions.

```
spotify-10k-expanded/
├── README.md                           # Comprehensive documentation & attribution
├── dataset_manifest.json               # SHA256 hashes, row counts, creation dates
├── metadata/
│   ├── songs_clean.parquet             # Cleaned core metadata (no raw lyrics)
│   ├── audio_descriptors.parquet       # Low-dimensional MIR scalar features
│   └── lyric_features.parquet          # Interpretable lyric statistics & emotion tags
├── embeddings/
│   ├── audio_clap_512.npy              # LAION-CLAP audio embeddings (float32)
│   ├── audio_mert_768.npy              # MERT v1 95M embeddings (float32)
│   ├── audio_panns_2048.npy            # PANNs Cnn14 embeddings (float32)
│   ├── lyrics_bgem3_1024.npy           # BGE-M3 full-context lyric embeddings (float32)
│   └── track_ids.npy                   # Master ordered track_id alignment array
└── evaluation/
    ├── splits_artist_grouped.parquet   # 5-fold cross-validation split indices
    └── lyric_similarity_benchmark.json # 500 annotated evaluation pairs

```

### Data Formats & Schemas

* **Tabular Metadata (`.parquet`):** Use Apache Parquet with Snappy compression for structured tables. It preserves column data types, executes 10x faster than CSV, and reduces disk storage by 75%.
* **Embeddings (`.npy` or `.safetensors`):** Store feature matrices as raw 2D NumPy float32 arrays `(10000, D)` aligned 1:1 with `track_ids.npy`.

### What NOT to Distribute (Legal & Licensing Safeguards)

* **NO Raw Audio Files:** Do not publish Opus/MP3 audio clips. Distribute only derived numeric feature representations.
* **NO Full Raw Lyric Text:** Do not include a raw `lyrics` string column in public Kaggle releases. Full song lyrics are protected under literary copyright. Distribute only statistical metrics, line counts, emotion logits, and high-dimensional embeddings (which are non-invertible and legally transformative).
* **NO Spotify Proprietary Previews:** Do not distribute scraped 30-second MP3 previews.

---

## 8. Staged 10k-Song Roadmap

```
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                            4-Tier Execution Workflow                                   │
  └────────────────────────────────────────────────────────────────────────────────────────┘
     │
     ├──► [Tier 0: Data Hygiene & Baseline Extraction] (~0.5 hr, CPU)
     │     • Metadata cleaning, silence/clipping flags, missing lyric tokens.
     │     • Gate: Zero NaNs, 100% ID alignment.
     │
     ├──► [Tier 1: Core Feature Expansion] (~5.5 hrs, GPU/CPU)
     │     • LAION-CLAP, Essentia EffNet/Jamendo, BGE-M3, Lexical Richness.
     │     • Gate: Cross-modal retrieval MRR > 0.40; Lyric nDCG@10 > MPNet (+10%).
     │
     ├──► [Tier 2: Targeted Pilots] (~4 hrs, GPU/CPU)
     │     • 200-song Demucs pilot, 500-song Basic Pitch pilot, BERTopic.
     │     • Gate: Feature importance > 5% in ablation; compute feasible.
     │
     └──► [Tier 3: Scaling & Optimization] (~4 hrs, CPU/GPU)
           • HNSW indexing, UMAP 2D projection, Parquet packaging.
           • Gate: Sub-10ms search latency, verified SHA256 manifest.

```

### Tier 0: Data Hygiene, Quality Control, and Free Features

* **Tasks:**
* Clean metadata: Standardize release dates to ISO-8601, parse genres into JSON arrays.
* Audio quality check: Scan 10,000 Opus files for digital silence ($> 5\text{s}$ leading/trailing silence), RMS loudness, and duration delta ($\vert{}\text{audio\_dur} - \text{spotify\_dur\_ms}\vert{} > 10\text{s}$). Flag mismatches in a `qc_mismatch_flag` column.
* Text cleaning: Strip scraping headers (e.g., *"Embed"*, *"Contributors"*, *"Lyrics for this song"*), normalize Unicode, identify languages using `fasttext` (`lid.176`). Assign missing-lyric placeholder tokens to the 203 empty tracks.


* **Compute / Storage:** ~30 minutes on CPU (multi-core); +5 MB storage.
* **Go/No-Go Gate:** 100% of tracks pass shape/ID verification; zero unhandled `NaN` or `Inf` values in feature arrays.

### Tier 1: High-Value, Low-Risk Core Additions

* **Tasks:**
* Audio: Extract LAION-CLAP (`laion/clap-htsat-unfused`, 512-D), Essentia Discogs-EffNet (400 styles + 128-D), and Essentia MTG-Jamendo (56 mood + 40 instrument tags).
* Rhythm/MIR: Compute Madmom beat/downbeat stability and librosa tonnetz/chroma variance.
* Lyrics: Extract full-context BGE-M3 dense embeddings (1,024-D), lexical richness suite (18 features), and RoBERTa GoEmotions sectional arcs.


* **Hardware & Runtime on GTX 1660 Ti:**
* CLAP: ~3.5 hours (batch size 4, fp16).
* Essentia Taggers: ~1.5 hours (batch size 16).
* BGE-M3: ~15 minutes (batch size 8, fp16).
* Interpretable text/audio scripts: ~2 hours (CPU parallel).
* Total Runtime: **~7.2 hours**.
* Total New Storage: **~180 MB** (Parquet + NPY).


* **Go/No-Go Gate:** BGE-M3 achieves statistically significant improvement ($p < 0.01$) over MPNet baseline on in-domain lyric retrieval nDCG@10; CLAP zero-shot genre retrieval achieves Top-5 Accuracy $> 60\%$.

### Tier 2: Empirical Pilots

* **Pilot A (Demucs v4 Source Separation):** Run `htdemucs` on 200 diverse songs (batch size 1, `--segment 10`). Compute vocal-to-music energy ratio and drum prominence.
* *Gate:* Vocal stem features must increase downstream emotion/genre prediction $R^2 / F_1$ by at least $+0.03$ over whole-mix features. If not, reject full 10k extraction (saves ~40 hours of compute).


* **Pilot B (Basic Pitch Polyphonic AMT):** Transcribe 500 tracks to MIDI note events; compute pitch entropy and average note velocity.
* *Gate:* Note density and pitch complexity metrics must show low collinearity ($\vert{}r\vert{} < 0.50$) with Spotify danceability/acousticness descriptors.


* **Pilot C (BERTopic Modeling):** Fit 32 lyric topics across all 9,797 lyric tracks using BGE-M3 embeddings.
* *Gate:* Topic coherence score ($C_v > 0.55$) and distinct cluster interpretability.



### Tier 3: Indexing, Dimensionality Reduction & Web Preparation

* **Tasks:**
* Fit PCA (preserving 95% variance) on audio and lyric embeddings **strictly on the training split**.
* Build HNSW vector indexes (`hnswlib`) for CLAP, BGE-M3, and multimodal concatenations ($M=16$, $efConstruction=200$).
* Compute 2-D UMAP exploratory layout coordinates for front-end rendering.


* **Compute / Storage:** ~40 minutes on CPU/GPU; +45 MB for HNSW index files.

---

## 9. Ablation & Evaluation Protocol

To prevent methodological bias and validate feature additions, enforce this standardized evaluation harness across both predictive and retrieval tasks.

```
                    ┌────────────────────────────────────────────────────────┐
                    │            Artist-Disjoint Evaluation Harness          │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
   ┌───────────────────┐                                               ┌───────────────────┐
   │ Task 1: Retrieval │                                               │ Task 2: Prediction│
   │ (Ranking Metrics) │                                               │ (Supervised CV)   │
   └─────────┬─────────┘                                               └─────────┬─────────┘
             │                                                                   │
             ├──► Audio Text-to-Audio (nDCG@10, MRR)                             ├──► XGBoost / LightGBM Baselines
             ├──► Lyrical Semantic Similarity (Spearman ρ)                       ├──► Targets: Valence, Genre, Year
             └──► Cover / Version Search (MAP@R)                                 └──► Metric: R² / Multi-class F1

```

### 1. Data Splitting & Leakage Controls

* **Artist-Grouped Cross-Validation:** Partition the 10,000 tracks into 5 folds using `GroupKFold(n_splits=5)` grouped on `artist_names`. This guarantees that no artist present in the training set appears in the validation/test set, preventing models from memorizing artist-specific production styles.
* **Target Leakage Prohibition:** The target column (e.g., `popularity`) must be strictly removed from the input matrix during supervised training.
* **Preprocessing Isolation:** All scalers (`StandardScaler`), encoders, and dimensionality reduction models (`PCA`, `UMAP`) must be `fit` solely on the training fold and applied to the test fold via `transform`.

### 2. Supervised Prediction Benchmark Suite

* **Models:** LightGBM / XGBoost Regressors and Classifiers (fast, tabular-optimized, handles non-linear feature interactions).
* **Downstream Tasks:**
1. *Continuous Musical Valence & Energy Prediction* (Metric: $R^2$, RMSE).
2. *Macro Genre Classification* (10 classes, Metric: Macro-$F_1$).
3. *Release Year Estimation* (Metric: Mean Absolute Error in years).


* **Ablation Comparison Groups:**
* **Baseline A:** Spotify native audio descriptors only (12 features).
* **Baseline B:** Current Audio Embeddings (MERT-95M + PANNs + VGGish + Mel Stats = 3,456-D).
* **Baseline C:** Current Lyric Embeddings (MPNet + MiniLM + TextBlob + Stats = 1,159-D).
* **Baseline D:** Combined Current Pipeline (4,254 features).
* **Experimental Tiers:** Baseline + LAION-CLAP; Baseline + Essentia Tags; Baseline + BGE-M3; Full Proposed Feature Store.



### 3. Unsupervised Retrieval Benchmark Suite

* **Audio Retrieval:** Query audio vectors against annotated genre/style clusters. Measure Mean Average Precision ($\text{MAP}@10$) and Precision@5.
* **Lyric Retrieval:** Query against the 500 human-annotated lyric similarity pairs. Measure Spearman's rank correlation ($\rho$) against human consensus scores and $\text{nDCG}@10$.
* **Statistical Significance:** Compute 95% bootstrap confidence intervals (1,000 resamples) for all metric differences ($\Delta R^2$, $\Delta \text{nDCG}$). Differences are accepted as meaningful only if $p < 0.01$.

---

## 10. Comprehensive Risk Register

| Risk ID | Risk Category | Failure Scenario / Description | Severity | Likelihood | Concrete Mitigation Strategy |
| --- | --- | --- | --- | --- | --- |
| **R-01** | **Copyright / Legal** | Distributing raw copyrighted lyrics or YouTube audio files leads to DMCA takedown on Kaggle. | **Critical** | High | Distribute **only** derived mathematical features, embeddings, and non-invertible statistical metrics. Zero raw audio/text distribution. |
| **R-02** | **Target Leakage** | Including `popularity` or Spotify follower counts in feature sets when predicting popularity yields artificially inflated $R^2 \approx 0.99$. | **High** | Medium | Isolate metadata columns; maintain strict programmatic blacklists in training scripts that drop all target columns prior to `fit()`. |
| **R-03** | **Artist / Near-Duplicate Leakage** | Alternate versions/remasters of the same song appear in both train and test folds, inflating validation performance. | **High** | High | Implement `GroupKFold` grouped on `artist_names`. Run acoustic fingerprint deduplication via Chromaprint (`fpcalc`) to drop exact audio duplicates. |
| **R-04** | **Hardware OOM (6 GB VRAM)** | Loading large transformer models (MERT-330M, CLAP) with long audio buffers causes PyTorch CUDA Out-of-Memory crashes. | **High** | High | Enforce fp16 inference (`torch.autocast`); use strict chunking (10s audio segments); cap batch size at 1 or 2; call `torch.cuda.empty_cache()` after each batch. |
| **R-05** | **Audio Provenance Mismatch** | YouTube downloads contain unofficial covers, live concert audio, or extended intros differing from the Spotify metadata. | **Medium** | High | Compute an audio quality flag: compare Opus duration against Spotify `duration_ms`. Filter out tracks with $\vert{}\Delta_{\text{duration}}\vert{} > 10\text{s}$ during core evaluations. |
| **R-06** | **Multilingual NLP Bias** | English-trained sentiment/topic models produce gibberish or severe score distortions on Spanish, Turkish, or Korean lyrics. | **Medium** | High | Run fastText language identification. Route non-English lyrics through multilingual encoders (BGE-M3); isolate English-only models (GoEmotions) to English-tagged subsets. |
| **R-07** | **Feature Alignment Drift** | Re-ordering or dropping rows during text cleaning misaligns `.npy` embedding matrices with `songs.csv`. | **Critical** | Medium | Use immutable string `track_id` arrays saved alongside every feature matrix. Validate row alignment via runtime assertions before training. |
| **R-08** | **Model Weight Licensing** | Using non-commercial (CC BY-NC) weights in a commercial web product causes legal compliance issues. | **Medium** | Low | Segregate features by license: use Apache 2.0 / MIT models (CLAP, BGE-M3, Basic Pitch) for the commercial product; reserve CC BY-NC-SA (Essentia MTG) for Kaggle research. |

---

## 11. Annotated Bibliography & Primary Sources

### Audio Models & MIR Libraries

* **LAION-CLAP:** Wu, Y., et al. (2023). *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.* ICASSP 2023.
* Source Code & Model Hub: [github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) | Hugging Face: [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused).
* License: Apache-2.0.


* **MERT:** Li, Y., et al. (2023). *MERT: Acoustic Music Understanding with Large-Scale Self-supervised Pre-training.* arXiv:2306.00107.
* Source Code & Model Hub: [github.com/yizheliu/MERT](https://www.google.com/search?q=https://github.com/yizheliu/MERT) | Hugging Face: [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M), [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M).
* License: CC BY-NC-SA 4.0.


* **Essentia & Discogs-EffNet:** Bogdanov, D., et al. (2013/2023). *Essentia: An open-source library for sound and music analysis.* ACM Multimedia; and *Discogs-EffNet Models for Music Style and Mood Classification.*
* Documentation & Models: [essentia.upf.edu/models.html](https://essentia.upf.edu/models.html).
* License: Library (AGPL-3.0 / Commercial); Pre-trained MTG weights (CC BY-NC-SA 4.0).


* **Madmom:** Böck, S., et al. (2016). *Madmom: A new Python Audio Processing and Music Information Retrieval Library.* ACM Multimedia.
* Documentation & Repo: [github.com/CPJKU/madmom](https://github.com/CPJKU/madmom) | [madmom.readthedocs.io](https://madmom.readthedocs.io/).
* License: BSD 2-Clause.


* **Basic Pitch:** Bittner, R. M., et al. (2022). *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation.* ICASSP 2022.
* Source Code: [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch).
* License: Apache-2.0.


* **Demucs v4:** Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation.* arXiv:2111.12203.
* Source Code: [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs).
* License: MIT.



### NLP & Lyric Representation Models

* **BGE-M3:** Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216.
* Source Code & Model Hub: [github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | Hugging Face: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3).
* License: MIT.


* **Jina Embeddings v3:** Günther, M., et al. (2024). *jina-embeddings-v3: Multilingual Embeddings with Task-Specific LoRA Adapters.*
* Model Card: Hugging Face: [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3).
* License: Apache-2.0.


* **LyricSIM Benchmark:** Benito-Santos, A., Ghajari, A., Hernández, P., Fresno, V., Ros, S., & González-Blanco, E. (2023). *LyricSIM: A novel dataset and benchmark for similarity detection in Spanish song lyrics.* arXiv:2306.01325.
* Primary Paper: [arXiv:2306.01325](https://arxiv.org/abs/2306.01325).
* License: CC BY-SA 4.0.


* **4MuLA Dataset:** da Silva, F. F., Silva, D. F., & Marcacini, R. M. (2020). *4MuLA: A Multilingual and Multimodal Music Dataset for Music Information Retrieval.*
* Primary Paper: [doi:10.1007/978-3-030-61377-8_4](https://www.google.com/search?q=https://doi.org/10.1007/978-3-030-61377-8_4).


* **GoEmotions:** Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020.
* Model Checkpoint: Hugging Face: [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions).
* License: Apache-2.0.


* **BERTopic:** Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
* Source Code: [github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic).
* License: MIT.