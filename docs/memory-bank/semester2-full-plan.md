# Semester 2 Deep Learning Plan

CRITICAL NOTE:
 THIS is the external outdated reference for the Semester 2 plan. DO NOT blindly follow this - check ROADMAP.md for the current plan. This file is preserved for archival reference but may not reflect the latest priorities or adjustments made during the semester.

## Music Prediction Project — Extension of HitMusicNet (IEEE Access, 2020)

---

## Project Background

### What exists from Semester 1
- Dataset: 550,622 songs with 414 features saved at data/processed/songs.csv
- Feature arrays saved at ml/features/*.npy
- Experiment results at results/metrics/
- Analysis notebooks numbered 01 through 07
- Best results: Energy R²=0.81, Danceability R²=0.55, Valence R²=0.45, Popularity R²=0.13
- Best models: CatBoost and XGBoost via gradient boosting

### Reference Paper We Are Extending
- Title: A Multimodal End-to-End Deep Learning Architecture for Music Popularity Prediction
- Authors: Martín-Gutiérrez et al.
- Published: IEEE Access, 2020
- DOI: 10.1109/ACCESS.2020.2976033
- Their dataset: 101,939 songs (ours is 5.4x larger)
- Their targets: popularity only (we predict 4 targets)
- Their architecture: feature extraction → autoencoder compression → MLP prediction
- Their audio: 30-second Spotify preview clips processed with classical signal processing
- Their text: 6 hand-crafted stylometric features from lyrics
- Their best result: MAE of 0.0855 (normalized), classification F1 of 83%
- Their stated future work: add CNN for audio and word embeddings for text — this is exactly what we implement

### Our Narrative
We replicate HitMusicNet on a 5.4x larger dataset, then implement the improvements the authors said they would do next: replacing their 6 stylometric text features with DistilBERT embeddings, and adding modern audio embeddings alongside classical features. We also extend from single-target to multi-task prediction across 4 targets simultaneously.

---

## Target Improvements

| Target | Semester 1 | After Phase 1 | After Phase 2 | After Phase 3 | Final Goal |
|---|---|---|---|---|---|
| Energy | R² 0.81 | 0.82 | 0.83 | 0.86 | 0.88+ |
| Danceability | R² 0.55 | 0.57 | 0.60 | 0.63 | 0.66+ |
| Valence | R² 0.45 | 0.47 | 0.58 | 0.62 | 0.65+ |
| Popularity | R² 0.13 | 0.16 | 0.21 | 0.23 | 0.27+ |

Valence benefits most from lyrics DL because emotional content in lyrics is directly measurable.
Popularity is the hardest target — do not be discouraged if it stays low.

---

## Tech Stack

Install in this order, one phase at a time — do not install everything upfront:

- Phase 0: torch, wandb, scikit-learn, pandas, numpy
- Phase 1: librosa, torchaudio, spotipy, requests
- Phase 2: transformers, sentence-transformers, datasets, vaderSentiment, pronouncing
- Phase 3: no new installs needed
- Phase 4: optuna
- Phase 5: gradio, shap

---

## Directory Structure to Create

```
project/
├── data/
│   ├── processed/songs.csv              (existing)
│   ├── audio/previews/                  (30-sec MP3 files, populated in Phase 3)
│   └── embeddings/                      (saved numpy arrays, one file per phase)
├── ml/features/                         (existing)
├── dl/
│   ├── phase0_foundations/
│   ├── phase1_replicate_paper/
│   ├── phase2_lyrics_dl/
│   ├── phase3_audio_dl/
│   ├── phase4_fusion/
│   └── phase5_research/
├── models/checkpoints/                  (saved .pt model weight files)
├── results/dl_metrics/                  (new DL experiment results)
└── app/                                 (Gradio demo)
```

---

## Phase 0: Foundations
### Weeks 1–2

### Goal
Learn PyTorch fundamentals using the tabular data you already know well.
Do not skip this phase. Building on familiar data prevents confusion in all later phases.
By the end of this phase you should be able to write a complete training loop from scratch without looking anything up.

### Learning Resources (complete before writing any code)
- Watch Andrej Karpathy's "Neural Networks: Zero to Hero" series on YouTube — focus specifically on the micrograd and makemore episodes
- Read the official PyTorch beginner tutorial at pytorch.org/tutorials/beginner/basics/intro.html
- Concepts to understand before moving on: tensors, forward pass, loss function, backpropagation at a high level (you do not need to derive it), gradient descent, what a learning rate controls, what a batch is, what an epoch is

### Week 1 Tasks
- Install PyTorch following the official guide for your OS and whether you have a GPU
- Create notebook dl/phase0_foundations/00_pytorch_basics.ipynb
- In this notebook: create tensors, practice basic math operations, understand shapes, move data to GPU if available
- Build a tiny network that learns the XOR problem (4 data points, 2 inputs, 1 output) — train it until loss reaches near zero
- This confirms your training loop is correctly implemented before touching real data

### Week 2 Tasks
- Set up Weights and Biases for experiment tracking — go to wandb.ai, create a free account, run wandb login in terminal — this takes 30 minutes and prevents losing experimental results for the entire semester
- Create notebook dl/phase0_foundations/01_tabular_mlp.ipynb
- Load songs.csv, use the 414 features as input, use energy/danceability/valence/popularity as the 4 outputs
- Normalize all 4 targets to the range 0 to 1 before training — energy/danceability/valence are already in this range, divide popularity by 100
- Normalize input features using StandardScaler from scikit-learn — fit on train set only, transform both train and test
- Split data 80% train and 20% test using a fixed random seed
- Build a 3-hidden-layer MLP where layer sizes decrease as input_size → input_size/2 → input_size/3 → 4 outputs — this is the exact MusicPopNet architecture from the paper, learning it here on tabular data means you already understand it when you need it again in Phase 1
- Use ReLU activation between hidden layers
- Use Sigmoid at the output layer to constrain predictions to 0–1
- Use Dropout with probability 0.5 between hidden layers
- Train with MSE loss, Adam optimizer at learning rate 0.001, batch size 256
- Add early stopping: if validation loss does not improve for 10 consecutive epochs, stop training
- Log train loss, validation loss, and R² for each of the 4 targets to W&B every epoch
- Compare your final R² against the Semester 1 CatBoost results and note the difference

### Expected Outcome
The MLP on tabular data will probably not beat CatBoost. This is normal and expected.
Gradient boosting usually outperforms MLPs on tabular data. The goal here is to learn the workflow, not to beat the baseline.

---

## Phase 1: Replicate HitMusicNet
### Weeks 3–5

### Goal
Implement the exact architecture from the paper on our 550K dataset.
This establishes a proper DL baseline tied to published research.
You should outperform the paper's results due to having 5.4x more data even without any architectural improvements.

### Spotify Preview Downloader (start Week 3, runs in background throughout)
- Create script dl/phase1_replicate_paper/spotify_downloader.py
- Register for a free Spotify Developer account at developer.spotify.com and create an app to get a client ID and client secret
- Use the spotipy Python library to authenticate
- For each track_id in songs.csv, call the Spotify API, retrieve the preview_url field, download the MP3 file to data/audio/previews/{track_id}.mp3
- About 70–80% of tracks will have preview URLs — the rest return null and should be skipped
- Add a small delay between requests to respect API rate limits
- Target at minimum 50,000 downloaded previews before starting audio ML experiments
- Storage estimate: 50,000 files × 240KB = approximately 12GB

### Audio Feature Extraction (Week 3)
- Create notebook dl/phase1_replicate_paper/02_audio_feature_extraction.ipynb
- Install librosa
- Start with a pilot subset of 5,000 songs to test the pipeline before scaling to all 550K
- For each 30-second MP3 file extract the following features using librosa with sampling rate 44100 and hop length 512:
  - MFCCs: 40 coefficients — captures timbral texture, what the instrument sounds like
  - Mel spectrogram: 127 filter bands, frequency range 50Hz to 22500Hz, FFT size 2048, Hamming window — captures time-frequency energy distribution
  - Tonnetz: 6 dimensions — captures harmonic and tonal content, chord relationships
  - Chromagram: 12 dimensions — captures which musical pitch classes (notes) are present
  - Spectral Contrast: 7 sub-bands — ratio of harmonic vs non-harmonic components
  - Spectral Centroid: 1 value — represents the brightness or frequency center of mass of the sound
  - Spectral Bandwidth: 1 value — width of the frequency distribution around the centroid
  - Zero Crossing Rate: 1 value — how often the signal crosses zero, indicates noisiness vs tonality
- For every feature that has a time dimension, compute both the mean and standard deviation across all time frames — this collapses each clip to a fixed-length flat vector regardless of clip duration
- Concatenate all computed statistics into one flat numpy array per song
- Save the complete array to data/embeddings/classical_audio_features.npy with shape (N, ~390)
- For songs where no audio file exists, fill that row with zeros

### Text Stylometric Features (Week 3)
- Create notebook dl/phase1_replicate_paper/03_text_stylometric_features.ipynb
- Use the lyrics column from songs.csv
- Implement these 6 features exactly as described in the paper:
  1. Total number of sentences — count non-empty lines in the lyric
  2. Average number of words per sentence
  3. Total number of words
  4. Average number of syllables per word — use a simple vowel-counting heuristic
  5. Sentence similarity coefficient — for all pairs of sentences compute TF-IDF cosine similarity, count the fraction of pairs above 0.75 threshold; ranges 0 (all unique) to 1 (all identical); measures how repetitive the lyrics are
  6. Vocabulary wealth coefficient — rank words by frequency, find the words that cover 85% of the cumulative frequency, divide that count by total word count; higher values mean more diverse vocabulary
- Save results as a numpy array with shape (550622, 6)
- For songs with missing lyrics fill with zeros

### Autoencoder MusicAENet (Week 4)
- Create notebook dl/phase1_replicate_paper/04_musicaenet_autoencoder.ipynb
- Build the autoencoder in PyTorch
- Input: the full concatenated feature vector (Spotify features + metadata + classical audio + stylometric text)
- Encoder architecture: two hidden layers at sizes d/2 and d/3, then a bottleneck layer at d/5 — where d is the input dimension
- Decoder architecture: mirrors the encoder in reverse order
- The compression ratio delta=1/5 is the paper's best configuration — also test delta=1/4 and delta=1/7 for comparison
- Train with MSE loss (reconstruction error) — the autoencoder just learns to reconstruct its own input
- Use Adam optimizer at learning rate 0.001
- Train until reconstruction loss is around 1e-5 — the paper reports this as a good convergence target
- Save the trained encoder weights to models/checkpoints/musicaenet_encoder.pt
- After training, run all songs through the encoder and save the compressed representations to data/embeddings/compressed_features.npy — you will use this file as input to the MLP

### MLP MusicPopNet (Week 4–5)
- Create notebook dl/phase1_replicate_paper/05_musicpopnet_mlp.ipynb
- Load the compressed features from data/embeddings/compressed_features.npy
- Build MusicPopNet using the same architecture from your Phase 0 MLP (it is the same architecture — 3 hidden layers decreasing in size) but now on compressed inputs and with 4 output neurons
- Try these optimizers and log each to W&B: Adam, Adadelta (the paper's best), Nadam
- Try these weight initializations: Xavier uniform (the paper's best), Xavier normal, He uniform, He normal
- Try these dropout rates: 0.25, 0.5, 0.75
- Use early stopping with patience of 10 epochs, maximum 100 epochs, batch size 256
- Report final R² for all 4 targets on the test set — this is your Phase 1 baseline

### Expected Outcome
This gives you a legitimate DL baseline tied to a published paper.
Your results should exceed the paper's because of more data.
The paper achieved MAE of 0.0855 on normalized popularity — use this as your comparison point.

---

## Phase 2: Lyrics Deep Learning
### Weeks 6–9

### Goal
Replace the paper's 6 stylometric text features with transformer-based embeddings.
This is where the biggest performance improvements are expected, especially for valence and popularity.
The paper explicitly listed adding word embeddings as their planned next step — you are completing their roadmap.

### MiniLM Baseline (Week 6)
- Create notebook dl/phase2_lyrics_dl/06_minilm_baseline.ipynb
- Install sentence-transformers library
- Load the model called all-MiniLM-L6-v2 from sentence-transformers — no training is required, you use it out of the box
- This model encodes any text into a 384-dimensional vector that captures semantic meaning
- Encode all lyrics from songs.csv in batches — a batch size of 256 works well
- Save the result to data/embeddings/minilm_lyrics.npy with shape (550622, 384)
- For songs with missing lyrics use a zero vector
- Concatenate these embeddings with your metadata features and train MusicPopNet on the combined vector
- Compare R² to your Phase 1 baseline — note which targets improve most
- This step alone should show a meaningful improvement on valence

### Fine-tune DistilBERT (Week 7)
- Create notebook dl/phase2_lyrics_dl/07_distilbert_finetune.ipynb
- Install HuggingFace transformers library
- Load distilbert-base-uncased from HuggingFace — DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of BERT's performance, making it the right choice for a beginner without large GPU resources
- Tokenize all lyrics using the DistilBERT tokenizer — truncate at 512 tokens, pad shorter sequences
- Add a regression head on top of the model: a dropout layer followed by a linear layer from 768 dimensions down to 4 output neurons, with Sigmoid activation at the end
- Fine-tune end-to-end for 3 to 5 epochs — BERT-based models need very few epochs, more will overfit
- Use AdamW optimizer at learning rate 2e-5 with weight decay 0.01 — these are standard BERT fine-tuning values
- Use batch size 16 or 32 depending on your available GPU memory
- The model predicts all 4 targets simultaneously from lyrics alone — this is multi-task fine-tuning
- Use the CLS token output (first token position) as the song-level representation
- Save the final model weights to models/checkpoints/distilbert_lyrics.pt
- Extract and save the raw 768-d CLS embeddings (before the regression head) to data/embeddings/bert_lyrics.npy — saving these means you never need to run BERT inference again in later phases
- Log training loss and validation R² per target per epoch to W&B
- If you do not have a GPU: use Google Colab with a free T4 GPU, or reduce the dataset to 100,000 songs for fine-tuning

### Lyric Structure Features (Week 8)
- Create notebook dl/phase2_lyrics_dl/08_lyric_structure_features.ipynb
- Install vaderSentiment and pronouncing libraries
- These features go beyond what the paper computed and represent your original contribution to text analysis
- Compute the following for each song's lyrics:
  - Rhyme density: fraction of adjacent line pairs where the last words rhyme — use the pronouncing library to check phonetic endings
  - Sentiment mean: average VADER compound sentiment score across all lines, ranging from -1 (most negative) to +1 (most positive)
  - Sentiment standard deviation: how much the sentiment fluctuates between lines — high variance means the song has emotional contrast
  - Sentiment range: difference between the most positive and most negative single line
  - Repetition ratio: fraction of lines that appear more than once in the entire lyric — the paper found repetitive lyrics correlate with popularity
  - Word uniqueness: unique word count divided by total word count — lower means more repetitive vocabulary
- Save as a numpy array with shape (550622, 6)
- Try combining these with BERT embeddings and check whether they add predictive signal beyond what BERT already captures — if R² does not improve when adding them on top of BERT, BERT has already learned the same information implicitly

### Lyrics Ablation Study (Week 9)
- Create notebook dl/phase2_lyrics_dl/09_lyrics_ablation.ipynb
- Run and record R² for every configuration listed below, all 4 targets, test set only:
  1. Metadata only (Semester 1 CatBoost baseline — reference point, already done)
  2. Metadata + paper's 6 stylometric features (Phase 1, already done)
  3. Metadata + MiniLM 384-d embeddings (Week 6, already done)
  4. Metadata + fine-tuned DistilBERT 768-d embeddings (Week 7, already done)
  5. Metadata + DistilBERT + 6 structure features from Week 8
- Compile all results into a single clean table
- Identify which text representation helps each target most and by how much
- This table is a core results section of your academic report

---

## Phase 3: Audio Deep Learning
### Weeks 10–12

### Goal
Add modern audio representations on top of classical features.
The paper said they planned to add CNN-based audio processing — you implement this.

### Audio Data Review (Week 10)
- Check how many preview files were successfully downloaded since Week 3
- If fewer than 50,000 files are available, continue downloading before proceeding with audio DL experiments
- As a fully open-license alternative: download FMA-medium from freemusicarchive.org — 25,000 songs, 22GB total, Creative Commons licensed with no legal concerns — this is a clean benchmark dataset
- FMA is useful for experimenting even if you also use Spotify previews for the main experiments

### Pre-trained Audio Embeddings with VGGish (Week 10–11)
- Create notebook dl/phase3_audio_dl/11_vggish_embeddings.ipynb
- VGGish is a model trained by Google on millions of YouTube audio clips — it outputs 128-dimensional embeddings per second of audio and is freely available
- Load each audio file at 16kHz (VGGish's required sample rate), run through the model, mean-pool the per-second embeddings to get one 128-d vector per clip
- Save all embeddings to data/embeddings/vggish_audio.npy with shape (N, 128)
- For songs with missing audio files fill the corresponding row with zeros
- Concatenate VGGish embeddings with metadata and BERT lyrics embeddings, then train MusicPopNet on the combined vector
- Record R² improvements over Phase 2 results

### Pre-trained Audio Embeddings with MERT (Week 11)
- MERT is a music-specific model trained with self-supervised learning on large music collections — unlike VGGish which was trained on general YouTube audio, MERT was designed specifically for music understanding tasks
- Model identifier on HuggingFace: m-a-p/MERT-v1-95M
- Load each audio file at 24kHz (MERT's required sample rate), process through the model, mean-pool over the time dimension to get one 768-d vector per clip
- Save all embeddings to data/embeddings/mert_audio.npy with shape (N, 768)
- Compare VGGish (128-d, general audio) against MERT (768-d, music-specific) — MERT should outperform VGGish on music-specific targets like energy and danceability
- Note: MERT inference is slower than VGGish — run on a GPU if available

### Mel Spectrogram CNN (Week 11–12)
- Create notebook dl/phase3_audio_dl/12_mel_spectrogram_cnn.ipynb
- Convert each 30-second clip to a mel spectrogram: use 128 mel frequency bands, hop length 512, sampling rate 44100 as the paper does, FFT size 2048
- Apply log-power scaling (convert to decibels) to make the dynamic range manageable for the neural network
- The spectrogram is a 2D image with dimensions approximately 128 × 1292 for a 30-second clip — the CNN treats this exactly like a photograph
- Load a pre-trained ResNet-18 from torchvision.models — this was pre-trained on ImageNet photographs but we adapt it to spectrograms via fine-tuning
- Replace the final classification layer with a regression head outputting 4 values with Sigmoid activation
- Fine-tune for 10 to 20 epochs using a small learning rate of 1e-4 — small because we are adjusting pre-trained weights rather than training from scratch
- This is your first CNN training experience — the spectrogram CNN often captures features that classical signal processing features cannot represent

### Audio Ablation Study (Week 12)
- Create notebook dl/phase3_audio_dl/13_audio_ablation.ipynb
- Train and record R² for all 4 targets for each configuration:
  1. Classical audio features only (the paper's approach, done in Phase 1)
  2. Classical features + VGGish 128-d embeddings
  3. Classical features + MERT 768-d embeddings
  4. Fine-tuned ResNet-18 CNN on mel spectrograms
  5. Classical features + MERT + CNN (best audio combination)
- Energy and danceability should benefit most from audio — confirm this
- Valence should benefit more from lyrics than audio — confirm this
- These patterns will support the modality contribution analysis in Phase 4

---

## Phase 4: Multimodal Fusion
### Weeks 13–14

### Goal
Combine all three modalities into one unified system.
This is the most novel contribution and the main story of any conference paper submission.

### Late Fusion (Week 13)
- Create notebook dl/phase4_fusion/14_late_fusion.ipynb
- Train three separate best models using only one modality each: metadata model, lyrics model, audio model
- Combine their test-set predictions using a weighted average — start with equal weights (1/3 each), then optimize the weights on a validation set
- Also train a small stacking model: a linear regression or tiny MLP that takes all three models' predictions as input and outputs final predictions — this is called a meta-learner or stacking
- Record R² for all 4 targets — this is your first true multimodal result
- This approach is called late fusion because predictions are combined only at the end

### Intermediate Fusion (Week 13)
- Create notebook dl/phase4_fusion/15_intermediate_fusion.ipynb
- Concatenate all learned representations into one large vector per song: metadata features + BERT lyrics embeddings (768-d) + MERT audio embeddings (768-d) + classical audio features
- Pass this large concatenated vector through the MusicAENet autoencoder for compression — use the same delta=1/5 compression ratio from Phase 1
- Then train MusicPopNet on the compressed representation
- This directly mirrors the HitMusicNet paper architecture but with far richer inputs: BERT instead of 6 stylometric features, and MERT in addition to classical audio
- Compare to late fusion results — the better approach depends on how much shared information exists across modalities
- This approach is called intermediate fusion because the modalities are combined before the final prediction layer

### Modality Contribution Analysis (Week 14)
- Create notebook dl/phase4_fusion/16_modality_contribution_analysis.ipynb
- For each of the 4 targets, measure how much each modality contributes by ablation:
  - Train the full fusion model, then remove one modality at a time and measure how much R² drops
  - The modality whose removal causes the largest R² drop is the most important for that target
- Expected findings based on domain knowledge:
  - Energy: audio-dominant — tempo, loudness, and spectral features directly measure energy
  - Valence: lyrics-dominant — emotional sentiment and word choice drive valence more than audio does
  - Danceability: mixed audio and metadata — tempo and rhythm are in audio, but Spotify's own danceability feature is already in metadata
  - Popularity: metadata-dominant — artist follower count, number of available markets, and artist popularity score are strong predictors that no amount of audio or lyrics can replace
- Create a visualization showing modality importance per target — this becomes a figure in your report

### Full Comparison Table (Week 14)
- Create notebook dl/phase4_fusion/17_full_ablation_table.ipynb
- Compile every experimental result from Phases 1 through 4 into one master table
- Rows: every configuration tested across all phases
- Columns: R² for energy, danceability, valence, popularity
- Include the paper's original results as a reference row (convert their MAE to R² if possible using the test set distribution)
- Mark clearly which configuration beats the paper and by how much
- This table is the centerpiece of your academic report

---

## Phase 5: Research, Writing, and Demo
### Weeks 15–16

### Goal
Package everything for maximum academic and public impact.

### Gradio Demo App (Week 15)
- Create app/gradio_demo.py
- Install gradio
- User inputs: song name and artist name as free text fields
- Backend behavior: use spotipy to look up the song on Spotify and retrieve its features, then run through your best fusion model to generate predictions
- Display: predicted energy, danceability, valence, and popularity as percentage bars or gauges
- Also display which modality was most influential for each prediction, based on your Phase 4 contribution analysis
- Deploy to HuggingFace Spaces for free public hosting — create an account at huggingface.co, follow their Spaces deployment guide for Gradio apps
- Add the public demo link to your Kaggle dataset description and GitHub README

### Kaggle Notebook Update (Week 15)
- Publish a new public notebook on your existing 550K dataset page
- Frame it as: from gradient boosting to deep learning on the same dataset
- Show the complete ablation table from Phase 4 in a clean format
- Briefly describe each phase's approach and what it contributed
- Include a link to the live Gradio demo
- Your existing Bronze medal and dataset visibility will drive traffic to this new notebook

### Semester 2 Report Structure (Week 16)
Write the report in this order, one section per day:

1. Introduction — motivate why multimodal music prediction matters, state the problem of predicting popularity/energy/valence/danceability, introduce your approach, cite the HitMusicNet paper as the baseline you extend
2. Related Work — summarize HitMusicNet in one paragraph (dataset, architecture, results, limitations), mention 2–3 other papers on music feature prediction or multimodal learning in music
3. Dataset — describe your 550K dataset: sources, features, target distributions, comparison to the paper's 101K dataset
4. Methodology — describe each phase: classical audio features (what they are and why), BERT lyrics fine-tuning (what pre-trained means, why DistilBERT), modern audio embeddings (VGGish vs MERT), autoencoder compression (why it helps), fusion strategies
5. Experiments — present the master ablation table, describe what each row adds
6. Discussion — explain why valence improves most from lyrics, why energy improves most from audio, why popularity remains the hardest target, why the modality contribution patterns match musical intuition
7. Conclusion — summarize improvements over the paper, state limitations, describe natural future directions

Key opening sentence for the report: "We replicate and extend the HitMusicNet architecture of Martín-Gutiérrez et al. (2020) on a 5.4× larger dataset, implementing the CNN audio and word embedding improvements they identified as future work, and extending the task from single-target popularity prediction to multi-task prediction of four musical attributes simultaneously."

### Conference Paper Decision (Week 16)
Submit a conference paper if any of these thresholds are met:
- Valence R² exceeds 0.60 — current literature typically sits around 0.45 to 0.55
- Popularity R² exceeds 0.25 — this target is notoriously difficult, 0.25 would be a notable result
- Any target improves over the paper's comparable result by more than 0.05 R²

Target venues if results are strong enough:
- ISMIR 2026 — International Society for Music Information Retrieval, the top dedicated venue for this research area, check current year's deadline
- IEEE Transactions on Multimedia — the same journal where the reference paper was published, making your work a direct successor paper
- ICASSP 2026 — IEEE International Conference on Acoustics Speech and Signal Processing, large respected venue with an audio track

If results do not meet the conference threshold, the Semester 2 report plus the Kaggle notebook plus the GitHub repository are still strong deliverables for an undergraduate or graduate student.

---

## Weekly Schedule Summary

| Weeks | Phase | Primary Focus |
|---|---|---|
| 1–2 | Phase 0: Foundations | PyTorch, training loop, tabular MLP, W&B setup |
| 3–5 | Phase 1: Replicate paper | Classical audio features, stylometric text, autoencoder, MusicPopNet |
| 6–9 | Phase 2: Lyrics DL | MiniLM baseline, DistilBERT fine-tuning, structure features, ablation |
| 10–12 | Phase 3: Audio DL | Preview downloads, VGGish, MERT, mel spectrogram CNN, ablation |
| 13–14 | Phase 4: Fusion | Late fusion, intermediate fusion, modality contribution analysis |
| 15–16 | Phase 5: Research | Gradio demo, Kaggle update, report writing, conference paper decision |

---

## Tool Learning Order

Learn and install tools in this order — do not install everything at once:

1. PyTorch — weeks 1–2, the foundation of all deep learning work in this project
2. W&B — week 2, 30-minute setup, use continuously from then on for all experiments
3. scikit-learn — week 2, for StandardScaler, train/test split, R² scoring
4. librosa — week 3, for classical audio feature extraction from MP3 files
5. spotipy — week 3, for Spotify API access and preview MP3 downloading
6. sentence-transformers — week 6, the easiest BERT wrapper, requires no training to use
7. HuggingFace transformers — week 7, for loading and fine-tuning DistilBERT
8. vaderSentiment — week 8, for line-by-line sentiment scoring of lyrics
9. pronouncing — week 8, for phonetic rhyme detection in lyrics
10. optuna — week 13, for hyperparameter search on fusion model configurations
11. gradio — week 15, for building the demo app, very beginner-friendly

---

## Key Architectural Decisions and Why

### Why DistilBERT and not BERT-base or RoBERTa?
DistilBERT is 40% smaller and 60% faster than BERT-base while keeping 97% of BERT's performance.
For a beginner with limited GPU resources and a tight semester timeline, DistilBERT is the right trade-off.
RoBERTa is marginally stronger but slower and harder to fine-tune without more experience.

### Why VGGish and MERT instead of Wav2Vec 2.0?
VGGish is simple to use, outputs compact 128-d embeddings, and was trained on a huge diverse audio collection.
MERT is music-specific (trained on music, not speech like Wav2Vec), outputs richer 768-d embeddings.
Wav2Vec 2.0 was designed for speech recognition — it understands spoken language, not music structure.

### Why 30-second Spotify preview clips and not full songs?
The reference paper also used 30-second previews — this makes our results directly comparable.
Storage: 550K clips at 30 seconds each = approximately 132GB, well within a 1TB budget.
Full songs would exceed 1TB and downloading them from Spotify violates their Terms of Service.
The 30-second clips are also legally available through the public Spotify API preview URLs.

### Why use an autoencoder before the MLP?
The paper showed that training the MLP directly on the raw high-dimensional feature vector causes overfitting.
Compressing to 1/5 of the original size acts as regularization — it forces the model to keep only the most useful information.
The paper tested delta=1/4, 1/5, and 1/7 and found 1/5 gave the best performance — start with this value.

### Why predict all 4 targets simultaneously instead of 4 separate models?
Energy, danceability, valence, and popularity share musical signal — an upbeat song tends to be energetic and danceable simultaneously.
Multi-task learning lets the network share internal representations across targets, which helps especially for targets with less signal (popularity).
It is also 4x faster to train than four separate models.
If a specific target underperforms in the multi-task setup, you can train a single-target specialist model for it later.

---

## Evaluation Metrics

Use these consistently across all experiments so all results are comparable:

- Primary metric: R² (coefficient of determination) — easy to interpret, ranges from negative infinity to 1.0, was used in Semester 1 so you have direct comparison points
- Secondary metric: MAE (mean absolute error on normalized targets, i.e., targets divided to be in range 0–1) — used in the paper, allows direct comparison of your results to their published numbers
- Secondary metric: MSE (mean squared error on normalized targets) — also used in the paper for their tables

For the autoencoder only: use MSE as reconstruction error (R² does not apply to unsupervised reconstruction).

Always report results on the held-out test set (20% of data that was set aside before any training began).
Never tune hyperparameters using the test set — use a validation set split from the training data for that purpose.

---

## Common Beginner Mistakes to Avoid

1. Not normalizing targets — always scale all 4 targets to the range 0–1 before training any neural network, otherwise loss values will be on incompatible scales
2. Not normalizing input features — always apply StandardScaler to input features, fit only on the training set and apply to both train and test
3. Overfitting by training too long — use dropout and early stopping in every model, no exceptions
4. Recomputing embeddings from scratch each experiment — save BERT embeddings, VGGish embeddings, and MERT embeddings to .npy files immediately after computing them; recomputing takes hours and is unnecessary
5. Forgetting model.eval() during evaluation — this disables dropout during inference; without it your evaluation metrics will be artificially noisy
6. Tuning hyperparameters on the test set — use a separate validation split for all tuning decisions, only look at test set results for final reporting
7. Not logging experiments — run wandb.init at the start of every notebook from week 2 onward; it is very easy to lose track of which hyperparameter combination gave which result
8. Skipping Phase 0 or Phase 1 to jump straight to BERT — the foundations matter; doing the paper replication first gives you a baseline and teaches the architecture you will use throughout
9. Installing all libraries at once — follow the phase-by-phase install order; conflicting library versions are a common source of painful debugging
10. Expecting popularity prediction to be high — it is genuinely hard due to external cultural and social factors; if you reach R²=0.20 that is a good result, 0.25+ would be excellent

---

## End of Plan