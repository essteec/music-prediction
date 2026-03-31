# Deep Learning Roadmap - Semester 2

## 🎯 Goal
Beat Semester 1 ML baseline using Deep Learning neural networks

## 📊 Current Phase
Phase 1A - MPNet Embeddings ✅ **COMPLETE**

---

## ✅ Semester 1 - ML Baseline (COMPLETE)

### Achievements
- **Dataset**: 550,622 English songs with 414 features
- **Features**: 23 audio + 5 text stats + 2 sentiment + 384 embeddings (MiniLM)
- **Preprocessing**: All .npy files ready at `ml/features/`
- **Splits**: Artist-aware train/val/test (zero artist overlap)
- **Models**: 28+ algorithms compared (CatBoost, XGBoost, LightGBM dominated)

### Baseline to Beat (Test Set R²)
| Target | Best Model | R² | Status |
|--------|-----------|-----|--------|
| **Energy** | CatBoost_tuned | **0.81** | Strong baseline |
| **Danceability** | XGBoost_tuned | **0.55** | Moderate baseline |
| **Valence** | XGBoost_tuned | **0.45** | Weak - main opportunity |
| **Popularity** | CatBoost | **0.13** | Very weak - hard target |

### Available Data (Ready to Use)
```
ml/features/
├── X_train_audio.npy          (374997, 23)
├── X_train_text_stats.npy     (374997, 5)
├── X_train_sentiment.npy      (374997, 2)
├── X_train_embeddings.npy     (374997, 384)
├── y_train_{target}.npy       (4 target files)
└── Same structure for val and test splits

Total: 414 features ready for DL training
```

---

## 🚀 Semester 2 - Deep Learning Extension

**Strategy**: Start simple, iterate improvements based on semester2-full-plan.md inspiration

**Reference**: HitMusicNet paper (Martín-Gutiérrez et al., IEEE Access 2020) provides architectural ideas

**Our Approach**: Beat our ML baseline using DL, improve incrementally

**Success Target**: Ambitious - beat ML baseline on all 4 targets

---

## Phase 0: PyTorch MLP Baseline ✅ COMPLETE

**Goal**: Establish PyTorch baseline using existing 414 features, prove DL can match/beat ML

**Status**: ✅ Complete (March 30-31, 2026)

### Final Results (Test Set R²)
| Target | ML R² | DL R² (Unweighted) | DL R² (Weighted) | Improvement | Gap to ML |
|--------|-------|-------------------|-----------------|-------------|-----------|
| Valence | 0.45 | 0.27 | **0.35** | +0.08 | -0.10 |
| Energy | 0.81 | 0.71 | **0.75** | +0.04 | -0.06 |
| Danceability | 0.55 | 0.37 | **0.47** | +0.10 | -0.08 |
| Popularity | 0.13 | 0.11 | **0.12** | +0.01 | -0.01 |
| **Average** | **0.49** | **0.36** | **0.42** | **+0.06** | **-0.07** |

**Key Findings:**
- ✅ Weighted loss improved performance by 16.7% (0.36 → 0.42 avg R²)
- ✅ DL ~7 points behind ML (expected for tabular data)
- ✅ Multi-task learning works - no gradient domination
- ✅ Reproducibility achieved with seed=42

### Implementation Details

**Architecture**: MusicMLP (115K parameters)
```python
# CORRECTED: No final activation (raw regression)
Input (414) → FC1 (207) → ReLU → Dropout(0.5) 
           → FC2 (138) → ReLU → Dropout(0.5)
           → FC3 (4) → No activation
```

**Training Configuration:**
- Loss: MSELoss with weighted reduction `[2.0, 1.0, 2.0, 0.5]`
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Early stopping: patience=10 (stopped at epoch 32)
- Seed: 42 (deterministic, reproducible)
- Device: CUDA (GPU)

**Files Created:**
```
dl/
├── 01_xor_network.py          # XOR learning exercise
├── 02_train_mlp.py             # Main MLP trainer (weighted multi-task)
├── test_setup.py               # Setup verification
├── README.md                   # Phase 0 documentation
└── utils/
    ├── data_loaders.py         # Deterministic PyTorch Dataset
    ├── models.py               # MusicMLP (no final activation)
    ├── metrics.py              # R², RMSE, MAE
    └── reproducibility.py      # Centralized seed management

models/checkpoints/
└── mlp_baseline_best.pt        # Best model (epoch 22)

results/dl_metrics/
├── mlp_baseline_20260331_022237.csv  # Unweighted results
└── mlp_baseline_20260331_024129.csv  # Weighted results (FINAL)
```

### Critical Bugs Fixed

1. **Sigmoid vs log1p Mismatch**: Popularity transformed to [0-4.6] but Sigmoid constrains [0-1] → Removed final activation
2. **Reproducibility**: Random seeds not set → Added set_seed(42) everywhere
3. **Multi-task Imbalance**: Popularity dominated gradient → Weighted loss balancing
4. **PyTorch 2.6 Change**: torch.load() broke → Added weights_only=False

### Lessons Learned

- **Gradient Boosting dominates tabular data**: Expected that engineered features favor ML
- **DL needs better representations**: Next phases focus on learned embeddings (BERT, audio)
- **Multi-task requires balance**: Different target scales need weighted loss
- **Reproducibility is critical**: Always set seeds for research methodology

### Success Criteria
- ✅ **Minimum**: MLP R² within 10% of ML baseline → Achieved (7% gap)
- ✅ **Reproducibility**: Deterministic results every run → Achieved
- ⚠️ **Target**: Beat ML on at least 1 target → Not achieved (expected for tabular MLP)

**Next Phase**: Phase 1 - BERT fine-tuning to improve text representations

---

## Phase 1A: MPNet Embeddings ✅ COMPLETE

**Goal**: Replace MiniLM-L6-v2 (384-d) with MPNet (768-d) for better semantic understanding

**Status**: ✅ Complete (March 31, 2026)

### Results Achieved

**MPNet Embeddings Extracted:**
- Train: 374,997 songs → 1098.6 MB
- Val: 89,172 songs → 261.2 MB  
- Test: 86,453 songs → 253.3 MB
- **Total**: 1.6GB of 768-d embeddings for all 550K songs

**Model**: sentence-transformers/all-mpnet-base-v2
- 768-d embeddings (2× Phase 0's 384-d MiniLM)
- Better semantic understanding from Microsoft's training
- Frozen (no fine-tuning needed initially)

**Technical Details:**
- Batch size: 32 (6GB VRAM conservative)
- Processing: 4 hours runtime, ~1.09 it/s
- Checkpoint logic: Rerun-safe
- Script: `dl/03_extract_better_embeddings.py` (207 lines)

### Key Decisions

1. **MPNet over GTE**: GTE had CUDA compatibility issues, MPNet proven
2. **Frozen first**: Test improvement before fine-tuning investment  
3. **Feature count**: 798 total (23 audio + 5 text + 2 sentiment + 768 MPNet)

### Expected Improvement
- Valence R²: 0.45 → 0.50+ (better emotion understanding from lyrics)
- Energy/Danceability: Minimal change (not lyric-dependent)

### Files Created
```
dl/03_extract_better_embeddings.py       # Extraction script
data/embeddings/mpnet_lyrics_768d_*.npy  # 3 files (train/val/test)
```

### Next: Phase 1B
Train MLP with MPNet embeddings and compare to Phase 0 baseline

---

## Phase 1B: Train MLP with MPNet (NEXT)

**Goal**: Train MusicMLP on 798 features (798 vs 414 in Phase 0) and compare performance

### Approach
- MiniLM embeddings are frozen (384-d, zero-shot)
- No task-specific tuning for music prediction

### Approach 1: Fine-tune DistilBERT

**Script**: `dl/03_finetune_distilbert.py`
```python
from transformers import DistilBertModel, DistilBertTokenizer

# Why DistilBERT: 40% smaller, 60% faster than BERT, 97% performance
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Add regression head: Dropout → Linear(768 → 4) → Sigmoid
# Multi-task fine-tuning: all 4 targets from lyrics alone
# 3-5 epochs (BERT overfits quickly)
# AdamW lr=2e-5, weight_decay=0.01
# Batch size 16-32 (GPU memory dependent)
# Use CLS token output as song-level representation

# Save:
# - Model weights: models/checkpoints/distilbert_finetuned.pt
# - 768-d embeddings: data/embeddings/bert_lyrics_768d.npy
# - Metrics: results/dl_metrics/distilbert.csv
```

**GPU Note**: Use Google Colab free T4 or reduce dataset to 100K songs if needed

### Approach 2: Lyric Structure Features

**Script**: `dl/04_extract_lyric_structure.py`
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pronouncing

# 6 new features per song:
# 1. Rhyme density: fraction of adjacent line pairs with rhyming endings
# 2. Sentiment mean: avg VADER compound score (-1 to +1)
# 3. Sentiment std: emotional contrast across lines
# 4. Sentiment range: max - min sentiment per line
# 5. Repetition ratio: fraction of lines appearing >1 time
# 6. Word uniqueness: unique words / total words

# Save: data/embeddings/lyric_structure_6d.npy (550622, 6)
```

### Evaluation
Test configurations:
1. Baseline: 414 features (Phase 0 result)
2. Replace MiniLM: metadata + audio + DistilBERT 768-d
3. Augment: 414 features + lyric structure 6-d = 420 total

**Notebook**: `notebooks/09_phase1_lyrics_analysis.ipynb`
- Compare R² across configurations
- Focus on Valence improvement
- Visualize BERT embeddings (t-SNE colored by valence)

### Expected Improvement
- Valence R²: 0.45 → 0.55+ (main target)
- Energy/Danceability: Minimal change (not lyric-dependent)
- Popularity: Slight improvement possible

---

## Phase 2: Autoencoder Compression (Optional)

**Goal**: Apply HitMusicNet's MusicAENet architecture - compress features before MLP

### Motivation
- HitMusicNet compresses features by 5× before prediction
- Reduces noise, forces model to learn compact representation
- May improve generalization

### Script: `dl/05_train_autoencoder.py`
```python
# Autoencoder architecture
class MusicAutoencoder(nn.Module):
    def __init__(self, input_size=414):
        super().__init__()
        # Encoder: 414 → 207 → 138 → 83 (bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size//3),
            nn.ReLU(),
            nn.Linear(input_size//3, input_size//5)
        )
        # Decoder: 83 → 138 → 207 → 414 (mirrors encoder)
        self.decoder = nn.Sequential(
            nn.Linear(input_size//5, input_size//3),
            nn.ReLU(),
            nn.Linear(input_size//3, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size)
        )
    
    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed

# Train until MSE reconstruction loss < 1e-5
# Save compressed representations: data/embeddings/autoencoder_compressed_83d.npy
# Then train MusicPopNet MLP on 83-d compressed features
```

### Evaluation
Compare:
- MLP on 414 raw features (Phase 0)
- MLP on 83 compressed features (Phase 2)

**Notebook**: `notebooks/10_phase2_autoencoder_analysis.ipynb`

---

## Phase 3: Audio Deep Learning (Future - Pending Data Acquisition)

**Goal**: Add audio embeddings to improve Energy/Danceability

### Current Blocker
- No audio files exist (only Spotify API metadata)
- Need to acquire actual song audio

### Acquisition Options

**Option A: yt-dlp (YouTube extraction)**
- Pros: Widely available, free
- Cons: Legal gray area, quality varies, matching songs difficult
- Storage: 150GB available (tight for 550K songs)
- Workflow: Download subset (50K songs), prioritize test set

**Option B: FMA-medium dataset**
- Pros: Legal (Creative Commons), pre-processed
- Cons: Only 25K songs, different catalog, need ID matching
- Size: 22GB download

**Option C: Skip audio embeddings**
- Focus on text/metadata improvements only
- Audio features from Spotify API may be sufficient

### IF Audio Acquired - Embedding Extraction

**Script**: `dl/06_extract_audio_embeddings.py`
```python
# VGGish: Google's general audio model (128-d)
# OR MERT: Music-specific model (768-d, better for music)

# For each audio file:
# 1. Load audio at required sample rate (16kHz VGGish, 24kHz MERT)
# 2. Extract embeddings
# 3. Mean-pool over time → one vector per song
# 4. Save: data/embeddings/audio_embeddings.npy

# Combine with existing features:
# 414 + 128 (VGGish) = 542 total features
# OR 414 + 768 (MERT) = 1182 total features
```

### Evaluation
Compare:
- Baseline: 414 features
- With VGGish: 542 features
- With MERT: 1182 features

**Expected**: Energy/Danceability improve most, Valence unchanged

---

## Phase 4: Architecture Experiments

**Goal**: Try different network architectures beyond simple MLP

### Experiments to Try

**1. Deeper Networks**
```python
# 5-layer instead of 3-layer
# 414 → 256 → 128 → 64 → 32 → 4
# Risk: overfitting (use more dropout)
```

**2. Residual Connections**
```python
# Add skip connections like ResNet
# Helps gradient flow in deeper networks
```

**3. Attention Mechanisms**
```python
# Self-attention over feature groups
# Learn to weight audio vs text vs embeddings
```

**4. Separate Encoders per Modality**
```python
# Audio encoder → audio representation
# Text encoder → text representation  
# Metadata encoder → metadata representation
# Fusion layer → combine → prediction
```

**5. Multi-task Learning Variations**
```python
# Current: Single network predicts all 4 targets
# Alternative: Shared encoder + task-specific heads
# May improve by sharing knowledge across targets
```

### Evaluation
**Notebook**: `notebooks/11_phase4_architecture_analysis.ipynb`
- Ablation study: test each architecture
- Compare training time vs performance
- Identify best architecture per target

---

## Phase 5: Final Model & Deployment

**Goal**: Package best model for production use

### Tasks

1. **Select Best Configuration**
   - Compare all phases, pick highest R² per target
   - May use different models per target (like ML did)

2. **Train on Full Data**
   - Train + Val combined (475K songs)
   - Test on held-out test set (86K songs) ONCE ONLY

3. **Update Gradio App**
   - Replace ML models in `app/gradio_app.py`
   - Load best DL models instead
   - Add modality contribution visualization

4. **Results Notebook**
   - `notebooks/12_final_dl_results.ipynb`
   - Full comparison: ML baseline vs DL final
   - Modality importance analysis
   - Error analysis: where DL helps vs hurts

5. **Documentation**
   - Update README.md with DL results
   - Document improvement per target
   - Note which phases contributed most

---

## 📦 Tech Stack

**Install as needed** (don't install everything upfront):

### Phase 0
```bash
pip install torch wandb
```

### Phase 1
```bash
pip install transformers datasets
pip install vaderSentiment pronouncing
```

### Phase 2
No new installs (uses PyTorch)

### Phase 3 (if audio acquired)
```bash
pip install librosa torchaudio
# For VGGish: tensorflow-hub
# For MERT: transformers (already installed)
```

### Phase 4
No new installs (PyTorch experiments)

### Phase 5
```bash
# Gradio already installed from Semester 1
pip install shap  # for explainability
```

---

## 📁 Directory Structure

```
dl/                              # Deep learning code
├── 01_xor_network.py           # Phase 0: Learn PyTorch
├── 02_train_mlp.py             # Phase 0: MLP baseline
├── 03_finetune_distilbert.py   # Phase 1: BERT fine-tuning
├── 04_extract_lyric_structure.py  # Phase 1: Structure features
├── 05_train_autoencoder.py     # Phase 2: Compression
├── 06_extract_audio_embeddings.py  # Phase 3: Audio (if acquired)
├── 07_train_*.py               # Phase 4: Architecture experiments
└── utils/
    ├── data_loaders.py         # Shared PyTorch data loading
    ├── metrics.py              # Evaluation utilities
    └── models.py               # Model definitions

data/
├── embeddings/                 # New DL embeddings
│   ├── bert_lyrics_768d.npy   # From Phase 1
│   ├── lyric_structure_6d.npy # From Phase 1
│   ├── autoencoder_compressed_83d.npy  # From Phase 2
│   └── audio_embeddings.npy   # From Phase 3 (if acquired)
└── audio/                      # Future: raw audio files
    └── previews/               # (pending acquisition)

models/checkpoints/             # Saved PyTorch models
├── mlp_baseline.pt
├── distilbert_finetuned.pt
├── autoencoder.pt
└── best_model.pt

results/dl_metrics/             # Training results CSVs
├── mlp_baseline.csv
├── distilbert.csv
├── autoencoder.csv
└── final_comparison.csv

notebooks/                      # Visualization only
├── 08_phase0_mlp_analysis.ipynb
├── 09_phase1_lyrics_analysis.ipynb
├── 10_phase2_autoencoder_analysis.ipynb
├── 11_phase4_architecture_analysis.ipynb
└── 12_final_dl_results.ipynb
```

---

## 🎯 Success Metrics

### Conservative (Minimum Target)
- Beat ML baseline on Valence (0.45 → 0.50+)
- Match ML on Energy/Danceability (within 2%)
- Any improvement on Popularity

### Moderate (Expected Target)
- Beat ML on Valence and Danceability
- Match ML on Energy
- Slight improvement on Popularity

### Ambitious (Goal - You Requested This)
- Beat ML baseline on ALL 4 targets
- Significant Valence improvement (0.45 → 0.55+)
- Meaningful Popularity improvement (0.13 → 0.18+)

---

## 📊 Key Conventions

### From Semester 1 (Keep Using)

1. **Artist-Aware Splits**: Zero artist overlap between train/val/test
2. **Cache Expensive Operations**: Save embeddings to .npy immediately
3. **Test Set Sacred**: Evaluate on test set ONCE at the very end
4. **Normalize Everything**: StandardScaler on inputs, targets in [0,1]
5. **Early Stopping**: Patience=10 epochs prevents overfitting
6. **Log Everything**: W&B for experiment tracking

### DL-Specific Additions

1. **Training Scripts**: Python .py files, not notebooks
2. **Visualization Notebooks**: Load results from CSVs
3. **Model Checkpoints**: Save best model weights (.pt files)
4. **Config Files**: YAML/JSON for hyperparameters (future)
5. **Modular Code**: Separate data loading, models, training logic

---

## 🔄 Iteration Strategy

**Don't try everything at once**. Work through phases sequentially:

1. **Phase 0**: Establish DL baseline, validate pipeline works
2. **Phase 1**: Focus on text (biggest opportunity)
3. **Phase 2**: Compression (optional, test if helps)
4. **Phase 3**: Audio (only if acquisition solved)
5. **Phase 4**: Architecture search (after knowing what data works)
6. **Phase 5**: Final model selection & deployment

**After each phase**:
- Compare to ML baseline (is DL winning?)
- Decide: continue to next phase OR iterate on current phase
- Update this roadmap with learnings

---

## 📝 Notes

- **semester2-full-plan.md**: External inspiration, not exact roadmap
- **This ROADMAP.md**: Actual plan specific to this project
- **Flexible timeline**: Work at your own pace, no deadlines
- **Audio optional**: Can achieve success without solving audio acquisition
- **Focus on wins**: Valence improvement via BERT is the clearest path

---

## ❓ Open Questions (To Resolve Later)

1. **Audio acquisition**: Which approach? yt-dlp vs FMA vs skip?
2. **Storage strategy**: If using yt-dlp, download full songs or clips?
3. **GPU access**: Local GPU, Colab, or CPU-only for now?
4. **Subset strategy**: Train on full 550K or subset first?
5. **Publication goal**: Still planning conference paper or just learning?

These can be decided during execution, not blockers to start Phase 0.
