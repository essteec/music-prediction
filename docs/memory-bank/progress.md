# Progress: Music Prediction Project

## 📅 Current Phase: Phase 1 - BERT Fine-tuning (Next)

**Status**: Phase 0 complete - Weighted MLP baseline established
**Current Focus**: Ready to start Phase 1 - BERT fine-tuning for lyrics
**Last Completed**: Phase 0 - PyTorch MLP Baseline ✅

---

## ✅ Semester 1: ML Baseline (COMPLETE)

### Achievements
- **Dataset**: 550,622 English songs with 414 features
- **Features**: 23 audio + 5 text stats + 2 sentiment + 384 embeddings (MiniLM)
- **Preprocessing**: All .npy files at `ml/features/` (941 MB total)
- **Splits**: Artist-aware train/val/test (zero artist overlap)
- **Models**: 28+ algorithms tested (CatBoost, XGBoost, LightGBM dominated)
- **Thesis**: Complete and submitted
- **Kaggle**: Bronze Medal (48 votes)

### ML Baseline to Beat (Test Set R²)
| Target | Best Model | R² | Notes |
|--------|-----------|-----|-------|
| **Energy** | CatBoost_tuned | **0.81** | Strong baseline - highly predictable |
| **Danceability** | XGBoost_tuned | **0.55** | Moderate baseline |
| **Valence** | XGBoost_tuned | **0.45** | Weak - main opportunity for improvement |
| **Popularity** | CatBoost | **0.13** | Very weak - external factors dominate |

### Available Data (Ready for DL)
```
ml/features/
├── X_train_audio.npy          (374997, 23)
├── X_train_text_stats.npy     (374997, 5)
├── X_train_sentiment.npy      (374997, 2)
├── X_train_embeddings.npy     (374997, 384)
├── y_train_{target}.npy       (4 targets: valence, energy, danceability, popularity)
└── Same structure for val and test splits

Total: 414 features ready for training
```

---

## 🚀 Semester 2: Deep Learning Extension

**Goal**: Beat ML baseline using Deep Learning neural networks

**Strategy**: Start simple with PyTorch MLP on existing 414 features, then iterate improvements based on semester2-full-plan.md inspiration

---

## ✅ Phase 0: PyTorch MLP Baseline (COMPLETE)

### Timeline
- **Started**: March 30, 2026
- **Completed**: March 31, 2026
- **Duration**: 1 day

### Final Results - Weighted Loss (Test Set R²)

**Comparison to ML Baseline:**
| Target | ML R² | DL R² (Unweighted) | DL R² (Weighted) | Improvement | Gap to ML |
|--------|-------|-------------------|-----------------|-------------|-----------|
| Valence | 0.45 | 0.27 | **0.35** | +0.08 | -0.10 |
| Energy | 0.81 | 0.71 | **0.75** | +0.04 | -0.06 |
| Danceability | 0.55 | 0.37 | **0.47** | +0.10 | -0.08 |
| Popularity | 0.13 | 0.11 | **0.12** | +0.01 | -0.01 |
| **Average** | **0.49** | **0.36** | **0.42** | **+0.06** | **-0.07** |

**Key Findings:**
- ✅ **16.7% improvement** with weighted loss (0.36 → 0.42 avg R²)
- ✅ DL still ~7 points behind ML (expected for tabular data with engineered features)
- ✅ Weighted loss balanced learning: Valence & Danceability improved most (+0.08, +0.10)
- ✅ Multi-task learning works - no target domination issues
- ✅ Popularity essentially equal to ML (both struggle with external factors)

### Critical Bugs Discovered & Fixed

1. **Sigmoid vs log1p Mismatch (CRITICAL)**
   - **Problem**: Popularity uses `log1p()` transformation [0-4.6] but model used Sigmoid activation [0-1]
   - **Impact**: Model physically unable to predict popularity > 1.0
   - **Fix**: Removed Sigmoid from final layer, use raw regression
   - **File**: `dl/utils/models.py` line 48-51

2. **Reproducibility Issues**
   - **Problem**: Random initialization caused non-deterministic results
   - **Impact**: Cannot verify methodology or compare experiments
   - **Fix**: `set_seed(42)` at start of all scripts, deterministic DataLoader
   - **Files**: All training scripts + `dl/utils/reproducibility.py`

3. **Multi-Task Learning Imbalance**
   - **Problem**: Popularity's large errors (scale 0-4.6) dominated gradient
   - **Impact**: Model ignored [0-1] targets (valence, energy, danceability)
   - **Fix**: Weighted loss `[2.0, 1.0, 2.0, 0.5]` - down-weight popularity
   - **File**: `dl/02_train_mlp.py` lines 100-115

4. **PyTorch 2.6 Breaking Change**
   - **Problem**: `torch.load()` default changed to `weights_only=True`
   - **Impact**: Checkpoint loading failed with UnpicklingError
   - **Fix**: Added `weights_only=False` for trusted checkpoints
   - **File**: `dl/02_train_mlp.py` line 207

### Technical Implementation

**Architecture**: MusicMLP (from HitMusicNet paper)
```
Input (414) → FC1 (207) → ReLU → Dropout(0.5) 
           → FC2 (138) → ReLU → Dropout(0.5)
           → FC3 (4) → No activation (raw regression)
```

**Training Configuration:**
- Loss: MSELoss with weighted reduction `[2.0, 1.0, 2.0, 0.5]`
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Early stopping: patience=10 (stopped at epoch 32)
- Seed: 42 (reproducible)
- Device: CUDA (GPU)

**Files Created:**
```
dl/
├── 01_xor_network.py          # XOR learning exercise (10K epochs, 8 hidden units)
├── 02_train_mlp.py             # Main MLP trainer (weighted multi-task loss)
├── test_setup.py               # Setup verification script
├── README.md                   # Phase 0 documentation
└── utils/
    ├── __init__.py
    ├── data_loaders.py         # PyTorch Dataset with deterministic shuffling
    ├── models.py               # MusicMLP (no final activation)
    ├── metrics.py              # R², RMSE, MAE evaluation
    └── reproducibility.py      # Centralized seed management

models/checkpoints/
└── mlp_baseline_best.pt        # Best model (epoch 22, val_loss=0.2733)

results/dl_metrics/
├── mlp_baseline_20260331_022237.csv  # Unweighted results
└── mlp_baseline_20260331_024129.csv  # Weighted results (final)
```

### Lessons Learned

1. **Gradient Boosting dominates tabular data** - Expected that MLP on engineered features would underperform
2. **DL needs better representations** - Next phases focus on learned representations (BERT, audio embeddings)
3. **Multi-task learning requires balance** - Different target scales need weighted loss
4. **Reproducibility is critical** - Always set seeds for research methodology
5. **Target transformations matter** - Must match preprocessing (log1p) to model architecture

### Why DL Lost to ML (Expected)

- **Engineered features**: 384 sentence embeddings, 23 Spotify audio features already optimal for ML
- **Tabular data**: Gradient boosting (XGBoost, CatBoost) designed for this
- **No representation learning**: MLP just transforms existing features, doesn't learn new ones
- **Small model**: 115K parameters vs ensemble of 1000+ trees

### Where DL Will Win (Future Phases)

- **Phase 1 (BERT)**: Fine-tuned BERT will capture lyric semantics better than MiniLM embeddings
- **Phase 3 (Audio)**: Raw audio → learned embeddings will beat hand-crafted Spotify features  
- **Phase 4 (Architecture)**: Attention, multi-modal fusion, deeper networks

---

## 📋 Next Steps: Phase 1 - BERT Fine-tuning

**Ready to start**: Install transformers and begin BERT fine-tuning for lyrics

**Success Target**: Ambitious - beat ML on all 4 targets

**Reference**: HitMusicNet paper (Martín-Gutiérrez et al., IEEE Access 2020) provides architectural ideas

---

## 📋 Phase 0: PyTorch MLP Baseline (Starting Now)

**Goal**: Establish DL baseline using existing 414 features, prove DL can match/beat ML

### Completed
- ✅ All Semester 1 preprocessing complete
- ✅ 414 features ready as .npy files
- ✅ Artist-aware splits created

### In Progress
- ⏳ Install PyTorch and W&B
- ⏳ Learn PyTorch basics (Karpathy videos)
- ⏳ Build XOR network (validate pipeline)
- ⏳ Build MusicMLP on 414 features
- ⏳ Compare to ML baseline

### Success Criteria
- MLP R² within 5% of ML baseline (proves DL works)
- Training pipeline established (data loading, training loops, checkpointing)
- Ready for Phase 1 improvements

---

## 📋 Phase 1: Lyrics Deep Learning (Future)

**Goal**: Improve text understanding with transformer models (biggest opportunity for Valence)

### Approach
1. **Fine-tune DistilBERT**: Multi-task regression on all 4 targets from lyrics
2. **Lyric structure features**: Rhyme density, sentiment stats, repetition ratio, word uniqueness
3. **Ablation study**: Compare MiniLM vs DistilBERT vs augmented

### Expected Improvement
- Valence R²: 0.45 → 0.55+ (main target)
- Minimal change on Energy/Danceability (not lyric-dependent)

---

## 📋 Phase 2: Autoencoder Compression (Optional)

**Goal**: Apply HitMusicNet's compression strategy - reduce 414 → 83 features

### Approach
- Train autoencoder to compress features by 5×
- Train MLP on compressed 83-d representation
- Compare to MLP on 414-d raw features

### Expected
- May improve generalization by reducing noise
- Optional phase - depends on Phase 0/1 results

---

## 📋 Phase 3: Audio Deep Learning (Pending Data Acquisition)

**Goal**: Add audio embeddings to improve Energy/Danceability

### Current Blocker
- No audio files exist (only Spotify API metadata)
- Need to solve acquisition: yt-dlp vs FMA vs skip entirely
- Storage constraint: 150GB available (tight for 550K songs)

### If Audio Acquired
- VGGish embeddings (128-d, general audio)
- MERT embeddings (768-d, music-specific)
- Expected: Energy/Danceability improve most

---

## 📋 Phase 4: Architecture Experiments (Future)

**Goal**: Try different network architectures beyond simple MLP

### Ideas
- Deeper networks (5+ layers)
- Residual connections
- Attention mechanisms
- Separate encoders per modality (audio/text/metadata)
- Multi-task learning variations

### Approach
- Ablation study to find best architecture per target
- May use different models per target (like ML did)

---

## 📋 Phase 5: Final Model & Deployment (Future)

**Goal**: Package best model for production use

### Tasks
1. Select best configuration per target
2. Train on full data (train + val combined)
3. Test on held-out test set (ONCE only)
4. Update Gradio app with best DL models
5. Create final results notebook
6. Update README with DL results

---

## 📁 Directory Structure

### Existing (Semester 1)
```
ml/features/                    ✅ 24 .npy files (941 MB)
data/processed/                 ✅ CSV splits (train/val/test)
notebooks/01-07_*.ipynb         ✅ Semester 1 analysis
app/gradio_app.py               ✅ ML models deployed
```

### To Create (Semester 2)
```
dl/                             # Deep learning code (.py scripts)
├── 01_xor_network.py
├── 02_train_mlp.py
├── 03_finetune_distilbert.py
├── 04_extract_lyric_structure.py
├── 05_train_autoencoder.py
├── 06_extract_audio_embeddings.py
└── utils/

data/embeddings/                # New DL embeddings (.npy)
models/checkpoints/             # PyTorch .pt weights
results/dl_metrics/             # Training results CSVs
notebooks/08-12_*.ipynb         # DL analysis notebooks
```

---

## 🎯 Success Metrics

### Conservative (Minimum)
- Beat ML on Valence (0.45 → 0.50+)
- Match ML on Energy/Danceability

### Moderate (Expected)
- Beat ML on Valence and Danceability
- Slight Popularity improvement

### Ambitious (Goal)
- Beat ML baseline on ALL 4 targets
- Valence: 0.45 → 0.55+
- Popularity: 0.13 → 0.18+

---

## 📦 Tech Stack (Install as Needed)

### Phase 0 (Now)
```bash
pip install torch wandb
```

### Phase 1 (Later)
```bash
pip install transformers vaderSentiment pronouncing
```

### Phases 2-5 (Future)
- Phase 2: PyTorch only (no new installs)
- Phase 3: `librosa torchaudio` (if audio acquired)
- Phase 4: PyTorch only
- Phase 5: `shap` for explainability

---

## 📝 Key Notes

- **No timelines**: Flexible pace, work through phases sequentially
- **semester2-full-plan.md**: External inspiration (not exact roadmap)
- **ROADMAP.md**: Actual detailed plan for this project
- **Audio optional**: Can achieve success without solving audio acquisition
- **Convention**: Python scripts for training, notebooks for visualization
- **Focus**: BERT for Valence is clearest improvement path

---

## 🎯 Immediate Next Actions

1. ⏳ **Install PyTorch**: `pip install torch wandb`
2. ⏳ **Learn basics**: Watch Karpathy videos, understand tensors
3. ⏳ **Create dl/ directory**: `mkdir -p dl/utils`
4. ⏳ **Build XOR network**: Simple 4-point problem to validate pipeline
5. ⏳ **Build MLP baseline**: Train on 414 features
6. ⏳ **Create notebook**: Compare DL vs ML results
7. ⏳ **Decide next phase**: Based on Phase 0 results (likely Phase 1 BERT)
