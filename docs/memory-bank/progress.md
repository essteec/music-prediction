# Progress: Music Prediction Project

## 📅 Current Phase
Phase 2 - Architecture Improvements (READY TO START)

**Status**: Phase 1B complete with full metrics - ready for architecture experiments  
**Current Focus**: Implement deeper networks, residual connections, and attention mechanisms  
**Last Completed**: Phase 1B - MLP with MPNet embeddings (4.3% improvement) ✅

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

## ✅ Phase 1A: MPNet Embeddings (COMPLETE)

### Timeline
- **Started**: March 31, 2026
- **Completed**: March 31, 2026
- **Duration**: 4 hours runtime

### Achievements

**MPNet Embeddings Extracted:**
```
data/embeddings/
├── mpnet_lyrics_768d_train.npy  (374,997 songs, 1098.6 MB)
├── mpnet_lyrics_768d_val.npy    (89,172 songs, 261.2 MB)
└── mpnet_lyrics_768d_test.npy   (86,453 songs, 253.3 MB)

Total: 1.6GB of 768-d embeddings for all 550K songs
```

**Model Details:**
- **Model**: sentence-transformers/all-mpnet-base-v2 (Microsoft)
- **Embedding dimension**: 768-d (double Phase 0's 384-d MiniLM)
- **Max sequence**: 512 tokens (~3000 chars)
- **Processing**: Batch size 32, 4 hours total, ~1.09 it/s

**Technical Implementation:**
- Checkpoint logic: Rerun-safe (skips existing files)
- Memory management: torch.cuda.empty_cache() between splits
- Truncation: Lyrics limited to 3000 chars max
- Script: `dl/03_extract_better_embeddings.py` (207 lines)

### Key Decisions

1. **MPNet chosen over**:
   - DistilBERT: Sentence-transformers optimized for embeddings
   - GTE-base-v1.5: CUDA compatibility issues (rotary pos embeddings)
   - MiniLM-L6-v2: MPNet better semantic understanding

2. **Frozen embeddings first**:
   - Validate improvement before fine-tuning investment
   - Fast to test (no training needed)
   - Phase 1B will show if fine-tuning needed

3. **Feature composition**:
   - Phase 0: 414 features (23 audio + 5 text + 2 sentiment + 384 MiniLM)
   - Phase 1: 798 features (23 audio + 5 text + 2 sentiment + 768 MPNet)
   - 384 additional features from better text understanding

### Why This Helps

- **Better semantic understanding**: MPNet trained on massive corpus for similarity
- **Double capacity**: 768-d vs 384-d captures richer lyric semantics
- **Expected**: Valence R² improvement (0.45 → 0.50+)
- **Target**: Better emotion understanding from lyrics

### Technical Challenges Resolved

1. **OOM at batch_size=256**: Reduced to 32 for 6GB VRAM
2. **GTE CUDA errors**: Skipped in favor of proven MPNet
3. **Lyrics truncation**: Some songs exceed 512 token limit
4. **Memory fragmentation**: Added PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Files Created
```
dl/03_extract_better_embeddings.py  # 207 lines, MPNet extraction script
data/embeddings/mpnet_lyrics_768d_*.npy  # 3 files (train/val/test)
```

### Next Phase

**Phase 1B - Train MLP with MPNet:**
1. Load MPNet embeddings (768-d)
2. Concatenate with existing features (23 audio + 5 text + 2 sentiment)
3. Train MusicMLP on 798 total features
4. Compare to Phase 0 baseline (414 features)
5. Expected: Valence improvement from better text understanding

---

## ✅ Phase 1B: MLP with MPNet Embeddings (COMPLETE)

### Timeline
- **Started**: March 31, 2026
- **Completed**: April 1, 2026 (01:21 UTC)
- **Duration**: ~2 hours training time

### Final Results (Test Set R²) - VERIFIED

**Comparison to Phase 0 (414 features → 798 features):**
| Target | Phase 0 | Phase 1B | Improvement | Status |
|--------|---------|----------|-------------|--------|
| Valence | 0.3500 | **0.3792** | +0.0292 (+8.3%) | ✓ Better |
| Energy | 0.7500 | **0.7539** | +0.0039 (+0.5%) | ≈ Similar |
| Danceability | 0.4700 | **0.4978** | +0.0278 (+5.9%) | ✓ Better |
| Popularity | 0.1200 | **0.1311** | +0.0111 (+9.3%) | ✓ Better |
| **Average** | **0.4225** | **0.4405** | **+0.0180** | ✓ **4.3% improvement** |

**Comparison to ML Baseline (Semester 1):**
| Target | ML R² | DL R² | Gap | Status |
|--------|-------|-------|-----|--------|
| Valence | 0.45 | 0.3792 | -0.0708 | Still behind |
| Energy | 0.81 | 0.7539 | -0.0561 | Still behind |
| Danceability | 0.55 | 0.4978 | -0.0522 | Still behind |
| Popularity | 0.13 | 0.1311 | +0.0011 | ≈ Equal |

### Key Findings

**✅ Positive Results:**
- MPNet (768-d) improved over MiniLM (384-d) embeddings
- Valence gained +0.0292 R² from better text understanding
- Danceability gained +0.0278 R² (unexpected text benefit)
- 4.3% overall improvement with minimal code change
- Popularity now matches ML baseline

**⚠️ Reality Check:**
- DL still ~6-7% behind ML across all targets (expected for tabular data)
- Energy and Danceability improvements minimal from better embeddings
- Valence improvement exists but modest (+2.9% points)
- Bigger architectural changes needed to close gap

### Technical Details

**Model**: Same MusicMLP architecture as Phase 0
- Input: 798 features (23 audio + 5 text + 2 sentiment + 768 MPNet)
- Architecture: 798 → 399 → 266 → 4 (ReLU + Dropout 0.5)
- Parameters: 426,269 (vs 115K in Phase 0 with 414 features)

**Training:**
- Loss: MSELoss (unweighted - simpler for ablation)
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Early stopping: patience=10
- Seed: 42 (reproducible)
- Device: CUDA (GPU)

**Files Created:**
```
dl/04_train_mlp_with_mpnet.py                   # Training script with MPNet
models/checkpoints/mlp_mpnet_best.pt            # Best model checkpoint
results/dl_metrics/mlp_mpnet_20260401_012157.csv # Final metrics (timestamp-verified)
```

### Analysis: Did MPNet Help?

**For Valence**: ✓ Yes, but modest
- +0.0292 R² improvement (0.35 → 0.38)
- Still -0.07 behind ML baseline (0.45)
- Better text embeddings helped but insufficient alone

**For Energy/Danceability**: ≈ Minimal
- Energy: +0.0039 (noise-level improvement)
- Danceability: +0.0278 (surprising - some lyric signal)
- These targets rely more on audio features

**For Popularity**: ✓ Equal to ML now
- Crossed threshold: DL 0.1311 vs ML 0.13
- Still very weak prediction (external factors dominate)

### Decision Point: Next Phase Strategy

**Option A: Fine-tune BERT (Phase 1C)** ⚠️ NOT RECOMMENDED
- Rationale: MPNet only gave +2.9% on Valence
- Cost: 10-20 hours training time, complex pipeline
- Expected: Maybe +3-5% more (still behind ML's 0.45)
- Risk: Diminishing returns on text alone - not worth the investment

**Option B: Move to Phase 2 - Architecture Improvements** ✅ **CHOSEN PATH**
- Rationale: Bigger gains from better model architecture
- Options: Deeper networks, residual connections, attention
- Expected: 5-10% improvement across all targets
- Proven path: Focus on what DL does best (architecture vs feature engineering)

**Option C: Move to Phase 3 - Audio Embeddings** ⚠️ BLOCKED
- Rationale: Audio improvements would help Energy/Danceability most
- Blocker: No audio files (only Spotify metadata)
- Need to solve: yt-dlp acquisition or skip audio entirely

### ✅ Final Decision: Proceed to Phase 2

**Skip Phase 1C (BERT fine-tuning)** - Diminishing returns on text alone

**Start Phase 2 - Architecture Improvements:**
1. **Deeper networks**: 5-7 layers with residual connections
2. **Attention mechanisms**: Learn feature importance dynamically
3. **Better regularization**: LayerNorm, label smoothing, dropout tuning
4. **Modality-specific encoders**: Separate processing for audio/text/embeddings
5. **Task-specific heads**: Shared encoder + separate prediction heads per target
6. **Advanced optimizers**: Try AdamW with cosine annealing

### Why Phase 2 Is Right Choice

- MPNet proved text improvements help (+4.3% overall)
- But text alone won't close 8% gap to ML baseline
- Architecture improvements affect ALL modalities simultaneously
- Literature shows residual connections + attention consistently improve tabular DL
- Can combine with existing MPNet embeddings for additive gains

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
