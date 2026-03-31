# Active Context: Current Work Focus

## 📅 Current Status

**Phase**: Phase 0 - PyTorch MLP Baseline ✅ **COMPLETE**
**Status**: Weighted MLP baseline established - Ready for Phase 1
**Next**: Phase 1 - BERT Fine-tuning for Lyrics

---

## ✅ Semester 1: ML Baseline (COMPLETE)

### Achievements
- **Dataset**: 550,622 English songs with 414 features
- **Features**: 23 audio + 5 text stats + 2 sentiment + 384 embeddings (MiniLM)
- **Splits**: Artist-aware train/val/test (zero overlap)
- **Models**: 28+ algorithms tested
- **Best**: Gradient boosting (CatBoost, XGBoost, LightGBM)

### ML Baseline to Beat (Test Set R²)
| Target | R² | Best Model |
|--------|-----|-----------|
| Energy | **0.81** | CatBoost_tuned |
| Danceability | **0.55** | XGBoost_tuned |
| Valence | **0.45** | XGBoost_tuned (weak - opportunity!) |
| Popularity | **0.13** | CatBoost (very weak - hard target) |

---

## 🚀 Semester 2: Deep Learning Goal

**Simple Goal**: Beat ML baseline using DL neural networks

**Strategy**: Start with PyTorch MLP on existing 414 features, then iterate improvements based on semester2-full-plan.md inspiration

**Success Target**: Ambitious - beat ML on all 4 targets

---

## ✅ Phase 0: PyTorch MLP Baseline (COMPLETE)

### Final Results (Weighted Loss - Test Set)
| Target | ML R² | DL R² | Gap | Status |
|--------|-------|-------|-----|--------|
| Valence | 0.45 | **0.35** | -0.10 | Still behind but improved |
| Energy | 0.81 | **0.75** | -0.06 | Close to ML |
| Danceability | 0.55 | **0.47** | -0.08 | Still behind but improved |
| Popularity | 0.13 | **0.12** | -0.01 | Essentially equal |
| **Average** | 0.49 | **0.42** | -0.07 | 16.7% better than unweighted |

### Key Achievements
- ✅ **Working PyTorch pipeline** with CUDA support
- ✅ **Reproducibility**: seed=42, deterministic results every run
- ✅ **Multi-task weighted loss**: Balanced learning across 4 targets
- ✅ **Critical bugs fixed**: Sigmoid/log1p mismatch, checkpoint loading
- ✅ **Baseline established**: 0.42 avg R² (unweighted was 0.36)

### Critical Bugs Discovered & Fixed
1. **Sigmoid vs log1p mismatch**: Popularity uses log1p [0-4.6] but Sigmoid constrains to [0-1] → Removed final activation
2. **Reproducibility**: Random seed not set → Added set_seed(42) everywhere
3. **Multi-task imbalance**: Popularity's large errors dominated gradient → Weighted loss [2.0, 1.0, 2.0, 0.5]
4. **PyTorch 2.6 breaking change**: torch.load() defaults to weights_only=True → Added weights_only=False

### Files Created
```
dl/
├── 01_xor_network.py          # XOR learning exercise (reproducible)
├── 02_train_mlp.py             # Main MLP trainer (weighted loss)
├── test_setup.py               # Setup verification
├── README.md                   # Phase 0 documentation
└── utils/
    ├── data_loaders.py         # PyTorch Dataset (deterministic)
    ├── models.py               # MusicMLP (no final activation)
    ├── metrics.py              # R², RMSE, MAE
    └── reproducibility.py      # Centralized seed management
```

### Next Phase Prerequisites
1. ⏳ **Install transformers**: `pip install transformers datasets`

3. ⏳ **Learn PyTorch basics**: Karpathy's Neural Networks: Zero to Hero
   - Understand: tensors, forward pass, loss, backprop
   - Build XOR network (4 points) to validate pipeline

### Tasks
1. Create `dl/` directory structure
2. Build `dl/01_xor_network.py` (learn PyTorch)
3. Build `dl/02_train_mlp.py` (main baseline)
4. Train MusicMLP on 414 features (from HitMusicNet paper)
5. Create `notebooks/08_phase0_mlp_analysis.ipynb` (visualize results)
6. Compare to ML baseline

### Expected Result
- MLP R² within 5% of ML baseline (tabular MLP rarely beats gradient boosting)
- Proves DL pipeline works
- Ready for Phase 1 improvements

---

## 📁 Directory Structure

### Existing (Semester 1)
```
ml/features/                     # ✅ 24 .npy files ready (941 MB)
data/processed/                  # ✅ CSV splits (train/val/test)
notebooks/01-07_*.ipynb          # ✅ Semester 1 analysis
```

### To Create (Semester 2)
```
dl/                              # Deep learning code (.py scripts)
├── 01_xor_network.py           # Phase 0: Learn PyTorch
├── 02_train_mlp.py             # Phase 0: MLP baseline
├── 03_finetune_distilbert.py   # Phase 1: BERT
├── 04_extract_lyric_structure.py # Phase 1: Structure features
├── 05_train_autoencoder.py     # Phase 2: Compression
├── 06_extract_audio_embeddings.py # Phase 3: Audio (pending)
└── utils/
    ├── data_loaders.py
    ├── metrics.py
    └── models.py

data/embeddings/                 # New DL embeddings (.npy)
├── bert_lyrics_768d.npy        # Phase 1
├── lyric_structure_6d.npy      # Phase 1
├── autoencoder_compressed_83d.npy # Phase 2
└── audio_embeddings.npy        # Phase 3 (if audio acquired)

models/checkpoints/              # PyTorch .pt weights
results/dl_metrics/              # Training results CSVs
notebooks/08-12_*.ipynb          # DL analysis notebooks
```

**Convention**: Python scripts for training, notebooks for visualization only

---

## 🎯 Immediate Next Actions

1. **Install PyTorch**: `pip install torch wandb`
2. **Create dl/ directory**: `mkdir -p dl/utils`
3. **Learn PyTorch**: Watch Karpathy videos, understand tensors
4. **Build XOR network**: Simple 4-point problem to validate pipeline
5. **Build MLP baseline**: Train on 414 features
6. **Compare results**: Notebook to compare DL vs ML

**After Phase 0**: Decide next improvement based on results (likely Phase 1 BERT for Valence)

---

## 📦 Tech Stack (Install as Needed)

### Phase 0 (Now)
```bash
pip install torch wandb
```

### Phase 1 (Later - Text Improvements)
```bash
pip install transformers vaderSentiment pronouncing
```

### Phase 2-5 (Future)
- Phase 2: No new installs (PyTorch only)
- Phase 3: `librosa torchaudio` (if audio acquired)
- Phase 4: Architecture experiments (PyTorch only)
- Phase 5: `shap` for explainability

---

## 🔄 Phase Progression

**Don't do everything at once** - work sequentially:

1. **Phase 0**: Establish DL baseline (current)
2. **Phase 1**: Improve text (BERT fine-tuning - biggest opportunity)
3. **Phase 2**: Compression (autoencoder - optional)
4. **Phase 3**: Audio embeddings (only if acquisition solved)
5. **Phase 4**: Architecture experiments (after knowing what data works)
6. **Phase 5**: Final model & deployment

**After each phase**: Compare to ML baseline, decide to continue or iterate

---

## 🎯 Success Metrics

### Conservative
- Beat ML on Valence (0.45 → 0.50+)
- Match ML on Energy/Danceability

### Moderate
- Beat ML on Valence and Danceability
- Slight Popularity improvement

### Ambitious (Goal)
- Beat ML on ALL 4 targets
- Valence: 0.45 → 0.55+
- Popularity: 0.13 → 0.18+

---

## 📝 Key Notes

- **No timelines**: Work at your own pace
- **semester2-full-plan.md**: External inspiration (not exact roadmap)
- **ROADMAP.md**: Actual plan for this project
- **Audio optional**: Can succeed without solving audio acquisition
- **Storage limit**: 150GB (tight if downloading audio via yt-dlp)
- **Focus on wins**: BERT for Valence is clearest improvement path
