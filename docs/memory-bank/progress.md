# Progress: Music Prediction Project

## 📅 Current Phase: Semester 2 Planning (March 2026)

---

## ✅ Semester 1 Summary (Oct 2025 - Jan 2026)

### Final Deliverables
- ✅ **Thesis**: Written and submitted
- ✅ **GitHub**: Code published
- ✅ **Kaggle**: Dataset published (Bronze Medal 🥉, 48 votes)

### Dataset
- **Final Size**: 550,622 songs (English-only)
- **Features**: 414 total (23 audio + 5 text + 2 sentiment + 384 embeddings)
- **Splits**: 374,997 train / 89,171 val / 86,454 test (artist-aware)

### Best Test Results
| Target | Best Model | R² | Notes |
|--------|-----------|-----|-------|
| Energy | CatBoost_tuned | 0.81 | Highly predictable |
| Danceability | XGBoost_tuned | ~0.55 | Moderate |
| Valence | XGBoost_tuned | ~0.45 | Needs better text |
| Popularity | CatBoost | ~0.13 | External factors dominate |

### Key Methodology
- Artist-aware splits (GroupShuffleSplit)
- MiniLM-384d embeddings, TextBlob sentiment
- RFE feature selection, 28+ models compared

---

## 🚀 Semester 2: Deep Learning (March 2026+)

### Target Improvements
| Metric | Semester 1 | Goal |
|--------|------------|------|
| Energy | 0.81 | 0.85+ |
| Valence | ~0.45 | 0.60+ |
| Popularity | ~0.13 | 0.25+ |

### Phase 1: Lyrics Deep Learning (Weeks 1-4) ✅ NO BLOCKERS
- [ ] Fine-tune DistilBERT on existing lyrics
- [ ] Compare to MiniLM baseline
- [ ] Add rhyme/structure features
- [ ] Multi-task prediction network

**Expected**: Valence R² 0.45 → 0.55-0.60

### Phase 2: Audio Deep Learning (Weeks 5-8) ⚠️ DATA DEPENDENT
- [ ] Source legal audio (FMA ~100K CC tracks)
- [ ] Extract Mel spectrograms / MFCCs
- [ ] Use pretrained VGGish / Wav2Vec 2.0
- [ ] CNN on spectrograms

**Constraint**: 550K MP3s not feasible legally

### Phase 3: Multimodal Fusion (Weeks 9-12)
- [ ] Early fusion: Concatenate embeddings → MLP
- [ ] Late fusion: Separate networks → combine
- [ ] Cross-modal attention mechanisms
- [ ] Multi-task learning (4 targets simultaneously)

### Phase 4: Research & Publication (Weeks 13-16)
- [ ] Popularity deep dive (GNN, temporal features)
- [ ] Explainability (SHAP, attention visualization)
- [ ] Conference paper submission
- [ ] Demo app (Streamlit/Gradio)

---

## 📦 New Dependencies (Semester 2)

```
torch, transformers, wandb, optuna, librosa, torchaudio, gradio
```

---

## ⚠️ Known Risks

1. **Audio data**: 550K MP3s not feasible → Use FMA subset or lyrics-only
2. **Compute**: GPU costs → Start with Kaggle/Colab free tiers
3. **Overfitting**: More params → Strong validation needed

---

## 🎯 Quick Wins to Start

1. Fine-tune DistilBERT on lyrics (1-2 days)
2. Set up Weights & Biases for tracking
3. Download FMA small for audio tests
4. Create PyTorch dataloader for existing dataset
