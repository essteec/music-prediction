# Project Brief: ML Music Prediction

## 📅 Current Phase: Semester 2 - Deep Learning (March 2026)

---

## Project Overview

Final year thesis project predicting musical attributes from lyrics and audio features. 
- **Semester 1**: Traditional ML comparison (complete)
- **Semester 2**: Deep learning extension (current)

## Dataset
- **Size**: 550,622 songs (English-only)
- **Features**: 414 (audio + text + sentiment + embeddings)
- **Source**: Spotify + Chosic scraping

## Target Variables
| Target | Range | Semester 1 R² | Semester 2 Goal |
|--------|-------|---------------|-----------------|
| Energy | 0-1 | 0.81 | 0.85+ |
| Danceability | 0-1 | ~0.55 | 0.65+ |
| Valence | 0-1 | ~0.45 | 0.60+ |
| Popularity | 0-100 | ~0.13 | 0.25+ |

---

## Semester 1 Achievements ✅
- ✅ Thesis written and submitted
- ✅ GitHub code published
- ✅ Kaggle dataset: Bronze Medal 🥉 (48 votes)
- ✅ 28+ ML algorithms compared
- ✅ Best models: CatBoost, XGBoost, LightGBM

---

## Semester 2 Objectives 🚀

### Primary Goal
Extend project with deep learning to improve predictions

### Approaches
1. **Lyrics DL**: Fine-tune BERT/RoBERTa (vs MiniLM baseline)
2. **Audio DL**: CNN on spectrograms (if audio data available)
3. **Multimodal**: Fuse audio + text with attention

### Constraint
⚠️ 550K MP3s not legally feasible → Use FMA subset or lyrics-only

### Deliverables
- [ ] Semester 2 thesis/report
- [ ] Conference paper (if results strong)
- [ ] Demo application
- [ ] Extended GitHub repository
