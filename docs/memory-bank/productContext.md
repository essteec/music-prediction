# Product Context: ML Music Prediction System

## 📅 Current Phase: Semester 2 - Deep Learning

---

## What This Project Is

A thesis project exploring whether deep learning can predict musical attributes (valence, energy, danceability, popularity) from lyrics and audio features.

## Research Questions

### Semester 1 (Answered)
- ✅ Which traditional ML algorithms work best? → Gradient boosting (CatBoost, XGBoost)
- ✅ Do lyrics improve predictions? → Yes, especially for valence (+0.15 R²)
- ✅ Can popularity be predicted? → Poorly (R²=0.13, external factors dominate)

### Semester 2 (Current)
- Can BERT/transformers improve valence prediction beyond MiniLM?
- Do audio spectrograms help beyond audio features?
- Can multimodal fusion outperform single-modality models?

---

## System Flow

### Semester 1 (Complete)
```
Raw Dataset → Scraping → Cleaning → Feature Engineering → 
Traditional ML (28+ models) → Evaluation → Thesis
```

### Semester 2 (Current)
```
Existing Dataset → Fine-tune BERT → Audio Spectrograms (if data) →
Multimodal Fusion → Deep Learning Models → Comparison → Paper
```

---

## Results Summary

| Target | Semester 1 | Semester 2 Goal | Approach |
|--------|------------|-----------------|----------|
| Energy | R²=0.81 | 0.85+ | Audio DL |
| Valence | R²=0.45 | 0.60+ | BERT fine-tuning |
| Danceability | R²=0.55 | 0.65+ | Multimodal |
| Popularity | R²=0.13 | 0.25+ | Additional features |

---

## Key Constraints

### Audio Data
⚠️ 550K MP3s not legally obtainable
- **Solution**: FMA (~100K CC-licensed) or lyrics-only DL

### Compute
- Need GPU for deep learning
- Free options: Kaggle (30h/week), Colab

---

## Deliverables

### Semester 1 ✅
- Thesis document
- GitHub repository
- Kaggle dataset (Bronze Medal)

### Semester 2
- [ ] Extended thesis/report
- [ ] Conference paper
- [ ] Demo application
- [ ] Pretrained models
