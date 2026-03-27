# System Patterns: Architecture

## 📅 Current Phase: Semester 2 - Deep Learning

---

## Project Structure

```
music-prediction/
├── data/
│   ├── processed/          # Clean datasets
│   │   ├── songs.csv       # 550,622 songs
│   │   └── artists.csv     # Artist metadata
│   └── audio/              # [Semester 2] Audio files
├── ml/
│   ├── features/           # Cached features (.npy)
│   ├── models/             # Trained models
│   └── preprocessing/      # Feature extraction
├── notebooks/              # Analysis notebooks (01-07)
├── results/
│   ├── metrics/            # CSV results
│   └── figures/            # Visualizations
├── thesis/                 # Academic documents
└── docs/memory-bank/       # This knowledge base
```

---

## Semester 1 Pipeline (Complete)

```
Data → Features (414) → Traditional ML → Evaluation
         ↓
    ┌────┴────┐
    │ Audio   │ 23 features (loudness, tempo, genre, etc.)
    │ Text    │ 5 features (word count, uniqueness, etc.)
    │ Sent.   │ 2 features (polarity, subjectivity)
    │ Embed.  │ 384 features (MiniLM-L6-v2)
    └─────────┘
```

---

## Semester 2 Pipeline (Planned)

### Lyrics Deep Learning
```
Lyrics → Tokenizer → BERT/RoBERTa → Fine-tune → Predictions
                          ↓
              [CLS] token embedding (768-d)
```

### Audio Deep Learning
```
Audio → Mel Spectrogram → CNN/Transformer → Predictions
  ↓
librosa.feature.melspectrogram()
```

### Multimodal Fusion
```
┌─────────────┐     ┌─────────────┐
│ Text Model  │     │ Audio Model │
│   (BERT)    │     │   (CNN)     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               ↓
        Fusion Layer
        (Attention/MLP)
               ↓
         4 Predictions
```

---

## Key Patterns

### 1. Artist-Aware Splits (MANDATORY)
```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
train_idx, test_idx = next(gss.split(df, groups=df['artist_id']))
```

### 2. Embedding Caching
```python
# Compute once
embeddings = model.encode(texts)
joblib.dump(embeddings, 'cache.pkl')

# Reuse forever
embeddings = joblib.load('cache.pkl')
```

### 3. Experiment Tracking (Semester 2)
```python
import wandb
wandb.init(project='music-prediction-v2')
wandb.log({'val_r2': r2, 'val_loss': loss})
```

---

## Feature Groups

| Group | Count | Source |
|-------|-------|--------|
| Audio | 23 | Spotify API + artist data |
| Text Stats | 5 | Computed from lyrics |
| Sentiment | 2 | TextBlob |
| Embeddings | 384 | MiniLM-L6-v2 |
| **Total** | **414** | |

### Semester 2 Additions
| Group | Count | Source |
|-------|-------|--------|
| BERT embeddings | 768 | Fine-tuned BERT |
| Spectrogram | varies | librosa |
| VGGish | 128 | Pretrained |
