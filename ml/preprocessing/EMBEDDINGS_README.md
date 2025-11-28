# Lyric Embeddings Processing

This module extracts semantic embeddings from song lyrics using the `all-MiniLM-L6-v2` model from sentence-transformers.

## Overview

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384 (compact semantic vectors)
- **Optimized for**: English text, semantic similarity
- **Performance**: Fast inference, good quality

**Why Embeddings?**
- Captures semantic meaning beyond word frequency (vs TF-IDF)
- Compact representation (384-dim vs 1000+ for TF-IDF)
- Pre-trained on large corpus of text pairs

## Installation

```bash
pip install sentence-transformers torch
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### As Part of Pipeline

```bash
# Run just embeddings (skips if cached)
cd ml/preprocessing
python run_preprocessing.py --steps embeddings

# Force recomputation
python run_preprocessing.py --steps embeddings --force

# Run all preprocessing steps
python run_preprocessing.py
```

### Standalone

```bash
cd ml/preprocessing
python process_embeddings.py

# Quiet mode
python process_embeddings.py --quiet

# Custom batch size
python process_embeddings.py --batch-size=128
```

## Performance

**Initial Run** (computing embeddings):
- ~30-60 minutes for 700k songs
- Uses GPU if available (automatically detected)
- Shows progress bar with ETA

**Subsequent Runs** (loading from cache):
- ~2-5 seconds to load from disk
- Embeddings cached in `ml/features/X_*_embeddings.npy`

## Output Files

```
ml/features/
├── X_train_embeddings.npy  (514k × 384)
├── X_val_embeddings.npy    (109k × 384)
└── X_test_embeddings.npy   (109k × 384)
```

## Cache Behavior

The preprocessing pipeline uses intelligent caching:

1. **First run**: Computes embeddings (30-60 min)
2. **Cached runs**: Loads from disk (instant)
3. **Cache invalidation**: Triggered if:
   - Input CSV files change (train/val/test.csv)
   - Force flag used (`--force`)
   - Output files deleted

## Using Embeddings in Models

```python
import numpy as np

# Load embeddings
X_train_emb = np.load('ml/features/X_train_embeddings.npy')
X_val_emb = np.load('ml/features/X_val_embeddings.npy')

# Combine with audio features
X_train_audio = np.load('ml/features/X_train_audio.npy')
X_train_combined = np.hstack([X_train_audio, X_train_emb])

# Train model
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train_combined, y_train)
```

## Pre-built Model Scripts

Three scripts are available to train models with embeddings:

1. **`embedding_models.py`**: Audio + Embeddings (405 features)
2. **`full_features_models.py`**: Audio + Text Stats + Sentiment + Embeddings (412 features)
3. **`compare_text_approaches.py`**: Compares all text feature combinations

Run from `ml/models/`:
```bash
cd ml/models

# Train with embeddings
python embedding_models.py

# Train with all features
python full_features_models.py
```

## Expected Performance Gain

Based on literature and preliminary tests:

| Target | Baseline (Audio) | + Text Stats | + Embeddings | Expected Gain |
|--------|------------------|--------------|--------------|---------------|
| Valence | 0.35 | 0.37 | **0.39-0.42** | +0.02-0.05 |
| Energy | 0.83 | 0.83 | **0.83-0.84** | +0.00-0.01 |
| Danceability | 0.53 | 0.55 | **0.55-0.57** | +0.00-0.02 |
| Popularity | 0.09 | 0.12 | **0.12-0.14** | +0.00-0.02 |

**Key Insight**: Embeddings most useful for valence (emotional/semantic content).

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python process_embeddings.py --batch-size=32
```

### Slow Processing
- Uses CPU by default
- For GPU acceleration: Install `torch` with CUDA support
- Check GPU usage: `nvidia-smi` (if available)

### Missing Dependencies
```bash
pip install sentence-transformers torch
```

### Cache Issues
```bash
# Clear embeddings cache
python run_preprocessing.py --clear embeddings

# Force recomputation
python run_preprocessing.py --steps embeddings --force
```

## Technical Details

**Model Architecture**: 
- Based on BERT (transformer encoder)
- Mean pooling of token embeddings
- Normalized to unit length

**Empty Lyrics Handling**:
- Empty/null lyrics → zero vector (384 zeros)
- Model handles short text robustly

**Batch Processing**:
- Default batch size: 64
- Processes in chunks to manage memory
- Progress bar shows ETA

## References

- Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Paper: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
