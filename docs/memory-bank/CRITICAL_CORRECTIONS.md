# Critical Methodology Rules

**Status**: All issues from Semester 1 resolved. Rules still apply for Semester 2.

---

## ⚠️ Mandatory Rules

### 1. Artist-Aware Data Splits
**Problem**: Random split → same artist in train/test → inflated performance  
**Solution**: Always use GroupShuffleSplit by artist_id

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['artist_id']))
```

### 2. Cache Expensive Computations
**Problem**: Embeddings take hours for 550K songs  
**Solution**: Compute once, save to disk

```python
# Compute ONCE
embeddings = model.encode(lyrics_list, batch_size=64)
joblib.dump(embeddings, 'embeddings.pkl')

# Reuse forever
embeddings = joblib.load('embeddings.pkl')
```

### 3. Test Set Evaluation ONCE
**Problem**: Multiple test evaluations → overfitting to test set  
**Solution**: Touch test set only at final evaluation

### 4. Iterate, Don't Waterfall
**Problem**: Complete all features before testing → wasted work  
**Solution**: Build incrementally, validate continuously

---

## ✅ Semester 1 Issues (Resolved)

| Issue | Status | Solution Applied |
|-------|--------|------------------|
| Random splits | ✅ Fixed | GroupShuffleSplit |
| Embedding recomputation | ✅ Fixed | Cached to disk |
| TF-IDF memory issues | ✅ Fixed | Used MiniLM embeddings |
| Mixed key/mode encoding | ✅ Fixed | Proper mapping |
| Data leakage risk | ✅ Fixed | Artist-aware splits |

---

## 🚀 Semester 2 Considerations

### New Risks
1. **GPU memory**: Deep models need more memory
2. **Training time**: BERT fine-tuning is slower
3. **Overfitting**: More parameters = more risk

### New Rules
1. **Use mixed precision** (fp16) to save memory
2. **Checkpoint frequently** during training
3. **Use early stopping** based on validation loss
4. **Log everything** with Weights & Biases
