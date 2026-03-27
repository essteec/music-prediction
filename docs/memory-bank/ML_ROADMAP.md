# ML Pipeline Roadmap

## 📅 Current Phase: Semester 2 - Deep Learning (March 2026)

---

## ✅ Semester 1 Summary (Complete)

### What We Built
- **Dataset**: 550,622 English songs, 414 features
- **Features**: Audio (23) + Text stats (5) + Sentiment (2) + Embeddings (384)
- **Models**: 28+ algorithms compared
- **Best**: Gradient boosting (CatBoost, XGBoost, LightGBM)

### Key Methodology (Reusable)
```python
# Artist-aware splits (prevents data leakage)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['artist_id']))

# Embeddings (cache to disk)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(lyrics_list, batch_size=64)
joblib.dump(embeddings, 'embeddings.pkl')
```

### Final Results
| Target | R² | RMSE |
|--------|-----|------|
| Energy | 0.81 | 0.095 |
| Danceability | 0.55 | 0.106 |
| Valence | 0.45 | 0.181 |
| Popularity | 0.13 | 1.414 |

---

## 🚀 Semester 2: Deep Learning Roadmap

### Phase 7: Lyrics Deep Learning (Weeks 1-4)

**Goal**: Improve text understanding beyond MiniLM

#### 7.1 Pretrained Language Models
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Fine-tune for regression
model.classifier = nn.Linear(768, 4)  # 4 targets
```

**Models to Try**:
- [ ] DistilBERT (fast, good baseline)
- [ ] RoBERTa (larger, potentially better)
- [ ] Llama/Mistral embeddings (latest)

**Expected**: Valence R² 0.45 → 0.55-0.60

#### 7.2 Advanced Text Features
- [ ] Rhyme pattern detection
- [ ] Verse/chorus structure
- [ ] Repetition analysis
- [ ] Topic modeling (BERTopic)

### Phase 8: Audio Deep Learning (Weeks 5-8)

**⚠️ CONTINGENT ON AUDIO DATA**

#### 8.1 Legal Audio Sources
- **FMA**: ~100K CC-licensed tracks ✅ RECOMMENDED
- **MTG-Jamendo**: ~55K CC-licensed
- **Spotify previews**: 30-sec, limited

#### 8.2 Audio Representations
```python
import librosa
import torch

# Mel spectrogram
y, sr = librosa.load('song.mp3', sr=22050)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Pretrained embeddings
from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
```

**Models**:
- [ ] CNN on Mel spectrograms (ResNet/VGG)
- [ ] VGGish embeddings
- [ ] Wav2Vec 2.0 representations
- [ ] Audio Spectrogram Transformer (AST)

### Phase 9: Multimodal Fusion (Weeks 9-12)

#### 9.1 Fusion Strategies
```python
# Early fusion
combined = torch.cat([audio_emb, text_emb], dim=1)
output = mlp(combined)

# Late fusion
audio_pred = audio_model(audio)
text_pred = text_model(text)
final = (audio_pred + text_pred) / 2

# Cross-modal attention
attention = CrossModalAttention(d_audio=512, d_text=768)
fused = attention(audio_emb, text_emb)
```

#### 9.2 Multi-Task Learning
```python
class MultiTaskModel(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            'valence': nn.Linear(768, 1),
            'energy': nn.Linear(768, 1),
            'danceability': nn.Linear(768, 1),
            'popularity': nn.Linear(768, 1)
        })
```

### Phase 10: Research & Publication (Weeks 13-16)

#### 10.1 Explainability
- SHAP for deep models
- Attention visualization
- Grad-CAM for spectrograms

#### 10.2 Deliverables
- [ ] Conference paper
- [ ] Demo app (Gradio)
- [ ] HuggingFace model upload

---

## 📦 Dependencies

```txt
# Deep Learning
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
librosa>=0.10.0

# Experiment Tracking
wandb>=0.15.0
optuna>=3.0.0

# Demo
gradio>=3.0.0
```

---

## ⚠️ Critical Rules (Still Apply)

1. **Artist-aware splits** - GroupShuffleSplit mandatory
2. **Cache embeddings** - Compute once, reuse forever
3. **Test set ONCE** - Touch only at final evaluation
4. **Iterate** - Don't complete features before testing

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install torch transformers wandb librosa gradio

# 2. Fine-tune DistilBERT
python ml/models/finetune_bert.py --model distilbert --epochs 3

# 3. Track experiments
wandb login
wandb init --project music-prediction-v2
```
