# Tech Context: Technologies and Tools

## 📅 Current Phase: Semester 2 - Deep Learning

---

## ✅ Semester 1 Stack (Working)

### ML Framework
- **scikit-learn**: Preprocessing, metrics, splits
- **XGBoost/CatBoost/LightGBM**: Best performing models
- **sentence-transformers**: MiniLM-384d embeddings
- **TextBlob**: Sentiment analysis

### Data
- **pandas/numpy**: Data processing
- **joblib**: Model & embedding caching

### Visualization
- **matplotlib/seaborn**: Publication-quality plots
- **Academic styling**: 300 DPI, serif fonts

---

## 🚀 Semester 2 Stack (New)

### Deep Learning
```txt
torch>=2.0.0           # Primary DL framework
torchaudio>=2.0.0      # Audio processing
transformers>=4.30.0   # HuggingFace pretrained models
```

### Audio Processing
```txt
librosa>=0.10.0        # Spectrograms, MFCCs
soundfile              # Audio I/O
```

### Experiment Tracking
```txt
wandb>=0.15.0          # Experiment tracking, sweeps
optuna>=3.0.0          # Hyperparameter optimization
```

### Demo/Deployment
```txt
gradio>=3.0.0          # Interactive demos
streamlit              # Alternative demo framework
```

### Pretrained Models
- **Text**: DistilBERT, RoBERTa (HuggingFace)
- **Audio**: VGGish, Wav2Vec 2.0, CLAP

---

## 📁 Dataset Structure

### Current Files
```
data/processed/
├── songs.csv           # 550,622 songs (main dataset)
├── artists.csv         # Artist metadata
└── train/val/test.csv  # Split files

ml/features/
├── X_train_*.npy       # Feature arrays
├── y_train_*.npy       # Target arrays
└── embeddings.pkl      # Cached MiniLM embeddings
```

### Semester 2 Additions
```
data/audio/             # Raw audio files (if obtained)
├── fma_small/          # FMA dataset subset
└── spectrograms/       # Cached spectrograms

ml/models/
├── bert_valence.pt     # Fine-tuned models
└── multimodal.pt       # Fusion models
```

---

## 💻 Compute Requirements

### Semester 1 (CPU sufficient)
- RAM: 8GB+
- Storage: ~5GB

### Semester 2 (GPU needed)
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- Storage: ~50-100GB (if audio)

### Free Options
- **Kaggle**: 30h/week GPU
- **Colab**: Limited free GPU
- **University cluster**: If available

---

## ⚠️ Audio Data Constraint

**Problem**: 550K MP3s not legally obtainable

**Solutions**:
1. **FMA Dataset**: ~100K CC-licensed tracks
2. **MTG-Jamendo**: ~55K CC-licensed
3. **Lyrics-only DL**: No audio needed
4. **Pre-computed embeddings**: VGGish from research datasets

---

## 🔧 Setup Commands

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install Semester 2 dependencies
pip install torch torchaudio transformers
pip install librosa wandb optuna gradio

# Setup W&B
wandb login

# Download FMA (if using audio)
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
```
