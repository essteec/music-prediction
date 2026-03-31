# Deep Learning - Phase 0

## 🎯 Goal
Establish PyTorch baseline using existing 414 features and prove DL can match/beat ML.

## 📁 Files

### Learning Exercises
- **01_xor_network.py** - Simple XOR problem to learn PyTorch basics
  - 4-point dataset, 2→4→1 network
  - Validates: tensors, forward/backward pass, training loop
  - Should converge to 100% accuracy in ~5000 epochs

### Main Training
- **02_train_mlp.py** - MLP baseline on 414 features
  - Architecture: 414 → 207 → 138 → 4 (from HitMusicNet paper)
  - Multi-task: Predicts all 4 targets simultaneously
  - Early stopping with patience=10
  - Saves best model to `models/checkpoints/`
  - Results to `results/dl_metrics/`

### Utilities
- **utils/data_loaders.py** - PyTorch Dataset and DataLoader
  - Loads .npy files from `ml/features/`
  - Concatenates 4 feature types (414 total)
  - StandardScaler normalization
  
- **utils/models.py** - Neural network architectures
  - `MusicMLP`: Main 3-layer MLP for music prediction
  - `SimpleXORNetwork`: Learning exercise network
  
- **utils/metrics.py** - Evaluation utilities
  - R², RMSE, MAE computation
  - Pretty printing and CSV export

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install torch wandb
```

### 2. Learn PyTorch (Optional but Recommended)
```bash
cd dl
python 01_xor_network.py
```

Expected output:
```
✓ SUCCESS! Network learned XOR perfectly!
```

### 3. Train MLP Baseline
```bash
cd dl
python 02_train_mlp.py
```

Training will:
- Load 414 features from `ml/features/` (941 MB)
- Train on 374,997 songs
- Validate on 89,245 songs  
- Test on 86,380 songs (final evaluation)
- Save best model checkpoint
- Compare to ML baseline

Expected runtime: 
- CPU: ~20-30 min per epoch
- GPU: ~2-5 min per epoch

### 4. Monitor Training
Optional: Use W&B for experiment tracking
```bash
wandb login  # First time only
```

### 5. Analyze Results
Create notebook: `notebooks/08_phase0_mlp_analysis.ipynb`
- Load results from `results/dl_metrics/`
- Plot training curves
- Compare DL vs ML baseline
- Per-target scatter plots

## 📊 Expected Results

**Conservative Target** (MLP within 5% of ML):
- Energy: 0.77-0.81 (ML: 0.81)
- Danceability: 0.50-0.55 (ML: 0.55)
- Valence: 0.40-0.45 (ML: 0.45)
- Popularity: 0.10-0.13 (ML: 0.13)

**Note**: Tabular MLP rarely beats gradient boosting. This is expected!
The goal is to establish PyTorch pipeline, not beat ML yet.

## 📋 Next Steps After Phase 0

1. **If MLP works**: Move to Phase 1 (BERT fine-tuning for lyrics)
2. **If MLP struggles**: Debug data loading, try different hyperparameters
3. **Create visualization notebook**: Analyze where DL helps vs hurts

## 🔧 Hyperparameters

Default config in `02_train_mlp.py`:
```python
{
    'batch_size': 256,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 10,
    'dropout': 0.5,
    'optimizer': 'Adam',
    'loss': 'MSELoss'
}
```

Feel free to experiment with:
- Learning rate: [1e-4, 1e-3, 1e-2]
- Dropout: [0.3, 0.5, 0.7]
- Batch size: [128, 256, 512]

## 📝 Notes

- All targets are in [0, 1] range (except popularity normalized by /100)
- Artist-aware splits ensure zero artist overlap
- StandardScaler applied to inputs (fit on train, transform val/test)
- Early stopping prevents overfitting
- Test set evaluated ONCE at the end
