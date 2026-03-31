# Tech Context

## Core Stack in Use
- Python, NumPy, pandas
- PyTorch for DL training
- scikit-learn ecosystem for baseline ML and preprocessing artifacts
- Sentence-transformers embeddings (MiniLM historical, MPNet current extension)

## Data Artifacts
- Processed splits: `data/processed/{train,val,test}.csv`
- Engineered features: `ml/features/X_*` and `ml/features/y_*`
- MPNet embeddings: `data/embeddings/mpnet_lyrics_768d_{train,val,test}.npy`

## Training Artifacts
- Current MPNet trainer: `dl/04_train_mlp_with_mpnet.py`
- Current best checkpoint path: `models/checkpoints/mlp_mpnet_best.pt`
- Latest metrics snapshot: `results/dl_metrics/mlp_mpnet_20260401_012157.csv`

## Operational Notes
1. Keep output metric schema stable for cross-run comparison.
2. Use deterministic settings for reproducibility.
3. Maintain cache-first workflow to avoid unnecessary recomputation.
4. Ask user to run Python training scripts when execution is needed.
