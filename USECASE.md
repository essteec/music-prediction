# Use Case: Script Run Order

This file lists the exact scripts to run, in order, after you have downloaded
`data/processed/songs.csv` and the aligned audio embedding NPZ files into
`data/processed/`.

Required NPZ inputs:

- `data/processed/vggish_embeddings.npz`
- `data/processed/mel_stats_embeddings.npz`
- `data/processed/mert_embeddings.npz`
- `data/processed/panns_embeddings.npz`

## 1) Create splits and split NPZ embeddings

This creates `data/processed/{train,val,test}.csv` and writes audio splits to
`ml/features/X_{split}_{vggish|mel_stats|mert|panns}.npy`.

```bash
python ml/preprocessing/data_splitting.py
```

## 2) Build tabular/text/target features

This writes `ml/features/` arrays for audio metadata, text stats, sentiment,
MPNet embeddings, and targets.

```bash
python ml/preprocessing/run_preprocessing.py
```

## 3) ML validation (model selection)

```bash
python ml/models/thesis_ml_models.py --eval-split val
```

## 4) Optional HPO (validation-only)

```bash
python ml/models/hpo_catboost.py
python dl/15_hpo_attention_dl.py
```

## 5) ML final test (train+val -> test)

Default models:

```bash
python ml/models/thesis_ml_models.py --eval-split test
```

With tuned params (after HPO):

```bash
python ml/models/thesis_ml_models.py --eval-split test --models CatBoost --tuned-params results/hpo/catboost_best_params.json
```

## 6) DL validation (architecture selection)

```bash
python dl/14_thesis_architecture_comparison.py --eval-split val
```

## 7) DL final test (train+val -> test)

Default architectures (retrain for final test):

```bash
python dl/14_thesis_architecture_comparison.py --eval-split test --retrain --epochs 15 
```

With tuned params (after HPO):

```bash
python dl/14_thesis_architecture_comparison.py --eval-split test --retrain --tuned-params results/hpo/attention_dl_best_params.json --architectures AttentionTaskGatedFusionMLP 
```
