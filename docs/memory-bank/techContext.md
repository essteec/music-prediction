# Tech Context

## Core Stack In Use

- Python, NumPy, pandas
- PyTorch for DL training
- scikit-learn/XGBoost/CatBoost/LightGBM for ML baselines
- Sentence-transformers embeddings: MiniLM historical, MPNet current text representation
- Audio embeddings: VGGish, MERT, PANNs, Mel Stats

## Data Artifacts

- Processed splits: `data/processed/{train,val,test}.csv`
- Feature arrays: `ml/features/X_{split}_{modality}.npy`
- Target arrays: `ml/features/y_{split}_{target}.npy`
- Current key modalities: `audio`, `text_stats`, `sentiment`, `mpnet`, `vggish`, `mert`, `panns`, `mel_stats`

## Key Result Artifacts

- Corrected Ultimate ML baseline: `results/metrics/ultimate_test/ultimate_results_20260517_144042.csv`
- Invalid pre-fix Ultimate result: `results/metrics/ultimate_test/ultimate_results_20260516_184227.csv`
- Best single DL model result: `results/dl_metrics/exp_f_feat_eng_20260517_231126.csv`
- Loss-tuning diagnostic result: `results/dl_metrics/exp_h_loss_tuning_20260518_001715.csv`
- Focused HPO outputs:
  - `results/hpo/attention_dl_best_params.json`
  - `results/hpo/attention_dl_hpo_val_<timestamp>.csv`
  - `results/hpo/catboost_best_params.json`
  - `results/hpo/catboost_hpo_val_<target>_<timestamp>.csv`

## Current Code Artifacts

- ML all-feature baseline script: `ml/models/ultimate_models.py`
- Existing broad ML comparison pattern: `ml/models/enhanced_models.py`
- Focused HPO scripts: `ml/models/hpo_catboost.py`, `dl/15_hpo_attention_dl.py`
- DL fusion architectures: `dl/utils/fusion.py`, `dl/utils/fusion_attention.py`, `dl/utils/fusion_wide.py`
- DL architecture scripts: `dl/06_multimodal_fusion.py` through `dl/13_loss_tuning.py`

## Operational Notes

1. Keep output metric schema stable: target, model, feature set, split, RMSE, MAE, R2.
2. Make result filenames split-explicit: `*_val_*.csv` or `*_test_*.csv`.
3. Use validation for model selection and test for final thesis reporting only.
4. Use deterministic settings for DL reproducibility.
5. Maintain cache-first workflow to avoid unnecessary recomputation.
6. Ask user to run project Python training/evaluation scripts when execution is needed.
