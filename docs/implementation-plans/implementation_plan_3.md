# Implementation Plan 3: Thesis Notebook Refresh

## Summary

Create a new compact thesis notebook suite under notebooks/ using 10_ through 15_ prefixes. Do not modify the existing 01_-07_ notebooks. The new notebooks will use the current multimodal 4,254-feature setup, final ML/DL test results, HPO outputs, and audio embedding artifacts.
The implementation artifact should be implementation_plan_3.md, containing this plan. The notebooks themselves are planned only here and should not be created until execution is
requested outside Plan Mode.

## New Notebook Suite

Create these .ipynb files:

1. notebooks/10_thesis_dataset_and_target_eda.ipynb
    - Focus: updated dataset/split EDA for the final aligned dataset.
    - Inputs: data/processed/{train,val,test}.csv, ml/features/y_*_*.npy.
    - Figures: split-size bar chart, target distributions by split, target correlation heatmap, genre/year coverage, artist-aware split sanity check.
    - Thesis purpose: establish dataset validity, target difficulty, and split integrity.
2. notebooks/11_thesis_multimodal_feature_inventory.ipynb
    - Focus: current feature blocks and audio/text embedding coverage.
    - Inputs: ml/features/X_{split}_{modality}.npy.
    - Feature order must match ml/models/thesis_ml_models.py: audio 23, text_stats 5, sentiment 2, MPNet 768, VGGish 128, MERT 768, PANNs 2048, Mel Stats 512.
    - Figures: modality dimension table, 4,254-feature composition chart, audio-embedding zero-row/coverage chart, finite/missing-value checks, train/val/test shape table.
    - Thesis purpose: document the final multimodal representation and prove alignment.
3. notebooks/12_thesis_final_ml_vs_dl_comparison.ipynb
    - Focus: final test comparison.
    - Inputs:
        - results/metrics/thesis_ml_test/thesis_ml_results_test_20260528_230741.csv
        - results/metrics/thesis_ml_test/thesis_ml_results_test_20260601_160912.csv
        - results/dl_metrics/final_dl_test_20260601_212018.csv
        - results/dl_metrics/final_dl_test_20260601_224941.csv
    - Figures: per-target R² head-to-head, average R² comparison, RMSE/MAE comparison, winner/margin table, ML family heatmap.
    - Thesis purpose: main results section showing CatBoost tuned vs Attention DL tuned.
4. notebooks/13_thesis_architecture_and_hpo_analysis.ipynb
    - Focus: architecture progression and focused HPO.
    - Inputs:
        - results/dl_metrics/thesis_val/thesis_architecture_comparison_val_20260525_175338.csv
        - results/dl_metrics/final_dl_test_20260601_212018.csv
        - results/hpo/catboost_best_params.json
        - results/hpo/attention_dl_best_params.json
        - HPO trial CSVs in results/hpo/
    - Figures: DL architecture progression by target, validation-to-test comparison, CatBoost default vs tuned, Attention DL default vs tuned, HPO trial history.
    - Thesis purpose: show that HPO modestly improved both finalists but did not change the ML-vs-DL conclusion.
5. notebooks/14_thesis_modality_and_interpretability_analysis.ipynb
    - Focus: interpretable feature/modality contribution using both built-in importance and SHAP values.
    - Inputs: final CatBoost tuned models in ml/models/saved/thesis_ml_test/, feature block definitions from ml/models/thesis_ml_models.py, current feature arrays.
    - Figures: CatBoost feature importance aggregated by modality, per-target modality contribution heatmap, SHAP summary plots for top structured features, modality-level SHAP contribution summary.
    - Thesis purpose: provide model-agnostic (SHAP) and model-specific explanations of why CatBoost performs strongly and which modality groups matter by target.
6. notebooks/15_thesis_error_analysis_and_significance.ipynb
    - Focus: residual analysis, failure modes, and optional statistical support.
    - Inputs: final saved CatBoost tuned models, final test feature arrays, data/processed/test.csv, final DL checkpoint if usable.
    - Outputs when run: results/analysis/final_test_predictions.csv and results/analysis/final_error_summary.csv.
    - Figures: residual distributions by target, predicted-vs-actual scatter/hexbin, absolute error by target range, error by genre/year bins, worst-case examples, paired bootstrap or
      Wilcoxon summary for CatBoost vs Attention DL.
    - Thesis purpose: support the “Valence tie” claim and explain remaining weaknesses, especially Popularity.

## Implementation Details

- Save all generated figures under results/figures/thesis/ with notebook-number prefixes, for example 12_ml_vs_dl_r2_by_target.png.
- Save derived analysis tables under results/analysis/.
- Keep old notebooks untouched and treat them as archived Semester 1/Experiment 2 material.
- Use split-explicit final artifacts only. Do not use old misnamed *_test results affected by validation/test confusion.
- Use concise, thesis-ready visuals: labeled axes, consistent target order, consistent model names, and no emoji-heavy titles.
- For residual analysis, first export predictions from the saved final models. If the tuned DL final checkpoint is not available, run full residual plots for CatBoost and restrict DL
  comparison to aggregate metrics unless a valid DL prediction path is confirmed.

## Test Plan

- Open each new notebook and run from top to bottom without manual path edits.
- Verify every notebook uses current artifacts, not obsolete 414-feature-only Experiment 2 paths.
- Confirm every figure/table uses the target order: Valence, Energy, Danceability, Popularity.
- Confirm final comparison values match:
  - CatBoost tuned average R²: 0.6413
  - Attention DL tuned average R²: 0.6264
  - Valence near-tie: 0.7220 vs 0.7214
- Confirm created analysis outputs are reproducible and saved outside notebooks/.

## Assumptions

- 1X_ means numeric 10_ through 15_ prefixes.
- New artifacts should be .ipynb notebooks.
- The suite should be compact and thesis-focused rather than a one-for-one rewrite of every old notebook.
- Per-sample residual/error analysis is desired and should include prediction export.
