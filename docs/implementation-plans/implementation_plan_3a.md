# Implementation Plan 3a: Comprehensive Thesis Notebook Suite (20_* Series)

## Rationale

The 10_* thesis notebook suite is concise and narrative-heavy (~16 cells each). This plan creates a complementary 20_* suite that is **comprehensive and exploratory**, matching the depth of the original 01_*-07_* notebooks (~28 cells, 24K code chars each). The 20_* notebooks analyze **all models, all targets, all modalities** — not just the two tuned finalists. Both suites coexist: 10_* for thesis narrative, 20_* for complete exploratory depth.

## Data Sources

All notebooks use the **final test evaluation** where models were trained on `train+val` (417,059 samples) and evaluated once on `test` (74,573 samples, 4,254 features):

| CSV | Description |
|---|---:|
| `results/metrics/thesis_ml_test/thesis_ml_results_test_20260528_230741.csv` | All 8 ML families, default params |
| `results/metrics/thesis_ml_test/thesis_ml_results_test_20260601_160912.csv` | CatBoost, tuned params |
| `results/dl_metrics/final_dl_test_20260601_212018.csv` | All 5 DL architectures, default (15 epochs) |
| `results/dl_metrics/final_dl_test_20260601_224941.csv` | AttentionDL, tuned (11 epochs) |
| `results/hpo/catboost_best_params.json` | Per-target tuned CatBoost params |
| `results/hpo/attention_dl_best_params.json` | Tuned AttentionDL params |
| `results/hpo/catboost_hpo_val_*.csv` | CatBoost HPO trial records |
| `results/hpo/attention_dl_hpo_val_*.csv` | AttentionDL HPO trial records |

## Target Order (consistent across all notebooks)

Valence → Energy → Danceability → Popularity

---

## Notebook 20: Comprehensive Dataset and Target EDA

**File:** `notebooks/20_thesis_comprehensive_dataset_eda.ipynb`
**Maps to:** Old 01 (full EDA) + Old 02 (statistical tests)
**Target cells:** ~40

### Scope
Full exploratory analysis of the final aligned 550K-song dataset with 4254-feature multimodal pipeline. Covers basic dataset statistics, target variable analysis, and advanced statistical tests.

### Sections and Cells

#### A. Setup and Data Loading (4-5 cells)
- Imports centralized at top (pandas, numpy, matplotlib, seaborn, scipy, sklearn)
- `REPO = Path.home() / 'projects' / 'music-prediction'`
- Load `data/processed/{train,val,test}.csv`
- Load target arrays `ml/features/y_{split}_*.npy` for each split
- Print shapes, dtypes, memory usage per split

#### B. Dataset Overview (4-5 cells)
- **Table 1:** Split sizes and proportions (train 417K, val 74K, test 75K)
- Artist-aware split verification: unique artists per split, artist overlap check
- Feature dimension composition: bar chart of 8 modality sizes (audio 23, text_stats 5, sentiment 2, MPNet 768, VGGish 128, MERT 768, PANNs 2048, Mel Stats 512 = 4254)
- **Figure 1:** Split composition bar chart with artist counts

#### C. Target Variable Analysis (6-8 cells)
- **Figure 2:** Target distribution histograms/KDE per split (2x2 grid, all 4 targets, 3 splits overlaid)
- Descriptive statistics table: mean, std, min, max, skewness, kurtosis per target per split
- **Figure 3:** Target correlation heatmap (Pearson) — test set only, annotated
- Target range analysis: min/max verification, stationarity across splits
- **Figure 4:** Pairplot of all 4 targets (test set, colored by genre top-5)

#### D. Genre Analysis (4-5 cells)
- Genre distribution: top-10 genres bar chart across splits
- **Figure 5:** Per-target mean by genre (grouped bar chart, top-10 genres)
- ANOVA: F-statistic and p-value for genre effect on each target
- Post-hoc analysis: which genres differ most per target

#### E. Year Analysis (3-4 cells)
- Year distribution histogram across splits
- **Figure 6:** Per-target mean by year (line plot, smoothed, separate panel per target)
- Year correlation with targets

#### F. Advanced Statistical Tests (4-5 cells)
- Normality test (D'Agostino-Pearson or Shapiro, annotated: "largest known dataset to which this test is meaningful")
- Skewness/kurtosis table per target
- **Figure 7:** Q-Q plots per target (test set, 2x2 grid)
- Interaction effects: genre × year on each target (2-way ANOVA or visualization)

#### G. Key Findings Cell (2-3 cells)
- Bullet-point summary with specific metrics
- Save derived tables to `results/analysis/20_*`
- Compare to old notebook findings where applicable

---

## Notebook 21: Comprehensive Multimodal Feature Inventory

**File:** `notebooks/21_thesis_comprehensive_feature_inventory.ipynb`
**Maps to:** Old 03 (feature files EDA)
**Target cells:** ~30

### Scope
Deep inventory of all 8 feature modalities, dimension verification, quality checks, distributional analysis, correlation within and across modalities, and variance analysis.

### Sections and Cells

#### A. Setup (2-3 cells)
- Imports centralized
- Define modality dict with file paths:
  ```python
  MODALITIES = {
      'audio': (23, 'X_{split}_audio_23.npy'),
      'text_stats': (5, 'X_{split}_text_stats_5.npy'),
      'sentiment': (2, 'X_{split}_sentiment_2.npy'),
      'mpnet': (768, 'X_{split}_mpnet.npy'),
      'vggish': (128, 'X_{split}_vggish.npy'),
      'mert': (768, 'X_{split}_mert.npy'),
      'panns': (2048, 'X_{split}_panns.npy'),
      'mel_stats': (512, 'X_{split}_mel_stats.npy'),
  }
  ```
- Total verification: 23+5+2+768+128+768+2048+512 = 4254

#### B. Feature File Inventory (3-4 cells)
- **Table 1:** Modality summary — name, dimension, file existence, load time, dtype
- Verify feature order matches `ml/models/thesis_ml_models.py` concatenation order
- Train/val/test shape consistency across all modalities
- **Assertion cell:** Confirm all arrays load, shapes match, dtypes are float32/float64

#### C. Quality Checks (4-5 cells)
- NaN/Inf detection per modality: count, percentage, affected rows
- **Figure 1:** Missing value heatmap by modality (or zero-row coverage if no NaN)
- Finite value checks: all values finite per modality
- Zero-variance features per modality: count of constant columns
- Coverage: how many songs have non-zero embeddings per audio modality

#### D. Feature Distribution Analysis (5-6 cells)
- **Figure 2:** Feature distribution summary — for each modality, plot mean ± 2std across all features (or violin plot for a random sample of 5 features per modality)
- **Figure 3:** PCA visualization of each modality (2D scatter, colored by target quartile, 2x4 grid for 4 targets)
- Embedding sparsity: fraction of near-zero values per modality
- Scale range comparison across modalities (box plot of feature means)

#### E. Cross-Modality Correlation (4-5 cells)
- **Figure 4:** Inter-modality correlation heatmap — compute mean feature per modality, then correlate modality means
- Intra-modality correlation: average pairwise correlation within each modality
- **Table 2:** Within-modality correlation summary (mean, std, min, max per modality)

#### F. Variance Analysis (3-4 cells)
- Variance distribution per modality (box plot or violin)
- **Figure 5:** Cumulative variance explained by PCA per modality
- Low-variance feature identification: features with variance < 0.01, listed per modality

#### G. Summary Cell (2-3 cells)
- Key findings about feature quality, coverage, and modality characteristics
- Save summary table to `results/analysis/21_feature_inventory_summary.csv`

---

## Notebook 22: Comprehensive ML Baselines Comparison

**File:** `notebooks/22_thesis_comprehensive_ml_baselines.ipynb`
**Maps to:** Old 04 (enhanced models analysis) — ML portion
**Target cells:** ~35

### Scope
Full analysis of all 8 ML model families tested on the final test split. Includes default and tuned CatBoost. Heatmaps, top-5 per target, training time, RMSE/MAE analysis, and default vs tuned deltas.

### Sections and Cells

#### A. Setup and Data Loading (3-4 cells)
- Imports centralized
- Load `thesis_ml_results_test_20260528_230741.csv` (all 8 ML families, default)
- Load `thesis_ml_results_test_20260601_160912.csv` (CatBoost tuned)
- Print unique models, targets, and row counts

#### B. Overall ML Performance Overview (4-5 cells)
- **Table 1:** Full results table — model × target R² (styled, color-coded)
- **Figure 1:** ML family R² heatmap (8 models × 4 targets, annotated with values)
- Best model per target (highlighted in table or separate callout)
- Average R² ranking: bar chart of models sorted by mean R² across all targets

#### C. Per-Target Deep Dive (6-8 cells)
- For each target (4):
  - **Figure 2a-d:** Top-5 ML models bar chart (R² with error from CV if available)
  - Training time comparison (bar chart)
  - RMSE and MAE side-by-side
- Combined **Figure 3:** Multi-panel: top-5 per target (4 panels, shared legend)

#### D. Default vs Tuned CatBoost Comparison (4-5 cells)
- Extract default CatBoost and tuned CatBoost rows
- **Table 2:** Default vs tuned per target (R², RMSE, MAE, ΔR²)
- **Figure 4:** Default vs tuned bar chart (grouped, 4 targets)
- HPO effect summary: average improvement, per-target improvement, which targets benefited most
- Note: HPO improved avg R² from 0.6371 → 0.6413 (+0.004)

#### E. ML Family Characteristics (4-5 cells)
- **Figure 5:** Training time vs R² scatter plot (models as labeled points)
- Model type comparison: tree-based vs linear vs ensemble
- **Table 3:** Model complexity comparison (params, training time, inference time)
- Which model families are most robust across targets

#### F. RMSE and MAE Analysis (3-4 cells)
- **Figure 6:** RMSE heatmap (8 models × 4 targets)
- **Figure 7:** MAE heatmap (8 models × 4 targets)
- Scale comparison: R² vs RMSE vs MAE ranking consistency

#### G. Key Findings (2-3 cells)
- CatBoost wins overall: which targets and by what margin
- Second-best model per target
- Which models are not competitive and why
- Save comparison table to `results/analysis/22_ml_comparison_summary.csv`

---

## Notebook 23: Comprehensive DL Architecture Analysis

**File:** `notebooks/23_thesis_comprehensive_dl_architecture_analysis.ipynb`
**Maps to:** Old 04 (DL portion) + Old 07 (DL part)
**Target cells:** ~35

### Scope
Full analysis of all 5 DL architectures on the final test split. Architecture progression, per-target performance, validation-to-test consistency, training dynamics, and model complexity comparison.

### Sections and Cells

#### A. Setup and Data Loading (3-4 cells)
- Imports centralized (add seaborn, maybe numpy for aggregation)
- Load `final_dl_test_20260601_212018.csv` (5 architectures, default 15 epochs)
- Load `final_dl_test_20260601_224941.csv` (AttentionDL tuned, 11 epochs)
- Print architectures, targets, row counts

#### B. DL Architecture Overview (4-5 cells)
- Architecture descriptions table:
  | Architecture | Purpose | Encoder Params | Fusion Params | Total Params |
  |---|---|---|---|---|
  | FlatAllMLP | Simple neural baseline | — | — | ~2.1M |
  | MultiModalFusionMLP | Per-modality encoders | ~X | ~Y | ~Z |
  | TaskGatedFusionMLP | Per-target modality gates | ~X | ~Y | ~Z |
  | AttentionTaskGatedFusionMLP | Cross-modal attention | ~X | ~Y | ~Z |
  | TaskGatedFusionMLP_FeatEng | + metadata interactions | ~X | ~Y | ~Z |
  - **Note:** Read actual param counts from model definition or training logs
- **Figure 1:** Architecture complexity comparison (params vs avg R² scatter)

#### C. Architecture Progression (5-6 cells)
- **Figure 2:** Architecture progression by target — grouped bar chart (5 architectures, 4 targets, colored by architecture)
- **Figure 3:** Average R² progression (Flat → Fusion → Gated → Attention → FeatEng, line or bar)
- Per-target ranking: which architecture wins each target
- **Table 1:** Architecture ranking per target (1-5)

#### D. Per-Target Deep Dive (5-6 cells)
- For each target (4):
  - **Figure 4a-d:** All 5 architectures bar chart with R² values
  - Highlight best architecture
  - Compare to CatBoost baseline (add as horizontal line or separate bar)
- Combined **Figure 5:** Multi-panel per-target comparison (4 panels)

#### E. Validation-to-Test Consistency (4-5 cells)
- Load validation results from `results/dl_metrics/thesis_val/thesis_architecture_comparison_val_20260525_175338.csv`
- **Figure 6:** Validation R² vs Test R² scatter (each dot = architecture × target)
- Spearman rank correlation between val and test rankings
- **Table 2:** Val-to-test ranking changes (did ranking hold?)
- Check for overfitting: large val-test gaps

#### F. Default vs Tuned AttentionDL (3-4 cells)
- Extract default AttentionDL and tuned AttentionDL rows
- **Table 3:** Default vs tuned per target (R², RMSE, MAE, ΔR²)
- **Figure 7:** Default vs tuned bar chart (grouped, 4 targets)
- HPO effect: avg 0.6212 → 0.6264 (+0.005)
- Compare with CatBoost HPO effect (+0.004)

#### G. Training Dynamics (3-4 cells)
- If training logs available: loss curves per architecture
- **Figure 8:** Validation R² over epochs (if logged, one line per architecture)
- Early stopping analysis: which architectures converged fastest

#### H. Key Findings (2-3 cells)
- Strongest DL architecture overall and per target
- Architecture progression confirms: cross-modal attention adds value
- Feat Eng variant: worth the extra complexity?
- Save comparison table to `results/analysis/23_dl_comparison_summary.csv`

---

## Notebook 24: Comprehensive HPO Analysis

**File:** `notebooks/24_thesis_comprehensive_hpo_analysis.ipynb`
**Target cells:** ~30

### Scope
Full analysis of both HPO searches: CatBoost (12 trials × 4 targets) and AttentionDL (20 trials). Parameter importance, learning curves, trial comparison, and default-vs-tuned deltas.

### Sections and Cells

#### A. Setup (2-3 cells)
- Load `results/hpo/catboost_best_params.json`
- Load `results/hpo/attention_dl_best_params.json`
- Load CatBoost HPO trial CSVs from `results/hpo/catboost_hpo_val_*.csv`
- Load AttentionDL HPO trial CSV from `results/hpo/attention_dl_hpo_val_*.csv`

#### B. CatBoost HPO Analysis (6-8 cells)
- **Figure 1:** CatBoost HPO trial history per target (optimization curves, 4 panels)
- **Table 1:** CatBoost best params per target
- **Figure 2:** Parameter importance (Optuna hyperparameter importance, per target)
- **Figure 3:** Parallel coordinates plot for CatBoost trials (per target or combined)
- Default vs tuned comparison with validation R²
- **Table 2:** CatBoost default vs tuned validation R² per target

#### C. AttentionDL HPO Analysis (6-8 cells)
- **Figure 4:** AttentionDL HPO trial history (optimization curve, single panel)
- **Table 3:** AttentionDL best params
- **Figure 5:** Parameter importance (Optuna hyperparameter importance)
- **Figure 6:** Parallel coordinates plot for AttentionDL trials
- Default vs tuned comparison with validation R² (all 4 targets)
- **Table 4:** AttentionDL default vs tuned validation R² per target

#### D. Cross-Model HPO Comparison (4-5 cells)
- **Figure 7:** CatBoost vs AttentionDL — default vs tuned delta comparison (grouped bar, 4 targets)
- **Table 5:** Summary — model, default avg R², tuned avg R², ΔR², HPO budget (trials)
- Which model benefited more from HPO? By how much?
- Did HPO change the ranking between models?

#### E. Statistical Stability (3-4 cells)
- Trial distribution: histograms of validation R² for CatBoost and AttentionDL trials
- **Figure 8:** Trial R² distribution (histogram, both models overlaid if comparable)
- Best trial vs median trial: sensitivity analysis
- Did the default params fall within the trial distribution or was it an outlier?

#### F. Key Findings (2-3 cells)
- HPO improved both finalists modestly: CatBoost +0.004, AttentionDL +0.005
- Ranking unchanged: CatBoost still wins overall
- Parameter sensitivity insights
- Save HPO summary table to `results/analysis/24_hpo_summary.csv`

---

## Notebook 25: Comprehensive Feature and Modality Importance

**File:** `notebooks/25_thesis_comprehensive_feature_modality_importance.ipynb`
**Maps to:** Old 05 (feature importance) + Old 05-2 (SHAP/LIME)
**Target cells:** ~35

### Scope
Feature importance analysis using CatBoost built-in importance, modality-aggregated importance, SHAP analysis (bar + beeswarm + modality aggregation), and permutation importance comparison.

### Sections and Cells

#### A. Setup (3-4 cells)
- Load saved CatBoost tuned models from `ml/models/saved/thesis_ml_test/`
- Define feature names and modality ranges:
  ```python
  MODALITY_RANGES = {
      'audio': (0, 23),
      'text_stats': (23, 28),
      'sentiment': (28, 30),
      'mpnet': (30, 798),
      'vggish': (798, 926),
      'mert': (926, 1694),
      'panns': (1694, 3742),
      'mel_stats': (3742, 4254),
  }
  ```
- Load test features `X_test_*.npy` for SHAP
- Check SHAP availability with graceful degradation

#### B. CatBoost Built-in Feature Importance (5-6 cells)
- Extract feature importance from each target's tuned model
- **Figure 1:** Top-20 features per target (horizontal bar, 4 panels, one per target)
- **Table 1:** Top-10 features per target with names and importance scores
- Which raw (non-embedding) features appear in top-20 most often?
- Feature importance overlap across targets

#### C. Modality-Level Importance (5-6 cells)
- Aggregate importance by modality (sum or mean of feature importances within each modality range)
- **Figure 2:** Modality contribution heatmap (8 modalities × 4 targets, annotated with percentage)
- **Figure 3:** Modality contribution bar chart (stacked or grouped, per target)
- **Table 2:** Modality importance table (percentage per target, dominant modality highlighted)
- Key finding per target: which modality drives prediction

#### D. SHAP Analysis (6-8 cells)
- **Figure 4:** SHAP summary bar plot (top-20 features, 4 panels per target, or use a sample of 10K rows)
- **Figure 5:** SHAP beeswarm summary (top-15 features per target)
- **Table 3:** Top-15 SHAP features per target
- **Figure 6:** SHAP modality-level aggregation (bar chart, 8 modalities × 4 targets)
- SHAP dependence plots for top-2 features per target (optional, mark as appendix-level)

#### E. Permutation Importance Comparison (4-5 cells)
- Compute permutation importance on test set for CatBoost per target
- **Figure 7:** Permutation importance by modality (8 × 4 heatmap, comparable to Figure 2)
- **Table 4:** Built-in vs Permutation modality ranking per target
- Consistency analysis: which modalities are robust across importance methods

#### F. Modality Contribution to Thesis Narrative (3-4 cells)
- **Figure 8:** Summary radar chart — modality importance across all 4 targets (8 axes, 4 colored lines for targets)
- Combine findings into thesis-ready interpretation:
  - PANNs for Energy/Danceability
  - MERT + MPNet for Valence
  - Metadata/structured features for Popularity
- Open question: why does PANNs dominate? (2048-dim audio embedding, high capacity)

#### G. Key Findings (2-3 cells)
- Dominant modalities per target with quantified importance
- Consistency between built-in, SHAP, and permutation methods
- Implications for thesis narrative
- Save importance tables to `results/analysis/25_feature_importance_summary.csv`

---

## Notebook 26: Comprehensive Error Analysis

**File:** `notebooks/26_thesis_comprehensive_error_analysis.ipynb`
**Maps to:** Old 06 (error analysis)
**Target cells:** ~35

### Scope
Deep residual analysis for both CatBoost_tuned and AttentionDL_tuned. Residual distributions, predicted-vs-actual scatter, error by genre/year/target-range, worst-case failure analysis, per-sample export, and comparison across models.

### Sections and Cells

#### A. Setup and Prediction Export (3-4 cells)
- Load saved CatBoost tuned models (all 4 targets)
- Load test features `X_test_*.npy` and test targets `y_test_*.npy`
- Generate predictions for CatBoost on test set
- Load or generate AttentionDL predictions (load from file if available, or flag as unavailable)
- Combine predictions into a single DataFrame with metadata from `data/processed/test.csv`
- **Export:** Save full predictions CSV to `results/analysis/26_final_test_predictions.csv`

#### B. Residual Analysis (5-6 cells)
- **Figure 1:** Residual distribution per target (histogram + KDE, 2x2 grid, both models overlaid if available)
- **Table 1:** Residual summary statistics (mean, std, skewness, kurtosis, min, max per target per model)
- **Figure 2:** Q-Q plot of residuals per target (normality check)
- Residual bias analysis: mean residual significantly different from zero? (t-test)

#### C. Predicted vs Actual Analysis (4-5 cells)
- **Figure 3:** Predicted vs actual scatter/hexbin (2x2 grid, one per target, CatBoost only as primary)
- **Figure 4:** Predicted vs actual with AttentionDL overlay (if available, scatter with alpha)
- Per-target R² breakdown: what range predictions are most accurate?
- **Table 2:** R² by target quartile (bottom 25%, middle 50%, top 25% of actual values)

#### D. Error by Genre (4-5 cells)
- **Figure 5:** MAE by genre (grouped bar, top-10 genres, both models)
- **Figure 6:** R² by genre (bar chart, top-10 genres)
- **Table 3:** Best and worst predicted genres per target
- Genre-level error analysis: which genres are systematically over/under-predicted

#### E. Error by Year (3-4 cells)
- **Figure 7:** MAE by year (line plot, smoothed, per target panel)
- Prediction bias by decade: which decades are over/under-predicted
- Temporal drift analysis: does error increase for recent songs

#### F. Error by Target Range (3-4 cells)
- **Figure 8:** Absolute error by target value bins (binned scatter, one panel per target)
- Which ranges are hardest to predict (e.g., extreme valence, very low energy)
- Ceiling/floor effects at target extremes

#### G. Worst-Case Failure Analysis (4-5 cells)
- **Table 4:** Top-10 worst predictions per target (song metadata + actual vs predicted)
- **Figure 9:** Worst-predicted examples scatter (annotated with song IDs or genres)
- Pattern identification: are failures random or systematic?
- Popularity failure deep dive: why are the worst errors on Popularity?

#### H. Error Summary and Cross-Model Comparison (2-3 cells)
- **Table 5:** Overall error statistics — CatBoost vs AttentionDL (RMSE, MAE, R² per target)
- **Figure 10:** Error comparison bar chart (CatBoost vs AttentionDL, RMSE and MAE)
- Save error summary to `results/analysis/26_error_summary.csv`

#### I. Key Findings (2-3 cells)
- Most and least predictable targets
- Genre/year/systematic error patterns
- Top failure modes
- Actionable insights for thesis discussion

---

## Notebook 27: Comprehensive Final Comparison and Conclusions

**File:** `notebooks/27_thesis_comprehensive_final_comparison.ipynb`
**Maps to:** Old 07 (test evaluation analysis)
**Target cells:** ~35

### Scope
Full final comparison across all ML and DL models. Winner/margin analysis, combined leaderboard, statistical significance testing, ablation analysis, and thesis-ready conclusions.

### Sections and Cells

#### A. Setup (2-3 cells)
- Load all 4 final test CSVs
- Merge into unified DataFrame with a `family` column (ML vs DL) and `tuning` column (default vs tuned)
- Standardize model names for consistent display

#### B. Combined Leaderboard (4-5 cells)
- **Table 1:** Full leaderboard — all models/targets sorted by average R² (styled, color-coded)
- **Figure 1:** Combined heatmap — all models (ML + DL) × all targets, R² annotated
- **Figure 2:** Average R² bar chart across all models (sorted descending, color by family ML/DL)
- Top-5 models by average R²

#### C. Head-to-Head: CatBoost_tuned vs AttentionDL_tuned (1-2 cells)
- As requested: limited to 1-2 code blocks within the larger notebook
- **Table 2:** Head-to-head comparison
  | Target | CatBoost_tuned R² | AttentionDL_tuned R² | Winner | Margin |
  |---|---|---|---|---|
  | Valence | 0.7220 | 0.7214 | Tie | +0.001 |
  | Energy | 0.9212 | 0.9050 | CatBoost | +0.016 |
  | Danceability | 0.7903 | 0.7700 | CatBoost | +0.020 |
  | Popularity | 0.1316 | 0.1092 | CatBoost | +0.022 |
  | **Average** | **0.6413** | **0.6264** | **CatBoost** | **+0.015** |
- **Figure 3:** Per-target head-to-head bar chart (grouped, 4 targets)
- Brief interpretation of the near-tie on Valence

#### D. Statistical Significance Testing (4-5 cells)
- Load per-sample predictions from Notebook 26 export (`results/analysis/26_final_test_predictions.csv`)
- **Wilcoxon signed-rank test** on absolute errors: CatBoost vs AttentionDL per target
- **Table 3:** Wilcoxon test results per target (statistic, p-value, significant at α=0.05?)
- **Bootstrap test:** Paired bootstrap of R² difference (1000 resamples, 95% CI)
- **Figure 4:** Bootstrap distribution of ΔR² (CatBoost - AttentionDL) per target (4 panels, vertical line at 0)
- **Table 4:** Bootstrap results — mean ΔR², 95% CI, Pr(ΔR² > 0)
- Interpretation: Valence tie confirmed? Energy/Danceability/Popularity statistically significant?

#### E. Default vs Tuned Effect (3-4 cells)
- **Table 5:** All models — default vs tuned comparison (where both exist: CatBoost and AttentionDL)
- **Figure 5:** Default vs tuned comparison (grouped bar, both models, all 4 targets)
- Analysis: does HPO change conclusions?
- Combined HPO effect summary

#### F. Ablation Analysis (3-4 cells)
- ML family order: which modeling families are consistently strong?
- DL architecture order: does attention always add value?
- **Figure 6:** ML hierarchy — tree-based vs linear vs ensemble comparison
- **Figure 7:** DL hierarchy — flat → fusion → gated → attention comparison
- Key takeaway: which architectural choices matter most

#### G. Per-Model-Type Summary (3-4 cells)
- **Table 6:** Best model per target (overall)
- **Table 7:** Best DL per target
- **Table 8:** Best ML per target (non-tuned)
- **Table 9:** Modality contribution summary (from Notebook 25)

#### H. Thesis-Ready Conclusion (3-4 cells)
- Generate formatted LaTeX/Markdown tables for thesis inclusion
- **Figure 8:** Summary comparison figure — 4 panel: CatBoost vs AttentionDL per target with CI
- **Figure 9:** Radar chart — all 5 metrics (or all 4 targets) for CatBoost vs AttentionDL
- Top-5 key findings as bullet points
- Recommendation for thesis narrative

#### I. Save Final Artifacts (2-3 cells)
- Save combined leaderboard to `results/analysis/27_final_leaderboard.csv`
- Save head-to-head table to `results/analysis/27_head_to_head.csv`
- Save significance test results to `results/analysis/27_significance_tests.csv`
- Save all thesis-ready tables to `results/analysis/thesis_tables/`

---

## Implementation Notes

### Figure Convention
- All figures saved to `results/figures/thesis/` with notebook number prefix
- Example: `27_head_to_head_r2.png`, `27_leaderboard_heatmap.png`
- Publication quality: 300 DPI, consistent fonts, thesis-ready styling
- Consistent target colors: Valence=#4C72B0, Energy=#DD8452, Danceability=#55A868, Popularity=#C44E52

### Output Files
```text
results/analysis/
├── 20_dataset_eda_summary.csv
├── 21_feature_inventory_summary.csv
├── 22_ml_comparison_summary.csv
├── 23_dl_comparison_summary.csv
├── 24_hpo_summary.csv
├── 25_feature_importance_summary.csv
├── 26_final_test_predictions.csv
├── 26_error_summary.csv
├── 27_final_leaderboard.csv
├── 27_head_to_head.csv
├── 27_significance_tests.csv
└── thesis_tables/
    ├── final_comparison_table.tex
    └── final_comparison_table.csv
```

### Font and Styling Standards (Thesis Format)
- All markdown headers use **HTML bold tags** with larger font sizes for readability in thesis/report print:
  - Main title (H1): `<span style="font-size:16pt; font-weight:bold;">`
  - Section headers (H2): `<span style="font-size:14pt; font-weight:bold;">`
  - Subsection headers (H3): `<span style="font-size:12pt; font-weight:bold;">`
- Figure titles: bold, 11pt
- Table captions: bold, 11pt
- Code comments and print statements use consistent sentence-case, not code-style comments
- All figures use 14pt+ axis labels and 12pt tick labels

### Notebook Standards
- First code cell: centralized imports with `# %%` cell delimiters
- Use `REPO = Path.home() / 'projects' / 'music-prediction'` as root
- All paths relative to REPO
- Save figures with `plt.savefig(..., dpi=300, bbox_inches='tight')`
- Graceful degradation for optional dependencies (SHAP, DL checkpoints)
- Consistent target order: Valence, Energy, Danceability, Popularity
- Consistent model naming (no emojis, no abbreviations obscure enough to need a legend)

### Relation to 10_* Suite
- 10_* notebooks are concise thesis exhibits (~16 cells, narrative-first)
- 20_* notebooks are comprehensive exploration (~30-40 cells, code-first with full depth)
- Both use the same data sources and final test results
- 20_* suite does NOT replace 10_*; both coexist for different readers
