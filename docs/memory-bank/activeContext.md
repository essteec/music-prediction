# Active Context: Current Work Focus

## Current Status

**Phase**: Live app preprocessing standardization and debug  
**Status**: Final test evaluation is complete. Gradio app now serves the tuned CatBoost and tuned AttentionDL models on the 4,254-feature input space.  
**Current Goal**: Ensure app preprocessing (both base metadata and audio embedding extraction) matches training preprocessing exactly, so live predictions are valid.  
**Next**: Verify popularity output normalization (log1p revert via expm1) with debug logging.

---

## Final Test Results Summary (June 2026)

All models trained on `train+val` (417,059 samples), evaluated once on `test` (74,573 samples), 4,254 features.

### Best ML: CatBoost_tuned

| Target | R² |
|---|---:|
| Valence | 0.7220 |
| Energy | 0.9212 |
| Danceability | 0.7903 |
| Popularity | 0.1316 |
| Average | 0.6413 |

### Best DL: AttentionTaskGatedFusionMLP_tuned

| Target | R² |
|---|---:|
| Valence | 0.7214 |
| Energy | 0.9050 |
| Danceability | 0.7700 |
| Popularity | 0.1092 |
| Average | 0.6264 |

### Head-to-Head

| Target | CatBoost_tuned | AttentionDL_tuned | Winner | Margin |
|---|---:|---:|---:|---|---:|
| Valence | 0.7220 | 0.7214 | Tie | +0.001 |
| Energy | **0.9212** | 0.9050 | CatBoost | +0.016 |
| Danceability | **0.7903** | 0.7700 | CatBoost | +0.020 |
| Popularity | **0.1316** | 0.1092 | CatBoost | +0.022 |
| **Avg** | **0.6413** | **0.6264** | **CatBoost** | **+0.015** |

### HPO Effect

- CatBoost: default 0.6371 → tuned 0.6413 (+0.004)
- Attention DL: default 0.6212 → tuned 0.6264 (+0.005)
- HPO confirmed the ranking, did not change it.

---

## Key Conclusions From Test Results

1. CatBoost is the strongest model overall (avg R² 0.641), winning 3/4 targets.
2. DL ties CatBoost only on Valence (0.722 vs 0.721), suggesting cross-modal attention helps emotion prediction.
3. Architecture progression (Flat → Fusion → Gated → Attention) adds measurable value on test.
4. HPO improved both finalists modestly but did not change the ML vs DL conclusion.
5. Popularity remains the hardest target (best R² 0.132), driven by external factors not in audio/lyrics.

---

## Methodology Applied

1. Validation split used for model/architecture selection (completed earlier).
2. HPO conducted on validation only: CatBoost (12 trials/target), AttentionDL (20 trials).
3. Test split used exactly once for final reporting.
4. ML trained on train+val, DL retrained on train+val for fixed budgets.
5. All results include R², RMSE, MAE per target.

---

## Preferred DL Architecture Set

5 architectures in `dl/14_thesis_architecture_comparison.py`:

1. `FlatAllMLP`: simple concatenated all-feature neural baseline (4254d).
2. `MultiModalFusionMLP`: per-modality encoders with concatenation fusion.
3. `TaskGatedFusionMLP`: per-target modality gates.
4. `AttentionTaskGatedFusionMLP`: advanced cross-modal attention variant.
5. `TaskGatedFusionMLP_FeatEng`: best-optimized variant with metadata interactions.

---

## Comprehensive Notebook Suite (Complete)

- Notebooks: `notebooks/20_*` through `notebooks/27_*`
- Figures: `results/figures/thesis/`

---

## Recent App Preprocessing Standardization

Changes applied (June 2026) to `app/` only (no training pipeline modifications):

### Base Metadata
- Removed target fields (energy, danceability, valence) from automatic heuristic extractor — they are model outputs, not inputs.
- Added explicit required-field validation for automatic mode.
- UI now shows `Base Metadata Source` with two honest options: `Manual Spotify-like metadata` vs `Estimated from uploaded audio`.
- Added reliability warning when heuristic mode is active.

### Feature Contract Validation
- Added shape/dimension/NaN checks on every feature block before model prediction.
- Validates total concatenated width (4254) for ML and per-block widths for DL.

### YouTube Downloads
- Now uses training-time yt-dlp setting: format `251/bestaudio`, saves as Opus/WebM, no WAV re-encode.
- Eliminates FFmpegExtractAudio postprocessor mismatch.

### Uploaded Audio
- Converted to standardized Opus/WebM via FFmpeg before embedding extraction.

### Mel Stats Sample Rate
- Fixed: changed from `sr=16000` (app default) to `sr=22050` (training default) with explicit `n_fft=2048`, `hop_length=512` matching `scripts/audio-embedding-extraction/extract_mel_stats.py`.

### Preprocessing Parity Checker
- `validate_preprocessing_against_saved()` extended to check multiple rows against saved training feature arrays with configurable tolerance.

---

## Immediate Next Actions

1. **Verify popularity normalization**: debug log raw (pre-expm1) model outputs to confirm log1p revert is working as expected.
2. **Optional**: Install `soundfile` to eliminate librosa PySoundFile warnings.
3. **Optional**: Statistical significance tests (paired bootstrap or Wilcoxon) on per-sample predictions to strengthen the "tie on Valence" claim.
4. **Keep app in sync**: any future training pipeline changes to extraction/preprocessing must be mirrored in the app.
