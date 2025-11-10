# Phase 1: Data Validation & Cleaning - Progress Checklist

**Start Date**: November 10, 2025  
**Target Completion**: November 17-20, 2025  
**Current Status**: 🟡 Not Started

---

## 📋 Pre-Flight Checks

- [ ] Python environment setup (pandas, numpy, tqdm, jupyter installed)
- [ ] Folder structure created (raw/, scraped/, processed/, scripts/, notebooks/, reports/)
- [ ] Files organized into appropriate folders
- [ ] Git repository up to date
- [ ] Jupyter notebook server accessible

---

## 📊 PHASE 1.1: Data Profiling (Days 1-2)

### Step 1.1.1: Sample Inspection ⏱️ 2 hours
- [ ] Create `notebooks/01_data_profiling.ipynb`
- [ ] Load first 1000 rows of `songs_enhanced_full.csv`
- [ ] Load first 1000 rows of `failed_tracks.csv`
- [ ] Load first 1000 rows of `unknown_tracks.csv`
- [ ] Document schema for each file
- [ ] Take notes on any unexpected findings

**Output**: Schema documentation

---

### Step 1.1.2: Statistical Analysis ⏱️ 4-6 hours
- [ ] Write chunk processing script
- [ ] Count total rows in `songs_enhanced_full.csv`
- [ ] Count unique track IDs
- [ ] Calculate missing values per column
- [ ] Count NaN genres (exact number)
- [ ] Count year = 0 (exact number)
- [ ] Check for other year anomalies (< 1900, > 2025)
- [ ] Analyze popularity distribution
- [ ] Check explicit flag distribution
- [ ] Detect duplicates by track_id
- [ ] Generate `reports/data_quality_report.txt`

**Key Metrics Needed**:
```
✓ Total rows: ___________
✓ Unique tracks: ___________
✓ NaN genres: ___________
✓ Year = 0: ___________
✓ Duplicates: ___________
✓ Missing popularity: ___________
```

---

### Step 1.1.3: Failed Tracks Analysis ⏱️ 2-3 hours
- [ ] Load and inspect `failed_tracks.csv`
- [ ] Count total failed tracks
- [ ] Identify failure reasons (if available)
- [ ] Calculate % of original dataset that failed
- [ ] Check if any useful data can be extracted
- [ ] Make decision: retry / manual fix / accept loss
- [ ] Generate `reports/failed_tracks_analysis.txt`

**Decision Point**: 
```
Action: [ ] Retry  [ ] Manual Fix  [ ] Accept Loss
Rationale: _________________________________
```

---

### Step 1.1.4: Unknown Genres Analysis ⏱️ 2-3 hours
- [ ] Load and inspect `unknown_tracks.csv`
- [ ] Count total unknown genre tracks
- [ ] Check what other metadata exists (popularity, year?)
- [ ] Evaluate completeness of unknown tracks
- [ ] Explore genre inference options
- [ ] Make decision on handling strategy
- [ ] Generate `reports/unknown_genres_analysis.txt`

**Decision Point**:
```
Action: [ ] Keep as "Unknown"  [ ] Drop  [ ] Infer  [ ] Manual Map
Rationale: _________________________________
```

---

## 🧹 PHASE 1.2: Cleaning Strategy (Day 3)

### Step 1.2.1: Missing Value Strategy ⏱️ 2 hours
- [ ] Define genre handling rules
- [ ] Define year handling rules
- [ ] Define popularity handling rules
- [ ] Define explicit flag handling rules
- [ ] Document thresholds (e.g., drop if >5% missing)
- [ ] Create `ml/preprocessing/cleaning_strategy.md`

**Decisions Documented**:
```
Genre NaN: _________________________________
Year = 0: _________________________________
Missing popularity: _________________________________
```

---

### Step 1.2.2: Outlier Detection Strategy ⏱️ 2 hours
- [ ] Define valid ranges for audio features
- [ ] Define valid ranges for metadata
- [ ] Decide on outlier handling (drop/cap/investigate)
- [ ] Create `ml/preprocessing/outlier_rules.py`

---

### Step 1.2.3: Duplicate Handling Strategy ⏱️ 1 hour
- [ ] Define duplicate detection key (track_id)
- [ ] Define deduplication priority rules
- [ ] Document merge strategy

---

## 🔧 PHASE 1.3: Cleaning Execution (Days 4-5)

### Step 1.3.1: Clean Enhanced Data ⏱️ 4-6 hours
- [ ] Create `scripts/clean_enhanced_data.py`
- [ ] Implement chunk-based processing
- [ ] Fix NaN genres (apply strategy)
- [ ] Fix year = 0 (apply strategy)
- [ ] Validate outliers
- [ ] Standardize data types
- [ ] Remove duplicates
- [ ] Add quality flags (optional)
- [ ] Test on sample (1000 rows)
- [ ] Run on full dataset
- [ ] Generate `songs_enhanced_clean.csv`
- [ ] Create cleaning log

**Progress**:
```
Rows processed: ___________
Rows cleaned: ___________
Rows dropped: ___________
Issues fixed: ___________
```

---

### Step 1.3.2: Merge Datasets ⏱️ 3-4 hours
- [ ] Create `scripts/merge_datasets.py`
- [ ] Load `songs_enhanced_clean.csv`
- [ ] Load `songs_with_attributes_and_lyrics.csv`
- [ ] Implement merge logic (left join on track_id)
- [ ] Validate merge correctness
- [ ] Handle unmatched tracks
- [ ] Document merge statistics
- [ ] Generate `songs_final_merged.csv`

**Merge Stats**:
```
Original tracks: ___________
Enhanced tracks: ___________
Merged tracks: ___________
With metadata: ___________
Without metadata: ___________
```

---

### Step 1.3.3: Process Unknown Tracks ⏱️ 1-2 hours (if needed)
- [ ] Decision made in Step 1.1.4: __________
- [ ] If integrating: create integration script
- [ ] If dropping: document rationale
- [ ] Update final dataset accordingly

---

## ✅ PHASE 1.4: Validation (Day 6)

### Step 1.4.1: Schema Validation ⏱️ 1 hour
- [ ] Create `scripts/validate_schema.py`
- [ ] Check all required columns exist
- [ ] Verify data types are correct
- [ ] Check for unexpected columns
- [ ] Validate ID uniqueness
- [ ] Generate schema validation report

**Result**: [ ] PASS  [ ] FAIL

---

### Step 1.4.2: Quality Validation ⏱️ 2 hours
- [ ] Create `scripts/validate_quality.py`
- [ ] Run range checks on all features
- [ ] Run null checks on critical columns
- [ ] Run type checks
- [ ] Generate violation report
- [ ] Fix any violations found
- [ ] Generate `reports/data_quality_validation.txt`

**Quality Score**: __________ / 100

---

### Step 1.4.3: Statistical Validation ⏱️ 3-4 hours
- [ ] Create `notebooks/02_cleaned_data_analysis.ipynb`
- [ ] Calculate dataset size metrics
- [ ] Calculate feature completeness
- [ ] Analyze target variable distributions (valence, energy, danceability, popularity)
- [ ] Generate correlation matrix
- [ ] Analyze genre distribution
- [ ] Analyze year distribution
- [ ] Calculate lyrics statistics
- [ ] Create visualizations
- [ ] Compare with original dataset

**Key Stats**:
```
Final dataset size: ___________
Completeness: ____________%
Valence mean: __________
Energy mean: __________
Danceability mean: __________
Popularity mean: __________
```

---

## 🎨 PHASE 1.5: Normalization (Day 7)

### Step 1.5.1: Text Normalization ⏱️ 2-3 hours
- [ ] Create `ml/preprocessing/normalize_text.py`
- [ ] Implement lyrics cleaning function
- [ ] Apply to all lyrics
- [ ] Create `lyrics_cleaned` column
- [ ] Track cleaning statistics

**Text Stats**:
```
Rows with lyrics: ___________
Empty lyrics after cleaning: ___________
Avg lyric length: ___________
```

---

### Step 1.5.2: Categorical Encoding Prep ⏱️ 2-3 hours
- [ ] Create `ml/preprocessing/prepare_categories.py`
- [ ] Normalize genres using genre_mappings.csv
- [ ] Reduce to top 50 genres + "Other"
- [ ] Create decade feature from year
- [ ] Create era categories
- [ ] Save encoding mappings

**Category Stats**:
```
Unique genres (before): ___________
Unique genres (after): ___________
Top 5 genres: ___________
```

---

### Step 1.5.3: Final Export ⏱️ 1-2 hours
- [ ] Create `scripts/finalize_dataset.py`
- [ ] Export `songs_ml_ready.csv`
- [ ] Create 10K sample: `songs_ml_ready_sample.csv`
- [ ] Generate `data_dictionary.json`
- [ ] Export `encoding_mappings.json`
- [ ] Export `dataset_statistics.json`
- [ ] Create README for processed data
- [ ] Move all files to `dataset/processed/`

---

## 🎯 Final Deliverables Checklist

### Files Created:
- [ ] `dataset/processed/songs_ml_ready.csv`
- [ ] `dataset/processed/songs_ml_ready_sample.csv`
- [ ] `dataset/processed/data_dictionary.json`
- [ ] `dataset/processed/encoding_mappings.json`
- [ ] `dataset/processed/dataset_statistics.json`
- [ ] `dataset/processed/README.md`

### Scripts Created:
- [ ] `scripts/clean_enhanced_data.py`
- [ ] `scripts/merge_datasets.py`
- [ ] `scripts/validate_schema.py`
- [ ] `scripts/validate_quality.py`
- [ ] `scripts/finalize_dataset.py`

### Notebooks Created:
- [ ] `notebooks/01_data_profiling.ipynb`
- [ ] `notebooks/02_cleaned_data_analysis.ipynb`

### Reports Created:
- [ ] `reports/data_quality_report.txt`
- [ ] `reports/failed_tracks_analysis.txt`
- [ ] `reports/unknown_genres_analysis.txt`
- [ ] `reports/data_quality_validation.txt`

### Documentation Created:
- [ ] `ml/preprocessing/cleaning_strategy.md`
- [ ] `ml/preprocessing/outlier_rules.py`
- [ ] `ml/preprocessing/normalize_text.py`
- [ ] `ml/preprocessing/prepare_categories.py`

---

## 🏆 Success Criteria

- [ ] **Completeness**: >95% rows have all critical features
- [ ] **Validity**: 100% values pass range/type checks
- [ ] **Uniqueness**: 0 duplicate track_ids
- [ ] **Consistency**: Genre/year/popularity align with expectations
- [ ] **Documentation**: All decisions documented with rationale
- [ ] **Version Control**: All code committed to git
- [ ] **Reproducibility**: Sample dataset runs successfully

---

## 📝 Notes & Issues Log

### Issues Encountered:
```
1. [Date] Issue: ___________
   Solution: ___________

2. [Date] Issue: ___________
   Solution: ___________
```

### Important Decisions:
```
1. Genre handling: ___________
2. Year handling: ___________
3. Unknown tracks: ___________
4. Failed tracks: ___________
```

### Time Tracking:
```
Phase 1.1: _____ hours
Phase 1.2: _____ hours
Phase 1.3: _____ hours
Phase 1.4: _____ hours
Phase 1.5: _____ hours
Total: _____ hours
```

---

## ✅ Sign-Off

- [ ] All tasks completed
- [ ] All deliverables created
- [ ] All quality checks passed
- [ ] Documentation complete
- [ ] Code reviewed and committed
- [ ] Ready to proceed to Phase 2: Feature Engineering

**Completed By**: ___________  
**Completion Date**: ___________  
**Reviewer**: ___________  
**Approval Date**: ___________

---

**Next Phase**: Feature Engineering (TF-IDF, sentiment, encoding, scaling, train/test split)
