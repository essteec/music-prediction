# Data Validation & Normalization Roadmap
## Phase 1: Data Quality Assessment & Cleaning

**Status**: 🔥 ACTIVE (November 10, 2025)  
**Timeline**: 1-2 weeks  
**Goal**: Transform 3 scraped CSVs into a single, clean, ML-ready dataset

---

## 📊 Current State Analysis

### Files Overview
| File | Size | Lines | Status | Issues |
|------|------|-------|--------|--------|
| `songs_enhanced_full.csv` | 1.1GB | 31.8M | Primary data | NaN genres, year=0 |
| `failed_tracks.csv` | 288MB | 8.4M | Failed scrapes | Need analysis |
| `unknown_tracks.csv` | 78MB | 2.4M | Partial data | Genre detection failed |
| `songs_with_attributes_and_lyrics.csv` | 1.5GB | 42.5M | Original | Source data |
| `genre_mappings.csv` | 97KB | 4.6K | Reference | Genre normalization |

### Expected Final Dataset
- **Target**: ~950K unique songs (original dataset size)
- **Features**: Audio (13) + Text (1) + Metadata (4) = ~18 raw features
- **Targets**: valence, energy, danceability, popularity
- **Format**: Single clean CSV ready for ML pipeline

---

## 🎯 Roadmap Phases

### **PHASE 1.1: Data Profiling & Statistics** (Days 1-2)
> Goal: Understand exactly what we're dealing with

#### Step 1.1.1: Quick Sample Inspection
**Task**: Load first 1000 rows of each file to understand structure

**Action Items**:
- [ ] Create `notebooks/01_data_profiling.ipynb`
- [ ] Load samples from all 3 scraped files
- [ ] Inspect column names, dtypes, sample values
- [ ] Document schema differences (if any)

**Deliverable**: Schema documentation for each file

```python
# Pseudocode structure
import pandas as pd

# Sample inspection
enhanced_sample = pd.read_csv('songs_enhanced_full.csv', nrows=1000)
failed_sample = pd.read_csv('failed_tracks.csv', nrows=1000)
unknown_sample = pd.read_csv('unknown_tracks.csv', nrows=1000)

# Document schema
print(enhanced_sample.columns.tolist())
print(enhanced_sample.dtypes)
print(enhanced_sample.head())
```

---

#### Step 1.1.2: Statistical Analysis (Chunked Processing)
**Task**: Calculate comprehensive statistics without loading entire file

**Action Items**:
- [ ] Count total unique tracks in `songs_enhanced_full.csv`
- [ ] Calculate missing value statistics for each column
- [ ] Count NaN genres specifically
- [ ] Count year = 0 specifically
- [ ] Identify other data quality issues (negative values, outliers, duplicates)
- [ ] Generate statistical summary report

**Key Metrics to Calculate**:
```
songs_enhanced_full.csv:
├── Total rows
├── Unique track IDs (check id column uniqueness)
├── Missing values per column (%):
│   ├── id (should be 0%)
│   ├── name (should be 0%)
│   ├── album_name (can have missing)
│   ├── artists (should be 0%)
│   ├── danceability (should be 0% - TARGET) 🎯
│   ├── energy (should be 0% - TARGET) 🎯
│   ├── key (should be 0%)
│   ├── loudness (should be 0%)
│   ├── mode (should be 0%)
│   ├── speechiness (should be 0%)
│   ├── acousticness (should be 0%)
│   ├── instrumentalness (should be 0%)
│   ├── liveness (should be 0%)
│   ├── valence (should be 0% - CRITICAL TARGET) 🎯
│   ├── tempo (should be 0%)
│   ├── duration_ms (should be 0%)
│   ├── lyrics (can be empty for instrumentals)
│   ├── year (should be 0%, but zeros found ⚠️)
│   ├── genre (should be 0%, but NaNs found ⚠️)
│   └── popularity (can have missing for old songs - TARGET) 🎯
│
├── Range violations per column:
│   ├── danceability: values outside [0.0, 1.0]
│   ├── energy: values outside [0.0, 1.0]
│   ├── valence: values outside [0.0, 1.0] - CRITICAL
│   ├── key: values outside [0, 11]
│   ├── mode: values outside [0, 1]
│   ├── loudness: values outside [-60, 0]
│   ├── tempo: values outside [20, 300]
│   ├── duration_ms: values <= 0
│   ├── year: values = 0 or < 1900 or > 2025 ⚠️
│   ├── genre: NaN, empty, invalid ⚠️
│   └── popularity: values outside [0, 100]
│
├── Data type issues:
│   ├── id: must be string
│   ├── Numeric columns: must be int/float
│   └── Text columns: must be string
│
├── Target variable completeness (CRITICAL):
│   ├── valence: ___ complete (___ missing) 🎯
│   ├── energy: ___ complete (___ missing) 🎯
│   ├── danceability: ___ complete (___ missing) 🎯
│   └── popularity: ___ complete (___ missing - OK if old songs) 🎯
│
└── Duplicate track detection (by id column)
```

**Deliverable**: `reports/data_quality_report.txt` with all statistics

**Implementation Strategy**:
```python
# Comprehensive chunk processing for all columns
def analyze_large_csv(filepath, chunksize=50000):
    """
    Complete statistical analysis of dataset
    """
    stats = {
        'total_rows': 0,
        'unique_ids': set(),
        
        # Missing values tracking
        'missing_values': {},
        
        # Range violations tracking
        'range_violations': {
            'danceability': 0, 'energy': 0, 'valence': 0,
            'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0,
            'liveness': 0, 'key': 0, 'mode': 0, 'loudness': 0,
            'tempo': 0, 'duration_ms': 0, 'popularity': 0
        },
        
        # Specific issues
        'genre_issues': {
            'nan': 0,
            'empty': 0,
            'invalid': 0
        },
        'year_issues': {
            'zero': 0,
            'negative': 0,
            'future': 0,
            'too_old': 0
        },
        
        # Target variable tracking (CRITICAL)
        'target_completeness': {
            'valence': {'complete': 0, 'missing': 0},
            'energy': {'complete': 0, 'missing': 0},
            'danceability': {'complete': 0, 'missing': 0},
            'popularity': {'complete': 0, 'missing': 0}
        },
        
        # Data type issues
        'dtype_issues': {},
        
        # Duplicate tracking
        'duplicate_ids': []
    }
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        stats['total_rows'] += len(chunk)
        
        # Unique IDs
        chunk_ids = chunk['id'].tolist()
        duplicates_in_chunk = [x for x in chunk_ids if x in stats['unique_ids']]
        stats['duplicate_ids'].extend(duplicates_in_chunk)
        stats['unique_ids'].update(chunk_ids)
        
        # Missing values for ALL columns
        for col in chunk.columns:
            if col not in stats['missing_values']:
                stats['missing_values'][col] = 0
            stats['missing_values'][col] += chunk[col].isna().sum()
        
        # Range violations - Normalized features [0, 1]
        for col in ['danceability', 'energy', 'valence', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness']:
            if col in chunk.columns:
                stats['range_violations'][col] += (
                    (chunk[col] < 0) | (chunk[col] > 1)
                ).sum()
        
        # Range violations - Key (0-11)
        if 'key' in chunk.columns:
            stats['range_violations']['key'] += (
                (chunk['key'] < 0) | (chunk['key'] > 11)
            ).sum()
        
        # Range violations - Mode (0-1)
        if 'mode' in chunk.columns:
            stats['range_violations']['mode'] += (
                (chunk['mode'] < 0) | (chunk['mode'] > 1)
            ).sum()
        
        # Range violations - Loudness (-60 to 0)
        if 'loudness' in chunk.columns:
            stats['range_violations']['loudness'] += (
                (chunk['loudness'] < -60) | (chunk['loudness'] > 0)
            ).sum()
        
        # Range violations - Tempo (20-300)
        if 'tempo' in chunk.columns:
            stats['range_violations']['tempo'] += (
                (chunk['tempo'] < 20) | (chunk['tempo'] > 300)
            ).sum()
        
        # Range violations - Duration (>0)
        if 'duration_ms' in chunk.columns:
            stats['range_violations']['duration_ms'] += (
                chunk['duration_ms'] <= 0
            ).sum()
        
        # Range violations - Popularity (0-100)
        if 'popularity' in chunk.columns:
            valid_pop = chunk['popularity'].dropna()
            stats['range_violations']['popularity'] += (
                (valid_pop < 0) | (valid_pop > 100)
            ).sum()
        
        # Genre issues
        if 'genre' in chunk.columns:
            stats['genre_issues']['nan'] += chunk['genre'].isna().sum()
            stats['genre_issues']['empty'] += (chunk['genre'] == '').sum()
            # Check for 'Unknown', 'nan' string, etc.
            stats['genre_issues']['invalid'] += chunk['genre'].isin(
                ['Unknown', 'nan', 'NaN', 'None', 'null']
            ).sum()
        
        # Year issues
        if 'year' in chunk.columns:
            stats['year_issues']['zero'] += (chunk['year'] == 0).sum()
            stats['year_issues']['negative'] += (chunk['year'] < 0).sum()
            stats['year_issues']['future'] += (chunk['year'] > 2025).sum()
            stats['year_issues']['too_old'] += (
                (chunk['year'] < 1900) & (chunk['year'] != 0)
            ).sum()
        
        # Target variable completeness (CRITICAL)
        for target in ['valence', 'energy', 'danceability', 'popularity']:
            if target in chunk.columns:
                stats['target_completeness'][target]['complete'] += \
                    chunk[target].notna().sum()
                stats['target_completeness'][target]['missing'] += \
                    chunk[target].isna().sum()
    
    stats['unique_tracks'] = len(stats['unique_ids'])
    stats['total_duplicates'] = len(stats['duplicate_ids'])
    del stats['unique_ids']  # Remove set to save memory
    
    return stats
```

---

#### Step 1.1.3: Failed Tracks Analysis
**Task**: Understand why tracks failed and if we can recover any

**Action Items**:
- [ ] Count total failed tracks
- [ ] Categorize failure reasons (if stored)
- [ ] Check if failed tracks are subset of original dataset
- [ ] Decide: retry, manual fix, or accept loss
- [ ] Calculate impact on final dataset size

**Questions to Answer**:
1. What % of original dataset failed? (8.4M lines = ?)
2. Are failure reasons documented?
3. Can we extract partial data from failures?
4. Should we attempt re-scraping?

**Deliverable**: `reports/failed_tracks_analysis.txt`

---

#### Step 1.1.4: Unknown Genres Analysis
**Task**: Process tracks with successful scrapes but undetected genres

**Action Items**:
- [ ] Count total unknown genre tracks
- [ ] Check what other metadata exists (popularity, year, explicit?)
- [ ] Determine if we can infer genres from other features
- [ ] Decide: manual mapping, ML-based inference, or "Unknown" category
- [ ] Calculate impact on genre-based features

**Strategy Options**:
1. **Keep as "Unknown"**: Treat as separate genre category
2. **Drop**: If too small, exclude from dataset
3. **Infer**: Use k-NN on audio features to infer similar genres
4. **Manual**: If <1000, manually map using Spotify API

**Deliverable**: `reports/unknown_genres_analysis.txt`

---

### **PHASE 1.2: Data Cleaning Strategy** (Day 3)
> Goal: Define precise rules for handling each issue

#### Step 1.2.1: Missing Value Strategy

**Genre Handling**:
```
IF genre is NaN or empty:
    Option A: Drop row (if <5% of data)
    Option B: Infer from audio features using k-NN
    Option C: Mark as "Unknown" and create binary flag
    Option D: Use lyrics to infer genre (NLP-based)
    
DECISION: [To be made based on statistics]
RATIONALE: [Document why]
```

**Year Handling**:
```
IF year == 0:
    Option A: Drop row (if <2% of data)
    Option B: Impute with median year from same genre
    Option C: Impute with -1 and create "year_missing" flag
    Option D: Use external API to fetch correct year
    
DECISION: [To be made based on statistics]
RATIONALE: [Document why]
```

**Other Missing Values**:
- Popularity: Cannot impute (is a target variable in some models)
- Explicit: Default to False if missing (conservative)
- Audio features: Should already exist from original dataset

**Action Items**:
- [ ] Create decision matrix for each missing value type
- [ ] Document thresholds (e.g., drop if >5% missing)
- [ ] Define imputation strategies
- [ ] Create validation rules

**Deliverable**: `ml/preprocessing/cleaning_strategy.md`

---

#### Step 1.2.2: Outlier Detection Strategy

**Complete Column Validation Rules**:
```python
VALIDATION_RULES = {
    # Identity columns (no ML impact but must be valid)
    'id': {
        'type': str,
        'unique': True,
        'null_allowed': False,
        'ml_impact': None
    },
    'name': {
        'type': str,
        'null_allowed': False,
        'ml_impact': None
    },
    'album_name': {
        'type': str,
        'null_allowed': True,  # Some tracks may not have album
        'ml_impact': None
    },
    'artists': {
        'type': str,
        'null_allowed': False,
        'ml_impact': None
    },
    
    # Audio features - Normalized [0-1] range
    'danceability': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'HIGH',
        'target_variable': True,  # TARGET 🎯
        'check_decimals': True
    },
    'energy': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'HIGH',
        'target_variable': True,  # TARGET 🎯
        'check_decimals': True
    },
    'valence': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,  # CRITICAL - Primary target
        'ml_impact': 'HIGH',
        'target_variable': True,  # TARGET 🎯
        'check_decimals': True
    },
    'speechiness': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM'
    },
    'acousticness': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM'
    },
    'instrumentalness': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'MEDIUM'
    },
    'liveness': {
        'type': float,
        'range': (0.0, 1.0),
        'null_allowed': False,
        'ml_impact': 'LOW'
    },
    
    # Audio features - Categorical/Specific ranges
    'key': {
        'type': int,
        'range': (0, 11),  # 12 musical keys
        'null_allowed': False,
        'ml_impact': 'LOW'
    },
    'mode': {
        'type': int,
        'range': (0, 1),  # 0=minor, 1=major
        'null_allowed': False,
        'ml_impact': 'LOW'
    },
    'loudness': {
        'type': float,
        'range': (-60.0, 0.0),  # dB scale
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'warn_range': (-40.0, -5.0)  # Typical range
    },
    'tempo': {
        'type': float,
        'range': (20.0, 300.0),  # BPM
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'warn_range': (60.0, 180.0)  # Typical range
    },
    'duration_ms': {
        'type': int,
        'range': (1000, 3600000),  # 1 sec to 1 hour
        'null_allowed': False,
        'ml_impact': 'LOW',
        'warn_range': (30000, 600000)  # 30 sec to 10 min typical
    },
    
    # Text feature
    'lyrics': {
        'type': str,
        'null_allowed': True,  # Instrumentals have no lyrics
        'empty_allowed': True,
        'ml_impact': 'HIGH',  # Major NLP feature
        'check_encoding': True
    },
    
    # Metadata - Scraped features
    'year': {
        'type': int,
        'range': (1900, 2025),
        'null_allowed': False,
        'ml_impact': 'MEDIUM',
        'invalid_values': [0],  # KNOWN ISSUE ⚠️
        'action_on_invalid': 'flag_for_cleaning'
    },
    'genre': {
        'type': str,
        'null_allowed': False,
        'ml_impact': 'HIGH',  # Major categorical feature
        'invalid_values': ['', 'nan', 'NaN', 'None'],  # KNOWN ISSUE ⚠️
        'action_on_invalid': 'flag_for_cleaning',
        'validate_against_mapping': True  # Check genre_mappings.csv
    },
    'popularity': {
        'type': int,
        'range': (0, 100),
        'null_allowed': True,  # May be missing for old/obscure tracks
        'ml_impact': 'HIGH',
        'target_variable': True,  # TARGET 🎯
        'note': 'Can be NaN for very old songs'
    }
}
```

**Action Items**:
- [ ] Define valid ranges for each feature
- [ ] Decide on outlier handling (drop vs cap vs investigate)
- [ ] Create validation functions
- [ ] Document edge cases

**Deliverable**: `ml/preprocessing/outlier_rules.py`

---

#### Step 1.2.3: Duplicate Detection Strategy

**Duplicate Types**:
1. **Exact duplicates**: Same track_id, same features
2. **Partial duplicates**: Same track but different metadata
3. **Similar tracks**: Different versions of same song

**Strategy**:
```python
# Priority-based deduplication
1. Group by track_id
2. If multiple rows for same track:
   - Keep row with most complete metadata
   - If equal, keep row with most recent scrape
   - Document which version kept
3. Create duplicate_resolution_log.csv
```

**Action Items**:
- [ ] Identify duplicate detection key (track_id? name+artist?)
- [ ] Define deduplication priority rules
- [ ] Implement deduplication logic
- [ ] Log all duplicate resolutions

**Deliverable**: Deduplication function + resolution log

---

### **PHASE 1.3: Data Cleaning Execution** (Days 4-5)
> Goal: Apply cleaning strategies and create clean dataset

#### Step 1.3.1: Clean songs_enhanced_full.csv

**Script**: `scripts/clean_enhanced_data.py`

**Process Flow**:
```
1. Load data in chunks
2. For each chunk:
   a. Fix genre issues (apply strategy from 1.2.1)
   b. Fix year issues (apply strategy from 1.2.1)
   c. Validate outliers (apply strategy from 1.2.2)
   d. Standardize data types
   e. Create quality flags if needed
3. Concatenate cleaned chunks
4. Remove duplicates (apply strategy from 1.2.3)
5. Save to songs_enhanced_clean.csv
6. Generate cleaning report
```

**Quality Flags** (optional features):
```python
# Add these columns to track data quality
'genre_imputed': bool  # Genre was inferred/imputed
'year_imputed': bool   # Year was inferred/imputed
'data_quality_score': float  # Overall quality [0-1]
```

**Action Items**:
- [ ] Create cleaning script with progress bars
- [ ] Implement chunk-based processing
- [ ] Add extensive logging
- [ ] Create checkpoints for resumability
- [ ] Validate output after each step

**Deliverable**: `songs_enhanced_clean.csv` + cleaning log

---

#### Step 1.3.2: Merge with Original Dataset

**Script**: `scripts/merge_datasets.py`

**Process Flow**:
```
1. Load songs_enhanced_clean.csv (scraped metadata)
2. Load songs_with_attributes_and_lyrics.csv (original data)
3. Merge on track_id:
   - Left join: Keep all original tracks
   - Add scraped metadata where available
4. Handle tracks without scraped metadata:
   - Check failed_tracks.csv and unknown_tracks.csv
   - Decide inclusion based on availability
5. Final validation
6. Save to songs_final_merged.csv
```

**Merge Strategy**:
```python
# Pseudocode
original = pd.read_csv('songs_with_attributes_and_lyrics.csv')
enhanced_clean = pd.read_csv('songs_enhanced_clean.csv')

# Merge on track ID
merged = original.merge(
    enhanced_clean[['track_id', 'genre', 'year', 'popularity', 'explicit']],
    on='track_id',
    how='left'  # Keep all original tracks
)

# Track merge statistics
print(f"Original tracks: {len(original)}")
print(f"Enhanced tracks: {len(enhanced_clean)}")
print(f"Merged tracks: {len(merged)}")
print(f"Tracks with metadata: {merged['genre'].notna().sum()}")
print(f"Tracks without metadata: {merged['genre'].isna().sum()}")
```

**Action Items**:
- [ ] Implement merge logic
- [ ] Validate merge correctness (no data loss)
- [ ] Document merge statistics
- [ ] Handle unmatched tracks

**Deliverable**: `songs_final_merged.csv`

---

#### Step 1.3.3: Process Unknown Tracks (Conditional)

**Decision Point**: Only if unknown_tracks.csv contains valuable data

**Options**:
```
A. If unknown tracks have other metadata (popularity, year):
   → Extract and merge with appropriate handling
   
B. If unknown tracks only missing genre:
   → Add with genre='Unknown' or infer genre
   
C. If unknown tracks provide no value:
   → Exclude from final dataset
```

**Action Items**:
- [ ] Decide based on Step 1.1.4 analysis
- [ ] If including: create integration script
- [ ] Document decision and rationale

**Deliverable**: Decision document or integration script

---

### **PHASE 1.4: Data Validation & Quality Check** (Day 6)
> Goal: Verify cleaned data meets ML requirements

#### Step 1.4.1: Schema Validation

**Script**: `scripts/validate_schema.py`

**Checks**:
```python
REQUIRED_COLUMNS = [
    # Original audio features
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms',
    
    # Text features
    'lyrics',
    
    # Metadata
    'genre', 'year', 'popularity', 'explicit',
    
    # Identifiers
    'track_id', 'track_name', 'artist_name'
]

REQUIRED_DTYPES = {
    'danceability': 'float64',
    'energy': 'float64',
    'valence': 'float64',
    'popularity': 'int64',
    'year': 'int64',
    'explicit': 'bool',
    # ... etc
}
```

**Action Items**:
- [ ] Verify all required columns exist
- [ ] Verify data types are correct
- [ ] Check for any unexpected columns
- [ ] Validate ID uniqueness

**Deliverable**: Schema validation report (PASS/FAIL)

---

#### Step 1.4.2: Data Quality Validation

**Script**: `scripts/validate_quality.py`

**Comprehensive Checks**:
```python
VALIDATION_RULES = {
    # Range checks
    'danceability': (0, 1),
    'energy': (0, 1),
    'valence': (0, 1),
    'popularity': (0, 100),
    'year': (1900, 2025),
    'tempo': (20, 300),  # BPM
    'duration_ms': (1000, 3600000),  # 1 sec to 1 hour
    
    # Null checks (critical columns)
    'track_id': {'null_allowed': False},
    'valence': {'null_allowed': False},  # Target variable
    'energy': {'null_allowed': False},   # Target variable
    'danceability': {'null_allowed': False},  # Target variable
    'popularity': {'null_allowed': True},  # May be missing for old songs
    'lyrics': {'null_allowed': True},  # Some instrumentals lack lyrics
    'genre': {'null_allowed': False},  # Required after cleaning
    
    # Type checks
    'explicit': {'type': bool},
    'year': {'type': int},
}
```

**Action Items**:
- [ ] Run all validation rules
- [ ] Generate violation report
- [ ] Fix violations or document exceptions
- [ ] Calculate final data quality score

**Deliverable**: `reports/data_quality_validation.txt`

---

#### Step 1.4.3: Statistical Validation

**Script**: `notebooks/02_cleaned_data_analysis.ipynb`

**Analysis**:
```python
# Generate comprehensive statistics
1. Dataset size:
   - Total rows
   - Total unique tracks
   - Comparison to original dataset
   
2. Feature completeness:
   - Missing values per column (%)
   - Rows with all features complete (%)
   
3. Target variable distributions:
   - Valence: histogram, mean, std, quartiles
   - Energy: histogram, mean, std, quartiles
   - Danceability: histogram, mean, std, quartiles
   - Popularity: histogram, mean, std, quartiles
   
4. Feature correlations:
   - Correlation matrix for numerical features
   - Identify highly correlated features (>0.9)
   
5. Genre distribution:
   - Top 20 genres by count
   - Genre balance (for stratified sampling later)
   
6. Year distribution:
   - Songs per decade
   - Temporal trends in popularity
   
7. Lyrics statistics:
   - Average lyric length
   - Rows with empty lyrics
   - Language detection (if multilingual)
```

**Action Items**:
- [ ] Generate all statistics
- [ ] Create visualizations (histograms, correlation heatmaps)
- [ ] Compare with original dataset statistics
- [ ] Document any unexpected findings

**Deliverable**: Comprehensive analysis notebook with visualizations

---

### **PHASE 1.5: Data Normalization** (Day 7)
> Goal: Prepare data for ML pipeline (feature engineering pre-step)

#### Step 1.5.1: Text Normalization

**Script**: `ml/preprocessing/normalize_text.py`

**Lyrics Cleaning**:
```python
def normalize_lyrics(text):
    """
    Clean and normalize lyrics for NLP processing
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special markers (e.g., [Chorus], [Verse])
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove non-alphabetic (keep basic punctuation)
    # DON'T remove punctuation yet (needed for sentiment)
    
    # Handle empty result
    if not text.strip():
        return ""
    
    return text
```

**Action Items**:
- [ ] Implement text normalization function
- [ ] Apply to all lyrics
- [ ] Create `lyrics_cleaned` column
- [ ] Track cleaning statistics (% modified)

**Deliverable**: Dataset with cleaned lyrics

---

#### Step 1.5.2: Categorical Encoding Preparation

**Script**: `ml/preprocessing/prepare_categories.py`

**Genre Processing**:
```python
# 1. Genre normalization (using genre_mappings.csv)
# 2. Reduce to top N genres (e.g., top 50)
# 3. Map rare genres to "Other"
# 4. Create genre encoding mapping for later

def prepare_genre_encoding(df, top_n=50):
    """
    Prepare genre for encoding
    """
    # Count genres
    genre_counts = df['genre'].value_counts()
    
    # Keep top N
    top_genres = genre_counts.head(top_n).index.tolist()
    
    # Map others
    df['genre_normalized'] = df['genre'].apply(
        lambda x: x if x in top_genres else 'Other'
    )
    
    # Save mapping
    mapping = {g: i for i, g in enumerate(sorted(df['genre_normalized'].unique()))}
    
    return df, mapping
```

**Year Processing**:
```python
# Create decade feature
df['decade'] = (df['year'] // 10) * 10

# Create year categories (for analysis)
df['era'] = pd.cut(
    df['year'],
    bins=[0, 1970, 1990, 2000, 2010, 2020, 2030],
    labels=['pre-1970', '70s-80s', '90s', '2000s', '2010s', '2020s']
)
```

**Action Items**:
- [ ] Normalize genres using genre_mappings.csv
- [ ] Reduce genre cardinality (top 50 + Other)
- [ ] Create derived temporal features
- [ ] Save encoding mappings for later use

**Deliverable**: Dataset with normalized categories + encoding maps

---

#### Step 1.5.3: Final Dataset Export

**Script**: `scripts/finalize_dataset.py`

**Output Files**:
```
dataset/processed/
├── songs_ml_ready.csv          # Full cleaned dataset
├── songs_ml_ready_sample.csv   # 10K sample for quick experiments
├── data_dictionary.json        # Column descriptions
├── encoding_mappings.json      # Genre/category encodings
└── dataset_statistics.json     # Summary statistics
```

**Data Dictionary Structure**:
```json
{
  "track_id": {
    "type": "string",
    "description": "Unique Spotify track identifier",
    "nullable": false,
    "example": "7qiZfU4dY1lWllzX7mPBI"
  },
  "valence": {
    "type": "float",
    "description": "Musical positiveness [0-1]",
    "range": [0, 1],
    "nullable": false,
    "target_variable": true
  },
  // ... all columns
}
```

**Action Items**:
- [ ] Export final dataset
- [ ] Create 10K sample for fast iteration
- [ ] Generate data dictionary
- [ ] Export all mappings and statistics
- [ ] Create README for processed data

**Deliverable**: Complete processed dataset ready for ML

---

## 📁 Folder Structure for This Phase

```
bitirme/
├── dataset/
│   ├── raw/                    # Original files (move here)
│   │   ├── songs_with_attributes_and_lyrics.csv
│   │   └── songs_with_lyrics_and_timestamps.csv
│   ├── scraped/                # Scraped files (move here)
│   │   ├── songs_enhanced_full.csv
│   │   ├── failed_tracks.csv
│   │   ├── unknown_tracks.csv
│   │   └── genre_mappings.csv
│   └── processed/              # Clean outputs (create)
│       ├── songs_ml_ready.csv
│       ├── songs_ml_ready_sample.csv
│       ├── data_dictionary.json
│       ├── encoding_mappings.json
│       └── dataset_statistics.json
├── ml/
│   └── preprocessing/          # Cleaning scripts
│       ├── __init__.py
│       ├── cleaning_strategy.md
│       ├── outlier_rules.py
│       ├── normalize_text.py
│       ├── prepare_categories.py
│       └── validators.py
├── scripts/                    # Execution scripts
│   ├── clean_enhanced_data.py
│   ├── merge_datasets.py
│   ├── validate_schema.py
│   ├── validate_quality.py
│   └── finalize_dataset.py
├── notebooks/                  # Analysis notebooks
│   ├── 01_data_profiling.ipynb
│   └── 02_cleaned_data_analysis.ipynb
└── reports/                    # Generated reports
    ├── data_quality_report.txt
    ├── failed_tracks_analysis.txt
    ├── unknown_genres_analysis.txt
    └── data_quality_validation.txt
```

---

## 🔧 Technical Implementation Guide

### Recommended Libraries
```python
# Core
import pandas as pd
import numpy as np

# Text processing
import re
from collections import Counter

# Progress tracking
from tqdm import tqdm

# Validation
import json
import logging

# Visualization (for notebooks)
import matplotlib.pyplot as plt
import seaborn as sns
```

### Chunk Processing Template
```python
def process_large_csv_in_chunks(
    input_path,
    output_path,
    processing_func,
    chunksize=50000
):
    """
    Process large CSV file in chunks to avoid memory issues
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        processing_func: Function to apply to each chunk
        chunksize: Number of rows per chunk
    """
    # Setup
    first_chunk = True
    total_processed = 0
    
    # Get total lines for progress bar
    total_lines = sum(1 for _ in open(input_path)) - 1  # Exclude header
    
    # Process chunks
    with tqdm(total=total_lines, desc="Processing") as pbar:
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            # Apply processing function
            processed_chunk = processing_func(chunk)
            
            # Write to output
            processed_chunk.to_csv(
                output_path,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            
            first_chunk = False
            total_processed += len(processed_chunk)
            pbar.update(len(chunk))
    
    return total_processed
```

### Logging Setup
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

---

## ✅ Success Criteria

### Phase 1 Complete When:
- [ ] All 3 scraped CSVs analyzed and understood
- [ ] Data quality issues quantified and documented
- [ ] Cleaning strategies defined and documented
- [ ] Cleaning scripts implemented and tested
- [ ] Single merged dataset created (`songs_ml_ready.csv`)
- [ ] Dataset validates against all quality rules
- [ ] Data dictionary and mappings exported
- [ ] Comprehensive statistics generated
- [ ] All decisions documented with rationale
- [ ] Code is version controlled and commented
- [ ] Sample dataset available for quick iteration

### Quality Metrics:
- **Completeness**: >95% of rows have all critical features
- **Validity**: 100% of values pass range/type checks
- **Uniqueness**: 0 duplicate track_ids in final dataset
- **Consistency**: Genre/year/popularity align with expectations
- **Accuracy**: Spot-check 100 random rows manually

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
cd /home/esstee/documents/bitirme
python -m venv venv
source venv/bin/activate
pip install pandas numpy tqdm matplotlib seaborn jupyter

# 2. Create folder structure
mkdir -p dataset/{raw,scraped,processed}
mkdir -p ml/preprocessing
mkdir -p scripts
mkdir -p notebooks
mkdir -p reports

# 3. Move files to organized structure
mv dataset/*.csv dataset/scraped/
# (keep original songs_with_*.csv in dataset/raw/)

# 4. Start with profiling
jupyter notebook notebooks/01_data_profiling.ipynb

# 5. Run cleaning pipeline (after scripts are created)
python scripts/clean_enhanced_data.py
python scripts/merge_datasets.py
python scripts/validate_schema.py
python scripts/validate_quality.py
python scripts/finalize_dataset.py
```

---

## 📊 Expected Timeline

| Days | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Profiling & Analysis | Statistics reports + decision points |
| 3 | Strategy Definition | Cleaning strategy docs |
| 4-5 | Cleaning Execution | Clean merged dataset |
| 6 | Validation | Validation reports + fixes |
| 7 | Normalization | ML-ready dataset + documentation |

**Total**: 7-10 days depending on issues found

---

## 🎯 Next Phase Preview

After Phase 1 completion, we move to **Phase 2: Feature Engineering**:
- TF-IDF vectorization of lyrics
- Sentiment analysis extraction
- Genre encoding (one-hot or embedding)
- Feature scaling/normalization
- Train/validation/test split
- Feature selection

But first, let's get this data clean! 🧹

---

## 📝 Notes & Best Practices

1. **Always work on copies**: Never modify original scraped files
2. **Document everything**: Every decision needs rationale
3. **Version control**: Commit after each major step
4. **Test on samples first**: Validate logic on 1K rows before full run
5. **Checkpoint often**: Save intermediate results
6. **Log everything**: Comprehensive logging saves debugging time
7. **Validate early and often**: Don't wait until the end
8. **Keep it reproducible**: Set random seeds, document versions

---

**Last Updated**: November 10, 2025  
**Status**: Ready to execute  
**Owner**: Team (collaborative effort recommended)
