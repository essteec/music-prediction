# Dataset Module

This module handles data collection, cleaning, and preprocessing for the music prediction project.

## Structure

```
dataset/
├── scripts/          # Data processing scripts
├── notebooks/        # Exploratory data analysis
├── raw/             # Original datasets (not in git)
└── processed/       # Cleaned and feature-engineered data
```

## Scripts

### Data Collection

**`chosic_scraper.py`**
- Scrapes additional metadata from Chosic.com (genre, popularity, year, explicit flag)
- Uses Selenium + BeautifulSoup
- Includes rate limiting and sample-first approach
- See [CHOSIC_SCRAPER_README.md](CHOSIC_SCRAPER_README.md) for details

**`genre_mapper.py`**
- Normalizes genre labels across different sources
- Maps specific genres to broader categories
- See [GENRE_MAPPER_README.md](GENRE_MAPPER_README.md) for details

### Data Preprocessing (To Be Created)

**`data_cleaning.py`**
- Load and validate dataset
- Handle missing values
- Remove duplicates and invalid entries
- Clean text data (lyrics)

**`feature_engineering.py`**
- Extract features from lyrics (TF-IDF, sentiment)
- Engineer audio feature combinations
- Encode categorical variables
- Create final feature matrix

## Notebooks

**`01_data_inspection.ipynb`** (To Be Created)
- Dataset overview
- Data types and sizes
- Missing value analysis
- Initial observations

**`02_eda.ipynb`** (To Be Created)
- Valence distribution
- Feature correlations
- Visualizations
- Genre analysis
- Temporal patterns

## Data Files

### Current Files
- `songs_with_attributes_and_lyrics.csv`: Main dataset (>50MB, not in git)
- `genre_mappings.csv`: Genre standardization mappings
- Sample HTML files for scraper testing

### Expected Output Files
- `processed/songs_cleaned.csv`: After cleaning
- `processed/features_engineered.csv`: With all features
- `processed/X_train.csv`, `X_test.csv`, etc.: Split datasets

## Usage

### 1. Data Collection (if needed)
```bash
python scripts/chosic_scraper.py
```

### 2. Data Cleaning
```bash
python scripts/data_cleaning.py --input songs_with_attributes_and_lyrics.csv --output processed/songs_cleaned.csv
```

### 3. Feature Engineering
```bash
python scripts/feature_engineering.py --input processed/songs_cleaned.csv --output processed/features_engineered.csv
```

### 4. Exploratory Analysis
```bash
jupyter notebook notebooks/02_eda.ipynb
```

## Data Dictionary

### Audio Features (from Spotify)
- `danceability` (0-1): Suitability for dancing
- `energy` (0-1): Intensity and activity
- `valence` (0-1): Musical positivity ← **Target Variable**
- `tempo` (BPM): Track tempo
- `loudness` (dB): Overall loudness
- `speechiness` (0-1): Presence of spoken words
- `acousticness` (0-1): Acoustic vs electronic
- `instrumentalness` (0-1): Lack of vocals
- `liveness` (0-1): Audience presence
- `key` (0-11): Musical key
- `mode` (0-1): Major/minor
- `duration_ms`: Track length

### Text Features
- `lyrics`: Full song lyrics (raw)

### Metadata
- `id`: Spotify track ID
- `name`: Song title
- `album_name`: Album name
- `artists`: Artist name(s)
- `genre`: Musical genre (scraped)
- `year`: Release year (scraped)
- `popularity`: Popularity score (scraped)
- `explicit`: Explicit content flag (scraped)

## Next Steps

1. ✅ Complete data scraping
2. ⏳ Document dataset characteristics (size, distributions)
3. ⏳ Implement data cleaning script
4. ⏳ Create EDA notebook
5. ⏳ Implement feature engineering
6. ⏳ Create train/test splits

## Notes

- Large CSV files (>50MB) are excluded from git (see .gitignore)
- Use pandas chunking for processing very large files
- Always set random seeds for reproducibility
- Keep raw data untouched; save processed versions separately
