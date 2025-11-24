# Product Context: ML Music Prediction System

## Why This Exists
This project serves as a final year thesis exploring the intersection of music information retrieval, natural language processing, and machine learning. The goal is to understand whether and how song lyrics combined with audio features can predict musical characteristics.

## Problems Being Solved

### Academic Problem
- **Research Gap**: Comparing ML approaches for music attribute prediction using multimodal data (lyrics + audio features)
- **Learning Objective**: Hands-on experience with complete ML pipeline from data collection to model evaluation
- **Contribution**: Systematic comparison of algorithms with reproducible methodology

### Technical Problem
- **Data Gap**: ✅ RESOLVED - Dataset enhancement complete (November 10, 2025)
- **Data Quality**: ✅ COMPLETE - Cleaned dataset ready for ML pipeline (732,988 songs, English-only)
- **Prediction Challenge**: Can textual data (lyrics) enhance prediction of perceptual attributes (danceability, valence, energy, popularity)?
- **Model Selection**: Which algorithms work best for this specific domain?

## How It Should Work

### User Experience (For Evaluators/Reviewers)
1. **Clear Documentation**: 
   - Thesis explains methodology, decisions, and findings
   - README guides through repository structure
   - Code is well-commented and reproducible

2. **Reproducible Pipeline**:
   - Dataset preparation scripts clearly documented
   - Model training can be re-run with same results
   - Evaluation metrics are transparent

3. **Comparative Analysis**:
   - Multiple algorithms tested on same data
   - Clear performance comparison
   - Insights about which approaches work and why

### System Flow
```
Raw Dataset → ✅ Data Enrichment (Scraping COMPLETE) → 
✅ Data Validation & Cleaning (COMPLETE) → 
Feature Engineering → Model Training (4 Targets × Multiple Algorithms) → 
Evaluation & Comparison → Thesis Findings
```

**Final Dataset**: `data/processed/english_ml_ready.csv` (732,988 songs, English-only)
**Training Split**: 386,399 train / 82,187 val / 82,274 test (artist-aware)

## Target Variables - FINALIZED (October 10, 2025)

### Multi-Target Approach: All 4 Targets
**Decision**: Predict all 4 musical attributes using same ML pipeline

1. **Valence** (Emotional Positivity)
   - **Range**: 0-1 (negative to positive emotional tone)
   - **Expected R²**: 0.35-0.55
   - **Key Features**: Lyrics + mood features
   - **Difficulty**: Moderate

2. **Energy** (Intensity/Activity)
   - **Range**: 0-1 (low to high energy)
   - **Expected R²**: 0.60-0.75
   - **Key Features**: Loudness + tempo
   - **Difficulty**: Easy

3. **Danceability** (Dance Suitability)
   - **Range**: 0-1 (suitability for dancing)
   - **Expected R²**: 0.50-0.65
   - **Key Features**: Tempo + beat
   - **Difficulty**: Moderate

4. **Popularity** (Track Success)
   - **Range**: 0-100 (current track popularity)
   - **Expected R²**: 0.30-0.45
   - **Key Features**: Genre + year (external factors limit prediction)
   - **Difficulty**: Hard

### Why Multi-Target?
- **Comprehensive**: Demonstrates systematic methodology
- **Efficient**: Build pipeline once, apply to 4 targets
- **Risk Mitigation**: Success even if one target is challenging
- **Rich Analysis**: Comparative insights are the research contribution
- **Achievable**: 8-10 week timeline with traditional ML (no neural networks required)

## Key Design Goals
1. **Academic Rigor**: Proper train/test splits, cross-validation, statistical significance
2. **Reproducibility**: Version control, seed setting, documented dependencies
3. **Clarity**: Clear code structure, well-documented decisions
4. **Collaboration**: Two team members can work independently on different components
5. **Portfolio Quality**: GitHub-ready project demonstrating ML skills
