# Project Brief: ML Music Prediction Thesis

## Project Overview
Final year thesis project comparing machine learning algorithms for predicting musical attributes from lyrics and audio features. Two-person collaborative project focused on song characteristic prediction.

## Core Objectives
1. **Primary Goal**: Compare multiple ML algorithms for predicting song characteristics
2. **Target Variables**: Predict 4 musical attributes - valence, energy, danceability, and popularity
3. **Approach**: Multi-target prediction using 4 independent models with comparative analysis
4. **Academic Deliverable**: Comprehensive thesis documenting methodology, experiments, and findings
5. **Collaboration**: GitHub-based two-person team project with clear structure

## Dataset
- **Source**: Spotify songs dataset with attributes and lyrics
- **Current Features**: 
  - Audio features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration
  - Metadata: track id, name, album, artists
  - Text data: lyrics
- **Enhancements**: Custom scraper built to add missing values (popularity, explicit flag, genre, release year) from Chosic.com

## Target Decision ✅ FINALIZED

**Decision Made**: October 10, 2025

We will predict **4 target variables** using **4 separate models**:

1. **Valence** (0-1): Emotional positivity
   - Strong lyrical connection, NLP showcase
   - Expected R²: 0.35-0.55
   
2. **Energy** (0-1): Intensity/activity  
   - Strong audio feature connection
   - Expected R²: 0.60-0.75
   
3. **Danceability** (0-1): Dance suitability
   - Rhythm and tempo focused
   - Expected R²: 0.50-0.65
   
4. **Popularity** (0-100): Track success
   - Complex, external factors involved
   - Expected R²: 0.30-0.45

**Rationale**: Comprehensive multi-target approach provides rich comparative analysis (which features/algorithms work best for which targets), demonstrates systematic methodology, and mitigates risk through multiple successful predictions.
   
## Success Criteria
- Working ML pipeline from data preprocessing to prediction (reusable for all targets)
- 4 complete prediction models (valence, energy, danceability, popularity)
- Comparison of 5 ML algorithms per target (baseline, linear, ridge, RF, XGBoost)
- Comprehensive comparative analysis across targets
- Well-documented thesis with clear methodology
- Reproducible results with proper evaluation metrics
- Clean GitHub repository suitable for portfolio
- 20 total experiments (4 targets × 5 algorithms = professional depth)

## Timeline
- Current week (Oct 7-14, 2025): Dataset specification, reference thesis collection, abstract writing
- Project must be structured for ongoing development over academic term

## Technical Scope
- **Languages**: Python
- **Domain**: Music information retrieval, NLP, ML regression/classification
- **Deliverables**: Code, dataset, thesis document, results analysis
