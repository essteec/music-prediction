# Product Context: ML Music Prediction System

## Why This Exists
This project serves as a final year thesis exploring the intersection of music information retrieval, natural language processing, and machine learning. The goal is to understand whether and how song lyrics combined with audio features can predict musical characteristics.

## Problems Being Solved

### Academic Problem
- **Research Gap**: Comparing ML approaches for music attribute prediction using multimodal data (lyrics + audio features)
- **Learning Objective**: Hands-on experience with complete ML pipeline from data collection to model evaluation
- **Contribution**: Systematic comparison of algorithms with reproducible methodology

### Technical Problem
- **Data Gap**: Original dataset missing key features (popularity, genre, release year, explicit flag)
- **Prediction Challenge**: Can textual data (lyrics) enhance prediction of perceptual attributes (danceability, valence)?
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
Raw Dataset → Data Enrichment (Scraping) → Feature Engineering → 
Model Training (Multiple Algorithms) → Evaluation & Comparison → 
Thesis Findings
```

## Target Variables Under Consideration

### Option 1: Valence (Emotional Positivity)
- **Range**: 0-1 (negative to positive emotional tone)
- **Why Interesting**: Strong connection to lyrical content sentiment
- **Use Case**: "Can lyrics predict how happy/sad a song feels?"

### Option 2: Danceability
- **Range**: 0-1 (suitability for dancing)
- **Why Interesting**: Mix of rhythm, tempo, and beat strength
- **Use Case**: "Can we predict dance-friendliness from audio + lyrics?"

### Option 3: Popularity
- **Range**: 0-100 (current track popularity)
- **Why Interesting**: Real-world relevance, but many external factors
- **Use Case**: "What makes a song popular?"

## Key Design Goals
1. **Academic Rigor**: Proper train/test splits, cross-validation, statistical significance
2. **Reproducibility**: Version control, seed setting, documented dependencies
3. **Clarity**: Clear code structure, well-documented decisions
4. **Collaboration**: Two team members can work independently on different components
5. **Portfolio Quality**: GitHub-ready project demonstrating ML skills
