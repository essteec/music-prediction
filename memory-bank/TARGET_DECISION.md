# Target Variable Decision Guide

## The Question

Should we predict one target or multiple targets?

---

## 🎯 **FINAL DECISION: Predict All 4 Targets** ✅

**Decision Made**: October 10, 2025

We will predict **4 separate target variables** using **4 independent models**:
1. **Valence** (emotional positivity)
2. **Energy** (intensity/activity)
3. **Danceability** (suitability for dancing)
4. **Popularity** (track popularity score)

**Approach**: Same ML pipeline applied to each target separately, followed by comprehensive comparative analysis.

---

## Rationale for Multi-Target Approach

### Why Predict All 4 Targets?

1. **Comprehensive Thesis Scope**: Demonstrates systematic methodology across different prediction tasks
2. **Rich Comparative Analysis**: Shows which features/algorithms work best for which targets
3. **Risk Mitigation**: If one target is hard to predict, others provide strong results
4. **Reusable Pipeline**: Build once, apply four times
5. **Deeper Insights**: Understanding differences between targets IS the research contribution
6. **Realistic Timeline**: 4 targets × same pipeline = manageable in 8-10 weeks

### Technical Approach

**Four Independent Models** (NOT one multi-output model):
- Each target gets its own model trained separately
- Same preprocessing and feature engineering pipeline
- Same algorithms compared (Linear, Ridge, RF, XGBoost)
- Independent hyperparameter tuning per target
- Clearer interpretation of results

---

## Detailed Analysis of Each Target

### Target 1: Valence (Musical Positivity)

**What is it?**
- Measures emotional positivity/happiness of a song
- Scale: 0 (sad/negative) to 1 (happy/positive)
- Captures the overall emotional tone

#### ✅ Advantages

1. **Strong Lyrical Connection**
   - Lyrics naturally contain emotional content
   - Sentiment analysis directly relates to valence
   - Example: "I love this life" → high valence, "I'm so alone" → low valence
   - Makes intuitive sense: words matter for emotions

2. **Excellent NLP Showcase**
   - Demonstrates text processing skills
   - Multiple feature extraction approaches possible:
     - Sentiment scores (TextBlob, VADER)
     - TF-IDF word patterns
     - Word embeddings
     - Emotional word detection
   - Shows sophisticated feature engineering

3. **Compelling Thesis Narrative**
   - Clear research question: "Can lyrics predict emotion?"
   - Interesting to explore: Do happy words make happy songs?
   - Human-relatable topic (everyone understands emotions)
   - Good for both technical and non-technical audiences

4. **Rich Analysis Opportunities**
   - Compare pure audio vs pure lyrics vs combined
   - Discover which words/phrases indicate happiness
   - Genre-specific patterns (e.g., metal lyrics vs actual valence)
   - Interesting failure modes to analyze

5. **Academic Relevance**
   - Active research area in Music Information Retrieval
   - Lots of related work to cite
   - Clear contribution: comparing modern ML approaches

#### ❌ Challenges

1. **Complexity**
   - Valence isn't just lyrical sentiment
   - Musical factors matter: minor key can sound sad even with happy lyrics
   - Vocal delivery affects perception
   - → This is actually good! Makes the problem more interesting

2. **Subjectivity**
   - People may perceive valence differently
   - → But Spotify's metric is standardized (good for reproducibility)

3. **Moderate Difficulty**
   - Not trivially predictable (too easy = boring thesis)
   - Not impossibly hard (too hard = frustration)
   - → Just right for thesis complexity

#### Expected Performance
- R² between 0.35 - 0.55 (based on similar research)
- Better than danceability for demonstrating NLP value

---

### Target 2: Energy (Intensity/Activity)

**What is it?**
- Measures intensity and activity level
- Scale: 0 (calm/quiet) to 1 (energetic/intense)
- Based on loudness, tempo, dynamics

#### ✅ Advantages

1. **Strong Audio Feature Connection**
   - Loudness, tempo highly predictive
   - Clear physical measurements
   
2. **Objective Metric**
   - Less subjective than valence
   - Measurable attributes
   
3. **Good Baseline Expectations**
   - Should achieve high R² (0.6-0.7)
   - Builds confidence in pipeline

#### Expected Performance
- R² between 0.60 - 0.75
- Strong audio feature importance
- Lyrics may contribute less but could capture energy-related words ("fast", "loud", "intense")

---

### Target 3: Danceability

**What is it?**
- Measures how suitable a song is for dancing
- Scale: 0 (not danceable) to 1 (very danceable)
- Based on tempo, rhythm stability, beat strength

#### ✅ Advantages

1. **Rhythm-Focused**
   - Clear connection to tempo and beat
   - Objective measurement
   
2. **Interesting Comparison**
   - Shows when audio features dominate over lyrics
   - Good contrast with valence
   
3. **Moderate Difficulty**
   - Not trivial but achievable

#### Expected Performance
- R² between 0.50 - 0.65
- Audio features will dominate
- Lyrics likely contribute minimally (but testing this is valuable!)

---

### Target 4: Popularity

**What is it?**
- Measures track popularity/success
- Scale: 0-100 (based on streams, plays)
- Reflects current listener engagement

#### ✅ Advantages

1. **Real-World Relevance**
   - Practical application value
   - Interesting for music industry
   
2. **Complex Challenge**
   - Many confounding factors
   - Tests limits of your features

3. **Great Discussion Material**
   - Why is it hard to predict?
   - What's missing from the model?

#### ⚠️ Expected Challenges

1. **Missing Critical Features**
   - Artist fame, marketing budget, label support not in dataset
   - Release timing, playlist placement unknown
   
2. **Likely Lower Performance**
   - R² probably 0.30 - 0.45
   - External factors dominate
   
3. **Harder to Interpret**
   - Success depends on non-musical factors

#### Expected Performance
- R² between 0.30 - 0.45 (lowest of the four)
- Genre and year might be most predictive
- **This "failure" is valuable**: Shows limitations of content-based prediction

---

## Comparative Summary of All 4 Targets

| Criterion | Valence | Energy | Danceability | Popularity |
|-----------|---------|--------|--------------|------------|
| Lyrical relevance | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
| Audio relevance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Expected R² | 0.35-0.55 | 0.60-0.75 | 0.50-0.65 | 0.30-0.45 |
| Prediction difficulty | Moderate | Easy | Moderate | Hard |
| NLP value | High | Low | Low | Moderate |
| Interpretability | High | High | High | Low |
| Research interest | High | Moderate | Moderate | High |

### Key Insights from Multi-Target Approach

1. **Different Features Matter**: 
   - Valence → lyrics + mode + energy
   - Energy → loudness + tempo
   - Danceability → tempo + beat strength
   - Popularity → genre + year + ??? (external factors)

2. **Algorithm Performance Varies**:
   - Some targets favor linear models
   - Others benefit from tree-based models
   - This variation IS your research contribution!

3. **Comprehensive Understanding**:
   - Learn which musical attributes are "intrinsic" (predictable from content)
   - Learn which require external context (popularity)

---

## Implementation Strategy

### Phase-Based Approach

**Phase 1: Pipeline Development (Weeks 1-4)**
- Build complete pipeline for ONE target (valence recommended)
- Get end-to-end working: data → features → models → evaluation
- Document everything thoroughly

**Phase 2: Extension to Other Targets (Week 5)**
```python
# Reuse pipeline for other targets
targets = ['valence', 'energy', 'danceability', 'popularity']

for target in targets:
    results = run_ml_pipeline(
        target_variable=target,
        algorithms=['linear', 'ridge', 'rf', 'xgboost'],
        feature_sets=['audio', 'lyrics', 'metadata', 'all']
    )
    save_results(results, target)
```

**Phase 3: Comparative Analysis (Weeks 6-7)**
- Create comparison tables and visualizations
- Analyze feature importance across targets
- Identify patterns and insights

### Workload Distribution

**Person 1**: 
- Valence + Energy models
- Audio feature engineering
- Baseline implementations

**Person 2**:
- Danceability + Popularity models  
- Text feature engineering
- Evaluation framework

**Both**:
- Comparative analysis
- Thesis writing
- Literature review

---

## Why This Multi-Target Approach Works

### ✅ Advantages of Predicting All 4

1. **Comprehensive Thesis**: Suitable scope for final year project
2. **Efficient Reuse**: Build pipeline once, apply 4 times
3. **Risk Mitigation**: Multiple successful predictions likely
4. **Rich Analysis**: Compare what works for what target
5. **Clear Contribution**: "Systematic comparison across diverse targets"
6. **Professional Depth**: Shows thorough methodology
7. **No Neural Networks Needed**: Traditional ML sufficient for all 4

### 📊 Expected Thesis Structure

```
1. Introduction (5 pages)
   - Music prediction importance
   - Research questions
   - Multi-target approach justification

2. Literature Review (10 pages)
   - Music information retrieval
   - Emotion prediction (valence)
   - Audio feature prediction
   - Popularity prediction studies

3. Methodology (15 pages) ← DESCRIBE ONCE!
   - Dataset description
   - Preprocessing pipeline
   - Feature engineering (audio, text, metadata)
   - Algorithms (Linear, Ridge, RF, XGBoost)
   - Evaluation metrics (RMSE, R², MAE)
   - Cross-validation strategy

4. Experiments (30 pages)
   4.1 Valence Prediction (8 pages)
       - Results, feature importance, best model
   4.2 Energy Prediction (7 pages)
       - Results, feature importance, best model
   4.3 Danceability Prediction (7 pages)
       - Results, feature importance, best model
   4.4 Popularity Prediction (8 pages)
       - Results, why it's harder, limitations

5. Comparative Analysis (12 pages) ← YOUR CONTRIBUTION!
   5.1 Which targets are easiest/hardest?
   5.2 Which features matter for which targets?
   5.3 Which algorithms work best per target?
   5.4 Role of lyrics vs audio vs metadata
   5.5 Implications for music prediction tasks

6. Discussion (8 pages)
   - Interpretation of findings
   - Limitations
   - Intrinsic vs extrinsic predictability
   - Future work

7. Conclusion (3 pages)

Total: ~90 pages (typical thesis length)
```

### 🎯 Your Research Contribution

**Not**: "We built a valence predictor"  
**But**: "We systematically compared ML approaches across 4 diverse musical attributes and discovered which features and algorithms work best for which prediction tasks"

This is a **methodology contribution** - valuable for future researchers!

---

## Timeline Estimate

### Realistic 8-10 Week Schedule

**Weeks 1-2: Setup & EDA**
- ✅ Dataset specification
- ✅ Literature review (10 papers)
- ✅ Abstract draft
- EDA for all 4 targets
- Distribution analysis
- Correlation studies

**Weeks 3-4: Feature Engineering**
- Text preprocessing (lyrics)
- TF-IDF extraction
- Sentiment analysis
- Audio feature scaling
- Metadata encoding
- Train/test/val splits

**Week 5: First Target (Valence)**
- Train all 5 algorithms
- Hyperparameter tuning
- Evaluation
- Visualization
- **Get pipeline working end-to-end**

**Week 6: Extend to Other Targets**
- Energy prediction (1 day)
- Danceability prediction (1 day)
- Popularity prediction (1 day)
- Code refinement (2 days)

**Week 7: Evaluation & Analysis**
- Generate all comparison tables
- Create visualizations
- Feature importance analysis
- Statistical testing
- Results documentation

**Week 8-9: Thesis Writing**
- Write results sections
- Comparative analysis chapter
- Discussion
- Introduction & conclusion
- Figure polishing

**Week 10: Final Polish**
- Revisions
- Proofreading
- GitHub cleanup
- Presentation prep

---

## Next Steps (This Week)

1. ✅ **Decision Made**: Multi-target approach confirmed
2. **Update Documentation**: 
   - Update abstract for 4 targets
   - Update project brief in Memory Bank
3. **Dataset Analysis**: 
   - Check distributions for all 4 targets
   - Look for imbalances, outliers
   - Calculate correlations between targets
4. **Literature Review**: 
   - Find papers on each target type
   - Focus on comparative studies
5. **Team Discussion**:
   - Confirm with partner
   - Divide targets (Person 1: valence+energy, Person 2: dance+popularity)
   - Discuss with advisor

---

## Success Criteria

By end of project, you will have:

✅ **4 Complete Prediction Models**
- Each with 5 algorithms compared
- Full evaluation metrics
- Feature importance analysis

✅ **Comprehensive Comparison**
- Which targets are predictable from content?
- Which features matter most per target?
- Which algorithms win per target?

✅ **Strong Thesis**
- Clear methodology (reusable)
- Rich results (4 × 5 = 20 experiments)
- Insightful discussion
- Professional visualizations

✅ **Portfolio-Ready GitHub**
- Well-structured code
- Reproducible pipeline
- Clear documentation
- Example usage

---

## Questions to Consider

Before starting implementation:

- [ ] All 4 targets have good data quality (check for missing values)
- [ ] Dataset distributions are reasonable for all targets
- [ ] Team partner agrees with task division
- [ ] Advisor approves multi-target scope (if needed)
- [ ] Timeline seems achievable (adjust if needed)
- [ ] You understand why 4 separate models (not one multi-output)

---

## Final Thought

**This multi-target approach is excellent for your thesis.** It's:
- **Comprehensive** without being overwhelming
- **Systematic** (same methodology, different targets)
- **Achievable** (reuse code, parallel work possible)
- **Valuable** (comparison IS your contribution)
- **Professional** (demonstrates mature approach)

Your thesis will tell a story: 

> "We asked: can musical attributes be predicted from audio features and lyrics? We systematically tested this across four diverse targets - valence (emotion), energy (intensity), danceability (rhythm), and popularity (success). We discovered that intrinsic musical properties (valence, energy, danceability) are highly predictable from content alone, while extrinsic properties (popularity) depend heavily on external factors. Our comparative analysis reveals which features and algorithms work best for which prediction tasks, providing a roadmap for future music information retrieval research."

**That's a compelling thesis. Let's build it!** 🎯🎵📊
