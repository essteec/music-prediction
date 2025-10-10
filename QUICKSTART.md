# 🎵 Quick Start Guide

Welcome to your ML Music Prediction thesis project! This guide gets you started quickly.

## 📌 Quick Links

- **Main Documentation**: [README.md](README.md)
- **Target Variable Decision**: [memory-bank/TARGET_DECISION.md](memory-bank/TARGET_DECISION.md) ⭐
- **ML Pipeline Guide**: [memory-bank/ML_ROADMAP.md](memory-bank/ML_ROADMAP.md)
- **Collaboration Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Project Status**: [memory-bank/progress.md](memory-bank/progress.md)

## 🎯 Key Decision: Choose Valence!

**Recommendation**: Predict **valence** (emotional positivity) as your primary target.

**Why?**
- ✅ Strong connection to lyrics (showcases NLP skills)
- ✅ Compelling thesis narrative
- ✅ Rich feature engineering opportunities
- ✅ Perfect difficulty level

See [memory-bank/TARGET_DECISION.md](memory-bank/TARGET_DECISION.md) for detailed analysis.

## 📋 This Week's Tasks (Oct 7-14, 2025)

From your timeline:

### 1. Specify Dataset ✓
- [x] Dataset identified: Spotify songs with lyrics
- [ ] Document exact size (number of songs)
- [ ] Check valence distribution
- [ ] Create data dictionary
- [ ] Calculate missing value statistics

**Action**: Run this to check dataset:
```bash
cd dataset/scripts
python -c "
import pandas as pd
df = pd.read_csv('../songs_with_attributes_and_lyrics.csv')
print(f'Total songs: {len(df)}')
print(f'Valence range: {df[\"valence\"].min():.3f} - {df[\"valence\"].max():.3f}')
print(f'Valence mean: {df[\"valence\"].mean():.3f}')
print(f'Missing lyrics: {df[\"lyrics\"].isna().sum()}')
"
```

### 2. Get 10 Similar Thesis for Reference
**Search for**:
- Music emotion prediction + machine learning
- Valence prediction from lyrics
- Audio feature classification
- Music information retrieval

**Where to search**:
- Google Scholar
- IEEE Xplore
- ACM Digital Library
- arXiv

**Document in**: `thesis/references/`

See [thesis/README.md](thesis/README.md) for reference template.

### 3. Write Abstract
**Structure**:
1. Problem & motivation (2-3 sentences)
2. Approach (2-3 sentences)  
3. Methods (1-2 sentences)
4. Expected results (2-3 sentences)
5. Contribution (1-2 sentences)

**See example in**: [thesis/README.md](thesis/README.md#abstract-writing-this-week)

## 🚀 Next Steps (After This Week)

### Week 2: Setup & EDA
1. Setup virtual environment
2. Install dependencies
3. Create EDA notebook
4. Visualize data distributions

### Week 3-4: Feature Engineering
1. Text preprocessing
2. TF-IDF extraction
3. Sentiment analysis
4. Audio feature scaling

### Week 5-6: Model Training
1. Baseline models
2. Linear models
3. Tree-based models
4. Hyperparameter tuning

### Week 7-8: Evaluation
1. Model comparison
2. Feature importance
3. Error analysis
4. Generate figures

## 📁 Project Structure

```
bitirme/
├── dataset/           # Data collection & preprocessing
│   ├── scripts/      # Your scraper and processing scripts ✓
│   └── notebooks/    # For EDA (create these)
├── ml/               # Machine learning pipeline
│   ├── preprocessing/  # Feature engineering
│   ├── models/        # Model implementations
│   ├── evaluation/    # Metrics and visualization
│   └── experiments/   # Experiment orchestration
├── thesis/           # Academic writing
│   ├── references/   # Papers (collect this week!)
│   └── sections/     # Thesis chapters
├── memory-bank/      # Project knowledge (read first!) ⭐
└── results/          # Generated outputs (models, figures)
```

## 🤝 Team Collaboration

### GitHub Workflow
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit
3. Push: `git push origin feature/your-feature`
4. Create Pull Request
5. Partner reviews
6. Merge after approval

### Task Division (Suggested)
- **Person 1**: Data preprocessing, feature engineering, baseline models
- **Person 2**: Advanced models, evaluation, visualization
- **Both**: Literature review, thesis writing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🛠️ Environment Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 4. Test Installation
```python
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost
import nltk
print('All dependencies installed successfully!')
"
```

## 📖 Understanding the Memory Bank

The **memory-bank/** directory contains the project's knowledge base. These files document all important decisions, context, and progress.

**Read these files to understand the project**:

1. **projectbrief.md** - What we're building and why
2. **productContext.md** - Problems being solved
3. **systemPatterns.md** - Architecture and design
4. **techContext.md** - Technologies and tools
5. **activeContext.md** - Current focus (most important!)
6. **progress.md** - What's done, what's next
7. **TARGET_DECISION.md** - Valence vs danceability analysis ⭐
8. **ML_ROADMAP.md** - Step-by-step ML implementation guide ⭐

## 💡 Pro Tips

### For Success
1. **Start small**: Get baseline working before advanced models
2. **Document as you go**: Don't leave it until the end
3. **Set random seeds**: Ensure reproducibility (`random_state=42`)
4. **Save everything**: Models, scalers, results
5. **Visualize early**: Plots help understand data
6. **Ask questions**: Use GitHub Issues for discussion

### For Collaboration
1. **Communicate often**: Weekly sync meetings
2. **Review each other's code**: Catch mistakes early
3. **Write clear commits**: Future you will thank you
4. **Don't push large files**: Use .gitignore
5. **Be kind in reviews**: Constructive feedback

### For Thesis
1. **Write early**: Start abstract and intro now
2. **Keep experiment notes**: Document surprises
3. **Generate figures**: Visualizations make great thesis content
4. **Track references**: Use reference manager (Zotero, Mendeley)
5. **Get feedback**: Show drafts to advisor

## 🆘 Getting Help

**Stuck on something?**

1. Check documentation:
   - README files in each folder
   - Memory bank files
   - ML_ROADMAP.md for step-by-step guide

2. Search similar projects:
   - GitHub: "music emotion prediction"
   - Papers on Google Scholar

3. Ask your partner:
   - Create GitHub Issue
   - Use PR comments for code questions

4. Consult advisor:
   - For major decisions
   - When truly stuck

## ✅ This Week Checklist

- [ ] Read memory-bank/TARGET_DECISION.md
- [ ] Confirm valence as target (team + advisor)
- [ ] Document dataset characteristics
- [ ] Find 10 reference papers
- [ ] Write abstract first draft
- [ ] Setup virtual environment
- [ ] Install dependencies
- [ ] Push code to GitHub (if not already)
- [ ] Discuss task division with partner

## 🎯 Success Criteria

By end of project, you should have:

- ✅ Working ML pipeline (data → models → evaluation)
- ✅ Comparison of 3-5 algorithms
- ✅ Well-documented code in GitHub
- ✅ Complete thesis with figures and analysis
- ✅ Reproducible results
- ✅ Portfolio-ready project

---

**Remember**: This is a learning experience! Don't stress about perfection. Focus on:
1. Understanding the methods
2. Clear documentation
3. Solid methodology
4. Interesting insights

**You've got this!** 🎓🎵

---

**Next Action**: Read [memory-bank/TARGET_DECISION.md](memory-bank/TARGET_DECISION.md) and confirm valence as your target variable.
