# Contributing Guidelines

This is a collaborative two-person thesis project. These guidelines help us work together effectively.

## 🤝 Collaboration Principles

1. **Communication First**: Discuss major changes before implementing
2. **Code Review**: All code goes through review before merging
3. **Documentation**: Comment your code, update READMEs
4. **Reproducibility**: Ensure your work can be reproduced by your partner
5. **Respect**: Be constructive in feedback, supportive in challenges

## 🔄 Git Workflow

### Branch Strategy

We use **feature branch workflow**:

```
main (stable, working code only)
  ├── feature/data-cleaning (Person 1)
  ├── feature/text-features (Person 2)
  ├── experiment/random-forest (Person 1)
  └── docs/thesis-methodology (Person 2)
```

### Branch Naming Conventions

- `feature/<description>`: New features or functionality
- `fix/<description>`: Bug fixes
- `experiment/<model-name>`: ML experiments
- `docs/<section>`: Documentation updates
- `refactor/<component>`: Code refactoring

**Examples**:
- `feature/sentiment-analysis`
- `experiment/xgboost-tuning`
- `fix/missing-values-handling`
- `docs/related-work`

### Workflow Steps

#### 1. Starting New Work

```bash
# Update your local main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 2. Making Changes

```bash
# Make your changes
# Test your code
# Add files
git add <files>

# Commit with clear message
git commit -m "Add sentiment analysis feature extraction"

# Push to remote
git push origin feature/your-feature-name
```

#### 3. Creating Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template:
   - **What**: What does this change do?
   - **Why**: Why is this change needed?
   - **How**: How does it work?
   - **Testing**: How was it tested?
5. Request review from your partner
6. Address review comments
7. Merge after approval

#### 4. After Merge

```bash
# Switch back to main
git checkout main

# Pull latest changes
git pull origin main

# Delete local feature branch
git branch -d feature/your-feature-name
```

## 📝 Commit Message Guidelines

### Format
```
<type>: <short summary>

<optional detailed description>

<optional issue reference>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring (no feature change)
- `test`: Adding or updating tests
- `experiment`: ML experiments and analysis

### Examples

**Good**:
```
feat: Add TF-IDF vectorization for lyrics

Implemented TfidfVectorizer with max_features=1000 and 
removed stopwords. This will be used as baseline text 
representation for valence prediction.
```

```
experiment: Train Random Forest with hyperparameter tuning

Tested max_depth [5, 10, 20] and n_estimators [100, 200, 500].
Best params: max_depth=10, n_estimators=200
RMSE: 0.156, R²: 0.623
```

**Bad**:
```
fixed stuff
```
```
updated code
```

## 🔍 Code Review Guidelines

### As Reviewer

**Check for**:
- ✅ Code works and achieves stated goal
- ✅ No breaking changes to existing functionality
- ✅ Clear variable/function names
- ✅ Comments for complex logic
- ✅ No hardcoded paths (use relative paths or config)
- ✅ Updated documentation if needed
- ✅ Reproducible (random seeds set, dependencies documented)

**Review Etiquette**:
- Be specific: "Consider using pandas groupby instead of loop here"
- Be kind: "Great approach! One suggestion: ..."
- Ask questions: "Why did you choose this threshold?"
- Acknowledge good work: "Nice error handling here!"

### As Author

**Before Requesting Review**:
- Test your code thoroughly
- Run any existing tests
- Update README/docs if needed
- Ensure code follows project style
- Remove debug prints and commented-out code

**Responding to Review**:
- Don't take feedback personally
- Ask for clarification if unsure
- Implement suggestions or explain alternative approach
- Thank reviewer for their time

## 📂 File Organization

### Where Things Go

**Raw Data**: `dataset/raw/` (NOT in git, use .gitignore)
**Processed Data**: `dataset/processed/`
**Scripts**: In appropriate subfolder (`dataset/scripts/`, `ml/preprocessing/`, etc.)
**Notebooks**: `dataset/notebooks/` or `ml/notebooks/`
**Models**: `results/models/` (saved with timestamp and params)
**Figures**: `results/figures/` or `thesis/figures/`

### Naming Conventions

**Files**:
- Python: `lowercase_with_underscores.py`
- Notebooks: `01_eda.ipynb`, `02_baseline_models.ipynb` (numbered)
- Data: `songs_processed_v1.csv`, `features_engineered_20251010.csv`
- Models: `rf_model_20251010_r2_0.65.pkl`

**Variables/Functions**:
- Functions: `snake_case`: `extract_sentiment_features()`
- Classes: `PascalCase`: `TextFeatureExtractor`
- Constants: `UPPER_CASE`: `MAX_FEATURES = 1000`

## 🧪 Testing Guidelines

### What to Test
- Data loading functions
- Feature extraction functions
- Data transformation pipelines
- Model training/prediction (smoke tests)

### Running Tests
```bash
pytest tests/
```

### Writing Tests
```python
# tests/test_text_features.py
import pytest
from ml.preprocessing.text_features import extract_sentiment

def test_sentiment_extraction():
    text = "I love this song, it makes me so happy!"
    sentiment = extract_sentiment(text)
    assert sentiment > 0  # Should be positive
    assert -1 <= sentiment <= 1  # In valid range
```

## 📊 Experiment Tracking

### Document Your Experiments

Create entry in experiment log:

```markdown
## Experiment: Random Forest Baseline
**Date**: 2025-10-15
**Branch**: experiment/rf-baseline
**Person**: [Your Name]

### Goal
Establish baseline performance with Random Forest regressor.

### Configuration
- Features: Audio features only (no lyrics)
- Model: RandomForestRegressor
- Params: n_estimators=100, max_depth=10, random_state=42
- Split: 80/20 train/test

### Results
- Train RMSE: 0.142
- Test RMSE: 0.158
- R² Score: 0.612
- Training time: 23s

### Observations
- Model overfits slightly (train < test RMSE)
- Energy and tempo are top features
- Consider adding lyrics features next

### Next Steps
- Add TF-IDF features
- Try hyperparameter tuning
```

## 🚫 What NOT to Commit

Add to `.gitignore`:
```
# Large data files
*.csv
dataset/raw/
dataset/processed/

# Model files (unless small)
*.pkl
*.h5
*.pt

# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.log
```

**Exception**: Small example/sample files are OK (add explicitly with `git add -f`)

## 📅 Meeting Schedule

### Weekly Sync (Suggested)
- **When**: [Decide together - e.g., Monday 3pm]
- **Duration**: 30-60 minutes
- **Agenda**:
  - What did you complete?
  - What are you working on?
  - Any blockers or questions?
  - Plan next week's tasks

### Quick Updates
- Use GitHub Issues for async communication
- Use PR comments for code-specific discussion
- Use [Messaging App] for urgent matters

## 🎯 Task Management

### Using GitHub Issues

**Create Issue For**:
- New features to implement
- Bugs to fix
- Questions or discussions
- Thesis sections to write

**Issue Template**:
```markdown
**Type**: [Feature/Bug/Question/Documentation]

**Description**:
Clear description of what needs to be done

**Acceptance Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2

**Assigned To**: [Person 1/Person 2/Both]
**Priority**: [High/Medium/Low]
**Estimated Time**: [X hours/days]
```

### Issue Labels
- `data`: Data-related tasks
- `ml`: ML/modeling tasks
- `thesis`: Writing/documentation
- `bug`: Something broken
- `question`: Need discussion
- `high-priority`: Urgent
- `good-first-issue`: Easy tasks for getting started

## 🔐 Security & Privacy

- **No API Keys**: Never commit API keys or credentials
- **Use Environment Variables**: For any secrets
- **Sensitive Data**: Don't commit personal/copyrighted lyrics without permission

## 🆘 Getting Help

**Stuck on Something?**
1. Check documentation (README, memory-bank/)
2. Search GitHub Issues
3. Ask your partner
4. Create issue with "question" label
5. Consult advisor if needed

**Found a Bug?**
1. Check if already reported
2. Create issue with clear reproduction steps
3. Assign to person who worked on that code (or yourself)

## ✨ Best Practices

### For Reproducibility
- Set random seeds: `np.random.seed(42)`, `random_state=42`
- Document library versions: `pip freeze > requirements.txt`
- Use relative paths: `os.path.join()` or `pathlib.Path`
- Include setup instructions in notebooks

### For Collaboration
- Write code others can understand
- Don't optimize prematurely (clarity > cleverness)
- Small, focused PRs (easier to review)
- Update docs when changing functionality

### For Thesis
- Keep experiment notes as you go
- Save figures with descriptive names
- Document surprising findings immediately
- Track all hyperparameters and results

---

**Remember**: We're in this together! Good collaboration makes both the project and the experience better. 🎓🎵

