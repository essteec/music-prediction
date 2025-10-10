# Progress: What's Done and What's Next

## ✅ Completed

### Dataset Phase
- [x] **Data Source Identified**: Spotify songs with attributes and lyrics
- [x] **Base Dataset Acquired**: CSV with audio features, lyrics, and metadata
- [x] **Scraper Development**: Built Chosic.com scraper for missing values
  - Scrapes: popularity, genre, release year, explicit flag
  - Uses: Selenium + BeautifulSoup
  - Features: Headless mode, rate limiting, sample-first approach
- [x] **Genre Mapping**: Genre normalization system implemented
- [x] **Data Enhancement**: Successfully scraped additional metadata

### Project Structure
- [x] **Basic Folder Structure**: dataset/, ml/, thesis/, timeline/ created
- [x] **Version Control**: Git initialized
- [x] **Documentation**: Scraper README created
- [x] **Memory Bank**: Complete knowledge base established

### Planning
- [x] **Project Scope Defined**: ML comparison thesis on music prediction
- [x] **Timeline Started**: Week 1 tasks identified (Oct 7-14, 2025)

## 🚧 In Progress

### Week 1 Tasks (Oct 7-14, 2025)
- [ ] **Dataset Specification**: Document full dataset characteristics
  - Size, features, distributions, missing values
  - Create data dictionary
- [ ] **Reference Collection**: Gather 10 similar theses/papers
  - Focus on music prediction, lyric analysis, audio feature ML
- [ ] **Abstract Writing**: First draft of thesis abstract
  - Pending target variable decision

### Critical Decisions
- [x] **Target Variable Selection**: ✅ FINALIZED October 10, 2025
  - **Decision**: Predict all 4 targets (valence, energy, danceability, popularity)
  - **Approach**: 4 separate models using same ML pipeline
  - **Rationale**: Comprehensive comparative analysis, risk mitigation, rich insights
  - Awaiting team partner confirmation and advisor approval

## 📋 To Do (Upcoming)

### Project Setup (High Priority)
- [ ] **GitHub Repository Setup**
  - Create well-structured repo
  - Add README, .gitignore, LICENSE
  - Setup collaboration workflow (branching strategy)
  - Invite team partner

- [ ] **Environment Setup**
  - Create virtual environment
  - Consolidate requirements.txt (root level)
  - Document setup instructions
  - Test reproducibility

- [ ] **Project Structure Enhancement**
  - Create subdirectories in ml/
  - Add README files to each folder
  - Setup notebook directory
  - Create config/ for experiment parameters

### Data Pipeline (Next Phase)
- [ ] **Data Validation & Cleaning**
  - Check for duplicates
  - Handle missing values
  - Remove invalid entries
  - Document cleaning decisions

- [ ] **Exploratory Data Analysis (EDA)**
  - Jupyter notebook with visualizations
  - Feature distributions
  - Correlation analysis
  - Target variable analysis (valence distribution)
  - Identify outliers

- [ ] **Feature Engineering**
  - Text preprocessing pipeline (lowercase, tokenization, stopwords)
  - TF-IDF vectorization for lyrics
  - Sentiment analysis extraction
  - Word embeddings (optional)
  - Feature scaling/normalization
  - Train/test/validation split

### ML Pipeline (Core Work)
- [ ] **Baseline Models**
  - Mean predictor
  - Simple linear regression
  - Establish baseline performance

- [ ] **Model Implementation**
  - Ridge/Lasso regression
  - Random Forest
  - XGBoost/LightGBM
  - Optional: Neural network, SVM

- [ ] **Evaluation Framework**
  - Cross-validation setup
  - Metrics calculation (RMSE, R², MAE)
  - Results visualization
  - Statistical significance testing

- [ ] **Comparison Analysis**
  - Performance comparison across models
  - Feature importance analysis
  - Error analysis
  - Visualizations for thesis

### Documentation & Thesis
- [ ] **Code Documentation**
  - Docstrings for all functions
  - Inline comments
  - README updates

- [ ] **Thesis Writing**
  - Abstract ✅ (this week)
  - Introduction
  - Literature Review
  - Methodology
  - Experiments & Results
  - Discussion
  - Conclusion
  - References

- [ ] **Reproducibility**
  - Requirements.txt complete
  - Setup instructions tested
  - Notebook checkpoints saved
  - Model artifacts versioned

### Final Deliverables
- [ ] **GitHub Repository**
  - Clean, well-organized code
  - Comprehensive README
  - Example usage
  - Results summary

- [ ] **Thesis Document**
  - Complete academic paper
  - Figures and tables
  - Proper citations
  - Formatted per university guidelines

- [ ] **Presentation**
  - Slides for defense
  - Demo (optional)
  - Key findings highlighted

## 🐛 Known Issues

### Dataset
- **Issue**: Main CSV file >50MB (cannot open directly in VS Code)
  - **Impact**: Need alternative tools for inspection
  - **Solution**: Use pandas chunking, command-line tools, or Jupyter

### Scraper
- **Issue**: HTML parsing is fragile (relies on page structure)
  - **Impact**: May break if Chosic.com changes layout
  - **Solution**: Use more robust selectors (TODO in code), error handling
  
- **Issue**: Fixed sleep interval (3 seconds)
  - **Impact**: Inefficient, may still miss dynamically loaded content
  - **Solution**: Use WebDriverWait for specific elements

### Collaboration
- **Issue**: GitHub not yet setup
  - **Impact**: Cannot work simultaneously on code
  - **Solution**: Priority task for next phase

- **Issue**: Task division not finalized
  - **Impact**: Potential duplicated or missed work
  - **Solution**: Create task assignment document

## 📊 Project Status Overview

### Overall Progress: ~15%

**Phase Breakdown**:
- ✅ Data Collection: 80% (scraping done, validation pending)
- 🚧 Data Preprocessing: 0% (not started)
- 📋 Feature Engineering: 0% (not started)
- 📋 Model Training: 0% (not started)
- 📋 Evaluation: 0% (not started)
- 🚧 Thesis Writing: 5% (planning phase)

### Timeline Health: ✅ On Track
- Week 1 tasks are achievable
- Early planning phase is appropriate
- Scope is well-defined

## 🎯 Next Milestone

**Milestone 1: Complete Data Preparation (Target: End of Week 2-3)**
- Dataset fully cleaned and documented
- EDA complete with insights
- Features engineered and ready
- Train/test splits created
- First baseline model trained

**Success Criteria**:
- Can load and process full dataset efficiently
- Have visualizations showing data characteristics
- Baseline RMSE/R² established
- Code is in GitHub with proper documentation

## 📈 Evolution of Decisions

### Target Variable Decision
- **Initial**: Uncertain between valence, danceability, or popularity
- **Intermediate**: Considered single-target (valence) approach
- **Final Decision**: Multi-target approach - all 4 targets (valence, energy, danceability, popularity)
- **Reasoning**: Comprehensive scope, rich comparative analysis, efficient pipeline reuse, risk mitigation
- **Status**: ✅ Decision finalized, documented in Memory Bank

### Scope Evolution
- **Initial Thought**: Single model, single target
- **Evolution**: Comparative study of multiple algorithms
- **Final Scope**: 4 targets × 5 algorithms = 20 experiments
- **Reasoning**: Comprehensive comparison IS the research contribution
- **Key Insight**: Not just "we predicted X", but "we discovered which approaches work best for which tasks"

### Collaboration Approach
- **Initial**: Unclear division
- **Current Plan**: Person-based or phase-based division with GitHub workflow
- **Next**: Formalize task assignments
