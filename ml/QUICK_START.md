# Phase 1 Quick Start Guide
**Data Validation & Cleaning - Your First Steps**

---

## 🚀 Ready to Start? Here's What You Need to Know

### 📁 What We Created for You

1. **`ml/DATA_VALIDATION_ROADMAP.md`** ⭐ MAIN GUIDE
   - Comprehensive 7-day plan
   - 5 phases with detailed steps
   - Code examples and templates
   - Expected outcomes for each step
   - **READ THIS FIRST!**

2. **`ml/PHASE1_CHECKLIST.md`** ✅ PROGRESS TRACKER
   - Checkbox for every task
   - Time estimates
   - Decision points
   - Notes section
   - **USE THIS TO TRACK PROGRESS**

3. **`notebooks/01_data_profiling.ipynb`** 📊 STARTER CODE
   - Ready to run immediately
   - Analyzes all 3 CSV files
   - Generates statistics report
   - **START HERE!**

4. **Folder Structure** 📂 ORGANIZED
   ```
   bitirme/
   ├── dataset/
   │   ├── raw/           # Move original files here
   │   ├── scraped/       # Current location of scraped files
   │   └── processed/     # Clean outputs will go here
   ├── ml/
   │   └── preprocessing/ # Cleaning scripts (you'll create)
   ├── scripts/           # Execution scripts (you'll create)
   ├── notebooks/         # Jupyter notebooks
   └── reports/           # Generated reports
   ```

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Setup Environment
```bash
cd /home/esstee/documents/bitirme

# Create virtual environment (if not already done)
python -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn jupyter tqdm
```

### Step 2: Organize Files (Optional but Recommended)
```bash
# Move scraped files to scraped folder
# (They're currently in dataset/ root)
# Skip if you want to keep current structure
```

### Step 3: Start Jupyter
```bash
jupyter notebook notebooks/01_data_profiling.ipynb
```

### Step 4: Run the Notebook
- Run all cells in order
- Takes ~10-30 minutes depending on file size
- Generates `reports/data_quality_report.txt`

### Step 5: Review Results
- Check the report for key statistics
- Note the number of NaN genres
- Note the number of year = 0
- Use these numbers to make cleaning decisions

---

## 📋 The 7-Day Plan Overview

| Day | Phase | What You'll Do |
|-----|-------|----------------|
| **1-2** | **Profiling** | Run notebook, understand data, generate reports |
| **3** | **Strategy** | Decide how to handle issues (documented decisions) |
| **4-5** | **Cleaning** | Write scripts, clean data, merge datasets |
| **6** | **Validation** | Verify quality, check all rules pass |
| **7** | **Normalization** | Normalize text, prepare categories, export final dataset |

**Output**: `dataset/processed/songs_ml_ready.csv` - ready for ML!

---

## 🎯 Key Questions You'll Answer

During profiling (Days 1-2):
- How many rows have NaN genres? ___________
- How many rows have year = 0? ___________
- How many duplicates exist? ___________
- What % of tracks have failed scraping? ___________

During strategy (Day 3):
- **Genre NaN**: [ ] Drop [ ] Impute [ ] Mark as "Unknown" [ ] Infer from audio
- **Year = 0**: [ ] Drop [ ] Impute with median [ ] Flag as missing
- **Duplicates**: [ ] Keep first [ ] Keep most complete
- **Failed tracks**: [ ] Retry [ ] Accept loss
- **Unknown genres**: [ ] Keep as "Unknown" [ ] Drop [ ] Infer

During cleaning (Days 4-5):
- Write `scripts/clean_enhanced_data.py`
- Write `scripts/merge_datasets.py`
- Create `songs_enhanced_clean.csv`
- Create `songs_final_merged.csv`

During validation (Day 6):
- Write `scripts/validate_schema.py`
- Write `scripts/validate_quality.py`
- All checks must PASS

During normalization (Day 7):
- Clean lyrics text
- Normalize genres
- Create derived features
- Export final ML-ready dataset

---

## 🛠️ Tools You'll Use

### Python Libraries
```python
import pandas as pd           # Data manipulation
import numpy as np            # Numerical operations
from tqdm import tqdm         # Progress bars
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns         # Statistical plots
import json                   # Export metadata
import logging                # Track processing
```

### Key Techniques
- **Chunked processing**: Handle large files without memory issues
- **Progress bars**: Track long-running operations
- **Logging**: Document all decisions and transformations
- **Validation**: Ensure data quality at every step

---

## 📊 Expected Results

### Before Cleaning
- ~31M rows in songs_enhanced_full.csv
- Unknown % with NaN genres
- Unknown % with year = 0
- Unknown number of duplicates

### After Cleaning
- ~950K unique tracks (goal)
- 0% NaN genres (all handled)
- 0% year = 0 (all handled)
- 0 duplicates
- 100% pass validation rules

### Final Dataset
```
songs_ml_ready.csv:
├── ~950,000 rows (unique tracks)
├── ~18 columns (features)
├── 4 target variables (valence, energy, danceability, popularity)
├── 13 audio features
├── 1 text feature (lyrics)
├── 4 metadata features (genre, year, explicit, popularity)
└── Quality score: 95%+
```

---

## ⚠️ Common Pitfalls to Avoid

1. **Don't load entire CSV at once** → Use chunked processing
2. **Don't modify original files** → Always work on copies
3. **Don't skip validation** → Validate after each step
4. **Don't forget to document decisions** → Future you will thank you
5. **Don't rush strategy phase** → Good decisions save time later
6. **Don't skip edge cases** → Handle NaN, 0, negative, outliers
7. **Don't forget version control** → Commit after each phase

---

## 💡 Pro Tips

1. **Start with samples** (1000 rows) to test your code
2. **Use progress bars** for long operations (tqdm)
3. **Log everything** to files, not just print()
4. **Create checkpoints** - save intermediate results
5. **Test on subset first** before running on full dataset
6. **Keep a lab notebook** - document surprises and decisions
7. **Ask for help early** if stuck on a decision

---

## 🆘 If You Get Stuck

### Roadmap Too Detailed?
→ Focus on the **checklist** (`PHASE1_CHECKLIST.md`)
→ Just check boxes, don't worry about details yet

### Don't Know How to Decide?
→ See "Decision Points" in roadmap (Phase 1.2)
→ Use rule of thumb: if <5% affected, drop; else impute

### Code Not Working?
→ Test on small sample (1000 rows) first
→ Check data types and column names
→ Add print statements to debug

### Running Out of Time?
→ Prioritize critical issues (NaN genres, year = 0)
→ Accept some data loss if needed (document why)
→ Use simple strategies (drop rather than impute)

---

## ✅ Success Checklist

Phase 1 is complete when:
- [ ] All 3 CSV files analyzed and understood
- [ ] Data quality report generated
- [ ] Cleaning decisions documented
- [ ] Cleaning scripts written and tested
- [ ] Single merged dataset created
- [ ] All validation checks pass
- [ ] Data dictionary created
- [ ] Ready for Phase 2 (Feature Engineering)

---

## 📞 Need Help?

1. Check the **roadmap** for detailed steps
2. Check the **checklist** for what's next
3. Review the **starter notebook** for code examples
4. Check memory bank files for project context
5. Ask your team partner or advisor

---

## 🎉 Let's Get Started!

**Your first task**: Open `notebooks/01_data_profiling.ipynb` and run it!

**Estimated time to first results**: 30 minutes  
**Estimated time to complete Phase 1**: 7-10 days

**Remember**: This is the foundation for your entire ML pipeline.
Take your time, document everything, and make thoughtful decisions.

You've got this! 🚀

---

**Last Updated**: November 10, 2025  
**Version**: 1.0  
**For**: Phase 1 - Data Validation & Cleaning
