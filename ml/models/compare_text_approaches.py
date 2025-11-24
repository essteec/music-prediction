"""
Comprehensive Comparison of Text Feature Approaches

Compares:
1. Baseline: Audio only
2. Text Stats: Audio + Text Statistics
3. Sentiment: Audio + Sentiment
4. Combined: Audio + Text Stats + Sentiment (to be trained)

Creates comparison tables for thesis documentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("TEXT FEATURE APPROACH COMPARISON")
print("=" * 80)

results_dir = Path('../../results/metrics')

# Load all results
print("\nLoading results...")

baseline_files = sorted(results_dir.glob('only_audio_results/baseline_results_*.csv'))
textstats_files = sorted(results_dir.glob('textstats_results_*.csv'))
sentiment_files = sorted(results_dir.glob('sentiment_results_*.csv'))

if not baseline_files or not textstats_files or not sentiment_files:
    print("⚠️ Missing results files!")
    print(f"  Baseline files: {len(baseline_files)}")
    print(f"  Text stats files: {len(textstats_files)}")
    print(f"  Sentiment files: {len(sentiment_files)}")
    exit(1)

baseline_df = pd.read_csv(baseline_files[-1])
textstats_df = pd.read_csv(textstats_files[-1])
sentiment_df = pd.read_csv(sentiment_files[-1])

print(f"✅ Loaded:")
print(f"  Baseline:   {baseline_files[-1].name}")
print(f"  Text Stats: {textstats_files[-1].name}")
print(f"  Sentiment:  {sentiment_files[-1].name}")

# Focus on XGBoost results (best model)
print("\n" + "=" * 80)
print("XGBOOST COMPARISON ACROSS ALL APPROACHES")
print("=" * 80)

targets = ['valence', 'energy', 'danceability', 'popularity']

# Create comparison table
comparison_data = []

for target in targets:
    baseline_r2 = baseline_df[(baseline_df['target'] == target) & 
                               (baseline_df['model'] == 'XGBoost')]['r2'].values[0]
    baseline_rmse = baseline_df[(baseline_df['target'] == target) & 
                                 (baseline_df['model'] == 'XGBoost')]['rmse'].values[0]
    
    textstats_r2 = textstats_df[(textstats_df['target'] == target) & 
                                 (textstats_df['model'] == 'XGBoost')]['r2'].values[0]
    textstats_rmse = textstats_df[(textstats_df['target'] == target) & 
                                   (textstats_df['model'] == 'XGBoost')]['rmse'].values[0]
    
    sentiment_r2 = sentiment_df[(sentiment_df['target'] == target) & 
                                 (sentiment_df['model'] == 'XGBoost')]['r2'].values[0]
    sentiment_rmse = sentiment_df[(sentiment_df['target'] == target) & 
                                   (sentiment_df['model'] == 'XGBoost')]['rmse'].values[0]
    
    comparison_data.append({
        'target': target,
        'baseline_r2': baseline_r2,
        'baseline_rmse': baseline_rmse,
        'textstats_r2': textstats_r2,
        'textstats_rmse': textstats_rmse,
        'sentiment_r2': sentiment_r2,
        'sentiment_rmse': sentiment_rmse
    })

comparison_df = pd.DataFrame(comparison_data)

# Print R² comparison
print("\nR² SCORES (higher is better):")
print("-" * 80)
print(f"{'Target':<15s} {'Baseline':<12s} {'Text Stats':<12s} {'Sentiment':<12s} {'Best':<12s}")
print("-" * 80)

for _, row in comparison_df.iterrows():
    best_r2 = max(row['baseline_r2'], row['textstats_r2'], row['sentiment_r2'])
    if best_r2 == row['baseline_r2']:
        best = "Baseline"
    elif best_r2 == row['textstats_r2']:
        best = "Text Stats"
    else:
        best = "Sentiment"
    
    print(f"{row['target'].capitalize():<15s} "
          f"{row['baseline_r2']:<12.4f} "
          f"{row['textstats_r2']:<12.4f} "
          f"{row['sentiment_r2']:<12.4f} "
          f"{best:<12s}")

# Print RMSE comparison
print("\nRMSE SCORES (lower is better):")
print("-" * 80)
print(f"{'Target':<15s} {'Baseline':<12s} {'Text Stats':<12s} {'Sentiment':<12s} {'Best':<12s}")
print("-" * 80)

for _, row in comparison_df.iterrows():
    best_rmse = min(row['baseline_rmse'], row['textstats_rmse'], row['sentiment_rmse'])
    if best_rmse == row['baseline_rmse']:
        best = "Baseline"
    elif best_rmse == row['textstats_rmse']:
        best = "Text Stats"
    else:
        best = "Sentiment"
    
    print(f"{row['target'].capitalize():<15s} "
          f"{row['baseline_rmse']:<12.4f} "
          f"{row['textstats_rmse']:<12.4f} "
          f"{row['sentiment_rmse']:<12.4f} "
          f"{best:<12s}")

# Improvement analysis
print("\n" + "=" * 80)
print("IMPROVEMENT OVER BASELINE (Δ R²)")
print("=" * 80)

print(f"{'Target':<15s} {'Text Stats':<15s} {'Sentiment':<15s} {'Winner':<20s}")
print("-" * 80)

for _, row in comparison_df.iterrows():
    textstats_improvement = row['textstats_r2'] - row['baseline_r2']
    sentiment_improvement = row['sentiment_r2'] - row['baseline_r2']
    
    if abs(textstats_improvement - sentiment_improvement) < 0.01:
        winner = "⚠️ Similar"
    elif textstats_improvement > sentiment_improvement:
        winner = "✅ Text Stats"
    else:
        winner = "✅ Sentiment"
    
    print(f"{row['target'].capitalize():<15s} "
          f"{textstats_improvement:+.4f}{'':>10s} "
          f"{sentiment_improvement:+.4f}{'':>10s} "
          f"{winner:<20s}")

# Feature count summary
print("\n" + "=" * 80)
print("FEATURE COUNT BY APPROACH")
print("=" * 80)
print(f"{'Approach':<20s} {'Features':<10s} {'Description'}")
print("-" * 80)
print(f"{'Baseline':<20s} {'21':<10s} Audio + Genre + Year + Cyclical Key")
print(f"{'Text Stats':<20s} {'26':<10s} Baseline + 5 text statistics")
print(f"{'Sentiment':<20s} {'23':<10s} Baseline + 2 sentiment scores")
print(f"{'Combined (future)':<20s} {'28':<10s} Baseline + 5 text stats + 2 sentiment")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. VALENCE (Emotional Positivity):")
valence_row = comparison_df[comparison_df['target'] == 'valence'].iloc[0]
print(f"   - Baseline R²: {valence_row['baseline_r2']:.4f}")
print(f"   - Text Stats R²: {valence_row['textstats_r2']:.4f} (Δ: {valence_row['textstats_r2']-valence_row['baseline_r2']:+.4f})")
print(f"   - Sentiment R²: {valence_row['sentiment_r2']:.4f} (Δ: {valence_row['sentiment_r2']-valence_row['baseline_r2']:+.4f})")
if valence_row['textstats_r2'] > valence_row['sentiment_r2']:
    print("   → Text statistics (word count, uniqueness) are MORE valuable than sentiment")
else:
    print("   → Sentiment scores are MORE valuable than text statistics")

print("\n2. ENERGY (Intensity/Activity):")
energy_row = comparison_df[comparison_df['target'] == 'energy'].iloc[0]
print(f"   - Baseline R²: {energy_row['baseline_r2']:.4f} (already excellent)")
print(f"   - Text Stats R²: {energy_row['textstats_r2']:.4f} (Δ: {energy_row['textstats_r2']-energy_row['baseline_r2']:+.4f})")
print(f"   - Sentiment R²: {energy_row['sentiment_r2']:.4f} (Δ: {energy_row['sentiment_r2']-energy_row['baseline_r2']:+.4f})")
print("   → Audio features dominate; text features provide minimal gain")

print("\n3. DANCEABILITY:")
dance_row = comparison_df[comparison_df['target'] == 'danceability'].iloc[0]
print(f"   - Baseline R²: {dance_row['baseline_r2']:.4f}")
print(f"   - Text Stats R²: {dance_row['textstats_r2']:.4f} (Δ: {dance_row['textstats_r2']-dance_row['baseline_r2']:+.4f})")
print(f"   - Sentiment R²: {dance_row['sentiment_r2']:.4f} (Δ: {dance_row['sentiment_r2']-dance_row['baseline_r2']:+.4f})")
if dance_row['textstats_r2'] > dance_row['sentiment_r2']:
    print("   → Text statistics provide meaningful improvement")
else:
    print("   → Sentiment provides minimal improvement")

print("\n4. POPULARITY:")
pop_row = comparison_df[comparison_df['target'] == 'popularity'].iloc[0]
print(f"   - Baseline R²: {pop_row['baseline_r2']:.4f} (low - external factors)")
print(f"   - Text Stats R²: {pop_row['textstats_r2']:.4f} (Δ: {pop_row['textstats_r2']-pop_row['baseline_r2']:+.4f})")
print(f"   - Sentiment R²: {pop_row['sentiment_r2']:.4f} (Δ: {pop_row['sentiment_r2']-pop_row['baseline_r2']:+.4f})")
if pop_row['textstats_r2'] > pop_row['sentiment_r2']:
    print("   → Text statistics show significant improvement")
    print("   → Song complexity (word count) correlates with popularity")
else:
    print("   → Still difficult to predict (external factors dominate)")

# Save comparison to CSV
print("\n" + "=" * 80)
print("SAVING COMPARISON")
print("=" * 80)

comparison_path = results_dir / 'text_approach_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"✅ Saved to: {comparison_path}")

# Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION FOR THESIS")
print("=" * 80)

valence_best = "text stats" if valence_row['textstats_r2'] > valence_row['sentiment_r2'] else "sentiment"
dance_best = "text stats" if dance_row['textstats_r2'] > dance_row['sentiment_r2'] else "sentiment"
pop_best = "text stats" if pop_row['textstats_r2'] > pop_row['sentiment_r2'] else "sentiment"

print(f"\nBased on the results:")
print(f"  - Valence: {valence_best} is better")
print(f"  - Danceability: {dance_best} is better")
print(f"  - Popularity: {pop_best} is better")
print(f"  - Energy: Neither helps much (audio already excellent)")

print("\n✅ NEXT STEP: Train combined model (Audio + Text Stats + Sentiment)")
print("   This will test if combining both text approaches yields the best results.")
print("   Expected: Combined should be best for Valence and Popularity")

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE!")
print("=" * 80)
