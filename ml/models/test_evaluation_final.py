"""
Final Test Evaluation for Selected Models
==========================================
ONE-TIME ONLY evaluation on test set for thesis final numbers.

Selected Models (12 per target = 48 total):
- CatBoost, CatBoost_tuned
- LightGBM, LightGBM_tuned  
- XGBoost, XGBoost_tuned
- ExtraTrees, ExtraTrees_tuned
- MLPRegressor, MLPRegressor_tuned
- RandomForest, RandomForest_tuned

⚠️ WARNING: Run this ONLY ONCE after all development is complete!
Test set should never be used for model selection or tuning.

Author: Thesis Project
Date: December 6, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)

# Set paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "ml" / "features"
MODELS_DIR = PROJECT_ROOT / "ml" / "models" / "saved" / "enhanced"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Selected models for final evaluation
SELECTED_MODELS = [
    'CatBoost', 'CatBoost_tuned',
    'LightGBM', 'LightGBM_tuned',
    'XGBoost', 'XGBoost_tuned',
    'ExtraTrees', 'ExtraTrees_tuned',
    'MLPRegressor', 'MLPRegressor_tuned',
    'RandomForest', 'RandomForest_tuned'
]

TARGETS = ['valence', 'energy', 'danceability', 'popularity']


def load_test_data():
    """Load test features and targets"""
    print("\n📁 Loading test data...")
    
    # Load all feature arrays
    X_test_audio = np.load(FEATURES_DIR / "X_test_audio.npy")
    X_test_text_stats = np.load(FEATURES_DIR / "X_test_text_stats.npy")
    X_test_sentiment = np.load(FEATURES_DIR / "X_test_sentiment.npy")
    X_test_embeddings = np.load(FEATURES_DIR / "X_test_embeddings.npy")
    
    # Combine all features (same order as training)
    X_test = np.hstack([X_test_audio, X_test_text_stats, X_test_sentiment, X_test_embeddings])
    
    print(f"   Audio features: {X_test_audio.shape}")
    print(f"   Text stats: {X_test_text_stats.shape}")
    print(f"   Sentiment: {X_test_sentiment.shape}")
    print(f"   Embeddings: {X_test_embeddings.shape}")
    print(f"   Combined X_test: {X_test.shape}")
    
    # Load target arrays
    y_test = {}
    for target in TARGETS:
        y_test[target] = np.load(FEATURES_DIR / f"y_test_{target}.npy")
        print(f"   y_test_{target}: {y_test[target].shape}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name, target):
    """Comprehensive evaluation of a model on test set"""
    
    try:
        y_pred = model.predict(X_test)
        
        # Core metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        # Prediction statistics
        pred_mean = y_pred.mean()
        pred_std = y_pred.std()
        pred_min = y_pred.min()
        pred_max = y_pred.max()
        
        # Residual analysis
        residuals = y_test - y_pred
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        
        return {
            'target': target,
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'max_error': max_err,
            'mape': mape,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'pred_min': pred_min,
            'pred_max': pred_max,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'n_samples': len(y_test)
        }
        
    except Exception as e:
        print(f"   ❌ Error evaluating {model_name}: {e}")
        return None


def plot_test_results(results_df):
    """Create visualization of test results"""
    
    # 1. R² comparison across all targets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, len(SELECTED_MODELS)))
    
    for ax, target in zip(axes.flatten(), TARGETS):
        target_df = results_df[results_df['target'] == target].sort_values('r2', ascending=True)
        
        bars = ax.barh(range(len(target_df)), target_df['r2'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(target_df)))
        ax.set_yticklabels(target_df['model'])
        ax.set_xlabel('R² Score (Test Set)', fontsize=11)
        ax.set_title(f'{target.upper()} - Test Performance', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, target_df['r2']):
            ax.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
        
        ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.suptitle('🧪 FINAL TEST SET EVALUATION - R² Scores', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "test_evaluation_r2.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📈 Saved: test_evaluation_r2.png")
    
    # 2. RMSE comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, target in zip(axes.flatten(), TARGETS):
        target_df = results_df[results_df['target'] == target].sort_values('rmse', ascending=False)
        
        bars = ax.barh(range(len(target_df)), target_df['rmse'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(target_df)))
        ax.set_yticklabels(target_df['model'])
        ax.set_xlabel('RMSE (Test Set) - Lower is Better', fontsize=11)
        ax.set_title(f'{target.upper()} - Test RMSE', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, target_df['rmse']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=9)
    
    plt.suptitle('🧪 FINAL TEST SET EVALUATION - RMSE', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "test_evaluation_rmse.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📈 Saved: test_evaluation_rmse.png")
    
    # 3. Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(16, 10))
    
    pivot_r2 = results_df.pivot(index='model', columns='target', values='r2')
    pivot_r2 = pivot_r2[TARGETS]  # Order columns
    pivot_r2 = pivot_r2.sort_values(pivot_r2.columns.tolist(), ascending=False)
    
    sns.heatmap(pivot_r2, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
                ax=ax, linewidths=0.5, vmin=-0.1, vmax=0.9,
                cbar_kws={'label': 'R² Score'})
    ax.set_title('🧪 TEST SET R² Scores - All Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "test_evaluation_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📈 Saved: test_evaluation_heatmap.png")


def create_comparison_with_validation(test_results, val_results_path):
    """Compare test vs validation performance"""
    
    if not val_results_path.exists():
        print("   ⚠️ Validation results not found for comparison")
        return None
    
    val_df = pd.read_csv(val_results_path)
    test_df = test_results.copy()
    
    # Filter validation results to selected models
    val_df = val_df[val_df['model'].isin(SELECTED_MODELS)]
    
    # Merge
    comparison = test_df.merge(
        val_df[['target', 'model', 'r2', 'rmse']], 
        on=['target', 'model'], 
        suffixes=('_test', '_val')
    )
    
    comparison['r2_diff'] = comparison['r2_test'] - comparison['r2_val']
    comparison['rmse_diff'] = comparison['rmse_test'] - comparison['rmse_val']
    
    return comparison


def main():
    """Main test evaluation"""
    
    print("=" * 70)
    print("🧪 FINAL TEST SET EVALUATION")
    print("=" * 70)
    print("⚠️  WARNING: This should be run ONLY ONCE!")
    print("⚠️  Test set results are FINAL numbers for thesis!")
    print("=" * 70)
    
    # Confirmation prompt
    confirm = input("\n❓ Are you sure you want to run test evaluation? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ Test evaluation cancelled.")
        return
    
    print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Targets: {TARGETS}")
    print(f"🤖 Models: {len(SELECTED_MODELS)} selected models")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Evaluate all models
    all_results = []
    
    for target in TARGETS:
        print(f"\n🎯 Evaluating {target.upper()}...")
        
        for model_name in SELECTED_MODELS:
            model_path = MODELS_DIR / f"{model_name}_{target}.pkl"
            
            if not model_path.exists():
                print(f"   ⚠️ Model not found: {model_path.name}")
                continue
            
            model = joblib.load(model_path)
            result = evaluate_model(model, X_test, y_test[target], model_name, target)
            
            if result:
                all_results.append(result)
                print(f"   ✅ {model_name}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
    
    if not all_results:
        print("\n❌ No results collected. Check if models exist.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f"test_evaluation_final_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved: {results_path}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_test_results(results_df)
    
    # Compare with validation results
    val_results_path = RESULTS_DIR / "enhanced_results_summary_20251205_123928.csv"
    comparison = create_comparison_with_validation(results_df, val_results_path)
    
    if comparison is not None:
        comparison_path = RESULTS_DIR / f"test_vs_validation_comparison_{timestamp}.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"💾 Comparison saved: {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SET RESULTS SUMMARY")
    print("=" * 70)
    
    for target in TARGETS:
        target_df = results_df[results_df['target'] == target]
        best = target_df.loc[target_df['r2'].idxmax()]
        
        print(f"\n🎯 {target.upper()}:")
        print(f"   Best Model: {best['model']}")
        print(f"   R² = {best['r2']:.4f}")
        print(f"   RMSE = {best['rmse']:.4f}")
        print(f"   MAE = {best['mae']:.4f}")
    
    # Create final summary table
    print("\n" + "=" * 70)
    print("📋 BEST MODEL PER TARGET (FINAL TEST NUMBERS)")
    print("=" * 70)
    
    best_models = results_df.loc[results_df.groupby('target')['r2'].idxmax()]
    best_models = best_models[['target', 'model', 'r2', 'rmse', 'mae']].sort_values('r2', ascending=False)
    
    print(best_models.to_string(index=False))
    
    # Save best models summary
    best_path = RESULTS_DIR / f"best_models_test_{timestamp}.csv"
    best_models.to_csv(best_path, index=False)
    print(f"\n💾 Best models saved: {best_path}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("📈 OVERALL STATISTICS")
    print("=" * 70)
    print(f"   Total models evaluated: {len(results_df)}")
    print(f"   Average R² across all models/targets: {results_df['r2'].mean():.4f}")
    print(f"   Best overall R²: {results_df['r2'].max():.4f} ({results_df.loc[results_df['r2'].idxmax(), 'model']} on {results_df.loc[results_df['r2'].idxmax(), 'target']})")
    
    print("\n✅ Test evaluation complete!")
    print("📁 Results saved to:", RESULTS_DIR)
    print("📊 Figures saved to:", FIGURES_DIR)
    print("\n⚠️ These are your FINAL thesis numbers!")


if __name__ == "__main__":
    main()
