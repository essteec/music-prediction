"""
Final Test Evaluation for Experiment 2 (With Artist Features)
==============================================================
ONE-TIME ONLY evaluation on test set for thesis final numbers.

Evaluates TWO model sets:
1. Enhanced Models: Full 414 features (23 audio + 5 text + 2 sentiment + 384 embeddings)
2. RFE Models: Reduced features at optimal iterations (34-394 features per target)

Selected Models (12 per target = 48 total, × 2 sets = 96 evaluations):
- CatBoost, CatBoost_tuned
- LightGBM, LightGBM_tuned  
- XGBoost, XGBoost_tuned
- ExtraTrees, ExtraTrees_tuned
- MLPRegressor, MLPRegressor_tuned
- RandomForest, RandomForest_tuned

⚠️ WARNING: Run this ONLY ONCE after all development is complete!
Test set should never be used for model selection or tuning.

Author: Thesis Project
Date: January 1, 2026
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
MODELS_DIR_ENHANCED = PROJECT_ROOT / "ml" / "models" / "saved" / "experiment2_with_artist"
MODELS_DIR_RFE = PROJECT_ROOT / "ml" / "models" / "saved" / "experiment2_with_artist" / "rfe_best"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics" / "experiment2_with_artist"
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

# RFE best iterations (from retrain_rfe_best_iterations.py)
RFE_BEST_ITERATIONS = {
    'valence': 23,
    'energy': 38,
    'danceability': 34,
    'popularity': 2
}


def load_rfe_optimal_features(target, iteration):
    """Load optimal feature indices for RFE models"""
    rfe_results_dir = PROJECT_ROOT / "results" / "metrics" / "experiment2_with_artist" / "rfe_best"
    
    # Find the optimal features CSV for this target and iteration
    pattern = f"optimal_features_{target}_iter{iteration}_*.csv"
    matching_files = list(rfe_results_dir.glob(pattern))
    
    if not matching_files:
        print(f"   ⚠️ Warning: No optimal features file found for {target} iter {iteration}")
        return None
    
    # Use the most recent file
    features_file = sorted(matching_files)[-1]
    features_df = pd.read_csv(features_file)
    optimal_indices = features_df['feature_index'].values
    
    return optimal_indices


def load_test_data():
    """Load test features and targets (Experiment 2: 414 features with artist data)"""
    print("\n📁 Loading test data...")
    
    # Load all feature arrays
    # Note: X_test_audio.npy contains 23 features (21 audio + 2 artist)
    X_test_audio = np.load(FEATURES_DIR / "X_test_audio.npy")
    X_test_text_stats = np.load(FEATURES_DIR / "X_test_text_stats.npy")
    X_test_sentiment = np.load(FEATURES_DIR / "X_test_sentiment.npy")
    X_test_embeddings = np.load(FEATURES_DIR / "X_test_embeddings.npy")
    
    # Combine all features: 23 audio+artist + 5 text + 2 sentiment + 384 embeddings = 414
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


def evaluate_model(model, X_test, y_test, model_name, target, model_source='enhanced', num_features=414):
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
            'model_source': model_source,
            'num_features': num_features,
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
    
    # 1. R² comparison across all targets (both enhanced and RFE)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for ax, target in zip(axes.flatten(), TARGETS):
        target_df = results_df[results_df['target'] == target].sort_values('r2', ascending=True)
        
        # Separate enhanced and RFE
        enhanced_df = target_df[target_df['model_source'] == 'enhanced']
        rfe_df = target_df[target_df['model_source'] == 'rfe']
        
        y_pos = np.arange(len(SELECTED_MODELS))
        width = 0.35
        
        # Plot grouped bars
        for i, model in enumerate(SELECTED_MODELS):
            enhanced_r2 = enhanced_df[enhanced_df['model'] == model]['r2'].values
            rfe_r2 = rfe_df[rfe_df['model'] == model]['r2'].values
            
            if len(enhanced_r2) > 0:
                ax.barh(i - width/2, enhanced_r2[0], width, label='Enhanced' if i == 0 else '', 
                       color='#3498db', edgecolor='black', alpha=0.85)
                ax.text(enhanced_r2[0] + 0.005, i - width/2, f'{enhanced_r2[0]:.3f}', 
                       va='center', fontsize=8)
            
            if len(rfe_r2) > 0:
                ax.barh(i + width/2, rfe_r2[0], width, label='RFE' if i == 0 else '', 
                       color='#e74c3c', edgecolor='black', alpha=0.85)
                ax.text(rfe_r2[0] + 0.005, i + width/2, f'{rfe_r2[0]:.3f}', 
                       va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(SELECTED_MODELS, fontsize=9)
        ax.set_xlabel('R² Score (Test Set)', fontsize=11, fontweight='bold')
        ax.set_title(f'{target.upper()} - Enhanced vs RFE', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.2, axis='x')
    
    plt.suptitle('FINAL TEST SET EVALUATION - R² Scores (Enhanced vs RFE)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "test_evaluation_r2.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📈 Saved: test_evaluation_r2.png")
    
    # 2. RMSE comparison (both enhanced and RFE)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for ax, target in zip(axes.flatten(), TARGETS):
        target_df = results_df[results_df['target'] == target].sort_values('rmse', ascending=False)
        
        # Separate enhanced and RFE
        enhanced_df = target_df[target_df['model_source'] == 'enhanced']
        rfe_df = target_df[target_df['model_source'] == 'rfe']
        
        y_pos = np.arange(len(SELECTED_MODELS))
        width = 0.35
        
        # Plot grouped bars (lower is better for RMSE)
        for i, model in enumerate(SELECTED_MODELS):
            enhanced_rmse = enhanced_df[enhanced_df['model'] == model]['rmse'].values
            rfe_rmse = rfe_df[rfe_df['model'] == model]['rmse'].values
            
            if len(enhanced_rmse) > 0:
                ax.barh(i - width/2, enhanced_rmse[0], width, label='Enhanced' if i == 0 else '', 
                       color='#3498db', edgecolor='black', alpha=0.85)
                ax.text(enhanced_rmse[0] + 0.002, i - width/2, f'{enhanced_rmse[0]:.3f}', 
                       va='center', fontsize=8)
            
            if len(rfe_rmse) > 0:
                ax.barh(i + width/2, rfe_rmse[0], width, label='RFE' if i == 0 else '', 
                       color='#e74c3c', edgecolor='black', alpha=0.85)
                ax.text(rfe_rmse[0] + 0.002, i + width/2, f'{rfe_rmse[0]:.3f}', 
                       va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(SELECTED_MODELS, fontsize=9)
        ax.set_xlabel('RMSE (Test Set) - Lower is Better', fontsize=11, fontweight='bold')
        ax.set_title(f'{target.upper()} - Enhanced vs RFE', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2, axis='x')
    
    plt.suptitle('TEST SET EVALUATION - RMSE', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "test_evaluation_rmse.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📈 Saved: test_evaluation_rmse.png")
    
    # 3. Heatmap - separate for enhanced and RFE
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Enhanced models heatmap
    enhanced_df = results_df[results_df['model_source'] == 'enhanced']
    if len(enhanced_df) > 0:
        pivot_enhanced = enhanced_df.pivot(index='model', columns='target', values='r2')
        pivot_enhanced = pivot_enhanced[TARGETS]
        pivot_enhanced = pivot_enhanced.sort_values(pivot_enhanced.columns.tolist(), ascending=False)
        
        sns.heatmap(pivot_enhanced, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
                    ax=axes[0], linewidths=0.5, vmin=-0.1, vmax=0.9,
                    cbar_kws={'label': 'R² Score'})
        axes[0].set_title('Enhanced Models (414 features)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target Variable', fontsize=12)
        axes[0].set_ylabel('Model', fontsize=12)
    
    # RFE models heatmap
    rfe_df = results_df[results_df['model_source'] == 'rfe']
    if len(rfe_df) > 0:
        pivot_rfe = rfe_df.pivot(index='model', columns='target', values='r2')
        pivot_rfe = pivot_rfe[TARGETS]
        pivot_rfe = pivot_rfe.sort_values(pivot_rfe.columns.tolist(), ascending=False)
        
        sns.heatmap(pivot_rfe, annot=True, fmt='.4f', cmap='RdYlGn', center=0.3,
                    ax=axes[1], linewidths=0.5, vmin=-0.1, vmax=0.9,
                    cbar_kws={'label': 'R² Score'})
        axes[1].set_title('RFE Models (34-394 features)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Target Variable', fontsize=12)
        axes[1].set_ylabel('Model', fontsize=12)
    
    plt.suptitle('TEST SET R² Scores - Enhanced vs RFE', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
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
    print("🧪 FINAL TEST SET EVALUATION - EXPERIMENT 2")
    print("=" * 70)
    print("⚠️  WARNING: This should be run ONLY ONCE!")
    print("⚠️  Test set results are FINAL numbers for thesis!")
    print("")
    print("📊 Evaluating TWO model sets:")
    print("   1. Enhanced Models: 414 features (full)")
    print("   2. RFE Models: 34-394 features (optimal iterations)")
    print("   Dataset: 86,454 test songs")
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
        print(f"\n{'='*70}")
        print(f"🎯 TARGET: {target.upper()}")
        print(f"{'='*70}")
        
        # Load RFE optimal features for this target
        rfe_iteration = RFE_BEST_ITERATIONS[target]
        rfe_features = load_rfe_optimal_features(target, rfe_iteration)
        
        if rfe_features is not None:
            X_test_rfe = X_test[:, rfe_features]
            print(f"   RFE features loaded: {len(rfe_features)} features (iteration {rfe_iteration})")
        else:
            X_test_rfe = None
            print(f"   ⚠️ RFE features not available for {target}")
        
        for model_name in SELECTED_MODELS:
            # 1. Evaluate ENHANCED model (full features)
            model_path_enhanced = MODELS_DIR_ENHANCED / f"{model_name}_{target}.pkl"
            # 1. Evaluate ENHANCED model (full features)
            model_path_enhanced = MODELS_DIR_ENHANCED / f"{model_name}_{target}.pkl"
            
            if model_path_enhanced.exists():
                model = joblib.load(model_path_enhanced)
                result = evaluate_model(model, X_test, y_test[target], model_name, target, 
                                       model_source='enhanced', num_features=414)
                
                if result:
                    all_results.append(result)
                    print(f"   ✅ [Enhanced] {model_name}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
            else:
                print(f"   ⚠️ Enhanced model not found: {model_path_enhanced.name}")
            
            # 2. Evaluate RFE model (reduced features)
            if X_test_rfe is not None:
                model_path_rfe = MODELS_DIR_RFE / f"{model_name}_{target}_iter{rfe_iteration}.pkl"
                
                if model_path_rfe.exists():
                    model = joblib.load(model_path_rfe)
                    result = evaluate_model(model, X_test_rfe, y_test[target], model_name, target,
                                           model_source='rfe', num_features=len(rfe_features))
                    
                    if result:
                        all_results.append(result)
                        print(f"   ✅ [RFE]      {model_name}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f} ({len(rfe_features)} features)")
                else:
                    print(f"   ⚠️ RFE model not found: {model_path_rfe.name}")
    
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
        
        # Best enhanced model
        enhanced_df = target_df[target_df['model_source'] == 'enhanced']
        if len(enhanced_df) > 0:
            best_enhanced = enhanced_df.loc[enhanced_df['r2'].idxmax()]
        
        # Best RFE model
        rfe_df = target_df[target_df['model_source'] == 'rfe']
        if len(rfe_df) > 0:
            best_rfe = rfe_df.loc[rfe_df['r2'].idxmax()]
        
        print(f"\n🎯 {target.upper()}:")
        if len(enhanced_df) > 0:
            print(f"   Best Enhanced (414 features): {best_enhanced['model']} - R²={best_enhanced['r2']:.4f}, RMSE={best_enhanced['rmse']:.4f}")
        if len(rfe_df) > 0:
            print(f"   Best RFE ({int(best_rfe['num_features'])} features):     {best_rfe['model']} - R²={best_rfe['r2']:.4f}, RMSE={best_rfe['rmse']:.4f}")
            if len(enhanced_df) > 0:
                r2_diff = best_rfe['r2'] - best_enhanced['r2']
                print(f"   RFE vs Enhanced: {r2_diff:+.4f} R² ({(r2_diff/best_enhanced['r2']*100):+.2f}%)")
    
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
