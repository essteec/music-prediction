"""
Text Statistics Models - Audio + Text Statistics Features
Trains models with audio features + text statistics to measure improvement

Models:
1. Mean Predictor (sanity check)
2. Linear Regression
3. Ridge Regression
4. XGBoost

Trains separate model for each target variable
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("TEXT STATISTICS MODELS - AUDIO + TEXT STATISTICS FEATURES")
print("=" * 80)

# Paths
features_dir = Path('../features')
models_dir = Path('../models/saved')
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = Path('../../results/metrics')
results_dir.mkdir(exist_ok=True, parents=True)

# Load audio features
print("\nLoading audio features...")
X_train_audio = np.load(features_dir / 'X_train_audio.npy')
X_val_audio = np.load(features_dir / 'X_val_audio.npy')

print(f"Train audio features: {X_train_audio.shape}")
print(f"Val audio features:   {X_val_audio.shape}")

# Load text statistics features
print("\nLoading text statistics features...")
X_train_text = np.load(features_dir / 'X_train_text_stats.npy')
X_val_text = np.load(features_dir / 'X_val_text_stats.npy')

print(f"Train text features: {X_train_text.shape}")
print(f"Val text features:   {X_val_text.shape}")

# Combine features
print("\nCombining features...")
X_train = np.hstack([X_train_audio, X_train_text])
X_val = np.hstack([X_val_audio, X_val_text])

print(f"Combined train features: {X_train.shape}")
print(f"Combined val features:   {X_val.shape}")
print(f"  - Audio: 21 features (includes genre one-hot, cyclical key, year)")
print(f"  - Text Stats: 5 features")
print(f"  - Total: 26 features")

# Define targets
targets = ['valence', 'energy', 'danceability', 'popularity']

# Store all results
all_results = []

# Train models for each target
for target in targets:
    print("\n" + "=" * 80)
    print(f"TARGET: {target.upper()}")
    print("=" * 80)
    
    # Load target values
    y_train = np.load(features_dir / f'y_train_{target}.npy')
    y_val = np.load(features_dir / f'y_val_{target}.npy')
    
    print(f"\nTarget loaded: {target}")
    print(f"  Train: {y_train.shape[0]:,} samples, mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"  Val:   {y_val.shape[0]:,} samples, mean={y_val.mean():.3f}, std={y_val.std():.3f}")
    
    # 1. MEAN BASELINE
    print("\n" + "-" * 80)
    print("1. MEAN PREDICTOR (Sanity Check)")
    print("-" * 80)
    
    y_pred_mean = np.full_like(y_val, y_train.mean())
    
    rmse_mean = np.sqrt(mean_squared_error(y_val, y_pred_mean))
    mae_mean = mean_absolute_error(y_val, y_pred_mean)
    r2_mean = r2_score(y_val, y_pred_mean)
    
    print(f"RMSE: {rmse_mean:.4f}")
    print(f"MAE:  {mae_mean:.4f}")
    print(f"R²:   {r2_mean:.4f}")
    
    all_results.append({
        'target': target,
        'model': 'Mean',
        'rmse': rmse_mean,
        'mae': mae_mean,
        'r2': r2_mean
    })
    
    # 2. LINEAR REGRESSION
    print("\n" + "-" * 80)
    print("2. LINEAR REGRESSION")
    print("-" * 80)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    
    rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
    mae_lr = mean_absolute_error(y_val, y_pred_lr)
    r2_lr = r2_score(y_val, y_pred_lr)
    
    print(f"RMSE: {rmse_lr:.4f}")
    print(f"MAE:  {mae_lr:.4f}")
    print(f"R²:   {r2_lr:.4f}")
    
    all_results.append({
        'target': target,
        'model': 'LinearRegression',
        'rmse': rmse_lr,
        'mae': mae_lr,
        'r2': r2_lr
    })
    
    # Save model
    joblib.dump(lr, models_dir / f'linear_textstats_{target}.pkl')
    
    # 3. RIDGE REGRESSION
    print("\n" + "-" * 80)
    print("3. RIDGE REGRESSION (alpha=1.0)")
    print("-" * 80)
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_val)
    
    rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
    r2_ridge = r2_score(y_val, y_pred_ridge)
    
    print(f"RMSE: {rmse_ridge:.4f}")
    print(f"MAE:  {mae_ridge:.4f}")
    print(f"R²:   {r2_ridge:.4f}")
    
    all_results.append({
        'target': target,
        'model': 'Ridge',
        'rmse': rmse_ridge,
        'mae': mae_ridge,
        'r2': r2_ridge
    })
    
    # Save model
    joblib.dump(ridge, models_dir / f'ridge_textstats_{target}.pkl')
    
    # 4. XGBOOST
    print("\n" + "-" * 80)
    print("4. XGBOOST")
    print("-" * 80)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
    r2_xgb = r2_score(y_val, y_pred_xgb)
    
    print(f"RMSE: {rmse_xgb:.4f}")
    print(f"MAE:  {mae_xgb:.4f}")
    print(f"R²:   {r2_xgb:.4f}")
    
    all_results.append({
        'target': target,
        'model': 'XGBoost',
        'rmse': rmse_xgb,
        'mae': mae_xgb,
        'r2': r2_xgb
    })
    
    # Save model
    joblib.dump(xgb_model, models_dir / f'xgboost_textstats_{target}.pkl')
    
    # Feature importance for XGBoost
    print("\n" + "-" * 80)
    print("FEATURE IMPORTANCE (Top 10)")
    print("-" * 80)
    
    # Create feature names (must match audio_features.py output)
    audio_features = [
        'acousticness', 'instrumentalness', 'liveness', 'speechiness',
        'loudness', 'tempo', 'duration_ms', 'year', 'mode',
        'key_sin', 'key_cos',
        'genre_Blues', 'genre_Classical', 'genre_Country', 'genre_Electronic',
        'genre_Folk', 'genre_Hip-Hop', 'genre_Jazz', 'genre_Pop', 'genre_R&B', 'genre_Rock'
    ]
    text_features = ['word_count', 'unique_word_count', 'unique_ratio', 
                    'avg_word_length', 'char_count']
    feature_names = audio_features + text_features
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))
    
    # Summary for this target
    print("\n" + "-" * 80)
    print(f"SUMMARY FOR {target.upper()}")
    print("-" * 80)
    print(f"{'Model':<20s} {'RMSE':<10s} {'MAE':<10s} {'R²':<10s}")
    print("-" * 80)
    print(f"{'Mean':<20s} {rmse_mean:<10.4f} {mae_mean:<10.4f} {r2_mean:<10.4f}")
    print(f"{'Linear Regression':<20s} {rmse_lr:<10.4f} {mae_lr:<10.4f} {r2_lr:<10.4f}")
    print(f"{'Ridge':<20s} {rmse_ridge:<10.4f} {mae_ridge:<10.4f} {r2_ridge:<10.4f}")
    print(f"{'XGBoost':<20s} {rmse_xgb:<10.4f} {mae_xgb:<10.4f} {r2_xgb:<10.4f}")

# Save all results to CSV
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
results_path = results_dir / f'textstats_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_df.to_csv(results_path, index=False)

print(f"✅ Results saved to: {results_path}")

# Print overall comparison
print("\n" + "=" * 80)
print("OVERALL RESULTS - ALL TARGETS")
print("=" * 80)

# Pivot table for better visualization
pivot_rmse = results_df.pivot(index='model', columns='target', values='rmse')
pivot_r2 = results_df.pivot(index='model', columns='target', values='r2')

print("\nRMSE (lower is better):")
print(pivot_rmse.to_string())

print("\nR² (higher is better):")
print(pivot_r2.to_string())

# Best model for each target
print("\n" + "-" * 80)
print("BEST MODELS BY TARGET (based on RMSE)")
print("-" * 80)

for target in targets:
    target_results = results_df[results_df['target'] == target]
    best = target_results.loc[target_results['rmse'].idxmin()]
    print(f"{target.capitalize():<15s}: {best['model']:<20s} (RMSE={best['rmse']:.4f}, R²={best['r2']:.4f})")

# Load baseline results for comparison
print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE (AUDIO ONLY)")
print("=" * 80)

# Find latest baseline results
baseline_files = sorted(results_dir.glob('baseline_results_*.csv'))
if baseline_files:
    latest_baseline = baseline_files[-1]
    baseline_df = pd.read_csv(latest_baseline)
    
    print(f"\nComparing with: {latest_baseline.name}")
    print("\nIMPROVEMENT OVER BASELINE (XGBoost only):")
    print("-" * 80)
    print(f"{'Target':<15s} {'Baseline R²':<15s} {'Text Stats R²':<15s} {'Δ R²':<15s} {'Status'}")
    print("-" * 80)
    
    for target in targets:
        baseline_r2 = baseline_df[(baseline_df['target'] == target) & 
                                   (baseline_df['model'] == 'XGBoost')]['r2'].values[0]
        textstats_r2 = results_df[(results_df['target'] == target) & 
                                   (results_df['model'] == 'XGBoost')]['r2'].values[0]
        improvement = textstats_r2 - baseline_r2
        
        status = "✅ Better" if improvement > 0.01 else ("⚠️ Marginal" if improvement > 0 else "❌ Worse")
        
        print(f"{target.capitalize():<15s} {baseline_r2:<15.4f} {textstats_r2:<15.4f} "
              f"{improvement:+.4f}{'':>6s} {status}")
else:
    print("\n⚠️ No baseline results found for comparison")

print("\n" + "=" * 80)
print("✅ TEXT STATISTICS MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModels saved to: {models_dir}/")
print(f"Results saved to: {results_path}")
print("\nNext steps:")
print("  1. Review improvement over baseline")
print("  2. If improvement is significant, continue with sentiment features")
print("  3. If not, try sentiment features alone to see if they help")
print("=" * 80)
