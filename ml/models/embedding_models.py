"""
Embedding Models - Audio + Embeddings
Trains models with audio features + lyric embeddings (384-dim from all-MiniLM-L6-v2)

This tests if semantic embeddings improve predictions beyond text statistics.

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
print("EMBEDDING MODELS - AUDIO + LYRIC EMBEDDINGS")
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

# Load embeddings
print("\nLoading lyric embeddings...")
try:
    X_train_embeddings = np.load(features_dir / 'X_train_embeddings.npy')
    X_val_embeddings = np.load(features_dir / 'X_val_embeddings.npy')
    
    print(f"Train embeddings: {X_train_embeddings.shape}")
    print(f"Val embeddings:   {X_val_embeddings.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: Embeddings not found!")
    print("Run preprocessing first: python run_preprocessing.py --steps embeddings")
    exit(1)

# Combine features
print("\nCombining features...")
X_train = np.hstack([X_train_audio, X_train_embeddings])
X_val = np.hstack([X_val_audio, X_val_embeddings])

print(f"Combined train features: {X_train.shape}")
print(f"Combined val features:   {X_val.shape}")
print(f"  - Audio: 21 features (includes genre one-hot, cyclical key, year)")
print(f"  - Embeddings: 384 features (semantic vectors from all-MiniLM-L6-v2)")
print(f"  - Total: 405 features")

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
        'features': 'audio+embeddings',
        'rmse': rmse_mean,
        'mae': mae_mean,
        'r2': r2_mean,
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
        'model': 'Linear',
        'features': 'audio+embeddings',
        'rmse': rmse_lr,
        'mae': mae_lr,
        'r2': r2_lr,
    })
    
    # Save model
    joblib.dump(lr, models_dir / f'linear_embeddings_{target}.pkl')
    
    # 3. RIDGE REGRESSION
    print("\n" + "-" * 80)
    print("3. RIDGE REGRESSION")
    print("-" * 80)
    
    ridge = Ridge(alpha=1.0)
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
        'features': 'audio+embeddings',
        'rmse': rmse_ridge,
        'mae': mae_ridge,
        'r2': r2_ridge,
    })
    
    # Save model
    joblib.dump(ridge, models_dir / f'ridge_embeddings_{target}.pkl')
    
    # 4. XGBOOST
    print("\n" + "-" * 80)
    print("4. XGBOOST")
    print("-" * 80)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
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
        'features': 'audio+embeddings',
        'rmse': rmse_xgb,
        'mae': mae_xgb,
        'r2': r2_xgb,
    })
    
    # Save model
    joblib.dump(xgb_model, models_dir / f'xgboost_embeddings_{target}.pkl')
    
    # Print comparison
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY - {target.upper()}")
    print("=" * 80)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 80)
    print(f"{'Mean':<15} {rmse_mean:<10.4f} {mae_mean:<10.4f} {r2_mean:<10.4f}")
    print(f"{'Linear':<15} {rmse_lr:<10.4f} {mae_lr:<10.4f} {r2_lr:<10.4f}")
    print(f"{'Ridge':<15} {rmse_ridge:<10.4f} {mae_ridge:<10.4f} {r2_ridge:<10.4f}")
    print(f"{'XGBoost':<15} {rmse_xgb:<10.4f} {mae_xgb:<10.4f} {r2_xgb:<10.4f}")
    print("=" * 80)

# Save all results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_path = results_dir / f'embedding_results_{timestamp}.csv'
results_df.to_csv(results_path, index=False)

print(f"\n✅ Results saved to: {results_path}")

# Print final summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY - EMBEDDINGS (AUDIO + 384-DIM SEMANTIC VECTORS)")
print("=" * 80)
print("\nBest R² scores per target:")
for target in targets:
    target_results = results_df[results_df['target'] == target]
    best_idx = target_results['r2'].idxmax()
    best = target_results.loc[best_idx]
    print(f"{target.capitalize():<15} R²={best['r2']:.4f} ({best['model']})")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
