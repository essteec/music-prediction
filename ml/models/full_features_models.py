"""
Full Feature Models - Audio + Text Stats + Sentiment + Embeddings
Experiment 2: Trains models with ALL available features including artist features

Features:
- Audio: 23 features (includes genre, year, cyclical key, + 2 artist features)
  - NEW: log_total_artist_followers, avg_artist_popularity
- Text Stats: 5 features (word count, uniqueness, etc.)
- Sentiment: 2 features (polarity, subjectivity)
- Embeddings: 384 features (semantic vectors from all-MiniLM-L6-v2)
- Total: 414 features

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
print("FULL FEATURE MODELS - EXPERIMENT 2 (WITH ARTIST FEATURES)")
print("=" * 80)

# Paths
features_dir = Path('../features')
models_dir = Path('../models/saved/experiment2_with_artist')
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = Path('../../results/metrics')
results_dir.mkdir(exist_ok=True, parents=True)

# Load all features
print("\nLoading features...")

# Audio
X_train_audio = np.load(features_dir / 'X_train_audio.npy')
X_val_audio = np.load(features_dir / 'X_val_audio.npy')
print(f"✓ Audio features: train={X_train_audio.shape}, val={X_val_audio.shape}")

# Text stats
X_train_text = np.load(features_dir / 'X_train_text_stats.npy')
X_val_text = np.load(features_dir / 'X_val_text_stats.npy')
print(f"✓ Text stats: train={X_train_text.shape}, val={X_val_text.shape}")

# Sentiment
X_train_sentiment = np.load(features_dir / 'X_train_sentiment.npy')
X_val_sentiment = np.load(features_dir / 'X_val_sentiment.npy')
print(f"✓ Sentiment: train={X_train_sentiment.shape}, val={X_val_sentiment.shape}")

# Embeddings
try:
    X_train_embeddings = np.load(features_dir / 'X_train_mpnet.npy')
    X_val_embeddings = np.load(features_dir / 'X_val_mpnet.npy')
    print(f"✓ MPNet Embeddings: train={X_train_embeddings.shape}, val={X_val_embeddings.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: MPNet Embeddings not found!")
    print("Run preprocessing first: python run_preprocessing.py --steps mpnet")
    exit(1)

# Combine ALL features
print("\nCombining all features...")
X_train = np.hstack([X_train_audio, X_train_text, X_train_sentiment, X_train_embeddings])
X_val = np.hstack([X_val_audio, X_val_text, X_val_sentiment, X_val_embeddings])

print(f"\n{'=' * 80}")
print(f"COMBINED FEATURE MATRIX - MPNET VERSION")
print(f"{'=' * 80}")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"\nFeature breakdown:")
print(f"  - Audio:      23 features (genre, year, cyclical key, audio, + artist)")
print(f"  - Text Stats:  5 features (word count, uniqueness, avg length, etc.)")
print(f"  - Sentiment:   2 features (polarity, subjectivity)")
print(f"  - Embeddings: 768 features (semantic vectors from MPNet)")
print(f"  - TOTAL:      798 features")
print(f"{'=' * 80}")

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
        'features': 'all',
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
        'features': 'all',
        'rmse': rmse_lr,
        'mae': mae_lr,
        'r2': r2_lr,
    })
    
    # Save model
    joblib.dump(lr, models_dir / f'linear_full_{target}.pkl')
    
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
        'features': 'all',
        'rmse': rmse_ridge,
        'mae': mae_ridge,
        'r2': r2_ridge,
    })
    
    # Save model
    joblib.dump(ridge, models_dir / f'ridge_full_{target}.pkl')
    
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
        'features': 'all',
        'rmse': rmse_xgb,
        'mae': mae_xgb,
        'r2': r2_xgb,
    })
    
    # Save model
    joblib.dump(xgb_model, models_dir / f'xgboost_full_{target}.pkl')
    
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
results_path = results_dir / 'experiment2_with_artist' / f'full_features_results_{timestamp}.csv'
results_path.parent.mkdir(exist_ok=True, parents=True)
results_df.to_csv(results_path, index=False)

print(f"\n✅ Results saved to: {results_path}")

# Print final summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY - EXPERIMENT 2 (414 FEATURES WITH ARTIST)")
print("=" * 80)
print("\nBest R² scores per target:")
for target in targets:
    target_results = results_df[results_df['target'] == target]
    best_idx = target_results['r2'].idxmax()
    best = target_results.loc[best_idx]
    print(f"{target.capitalize():<15} R²={best['r2']:.4f} ({best['model']})")

print("\n" + "=" * 80)
print("DONE! All features have been tested.")
print("=" * 80)
