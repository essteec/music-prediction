"""
PANNs Models - Base Features + PANNs Audio Embeddings
Trains models with base features (audio, text, sentiment) + PANNs embeddings (2048-d).

Models:
1. Mean Predictor (sanity check)
2. Linear Regression
3. Ridge Regression
4. XGBoost
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
print("PANNS MODELS - BASE FEATURES + PANNS AUDIO EMBEDDINGS (2846 FEATURES)")
print("=" * 80)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
features_dir = REPO_ROOT / 'ml' / 'features'
audio_emb_dir = REPO_ROOT / 'data' / 'embeddings' / 'audio'
models_dir = REPO_ROOT / 'ml' / 'models' / 'saved' / 'panns_test'
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = REPO_ROOT / 'results' / 'metrics'
results_dir.mkdir(exist_ok=True, parents=True)

# Load all features
print("\nLoading features...")

# Audio (base)
X_train_audio = np.load(features_dir / 'X_train_audio.npy')
X_val_audio = np.load(features_dir / 'X_val_audio.npy')
print(f"✓ Base Audio features: train={X_train_audio.shape}, val={X_val_audio.shape}")

# Text stats
X_train_text = np.load(features_dir / 'X_train_text_stats.npy')
X_val_text = np.load(features_dir / 'X_val_text_stats.npy')
print(f"✓ Text stats: train={X_train_text.shape}, val={X_val_text.shape}")

# Sentiment
X_train_sentiment = np.load(features_dir / 'X_train_sentiment.npy')
X_val_sentiment = np.load(features_dir / 'X_val_sentiment.npy')
print(f"✓ Sentiment: train={X_train_sentiment.shape}, val={X_val_sentiment.shape}")

# MPNet Embeddings
try:
    X_train_mpnet = np.load(features_dir / 'X_train_mpnet.npy')
    X_val_mpnet = np.load(features_dir / 'X_val_mpnet.npy')
    print(f"✓ MPNet Embeddings: train={X_train_mpnet.shape}, val={X_val_mpnet.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: MPNet Embeddings not found!")
    exit(1)

# PANNs Embeddings
try:
    X_train_panns = np.load(features_dir / 'X_train_panns.npy')
    X_val_panns = np.load(features_dir / 'X_val_panns.npy')
    print(f"✓ PANNs Embeddings: train={X_train_panns.shape}, val={X_val_panns.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: PANNs Embeddings not found!")
    print("Run ml/preprocessing/data_splitting.py first.")
    exit(1)

# Combine ALL features
print("\nCombining all features...")
X_train = np.hstack([X_train_audio, X_train_text, X_train_sentiment, X_train_mpnet, X_train_panns])
X_val = np.hstack([X_val_audio, X_val_text, X_val_sentiment, X_val_mpnet, X_val_panns])

print(f"\n{'=' * 80}")
print(f"COMBINED FEATURE MATRIX - PANNS")
print(f"{'=' * 80}")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"\nFeature breakdown:")
print(f"  - Base Audio: 23 features")
print(f"  - Text Stats:  5 features")
print(f"  - Sentiment:   2 features")
print(f"  - MPNet:     768 features")
print(f"  - PANNs:    2048 features")
print(f"  - TOTAL:    2846 features")
print(f"{'=' * 80}")

# Define targets
targets = ['valence', 'energy', 'danceability', 'popularity']
all_results = []

for target in targets:
    print("\n" + "=" * 80)
    print(f"TARGET: {target.upper()}")
    print("=" * 80)
    
    y_train = np.load(features_dir / f'y_train_{target}.npy')
    y_val = np.load(features_dir / f'y_val_{target}.npy')
    
    # 1. MEAN BASELINE
    y_pred_mean = np.full_like(y_val, y_train.mean())
    rmse_mean = np.sqrt(mean_squared_error(y_val, y_pred_mean))
    mae_mean = mean_absolute_error(y_val, y_pred_mean)
    r2_mean = r2_score(y_val, y_pred_mean)
    all_results.append({'target': target, 'model': 'Mean', 'features': 'panns', 'rmse': rmse_mean, 'mae': mae_mean, 'r2': r2_mean})
    
    # 2. LINEAR REGRESSION
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
    mae_lr = mean_absolute_error(y_val, y_pred_lr)
    r2_lr = r2_score(y_val, y_pred_lr)
    all_results.append({'target': target, 'model': 'Linear', 'features': 'panns', 'rmse': rmse_lr, 'mae': mae_lr, 'r2': r2_lr})
    joblib.dump(lr, models_dir / f'linear_panns_{target}.pkl')
    
    # 3. RIDGE REGRESSION
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_val)
    rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
    r2_ridge = r2_score(y_val, y_pred_ridge)
    all_results.append({'target': target, 'model': 'Ridge', 'features': 'panns', 'rmse': rmse_ridge, 'mae': mae_ridge, 'r2': r2_ridge})
    joblib.dump(ridge, models_dir / f'ridge_panns_{target}.pkl')
    
    # 4. XGBOOST
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
    r2_xgb = r2_score(y_val, y_pred_xgb)
    all_results.append({'target': target, 'model': 'XGBoost', 'features': 'panns', 'rmse': rmse_xgb, 'mae': mae_xgb, 'r2': r2_xgb})
    joblib.dump(xgb_model, models_dir / f'xgboost_panns_{target}.pkl')
    
    print(f"Results for {target}:")
    print(f"  Linear: RMSE={rmse_lr:.4f}, R²={r2_lr:.4f}")
    print(f"  Ridge:  RMSE={rmse_ridge:.4f}, R²={r2_ridge:.4f}")
    print(f"  XGB:    RMSE={rmse_xgb:.4f}, R²={r2_xgb:.4f}")

# Save all results
results_df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_path = results_dir / 'panns_test' / f'panns_results_{timestamp}.csv'
results_path.parent.mkdir(exist_ok=True, parents=True)
results_df.to_csv(results_path, index=False)
print(f"\n✅ Results saved to: {results_path}")
