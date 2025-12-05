"""
Enhanced Models - Comprehensive Algorithm Comparison
Trains 14+ algorithms with default and tuned variants for complete comparison

Algorithms:
1. Mean Predictor (baseline)
2. Linear Regression
3. Ridge Regression
4. Lasso Regression
5. SGD Regressor
6. Decision Tree (default + tuned)
7. Random Forest (default + tuned)
8. Extra Trees (default + tuned)
9. AdaBoost Regressor
10. XGBoost (default + tuned)
11. CatBoost (default + tuned)
12. K-Neighbors Regressor
13. SVR with RBF kernel (default + tuned)
14. MLP Regressor (default + tuned)

Features: ALL 412 features (audio + text + sentiment + embeddings)
Outputs:
- results_summary.csv: Basic metrics (RMSE, MAE, R²)
- results_detailed.csv: Extended metrics with training time, predictions stats

Advanced Features:
- Incremental saving after each model
- Checkpoint system to resume training
- Progress tracking with JSON checkpoints
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error
)
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import joblib
from pathlib import Path
from datetime import datetime
import time
import json
import sys

print("=" * 80)
print("ENHANCED MODEL TRAINING - COMPREHENSIVE ALGORITHM COMPARISON")
print("=" * 80)

# Paths
features_dir = Path('../features')
models_dir = Path('../models/saved/enhanced')
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = Path('../../results/metrics')
results_dir.mkdir(exist_ok=True, parents=True)

# Checkpoint paths
checkpoint_dir = results_dir / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True, parents=True)
checkpoint_file = checkpoint_dir / 'enhanced_training_checkpoint.json'

# Session timestamp
session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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
    X_train_embeddings = np.load(features_dir / 'X_train_embeddings.npy')
    X_val_embeddings = np.load(features_dir / 'X_val_embeddings.npy')
    print(f"✓ Embeddings: train={X_train_embeddings.shape}, val={X_val_embeddings.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: Embeddings not found!")
    print("Run preprocessing first: python run_preprocessing.py --steps embeddings")
    exit(1)

# Combine ALL features
print("\nCombining all features...")
X_train = np.hstack([X_train_audio, X_train_text, X_train_sentiment, X_train_embeddings])
X_val = np.hstack([X_val_audio, X_val_text, X_val_sentiment, X_val_embeddings])

print(f"\n{'=' * 80}")
print(f"COMBINED FEATURE MATRIX")
print(f"{'=' * 80}")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Total Features: 412")
print(f"{'=' * 80}")

# Define targets
targets = ['valence', 'energy', 'danceability', 'popularity']

# Initialize or load checkpoint
def load_checkpoint():
    """Load existing checkpoint or create new one"""
    if checkpoint_file.exists():
        print("\n" + "=" * 80)
        print("CHECKPOINT FOUND - LOADING PROGRESS")
        print("=" * 80)
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"Session: {checkpoint['session_timestamp']}")
        print(f"Completed: {len(checkpoint['completed_models'])} models")
        print(f"Last update: {checkpoint['last_update']}")
        
        # Ask user if they want to resume
        response = input("\nResume from checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            print("✓ Resuming from checkpoint...")
            return checkpoint
        else:
            print("✗ Starting fresh training...")
            return create_new_checkpoint()
    else:
        return create_new_checkpoint()

def create_new_checkpoint():
    """Create new checkpoint structure"""
    return {
        'session_timestamp': session_timestamp,
        'last_update': datetime.now().isoformat(),
        'completed_models': [],
        'results_summary': [],
        'results_detailed': []
    }

def save_checkpoint(checkpoint):
    """Save checkpoint to disk"""
    checkpoint['last_update'] = datetime.now().isoformat()
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def is_model_completed(checkpoint, target, model_name):
    """Check if model has already been trained"""
    key = f"{target}_{model_name}"
    return key in checkpoint['completed_models']

def mark_model_completed(checkpoint, target, model_name):
    """Mark model as completed in checkpoint"""
    key = f"{target}_{model_name}"
    if key not in checkpoint['completed_models']:
        checkpoint['completed_models'].append(key)

# Load or create checkpoint
checkpoint = load_checkpoint()

# Store all results (load existing from checkpoint)
all_results_summary = checkpoint['results_summary']
all_results_detailed = checkpoint['results_detailed']

print(f"\nStarting with {len(all_results_summary)} existing results")

# Define all models with their configurations
def get_models():
    """Returns dictionary of all models to train
    
    Optimized for overnight training with 386k samples and 412 features:
    - Default versions: Fast baseline with standard sklearn defaults
    - Tuned versions: Heavily optimized for maximum performance (no speed constraints)
    - Consistent hyperparameter philosophy across all tuned models
    - Early stopping to prevent overfitting on validation set
    """
    models = {
        # Baseline
        'Mean': None,  # Special case, handled separately
        
        # Linear Models - Default unchanged
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Ridge_tuned': Ridge(alpha=0.2, random_state=42, max_iter=20000),
        
        'Lasso': Lasso(random_state=42),
        'Lasso_tuned': Lasso(alpha=0.2, random_state=42, max_iter=20000, tol=1e-5),
        
        'SGDRegressor': SGDRegressor(random_state=42),
        'SGDRegressor_tuned': SGDRegressor(
            alpha=0.0005,
            penalty='elasticnet',
            l1_ratio=0.2,
            random_state=42,
            max_iter=2000,
            tol=1e-4,
            early_stopping=True,
            n_iter_no_change=15,
            learning_rate='adaptive'
        ),
        
        # Tree-based Models - Default unchanged
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'DecisionTree_tuned': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=200,
            min_samples_leaf=100,
            max_features='sqrt',
            min_impurity_decrease=0.005,
            random_state=42
        ),
        
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'RandomForest_tuned': RandomForestRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=50,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            max_samples=0.5
        ),
        
        'ExtraTrees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'ExtraTrees_tuned': ExtraTreesRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=50,
            min_samples_leaf=10,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            max_samples=0.6
        ),
        
        # Boosting Models - Default unchanged, Tuned heavily optimized
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'AdaBoost_tuned': AdaBoostRegressor(
            n_estimators=150,
            learning_rate=0.01,
            loss='square',
            random_state=42,
        ),
        
        # XGBoost - Default unchanged
        'XGBoost': xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost_tuned': xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            colsample_bylevel=1.0,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        ),
        
        # CatBoost - Default unchanged
        'CatBoost': CatBoostRegressor(
            random_state=42,
            verbose=False
        ),
        'CatBoost_tuned': CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=8,
            subsample=0.8,
            bootstrap_type='Bernoulli',
            random_state=42,
            verbose=False,
            early_stopping_rounds=50,
            thread_count=-1,
            grow_policy='Lossguide',
            max_leaves=64
        ),
        
        # LightGBM - Default unchanged
        'LightGBM': lgb.LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'LightGBM_tuned': lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.06,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_split_gain=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            feature_fraction=0.7,
            bagging_freq=5,            
            bagging_fraction=0.7,
            importance_type='gain'
        ),
        
        # Instance-based - Default unchanged
        'KNeighbors': KNeighborsRegressor(n_jobs=-1),
        'KNeighbors_tuned': KNeighborsRegressor(
            n_neighbors=20,
            weights='distance',
            p=2,
            leaf_size=50,
            algorithm='ball_tree',
            n_jobs=-1,
            metric='minkowski' 
        ),
        
        # SVM - Default unchanged
        'LinearSVR': LinearSVR(random_state=42),
        'LinearSVR_tuned': LinearSVR(
            C=0.3,
            epsilon=0.05,
            loss='squared_epsilon_insensitive',
            dual=False,
            random_state=42,
            max_iter=20000,
            tol=1e-4,
            fit_intercept=True,             
            intercept_scaling=1.0,          
            verbose=0                       
        ),
        
        # Neural Network - Default unchanged
        'MLPRegressor': MLPRegressor(
            random_state=42,
            early_stopping=True
        ),
        'MLPRegressor_tuned': MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.005,
            batch_size=1024,
            learning_rate='adaptive',
            learning_rate_init=0.0005,
            power_t=0.5,
            random_state=42,
            max_iter=500,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        ),
    }
    return models


def save_results_to_csv(summary_results, detailed_results, timestamp):
    """Save results to CSV files immediately"""
    # Summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_path = results_dir / f'enhanced_results_summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Detailed CSV
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = results_dir / f'enhanced_results_detailed_{timestamp}.csv'
    detailed_df.to_csv(detailed_path, index=False)
    
    return summary_path, detailed_path


def calculate_detailed_metrics(y_true, y_pred, model_name, target, train_time):
    """Calculate comprehensive metrics for regression"""
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    explained_var = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    
    # MAPE (handle division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = np.nan
    
    # Prediction statistics
    pred_mean = np.mean(y_pred)
    pred_std = np.std(y_pred)
    pred_min = np.min(y_pred)
    pred_max = np.max(y_pred)
    
    # Residuals
    residuals = y_true - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    # Convert all numpy types to native Python types for JSON serialization
    # This handles both float32 (XGBoost) and float64 (sklearn) dtypes
    return {
        'target': target,
        'model': model_name,
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'explained_variance': float(explained_var),
        'max_error': float(max_err),
        'mape': float(mape),
        'pred_mean': float(pred_mean),
        'pred_std': float(pred_std),
        'pred_min': float(pred_min),
        'pred_max': float(pred_max),
        'residual_mean': float(residual_mean),
        'residual_std': float(residual_std),
        'train_time_seconds': float(train_time),
    }


# Train models for each target
for target in targets:
    print("\n" + "=" * 80)
    print(f"TARGET: {target.upper()}")
    print("=" * 80)
    
    # Load target values
    y_train = np.load(features_dir / f'y_train_{target}.npy')
    y_val = np.load(features_dir / f'y_val_{target}.npy')
    
    print(f"\nTarget statistics:")
    print(f"  Train: {y_train.shape[0]:,} samples, mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"  Val:   {y_val.shape[0]:,} samples, mean={y_val.mean():.3f}, std={y_val.std():.3f}")
    
    # Get all models
    models = get_models()
    
    # Train each model
    for i, (model_name, model) in enumerate(models.items(), 1):
        # Check if already completed
        if is_model_completed(checkpoint, target, model_name):
            print(f"\n[{i}/{len(models)}] {model_name} - SKIPPED (already completed)")
            continue
        
        print(f"\n[{i}/{len(models)}] Training {model_name}...")
        
        start_time = time.time()
        
        try:
            # Special case: Mean predictor
            if model_name == 'Mean':
                y_pred = np.full_like(y_val, y_train.mean(), dtype=float)
                train_time = time.time() - start_time
            else:
                # Train model with early stopping support for boosting models
                if 'XGBoost_tuned' in model_name or 'CatBoost_tuned' in model_name:
                    # Use validation set for early stopping
                    eval_set = [(X_val, y_val)]
                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                elif 'LightGBM_tuned' in model_name:
                    # LightGBM uses callbacks for early stopping
                    eval_set = [(X_val, y_val)]
                    model.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(50, verbose=False)])
                else:
                    # Standard training
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                train_time = time.time() - start_time
                
                # Save model
                model_filename = models_dir / f'{model_name}_{target}.pkl'
                joblib.dump(model, model_filename)
            
            # Calculate metrics
            detailed_metrics = calculate_detailed_metrics(
                y_val, y_pred, model_name, target, train_time
            )
            
            # Store results
            all_results_detailed.append(detailed_metrics)
            all_results_summary.append({
                'target': target,
                'model': model_name,
                'rmse': detailed_metrics['rmse'],
                'mae': detailed_metrics['mae'],
                'r2': detailed_metrics['r2']
            })
            
            # Update checkpoint
            checkpoint['results_summary'] = all_results_summary
            checkpoint['results_detailed'] = all_results_detailed
            mark_model_completed(checkpoint, target, model_name)
            
            # Save checkpoint and results immediately
            save_checkpoint(checkpoint)
            save_results_to_csv(all_results_summary, all_results_detailed, session_timestamp)
            
            # Print basic results with timestamp and beep notification
            completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ RMSE={detailed_metrics['rmse']:.4f}, "
                  f"MAE={detailed_metrics['mae']:.4f}, "
                  f"R²={detailed_metrics['r2']:.4f}, "
                  f"Time={train_time:.2f}s")
            print(f"  ✓ Results saved (total: {len(all_results_summary)} models)")
            print(f"  ⏰ Completed at: {completion_time}")
            
            # Audio notification (beep twice)
            import os
            os.system('echo -e "\a"')
            time.sleep(0.2)
            os.system('echo -e "\a"')
            
        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("TRAINING INTERRUPTED BY USER")
            print("=" * 80)
            print(f"Progress saved! {len(all_results_summary)} models completed.")
            print(f"Resume by running this script again.")
            save_checkpoint(checkpoint)
            save_results_to_csv(all_results_summary, all_results_detailed, session_timestamp)
            sys.exit(0)
            
# Final save of results
print("\n" + "=" * 80)
print("FINALIZING RESULTS")
print("=" * 80)

# Save final results
summary_path, detailed_path = save_results_to_csv(
    all_results_summary, 
    all_results_detailed, 
    session_timestamp
)
print(f"\n✓ Summary results saved to: {summary_path}")
print(f"✓ Detailed results saved to: {detailed_path}")

# Clean up checkpoint file (training complete)
if checkpoint_file.exists():
    checkpoint_backup = checkpoint_dir / f'checkpoint_completed_{session_timestamp}.json'
    checkpoint_file.rename(checkpoint_backup)
    print(f"✓ Checkpoint archived to: {checkpoint_backup}")

summary_df = pd.DataFrame(all_results_summary)
detailed_df = pd.DataFrame(all_results_detailed)
print("=" * 80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Summary CSV (simple format)
summary_df = pd.DataFrame(all_results_summary)
summary_path = results_dir / f'enhanced_results_summary_{timestamp}.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary results saved to: {summary_path}")

# Detailed CSV (extended metrics)
detailed_df = pd.DataFrame(all_results_detailed)
detailed_path = results_dir / f'enhanced_results_detailed_{timestamp}.csv'
detailed_df.to_csv(detailed_path, index=False)
print(f"✓ Detailed results saved to: {detailed_path}")

# Print overall analysis
print("\n" + "=" * 80)
print("OVERALL BEST MODELS BY TARGET")
print("=" * 80)

for target in targets:
    target_results = summary_df[summary_df['target'] == target]
    best = target_results.loc[target_results['r2'].idxmax()]
    print(f"\n{target.upper()}:")
    print(f"  Best Model: {best['model']}")
    print(f"  RMSE: {best['rmse']:.4f}")
    print(f"  MAE:  {best['mae']:.4f}")
    print(f"  R²:   {best['r2']:.4f}")

# Print model ranking across all targets
print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS ALL TARGETS")
print("=" * 80)

avg_performance = summary_df.groupby('model').agg({
    'rmse': 'mean',
    'mae': 'mean',
    'r2': 'mean'
}).sort_values('r2', ascending=False)

print("\nTop 10 models by average R²:")
print(avg_performance.head(10).to_string())

# Training time analysis
print("\n" + "=" * 80)
print("TRAINING TIME ANALYSIS")
print("=" * 80)

avg_time = detailed_df.groupby('model')['train_time_seconds'].mean().sort_values()
print("\nAverage training time per target:")
print(avg_time.to_string())

print("\n" + "=" * 80)
print("✅ ENHANCED MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModels saved to: {models_dir}/")
print(f"Summary results: {summary_path}")
print(f"Detailed results: {detailed_path}")
print(f"\nTotal models trained: {len(all_results_summary)}")
print(f"Total targets: {len(targets)}")
print(f"Total time: {detailed_df['train_time_seconds'].sum():.2f} seconds")
print("=" * 80)
