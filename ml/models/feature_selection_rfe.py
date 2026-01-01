"""
Recursive Feature Elimination (RFE) - Conservative Approach
EXPERIMENT 2 (414 Features with Artist Features)

Methodology:
- Model: CatBoost_tuned (single authority)
- Strategy: Conservative (remove 10 features per iteration)
- Stop Condition: When R² drops > 1% from previous iteration
- Metric: R² score
- Approach: Separate RFE for each target (4 independent feature subsets)

Outputs:
- CSV logs with iteration details
- Final optimal feature lists per target
- Visualization plots (R² vs feature count)
- Feature importance rankings
- Feature group analysis (audio, text, sentiment, embeddings)

Post-RFE: Retrain 6 models (XGBoost_tuned, CatBoost, CatBoost_tuned, LightGBM_tuned, MLPRegressor, MLPRegressor_tuned)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("RECURSIVE FEATURE ELIMINATION (RFE)")
print("EXPERIMENT 2 - Conservative Approach with CatBoost_tuned")
print("=" * 80)

# Configuration
FEATURES_PER_ITERATION = 10  # Conservative: remove 10 features per step
R2_DROP_THRESHOLD = 0.01  # Stop if R² drops > 1%
MIN_FEATURES = 20  # Safety minimum (don't go below this)

# Paths
features_dir = Path('../features')
models_dir = Path('../models/saved/experiment2_with_artist/rfe')
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = Path('../../results/metrics/experiment2_with_artist/rfe')
results_dir.mkdir(exist_ok=True, parents=True)

plots_dir = Path('../../results/figures/rfe')
plots_dir.mkdir(exist_ok=True, parents=True)

# Checkpoint
checkpoint_dir = results_dir / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True, parents=True)
checkpoint_file = checkpoint_dir / 'rfe_checkpoint.json'

session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Feature group definitions
FEATURE_GROUPS = {
    'audio': list(range(0, 23)),  # 23 audio features (including artist)
    'text': list(range(23, 28)),  # 5 text stats
    'sentiment': list(range(28, 30)),  # 2 sentiment features
    'embeddings': list(range(30, 414))  # 384 embedding dimensions
}

# Load feature names
print("\nLoading feature names...")
# Note: X_train_audio.npy contains 23 features (21 audio + 2 artist combined)
audio_names = [
    'acousticness', 'instrumentalness', 'speechiness', 'liveness',
    'loudness', 'tempo', 'duration_ms', 'year', 'mode',
    'key_sin', 'key_cos',
    'genre_Blues', 'genre_Classical', 'genre_Country', 'genre_Electronic',
    'genre_Folk', 'genre_Hip-Hop', 'genre_Jazz', 'genre_Pop', 'genre_R&B', 'genre_Rock',
    'log_total_artist_followers', 'avg_artist_popularity'
]  # 23 features total (21 audio + 2 artist)

text_names = ['word_count', 'unique_words', 'unique_ratio', 'avg_word_length', 'char_count']
sentiment_names = ['polarity', 'subjectivity']
embedding_names = [f'embedding_{i}' for i in range(384)]

all_feature_names = audio_names + text_names + sentiment_names + embedding_names

print(f"Total features: {len(all_feature_names)}")
print(f"  Audio+Artist: {len(audio_names)}")
print(f"  Text: {len(text_names)}")
print(f"  Sentiment: {len(sentiment_names)}")
print(f"  Embeddings: {len(embedding_names)}")

# Load data
print("\n" + "=" * 80)
print("LOADING FEATURES")
print("=" * 80)

# Audio
X_train_audio = np.load(features_dir / 'X_train_audio.npy')
X_val_audio = np.load(features_dir / 'X_val_audio.npy')
print(f"✓ Audio: train={X_train_audio.shape}, val={X_val_audio.shape}")

# Text
X_train_text = np.load(features_dir / 'X_train_text_stats.npy')
X_val_text = np.load(features_dir / 'X_val_text_stats.npy')
print(f"✓ Text: train={X_train_text.shape}, val={X_val_text.shape}")

# Sentiment
X_train_sentiment = np.load(features_dir / 'X_train_sentiment.npy')
X_val_sentiment = np.load(features_dir / 'X_val_sentiment.npy')
print(f"✓ Sentiment: train={X_train_sentiment.shape}, val={X_val_sentiment.shape}")

# Embeddings
X_train_embeddings = np.load(features_dir / 'X_train_embeddings.npy')
X_val_embeddings = np.load(features_dir / 'X_val_embeddings.npy')
print(f"✓ Embeddings: train={X_train_embeddings.shape}, val={X_val_embeddings.shape}")

# Combine
X_train_full = np.hstack([X_train_audio, X_train_text, X_train_sentiment, X_train_embeddings])
X_val_full = np.hstack([X_val_audio, X_val_text, X_val_sentiment, X_val_embeddings])

print(f"\nCombined features: {X_train_full.shape[1]}")

# Define targets
targets = ['valence', 'energy', 'danceability', 'popularity']

# CatBoost configuration (tuned version from enhanced_models.py)
def get_catboost_model():
    """Returns CatBoost_tuned configuration"""
    return CatBoostRegressor(
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
    )

def load_checkpoint():
    """Load checkpoint or create new"""
    if checkpoint_file.exists():
        print("\n" + "=" * 80)
        print("CHECKPOINT FOUND")
        print("=" * 80)
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"Completed targets: {checkpoint.get('completed_targets', [])}")
        response = input("\nResume from checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            return checkpoint
    return {
        'session_timestamp': session_timestamp,
        'completed_targets': [],
        'results': {}
    }

def save_checkpoint(checkpoint):
    """Save checkpoint"""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def analyze_feature_groups(feature_indices):
    """Analyze which feature groups are represented"""
    groups_count = {
        'audio': sum(1 for i in feature_indices if i in FEATURE_GROUPS['audio']),
        'text': sum(1 for i in feature_indices if i in FEATURE_GROUPS['text']),
        'sentiment': sum(1 for i in feature_indices if i in FEATURE_GROUPS['sentiment']),
        'embeddings': sum(1 for i in feature_indices if i in FEATURE_GROUPS['embeddings'])
    }
    return groups_count

def perform_rfe_for_target(target, X_train, X_val, y_train, y_val, checkpoint):
    """Perform RFE for a single target"""
    
    print("\n" + "=" * 80)
    print(f"RFE FOR TARGET: {target.upper()}")
    print("=" * 80)
    
    # Check if already completed
    if target in checkpoint.get('completed_targets', []):
        print(f"✓ {target} already completed, skipping...")
        return checkpoint['results'][target]
    
    # Initialize with all features
    current_features = list(range(X_train.shape[1]))
    iteration_results = []
    
    # Baseline: train with all features
    print(f"\nIteration 0: Training with {len(current_features)} features...")
    model = get_catboost_model()
    eval_set = [(X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    y_pred = model.predict(X_val)
    baseline_r2 = r2_score(y_val, y_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    baseline_mae = mean_absolute_error(y_val, y_pred)
    
    print(f"  Baseline R²: {baseline_r2:.4f}, RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")
    
    # Store baseline
    groups = analyze_feature_groups(current_features)
    iteration_results.append({
        'iteration': 0,
        'num_features': len(current_features),
        'r2': float(baseline_r2),
        'rmse': float(baseline_rmse),
        'mae': float(baseline_mae),
        'r2_drop': 0.0,  # Total drop from baseline
        'r2_drop_iteration': 0.0,  # Drop from previous iteration
        'removed_features': [],
        'audio_count': groups['audio'],
        'text_count': groups['text'],
        'sentiment_count': groups['sentiment'],
        'embeddings_count': groups['embeddings']
    })
    
    previous_r2 = baseline_r2
    best_r2 = baseline_r2  # Track best R² achieved
    best_iteration = 0  # Track which iteration had best R²
    best_features = current_features.copy()  # Track features at best iteration
    iteration = 1
    
    # RFE loop
    while len(current_features) > MIN_FEATURES:
        print(f"\nIteration {iteration}: {len(current_features)} features remaining")
        
        # Store current features BEFORE modification (critical for restoration)
        previous_features = current_features.copy()
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Map importance to current feature indices
        # Note: This assumes X_train[:, current_features] preserves order (standard NumPy behavior)
        feature_importance = list(zip(current_features, importance))
        feature_importance.sort(key=lambda x: x[1])  # Sort by importance (ascending)
        
        # Determine how many to remove
        n_to_remove = min(FEATURES_PER_ITERATION, len(current_features) - MIN_FEATURES)
        if n_to_remove <= 0:
            break
        
        # Remove least important features
        features_to_remove = [f[0] for f in feature_importance[:n_to_remove]]
        current_features = [f for f in current_features if f not in features_to_remove]
        
        print(f"  Removing {n_to_remove} least important features...")
        removed_names = [all_feature_names[i] for i in features_to_remove]
        print(f"  Features removed: {removed_names}")
        
        # Train with reduced feature set
        X_train_reduced = X_train[:, current_features]
        X_val_reduced = X_val[:, current_features]
        
        model = get_catboost_model()
        eval_set = [(X_val_reduced, y_val)]
        model.fit(X_train_reduced, y_train, eval_set=eval_set, verbose=False)
        
        y_pred = model.predict(X_val_reduced)
        current_r2 = r2_score(y_val, y_pred)
        current_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        current_mae = mean_absolute_error(y_val, y_pred)
        
        # Calculate drops: both from previous iteration and from baseline
        r2_drop_from_previous = previous_r2 - current_r2
        r2_drop_from_baseline = baseline_r2 - current_r2
        
        print(f"  R²: {current_r2:.4f} (drop from prev: {r2_drop_from_previous:.4f}, total drop: {r2_drop_from_baseline:.4f})")
        print(f"  RMSE: {current_rmse:.4f}, MAE: {current_mae:.4f}")
        
        # Store results (use drop from baseline for logging total degradation)
        groups = analyze_feature_groups(current_features)
        iteration_results.append({
            'iteration': iteration,
            'num_features': len(current_features),
            'r2': float(current_r2),
            'rmse': float(current_rmse),
            'mae': float(current_mae),
            'r2_drop': float(r2_drop_from_baseline),  # Total drop from baseline
            'r2_drop_iteration': float(r2_drop_from_previous),  # Drop from previous
            'removed_features': [all_feature_names[i] for i in features_to_remove],
            'audio_count': groups['audio'],
            'text_count': groups['text'],
            'sentiment_count': groups['sentiment'],
            'embeddings_count': groups['embeddings']
        })
        
        # Check stopping condition: total drop from baseline exceeds threshold
        if r2_drop_from_baseline > R2_DROP_THRESHOLD:
            print(f"\n  ⚠️ Total R² drop from baseline: {r2_drop_from_baseline:.4f} (> {R2_DROP_THRESHOLD} threshold)")
            print(f"  Stopping RFE. Restoring previous feature set.")
            print(f"  Optimal feature count: {len(previous_features)}")
            # CORRECTED: Restore the feature set from BEFORE this iteration
            optimal_features = previous_features
            break
        
        # Update tracking variables
        previous_r2 = current_r2
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_iteration = iteration
            best_features = current_features.copy()
        iteration += 1
    
    else:
        # Reached minimum features without threshold breach
        optimal_features = current_features
        print(f"\n  ✓ Reached minimum feature count: {MIN_FEATURES}")
    
    # Use the iteration with the BEST R² score (not just the stopping point)
    if best_r2 > baseline_r2 and best_iteration > 0:
        print(f"\n  ℹ️  Best R² was at iteration {best_iteration}: {best_r2:.4f}")
        print(f"  Using features from iteration {best_iteration} for final models")
        optimal_features = best_features
        optimal_r2 = best_r2
    else:
        # If no improvement, use baseline
        print(f"\n  ℹ️  No improvement over baseline. Using baseline features.")
        optimal_features = list(range(414))
        optimal_r2 = baseline_r2
    
    # Save results for this target
    target_results = {
        'iterations': iteration_results,
        'optimal_features': optimal_features,
        'optimal_feature_names': [all_feature_names[i] for i in optimal_features],
        'baseline_r2': float(baseline_r2),
        'optimal_r2': float(optimal_r2),
        'best_iteration': best_iteration,
        'n_features_removed': 414 - len(optimal_features)
    }
    
    # Save iteration log
    iterations_df = pd.DataFrame(iteration_results)
    iterations_path = results_dir / f'rfe_iterations_{target}_{session_timestamp}.csv'
    iterations_df.to_csv(iterations_path, index=False)
    print(f"\n✓ Iteration log saved: {iterations_path}")
    
    # Save optimal feature list
    optimal_features_df = pd.DataFrame({
        'feature_index': optimal_features,
        'feature_name': [all_feature_names[i] for i in optimal_features]
    })
    optimal_path = results_dir / f'optimal_features_{target}_{session_timestamp}.csv'
    optimal_features_df.to_csv(optimal_path, index=False)
    print(f"✓ Optimal features saved: {optimal_path}")
    
    # Create visualization
    plot_rfe_results(iterations_df, target)
    
    return target_results

def plot_rfe_results(iterations_df, target):
    """Create visualization of RFE results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'RFE Results: {target.capitalize()}', fontsize=18, fontweight='bold')
    
    # R² vs Feature Count
    ax = axes[0, 0]
    ax.plot(iterations_df['num_features'], iterations_df['r2'], 
            marker='o', linewidth=2.5, markersize=8, color='#2ecc71')
    ax.axhline(y=iterations_df['r2'].iloc[0], color='red', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.7)
    ax.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('R² vs Feature Count', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # R² Drop per Iteration
    ax = axes[0, 1]
    ax.bar(iterations_df['iteration'], iterations_df['r2_drop'], 
           color='#e74c3c', edgecolor='black', linewidth=1.2, label='Total from Baseline')
    if 'r2_drop_iteration' in iterations_df.columns:
        ax.bar(iterations_df['iteration'], iterations_df['r2_drop_iteration'], 
               color='#f39c12', edgecolor='black', linewidth=1.2, alpha=0.7, label='From Previous')
    ax.axhline(y=R2_DROP_THRESHOLD, color='darkred', linestyle='--', 
               linewidth=2, label=f'Threshold ({R2_DROP_THRESHOLD})', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Drop', fontsize=14, fontweight='bold')
    ax.set_title('R² Drop per Iteration', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Feature Group Evolution
    ax = axes[1, 0]
    ax.plot(iterations_df['num_features'], iterations_df['audio_count'], 
            marker='o', linewidth=2, label='Audio', color='#3498db')
    ax.plot(iterations_df['num_features'], iterations_df['text_count'], 
            marker='s', linewidth=2, label='Text', color='#e67e22')
    ax.plot(iterations_df['num_features'], iterations_df['sentiment_count'], 
            marker='^', linewidth=2, label='Sentiment', color='#9b59b6')
    ax.plot(iterations_df['num_features'], iterations_df['embeddings_count'], 
            marker='d', linewidth=2, label='Embeddings', color='#1abc9c')
    ax.set_xlabel('Total Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature Count', fontsize=14, fontweight='bold')
    ax.set_title('Feature Group Evolution', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # RMSE vs Feature Count
    ax = axes[1, 1]
    ax.plot(iterations_df['num_features'], iterations_df['rmse'], 
            marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
    ax.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title('RMSE vs Feature Count', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / f'rfe_analysis_{target}_{session_timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")

def retrain_with_optimal_features(target, optimal_features, X_train, X_val, y_train, y_val):
    """Retrain 4 models with optimal features"""
    
    print(f"\n" + "=" * 80)
    print(f"RETRAINING MODELS WITH OPTIMAL FEATURES: {target.upper()}")
    print(f"Features: {len(optimal_features)}")
    print("=" * 80)
    
    # Reduce feature sets
    X_train_opt = X_train[:, optimal_features]
    X_val_opt = X_val[:, optimal_features]
    
    models = {
        'XGBoost_tuned': xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=10,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            colsample_bylevel=1.0, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            random_state=42, n_jobs=-1, early_stopping_rounds=50
        ),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
        'CatBoost_tuned': CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=10,
            l2_leaf_reg=8, subsample=0.8, bootstrap_type='Bernoulli',
            random_state=42, verbose=False, early_stopping_rounds=50,
            thread_count=-1, grow_policy='Lossguide', max_leaves=64
        ),
        'LightGBM_tuned': lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.06, num_leaves=63,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.5, min_split_gain=0.05, 
            random_state=42, n_jobs=-1, verbose=-1, feature_fraction=0.7,
            bagging_freq=5, bagging_fraction=0.7, importance_type='gain'
        ),
        'MLPRegressor': MLPRegressor(random_state=42, early_stopping=True),
        'MLPRegressor_tuned': MLPRegressor(
            hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
            alpha=0.005, batch_size=1024, learning_rate='adaptive',
            learning_rate_init=0.0005, power_t=0.5, random_state=42, 
            max_iter=500, shuffle=True, early_stopping=True, 
            validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
        )
    }
    
    retraining_results = []
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        start_time = time.time()
        
        # Train with early stopping for boosting models
        if 'XGBoost' in model_name or 'CatBoost' in model_name:
            eval_set = [(X_val_opt, y_val)]
            model.fit(X_train_opt, y_train, eval_set=eval_set, verbose=False)
        elif 'LightGBM' in model_name:
            eval_set = [(X_val_opt, y_val)]
            model.fit(X_train_opt, y_train, eval_set=eval_set, 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model.fit(X_train_opt, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_val_opt)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"    R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {train_time:.2f}s")
        
        # Save model
        model_path = models_dir / f'{model_name}_{target}_rfe.pkl'
        joblib.dump(model, model_path)
        
        retraining_results.append({
            'target': target,
            'model': model_name,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'train_time': float(train_time),
            'num_features': len(optimal_features)
        })
    
    return retraining_results

# Main execution
checkpoint = load_checkpoint()
all_target_results = checkpoint.get('results', {})
all_retraining_results = []

for target in targets:
    # Load target values
    y_train = np.load(features_dir / f'y_train_{target}.npy')
    y_val = np.load(features_dir / f'y_val_{target}.npy')
    
    # Perform RFE
    target_results = perform_rfe_for_target(
        target, X_train_full, X_val_full, y_train, y_val, checkpoint
    )
    
    all_target_results[target] = target_results
    
    # Update checkpoint
    if target not in checkpoint['completed_targets']:
        checkpoint['completed_targets'].append(target)
    checkpoint['results'] = all_target_results
    save_checkpoint(checkpoint)
    
    # Retrain models with optimal features
    retraining_results = retrain_with_optimal_features(
        target, target_results['optimal_features'],
        X_train_full, X_val_full, y_train, y_val
    )
    all_retraining_results.extend(retraining_results)

# Save final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

summary_data = []
for target, results in all_target_results.items():
    summary_data.append({
        'target': target,
        'baseline_r2': results['baseline_r2'],
        'optimal_r2': results['optimal_r2'],
        'best_iteration': results.get('best_iteration', 0),
        'original_features': 414,
        'optimal_features': len(results['optimal_features']),
        'features_removed': results['n_features_removed'],
        'reduction_pct': (results['n_features_removed'] / 414) * 100
    })

summary_df = pd.DataFrame(summary_data)
summary_path = results_dir / f'rfe_summary_{session_timestamp}.csv'
summary_df.to_csv(summary_path, index=False)

print(f"\n{summary_df.to_string(index=False)}")
print(f"\n✓ Summary saved: {summary_path}")

# Save retraining results
retraining_df = pd.DataFrame(all_retraining_results)
retraining_path = results_dir / f'rfe_retraining_results_{session_timestamp}.csv'
retraining_df.to_csv(retraining_path, index=False)
print(f"✓ Retraining results saved: {retraining_path}")

# Clean up checkpoint
if checkpoint_file.exists():
    backup_path = checkpoint_dir / f'rfe_checkpoint_completed_{session_timestamp}.json'
    checkpoint_file.rename(backup_path)
    print(f"✓ Checkpoint archived: {backup_path}")

print("\n" + "=" * 80)
print("✅ RFE COMPLETE!")
print("=" * 80)
print(f"\nResults directory: {results_dir}/")
print(f"Plots directory: {plots_dir}/")
print(f"Models directory: {models_dir}/")
