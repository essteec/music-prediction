"""
Retrain Models at Best RFE Iterations
Loads features from the best performing iterations and retrains 6 models

Best Iterations:
- valence: 23rd iteration (184 features)
- energy: 38th iteration
- danceability: 34th iteration
- popularity: 2nd iteration
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

print("=" * 80)
print("RETRAIN MODELS AT BEST RFE ITERATIONS")
print("=" * 80)

# Configuration
BEST_ITERATIONS = {
    'valence': 23,
    'energy': 38,
    'danceability': 34,
    'popularity': 2
}

# Paths
features_dir = Path('../features')
models_dir = Path('../models/saved/experiment2_with_artist/rfe_best')
models_dir.mkdir(exist_ok=True, parents=True)

results_dir = Path('../../results/metrics/experiment2_with_artist/rfe')
output_dir = Path('../../results/metrics/experiment2_with_artist/rfe_best')
output_dir.mkdir(exist_ok=True, parents=True)

session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Feature names (matching file structure)
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

print(f"\nTotal features: {len(all_feature_names)}")

# Load data
print("\n" + "=" * 80)
print("LOADING FEATURES")
print("=" * 80)

X_train_audio = np.load(features_dir / 'X_train_audio.npy')
X_val_audio = np.load(features_dir / 'X_val_audio.npy')
X_train_text = np.load(features_dir / 'X_train_text_stats.npy')
X_val_text = np.load(features_dir / 'X_val_text_stats.npy')
X_train_sentiment = np.load(features_dir / 'X_train_sentiment.npy')
X_val_sentiment = np.load(features_dir / 'X_val_sentiment.npy')
X_train_embeddings = np.load(features_dir / 'X_train_embeddings.npy')
X_val_embeddings = np.load(features_dir / 'X_val_embeddings.npy')

X_train_full = np.hstack([X_train_audio, X_train_text, X_train_sentiment, X_train_embeddings])
X_val_full = np.hstack([X_val_audio, X_val_text, X_val_sentiment, X_val_embeddings])

print(f"✓ Combined features: {X_train_full.shape[1]}")

# Load RFE iteration logs
print("\n" + "=" * 80)
print("LOADING RFE ITERATION LOGS")
print("=" * 80)

iterations_file = results_dir / 'rfe_iterations_20260101_023946.txt'
iterations_df = pd.read_csv(iterations_file)

print(f"✓ Loaded {len(iterations_df)} iteration records")

def get_optimal_features_at_iteration(target, best_iteration):
    """
    Reconstruct the feature set at a specific iteration by tracking removals
    """
    print(f"\n  Reconstructing features for {target} at iteration {best_iteration}...")
    
    # Start with all 414 features
    remaining_features = set(range(414))
    
    # Get all iterations up to and including the best iteration
    target_iterations = iterations_df[
        (iterations_df['target'] == target) & 
        (iterations_df['iteration'] <= best_iteration) &
        (iterations_df['iteration'] > 0)  # Skip baseline (iteration 0)
    ].sort_values('iteration')
    
    print(f"  Processing {len(target_iterations)} iterations...")
    
    # Remove features that were eliminated in each iteration
    for _, row in target_iterations.iterrows():
        removed_str = row['removed_features']
        if removed_str and removed_str != '[]':
            # Parse the string list
            removed_str = removed_str.strip('[]')
            if removed_str:
                removed_features = [f.strip().strip("'\"") for f in removed_str.split(',')]
                
                # Convert feature names to indices
                for feat_name in removed_features:
                    if feat_name in all_feature_names:
                        feat_idx = all_feature_names.index(feat_name)
                        remaining_features.discard(feat_idx)
    
    optimal_features = sorted(list(remaining_features))
    print(f"  ✓ {len(optimal_features)} features remaining")
    
    return optimal_features

def get_models():
    """Return dictionary of models to train"""
    return {
        'XGBoost_tuned': xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=10,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            colsample_bylevel=1.0, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            random_state=42, n_jobs=-1, early_stopping_rounds=50
        ),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
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
        'LightGBM_tuned': lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.06, num_leaves=63,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.5, min_split_gain=0.05, 
            random_state=42, n_jobs=-1, verbose=-1, feature_fraction=0.7,
            bagging_freq=5, bagging_fraction=0.7, importance_type='gain'
        ),
        'MLPRegressor': MLPRegressor(random_state=42, early_stopping=True, max_iter=500),
        'MLPRegressor_tuned': MLPRegressor(
            hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
            alpha=0.005, batch_size=1024, learning_rate='adaptive',
            learning_rate_init=0.0005, power_t=0.5, random_state=42, 
            max_iter=500, shuffle=True, early_stopping=True, 
            validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
        )
    }

# Main execution
all_results = []

for target, best_iter in BEST_ITERATIONS.items():
    print("\n" + "=" * 80)
    print(f"TARGET: {target.upper()} (Iteration {best_iter})")
    print("=" * 80)
    
    # Handle valance/valence spelling
    target_in_file = 'valance' if target == 'valence' else target
    
    # Get optimal features at this iteration
    optimal_features = get_optimal_features_at_iteration(target_in_file, best_iter)
    
    # Save optimal feature list
    optimal_features_df = pd.DataFrame({
        'feature_index': optimal_features,
        'feature_name': [all_feature_names[i] for i in optimal_features]
    })
    optimal_path = output_dir / f'optimal_features_{target}_iter{best_iter}_{session_timestamp}.csv'
    optimal_features_df.to_csv(optimal_path, index=False)
    print(f"\n✓ Optimal features saved: {optimal_path}")
    
    # Load target values
    y_train = np.load(features_dir / f'y_train_{target}.npy')
    y_val = np.load(features_dir / f'y_val_{target}.npy')
    
    # Reduce feature sets
    X_train_opt = X_train_full[:, optimal_features]
    X_val_opt = X_val_full[:, optimal_features]
    
    print(f"\nTraining shape: {X_train_opt.shape}")
    print(f"Validation shape: {X_val_opt.shape}")
    
    # Train all models
    models = get_models()
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        start_time = time.time()
        
        # Train with early stopping for boosting models
        try:
            if 'XGBoost' in model_name:
                eval_set = [(X_val_opt, y_val)]
                model.fit(X_train_opt, y_train, eval_set=eval_set, verbose=False)
            elif 'CatBoost' in model_name:
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
            model_path = models_dir / f'{model_name}_{target}_iter{best_iter}.pkl'
            joblib.dump(model, model_path)
            
            all_results.append({
                'target': target,
                'iteration': best_iter,
                'model': model_name,
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'train_time': float(train_time),
                'num_features': len(optimal_features)
            })
            
        except Exception as e:
            print(f"    ✗ Error training {model_name}: {str(e)}")

# Save all results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
results_path = output_dir / f'retrained_models_best_iterations_{session_timestamp}.csv'
results_df.to_csv(results_path, index=False)

# Print summary
print("\nSummary by Target:")
for target in BEST_ITERATIONS.keys():
    target_results = results_df[results_df['target'] == target]
    if len(target_results) > 0:
        best_model = target_results.loc[target_results['r2'].idxmax()]
        print(f"\n{target.upper()}:")
        print(f"  Best iteration: {BEST_ITERATIONS[target]}")
        print(f"  Features: {best_model['num_features']}")
        print(f"  Best model: {best_model['model']}")
        print(f"  R²: {best_model['r2']:.4f}")
        print(f"  RMSE: {best_model['rmse']:.4f}")

print(f"\n✓ All results saved: {results_path}")
print(f"✓ Models saved to: {models_dir}/")

print("\n" + "=" * 80)
print("✅ RETRAINING COMPLETE!")
print("=" * 80)
