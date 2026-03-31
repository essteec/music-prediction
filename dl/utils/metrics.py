"""
Evaluation metrics for music prediction.
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred, target_names=['valence', 'energy', 'danceability', 'popularity']):
    """
    Compute R², RMSE, MAE for predictions.
    
    Args:
        y_true: True values (numpy array or tensor)
        y_pred: Predicted values (numpy array or tensor)
        target_names: Names of targets for reporting
        
    Returns:
        dict with metrics per target
    """
    # Convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Handle single target (reshape to 2D)
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    metrics = {}
    num_targets = y_true.shape[1]
    
    for i in range(num_targets):
        target_name = target_names[i] if i < len(target_names) else f'target_{i}'
        
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        
        metrics[target_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Target':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 60)
    
    for target, values in metrics.items():
        print(f"{target:<15} {values['r2']:>9.4f} {values['rmse']:>9.4f} {values['mae']:>9.4f}")
    
    # Average metrics
    avg_r2 = np.mean([v['r2'] for v in metrics.values()])
    avg_rmse = np.mean([v['rmse'] for v in metrics.values()])
    avg_mae = np.mean([v['mae'] for v in metrics.values()])
    
    print("-" * 60)
    print(f"{'Average':<15} {avg_r2:>9.4f} {avg_rmse:>9.4f} {avg_mae:>9.4f}")
    print("=" * 60)


def save_metrics_csv(metrics, filepath, epoch=None):
    """Save metrics to CSV file."""
    import pandas as pd
    
    # Flatten metrics
    rows = []
    for target, values in metrics.items():
        row = {'target': target}
        if epoch is not None:
            row['epoch'] = epoch
        row.update(values)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Append or create
    import os
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    return df
