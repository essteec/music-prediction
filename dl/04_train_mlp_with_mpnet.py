"""
Train MLP with MPNet Embeddings (Phase 1B)

Comparison to Phase 0:
- Phase 0: 414 features (23 audio + 5 text + 2 sentiment + 384 MiniLM)
- Phase 1B: 798 features (23 audio + 5 text + 2 sentiment + 768 MPNet)

Goal: Beat Phase 0 baseline with better text representations
Expected: Valence R² improvement (0.35 → 0.50+)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import sys
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from models import MusicMLP
from metrics import compute_metrics, print_metrics


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_mpnet_data(batch_size=256, num_workers=4):
    """Load data with MPNet embeddings (798 features total)."""
    print("\nLoading features:")
    
    # Feature directory
    feat_dir = Path('ml/features')
    emb_dir = Path('data/embeddings')
    
    # Load all feature types for each split
    splits = {}
    for split_name in ['train', 'val', 'test']:
        print(f"\n  {split_name.capitalize()} split:")
        
        # Load existing features
        audio = np.load(feat_dir / f'X_{split_name}_audio.npy')
        text = np.load(feat_dir / f'X_{split_name}_text_stats.npy')
        sentiment = np.load(feat_dir / f'X_{split_name}_sentiment.npy')
        
        # Load MPNet embeddings (768-d)
        mpnet = np.load(feat_dir / f'X_{split_name}_mpnet.npy')
        
        # Concatenate all features
        X = np.hstack([audio, text, sentiment, mpnet])
        
        # Load targets (all 4)
        y_valence = np.load(feat_dir / f'y_{split_name}_valence.npy')
        y_energy = np.load(feat_dir / f'y_{split_name}_energy.npy')
        y_dance = np.load(feat_dir / f'y_{split_name}_danceability.npy')
        y_pop = np.load(feat_dir / f'y_{split_name}_popularity.npy')
        
        # Stack targets: [valence, energy, danceability, popularity]
        y = np.stack([y_valence, y_energy, y_dance, y_pop], axis=1)
        
        print(f"    Features: {X.shape} (audio={audio.shape[1]}, text={text.shape[1]}, "
              f"sentiment={sentiment.shape[1]}, mpnet={mpnet.shape[1]})")
        print(f"    Targets: {y.shape}")
        print(f"    Total features: {X.shape[1]}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        # Use shuffle=True for train, False for val/test (deterministic)
        shuffle = (split_name == 'train')
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(42)  # Deterministic shuffling
        )
        
        splits[split_name] = loader
    
    return splits['train'], splits['val'], splits['test']


def train_epoch(model, train_loader, criterion, optimizer, device, weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        predictions = model(X_batch)
        
        # Compute loss (criterion is MSE with reduction='none')
        loss = criterion(predictions, y_batch)
        
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device, weights=None):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            
            # Compute loss
            loss = criterion(predictions, y_batch)
            
            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(targets, predictions)
    
    return avg_loss, metrics


def main():
    # Set seed for reproducibility (FIRST THING!)
    set_seed(42)
    
    print("=" * 70)
    print("Phase 1B: Training MLP with MPNet Embeddings")
    print("=" * 70)
    
    # Configuration
    config = {
        'seed': 42,
        'batch_size': 256,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 10,  # Early stopping
        'dropout': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'input_size': 798,  # 23 audio + 5 text + 2 sentiment + 768 MPNet
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data with MPNet embeddings
    print("\n" + "=" * 70)
    print("Loading Data with MPNet Embeddings")
    print("=" * 70)
    
    train_loader, val_loader, test_loader = load_mpnet_data(
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Create model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)
    
    device = torch.device(config['device'])
    model = MusicMLP(input_size=798, num_targets=4, dropout=config['dropout'])
    model = model.to(device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input size: {config['input_size']}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    
    # Loss and optimizer
    # Use reduction='none' to allow manual weighting per target
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Loss weighting: same as Phase 0 baseline
    # Down-weight popularity (0.5), up-weight valence and danceability (2.0)
    loss_weights = torch.tensor([2.0, 1.0, 2.0, 0.5]).to(device)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, weights=loss_weights)
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, weights=loss_weights)
        
        # Track history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_r2_avg': np.mean([m['r2'] for m in val_metrics.values()])
        })
        
        # Print progress
        val_r2_avg = history[-1]['val_r2_avg']
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val R²: {val_r2_avg:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = 'models/checkpoints/mlp_mpnet_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config,
                'loss_weights': loss_weights
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    checkpoint = torch.load('models/checkpoints/mlp_mpnet_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on all splits
    train_loss, train_metrics = evaluate(model, train_loader, criterion, device, weights=loss_weights)
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device, weights=loss_weights)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, weights=loss_weights)
    
    print_metrics(train_metrics, "Train Metrics")
    print_metrics(val_metrics, "Validation Metrics")
    print_metrics(test_metrics, "Test Metrics")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/dl_metrics/mlp_mpnet_{timestamp}.csv'
    
    # Combine metrics
    all_metrics = []
    for split, metrics in [('train', train_metrics), 
                           ('val', val_metrics), 
                           ('test', test_metrics)]:
        for target, values in metrics.items():
            all_metrics.append({
                'timestamp': timestamp,
                'model': 'MLP_MPNet',
                'phase': '1B',
                'split': split,
                'target': target,
                'input_features': 798,
                **values
            })
    
    df = pd.DataFrame(all_metrics)
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Compare to Phase 0 baseline
    print("\n" + "=" * 70)
    print("Comparison to Phase 0 Baseline (414 features)")
    print("=" * 70)
    
    phase0_baseline = {
        'valence': 0.35,
        'energy': 0.75,
        'danceability': 0.47,
        'popularity': 0.12
    }
    
    print(f"\n{'Target':<15} {'Phase 0':<10} {'Phase 1B':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)
    
    for target in ['valence', 'energy', 'danceability', 'popularity']:
        p0_r2 = phase0_baseline.get(target, 0)
        p1b_r2 = test_metrics[target]['r2']
        diff = p1b_r2 - p0_r2
        
        if diff > 0.02:
            status = "✓ Better"
        elif diff < -0.02:
            status = "✗ Worse"
        else:
            status = "≈ Similar"
        
        print(f"{target:<15} {p0_r2:>9.4f} {p1b_r2:>9.4f} {diff:>+9.4f}  {status}")
    
    # Compare to ML baseline
    print("\n" + "=" * 70)
    print("Comparison to ML Baseline (Semester 1)")
    print("=" * 70)
    
    ml_baseline = {
        'energy': 0.81,
        'danceability': 0.55,
        'valence': 0.45,
        'popularity': 0.13
    }
    
    print(f"\n{'Target':<15} {'ML R²':<10} {'DL R²':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)
    
    for target in ['valence', 'energy', 'danceability', 'popularity']:
        ml_r2 = ml_baseline.get(target, 0)
        dl_r2 = test_metrics[target]['r2']
        diff = dl_r2 - ml_r2
        status = "✓ Better" if diff > 0 else "✗ Worse" if diff < -0.05 else "≈ Similar"
        
        print(f"{target:<15} {ml_r2:>9.4f} {dl_r2:>9.4f} {diff:>+9.4f}  {status}")
    
    print("=" * 70)
    print("\n✓ Phase 1B Training complete!")
    print(f"Best model saved at: models/checkpoints/mlp_mpnet_best.pt")
    print(f"\nNext steps:")
    print("  1. Analyze if MPNet improved Valence R²")
    print("  2. If yes: Try fine-tuning BERT (Phase 1C)")
    print("  3. If no: Consider other improvements or move to Phase 2")


if __name__ == '__main__':
    main()
