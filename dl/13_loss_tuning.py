"""
Experiment H: Loss Tuning (Huber Loss + Uncertainty Weighting)

Key changes vs Exp F:
1. Replaces MSELoss with HuberLoss. MSE heavily penalizes outliers, which is 
   especially problematic for Popularity (which has a long tail).
2. Replaces static loss weights [2.0, 1.0, 2.0, 0.5] with homoscedastic 
   uncertainty weighting (Kendall et al., 2017). The model learns optimal 
   weights `log_var` dynamically during training.
3. Lower patience (24) to save time, as best models emerge early.

Usage:
    python dl/13_loss_tuning.py
"""

import os
import sys
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.fusion import TaskGatedFusionMLP
from utils.metrics import compute_metrics, print_metrics
import importlib
feat_eng = importlib.import_module("11_feature_engineering")
load_engineered_data = feat_eng.load_engineered_data

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_NAMES = ['valence', 'energy', 'danceability', 'popularity']
XGBOOST_BASELINE = {
    'valence': 0.6728, 'energy': 0.9073, 'danceability': 0.7693, 'popularity': 0.1478
}
EXP_C = {'valence': 0.6998, 'energy': 0.8894, 'danceability': 0.7306, 'popularity': 0.1133}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss with Uncertainty Weighting
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Computes Huber Loss and weights each task dynamically using homoscedastic
    uncertainty (Kendall et al., 2017).
    """
    def __init__(self, num_tasks=4):
        super().__init__()
        # log_vars initialized to 0. (precision = exp(-0) = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        # HuberLoss is more robust to outliers than MSE
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        
    def forward(self, preds, targets):
        # Calculate unweighted loss per element (B, 4)
        losses = self.criterion(preds, targets)
        # Mean across batch for each task -> (4,)
        task_losses = losses.mean(dim=0)
        
        # Apply uncertainty weighting: loss * exp(-log_var) + log_var
        precision = torch.exp(-self.log_vars)
        weighted_loss = torch.sum(precision * task_losses + self.log_vars)
        return weighted_loss

    def get_weights(self):
        """Returns the current effective weight per task (exp(-log_var))."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def unpack(batch, device):
    *mods, targets = batch
    return [m.to(device, non_blocking=True) for m in mods], targets.to(device, non_blocking=True)

def train_epoch(model, loss_fn, loader, optimizer, amp_scaler, device):
    model.train(); loss_fn.train(); total = 0.0
    for batch in loader:
        mods, tgts = unpack(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loss = loss_fn(model(*mods), tgts)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer); amp_scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loss_fn, loader, device):
    model.eval(); loss_fn.eval(); total = 0.0; preds_l, tgts_l = [], []
    for batch in loader:
        mods, tgts = unpack(batch, device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            p = model(*mods)
            total += loss_fn(p, tgts).item()
        preds_l.append(p.cpu().float()); tgts_l.append(tgts.cpu().float())
    metrics = compute_metrics(torch.cat(tgts_l), torch.cat(preds_l))
    return total / len(loader), metrics

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment H: Loss Tuning")
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--patience',       type=int,   default=24)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--dropout_enc',    type=float, default=0.2)
    parser.add_argument('--dropout_fusion', type=float, default=0.4)
    parser.add_argument('--feat_dir',       type=str,   default='ml/features')
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--no_feat_eng',    action='store_true')
    args = parser.parse_args()

    engineer = not args.no_feat_eng
    set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"exp_h_loss_tuning_{timestamp}"

    print("=" * 70)
    print("Experiment H: Huber Loss + Uncertainty Weighting")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    train_loader, val_loader, test_loader = load_engineered_data(
        args.batch_size, args.feat_dir, args.num_workers, engineer
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    meta_dim = 36 if engineer else 30
    model = TaskGatedFusionMLP(
        num_targets=4,
        dropout_enc=args.dropout_enc,
        dropout_fusion=args.dropout_fusion,
        metadata_dim=meta_dim,
    ).to(device)
    
    loss_fn = MultiTaskLoss(num_tasks=4).to(device)

    print(f"Model: {model.__class__.__name__} (metadata={meta_dim}d)  |  "
          f"Params: {model.count_parameters():,}")

    # Optimize both model parameters and loss_fn parameters (log_vars)
    optimizer  = optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), 
                             lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )
    amp_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    ckpt_path = Path('models/checkpoints') / f"{exp_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_r2    = -float('inf')
    patience_count = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val R²':>8}  {'LR':>10}  Best")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loss_fn, train_loader, optimizer,
                                 amp_scaler, device)
        val_loss, val_metrics = evaluate(model, loss_fn, val_loader, device)
        scheduler.step()
        val_r2_avg = float(np.mean([m['r2'] for m in val_metrics.values()]))
        current_lr = optimizer.param_groups[0]['lr']

        is_best = val_r2_avg > best_val_r2
        marker  = " ✓" if is_best else ""

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_r2_avg:>8.4f}  "
              f"{current_lr:>10.2e}{marker}")

        if is_best:
            best_val_r2    = val_r2_avg
            patience_count = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'loss_state_dict': loss_fn.state_dict(),
                        'val_r2': val_r2_avg, 'val_metrics': val_metrics,
                        'args': vars(args)}, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break
                
        # Optional: Print learned weights every 5 epochs
        if epoch % 5 == 0:
            weights = loss_fn.get_weights()
            w_str = ", ".join([f"{n}: {w:.3f}" for n, w in zip(TARGET_NAMES, weights)])
            print(f"        Learned Weights -> {w_str}")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    loss_fn.load_state_dict(ckpt['loss_state_dict'])
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val_r2={ckpt['val_r2']:.4f})")
    
    weights = loss_fn.get_weights()
    w_str = ", ".join([f"{n}: {w:.3f}" for n, w in zip(TARGET_NAMES, weights)])
    print(f"Final Learned Weights: {w_str}")

    _, test_metrics = evaluate(model, loss_fn, test_loader, device)
    print_metrics(test_metrics, "Test Metrics")

    results_dir  = Path('results/dl_metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{exp_name}.csv"
    rows = [{'timestamp': timestamp, 'experiment': 'H', 'model': 'TaskGatedFusionMLP',
              'phase': '4H', 'split': 'test', 'target': t, **test_metrics[t]}
            for t in TARGET_NAMES]
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n✓ Results: {results_path}\n✓ Checkpoint: {ckpt_path}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Comparison: ExpC → ExpH → XGBoost")
    print("=" * 70)
    print(f"{'Target':<14} {'ExpC':>8}  {'ExpH':>8}  {'ΔvsC':>8}  {'XGB':>8}  Status")
    print("-" * 65)
    beaten = 0
    for t in TARGET_NAMES:
        r2  = test_metrics[t]['r2']
        d_c = r2 - EXP_C[t]
        if r2 > XGBOOST_BASELINE[t]:
            status = "✅ beats XGB"; beaten += 1
        elif r2 > EXP_C[t]:
            status = "↑ improved"
        else:
            status = "↓ regressed"
        print(f"{t:<14} {EXP_C[t]:>8.4f}  {r2:>8.4f}  {'+' if d_c>=0 else ''}{d_c:>7.4f}  "
              f"{XGBOOST_BASELINE[t]:>8.4f}  {status}")
    avg_r2 = np.mean([test_metrics[t]['r2'] for t in TARGET_NAMES])
    print(f"\n  Avg R²: {avg_r2:.4f}  |  XGB avg: 0.6243  |  Exp C avg: 0.6083  "
          f"|  Beat XGB on {beaten}/4 targets")


if __name__ == '__main__':
    main()
