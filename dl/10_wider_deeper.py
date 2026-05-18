"""
Experiment E: Wider / Deeper Model

Hypothesis: Exp C/D peaked at epoch 4 with 3.2M params because
ENCODED_DIM=128 is a hard representational bottleneck. The PANNS encoder
crushes 2048 acoustic dims into 128 in two hops — too lossy for Energy/Dance.

Changes vs Exp C:
  - ENCODED_DIM: 128 → 256  (encoders retain 2× more information)
  - Fused dim: 768 → 1536
  - Fusion MLP: 2 ResBlocks instead of 1, 1024→256 instead of 512→256
  - dropout_enc=0.3, dropout_fusion=0.5  (up from 0.2/0.4 to handle extra capacity)
  - Same CosineAnnealingLR from Exp D (full LR from epoch 1)
  - Total params: ~10M

Usage (from project root):
    python dl/10_wider_deeper.py
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
from utils.fusion_wide import WideTaskGatedFusionMLP
from utils.data_loaders import load_multimodal_data
from utils.metrics import compute_metrics, print_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 2.0, 0.5])
TARGET_NAMES = ['valence', 'energy', 'danceability', 'popularity']
XGBOOST_BASELINE = {
    'valence': 0.6728, 'energy': 0.9073, 'danceability': 0.7693, 'popularity': 0.1478
}
EXP_C = {'valence': 0.6998, 'energy': 0.8894, 'danceability': 0.7306, 'popularity': 0.1133}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def unpack_batch(batch, device):
    *modalities, targets = batch
    return [m.to(device, non_blocking=True) for m in modalities], targets.to(device, non_blocking=True)


def train_epoch(model, loader, criterion, optimizer, scaler_amp, device, weights):
    model.train()
    total_loss = 0.0
    for batch in loader:
        modalities, targets = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds = model(*modalities)
            loss  = (criterion(preds, targets) * weights.to(device)).mean()
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, weights):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        modalities, targets = unpack_batch(batch, device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds = model(*modalities)
            loss  = (criterion(preds, targets) * weights.to(device)).mean()
        total_loss += loss.item()
        all_preds.append(preds.cpu().float())
        all_targets.append(targets.cpu().float())
    metrics = compute_metrics(torch.cat(all_targets), torch.cat(all_preds))
    return total_loss / len(loader), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment E: Wider/Deeper Model")
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--patience',       type=int,   default=35)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--dropout_enc',    type=float, default=0.3)
    parser.add_argument('--dropout_fusion', type=float, default=0.5)
    parser.add_argument('--feat_dir',       type=str,   default='ml/features')
    parser.add_argument('--scaler_dir',     type=str,   default='',
                        help='Path to global modal_scaler_*.pkl files. Empty = no scaling.')
    parser.add_argument('--num_workers',    type=int,   default=4)
    args = parser.parse_args()

    scaler_dir = args.scaler_dir if args.scaler_dir else None

    set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"exp_e_wider_deeper_{timestamp}"

    print("=" * 70)
    print("Experiment E: Wider / Deeper Model (ENCODED_DIM=256, 2 ResBlocks)")
    print("Phase 4E")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = load_multimodal_data(
        batch_size=args.batch_size, feat_dir=args.feat_dir,
        num_workers=args.num_workers, scaler_dir=scaler_dir,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = WideTaskGatedFusionMLP(
        num_targets=4,
        dropout_enc=args.dropout_enc,
        dropout_fusion=args.dropout_fusion,
    ).to(device)
    print(f"Model: {model.__class__.__name__}  |  Parameters: {model.count_parameters():,}")

    criterion  = nn.MSELoss(reduction='none')
    optimizer  = optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler_amp = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_path = Path('models/checkpoints') / f"{exp_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val R²':>8}  {'LR':>10}  Best")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 scaler_amp, device, LOSS_WEIGHTS)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, LOSS_WEIGHTS)
        scheduler.step()
        val_r2_avg = float(np.mean([m['r2'] for m in val_metrics.values()]))
        current_lr = optimizer.param_groups[0]['lr']

        is_best = val_loss < best_val_loss
        marker  = " ✓" if is_best else ""

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_r2_avg:>8.4f}  "
              f"{current_lr:>10.2e}{marker}")

        if is_best:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_metrics': val_metrics,
                        'args': vars(args), 'scaler_dir': scaler_dir},
                       ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    _, test_metrics = evaluate(model, test_loader, criterion, device, LOSS_WEIGHTS)
    print_metrics(test_metrics, "Test Metrics")

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir  = Path('results/dl_metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{exp_name}.csv"
    rows = []
    for target, values in test_metrics.items():
        rows.append({'timestamp': timestamp, 'experiment': 'E', 'model': 'WideTaskGatedFusionMLP',
                     'phase': '4E', 'split': 'test', 'target': target, **values})
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")
    print(f"✓ Checkpoint:    {ckpt_path}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Comparison: ExpC → ExpE → XGBoost")
    print("=" * 70)
    print(f"{'Target':<14} {'ExpC':>8}  {'ExpE':>8}  {'ΔvsC':>8}  {'XGB':>8}  {'Status'}")
    print("-" * 65)
    beaten = 0
    for t in TARGET_NAMES:
        r2  = test_metrics[t]['r2']
        d_c = r2 - EXP_C[t]
        sign = "+" if d_c >= 0 else ""
        if r2 > XGBOOST_BASELINE[t]:
            status = "✅ beats XGB"; beaten += 1
        elif r2 > EXP_C[t]:
            status = "↑ improved"
        else:
            status = "↓ regressed"
        print(f"{t:<14} {EXP_C[t]:>8.4f}  {r2:>8.4f}  {sign}{d_c:>7.4f}  "
              f"{XGBOOST_BASELINE[t]:>8.4f}  {status}")
    avg_r2 = np.mean([test_metrics[t]['r2'] for t in TARGET_NAMES])
    print(f"\n  Avg R²: {avg_r2:.4f}  |  XGB avg: 0.6243  |  Exp C avg: 0.6332  "
          f"|  Beat XGB on {beaten}/4 targets")


if __name__ == '__main__':
    main()
