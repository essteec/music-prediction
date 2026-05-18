"""
Experiment A: Multi-Branch Fusion MLP (Phase 4)

Baseline multi-modal architecture. Each modality gets its own encoder that
compresses it to 128d BEFORE fusion. This is the hypothesis test for:

  "Does per-modality compression alone eliminate the audio degradation seen
   when feeding raw 4,254-d flat vectors to a generic MLP?"

Expected result:
  - Audio no longer hurts any target vs Phase 1B (0.4405 avg R²)
  - Energy R² improves from 0.7539 → ≥ 0.78
  - Average R² ≥ 0.46

Usage (from project root):
    python dl/06_multimodal_fusion.py
    python dl/06_multimodal_fusion.py --epochs 150 --batch_size 512
    python dl/06_multimodal_fusion.py --feat_dir ml/features --num_workers 2
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
from utils.fusion import MultiModalFusionMLP
from utils.data_loaders import load_multimodal_data
from utils.metrics import compute_metrics, print_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 2.0, 0.5])  # [valence, energy, dance, pop]
TARGET_NAMES = ['valence', 'energy', 'danceability', 'popularity']


def unpack_batch(batch, device):
    """Unpack a multi-modal batch and send everything to device."""
    *modalities, targets = batch
    modalities = [m.to(device, non_blocking=True) for m in modalities]
    targets = targets.to(device, non_blocking=True)
    return modalities, targets


def train_epoch(model, loader, criterion, optimizer, scaler, device, weights):
    model.train()
    total_loss = 0.0
    for batch in loader:
        modalities, targets = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            preds = model(*modalities)
            loss = criterion(preds, targets)              # (B, 4)
            loss = (loss * weights.to(device)).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, weights):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        modalities, targets = unpack_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            preds = model(*modalities)
            loss = criterion(preds, targets)
            loss = (loss * weights.to(device)).mean()
        total_loss += loss.item()
        all_preds.append(preds.cpu().float())
        all_targets.append(targets.cpu().float())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = compute_metrics(targets_cat, preds_cat)
    return total_loss / len(loader), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment A: Multi-Branch Fusion MLP")
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--patience',     type=int,   default=25)
    parser.add_argument('--batch_size',   type=int,   default=512)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout_enc',  type=float, default=0.2)
    parser.add_argument('--dropout_fusion', type=float, default=0.4)
    parser.add_argument('--feat_dir',     type=str,   default='ml/features')
    parser.add_argument('--num_workers',  type=int,   default=4)
    args = parser.parse_args()

    set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"exp_a_multimodal_fusion_{timestamp}"

    print("=" * 70)
    print("Experiment A: Multi-Branch Fusion MLP")
    print("Phase 4 — Fixing the audio degradation problem")
    print("=" * 70)
    print(f"\nConfig:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = load_multimodal_data(
        batch_size=args.batch_size,
        feat_dir=args.feat_dir,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = MultiModalFusionMLP(
        num_targets=4,
        dropout_enc=args.dropout_enc,
        dropout_fusion=args.dropout_fusion,
    ).to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_path  = Path('models/checkpoints') / f"{exp_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val R²':>8}  {'Best':>5}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, LOSS_WEIGHTS)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, LOSS_WEIGHTS)
        val_r2_avg = float(np.mean([m['r2'] for m in val_metrics.values()]))

        is_best = val_loss < best_val_loss
        marker  = " ✓" if is_best else ""

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_r2_avg:>8.4f}{marker}")

        if is_best:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, ckpt_path)
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
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    _, test_metrics = evaluate(model, test_loader, criterion, device, LOSS_WEIGHTS)
    print_metrics(test_metrics, "Test Metrics")

    # ── Results to CSV ────────────────────────────────────────────────────────
    results_dir = Path('results/dl_metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{exp_name}.csv"

    rows = []
    for target, values in test_metrics.items():
        rows.append({'timestamp': timestamp, 'experiment': 'A', 'model': 'MultiModalFusionMLP',
                     'phase': '4', 'split': 'test', 'target': target, **values})
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Checkpoint saved:  {ckpt_path}")

    # Comparison summary
    print("\n" + "=" * 70)
    print("Comparison to Phase 1B baseline (MPNet flat MLP)")
    print("=" * 70)
    baseline = {'valence': 0.3792, 'energy': 0.7539, 'danceability': 0.4978, 'popularity': 0.1311}
    ml_base  = {'valence': 0.45,   'energy': 0.81,   'danceability': 0.55,   'popularity': 0.13  }
    print(f"{'Target':<14} {'Phase1B':>8}  {'ExpA':>8}  {'Delta':>8}  {'ML Base':>8}")
    print("-" * 55)
    for t in TARGET_NAMES:
        r2   = test_metrics[t]['r2']
        diff = r2 - baseline[t]
        sign = "+" if diff >= 0 else ""
        print(f"{t:<14} {baseline[t]:>8.4f}  {r2:>8.4f}  {sign}{diff:>7.4f}  {ml_base[t]:>8.4f}")


if __name__ == '__main__':
    main()
