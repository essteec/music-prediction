"""
Experiment C: Task-Gated Fusion MLP (Phase 4)

Each prediction target gets its own gating network and fusion pathway.
This is the key architectural hypothesis:

  Valence    → should up-weight MPNet (lyrics carry emotion)
  Energy     → should up-weight MERT, PANNs (acoustics carry energy)
  Danceability → should up-weight PANNs, MERT (rhythm from audio)
  Popularity → should up-weight metadata (artist, genre, BPM)

A single global gate (Experiment B) can't express these simultaneously.
Per-target gates allow the model to discover different modality preferences
for different prediction tasks without any hard-coded priors.

Hypothesis:
  "Do different targets benefit from different modality mixtures?"

Expected result vs Experiment B:
  - Larger improvements on targets where modality mismatch was worst
  - Gate analysis should show clearly different weight distributions per target
  - Average R² ≥ 0.49 (matching or beating ML baseline average)

Usage (from project root):
    python dl/08_task_gated_multimodal.py
    python dl/08_task_gated_multimodal.py --epochs 150 --batch_size 512
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
from utils.fusion import TaskGatedFusionMLP, MODALITY_DIMS
from utils.data_loaders import load_multimodal_data
from utils.metrics import compute_metrics, print_metrics


LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 2.0, 0.5])
TARGET_NAMES = ['valence', 'energy', 'danceability', 'popularity']
MODALITY_NAMES = list(MODALITY_DIMS.keys())


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def unpack_batch(batch, device):
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
            loss = (criterion(preds, targets) * weights.to(device)).mean()

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
            loss = (criterion(preds, targets) * weights.to(device)).mean()
        total_loss += loss.item()
        all_preds.append(preds.cpu().float())
        all_targets.append(targets.cpu().float())
    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return total_loss / len(loader), compute_metrics(targets_cat, preds_cat)


@torch.no_grad()
def inspect_task_gates(model, loader, device, n_batches: int = 20):
    """
    For each of the 4 targets, print the learned average modality weights.
    This is the most interpretable output of this experiment — it shows
    which modalities the model learned to rely on for each prediction task.
    """
    model.eval()
    # Accumulate per-target gate weights
    per_target_gates = [[] for _ in range(4)]

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        modalities, _ = unpack_batch(batch, device)

        encs = [model.encoders[k](v) for k, v in zip(MODALITY_DIMS.keys(), modalities)]
        presence = torch.stack(
            [(x.abs().sum(-1) > 0).float() for x in modalities], dim=-1
        )
        stacked = torch.stack(encs, dim=1)
        summary = stacked.mean(dim=1)

        for t_idx, gate_net in enumerate(model.task_gates):
            logits = gate_net(summary)
            logits = logits + (1.0 - presence) * (-1e9)
            gates  = torch.softmax(logits, dim=-1)  # (B, 6)
            per_target_gates[t_idx].append(gates.cpu())

    print("\n  Learned gate weights per target:")
    print(f"  {'Modality':<14}", end="")
    for t in TARGET_NAMES:
        print(f"  {t:>12}", end="")
    print()
    print("  " + "-" * (14 + 4 * 14))

    per_target_avg = [torch.cat(g).mean(dim=0).numpy() for g in per_target_gates]
    for m_idx, m_name in enumerate(MODALITY_NAMES):
        print(f"  {m_name:<14}", end="")
        for t_idx in range(4):
            w = per_target_avg[t_idx][m_idx]
            print(f"  {w:>12.4f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Experiment C: Task-Gated Fusion MLP")
    parser.add_argument('--epochs',         type=int,   default=150)
    parser.add_argument('--patience',       type=int,   default=25)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--dropout_enc',    type=float, default=0.2)
    parser.add_argument('--dropout_fusion', type=float, default=0.4)
    parser.add_argument('--feat_dir',       type=str,   default='ml/features')
    parser.add_argument('--num_workers',    type=int,   default=4)
    args = parser.parse_args()

    set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"exp_c_task_gated_{timestamp}"

    print("=" * 70)
    print("Experiment C: Task-Gated Fusion MLP")
    print("Phase 4 — Per-target modality attention")
    print("=" * 70)

    train_loader, val_loader, test_loader = load_multimodal_data(
        batch_size=args.batch_size, feat_dir=args.feat_dir, num_workers=args.num_workers,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = TaskGatedFusionMLP(
        num_targets=4,
        dropout_enc=args.dropout_enc,
        dropout_fusion=args.dropout_fusion,
    ).to(device)
    print(f"Model: {model.__class__.__name__}  |  Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    ckpt_path = Path('models/checkpoints') / f"{exp_name}_best.pt"
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
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_metrics': val_metrics, 'args': vars(args)},
                       ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
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

    # ── Gate analysis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Task Gate Analysis — Modality preferences per target")
    print("(Higher weight = model trusts this modality more for this target)")
    print("=" * 70)
    inspect_task_gates(model, val_loader, device)

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir  = Path('results/dl_metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{exp_name}.csv"
    rows = []
    for target, values in test_metrics.items():
        rows.append({'timestamp': timestamp, 'experiment': 'C', 'model': 'TaskGatedFusionMLP',
                     'phase': '4', 'split': 'test', 'target': target, **values})
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Comparison to baselines")
    print("=" * 70)
    phase1b = {'valence': 0.3792, 'energy': 0.7539, 'danceability': 0.4978, 'popularity': 0.1311}
    ml_base = {'valence': 0.45,   'energy': 0.81,   'danceability': 0.55,   'popularity': 0.13}
    print(f"{'Target':<14} {'Phase1B':>8}  {'ExpC':>8}  {'Delta':>8}  {'ML Base':>8}  {'Status'}")
    print("-" * 68)
    beaten = 0
    for t in TARGET_NAMES:
        r2   = test_metrics[t]['r2']
        diff = r2 - phase1b[t]
        sign = "+" if diff >= 0 else ""
        if r2 > ml_base[t]:
            status = "✅ beats ML"
            beaten += 1
        elif r2 > phase1b[t]:
            status = "↑ improved"
        else:
            status = "↓ regressed"
        print(f"{t:<14} {phase1b[t]:>8.4f}  {r2:>8.4f}  {sign}{diff:>7.4f}  {ml_base[t]:>8.4f}  {status}")

    avg_r2 = float(np.mean([test_metrics[t]['r2'] for t in TARGET_NAMES]))
    print(f"\n  Average R²: {avg_r2:.4f}  |  ML baseline avg: 0.49  |  Beat ML on {beaten}/4 targets")


if __name__ == '__main__':
    main()
