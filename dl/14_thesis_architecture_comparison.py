"""
Thesis architecture comparison — DL vs ML.

Trains/evaluates a small set of interpretable DL architectures on the same
4254-feature input space used by the thesis ML baseline (thesis_ml_models.py).

Methodology:
  - Validation mode: train on train split, select by avg val R²,
    save checkpoints, write one comparison CSV.
  - Test mode: load selected checkpoints, evaluate on test split,
    write final test CSV. Does not change architecture/hyperparameter choices.

Usage:
    python dl/14_thesis_architecture_comparison.py --eval-split val
    python dl/14_thesis_architecture_comparison.py --eval-split test \\
        --checkpoint-dir models/checkpoints/thesis

Outputs (val):
    results/dl_metrics/thesis_architecture_comparison_val_<timestamp>.csv
    models/checkpoints/thesis/<Architecture>_best.pt

Outputs (test):
    results/dl_metrics/final_dl_test_<timestamp>.csv
"""

import os
import sys
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loaders import MultiModalDataset, set_worker_seed
from utils.metrics import compute_metrics, print_metrics
from utils.thesis_models import (
    FlatAllMLP, MultiModalFusionMLP, TaskGatedFusionMLP,
    AttentionTaskGatedFusionMLP, engineer_metadata,
    TARGET_NAMES, TOTAL_FLAT_DIM,
)
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 2.0, 0.5])  # valence, energy, danceability, popularity
SCALER_DIR = "ml/features/scalers"

ARCHITECTURES = [
    {
        "name": "FlatAllMLP",
        "model_cls": FlatAllMLP,
        "model_kwargs": {"input_dim": TOTAL_FLAT_DIM, "dropout": 0.5},
        "feat_eng": False,
        "complexity": "low",
    },
    {
        "name": "MultiModalFusionMLP",
        "model_cls": MultiModalFusionMLP,
        "model_kwargs": {"dropout_enc": 0.2, "dropout_fusion": 0.4},
        "feat_eng": False,
        "complexity": "medium",
    },
    {
        "name": "TaskGatedFusionMLP",
        "model_cls": TaskGatedFusionMLP,
        "model_kwargs": {"dropout_enc": 0.2, "dropout_fusion": 0.4, "metadata_dim": 30},
        "feat_eng": False,
        "complexity": "medium",
    },
    {
        "name": "AttentionTaskGatedFusionMLP",
        "model_cls": AttentionTaskGatedFusionMLP,
        "model_kwargs": {"dropout_enc": 0.2, "dropout_fusion": 0.4, "metadata_dim": 30},
        "feat_eng": False,
        "complexity": "high",
    },
    {
        "name": "TaskGatedFusionMLP_FeatEng",
        "model_cls": TaskGatedFusionMLP,
        "model_kwargs": {"dropout_enc": 0.2, "dropout_fusion": 0.4, "metadata_dim": 36},
        "feat_eng": True,
        "complexity": "high",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Engineered Dataset (for feat eng variant)
# ──────────────────────────────────────────────────────────────────────────────

class EngineeredMultiModalDataset(MultiModalDataset):
    """MultiModalDataset with metadata feature engineering applied at load time."""

    def __init__(self, split: str, feat_dir: str = "ml/features", scaler_dir: str = None):
        super().__init__(split, feat_dir, scaler_dir)
        old_dim = self.metadata.shape[1]
        self.metadata = engineer_metadata(self.metadata)
        print(f"    Metadata engineered: {old_dim}d -> {self.metadata.shape[1]}d")


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=g,
        persistent_workers=(num_workers > 0),
    )


def load_data(batch_size: int, feat_dir: str, num_workers: int, seed: int, scaler_dir: str = None):
    """Load regular multimodal data loaders for train/val/test."""
    print("\nLoading multi-modal datasets:")
    train_ds = MultiModalDataset("train", feat_dir, scaler_dir)
    val_ds = MultiModalDataset("val", feat_dir, scaler_dir)
    test_ds = MultiModalDataset("test", feat_dir, scaler_dir)
    train_loader = _build_loader(train_ds, batch_size, True, num_workers, seed)
    val_loader = _build_loader(val_ds, batch_size, False, num_workers, seed)
    test_loader = _build_loader(test_ds, batch_size, False, num_workers, seed)
    print(f"\n  Train: {len(train_loader):,} batches  Val: {len(val_loader):,}  Test: {len(test_loader):,}")
    return train_loader, val_loader, test_loader


def load_engineered_data(batch_size: int, feat_dir: str, num_workers: int, seed: int, scaler_dir: str = None):
    """Load data with engineered metadata for feat eng variant."""
    print("\nLoading multi-modal datasets (engineered metadata):")
    train_ds = EngineeredMultiModalDataset("train", feat_dir, scaler_dir)
    val_ds = EngineeredMultiModalDataset("val", feat_dir, scaler_dir)
    test_ds = EngineeredMultiModalDataset("test", feat_dir, scaler_dir)
    train_loader = _build_loader(train_ds, batch_size, True, num_workers, seed)
    val_loader = _build_loader(val_ds, batch_size, False, num_workers, seed)
    test_loader = _build_loader(test_ds, batch_size, False, num_workers, seed)
    print(f"\n  Train: {len(train_loader):,} batches  Val: {len(val_loader):,}  Test: {len(test_loader):,}")
    return train_loader, val_loader, test_loader

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ──────────────────────────────────────────────────────────────────────────────
# Training / Evaluation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def unpack(batch, device):
    """Unpack a MultiModalDataset batch into list of modality tensors + targets."""
    *mods, targets = batch
    return [m.to(device, non_blocking=True) for m in mods], targets.to(device, non_blocking=True)


def train_epoch(model, loader, criterion, optimizer, scaler, device, weights):
    model.train()
    total = 0.0
    for batch in loader:
        mods, tgts = unpack(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss = (criterion(model(*mods), tgts) * weights).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, weights):
    model.eval()
    total = 0.0
    preds_l, tgts_l = [], []
    for batch in loader:
        mods, tgts = unpack(batch, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p = model(*mods)
            total += (criterion(p, tgts) * weights).mean().item()
        preds_l.append(p.cpu().float())
        tgts_l.append(tgts.cpu().float())
    metrics = compute_metrics(torch.cat(tgts_l), torch.cat(preds_l))
    return total / len(loader), metrics


def train_model(model, train_loader, val_loader, arch, args, device):
    """
    Train a single architecture, track best val R², save checkpoint.

    Returns:
        (best_val_r2, best_metrics, best_epoch, checkpoint_path)
    """
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    weights = LOSS_WEIGHTS.to(device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{arch['name']}_best.pt"

    best_val_r2 = -float("inf")
    best_metrics = None
    best_epoch = 0
    patience_count = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val R²':>8}  Best")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, amp_scaler, device, weights)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, weights)
        scheduler.step()
        val_r2_avg = float(np.mean([m["r2"] for m in val_metrics.values()]))

        is_best = val_r2_avg > best_val_r2
        marker = " ✓" if is_best else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_r2_avg:>8.4f}{marker}")

        if is_best:
            best_val_r2 = val_r2_avg
            best_metrics = val_metrics
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_r2": val_r2_avg,
                "val_metrics": val_metrics,
                "arch": arch["name"],
                "feat_eng": arch["feat_eng"],
                "args": vars(args),
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    return best_val_r2, best_metrics, best_epoch, ckpt_path

# ──────────────────────────────────────────────────────────────────────────────
# Result Writing
# ──────────────────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_result_row(arch, target, metrics, epoch, selection_metric, timestamp, split, notes=""):
    return {
        "timestamp": timestamp,
        "split": split,
        "experiment": "thesis_dl",
        "model": arch["name"],
        "target": target,
        "r2": metrics[target]["r2"],
        "rmse": metrics[target]["rmse"],
        "mae": metrics[target]["mae"],
        "epoch": epoch,
        "selection_metric": selection_metric,
        "notes": notes,
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Thesis DL Architecture Comparison")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                        help="Evaluation split. Val trains on train; test loads checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints/thesis",
                        help="Directory for saving/loading model checkpoints.")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--feat-dir", type=str, default="ml/features", help="Feature directory.")
    parser.add_argument("--results-root", type=str, default="results/dl_metrics",
                        help="Root directory for metric CSV outputs.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-scaler", action="store_true",
                        help="Disable per-modality scaling (not recommended).")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_dir = None if args.no_scaler else SCALER_DIR

    print("=" * 70)
    print("THESIS DL ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"Eval split:         {args.eval_split}")
    print(f"Device:             {device}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Learning rate:      {args.lr}")
    print(f"Weight decay:       {args.weight_decay}")
    print(f"Max epochs:         {args.epochs}")
    print(f"Patience:           {args.patience}")
    print(f"Per-modality scale: {scaler_dir is not None}")
    print(f"Checkpoint dir:     {args.checkpoint_dir}")

    needs_feat_eng = any(a["feat_eng"] for a in ARCHITECTURES)

    if args.eval_split == "val":
        results_dir = Path(args.results_root) / "thesis_val"
        results_path = results_dir / f"thesis_architecture_comparison_val_{timestamp}.csv"
        results_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, _ = load_data(
            args.batch_size, args.feat_dir, args.num_workers, args.seed, scaler_dir
        )
        if needs_feat_eng:
            train_eng, val_eng, _ = load_engineered_data(
                args.batch_size, args.feat_dir, args.num_workers, args.seed, scaler_dir
            )

        rows = []
        for arch in ARCHITECTURES:
            print("\n" + "=" * 70)
            print(f"ARCHITECTURE: {arch['name']}  (complexity={arch['complexity']})")
            print("=" * 70)

            use_eng = arch["feat_eng"]
            tr = train_eng if use_eng else train_loader
            vl = val_eng if use_eng else val_loader

            model = arch["model_cls"](**arch["model_kwargs"]).to(device)
            print(f"Parameters: {model.count_parameters():,}")

            best_val_r2, best_metrics, best_epoch, ckpt_path = train_model(
                model, tr, vl, arch, args, device
            )

            notes = "feat_eng" if arch["feat_eng"] else ""
            selection_metric = f"val_avg_r2={best_val_r2:.4f}"

            for target in TARGET_NAMES:
                rows.append(build_result_row(
                    arch, target, best_metrics, best_epoch,
                    selection_metric, timestamp, "val", notes
                ))

            print(f"\n  Best val R²: {best_val_r2:.4f} at epoch {best_epoch}")
            print(f"  Checkpoint: {ckpt_path}")

        save_results(rows, results_path)
        print(f"\n✓ Results saved to: {results_path}")

    else:
        results_dir = Path(args.results_root)
        results_path = results_dir / f"final_dl_test_{timestamp}.csv"
        results_dir.mkdir(parents=True, exist_ok=True)

        _, _, test_loader = load_data(
            args.batch_size, args.feat_dir, args.num_workers, args.seed, scaler_dir
        )
        if needs_feat_eng:
            _, _, test_eng = load_engineered_data(
                args.batch_size, args.feat_dir, args.num_workers, args.seed, scaler_dir
            )

        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"\nError: checkpoint directory '{checkpoint_dir}' not found.")
            print("Run validation mode first to train and save checkpoints.")
            sys.exit(1)

        weights = LOSS_WEIGHTS.to(device)
        rows = []
        for arch in ARCHITECTURES:
            ckpt_path = checkpoint_dir / f"{arch['name']}_best.pt"
            if not ckpt_path.exists():
                print(f"\n  Warning: checkpoint not found for {arch['name']} — skipping.")
                continue

            print("\n" + "=" * 70)
            print(f"ARCHITECTURE: {arch['name']}  (test evaluation)")
            print("=" * 70)

            use_eng = arch["feat_eng"]
            tl = test_eng if use_eng else test_loader

            model = arch["model_cls"](**arch["model_kwargs"]).to(device)
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            loaded_epoch = ckpt.get("epoch", "?")
            loaded_r2 = ckpt.get("val_r2", "?")
            print(f"  Loaded checkpoint: epoch={loaded_epoch}, val_r2={loaded_r2}")

            criterion = nn.MSELoss(reduction="none")
            _, test_metrics = evaluate(model, tl, criterion, device, weights)
            print_metrics(test_metrics, f"Test — {arch['name']}")

            test_avg_r2 = float(np.mean([m["r2"] for m in test_metrics.values()]))
            notes = "feat_eng" if arch["feat_eng"] else ""
            selection_metric = f"test_avg_r2={test_avg_r2:.4f}"

            for target in TARGET_NAMES:
                rows.append(build_result_row(
                    arch, target, test_metrics, loaded_epoch,
                    selection_metric, timestamp, "test", notes
                ))

        if rows:
            save_results(rows, results_path)
            print(f"\n✓ Final test results saved to: {results_path}")
        else:
            print("\nNo results to save — all architectures were skipped.")


if __name__ == "__main__":
    main()
