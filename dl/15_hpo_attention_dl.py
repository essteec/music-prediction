"""
AttentionTaskGatedFusionMLP HPO — Focused validation-only hyperparameter optimization.

Optimizes average validation R² across all 4 targets.
Uses Optuna with pruning. Default: 20 trials.

Usage:
    python dl/15_hpo_attention_dl.py
    python dl/15_hpo_attention_dl.py --trials 15 --epochs 100
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loaders import MultiModalDataset
from utils.metrics import compute_metrics
from utils.thesis_models import AttentionTaskGatedFusionMLP

LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 2.0, 0.5])
SCALER_DIR = "ml/features/scalers"


def parse_args():
    parser = argparse.ArgumentParser(description="Attention DL HPO on validation")
    parser.add_argument("--trials", type=int, default=20, help="Number of HPO trials")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--feat-dir", type=str, default="ml/features", help="Feature directory")
    parser.add_argument("--results-root", type=str, default="results/hpo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-scaler", action="store_true", help="Disable per-modality scaling")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_worker_seed_hpo(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, shuffle, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_worker_seed_hpo,
        generator=g,
        persistent_workers=(num_workers > 0),
    )


def unpack(batch, device):
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


def load_datasets(feat_dir, scaler_dir):
    """Load multi-modal datasets once, cache across trials."""
    print("\nLoading multi-modal datasets (once)...")
    train_ds = MultiModalDataset("train", feat_dir, scaler_dir)
    val_ds = MultiModalDataset("val", feat_dir, scaler_dir)
    print(f"\n  Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")
    return train_ds, val_ds


def _to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_native(v) for v in obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def objective(trial, args, device, train_ds, val_ds):
    lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    dropout_enc = trial.suggest_float("dropout_enc", 0.1, 0.4)
    dropout_fusion = trial.suggest_float("dropout_fusion", 0.2, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])

    model = AttentionTaskGatedFusionMLP(
        dropout_enc=dropout_enc,
        dropout_fusion=dropout_fusion,
    ).to(device)

    # Build loaders from cached datasets (no disk I/O)
    trial_train_loader = build_loader(train_ds, batch_size, True, args.num_workers, args.seed)
    trial_val_loader = build_loader(val_ds, batch_size, False, args.num_workers, args.seed)

    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    weights = LOSS_WEIGHTS.to(device)

    best_val_r2 = -float("inf")
    best_val_metrics = None
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, trial_train_loader, criterion, optimizer, scaler, device, weights)
        val_loss, val_metrics = evaluate(model, trial_val_loader, criterion, device, weights)
        scheduler.step()
        val_r2_avg = float(np.mean([m["r2"] for m in val_metrics.values()]))
        trial.report(val_r2_avg, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_r2_avg > best_val_r2:
            best_val_r2 = val_r2_avg
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_val_metrics", _to_native(best_val_metrics))
    trial.set_user_attr("best_val_r2", float(best_val_r2))

    return best_val_r2


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_dir = None if args.no_scaler else SCALER_DIR

    num_workers = min(4, os.cpu_count() or 1)
    args.num_workers = num_workers

    print("=" * 70)
    print("ATTENTION DL HPO — FOCUSED VALIDATION OPTIMIZATION")
    print("=" * 70)
    print(f"Device:            {device}")
    print(f"Trials:            {args.trials}")
    print(f"Max epochs/trial:  {args.epochs}")
    print(f"Patience:          {args.patience}")

    # Store scaler_dir in args so objective can access it
    args.scaler_dir = scaler_dir

    # Load datasets once — cached across all trials (Bug #3 fix)
    train_ds, val_ds = load_datasets(args.feat_dir, scaler_dir)

    study = optuna.create_study(
        direction="maximize",
        study_name="attention_dl_hpo",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        ),
    )

    print(f"\nRunning {args.trials} trials...")

    def objective_wrapper(trial):
        return objective(trial, args, device, train_ds, val_ds)

    study.optimize(objective_wrapper, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest average val R²: {study.best_value:.4f}")
    print(f"Best params:         {study.best_params}")
    best_trial = study.best_trial
    best_epoch = int(best_trial.user_attrs.get("best_epoch", args.epochs))
    best_metrics = best_trial.user_attrs.get("best_val_metrics", {})
    print(f"Best epoch:          {best_epoch}")

    # Save trial history
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trials_df = study.trials_dataframe()
    trials_path = results_dir / f"attention_dl_hpo_val_{timestamp}.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Trial history: {trials_path}")

    # Save best params
    best_path = results_dir / "attention_dl_best_params.json"
    with open(best_path, "w") as f:
        json.dump(_to_native({
            "timestamp": timestamp,
            "model_name": "AttentionTaskGatedFusionMLP",
            "best_value": float(study.best_value),
            "n_trials": args.trials,
            "best_params": dict(study.best_params),
            "best_epoch": best_epoch,
            "best_trial_number": int(best_trial.number),
            "best_val_metrics": best_metrics,
            "train_protocol": {
                "loss_weights": LOSS_WEIGHTS.tolist(),
                "max_epochs_per_trial": args.epochs,
                "early_stopping_patience": args.patience,
                "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)",
                "objective": "average validation R2 across 4 targets",
                "scaler_enabled": not args.no_scaler,
            },
            "final_retrain_recommendation": (
                "For final test, retrain AttentionTaskGatedFusionMLP on train+val only "
                "using best_params and best_epoch, then evaluate once on test."
            ),
        }), f, indent=2)
    print(f"Best params saved to: {best_path}")

    # Pruning report
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nTrials completed: {n_completed}  Pruned: {n_pruned}")


if __name__ == "__main__":
    main()
