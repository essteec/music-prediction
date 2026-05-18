"""
Ultimate Feature Models - ALL Features Combined

Trains classical models with every available feature and evaluates them on an
explicit split. This script is intentionally split-aware because older runs
wrote validation results under `ultimate_test`, which is not thesis-safe.

Features:
- Audio/base metadata: 23
- Text stats: 5
- Sentiment: 2
- MPNet text embeddings: 768
- VGGish audio embeddings: 128
- PANNs audio embeddings: 2048
- Mel Stats audio embeddings: 512
- MERT audio embeddings: 768

Total: 4254 features

Usage:
    python ml/models/ultimate_models.py --eval-split val
    python ml/models/ultimate_models.py --eval-split test
"""

import argparse
import gc
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["valence", "energy", "danceability", "popularity"]
FEATURE_PARTS = [
    ("audio", "Base Audio", 23),
    ("text_stats", "Text Stats", 5),
    ("sentiment", "Sentiment", 2),
    ("mpnet", "MPNet", 768),
    ("vggish", "VGGish", 128),
    ("panns", "PANNs", 2048),
    ("mel_stats", "Mel Stats", 512),
    ("mert", "MERT", 768),
]


def load_feature_matrix(features_dir: Path, split: str) -> np.ndarray:
    """Load and concatenate all Ultimate feature groups for one split."""
    arrays = []
    print(f"\nLoading {split} features...")
    for key, label, expected_dim in FEATURE_PARTS:
        path = features_dir / f"X_{split}_{key}.npy"
        arr = np.load(path).astype(np.float32)
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected {label} dimension for {path}: "
                f"got {arr.shape[1]}, expected {expected_dim}"
            )
        arrays.append(arr)
        print(f"  {label:<12}: {arr.shape}")

    print("Combining feature groups...")
    matrix = np.hstack(arrays)
    del arrays
    gc.collect()
    return matrix


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultimate ML models on all feature groups")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split. Use val for selection and test only for final thesis reporting.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    features_dir = repo_root / "ml" / "features"
    models_dir = repo_root / "ml" / "models" / "saved" / f"ultimate_{args.eval_split}"
    results_dir = repo_root / "results" / "metrics" / f"ultimate_{args.eval_split}"
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(f"ULTIMATE MODELS - ALL FEATURES COMBINED (eval_split={args.eval_split})")
    print("=" * 80)

    X_train = load_feature_matrix(features_dir, "train")
    X_eval = load_feature_matrix(features_dir, args.eval_split)

    print("\n" + "=" * 80)
    print("COMBINED FEATURE MATRIX - ULTIMATE")
    print("=" * 80)
    print(f"Train: {X_train.shape}")
    print(f"Eval ({args.eval_split}): {X_eval.shape}")
    print("Feature breakdown:")
    for _, label, dim in FEATURE_PARTS:
        print(f"  - {label:<12}: {dim} features")
    print("  - TOTAL       : 4254 features")
    print("=" * 80)

    all_results = []

    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"TARGET: {target.upper()}")
        print("=" * 80)

        y_train = np.load(features_dir / f"y_train_{target}.npy")
        y_eval = np.load(features_dir / f"y_{args.eval_split}_{target}.npy")

        y_pred_mean = np.full_like(y_eval, y_train.mean())
        all_results.append({
            "split": args.eval_split,
            "target": target,
            "model": "Mean",
            "features": "ultimate",
            **evaluate_predictions(y_eval, y_pred_mean),
        })

        print("Training Ridge Regression (SAG solver)...")
        ridge = Ridge(alpha=1.0, solver="sag", random_state=42)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_eval)
        ridge_metrics = evaluate_predictions(y_eval, y_pred_ridge)
        all_results.append({
            "split": args.eval_split,
            "target": target,
            "model": "Ridge",
            "features": "ultimate",
            **ridge_metrics,
        })
        joblib.dump(ridge, models_dir / f"ridge_ultimate_{target}.pkl")

        print("Training XGBoost (tree_method='hist' for memory efficiency)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_eval)
        xgb_metrics = evaluate_predictions(y_eval, y_pred_xgb)
        all_results.append({
            "split": args.eval_split,
            "target": target,
            "model": "XGBoost",
            "features": "ultimate",
            **xgb_metrics,
        })
        joblib.dump(xgb_model, models_dir / f"xgboost_ultimate_{target}.pkl")

        print(f"\nResults for {target} ({args.eval_split}):")
        print(f"  Ridge: RMSE={ridge_metrics['rmse']:.4f}, R2={ridge_metrics['r2']:.4f}")
        print(f"  XGB:   RMSE={xgb_metrics['rmse']:.4f}, R2={xgb_metrics['r2']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"ultimate_results_{args.eval_split}_{timestamp}.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
