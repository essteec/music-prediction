"""
Thesis ML Models - Same Inputs as DL Setup

Runs a compact, thesis-ready classical ML comparison on the same 4254-feature
input space used by the multimodal DL experiments:

- Base audio / metadata: 23
- Text stats: 5
- Sentiment: 2
- MPNet lyric embeddings: 768
- VGGish audio embeddings: 128
- MERT audio embeddings: 768
- PANNs audio embeddings: 2048
- Mel Stats audio embeddings: 512

Methodology:
- `--eval-split val`: train on train, evaluate on val for model-family selection.
- `--eval-split test`: train on train+val, evaluate on test for final thesis reporting.
- No hand-written "tuned" variants. Hyperparameter optimization should be a separate stage.
- Result files are split-explicit and include training/prediction timings.

Usage:
    python ml/models/thesis_ml_models.py --eval-split val
    python ml/models/thesis_ml_models.py --eval-split test
    python ml/models/thesis_ml_models.py --eval-split val --models Mean Ridge XGBoost
    python ml/models/thesis_ml_models.py --eval-split val --no-save-models
"""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


warnings.filterwarnings("ignore")


TARGETS = ["valence", "energy", "danceability", "popularity"]
FEATURE_PARTS = [
    ("audio", "Base Audio", 23),
    ("text_stats", "Text Stats", 5),
    ("sentiment", "Sentiment", 2),
    ("mpnet", "MPNet", 768),
    ("vggish", "VGGish", 128),
    ("mert", "MERT", 768),
    ("panns", "PANNs", 2048),
    ("mel_stats", "Mel Stats", 512),
]
FEATURE_SET_NAME = "ultimate_4254"
N_FEATURES = sum(dim for _, _, dim in FEATURE_PARTS)
DEFAULT_MODELS = [
    "Mean",
    "Ridge",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLPRegressor",
    "ExtraTrees",
    "RandomForest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thesis ML model comparison on DL-equivalent features")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split. Val trains on train; test trains on train+val.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Skip saving trained model artifacts.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/metrics",
        help="Root directory for metric CSV outputs.",
    )
    parser.add_argument(
        "--mlp-max-samples",
        type=int,
        default=120000,
        help=(
            "Max number of training samples for MLPRegressor. "
            "Use 0 to disable capping. Default keeps memory usage manageable."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and model reproducibility.",
    )
    return parser.parse_args()


def load_feature_matrix(features_dir: Path, split: str) -> np.ndarray:
    """Load all feature groups for one split and concatenate to 4254 dims."""
    arrays = []
    print(f"\nLoading {split} features...")
    for key, label, expected_dim in FEATURE_PARTS:
        path = features_dir / f"X_{split}_{key}.npy"
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be 2D, got shape {arr.shape}")
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected {label} dimension for {path}: "
                f"got {arr.shape[1]}, expected {expected_dim}"
            )
        arrays.append(arr)
        print(f"  {label:<12}: {arr.shape}")

    X = np.hstack(arrays).astype(np.float32, copy=False)
    del arrays
    gc.collect()
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {X.shape[1]}")
    return X


def load_targets(features_dir: Path, split: str, target: str) -> np.ndarray:
    return np.load(features_dir / f"y_{split}_{target}.npy").astype(np.float32)


def build_training_data(features_dir: Path, eval_split: str) -> tuple[np.ndarray, str]:
    """Return X matrix used for fitting and a label describing its source split."""
    X_train = load_feature_matrix(features_dir, "train")
    if eval_split == "val":
        return X_train, "train"

    X_val = load_feature_matrix(features_dir, "val")
    X_train_val = np.vstack([X_train, X_val]).astype(np.float32, copy=False)
    del X_train, X_val
    gc.collect()
    return X_train_val, "train+val"


def build_training_target(features_dir: Path, eval_split: str, target: str) -> np.ndarray:
    y_train = load_targets(features_dir, "train", target)
    if eval_split == "val":
        return y_train
    y_val = load_targets(features_dir, "val", target)
    return np.concatenate([y_train, y_val]).astype(np.float32, copy=False)


def get_model(model_name: str, seed: int):
    """Return a single default/simple model instance.

    Memory-first defaults:
    - Ridge uses SAG solver directly on raw features to avoid StandardScaler copies.
    - MLP uses smaller architecture and adaptive LR; training set is optionally capped.
    """
    if model_name == "Ridge":
        return Ridge(alpha=1.0, solver="sag", random_state=seed)
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    if model_name == "LightGBM":
        return lgb.LGBMRegressor(
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "CatBoost":
        return CatBoostRegressor(
            random_state=seed,
            verbose=False,
            thread_count=-1,
        )
    if model_name == "MLPRegressor":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            random_state=seed,
            early_stopping=True,
            max_iter=120,
            batch_size=2048,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            verbose=False,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1)
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown trainable model: {model_name}")


def maybe_subsample_for_model(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    model_name: str,
    mlp_max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Optionally subsample training data for memory-heavy models.

    Returns sampled X, y, and number of rows used.
    """
    n = len(y_fit)
    if model_name != "MLPRegressor" or mlp_max_samples <= 0 or n <= mlp_max_samples:
        return X_fit, y_fit, n

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=mlp_max_samples, replace=False)
    idx.sort()
    return X_fit[idx], y_fit[idx], mlp_max_samples


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_results(rows: list[dict], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_path, index=False)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = repo_root / "ml" / "features"
    results_root = repo_root / args.results_root
    results_dir = results_root / f"thesis_ml_{args.eval_split}"
    models_dir = repo_root / "ml" / "models" / "saved" / f"thesis_ml_{args.eval_split}"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"thesis_ml_results_{args.eval_split}_{timestamp}.csv"
    summary_path = results_dir / f"thesis_ml_model_summary_{args.eval_split}_{timestamp}.csv"

    print("=" * 80)
    print("THESIS ML MODELS - DL-EQUIVALENT INPUTS")
    print("=" * 80)
    print(f"Eval split:     {args.eval_split}")
    print(f"Models:         {', '.join(args.models)}")
    print(f"Feature set:    {FEATURE_SET_NAME} ({N_FEATURES} features)")
    print(f"Save models:    {not args.no_save_models}")
    print(f"MLP max samples:{args.mlp_max_samples}")

    X_fit, train_split_label = build_training_data(features_dir, args.eval_split)
    X_eval = load_feature_matrix(features_dir, args.eval_split)

    print("\n" + "=" * 80)
    print("MATRIX SUMMARY")
    print("=" * 80)
    print(f"Train split used: {train_split_label}")
    print(f"X_fit:           {X_fit.shape}")
    print(f"X_eval:          {X_eval.shape}")

    rows = []
    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"TARGET: {target.upper()}")
        print("=" * 80)

        y_fit = build_training_target(features_dir, args.eval_split, target)
        y_eval = load_targets(features_dir, args.eval_split, target)
        print(f"y_fit:  {y_fit.shape}  mean={y_fit.mean():.4f}  std={y_fit.std():.4f}")
        print(f"y_eval: {y_eval.shape}  mean={y_eval.mean():.4f}  std={y_eval.std():.4f}")

        for model_name in args.models:
            print(f"\nTraining/evaluating {model_name} for {target}...")
            start_total = time.perf_counter()
            train_time = 0.0

            if model_name == "Mean":
                y_pred = np.full_like(y_eval, y_fit.mean(), dtype=np.float32)
                model = None
                n_train_used = int(len(y_fit))
            else:
                model = get_model(model_name, args.seed)
                X_fit_used, y_fit_used, n_train_used = maybe_subsample_for_model(
                    X_fit, y_fit, model_name, args.mlp_max_samples, args.seed
                )
                start_train = time.perf_counter()
                model.fit(X_fit_used, y_fit_used)
                train_time = time.perf_counter() - start_train
                del X_fit_used, y_fit_used
                start_predict = time.perf_counter()
                y_pred = model.predict(X_eval)
                predict_time = time.perf_counter() - start_predict
            if model_name == "Mean":
                predict_time = time.perf_counter() - start_total

            total_time = time.perf_counter() - start_total
            metrics = evaluate_predictions(y_eval, y_pred)
            row = {
                "timestamp": timestamp,
                "split": args.eval_split,
                "train_split": train_split_label,
                "target": target,
                "model": model_name,
                "features": FEATURE_SET_NAME,
                "n_features": N_FEATURES,
                "n_train": int(len(y_fit)),
                "n_train_used": int(n_train_used),
                "n_eval": int(len(y_eval)),
                **metrics,
                "train_time_seconds": float(train_time),
                "predict_time_seconds": float(predict_time),
                "total_time_seconds": float(total_time),
            }
            rows.append(row)

            if model is not None and not args.no_save_models:
                model_path = models_dir / f"{model_name}_{target}.pkl"
                joblib.dump(model, model_path)
                row["model_path"] = str(model_path.relative_to(repo_root))
            else:
                row["model_path"] = ""

            save_results(rows, results_path)
            print(
                f"  R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
                f"MAE={metrics['mae']:.4f}  train={train_time:.1f}s  total={total_time:.1f}s"
            )

    results_df = pd.DataFrame(rows)
    save_results(rows, results_path)

    print("\n" + "=" * 80)
    print("BEST MODEL BY TARGET")
    print("=" * 80)
    for target in TARGETS:
        target_df = results_df[results_df["target"] == target]
        best = target_df.loc[target_df["r2"].idxmax()]
        print(f"{target:<14} {best['model']:<14} R2={best['r2']:.4f}")

    print("\n" + "=" * 80)
    print("AVERAGE R2 BY MODEL")
    print("=" * 80)
    avg = results_df.groupby("model", as_index=False).agg(
        r2=("r2", "mean"),
        total_time_seconds=("total_time_seconds", "sum"),
        train_time_seconds=("train_time_seconds", "sum"),
        predict_time_seconds=("predict_time_seconds", "sum"),
    ).sort_values("r2", ascending=False)
    avg.to_csv(summary_path, index=False)
    print(avg.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Model summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
