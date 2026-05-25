"""
CatBoost HPO — Focused validation-only hyperparameter optimization.

Reflects best practices from:
  - CatBoost official docs (border_count=254, bagging_temperature,
    use_best_model, depth 6-10)
  - Comprehensive HPO benchmarking paper (od_wait=300, random_strength,
    leaf_estimation_iterations, drop rsm)

Uses Optuna with TPE sampler and CatBoostPruningCallback.

Usage:
    python ml/models/hpo_catboost.py
    python ml/models/hpo_catboost.py --trials 15
    python ml/models/hpo_catboost.py --targets valence energy
"""

import argparse
import gc
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import r2_score

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


def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost HPO on validation")
    parser.add_argument("--trials", type=int, default=12, help="Trials per target")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS, help="Targets to tune")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-root", type=str, default="results/hpo", help="Output directory")
    return parser.parse_args()


def load_feature_matrix(features_dir: Path, split: str) -> np.ndarray:
    arrays = []
    print(f"\nLoading {split} features...")
    for key, label, expected_dim in FEATURE_PARTS:
        path = features_dir / f"X_{split}_{key}.npy"
        arr = np.load(path).astype(np.float32)
        if arr.shape[1] != expected_dim:
            raise ValueError(f"Unexpected {label} dimension: got {arr.shape[1]}, expected {expected_dim}")
        arrays.append(arr)
        print(f"  {label:<12}: {arr.shape}")
    X = np.hstack(arrays).astype(np.float32, copy=False)
    del arrays
    gc.collect()
    return X


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective for CatBoost.

    Search space rationale:
      - learning_rate: wide log range from very small to moderate.
        Paper tuned default: ~0.09 for regression.
      - depth: 6-10 per official CatBoost docs (optimal range).
        Paper tuned default: 7 (classification), 9 (regression).
      - l2_leaf_reg: log-uniform. Paper tuned default: 1e-5.
      - random_strength: controls randomness of split selection.
        Paper search: int 1-20.
      - bagging_temperature: controls Bayesian bootstrap aggressiveness.
        0 = uniform weights, higher = more aggressive sampling.
        Alternative: set bootstrap_type="Bernoulli" + tune subsample instead.
      - leaf_estimation_iterations: gradient steps per leaf.
        Paper search: int 1-20. Recommended: 1-10 for regression.

    Fixed parameters:
      - iterations=1500 (large enough; early stopping cuts it short).
      - od_type="Iter", od_wait=300 (paper: patience too low hurts accuracy).
      - use_best_model=True (reverts to best iteration after early stop).
      - border_count=254 (max quality per CatBoost docs).
    """
    fixed_params = {
        # Fixed quality / early stopping
        "iterations": 1500,
        "od_type": "Iter",
        "od_wait": 300,
        "use_best_model": True,
        "border_count": 254,
        "eval_metric": "R2",
        "random_state": 42,
        "verbose": False,
        "thread_count": -1,
    }
    tuned_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
    }
    params = {**fixed_params, **tuned_params}

    pruning_callback = CatBoostPruningCallback(trial, "R2")

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()

    y_pred = model.predict(X_val)
    val_r2 = float(r2_score(y_val, y_pred))
    trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
    trial.set_user_attr("fixed_params", fixed_params)
    trial.set_user_attr("full_params", params)
    trial.set_user_attr("val_r2", val_r2)
    return val_r2


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = repo_root / "ml" / "features"
    results_dir = repo_root / args.results_root
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CATBOOST HPO — FOCUSED VALIDATION OPTIMIZATION")
    print("=" * 80)
    print(f"Trials per target: {args.trials}")
    print(f"Targets:           {', '.join(args.targets)}")
    print(f"Features:          4254 dim")
    print(f"Sampler:           TPE")
    print(f"Pruning:           CatBoostPruningCallback (R2)")
    print(f"Output:            {results_dir}")

    X_train = load_feature_matrix(features_dir, "train")
    X_val = load_feature_matrix(features_dir, "val")
    print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}")

    summary_by_target = {}

    for target in args.targets:
        print("\n" + "=" * 80)
        print(f"TARGET: {target.upper()}")
        print("=" * 80)

        y_train = np.load(features_dir / f"y_train_{target}.npy").astype(np.float32)
        y_val = np.load(features_dir / f"y_val_{target}.npy").astype(np.float32)
        print(f"y_train: {y_train.shape}  mean={y_train.mean():.4f}")
        print(f"y_val:   {y_val.shape}  mean={y_val.mean():.4f}")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"catboost_{target}",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

        print(f"\nRunning {args.trials} trials...")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=args.trials,
            show_progress_bar=True,
        )

        best_trial = study.best_trial
        best_r2 = float(study.best_value)
        best_iteration = int(best_trial.user_attrs.get("best_iteration", -1))
        fixed_params = best_trial.user_attrs.get("fixed_params", {})
        tuned_params = dict(study.best_params)
        full_params = dict(best_trial.user_attrs.get("full_params", {}))
        best_iterations_for_retrain = best_iteration + 1 if best_iteration >= 0 else fixed_params.get("iterations", 1500)

        summary_by_target[target] = {
            "best_val_r2": best_r2,
            "best_iteration_on_val": best_iteration,
            "recommended_iterations_for_retrain": int(best_iterations_for_retrain),
            "tuned_params": tuned_params,
            "fixed_params": fixed_params,
            "full_params_for_refit": full_params,
        }

        print(f"\n  Best R²: {best_r2:.4f}")
        print(f"  Best iteration (val): {best_iteration}")
        print(f"  Suggested iterations for train+val refit: {best_iterations_for_retrain}")
        print(f"  Best tuned params: {tuned_params}")

        trials_df = study.trials_dataframe()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trials_path = results_dir / f"catboost_hpo_val_{target}_{timestamp}.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"  Trial history: {trials_path}")

        # Pruning report
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"  Trials completed: {n_completed}  Pruned: {n_pruned}")

    best_path = results_dir / "catboost_best_params.json"
    with open(best_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "trials_per_target": args.trials,
            "sampler": "TPESampler",
            "pruning": "CatBoostPruningCallback(R2)",
            "feature_dim": 4254,
            "best_by_target": summary_by_target,
            "recommendation": (
                "Tuned on val split only. For final thesis test, retrain each target on train+val "
                "using full_params_for_refit, but set iterations to recommended_iterations_for_retrain. "
                "Do not use test split for early stopping or hyperparameter selection."
            ),
        }, f, indent=2)
    print(f"\nAll best params saved to: {best_path}")

    print("\nSummary by target:")
    for target, info in summary_by_target.items():
        print(
            f"  {target:<14}: R²={info['best_val_r2']:.4f}  "
            f"best_iter={info['best_iteration_on_val']}  "
            f"retrain_iter={info['recommended_iterations_for_retrain']}"
        )


if __name__ == "__main__":
    main()
