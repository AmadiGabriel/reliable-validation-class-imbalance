#!/usr/bin/env python3
"""Nested-CV aero-engine validation for ALICE.

Key methodological rules implemented here
-----------------------------------------
1. The target problem is defined only on the original Class 0 / Class 1 data.
2. For FPs, original Class 2 observations are used as auxiliary positive
   training observations (surrogates).
3. Surrogates are never used in an inner-validation fold or an outer-test fold.
4. For permutation runs, only the Class 0 / Class 1 target labels are permuted.
   The Class 2 surrogate pool remains fixed and is still training-only.
5. Feature selection (SelectFromModel) is fitted inside every inner-training
   fit as part of the pipeline; the maximum retained feature count is fixed at 50.

The script performs one complete nested-CV run for one problem/classifier/seed.
Use the accompanying SLURM array scripts to run all combinations and the 90
independent label permutations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline


DROP_COLUMNS = [
    "L_-b/2a True Airspeed (knots)",
    "L_-b/2a CHT 3 (deg C)",
    "D_-b/2a Oil Pressure (PSI)",
    "L_-b/2a CHT 6 (deg C)",
    "C_-b/2a Barometer Setting (inHg)",
    "TO_-b/2a Barometer Setting (inHg)",
    "L_-b/2a Barometer Setting (inHg)",
    "D_-b/2a Barometer Setting (inHg)",
]

CLASSIFIERS = ("ada", "lgbm", "rf", "xgb")
PROBLEMS = ("FP", "FPs")
RUN_TYPES = ("true", "perm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    p.add_argument("--run-type", choices=RUN_TYPES, required=True,
                   help="'true' uses original target labels; 'perm' permutes target labels.")
    p.add_argument("--seed", type=int, required=True,
                   help="Run seed. Production permutation runs use seeds 1..90; true run uses 0.")
    p.add_argument("--data-dir", type=Path, default=Path("engine_data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--max-features", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Parallel hyperparameter configurations. Defaults to SLURM_CPUS_PER_TASK or 1.")
    p.add_argument("--max-configs", type=int, default=None,
                   help="Testing only: limit ParameterGrid to first N configurations.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Exit successfully if this run's summary JSON already exists.")
    return p.parse_args()


def normalise_n_jobs(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))


def load_data(data_dir: Path) -> pd.DataFrame:
    files = [data_dir / f"aeroeng{i}.csv" for i in range(1, 11)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing engine CSV files:\n  " + "\n  ".join(missing)
        )

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).fillna(0.1)

    if "Fault" not in df.columns:
        raise ValueError("Input data must contain a 'Fault' column.")

    missing_drop = [c for c in DROP_COLUMNS if c not in df.columns]
    if missing_drop:
        warnings.warn(
            "Some pre-identified VIF columns were not found and will not be dropped: "
            + ", ".join(missing_drop)
        )
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Fixed row shuffle for stable row ordering across every classifier/problem/run.
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_pipeline(classifier: str, model_seed: int, max_features: int):
    """Return (pipeline, classifier-only hyperparameter grid).

    SelectFromModel's max_features is fixed, not tuned.
    Estimator-level n_jobs is kept at 1 because joblib parallelises candidate
    configurations at the outer Python level.
    """
    if classifier == "lgbm":
        from lightgbm import LGBMClassifier

        base_est = LGBMClassifier(
            random_state=model_seed, n_jobs=1, verbosity=-1
        )
        clf_est = LGBMClassifier(
            random_state=model_seed, n_jobs=1, verbosity=-1
        )
        param_grid = {
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__n_estimators": [100, 150, 200, 250],
            "clf__max_depth": [3, 5, 7],
            "clf__num_leaves": [31, 63, 127],
        }

    elif classifier == "xgb":
        from xgboost import XGBClassifier

        base_est = XGBClassifier(
            random_state=model_seed,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        )
        clf_est = XGBClassifier(
            random_state=model_seed,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        )
        param_grid = {
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__n_estimators": [150, 200, 250],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.6, 0.8, 1.0],
        }

    elif classifier == "ada":
        from sklearn.ensemble import AdaBoostClassifier

        base_est = AdaBoostClassifier(random_state=model_seed)
        clf_est = AdaBoostClassifier(random_state=model_seed)
        param_grid = {
            "clf__n_estimators": [100, 150, 200, 250],
            "clf__learning_rate": [0.01, 0.1, 1.0],
        }

    elif classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        base_est = RandomForestClassifier(
            random_state=model_seed, n_jobs=1
        )
        clf_est = RandomForestClassifier(
            random_state=model_seed, n_jobs=1
        )
        param_grid = {
            "clf__n_estimators": [150, 200, 250, 300],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2", None],
        }

    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown classifier: {classifier}")

    selector = SelectFromModel(
        estimator=base_est,
        threshold=-np.inf,
        max_features=max_features,
    )
    pipe = Pipeline([("fs", selector), ("clf", clf_est)])
    return pipe, param_grid


def augment_training(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    problem: str,
    X_surrogate: pd.DataFrame,
    y_surrogate: np.ndarray,
):
    """Add surrogates to training data only for FPs."""
    if problem == "FPs":
        X_fit = pd.concat([X_train, X_surrogate], ignore_index=True)
        y_fit = np.concatenate([y_train, y_surrogate])
        return X_fit, y_fit
    return X_train, y_train


def evaluate_configuration(
    params: dict,
    pipe: Pipeline,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    problem: str,
    X_surrogate: pd.DataFrame,
    y_surrogate: np.ndarray,
) -> tuple[float, list[float], dict]:
    """Evaluate one classifier configuration over untouched inner validation folds."""
    inner_aucs: list[float] = []

    for inner_train_idx, inner_val_idx in inner_splits:
        X_inner_train = X_outer_train.iloc[inner_train_idx].reset_index(drop=True)
        y_inner_train = y_outer_train[inner_train_idx]

        # IMPORTANT: validation contains original target observations only.
        X_inner_val = X_outer_train.iloc[inner_val_idx].reset_index(drop=True)
        y_inner_val = y_outer_train[inner_val_idx]

        # Augmentation is applied only after the split and only to training.
        X_fit, y_fit = augment_training(
            X_inner_train,
            y_inner_train,
            problem,
            X_surrogate,
            y_surrogate,
        )

        candidate = clone(pipe)
        candidate.set_params(**params)
        candidate.fit(X_fit, y_fit)
        y_prob = candidate.predict_proba(X_inner_val)[:, 1]
        inner_aucs.append(float(roc_auc_score(y_inner_val, y_prob)))

    return float(np.mean(inner_aucs)), inner_aucs, params


def serialisable_params(params: dict) -> dict:
    result = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def run(args: argparse.Namespace) -> Path:
    n_jobs = normalise_n_jobs(args.n_jobs)
    started = time.time()

    # Separate deterministic streams for permutation, CV splitting, and model randomness.
    perm_seed = int(args.seed)
    cv_seed = 100_000 + int(args.seed)
    model_seed = 200_000 + int(args.seed)

    run_tag = f"{args.problem}_{args.classifier}_{args.run_type}_seed{args.seed:03d}"
    run_dir = args.results_dir / args.problem / args.classifier / args.run_type
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / f"summary_{run_tag}.json"

    if args.skip_existing and summary_path.exists():
        print(f"SKIP existing completed run: {summary_path}")
        return summary_path

    df = load_data(args.data_dir)

    # Define the original target problem BEFORE any permutation.
    target = df[df["Fault"].isin([0, 1])].reset_index(drop=True)
    surrogate = df[df["Fault"] == 2].reset_index(drop=True)

    X = target.drop(columns="Fault")
    y_original = target["Fault"].to_numpy(dtype=int)

    # LightGBM normalises some pandas feature names internally (for example,
    # spaces may become underscores). With recent scikit-learn feature-name
    # checks, SelectFromModel can then see different names at transform time.
    # Use stable, collision-free internal names for LightGBM only, while
    # retaining a map so selected features are written using their original
    # scientific names. This changes names only; feature values and order are
    # unchanged.
    original_feature_names = X.columns.to_list()
    lgbm_feature_name_map = None

    # Permute ONLY the original Class 0/1 target labels.
    if args.run_type == "perm":
        rng = np.random.default_rng(perm_seed)
        y = rng.permutation(y_original)
    else:
        y = y_original.copy()

    # Fixed Class-2 surrogate pool; used only in fitting for FPs.
    X_surrogate = surrogate.drop(columns="Fault")
    y_surrogate = np.ones(len(X_surrogate), dtype=int)

    if args.classifier == "lgbm":
        if list(X_surrogate.columns) != original_feature_names:
            raise ValueError("Target and surrogate feature columns do not match.")
        safe_names = [f"f_{k:04d}" for k in range(X.shape[1])]
        lgbm_feature_name_map = dict(zip(safe_names, original_feature_names))
        X = X.copy()
        X_surrogate = X_surrogate.copy()
        X.columns = safe_names
        X_surrogate.columns = safe_names

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    if len(class_counts) != 2:
        raise ValueError(f"Target labels must contain two classes, found: {class_counts}")

    min_minority_required = max(args.outer_folds, args.inner_folds)
    if min(class_counts.values()) < min_minority_required:
        raise ValueError(
            f"Too few observations in the smaller class for requested CV: {class_counts}"
        )

    pipe, param_grid = build_pipeline(
        args.classifier, model_seed=model_seed, max_features=args.max_features
    )
    configurations = list(ParameterGrid(param_grid))
    if args.max_configs is not None:
        configurations = configurations[: max(1, args.max_configs)]

    print("=" * 78)
    print(f"Run: {run_tag}")
    print(f"Problem: {args.problem}; classifier: {args.classifier}; run type: {args.run_type}")
    print(f"Target rows: {len(X)}; target class counts: {class_counts}")
    print(f"Surrogate rows available: {len(X_surrogate)} (training-only for FPs)")
    print(f"Features after fixed VIF removal: {X.shape[1]}")
    print(f"Outer/inner folds: {args.outer_folds}/{args.inner_folds}")
    print(f"Candidate configurations: {len(configurations)}; joblib workers: {n_jobs}")
    print("=" * 78, flush=True)

    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=cv_seed,
    )

    fold_records: list[dict] = []
    selected_feature_records: list[dict] = []
    roc_records: list[dict] = []
    prediction_records: list[dict] = []

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_train = y[outer_train_idx]

        # IMPORTANT: the outer test set is always original target data only.
        X_outer_test = X.iloc[outer_test_idx].reset_index(drop=True)
        y_outer_test = y[outer_test_idx]

        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=cv_seed + 1_000 + fold_idx,
        )
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))

        evaluated = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_configuration)(
                params,
                pipe,
                inner_splits,
                X_outer_train,
                y_outer_train,
                args.problem,
                X_surrogate,
                y_surrogate,
            )
            for params in configurations
        )

        mean_scores = np.asarray([x[0] for x in evaluated], dtype=float)
        best_idx = int(np.argmax(mean_scores))
        best_mean_inner_auc, best_inner_aucs, best_params = evaluated[best_idx]

        # Refit selected configuration on complete outer development data.
        X_outer_fit, y_outer_fit = augment_training(
            X_outer_train,
            y_outer_train,
            args.problem,
            X_surrogate,
            y_surrogate,
        )
        best_pipe = clone(pipe)
        best_pipe.set_params(**best_params)
        best_pipe.fit(X_outer_fit, y_outer_fit)

        y_prob = best_pipe.predict_proba(X_outer_test)[:, 1]
        outer_auc = float(roc_auc_score(y_outer_test, y_prob))

        fpr, tpr, thresholds = roc_curve(y_outer_test, y_prob)
        for point_idx, (fpr_v, tpr_v, thr_v) in enumerate(zip(fpr, tpr, thresholds)):
            roc_records.append({
                "fold": fold_idx,
                "point": point_idx,
                "fpr": float(fpr_v),
                "tpr": float(tpr_v),
                "threshold": float(thr_v),
            })

        support = best_pipe.named_steps["fs"].get_support()
        selected_features_internal = X.columns[support].tolist()
        if lgbm_feature_name_map is not None:
            selected_features = [
                lgbm_feature_name_map[name] for name in selected_features_internal
            ]
        else:
            selected_features = selected_features_internal
        for feature in selected_features:
            selected_feature_records.append({"fold": fold_idx, "feature": feature})

        for local_pos, global_idx in enumerate(outer_test_idx):
            prediction_records.append({
                "fold": fold_idx,
                "target_index": int(global_idx),
                "y_true": int(y_outer_test[local_pos]),
                "y_prob": float(y_prob[local_pos]),
            })

        fold_record = {
            "fold": fold_idx,
            "outer_auc": outer_auc,
            "best_mean_inner_auc": float(best_mean_inner_auc),
            "best_inner_aucs": json.dumps([float(v) for v in best_inner_aucs]),
            "best_params": json.dumps(serialisable_params(best_params), sort_keys=True),
            "n_selected_features": int(len(selected_features)),
            "n_outer_train_original": int(len(X_outer_train)),
            "n_surrogates_added": int(len(X_surrogate) if args.problem == "FPs" else 0),
            "n_outer_test_original": int(len(X_outer_test)),
        }
        fold_records.append(fold_record)

        print(
            f"Fold {fold_idx + 1}/{args.outer_folds}: "
            f"inner AUC={best_mean_inner_auc:.4f}, outer AUC={outer_auc:.4f}, "
            f"selected={len(selected_features)}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.DataFrame(prediction_records)
    selected_df = pd.DataFrame(selected_feature_records)
    roc_df = pd.DataFrame(roc_records)

    mean_outer_auc = float(fold_df["outer_auc"].mean())
    std_outer_auc = float(fold_df["outer_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0
    pooled_outer_auc = float(roc_auc_score(pred_df["y_true"], pred_df["y_prob"]))

    fold_path = run_dir / f"fold_results_{run_tag}.csv"
    pred_path = run_dir / f"oof_predictions_{run_tag}.csv"
    selected_path = run_dir / f"selected_features_{run_tag}.csv"
    roc_path = run_dir / f"roc_{run_tag}.csv"

    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    roc_df.to_csv(roc_path, index=False)

    summary = {
        "run_tag": run_tag,
        "problem": args.problem,
        "classifier": args.classifier,
        "run_type": args.run_type,
        "seed": int(args.seed),
        "perm_seed": perm_seed if args.run_type == "perm" else None,
        "cv_seed": cv_seed,
        "model_seed": model_seed,
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "fixed_max_features": int(args.max_features),
        "n_configurations": int(len(configurations)),
        "n_jobs": int(n_jobs),
        "n_target": int(len(X)),
        "n_target_class0": int(np.sum(y == 0)),
        "n_target_class1": int(np.sum(y == 1)),
        "n_surrogate": int(len(X_surrogate)),
        "surrogates_training_only": True,
        "permutation_scope": "target_class_0_1_labels_only" if args.run_type == "perm" else "none",
        "mean_outer_auc": mean_outer_auc,
        "std_outer_auc": std_outer_auc,
        "pooled_outer_auc": pooled_outer_auc,
        "elapsed_seconds": float(time.time() - started),
        "fold_results_file": str(fold_path),
        "oof_predictions_file": str(pred_path),
        "selected_features_file": str(selected_path),
        "roc_file": str(roc_path),
    }

    # Write summary last: its presence is the completion marker for --skip-existing.
    tmp_summary = summary_path.with_suffix(".json.tmp")
    with tmp_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    tmp_summary.replace(summary_path)

    print("-" * 78)
    print(f"Mean outer-fold AUC: {mean_outer_auc:.4f}")
    print(f"Pooled out-of-fold AUC: {pooled_outer_auc:.4f}")
    print(f"Completed in {(time.time() - started) / 60:.1f} minutes")
    print(f"Summary: {summary_path}")
    print("-" * 78, flush=True)
    return summary_path


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
