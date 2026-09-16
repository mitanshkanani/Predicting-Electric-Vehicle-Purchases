"""
baseline_catboost.py

Phase 3B — First nonlinear baseline using CatBoost and the frozen 5-fold split.

Experiment question:
    Does a nonlinear tree model with native categorical handling materially
    improve ROC-AUC over the raw logistic-regression baseline?

Experimental-control rule:
    Keep the same raw 13 model features and the same current feature-type policy
    used by baseline_logistic.py. Only the model family changes.

Important:
- Uses the exact frozen folds created by validation_setup.py.
- Excludes the ID column.
- Treats only raw string/object/category/bool columns as categorical.
- Keeps low-cardinality numeric columns numeric for THIS experiment.
- Uses early stopping on each validation fold.
- Saves OOF/test predictions, fold metrics, and mean feature importance.
- Does NOT create a Kaggle submission.

Run:
    python baseline_catboost.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise SystemExit(
        "\nCatBoost is not installed.\n"
        "Install it with:\n"
        "    python -m pip install catboost\n"
        "Then run this script again.\n"
    ) from exc


SEED = 42
DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "experiments" / "catboost_raw"
DEFAULT_LOGISTIC_OOF = (
    Path("artifacts") / "experiments" / "logistic_raw" / "oof_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first CatBoost baseline on the frozen folds."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logistic-oof", type=Path, default=DEFAULT_LOGISTIC_OOF)
    return parser.parse_args()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise ValueError(
            "Could not safely infer target. Expected exactly one train-only "
            f"column, found: {train_only}"
        )
    return train_only[0]


def encode_binary_target(y: pd.Series) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise ValueError(
            f"Expected a binary target, found {len(values)} values: {values}"
        )

    preferred_positive = {"yes", "true", "1", "positive", "buy", "will_buy"}
    positive_label = None
    for value in values:
        if str(value).strip().lower() in preferred_positive:
            positive_label = value
            break
    if positive_label is None:
        positive_label = y.value_counts().idxmin()

    return (y == positive_label).astype(np.int8).to_numpy(), positive_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_folds(
    folds: pd.DataFrame, train: pd.DataFrame
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"Fold file is missing columns: {sorted(missing)}")

    if len(folds) != len(train):
        raise ValueError(
            f"Fold file has {len(folds):,} rows but train has {len(train):,}."
        )

    expected_index = np.arange(len(train), dtype=np.int64)
    if not np.array_equal(folds["row_index"].to_numpy(), expected_index):
        raise ValueError("Fold row_index does not exactly match train row order.")

    unique_folds = sorted(folds["fold"].unique().tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"Expected frozen folds [0, 1, 2, 3, 4], found {unique_folds}."
        )

    possible_id_columns = [
        c for c in folds.columns if c not in {"row_index", "fold"}
    ]
    if len(possible_id_columns) > 1:
        raise ValueError(
            "Fold file contains more than one possible ID column: "
            f"{possible_id_columns}"
        )

    id_col = possible_id_columns[0] if possible_id_columns else None
    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(f"Fold ID column {id_col!r} is not in train.csv.")
        if not np.array_equal(folds[id_col].to_numpy(), train[id_col].to_numpy()):
            raise ValueError(f"Fold-file {id_col!r} values do not align with train.csv.")

    return folds["fold"].to_numpy(dtype=np.int16), id_col


def detect_categorical_features(
    train: pd.DataFrame, features: list[str]
) -> list[str]:
    categorical = []
    for c in features:
        dtype = train[c].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            categorical.append(c)
    return categorical


def safe_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(a).rank(method="average").to_numpy()
    b_rank = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def load_logistic_comparison(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray | None:
    if not path.exists():
        return None

    previous = pd.read_csv(path)
    required = {"row_index", "fold", "oof_prediction"}
    if not required.issubset(previous.columns) or len(previous) != len(train):
        print("Previous logistic OOF is incompatible; comparison skipped.")
        return None

    if not np.array_equal(
        previous["row_index"].to_numpy(), np.arange(len(train), dtype=np.int64)
    ):
        print("Previous logistic OOF row order differs; comparison skipped.")
        return None

    if not np.array_equal(previous["fold"].to_numpy(), fold_ids):
        print("Previous logistic OOF uses different folds; comparison skipped.")
        return None

    if id_col is not None and id_col in previous.columns:
        if not np.array_equal(previous[id_col].to_numpy(), train[id_col].to_numpy()):
            print("Previous logistic OOF IDs differ; comparison skipped.")
            return None

    return previous["oof_prediction"].to_numpy(dtype=np.float64)


def build_model() -> CatBoostClassifier:
    # Intentionally untuned. This experiment isolates nonlinear modeling.
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        thread_count=-1,
        allow_writing_files=False,
    )


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    for path in (train_path, test_path, args.folds_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and frozen folds...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(train, test)
    y, positive_label = encode_binary_target(train[target])
    fold_ids, id_col = validate_folds(folds_df, train)

    common_features = [c for c in test.columns if c in train.columns]
    features = [c for c in common_features if c != id_col]
    categorical_features = detect_categorical_features(train, features)
    numeric_features = [c for c in features if c not in categorical_features]

    X = train[features]
    X_test = test[features]

    previous_logistic_oof = load_logistic_comparison(
        args.logistic_oof, train, fold_ids, id_col
    )

    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"ID excluded from model: {id_col!r}")
    print(f"Model features: {len(features)}")
    print(f"  numeric: {len(numeric_features)}")
    print(f"  native categorical/binary: {len(categorical_features)}")
    print("\nNumeric features:")
    for c in numeric_features:
        print(f"  - {c}")
    print("Native categorical/binary features:")
    for c in categorical_features:
        print(f"  - {c}")

    print("\nCatBoost baseline parameters:")
    print("  iterations=2000")
    print("  learning_rate=0.05")
    print("  depth=6")
    print("  early_stopping_rounds=150")

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=features,
    )

    oof = np.full(len(train), np.nan, dtype=np.float64)
    test_fold_predictions: list[np.ndarray] = []
    metric_rows = []
    importance_rows = []

    total_start = time.perf_counter()
    print("\nRunning frozen 5-fold CatBoost CV...")

    for fold in range(5):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)

        train_pool = Pool(
            X.iloc[train_idx],
            label=y[train_idx],
            cat_features=categorical_features,
            feature_names=features,
        )
        valid_pool = Pool(
            X.iloc[valid_idx],
            label=y[valid_idx],
            cat_features=categorical_features,
            feature_names=features,
        )

        model = build_model()
        fit_start = time.perf_counter()
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=200,
        )
        fit_seconds = time.perf_counter() - fit_start

        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(valid_pool)[:, 1]
        test_pred = model.predict_proba(test_pool)[:, 1]
        inference_seconds = time.perf_counter() - infer_start

        oof[valid_idx] = valid_pred
        test_fold_predictions.append(test_pred.astype(np.float32))
        fold_auc = float(roc_auc_score(y[valid_idx], valid_pred))

        best_iteration = int(model.get_best_iteration())
        tree_count = int(model.tree_count_)
        metric_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "positive_rate_valid": float(y[valid_idx].mean()),
                "auc": fold_auc,
                "best_iteration_zero_based": best_iteration,
                "tree_count": tree_count,
                "fit_seconds": float(fit_seconds),
                "inference_seconds_valid_plus_test": float(inference_seconds),
            }
        )

        for feature, importance in zip(
            features,
            model.get_feature_importance(type="PredictionValuesChange"),
        ):
            importance_rows.append(
                {"fold": fold, "feature": feature, "importance": float(importance)}
            )

        print(
            f"Fold {fold}: AUC={fold_auc:.6f} | "
            f"best_iter={best_iteration} | trees={tree_count} | "
            f"fit={fit_seconds:.1f}s | inference={inference_seconds:.1f}s"
        )
        del model, train_pool, valid_pool

    total_seconds = time.perf_counter() - total_start
    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions contain missing values.")

    fold_metrics = pd.DataFrame(metric_rows)
    mean_fold_auc = float(fold_metrics["auc"].mean())
    std_fold_auc = float(fold_metrics["auc"].std(ddof=1))
    oof_auc = float(roc_auc_score(y, oof))
    test_prediction = np.mean(np.vstack(test_fold_predictions), axis=0)

    feature_importance_by_fold = pd.DataFrame(importance_rows)
    feature_importance = (
        feature_importance_by_fold.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    logistic_auc = auc_delta = prediction_corr = rank_corr = None
    if previous_logistic_oof is not None:
        logistic_auc = float(roc_auc_score(y, previous_logistic_oof))
        auc_delta = float(oof_auc - logistic_auc)
        prediction_corr = float(np.corrcoef(oof, previous_logistic_oof)[0, 1])
        rank_corr = safe_rank_correlation(oof, previous_logistic_oof)

    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    feature_importance_by_fold.to_csv(
        args.output_dir / "feature_importance_by_fold.csv", index=False
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": oof.astype(np.float32),
        }
    )
    if id_col is not None:
        oof_output.insert(1, id_col, train[id_col].to_numpy())
    oof_output.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    test_output = pd.DataFrame(
        {"prediction": test_prediction.astype(np.float32)}
    )
    if id_col is not None:
        test_output.insert(0, id_col, test[id_col].to_numpy())
    test_output.to_csv(args.output_dir / "test_predictions.csv", index=False)

    summary_lines = [
        "EXPERIMENT: CATBOOST_RAW_BASELINE",
        "=" * 72,
        "",
        "QUESTION",
        "Does a nonlinear tree model with native categorical handling",
        "materially improve ROC-AUC over the raw logistic baseline?",
        "",
        "EXPERIMENTAL CONTROL",
        "Same frozen folds.",
        "Same 13 raw model features.",
        "Same current feature-type policy.",
        "Only the model family changes.",
        "",
        "VALIDATION",
        "Frozen 5-fold StratifiedKFold assignment from validation_setup.py",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "FEATURE POLICY",
        f"Excluded ID: {id_col!r}",
        f"Numeric features: {numeric_features}",
        f"Native categorical/binary features: {categorical_features}",
        "",
        "MODEL",
        "CatBoostClassifier",
        "iterations=2000",
        "learning_rate=0.05",
        "depth=6",
        "loss_function='Logloss'",
        "eval_metric='AUC'",
        "early_stopping_rounds=150",
        "",
        "RESULTS",
        f"Mean fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {std_fold_auc:.8f}",
        f"OOF AUC: {oof_auc:.8f}",
        f"Mean best iteration: {fold_metrics['best_iteration_zero_based'].mean():.1f}",
        f"Mean tree count: {fold_metrics['tree_count'].mean():.1f}",
        f"Total CV + test inference time: {total_seconds:.2f} seconds",
    ]

    if logistic_auc is not None:
        summary_lines.extend(
            [
                "",
                "DIRECT COMPARISON VS LOGISTIC RAW",
                f"Logistic OOF AUC: {logistic_auc:.8f}",
                f"CatBoost OOF AUC: {oof_auc:.8f}",
                f"AUC delta: {auc_delta:+.8f}",
                f"OOF probability correlation: {prediction_corr:.6f}",
                f"OOF rank correlation: {rank_corr:.6f}",
            ]
        )

    summary_lines.extend(["", "TOP MEAN FEATURE IMPORTANCES"])
    for row in feature_importance.head(10).itertuples():
        summary_lines.append(f"{row.feature}: {row.mean_importance:.6f}")

    summary_lines.extend(
        [
            "",
            "NOTE",
            "This is a baseline, not tuned CatBoost.",
            "Low-cardinality numeric columns remain numeric deliberately.",
            "test_predictions.csv is NOT a Kaggle submission.",
        ]
    )

    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("CATBOOST RAW BASELINE COMPLETE")
    print("=" * 78)
    print(f"Mean fold AUC : {mean_fold_auc:.8f}")
    print(f"Fold AUC std  : {std_fold_auc:.8f}")
    print(f"OOF AUC       : {oof_auc:.8f}")
    if logistic_auc is not None:
        print(f"Logistic OOF  : {logistic_auc:.8f}")
        print(f"Delta         : {auc_delta:+.8f}")
        print(f"OOF corr      : {prediction_corr:.6f}")
        print(f"Rank corr     : {rank_corr:.6f}")
    print(f"Total runtime : {total_seconds:.2f}s")
    print(f"Artifacts     : {args.output_dir.resolve()}")
    print("=" * 78)
    print("\nSend me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. feature_importance.csv")


if __name__ == "__main__":
    main()
