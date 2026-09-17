"""
baseline_catboost_gpu.py

Phase 3B-GPU — Raw CatBoost baseline using the frozen 5-fold split.

Goal:
    Re-establish our CatBoost baseline on GPU so every later CatBoost
    experiment can be compared fairly against the same compute mode.

Important:
- Uses the exact frozen folds from validation_setup.py.
- Excludes the ID column.
- Uses the same 13 raw model features as the CPU CatBoost baseline.
- Keeps the same feature-type policy.
- Uses GPU training: task_type="GPU", devices="0".
- Saves OOF predictions, test predictions, fold metrics, and feature importance.
- Does NOT create a Kaggle submission.

Run:
    python baseline_catboost_gpu.py

If CatBoost is missing:
    python -m pip install catboost
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
DEFAULT_OUTPUT_DIR = Path("artifacts") / "experiments" / "catboost_raw_gpu"
DEFAULT_LOGISTIC_OOF = (
    Path("artifacts")
    / "experiments"
    / "logistic_raw"
    / "oof_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the raw CatBoost baseline on GPU using frozen folds."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
        help="Frozen fold assignment produced by validation_setup.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for experiment artifacts.",
    )
    parser.add_argument(
        "--logistic-oof",
        type=Path,
        default=DEFAULT_LOGISTIC_OOF,
        help="Optional logistic OOF file for direct comparison.",
    )
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

    preferred_positive = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive_label = None

    for value in values:
        if str(value).strip().lower() in preferred_positive:
            positive_label = value
            break

    if positive_label is None:
        positive_label = y.value_counts().idxmin()

    encoded = (y == positive_label).astype(np.int8).to_numpy()

    return encoded, positive_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}

    missing = required - set(folds.columns)
    if missing:
        raise ValueError(
            f"Fold file is missing required columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            f"Fold file has {len(folds):,} rows but train has {len(train):,}."
        )

    expected_index = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        expected_index,
    ):
        raise ValueError(
            "row_index in the fold file does not exactly match train row order."
        )

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

    id_col = None

    if possible_id_columns:
        id_col = possible_id_columns[0]

        if id_col not in train.columns:
            raise ValueError(
                f"Fold-file ID column {id_col!r} does not exist in train.csv."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                f"Fold-file {id_col!r} values do not align with train.csv."
            )

    return folds["fold"].to_numpy(dtype=np.int16), id_col


def detect_categorical_features(
    train: pd.DataFrame,
    features: list[str],
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

    if not required.issubset(previous.columns):
        print(
            "Previous logistic OOF exists but has unexpected columns; "
            "comparison will be skipped."
        )
        return None

    if len(previous) != len(train):
        print(
            "Previous logistic OOF row count does not match train; "
            "comparison will be skipped."
        )
        return None

    expected_index = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(
        previous["row_index"].to_numpy(),
        expected_index,
    ):
        print(
            "Previous logistic OOF row order does not match train; "
            "comparison will be skipped."
        )
        return None

    if not np.array_equal(
        previous["fold"].to_numpy(),
        fold_ids,
    ):
        print(
            "Previous logistic OOF folds differ from frozen folds; "
            "comparison will be skipped."
        )
        return None

    if id_col is not None and id_col in previous.columns:
        if not np.array_equal(
            previous[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            print(
                "Previous logistic OOF IDs do not align; "
                "comparison will be skipped."
            )
            return None

    return previous["oof_prediction"].to_numpy(dtype=np.float64)


def build_model() -> CatBoostClassifier:
    """
    Same modeling settings as our CPU CatBoost baseline,
    with GPU enabled.

    We intentionally do NOT tune hyperparameters yet.
    """
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=2000,
        learning_rate=0.05,
        depth=6,

        random_seed=SEED,

        # GPU SETTINGS
        task_type="GPU",
        devices="0",

        # Keep CatBoost from creating catboost_info/ folders.
        allow_writing_files=False,
    )


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    for path in (train_path, test_path, args.folds_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and frozen folds...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(train, test)

    y, positive_label = encode_binary_target(
        train[target]
    )

    fold_ids, id_col = validate_folds(
        folds_df,
        train,
    )

    common_features = [
        c
        for c in test.columns
        if c in train.columns
    ]

    features = [
        c
        for c in common_features
        if c != id_col
    ]

    categorical_features = detect_categorical_features(
        train,
        features,
    )

    numeric_features = [
        c
        for c in features
        if c not in categorical_features
    ]

    X = train[features]
    X_test = test[features]

    previous_logistic_oof = load_logistic_comparison(
        path=args.logistic_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
    )

    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(f"ID excluded from model: {id_col!r}")
    print(f"Model features: {len(features)}")
    print(f"  numeric: {len(numeric_features)}")
    print(
        f"  native categorical/binary: "
        f"{len(categorical_features)}"
    )

    print()
    print("Numeric features:")

    for c in numeric_features:
        print(f"  - {c}")

    print("Native categorical/binary features:")

    for c in categorical_features:
        print(f"  - {c}")

    print()
    print("CatBoost GPU baseline parameters:")
    print("  task_type='GPU'")
    print("  devices='0'")
    print("  iterations=2000")
    print("  learning_rate=0.05")
    print("  depth=6")
    print("  loss_function='Logloss'")
    print("  eval_metric='AUC'")
    print("  early_stopping_rounds=150")
    print()

    # Build test Pool once and reuse across folds.
    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=features,
    )

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []

    metric_rows = []
    importance_rows = []

    total_start = time.perf_counter()

    print("Running frozen 5-fold CatBoost GPU CV...")

    for fold in range(5):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

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

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        infer_start = time.perf_counter()

        valid_pred = model.predict_proba(
            valid_pool
        )[:, 1]

        test_pred = model.predict_proba(
            test_pool
        )[:, 1]

        inference_seconds = (
            time.perf_counter()
            - infer_start
        )

        oof[valid_idx] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        best_iteration = int(
            model.get_best_iteration()
        )

        tree_count = int(
            model.tree_count_
        )

        metric_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "positive_rate_valid": float(
                    y[valid_idx].mean()
                ),
                "auc": fold_auc,
                "best_iteration_zero_based": best_iteration,
                "tree_count": tree_count,
                "fit_seconds": float(fit_seconds),
                "inference_seconds_valid_plus_test": float(
                    inference_seconds
                ),
            }
        )

        feature_importance = model.get_feature_importance(
            type="PredictionValuesChange"
        )

        for feature, importance in zip(
            features,
            feature_importance,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

        print()
        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.6f} | "
            f"best_iter={best_iteration} | "
            f"trees={tree_count} | "
            f"fit={fit_seconds:.1f}s | "
            f"inference={inference_seconds:.1f}s"
        )
        print()

        del model, train_pool, valid_pool

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "OOF predictions contain missing values."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    mean_fold_auc = float(
        fold_metrics["auc"].mean()
    )

    std_fold_auc = float(
        fold_metrics["auc"].std(ddof=1)
    )

    oof_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    # ----------------------------
    # Feature importance
    # ----------------------------

    feature_importance_by_fold = pd.DataFrame(
        importance_rows
    )

    feature_importance = (
        feature_importance_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),
            std_importance=(
                "importance",
                "std",
            ),
        )
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ----------------------------
    # Logistic comparison
    # ----------------------------

    logistic_auc = None
    auc_delta = None
    probability_corr = None
    rank_corr = None

    if previous_logistic_oof is not None:
        logistic_auc = float(
            roc_auc_score(
                y,
                previous_logistic_oof,
            )
        )

        auc_delta = (
            oof_auc
            - logistic_auc
        )

        probability_corr = float(
            np.corrcoef(
                oof,
                previous_logistic_oof,
            )[0, 1]
        )

        rank_corr = safe_rank_correlation(
            oof,
            previous_logistic_oof,
        )

    # ----------------------------
    # Save artifacts
    # ----------------------------

    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )

    feature_importance.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
    )

    feature_importance_by_fold.to_csv(
        args.output_dir
        / "feature_importance_by_fold.csv",
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": oof.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[id_col].to_numpy(),
        )

    oof_output.to_csv(
        args.output_dir / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": test_prediction.astype(
                np.float32
            )
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    # ----------------------------
    # Summary
    # ----------------------------

    summary_lines = [
        "EXPERIMENT: CATBOOST_RAW_GPU_BASELINE",
        "=" * 72,
        "",
        "QUESTION",
        "What is our raw CatBoost baseline when training on GPU?",
        "",
        "PURPOSE",
        "This becomes the fair comparison baseline for all later GPU CatBoost",
        "experiments such as source augmentation and feature-type ablations.",
        "",
        "VALIDATION",
        "Frozen 5-fold StratifiedKFold assignment from validation_setup.py",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "FEATURE POLICY",
        f"Excluded ID: {id_col!r}",
        f"Raw model features: {len(features)}",
        f"Numeric features: {numeric_features}",
        f"Native categorical/binary features: {categorical_features}",
        "",
        "MODEL",
        "CatBoostClassifier",
        "task_type='GPU'",
        "devices='0'",
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
        f"Mean best iteration: "
        f"{fold_metrics['best_iteration_zero_based'].mean():.1f}",
        f"Mean tree count: "
        f"{fold_metrics['tree_count'].mean():.1f}",
        f"Total CV + test inference time: "
        f"{total_seconds:.2f} seconds",
    ]

    if logistic_auc is not None:
        summary_lines.extend(
            [
                "",
                "DIRECT COMPARISON VS LOGISTIC RAW",
                f"Logistic OOF AUC: {logistic_auc:.8f}",
                f"CatBoost GPU OOF AUC: {oof_auc:.8f}",
                f"AUC delta: {auc_delta:+.8f}",
                f"OOF probability correlation: "
                f"{probability_corr:.6f}",
                f"OOF rank correlation: "
                f"{rank_corr:.6f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "TOP MEAN FEATURE IMPORTANCES",
        ]
    )

    for row in feature_importance.head(10).itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "IMPORTANT",
            "Do NOT compare tiny score differences between this GPU model and",
            "the previous CPU CatBoost run as a feature/model improvement.",
            "GPU CatBoost is not bit-for-bit deterministic.",
            "",
            "All future CatBoost experiments should compare against THIS GPU",
            "baseline using the same frozen folds.",
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CATBOOST RAW GPU BASELINE COMPLETE"
    )
    print("=" * 78)
    print(
        f"Mean fold AUC : "
        f"{mean_fold_auc:.8f}"
    )
    print(
        f"Fold AUC std  : "
        f"{std_fold_auc:.8f}"
    )
    print(
        f"OOF AUC       : "
        f"{oof_auc:.8f}"
    )

    if logistic_auc is not None:
        print(
            f"Logistic OOF  : "
            f"{logistic_auc:.8f}"
        )
        print(
            f"Delta         : "
            f"{auc_delta:+.8f}"
        )

    print(
        f"Total runtime : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts     : "
        f"{args.output_dir.resolve()}"
    )
    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. feature_importance.csv")


if __name__ == "__main__":
    main()
