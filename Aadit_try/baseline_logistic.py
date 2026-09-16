"""
baseline_logistic.py

Phase 3A — Linear/additive baseline using the frozen 5-fold split.

Experiment question:
    How much ROC-AUC can a simple additive linear model extract from the
    raw features before we introduce tree boosting or feature engineering?

Important:
- Uses the exact fold assignment created by validation_setup.py.
- Excludes the likely ID column from modeling.
- One-hot encodes object/string/category/bool features.
- Standardizes numeric features.
- Saves OOF predictions and fold metrics.
- Does NOT create a Kaggle submission.

Run:
    python baseline_logistic.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "experiments" / "logistic_raw"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe 5-fold logistic-regression baseline."
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
        raise ValueError(f"Fold file is missing required columns: {sorted(missing)}")

    if len(folds) != len(train):
        raise ValueError(
            f"Fold file has {len(folds):,} rows but train has {len(train):,}."
        )

    expected_index = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(folds["row_index"].to_numpy(), expected_index):
        raise ValueError(
            "row_index in the fold file does not exactly match train row order."
        )

    unique_folds = sorted(folds["fold"].unique().tolist())

    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"Expected frozen folds [0, 1, 2, 3, 4], found {unique_folds}."
        )

    id_col = None

    # validation_setup.py writes the detected ID between row_index and fold.
    possible_id_columns = [
        c for c in folds.columns if c not in {"row_index", "fold"}
    ]

    if len(possible_id_columns) > 1:
        raise ValueError(
            "Fold file contains more than one possible ID column: "
            f"{possible_id_columns}"
        )

    if possible_id_columns:
        id_col = possible_id_columns[0]

        if id_col not in train.columns:
            raise ValueError(
                f"Fold file ID column {id_col!r} is not present in train.csv."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                f"Fold-file {id_col!r} values do not align with train.csv."
            )

    return folds["fold"].to_numpy(dtype=np.int16), id_col


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No usable features remain after preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
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

    # ID is useful for row tracking, but our diagnostics showed no useful
    # predictive relationship and test IDs continue beyond train IDs.
    if id_col is not None:
        features = [c for c in common_features if c != id_col]
    else:
        features = common_features

    X = train[features]
    X_test = test[features]

    categorical_features = [
        c
        for c in features
        if (
            pd.api.types.is_object_dtype(train[c].dtype)
            or isinstance(train[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train[c].dtype)
            or pd.api.types.is_string_dtype(train[c].dtype)
        )
    ]

    numeric_features = [
        c for c in features if c not in categorical_features
    ]

    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"ID excluded from model: {id_col!r}")
    print(f"Model features: {len(features)}")
    print(f"  numeric: {len(numeric_features)}")
    print(f"  categorical/binary one-hot: {len(categorical_features)}")
    print()
    print("Numeric features:")
    for c in numeric_features:
        print(f"  - {c}")
    print("Categorical/binary features:")
    for c in categorical_features:
        print(f"  - {c}")

    oof = np.full(len(train), np.nan, dtype=np.float64)
    test_fold_predictions = []
    metric_rows = []

    total_start = time.perf_counter()

    print()
    print("Running frozen 5-fold CV...")

    for fold in range(5):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)

        pipeline = build_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        fit_start = time.perf_counter()
        pipeline.fit(
            X.iloc[train_idx],
            y[train_idx],
        )
        fit_seconds = time.perf_counter() - fit_start

        infer_start = time.perf_counter()

        valid_pred = pipeline.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

        test_pred = pipeline.predict_proba(X_test)[:, 1]

        inference_seconds = time.perf_counter() - infer_start

        oof[valid_idx] = valid_pred
        test_fold_predictions.append(test_pred.astype(np.float32))

        fold_auc = roc_auc_score(
            y[valid_idx],
            valid_pred,
        )

        transformed_feature_count = len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        metric_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "positive_rate_valid": float(y[valid_idx].mean()),
                "auc": float(fold_auc),
                "fit_seconds": float(fit_seconds),
                "inference_seconds_valid_plus_test": float(inference_seconds),
                "transformed_feature_count": transformed_feature_count,
                "model_n_iter": int(
                    np.max(pipeline.named_steps["model"].n_iter_)
                ),
            }
        )

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.6f} | "
            f"fit={fit_seconds:.1f}s | "
            f"inference={inference_seconds:.1f}s | "
            f"features_after_encoding={transformed_feature_count}"
        )

    total_seconds = time.perf_counter() - total_start

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions contain missing values.")

    fold_metrics = pd.DataFrame(metric_rows)

    mean_fold_auc = float(fold_metrics["auc"].mean())
    std_fold_auc = float(fold_metrics["auc"].std(ddof=1))
    oof_auc = float(roc_auc_score(y, oof))

    test_prediction = np.mean(
        np.vstack(test_fold_predictions),
        axis=0,
    )

    # ----------------------------
    # Save experiment artifacts
    # ----------------------------
    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
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
            "prediction": test_prediction.astype(np.float32),
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

    summary_lines = [
        "EXPERIMENT: LOGISTIC_RAW_BASELINE",
        "=" * 72,
        "",
        "QUESTION",
        "How much ROC-AUC can a simple additive linear model extract from",
        "the raw features before nonlinear tree models or feature engineering?",
        "",
        "VALIDATION",
        "Frozen 5-fold StratifiedKFold assignment from validation_setup.py",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "FEATURE POLICY",
        f"Excluded ID: {id_col!r}",
        f"Raw model features: {len(features)}",
        f"Numeric features: {len(numeric_features)}",
        f"Categorical/binary one-hot features: {len(categorical_features)}",
        f"Numeric: {numeric_features}",
        f"Categorical/binary: {categorical_features}",
        "",
        "MODEL",
        "LogisticRegression(C=1.0, L2 penalty, solver='lbfgs')",
        "Numeric preprocessing: StandardScaler",
        "Categorical preprocessing: OneHotEncoder(handle_unknown='ignore')",
        "",
        "RESULTS",
        f"Mean fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {std_fold_auc:.8f}",
        f"OOF AUC: {oof_auc:.8f}",
        f"Total CV + test inference time: {total_seconds:.2f} seconds",
        "",
        "NOTE",
        "test_predictions.csv is saved only so this model can later be compared",
        "or blended consistently. It is NOT a Kaggle submission.",
    ]

    summary_text = "\n".join(summary_lines)

    (args.output_dir / "summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("LOGISTIC RAW BASELINE COMPLETE")
    print("=" * 78)
    print(f"Mean fold AUC : {mean_fold_auc:.8f}")
    print(f"Fold AUC std  : {std_fold_auc:.8f}")
    print(f"OOF AUC       : {oof_auc:.8f}")
    print(f"Total runtime : {total_seconds:.2f}s")
    print(f"Artifacts     : {args.output_dir.resolve()}")
    print("=" * 78)
    print()
    print("Do NOT judge the final model choice from this baseline alone.")
    print("Send me the terminal output, summary.txt, and fold_metrics.csv.")


if __name__ == "__main__":
    main()
