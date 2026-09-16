"""
baseline_xgboost_gpu.py

Phase 4A — XGBoost GPU baseline on the frozen 5-fold split.

EXPERIMENT QUESTION
-------------------
Can XGBoost capture useful structure that differs from CatBoost while using
the same raw 13 features and frozen validation folds?

Why this experiment matters:
- CatBoost GPU OOF is our current reference.
- A second strong booster gives us model diversity.
- Even if XGBoost scores slightly lower, low OOF correlation can make it
  valuable later in an ensemble.

This script:
- uses the exact frozen 5-fold assignment
- excludes id
- keeps the same 7 numeric features numeric
- treats the same 6 string/binary features as native XGBoost categoricals
- trains on GPU
- uses early stopping
- saves OOF predictions, test predictions, fold metrics, and feature importance
- directly compares OOF predictions against CatBoost GPU
- does NOT create a Kaggle submission

Run:
    python baseline_xgboost_gpu.py

Install if needed:
    python -m pip install -U xgboost
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
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U xgboost\n"
    ) from exc


SEED = 42

DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"

DEFAULT_CATBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_LOGISTIC_OOF = (
    Path("artifacts")
    / "experiments"
    / "logistic_raw"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_raw_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GPU XGBoost baseline using the frozen folds."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
    )

    parser.add_argument(
        "--catboost-oof",
        type=Path,
        default=DEFAULT_CATBOOST_OOF,
    )

    parser.add_argument(
        "--logistic-oof",
        type=Path,
        default=DEFAULT_LOGISTIC_OOF,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def detect_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    train_only = [
        c
        for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            "Expected exactly one train-only target column; "
            f"found: {train_only}"
        )

    return train_only[0]


def encode_binary_target(
    y: pd.Series,
) -> tuple[np.ndarray, object]:
    values = list(
        pd.unique(
            y.dropna()
        )
    )

    if len(values) != 2:
        raise ValueError(
            f"Expected binary target; found {values}"
        )

    preferred = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive = None

    for value in values:
        if (
            str(value)
            .strip()
            .lower()
            in preferred
        ):
            positive = value
            break

    if positive is None:
        positive = (
            y.value_counts()
            .idxmin()
        )

    encoded = (
        y == positive
    ).astype(
        np.int8
    ).to_numpy()

    return encoded, positive


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {
        "row_index",
        "fold",
    }

    missing = (
        required
        - set(
            folds.columns
        )
    )

    if missing:
        raise ValueError(
            f"Fold file is missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen fold row count does not match train.csv."
        )

    if not np.array_equal(
        folds[
            "row_index"
        ].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    unique_folds = sorted(
        folds[
            "fold"
        ].unique().tolist()
    )

    if unique_folds != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise ValueError(
            f"Expected folds [0,1,2,3,4], found {unique_folds}"
        )

    extra = [
        c
        for c in folds.columns
        if c
        not in {
            "row_index",
            "fold",
        }
    ]

    if len(extra) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extra}"
        )

    id_col = (
        extra[0]
        if extra
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} is missing from train.csv."
            )

        if not np.array_equal(
            folds[
                id_col
            ].to_numpy(),
            train[
                id_col
            ].to_numpy(),
        ):
            raise ValueError(
                "Frozen fold IDs do not align with train.csv."
            )

    return (
        folds[
            "fold"
        ].to_numpy(
            dtype=np.int16
        ),
        id_col,
    )


def detect_categorical_features(
    train: pd.DataFrame,
    features: list[str],
) -> list[str]:
    categorical = []

    for c in features:
        dtype = train[
            c
        ].dtype

        if (
            pd.api.types.is_object_dtype(
                dtype
            )
            or
            pd.api.types.is_string_dtype(
                dtype
            )
            or
            pd.api.types.is_bool_dtype(
                dtype
            )
            or
            isinstance(
                dtype,
                pd.CategoricalDtype,
            )
        ):
            categorical.append(
                c
            )

    return categorical


def prepare_xgboost_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make categorical dtypes consistent across train/test.

    We use the union of train and test category labels. This does NOT use
    test labels/target information; it only ensures both DataFrames share
    the same category vocabulary.
    """
    x_train = train[
        features
    ].copy()

    x_test = test[
        features
    ].copy()

    for c in categorical_features:
        train_values = (
            x_train[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        test_values = (
            x_test[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        categories = pd.Index(
            pd.concat(
                [
                    train_values,
                    test_values,
                ],
                axis=0,
                ignore_index=True,
            ).unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        x_train[
            c
        ] = train_values.astype(
            dtype
        )

        x_test[
            c
        ] = test_values.astype(
            dtype
        )

    return (
        x_train,
        x_test,
    )


def load_oof(
    path: Path,
    prediction_column: str,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    required: bool,
) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required comparison OOF file not found:\n{path.resolve()}"
            )
        return None

    df = pd.read_csv(
        path
    )

    required_columns = {
        "row_index",
        "fold",
        prediction_column,
    }

    if not required_columns.issubset(
        df.columns
    ):
        if required:
            raise ValueError(
                f"OOF file has unexpected columns: {path}"
            )
        return None

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch: {path}"
        )

    if not np.array_equal(
        df[
            "row_index"
        ].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            f"OOF row order mismatch: {path}"
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF folds mismatch: {path}"
        )

    if (
        id_col is not None
        and
        id_col in df.columns
    ):
        if not np.array_equal(
            df[
                id_col
            ].to_numpy(),
            train[
                id_col
            ].to_numpy(),
        ):
            raise ValueError(
                f"OOF ID mismatch: {path}"
            )

    return df[
        prediction_column
    ].to_numpy(
        dtype=np.float64
    )


def safe_rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def build_model() -> xgb.XGBClassifier:
    """
    Deliberately sensible baseline, not tuned XGBoost.

    We want to measure model-family behavior before hyperparameter search.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",

        n_estimators=5000,
        learning_rate=0.03,

        max_depth=6,
        min_child_weight=8.0,

        subsample=0.90,
        colsample_bytree=0.90,

        reg_lambda=2.0,
        reg_alpha=0.0,
        gamma=0.0,

        max_bin=256,

        tree_method="hist",
        device="cuda",

        enable_categorical=True,
        max_cat_to_onehot=8,

        early_stopping_rounds=200,

        random_state=SEED,

        n_jobs=-1,

        importance_type="gain",
    )


def main() -> None:
    args = parse_args()

    train_path = (
        args.data_dir
        / "train.csv"
    )

    test_path = (
        args.data_dir
        / "test.csv"
    )

    for path in [
        train_path,
        test_path,
        args.folds_path,
        args.catboost_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading competition data and frozen folds..."
    )

    print(
        f"XGBoost version: {xgb.__version__}"
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    folds_df = pd.read_csv(
        args.folds_path
    )

    target = detect_target(
        train,
        test,
    )

    y, positive_label = (
        encode_binary_target(
            train[
                target
            ]
        )
    )

    fold_ids, id_col = (
        validate_folds(
            folds_df,
            train,
        )
    )

    features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and
            c != id_col
        )
    ]

    categorical_features = (
        detect_categorical_features(
            train,
            features,
        )
    )

    numeric_features = [
        c
        for c in features
        if c
        not in categorical_features
    ]

    X, X_test = (
        prepare_xgboost_frames(
            train=train,
            test=test,
            features=features,
            categorical_features=categorical_features,
        )
    )

    catboost_oof = load_oof(
        path=args.catboost_oof,
        prediction_column="oof_prediction",
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    logistic_oof = load_oof(
        path=args.logistic_oof,
        prediction_column="oof_prediction",
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=False,
    )

    catboost_auc = float(
        roc_auc_score(
            y,
            catboost_oof,
        )
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"ID excluded from model: {id_col!r}"
    )

    print(
        f"Model features: {len(features)}"
    )

    print(
        f"  numeric: {len(numeric_features)}"
    )

    print(
        "  native categorical/binary: "
        f"{len(categorical_features)}"
    )

    print()
    print(
        f"Current CatBoost GPU OOF AUC: "
        f"{catboost_auc:.8f}"
    )

    print()
    print(
        "XGBoost GPU baseline parameters:"
    )

    print(
        "  tree_method='hist'"
    )

    print(
        "  device='cuda'"
    )

    print(
        "  enable_categorical=True"
    )

    print(
        "  n_estimators=5000"
    )

    print(
        "  learning_rate=0.03"
    )

    print(
        "  max_depth=6"
    )

    print(
        "  min_child_weight=8"
    )

    print(
        "  subsample=0.90"
    )

    print(
        "  colsample_bytree=0.90"
    )

    print(
        "  early_stopping_rounds=200"
    )

    print()
    print(
        "Running frozen 5-fold XGBoost GPU CV..."
    )
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []

    metric_rows = []
    importance_rows = []

    total_start = (
        time.perf_counter()
    )

    for fold in range(5):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        model = build_model()

        fit_start = (
            time.perf_counter()
        )

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
            eval_set=[
                (
                    X.iloc[
                        valid_idx
                    ],
                    y[
                        valid_idx
                    ],
                )
            ],
            verbose=100,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        infer_start = (
            time.perf_counter()
        )

        valid_pred = (
            model.predict_proba(
                X.iloc[
                    valid_idx
                ]
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        inference_seconds = (
            time.perf_counter()
            - infer_start
        )

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                valid_pred,
            )
        )

        catboost_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                catboost_oof[
                    valid_idx
                ],
            )
        )

        fold_delta_vs_catboost = (
            fold_auc
            - catboost_fold_auc
        )

        best_iteration = int(
            model.best_iteration
        )

        best_score = float(
            model.best_score
        )

        metric_rows.append(
            {
                "fold": fold,

                "train_rows": len(
                    train_idx
                ),

                "valid_rows": len(
                    valid_idx
                ),

                "positive_rate_valid": float(
                    y[
                        valid_idx
                    ].mean()
                ),

                "xgboost_auc": (
                    fold_auc
                ),

                "catboost_gpu_auc": (
                    catboost_fold_auc
                ),

                "delta_vs_catboost": (
                    fold_delta_vs_catboost
                ),

                "best_iteration_zero_based": (
                    best_iteration
                ),

                "best_validation_auc_reported": (
                    best_score
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "inference_seconds_valid_plus_test": (
                    float(
                        inference_seconds
                    )
                ),
            }
        )

        importances = (
            model.feature_importances_
        )

        for (
            feature,
            importance,
        ) in zip(
            features,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print()
        print(
            f"Fold {fold}: "
            f"XGB={fold_auc:.6f} | "
            f"CatBoost={catboost_fold_auc:.6f} | "
            f"delta={fold_delta_vs_catboost:+.6f} | "
            f"best_iter={best_iteration} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del model

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "XGBoost OOF predictions contain missing values."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    xgb_oof_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_catboost = (
        xgb_oof_auc
        - catboost_auc
    )

    mean_fold_auc = float(
        fold_metrics[
            "xgboost_auc"
        ].mean()
    )

    std_fold_auc = float(
        fold_metrics[
            "xgboost_auc"
        ].std(
            ddof=1
        )
    )

    folds_beating_catboost = int(
        (
            fold_metrics[
                "delta_vs_catboost"
            ] > 0
        ).sum()
    )

    catboost_probability_corr = float(
        np.corrcoef(
            oof,
            catboost_oof,
        )[0, 1]
    )

    catboost_rank_corr = (
        safe_rank_correlation(
            oof,
            catboost_oof,
        )
    )

    logistic_auc = None
    logistic_probability_corr = None
    logistic_rank_corr = None

    if logistic_oof is not None:
        logistic_auc = float(
            roc_auc_score(
                y,
                logistic_oof,
            )
        )

        logistic_probability_corr = float(
            np.corrcoef(
                oof,
                logistic_oof,
            )[0, 1]
        )

        logistic_rank_corr = (
            safe_rank_correlation(
                oof,
                logistic_oof,
            )
        )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    # ---------------------------------------
    # Importance summary
    # ---------------------------------------

    importance_by_fold = (
        pd.DataFrame(
            importance_rows
        )
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_gain_importance=(
                "gain_importance",
                "mean",
            ),
            std_gain_importance=(
                "gain_importance",
                "std",
            ),
        )
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------
    # Save artifacts
    # ---------------------------------------

    fold_metrics.to_csv(
        args.output_dir
        / "fold_metrics.csv",
        index=False,
    )

    importance_summary.to_csv(
        args.output_dir
        / "feature_importance.csv",
        index=False,
    )

    importance_by_fold.to_csv(
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

            "fold": (
                fold_ids
            ),

            "target_encoded": y,

            "oof_prediction": (
                oof.astype(
                    np.float32
                )
            ),
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    oof_output.to_csv(
        args.output_dir
        / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": (
                test_prediction.astype(
                    np.float32
                )
            )
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[
                id_col
            ].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    summary_lines = [
        "EXPERIMENT: XGBOOST_RAW_GPU_BASELINE",
        "=" * 76,
        "",
        "QUESTION",
        "Can XGBoost capture useful structure that differs from CatBoost",
        "while using the same raw features and frozen folds?",
        "",
        "VALIDATION",
        "Same frozen 5-fold competition split.",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "FEATURE POLICY",
        f"Excluded ID: {id_col!r}",
        f"Raw model features: {len(features)}",
        f"Numeric features: {numeric_features}",
        f"Native categorical/binary features: {categorical_features}",
        "",
        "MODEL",
        f"XGBoost version: {xgb.__version__}",
        "XGBClassifier",
        "tree_method='hist'",
        "device='cuda'",
        "enable_categorical=True",
        "n_estimators=5000",
        "learning_rate=0.03",
        "max_depth=6",
        "min_child_weight=8",
        "subsample=0.90",
        "colsample_bytree=0.90",
        "reg_lambda=2.0",
        "early_stopping_rounds=200",
        "",
        "RESULTS",
        f"XGBoost OOF AUC: {xgb_oof_auc:.8f}",
        f"Mean fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {std_fold_auc:.8f}",
        f"CatBoost GPU OOF AUC: {catboost_auc:.8f}",
        f"Delta vs CatBoost: {delta_vs_catboost:+.8f}",
        f"Folds beating CatBoost: {folds_beating_catboost}/5",
        f"OOF probability correlation vs CatBoost: "
        f"{catboost_probability_corr:.6f}",
        f"OOF rank correlation vs CatBoost: "
        f"{catboost_rank_corr:.6f}",
        f"Total runtime: {total_seconds:.2f} seconds",
    ]

    if logistic_auc is not None:
        summary_lines.extend(
            [
                "",
                "DIVERSITY VS LOGISTIC",
                f"Logistic OOF AUC: {logistic_auc:.8f}",
                f"OOF probability correlation vs Logistic: "
                f"{logistic_probability_corr:.6f}",
                f"OOF rank correlation vs Logistic: "
                f"{logistic_rank_corr:.6f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "TOP MEAN GAIN IMPORTANCES",
        ]
    )

    for row in (
        importance_summary
        .head(10)
        .itertuples()
    ):
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "Do not judge XGBoost only by whether it beats CatBoost.",
            "If its OOF predictions are meaningfully different while its AUC",
            "is competitive, it may be useful later for blending.",
            "",
            "This is a baseline, not tuned XGBoost.",
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
        "XGBOOST RAW GPU BASELINE COMPLETE"
    )
    print("=" * 78)

    print(
        f"XGBoost OOF    : "
        f"{xgb_oof_auc:.8f}"
    )

    print(
        f"CatBoost OOF   : "
        f"{catboost_auc:.8f}"
    )

    print(
        f"Delta          : "
        f"{delta_vs_catboost:+.8f}"
    )

    print(
        f"Folds won      : "
        f"{folds_beating_catboost}/5"
    )

    print(
        f"OOF rank corr  : "
        f"{catboost_rank_corr:.6f}"
    )

    print(
        f"Total runtime  : "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Artifacts      : "
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
