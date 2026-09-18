"""
catboost_hierarchical_income_buckets_gpu.py

Kaggle Playground Series S6E9

Controlled structural experiment:
Add target-free hierarchical Annual_Income_USD categorical bucket copies to the
validated seed-42 CatBoost exact-value-ID model.

HYPOTHESIS
----------
Hierarchical income structure improved BOTH:
- XGBoost: +0.00004290, 5/5 folds
- LightGBM: +0.00004660, 5/5 folds

CatBoost's strongest breakthrough also came from treating exact numeric values as
categorical identities. Therefore the next clean question is whether CatBoost can
benefit from seeing income identity at multiple resolutions as categorical tokens.

ONLY CHANGE
-----------
Add three NEW categorical string features:

    CAT__income_1k_bucket
    CAT__income_10k_bucket
    CAT__income_100k_bucket

where:
    bucket = floor(round(Annual_Income_USD) / scale)

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw 13 features
- original categorical treatment
- raw numeric Annual_Income_USD retained
- raw numeric Daily_Commute_km retained
- exact string identity copies:
      Annual_Income_USD_id
      Daily_Commute_km_id
- CatBoost iterations=2000
- learning_rate=0.05
- depth=6
- loss_function="Logloss"
- eval_metric="AUC"
- random_seed=42
- task_type="GPU"
- devices="0"
- early_stopping_rounds=150
- all unspecified CatBoost parameters remain library defaults
- no public leaderboard optimization

BASELINE
--------
Existing validated seed-42 exact-value-ID CatBoost:
    OOF AUC = 0.94536428

Existing artifact:
    artifacts/experiments/catboost_exact_value_ids_gpu/oof_predictions.csv

Run:
    python catboost_hierarchical_income_buckets_gpu.py
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
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise SystemExit(
        "\nCatBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U catboost\n"
    ) from exc


SEED = 42

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94536428
AUC_CHECK_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_buckets_gpu"
)

NEW_BUCKET_SPECS = [
    ("CAT__income_1k_bucket", 1_000),
    ("CAT__income_10k_bucket", 10_000),
    ("CAT__income_100k_bucket", 100_000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CatBoost exact-value IDs + hierarchical income categorical buckets."
        )
    )

    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    p.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
    )

    p.add_argument(
        "--baseline-oof",
        type=Path,
        default=DEFAULT_BASELINE_OOF,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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
            f"Expected exactly one train-only target column, found {train_only}"
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
            f"Expected binary target, found {values}"
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

    return (
        (y == positive)
        .astype(np.int8)
        .to_numpy(),
        positive,
    )


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
        - set(folds.columns)
    )

    if missing:
        raise ValueError(
            f"Fold file missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen fold row count does not match train.csv."
        )

    expected = np.arange(
        len(train),
        dtype=np.int64,
    )

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        expected,
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    if sorted(
        folds["fold"].unique().tolist()
    ) != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected frozen folds [0,1,2,3,4]."
        )

    extras = [
        c
        for c in folds.columns
        if c not in {
            "row_index",
            "fold",
        }
    ]

    if len(extras) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extras}"
        )

    id_col = (
        extras[0]
        if extras
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} missing from train.csv."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                "Frozen fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(
            dtype=np.int16
        ),
        id_col,
    )


def detect_raw_categoricals(
    train: pd.DataFrame,
    raw_features: list[str],
) -> list[str]:
    out = []

    for c in raw_features:
        dtype = train[c].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(
                dtype,
                pd.CategoricalDtype,
            )
        ):
            out.append(c)

    return out


def exact_income_id(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    values = np.rint(
        numeric
    ).astype(np.int64)

    return pd.Series(
        values.astype(str),
        index=series.index,
        dtype="object",
    )


def exact_commute_id(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    return pd.Series(
        [
            f"{v:.1f}"
            for v in numeric
        ],
        index=series.index,
        dtype="object",
    )


def add_exact_value_ids(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[
        "Annual_Income_USD_id"
    ] = exact_income_id(
        train_source[
            "Annual_Income_USD"
        ]
    )

    X_test[
        "Annual_Income_USD_id"
    ] = exact_income_id(
        test_source[
            "Annual_Income_USD"
        ]
    )

    X_train[
        "Daily_Commute_km_id"
    ] = exact_commute_id(
        train_source[
            "Daily_Commute_km"
        ]
    )

    X_test[
        "Daily_Commute_km_id"
    ] = exact_commute_id(
        test_source[
            "Daily_Commute_km"
        ]
    )

    return (
        X_train,
        X_test,
        [
            "Annual_Income_USD_id",
            "Daily_Commute_km_id",
        ],
    )


def add_hierarchical_income_categories(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_income = np.rint(
        pd.to_numeric(
            train_source[
                "Annual_Income_USD"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )
    ).astype(np.int64)

    test_income = np.rint(
        pd.to_numeric(
            test_source[
                "Annual_Income_USD"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )
    ).astype(np.int64)

    if (
        (train_income < 0).any()
        or (test_income < 0).any()
    ):
        raise ValueError(
            "Annual_Income_USD unexpectedly contains negative values."
        )

    added = []

    for name, divisor in NEW_BUCKET_SPECS:
        X_train[
            name
        ] = (
            train_income
            // divisor
        ).astype(str)

        X_test[
            name
        ] = (
            test_income
            // divisor
        ).astype(str)

        added.append(name)

    return (
        X_train,
        X_test,
        added,
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"OOF file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(
        path
    )

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch in {path}"
        )

    if "row_index" not in df.columns:
        raise ValueError(
            f"OOF row_index missing in {path}"
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
            f"OOF row order mismatch in {path}"
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"OOF fold column missing in {path}"
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF fold mismatch in {path}"
        )

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(
            df[
                id_col
            ].to_numpy(),
            train[
                id_col
            ].to_numpy(),
        )
    ):
        raise ValueError(
            f"OOF ID mismatch in {path}"
        )

    candidate_columns = [
        "oof_prediction",
        "prediction",
    ]

    pred_col = None

    for c in candidate_columns:
        if c in df.columns:
            pred_col = c
            break

    if pred_col is None:
        excluded = {
            "row_index",
            "fold",
            "target",
            "target_encoded",
            "id",
        }

        numeric = [
            c
            for c in df.columns
            if (
                c not in excluded
                and pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )
        ]

        if len(numeric) == 1:
            pred_col = numeric[0]
        else:
            raise ValueError(
                "Could not safely identify baseline OOF prediction column.\n"
                f"File: {path.resolve()}\n"
                f"Columns: {list(df.columns)}"
            )

    pred = df[
        pred_col
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "Baseline OOF contains non-finite predictions."
        )

    return pred


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
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

    required = [
        train_path,
        test_path,
        args.folds_path,
        args.baseline_oof,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = sha256_file(
        args.folds_path
    )

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}\n"
            f"File    : {args.folds_path.resolve()}"
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

    baseline_oof = load_oof(
        args.baseline_oof,
        train,
        fold_ids,
        id_col,
    )

    baseline_auc = float(
        roc_auc_score(
            y,
            baseline_oof,
        )
    )

    if abs(
        baseline_auc
        - EXPECTED_BASELINE_AUC
    ) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            "Loaded exact-value CatBoost baseline does not match "
            "the documented seed-42 result.\n"
            f"Expected approximately: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Loaded artifact AUC   : {baseline_auc:.8f}\n"
            f"File                  : {args.baseline_oof.resolve()}"
        )

    raw_features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and c != id_col
        )
    ]

    raw_categoricals = (
        detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    X_train = train[
        raw_features
    ].copy()

    X_test = test[
        raw_features
    ].copy()

    # CatBoost categorical columns must have no NaN and consistent string type.
    for c in raw_categoricals:
        X_train[c] = (
            X_train[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        X_test[c] = (
            X_test[c]
            .fillna("__MISSING__")
            .astype(str)
        )

    (
        X_train,
        X_test,
        exact_id_columns,
    ) = add_exact_value_ids(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    (
        X_train,
        X_test,
        bucket_columns,
    ) = add_hierarchical_income_categories(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    categorical_columns = (
        raw_categoricals
        + exact_id_columns
        + bucket_columns
    )

    if len(set(categorical_columns)) != len(categorical_columns):
        raise ValueError(
            "Duplicate categorical feature names detected."
        )

    print("=" * 94)
    print("CATBOOST + HIERARCHICAL INCOME CATEGORICAL BUCKETS")
    print("=" * 94)
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Exact-value CatBoost seed-42 baseline: {baseline_auc:.8f}")
    print()
    print("HYPOTHESIS:")
    print(
        "  Hierarchical income structure improved XGBoost and LightGBM on "
        "all 5 folds. CatBoost may exploit the same structure naturally if "
        "income buckets are exposed as categorical identities."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add target-free categorical copies for $1k, $10k and $100k "
        "Annual_Income_USD buckets."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds + SHA256")
    print("  - raw 13 features")
    print("  - raw income + commute remain numeric")
    print("  - exact income + commute string IDs")
    print("  - iterations=2000")
    print("  - learning_rate=0.05")
    print("  - depth=6")
    print("  - loss_function=Logloss")
    print("  - eval_metric=AUC")
    print("  - random_seed=42")
    print("  - GPU device 0")
    print("  - early_stopping_rounds=150")
    print()
    print(f"Total model features : {X_train.shape[1]}")
    print(f"Categorical features : {len(categorical_columns)}")
    print("New categorical features:")
    for c in bucket_columns:
        print(f"  - {c}")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    fold_rows = []
    importance_rows = []

    total_start = time.perf_counter()

    for fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        model = build_model()

        model.fit(
            X_train.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
            cat_features=categorical_columns,
            eval_set=(
                X_train.iloc[
                    valid_idx
                ],
                y[
                    valid_idx
                ],
            ),
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=100,
        )

        valid_pred = model.predict_proba(
            X_train.iloc[
                valid_idx
            ]
        )[:, 1]

        test_pred = model.predict_proba(
            X_test
        )[:, 1]

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                baseline_oof[
                    valid_idx
                ],
            )
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                valid_pred,
            )
        )

        delta = (
            candidate_fold_auc
            - baseline_fold_auc
        )

        best_iteration = int(
            model.get_best_iteration()
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": fold,
                "baseline_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": best_iteration,
                "total_fold_seconds": fold_seconds,
            }
        )

        importances = (
            model.get_feature_importance()
        )

        for feature, importance in zip(
            X_train.columns,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "is_new_income_bucket": (
                        feature
                        in bucket_columns
                    ),
                    "is_exact_value_id": (
                        feature
                        in exact_id_columns
                    ),
                    "importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {fold}: "
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={best_iteration}"
        )

        del model

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "Candidate OOF contains NaNs."
        )

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_baseline = (
        candidate_auc
        - baseline_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    folds_improved = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] < 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    baseline_rank = (
        pd.Series(
            baseline_oof
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    candidate_rank = (
        pd.Series(
            oof
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    rank_correlation = float(
        np.corrcoef(
            baseline_rank,
            candidate_rank,
        )[0, 1]
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    importance_by_fold = pd.DataFrame(
        importance_rows
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            [
                "feature",
                "is_new_income_bucket",
                "is_exact_value_id",
            ],
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
        .reset_index(
            drop=True
        )
    )

    bucket_importance = (
        importance_summary[
            importance_summary[
                "is_new_income_bucket"
            ]
        ]
        .copy()
        .sort_values(
            "mean_importance",
            ascending=False,
        )
    )

    if (
        delta_vs_baseline > 0
        and folds_improved >= 3
    ):
        decision = (
            "KEEP_HIERARCHICAL_INCOME_BUCKETS_FOR_CATBOOST"
        )
    else:
        decision = (
            "REJECT_HIERARCHICAL_INCOME_BUCKETS_FOR_CATBOOST"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    bucket_importance.to_csv(
        args.output_dir
        / "hierarchical_income_bucket_importance.csv",
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

    summary_lines = [
        "EXPERIMENT: CATBOOST + HIERARCHICAL INCOME CATEGORICAL BUCKETS",
        "=" * 84,
        "",
        "HYPOTHESIS",
        "Does the hierarchical income representation that improved XGBoost and",
        "LightGBM also improve CatBoost when exposed as categorical identities?",
        "",
        "ONLY CHANGE",
        "Add target-free income bucket categories at $1k / $10k / $100k scales.",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- raw 13 features",
        "- raw numeric income + commute",
        "- exact income + commute categorical IDs",
        "- iterations=2000",
        "- learning_rate=0.05",
        "- depth=6",
        "- loss_function=Logloss",
        "- eval_metric=AUC",
        "- random_seed=42",
        "- GPU device 0",
        "- early stopping 150",
        "",
        "RESULTS",
        f"Seed-42 exact-value baseline OOF: {baseline_auc:.8f}",
        f"Hierarchical-bucket candidate OOF: {candidate_auc:.8f}",
        f"Delta vs baseline: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs baseline: {probability_corr:.6f}",
        f"Rank corr vs baseline: {rank_correlation:.6f}",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"baseline={row.baseline_auc:.8f} -> "
            f"candidate={row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "HIERARCHICAL INCOME BUCKET IMPORTANCE",
        ]
    )

    for row in bucket_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_importance:.8f}"
        )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("CATBOOST HIERARCHICAL-INCOME EXPERIMENT COMPLETE")
    print("=" * 94)
    print(f"Exact-value seed42 baseline : {baseline_auc:.8f}")
    print(f"Candidate                   : {candidate_auc:.8f}")
    print(f"Delta                       : {delta_vs_baseline:+.8f}")
    print(f"Folds improved              : {folds_improved}/5")
    print(f"Folds worse                 : {folds_worse}/5")
    print(f"Probability corr            : {probability_corr:.6f}")
    print(f"Rank corr                   : {rank_correlation:.6f}")
    print(f"Decision                    : {decision}")
    print(f"Runtime                     : {total_seconds:.2f}s")
    print(f"Artifacts                   : {args.output_dir.resolve()}")
    print()
    print("Hierarchical income bucket importances:")
    if len(
        bucket_importance
    ):
        print(
            bucket_importance[
                [
                    "feature",
                    "mean_importance",
                ]
            ].to_string(
                index=False
            )
        )
    print("=" * 94)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. hierarchical_income_bucket_importance.csv")


if __name__ == "__main__":
    main()
