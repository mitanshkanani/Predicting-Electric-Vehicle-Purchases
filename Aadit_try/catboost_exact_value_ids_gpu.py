"""
catboost_exact_value_ids_gpu.py

Phase 5C — Exact numeric value identity experiment.

HYPOTHESIS
----------
The exact values of:
    Annual_Income_USD
    Daily_Commute_km

may contain synthetic-generator identity signal that ordinary numeric tree
splits cannot fully exploit.

Instead of replacing the raw numeric columns, this experiment ADDS string
copies:

    Annual_Income_USD_id
    Daily_Commute_km_id

and lets CatBoost treat those exact values as categorical features.

WHY CATBOOST
------------
CatBoost can build ordered target statistics / CTR features from categorical
variables without us manually leaking the target.

CONTROLLED EXPERIMENT
---------------------
Compared with baseline_catboost_gpu.py:

SAME:
- exact frozen 5 folds
- original 13 raw features
- original feature types
- GPU training
- iterations=2000
- learning_rate=0.05
- depth=6
- early_stopping_rounds=150
- seed=42

ONLY CHANGE:
- add two exact-value categorical identity columns

This does NOT use the external/source dataset.
This does NOT create a Kaggle submission.

Run:
    python catboost_exact_value_ids_gpu.py
"""

from __future__ import annotations

import argparse
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
    ) from exc


SEED = 42

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_CATBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_XGBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test exact numeric value identities with CatBoost GPU."
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
        "--xgboost-oof",
        type=Path,
        default=DEFAULT_XGBOOST_OOF,
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
            "Expected exactly one train-only target column, "
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

    encoded = (
        y == positive
    ).astype(
        np.int8
    ).to_numpy()

    return encoded, positive


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
        folds[
            "row_index"
        ].to_numpy(),
        expected,
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

    extras = [
        c
        for c in folds.columns
        if c
        not in {
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


def detect_raw_categoricals(
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


def stable_exact_value_string(
    series: pd.Series,
    feature: str,
) -> pd.Series:
    """
    Convert exact numeric values to deterministic category strings.

    Annual income is integer-like in this competition.
    Daily commute is recorded to one decimal place.

    We format explicitly so train/test use identical textual keys.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{feature!r} unexpectedly contains missing/non-numeric values."
        )

    if feature == "Annual_Income_USD":
        rounded = np.rint(
            numeric.to_numpy(
                dtype=np.float64
            )
        ).astype(
            np.int64
        )

        return pd.Series(
            rounded.astype(str),
            index=series.index,
            name=f"{feature}_id",
        )

    if feature == "Daily_Commute_km":
        values = numeric.to_numpy(
            dtype=np.float64
        )

        return pd.Series(
            [
                f"{value:.1f}"
                for value in values
            ],
            index=series.index,
            name=f"{feature}_id",
        )

    # Generic fallback.
    return (
        numeric
        .map(
            lambda x: format(
                float(x),
                ".12g",
            )
        )
        .rename(
            f"{feature}_id"
        )
    )


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_features: list[str],
    raw_categoricals: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Preserve all raw features, then append the two exact-value ID features.
    """
    x_train = train[
        raw_features
    ].copy()

    x_test = test[
        raw_features
    ].copy()

    # Normalize existing CatBoost categorical columns.
    for c in raw_categoricals:
        x_train[
            c
        ] = (
            x_train[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        x_test[
            c
        ] = (
            x_test[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    value_id_columns = []

    for feature in VALUE_ID_FEATURES:
        if feature not in x_train.columns:
            raise ValueError(
                f"Required value-ID feature {feature!r} is missing."
            )

        id_col = (
            f"{feature}_id"
        )

        x_train[
            id_col
        ] = stable_exact_value_string(
            train[
                feature
            ],
            feature,
        )

        x_test[
            id_col
        ] = stable_exact_value_string(
            test[
                feature
            ],
            feature,
        )

        value_id_columns.append(
            id_col
        )

    categorical_features = (
        raw_categoricals
        + value_id_columns
    )

    return (
        x_train,
        x_test,
        categorical_features,
        value_id_columns,
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    required: bool,
) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required OOF file not found:\n{path.resolve()}"
            )

        return None

    df = pd.read_csv(
        path
    )

    required_cols = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required_cols.issubset(
        df.columns
    ):
        raise ValueError(
            f"Unexpected OOF columns in {path}"
        )

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch in {path}"
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
                f"OOF IDs mismatch in {path}"
            )

    return df[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )


def rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a_rank = (
        pd.Series(
            a
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    b_rank = (
        pd.Series(
            b
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            a_rank,
            b_rank,
        )[0, 1]
    )


def build_model() -> CatBoostClassifier:
    """
    Exact same settings as baseline_catboost_gpu.py.
    """
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=2000,
        learning_rate=0.05,
        depth=6,

        random_seed=SEED,

        task_type="GPU",
        devices="0",

        allow_writing_files=False,
    )


def exact_value_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in VALUE_ID_FEATURES:
        train_key = stable_exact_value_string(
            train[
                feature
            ],
            feature,
        )

        test_key = stable_exact_value_string(
            test[
                feature
            ],
            feature,
        )

        train_counts = train_key.value_counts()

        train_unique = int(
            train_key.nunique()
        )

        test_unique = int(
            test_key.nunique()
        )

        test_seen_rate = float(
            test_key.isin(
                set(
                    train_counts.index
                )
            ).mean()
        )

        rows.append(
            {
                "feature": feature,
                "train_unique_values": train_unique,
                "test_unique_values": test_unique,
                "mean_train_rows_per_value": float(
                    len(train)
                    / train_unique
                ),
                "median_train_rows_per_value": float(
                    train_counts.median()
                ),
                "min_train_rows_per_value": int(
                    train_counts.min()
                ),
                "max_train_rows_per_value": int(
                    train_counts.max()
                ),
                "test_values_seen_in_train_rate": test_seen_rate,
            }
        )

    return pd.DataFrame(
        rows
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
        args.xgboost_oof,
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
        "Loading competition data, frozen folds, and baseline OOF predictions..."
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

    raw_features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and
            c != id_col
        )
    ]

    raw_categoricals = (
        detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    (
        X,
        X_test,
        categorical_features,
        value_id_columns,
    ) = prepare_frames(
        train=train,
        test=test,
        raw_features=raw_features,
        raw_categoricals=raw_categoricals,
    )

    catboost_baseline_oof = load_oof(
        path=args.catboost_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    xgboost_oof = load_oof(
        path=args.xgboost_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    catboost_baseline_auc = float(
        roc_auc_score(
            y,
            catboost_baseline_oof,
        )
    )

    xgboost_auc = float(
        roc_auc_score(
            y,
            xgboost_oof,
        )
    )

    diagnostics = exact_value_diagnostics(
        train,
        test,
    )

    diagnostics.to_csv(
        args.output_dir
        / "value_identity_diagnostics.csv",
        index=False,
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Raw CatBoost GPU baseline: "
        f"{catboost_baseline_auc:.8f}"
    )

    print(
        f"Current best XGBoost depth-4: "
        f"{xgboost_auc:.8f}"
    )

    print()
    print(
        "Added exact-value categorical columns:"
    )

    for c in value_id_columns:
        print(
            f"  - {c}"
        )

    print()
    print(
        "Value identity diagnostics:"
    )

    for row in diagnostics.itertuples():
        print(
            f"  {row.feature}: "
            f"train_unique={row.train_unique_values:,} | "
            f"mean_rows/value={row.mean_train_rows_per_value:.1f} | "
            f"median_rows/value={row.median_train_rows_per_value:.1f} | "
            f"test_seen_in_train={row.test_values_seen_in_train_rate:.4f}"
        )

    print()
    print(
        f"Total model features: {len(X.columns)} "
        f"({len(raw_features)} raw + {len(value_id_columns)} value-ID)"
    )

    print(
        f"Total CatBoost categorical features: "
        f"{len(categorical_features)}"
    )

    print()
    print(
        "Running frozen 5-fold CatBoost GPU exact-value-ID CV..."
    )
    print()

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=list(
            X.columns
        ),
    )

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

        train_pool = Pool(
            X.iloc[
                train_idx
            ],
            label=y[
                train_idx
            ],
            cat_features=categorical_features,
            feature_names=list(
                X.columns
            ),
        )

        valid_pool = Pool(
            X.iloc[
                valid_idx
            ],
            label=y[
                valid_idx
            ],
            cat_features=categorical_features,
            feature_names=list(
                X.columns
            ),
        )

        model = build_model()

        fit_start = (
            time.perf_counter()
        )

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

        inference_start = (
            time.perf_counter()
        )

        valid_pred = (
            model.predict_proba(
                valid_pool
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                test_pool
            )[:, 1]
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
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

        cat_baseline_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                catboost_baseline_oof[
                    valid_idx
                ],
            )
        )

        xgb_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                xgboost_oof[
                    valid_idx
                ],
            )
        )

        importances = (
            model.get_feature_importance(
                type="PredictionValuesChange"
            )
        )

        for feature, importance in zip(
            X.columns,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "is_value_identity": (
                        feature
                        in value_id_columns
                    ),
                    "importance": float(
                        importance
                    ),
                }
            )

        metric_rows.append(
            {
                "fold": fold,

                "catboost_raw_gpu_auc": (
                    cat_baseline_fold_auc
                ),

                "xgboost_depth4_auc": (
                    xgb_fold_auc
                ),

                "catboost_value_id_auc": (
                    fold_auc
                ),

                "delta_vs_catboost_raw": (
                    fold_auc
                    - cat_baseline_fold_auc
                ),

                "delta_vs_xgboost_depth4": (
                    fold_auc
                    - xgb_fold_auc
                ),

                "best_iteration_zero_based": int(
                    model.get_best_iteration()
                ),

                "tree_count": int(
                    model.tree_count_
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "inference_seconds_valid_plus_test": float(
                    inference_seconds
                ),
            }
        )

        print()
        print(
            f"Fold {fold}: "
            f"raw_CAT={cat_baseline_fold_auc:.6f} | "
            f"valueID_CAT={fold_auc:.6f} | "
            f"delta_CAT={fold_auc - cat_baseline_fold_auc:+.6f} | "
            f"XGB4={xgb_fold_auc:.6f} | "
            f"delta_XGB={fold_auc - xgb_fold_auc:+.6f} | "
            f"best_iter={model.get_best_iteration()} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            train_pool,
            valid_pool,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "Value-ID CatBoost OOF contains NaN predictions."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    value_id_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_raw_cat = (
        value_id_auc
        - catboost_baseline_auc
    )

    delta_vs_xgb = (
        value_id_auc
        - xgboost_auc
    )

    folds_beating_raw_cat = int(
        (
            fold_metrics[
                "delta_vs_catboost_raw"
            ] > 0
        ).sum()
    )

    folds_beating_xgb = int(
        (
            fold_metrics[
                "delta_vs_xgboost_depth4"
            ] > 0
        ).sum()
    )

    rank_corr_vs_xgb = (
        rank_correlation(
            oof,
            xgboost_oof,
        )
    )

    probability_corr_vs_xgb = float(
        np.corrcoef(
            oof,
            xgboost_oof,
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
                "is_value_identity",
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

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

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
            "prediction": test_prediction.astype(
                np.float32
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

    decision = (
        "MAJOR_BREAKTHROUGH"
        if (
            delta_vs_xgb
            >= 0.001
            and
            folds_beating_xgb
            >= 3
        )
        else
        "KEEP"
        if (
            delta_vs_raw_cat
            > 0
            and
            folds_beating_raw_cat
            >= 3
        )
        else
        "REJECT_FOR_NOW"
    )

    summary_lines = [
        "EXPERIMENT: CATBOOST_GPU_EXACT_VALUE_IDENTITIES",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Exact Annual_Income_USD and Daily_Commute_km values may behave",
        "as synthetic-generator identities beyond their numeric magnitude.",
        "",
        "CONTROL",
        "Same raw CatBoost GPU baseline configuration.",
        "Only two exact-value categorical copies were added.",
        f"Value-ID columns: {value_id_columns}",
        "",
        "RESULTS",
        f"Raw CatBoost GPU OOF AUC: {catboost_baseline_auc:.8f}",
        f"Value-ID CatBoost OOF AUC: {value_id_auc:.8f}",
        f"Delta vs raw CatBoost: {delta_vs_raw_cat:+.8f}",
        f"Raw depth-4 XGBoost OOF AUC: {xgboost_auc:.8f}",
        f"Delta vs depth-4 XGBoost: {delta_vs_xgb:+.8f}",
        f"Folds beating raw CatBoost: {folds_beating_raw_cat}/5",
        f"Folds beating depth-4 XGBoost: {folds_beating_xgb}/5",
        f"OOF probability correlation vs XGBoost: "
        f"{probability_corr_vs_xgb:.6f}",
        f"OOF rank correlation vs XGBoost: "
        f"{rank_corr_vs_xgb:.6f}",
        f"Total runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "VALUE-ID DIAGNOSTICS",
    ]

    for row in diagnostics.itertuples():
        summary_lines.append(
            f"- {row.feature}: "
            f"train_unique={row.train_unique_values}, "
            f"mean_rows_per_value={row.mean_train_rows_per_value:.2f}, "
            f"median_rows_per_value={row.median_train_rows_per_value:.2f}, "
            f"test_seen_in_train={row.test_values_seen_in_train_rate:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "FOLD DELTAS VS RAW CATBOOST",
        ]
    )

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.catboost_raw_gpu_auc:.8f} -> "
            f"{row.catboost_value_id_auc:.8f} "
            f"({row.delta_vs_catboost_raw:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "VALUE-ID FEATURE IMPORTANCES",
        ]
    )

    value_id_importance = importance_summary[
        importance_summary[
            "is_value_identity"
        ]
    ]

    for row in value_id_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "This is different from the rejected binned target-encoding run.",
            "Here the exact numeric values themselves are preserved as category",
            "identities and CatBoost learns ordered target statistics internally.",
        ]
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
    print("=" * 78)
    print(
        "CATBOOST EXACT VALUE-ID EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Raw CatBoost OOF : "
        f"{catboost_baseline_auc:.8f}"
    )

    print(
        f"Value-ID CatBoost: "
        f"{value_id_auc:.8f}"
    )

    print(
        f"Delta vs raw CAT : "
        f"{delta_vs_raw_cat:+.8f}"
    )

    print(
        f"Depth-4 XGBoost  : "
        f"{xgboost_auc:.8f}"
    )

    print(
        f"Delta vs XGBoost : "
        f"{delta_vs_xgb:+.8f}"
    )

    print(
        f"XGB folds beaten : "
        f"{folds_beating_xgb}/5"
    )

    print(
        f"Rank corr vs XGB : "
        f"{rank_corr_vs_xgb:.6f}"
    )

    print(
        f"Decision         : "
        f"{decision}"
    )

    print(
        f"Runtime          : "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Artifacts        : "
        f"{args.output_dir.resolve()}"
    )

    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. feature_importance.csv")
    print("  5. value_identity_diagnostics.csv")


if __name__ == "__main__":
    main()
