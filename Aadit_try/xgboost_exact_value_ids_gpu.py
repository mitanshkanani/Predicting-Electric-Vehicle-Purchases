"""
xgboost_exact_value_ids_gpu.py

Phase 5G — Exact-value identity features for XGBoost GPU.

CURRENT LOCAL CHAMPION
----------------------
3-seed rank-averaged CatBoost + exact value identities:
    OOF AUC = 0.94543956

CURRENT XGBOOST REFERENCE
-------------------------
Depth-4 raw XGBoost:
    OOF AUC = 0.94206547

HYPOTHESIS
----------
The exact values of:
    Annual_Income_USD
    Daily_Commute_km

may behave like synthetic-generator identities. CatBoost benefited massively
from categorical copies of these exact values.

Now test the SAME idea in XGBoost:

    Annual_Income_USD      -> keep raw numeric
    Annual_Income_USD_id   -> add exact categorical identity

    Daily_Commute_km       -> keep raw numeric
    Daily_Commute_km_id    -> add exact categorical identity

CONTROLLED EXPERIMENT
---------------------
Compared with our winning depth-4 XGBoost:

SAME:
- exact frozen 5 folds
- original 13 raw features
- original categorical treatment
- GPU training
- n_estimators=5000
- learning_rate=0.03
- max_depth=4
- min_child_weight=8
- subsample=0.90
- colsample_bytree=0.90
- reg_lambda=2
- early_stopping_rounds=200
- random_state=42

ONLY CHANGE:
- add two exact-value categorical identity features

No source data.
No target encoding.
No Kaggle submission.

Run:
    python xgboost_exact_value_ids_gpu.py
"""

from __future__ import annotations

import argparse
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

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_RAW_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_oof_predictions.csv"
)

DEFAULT_CATBOOST_MULTI_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_ids_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test exact-value identities in depth-4 XGBoost."
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
        "--raw-xgb-oof",
        type=Path,
        default=DEFAULT_RAW_XGB_OOF,
    )

    parser.add_argument(
        "--catboost-multiseed-oof",
        type=Path,
        default=DEFAULT_CATBOOST_MULTI_OOF,
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

    if sorted(
        folds[
            "fold"
        ].unique().tolist()
    ) != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected frozen folds [0,1,2,3,4]."
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
                f"Fold ID {id_col!r} missing from train.csv."
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
    out = []

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
            out.append(c)

    return out


def stable_value_id(
    series: pd.Series,
    feature: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{feature!r} contains missing/non-numeric values."
        )

    if feature == "Annual_Income_USD":
        arr = np.rint(
            numeric.to_numpy(
                dtype=np.float64
            )
        ).astype(
            np.int64
        )

        return pd.Series(
            arr.astype(str),
            index=series.index,
            name=f"{feature}_id",
        )

    if feature == "Daily_Commute_km":
        arr = numeric.to_numpy(
            dtype=np.float64
        )

        return pd.Series(
            [
                f"{x:.1f}"
                for x in arr
            ],
            index=series.index,
            name=f"{feature}_id",
        )

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
    Preserve all raw columns and append exact-value categorical IDs.

    All categorical columns receive a shared train+test pandas category
    vocabulary so XGBoost sees consistent category codes.
    """
    x_train = train[
        raw_features
    ].copy()

    x_test = test[
        raw_features
    ].copy()

    value_id_columns = []

    for feature in VALUE_ID_FEATURES:
        id_col = (
            f"{feature}_id"
        )

        x_train[
            id_col
        ] = stable_value_id(
            train[
                feature
            ],
            feature,
        )

        x_test[
            id_col
        ] = stable_value_id(
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
                f"OOF ID mismatch in {path}"
            )

    return (
        df[
            "oof_prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


def rank01(
    values: np.ndarray,
) -> np.ndarray:
    return (
        pd.Series(
            values
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


def rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            rank01(a),
            rank01(b),
        )[0, 1]
    )


def build_model() -> xgb.XGBClassifier:
    """
    Same model settings as the winning raw depth-4 XGBoost experiment.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",

        n_estimators=5000,
        learning_rate=0.03,

        max_depth=4,
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

        # Preserve the same one-hot threshold as our raw XGB baseline.
        # The two new value-ID features are far above this threshold,
        # so XGBoost uses partition-based categorical splits for them.
        max_cat_to_onehot=8,

        early_stopping_rounds=200,

        random_state=SEED,
        n_jobs=-1,

        importance_type="gain",
    )


def value_identity_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in VALUE_ID_FEATURES:
        train_key = stable_value_id(
            train[
                feature
            ],
            feature,
        )

        test_key = stable_value_id(
            test[
                feature
            ],
            feature,
        )

        train_counts = (
            train_key
            .value_counts()
        )

        train_unique = int(
            train_key.nunique()
        )

        test_unique = int(
            test_key.nunique()
        )

        seen_values = set(
            train_counts.index
        )

        test_seen_rate = float(
            test_key.isin(
                seen_values
            ).mean()
        )

        rows.append(
            {
                "feature": feature,

                "train_unique_values": (
                    train_unique
                ),

                "test_unique_values": (
                    test_unique
                ),

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

                "test_values_seen_in_train_rate": (
                    test_seen_rate
                ),
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
        args.raw_xgb_oof,
        args.catboost_multiseed_oof,
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
        "Loading data, frozen folds, and baseline OOF predictions..."
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

    raw_xgb_oof = load_oof(
        path=args.raw_xgb_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    catboost_multi_oof = load_oof(
        path=args.catboost_multiseed_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    raw_xgb_auc = float(
        roc_auc_score(
            y,
            raw_xgb_oof,
        )
    )

    catboost_multi_auc = float(
        roc_auc_score(
            y,
            catboost_multi_oof,
        )
    )

    diagnostics = (
        value_identity_diagnostics(
            train,
            test,
        )
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
        f"Raw depth-4 XGBoost OOF: "
        f"{raw_xgb_auc:.8f}"
    )

    print(
        f"CatBoost 3-seed rank champion OOF: "
        f"{catboost_multi_auc:.8f}"
    )

    print()
    print(
        "Added exact-value categorical features:"
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
        f"({len(raw_features)} raw + "
        f"{len(value_id_columns)} value-ID)"
    )

    print(
        f"Native categorical features: "
        f"{len(categorical_features)}"
    )

    print()
    print(
        "Running frozen 5-fold XGBoost GPU value-ID CV..."
    )
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[
        np.ndarray
    ] = []

    fold_rows = []
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

        raw_xgb_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                raw_xgb_oof[
                    valid_idx
                ],
            )
        )

        catboost_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                catboost_multi_oof[
                    valid_idx
                ],
            )
        )

        fold_rows.append(
            {
                "fold": fold,

                "raw_xgb_depth4_auc": (
                    raw_xgb_fold_auc
                ),

                "value_id_xgb_auc": (
                    fold_auc
                ),

                "delta_vs_raw_xgb": (
                    fold_auc
                    - raw_xgb_fold_auc
                ),

                "catboost_multiseed_auc": (
                    catboost_fold_auc
                ),

                "delta_vs_catboost_multiseed": (
                    fold_auc
                    - catboost_fold_auc
                ),

                "best_iteration_zero_based": int(
                    model.best_iteration
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "inference_seconds_valid_plus_test": float(
                    inference_seconds
                ),
            }
        )

        importances = (
            model.feature_importances_
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

                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print()
        print(
            f"Fold {fold}: "
            f"raw_XGB={raw_xgb_fold_auc:.6f} | "
            f"valueID_XGB={fold_auc:.6f} | "
            f"delta_raw={fold_auc - raw_xgb_fold_auc:+.6f} | "
            f"CAT3={catboost_fold_auc:.6f} | "
            f"delta_CAT3={fold_auc - catboost_fold_auc:+.6f} | "
            f"best_iter={model.best_iteration} | "
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
            "Value-ID XGBoost OOF contains NaN predictions."
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    value_id_xgb_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_raw_xgb = (
        value_id_xgb_auc
        - raw_xgb_auc
    )

    delta_vs_catboost = (
        value_id_xgb_auc
        - catboost_multi_auc
    )

    folds_beating_raw_xgb = int(
        (
            fold_metrics[
                "delta_vs_raw_xgb"
            ] > 0
        ).sum()
    )

    folds_beating_catboost = int(
        (
            fold_metrics[
                "delta_vs_catboost_multiseed"
            ] > 0
        ).sum()
    )

    probability_corr_vs_catboost = float(
        np.corrcoef(
            oof,
            catboost_multi_oof,
        )[0, 1]
    )

    rank_corr_vs_catboost = (
        rank_correlation(
            oof,
            catboost_multi_oof,
        )
    )

    probability_corr_vs_raw_xgb = float(
        np.corrcoef(
            oof,
            raw_xgb_oof,
        )[0, 1]
    )

    rank_corr_vs_raw_xgb = (
        rank_correlation(
            oof,
            raw_xgb_oof,
        )
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

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    if (
        delta_vs_catboost > 0
        and
        folds_beating_catboost >= 3
    ):
        decision = (
            "NEW_SINGLE_MODEL_CHAMPION"
        )

    elif (
        delta_vs_raw_xgb > 0
        and
        folds_beating_raw_xgb >= 3
    ):
        decision = (
            "KEEP_FOR_DIVERSITY"
        )

    else:
        decision = (
            "REJECT_FOR_NOW"
        )

    summary_lines = [
        "EXPERIMENT: XGBOOST_GPU_EXACT_VALUE_IDENTITIES",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Do the same exact-value identity features that transformed CatBoost",
        "also improve our depth-4 XGBoost model?",
        "",
        "CONTROL",
        "Same frozen folds and same winning depth-4 XGBoost configuration.",
        "Only two exact-value categorical copies were added.",
        f"Value-ID columns: {value_id_columns}",
        "",
        "RESULTS",
        f"Raw depth-4 XGBoost OOF AUC: {raw_xgb_auc:.8f}",
        f"Value-ID XGBoost OOF AUC: {value_id_xgb_auc:.8f}",
        f"Delta vs raw XGBoost: {delta_vs_raw_xgb:+.8f}",
        f"Folds beating raw XGBoost: {folds_beating_raw_xgb}/5",
        "",
        f"CatBoost 3-seed rank champion OOF AUC: {catboost_multi_auc:.8f}",
        f"Delta vs CatBoost champion: {delta_vs_catboost:+.8f}",
        f"Folds beating CatBoost champion: {folds_beating_catboost}/5",
        "",
        f"Probability correlation vs CatBoost champion: "
        f"{probability_corr_vs_catboost:.6f}",
        f"Rank correlation vs CatBoost champion: "
        f"{rank_corr_vs_catboost:.6f}",
        f"Probability correlation vs raw XGBoost: "
        f"{probability_corr_vs_raw_xgb:.6f}",
        f"Rank correlation vs raw XGBoost: "
        f"{rank_corr_vs_raw_xgb:.6f}",
        "",
        f"Total runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in (
        fold_metrics
        .itertuples()
    ):
        summary_lines.append(
            f"Fold {row.fold}: "
            f"raw_XGB={row.raw_xgb_depth4_auc:.8f}, "
            f"valueID_XGB={row.value_id_xgb_auc:.8f} "
            f"({row.delta_vs_raw_xgb:+.8f}), "
            f"CAT3={row.catboost_multiseed_auc:.8f}"
        )

    summary_lines.extend(
        [
            "",
            "VALUE-ID FEATURE IMPORTANCE",
        ]
    )

    value_importance = (
        importance_summary[
            importance_summary[
                "is_value_identity"
            ]
        ]
    )

    for row in (
        value_importance
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
            "If XGBoost improves strongly but remains below CatBoost, its",
            "rank correlation determines whether it is worth blending later.",
            "",
            "If it approaches or beats CatBoost, we will build a rigorous",
            "cross-model blend audit using the aligned OOF predictions.",
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
        "XGBOOST EXACT VALUE-ID EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Raw XGBoost OOF   : "
        f"{raw_xgb_auc:.8f}"
    )

    print(
        f"Value-ID XGB OOF  : "
        f"{value_id_xgb_auc:.8f}"
    )

    print(
        f"Delta vs raw XGB  : "
        f"{delta_vs_raw_xgb:+.8f}"
    )

    print(
        f"CatBoost 3-seed   : "
        f"{catboost_multi_auc:.8f}"
    )

    print(
        f"Delta vs CAT3     : "
        f"{delta_vs_catboost:+.8f}"
    )

    print(
        f"Rank corr vs CAT3 : "
        f"{rank_corr_vs_catboost:.6f}"
    )

    print(
        f"Decision          : "
        f"{decision}"
    )

    print(
        f"Runtime           : "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Artifacts         : "
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
