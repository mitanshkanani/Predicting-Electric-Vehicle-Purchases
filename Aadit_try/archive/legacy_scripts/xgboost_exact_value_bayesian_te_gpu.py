"""
xgboost_exact_value_bayesian_te_gpu.py

Phase 5J — Exact-value Bayesian target encoding for XGBoost GPU.

CURRENT REFERENCES
------------------
Raw depth-4 XGBoost:
    OOF AUC ≈ 0.94206547

Current overall champion:
    3-seed rank-averaged CatBoost + exact value identities
    OOF AUC ≈ 0.94543956

HYPOTHESIS
----------
XGBoost's native categorical handling failed to exploit exact-value identity
features, but XGBoost may benefit if we expose those identities as
leakage-safe Bayesian target statistics.

Exact identity keys:
    Annual_Income_USD
    Daily_Commute_km

Bayesian target encoding:
    TE(value) = (sum_target(value) + m * global_prior) / (count(value) + m)

with:
    m = 100

LEAKAGE SAFETY
--------------
For each OUTER frozen validation fold:

1. The outer validation fold is NEVER used to build any target encoding.

2. Outer-training rows receive INNER-OOF target encodings:
       each of the remaining four frozen folds is encoded using the other
       three outer-training folds only.

3. Outer validation + competition test rows are encoded using ALL four
   outer-training folds.

This means no row ever sees its own target inside its TE feature.

CONTROLLED EXPERIMENT
---------------------
Compared with our winning raw depth-4 XGBoost:

SAME:
- exact frozen 5 folds
- original 13 raw features
- same categorical treatment
- GPU
- n_estimators=5000
- learning_rate=0.03
- max_depth=4
- min_child_weight=8
- subsample=0.90
- colsample_bytree=0.90
- reg_lambda=2
- early_stopping_rounds=200
- seed=42

ONLY CHANGE:
- add two leakage-safe exact-value TE features:
      TE__Annual_Income_USD_exact
      TE__Daily_Commute_km_exact

Run:
    python xgboost_exact_value_bayesian_te_gpu.py
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
SMOOTHING = 100.0

EXACT_TE_FEATURES = [
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

DEFAULT_CATBOOST_CHAMPION_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe exact-value Bayesian TE for depth-4 XGBoost."
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
        "--catboost-champion-oof",
        type=Path,
        default=DEFAULT_CATBOOST_CHAMPION_OOF,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=SMOOTHING,
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
    features: list[str],
) -> list[str]:
    out = []

    for c in features:
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


def prepare_base_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_features: list[str],
    categorical_features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    x_train = train[
        raw_features
    ].copy()

    x_test = test[
        raw_features
    ].copy()

    for c in categorical_features:
        train_values = (
            x_train[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        test_values = (
            x_test[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        categories = pd.Index(
            pd.concat(
                [
                    train_values,
                    test_values,
                ],
                ignore_index=True,
            ).unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        x_train[c] = train_values.astype(dtype)
        x_test[c] = test_values.astype(dtype)

    return (
        x_train,
        x_test,
    )


def exact_key(
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
            name=feature,
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
            name=feature,
        )

    return (
        numeric
        .map(
            lambda x: format(
                float(x),
                ".12g",
            )
        )
        .rename(feature)
    )


def fit_mapping(
    keys: pd.Series,
    y: np.ndarray,
    prior: float,
    smoothing: float,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "target": y,
        }
    )

    stats = (
        frame
        .groupby(
            "key",
            observed=True,
        )["target"]
        .agg(
            ["sum", "count"]
        )
    )

    encoded = (
        stats["sum"]
        + smoothing * prior
    ) / (
        stats["count"]
        + smoothing
    )

    return encoded


def apply_mapping(
    keys: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> np.ndarray:
    return (
        keys
        .map(mapping)
        .fillna(prior)
        .to_numpy(
            dtype=np.float32
        )
    )


def build_exact_te_for_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
    smoothing: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build leakage-safe exact-value TE features.

    Outer train:
        OOF encoding across its remaining 4 frozen folds.

    Outer validation/test:
        mapping fit on all outer-training rows.
    """
    outer_train_mask = (
        fold_ids != outer_fold
    )

    outer_valid_mask = (
        fold_ids == outer_fold
    )

    outer_train_idx = np.flatnonzero(
        outer_train_mask
    )

    outer_valid_idx = np.flatnonzero(
        outer_valid_mask
    )

    outer_train_original_folds = (
        fold_ids[
            outer_train_idx
        ]
    )

    outer_train_y = (
        y[
            outer_train_idx
        ]
    )

    train_te = pd.DataFrame(
        index=np.arange(
            len(outer_train_idx)
        )
    )

    valid_te = pd.DataFrame(
        index=np.arange(
            len(outer_valid_idx)
        )
    )

    test_te = pd.DataFrame(
        index=np.arange(
            len(test)
        )
    )

    diagnostics = []

    for feature in EXACT_TE_FEATURES:
        outer_train_keys = exact_key(
            train.iloc[
                outer_train_idx
            ][feature].reset_index(
                drop=True
            ),
            feature,
        )

        valid_keys = exact_key(
            train.iloc[
                outer_valid_idx
            ][feature].reset_index(
                drop=True
            ),
            feature,
        )

        test_keys = exact_key(
            test[feature].reset_index(
                drop=True
            ),
            feature,
        )

        te_name = (
            f"TE__{feature}_exact"
        )

        train_encoded = np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )

        # Inner OOF encoding for outer-training rows.
        for inner_valid_fold in sorted(
            np.unique(
                outer_train_original_folds
            ).tolist()
        ):
            inner_valid_mask = (
                outer_train_original_folds
                == inner_valid_fold
            )

            inner_fit_mask = (
                ~inner_valid_mask
            )

            inner_prior = float(
                outer_train_y[
                    inner_fit_mask
                ].mean()
            )

            mapping = fit_mapping(
                keys=outer_train_keys[
                    inner_fit_mask
                ],
                y=outer_train_y[
                    inner_fit_mask
                ],
                prior=inner_prior,
                smoothing=smoothing,
            )

            train_encoded[
                inner_valid_mask
            ] = apply_mapping(
                keys=outer_train_keys[
                    inner_valid_mask
                ],
                mapping=mapping,
                prior=inner_prior,
            )

        if np.isnan(
            train_encoded
        ).any():
            raise RuntimeError(
                f"Train TE contains NaN values for {feature}."
            )

        full_prior = float(
            outer_train_y.mean()
        )

        full_mapping = fit_mapping(
            keys=outer_train_keys,
            y=outer_train_y,
            prior=full_prior,
            smoothing=smoothing,
        )

        valid_encoded = apply_mapping(
            keys=valid_keys,
            mapping=full_mapping,
            prior=full_prior,
        )

        test_encoded = apply_mapping(
            keys=test_keys,
            mapping=full_mapping,
            prior=full_prior,
        )

        train_te[
            te_name
        ] = train_encoded

        valid_te[
            te_name
        ] = valid_encoded

        test_te[
            te_name
        ] = test_encoded

        known_keys = set(
            full_mapping.index
        )

        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "feature": feature,
                "smoothing": smoothing,

                "outer_train_unique_keys": int(
                    outer_train_keys.nunique()
                ),

                "valid_unseen_rate": float(
                    (~valid_keys.isin(
                        known_keys
                    )).mean()
                ),

                "test_unseen_rate": float(
                    (~test_keys.isin(
                        known_keys
                    )).mean()
                ),

                "outer_train_prior": (
                    full_prior
                ),
            }
        )

    return (
        train_te,
        valid_te,
        test_te,
        pd.DataFrame(
            diagnostics
        ),
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

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(
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
        df["row_index"].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            f"OOF row order mismatch in {path}"
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF folds mismatch in {path}"
        )

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(
            df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        )
    ):
        raise ValueError(
            f"OOF ID mismatch in {path}"
        )

    return df[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            method="average",
            pct=True,
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
        args.raw_xgb_oof,
        args.catboost_champion_oof,
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
        "Loading data, frozen folds, and reference OOF predictions..."
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
        X_base,
        X_test_base,
    ) = prepare_base_frames(
        train=train,
        test=test,
        raw_features=raw_features,
        categorical_features=raw_categoricals,
    )

    raw_xgb_oof = load_oof(
        args.raw_xgb_oof,
        train,
        fold_ids,
        id_col,
    )

    catboost_oof = load_oof(
        args.catboost_champion_oof,
        train,
        fold_ids,
        id_col,
    )

    raw_xgb_auc = float(
        roc_auc_score(
            y,
            raw_xgb_oof,
        )
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
        f"Raw depth-4 XGBoost OOF: "
        f"{raw_xgb_auc:.8f}"
    )

    print(
        f"Current CatBoost champion OOF: "
        f"{catboost_auc:.8f}"
    )

    print()
    print(
        f"Bayesian smoothing m = "
        f"{args.smoothing:g}"
    )

    print(
        "Exact-value TE features:"
    )

    for feature in EXACT_TE_FEATURES:
        print(
            f"  - TE__{feature}_exact"
        )

    print()
    print(
        "Running frozen 5-fold leakage-safe exact-value TE CV..."
    )
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []

    fold_rows = []
    importance_rows = []
    diagnostic_frames = []

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        te_start = time.perf_counter()

        (
            train_te,
            valid_te,
            test_te,
            diagnostics,
        ) = build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=args.smoothing,
        )

        te_seconds = (
            time.perf_counter()
            - te_start
        )

        diagnostic_frames.append(
            diagnostics
        )

        X_train = (
            X_base.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        X_valid = (
            X_base.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        X_test = (
            X_test_base
            .reset_index(
                drop=True
            )
            .copy()
        )

        te_columns = list(
            train_te.columns
        )

        for c in te_columns:
            X_train[c] = train_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

        model = build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[
                train_idx
            ],

            eval_set=[
                (
                    X_valid,
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

        infer_start = time.perf_counter()

        valid_pred = (
            model.predict_proba(
                X_valid
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

        candidate_fold_auc = float(
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
                catboost_oof[
                    valid_idx
                ],
            )
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,

                "raw_xgb_auc": (
                    raw_xgb_fold_auc
                ),

                "exact_te_xgb_auc": (
                    candidate_fold_auc
                ),

                "delta_vs_raw_xgb": (
                    candidate_fold_auc
                    - raw_xgb_fold_auc
                ),

                "catboost_champion_auc": (
                    catboost_fold_auc
                ),

                "delta_vs_catboost_champion": (
                    candidate_fold_auc
                    - catboost_fold_auc
                ),

                "best_iteration_zero_based": int(
                    model.best_iteration
                ),

                "te_generation_seconds": float(
                    te_seconds
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "inference_seconds": float(
                    inference_seconds
                ),

                "total_fold_seconds": float(
                    fold_seconds
                ),
            }
        )

        importances = (
            model.feature_importances_
        )

        for feature, importance in zip(
            X_train.columns,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,

                    "feature": feature,

                    "is_exact_value_te": (
                        feature
                        in te_columns
                    ),

                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print()
        print(
            f"Fold {outer_fold}: "
            f"raw_XGB={raw_xgb_fold_auc:.6f} | "
            f"exact_TE={candidate_fold_auc:.6f} | "
            f"delta={candidate_fold_auc - raw_xgb_fold_auc:+.6f} | "
            f"CAT3={catboost_fold_auc:.6f} | "
            f"best_iter={model.best_iteration} | "
            f"TE_build={te_seconds:.1f}s | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            X_train,
            X_valid,
            X_test,
            train_te,
            valid_te,
            test_te,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "Exact-value TE OOF predictions contain NaNs."
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_raw_xgb = (
        candidate_auc
        - raw_xgb_auc
    )

    delta_vs_catboost = (
        candidate_auc
        - catboost_auc
    )

    folds_improved_vs_raw = int(
        (
            fold_metrics[
                "delta_vs_raw_xgb"
            ] > 0
        ).sum()
    )

    folds_beating_catboost = int(
        (
            fold_metrics[
                "delta_vs_catboost_champion"
            ] > 0
        ).sum()
    )

    probability_corr_vs_catboost = float(
        np.corrcoef(
            oof,
            catboost_oof,
        )[0, 1]
    )

    rank_corr_vs_catboost = (
        rank_corr(
            oof,
            catboost_oof,
        )
    )

    probability_corr_vs_raw_xgb = float(
        np.corrcoef(
            oof,
            raw_xgb_oof,
        )[0, 1]
    )

    rank_corr_vs_raw_xgb = (
        rank_corr(
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
                "is_exact_value_te",
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

    diagnostics = pd.concat(
        diagnostic_frames,
        ignore_index=True,
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

    diagnostics.to_csv(
        args.output_dir
        / "te_diagnostics.csv",
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
        folds_improved_vs_raw >= 3
    ):
        decision = (
            "KEEP_FOR_DIVERSITY"
        )

    else:
        decision = (
            "REJECT_FOR_NOW"
        )

    summary_lines = [
        "EXPERIMENT: XGBOOST_EXACT_VALUE_BAYESIAN_TE_GPU",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Can XGBoost exploit exact income/commute identities when they are",
        "expressed as leakage-safe Bayesian target statistics?",
        "",
        "LEAKAGE SAFETY",
        "For every outer fold, validation targets are never used in TE.",
        "Outer-training rows receive inner-OOF encodings from the other",
        "three outer-training folds.",
        "",
        "CONTROL",
        "Same raw depth-4 XGBoost setup.",
        "Only two numeric TE features are added.",
        f"Smoothing m: {args.smoothing:g}",
        "",
        "RESULTS",
        f"Raw depth-4 XGBoost OOF AUC: {raw_xgb_auc:.8f}",
        f"Exact-value TE XGBoost OOF AUC: {candidate_auc:.8f}",
        f"Delta vs raw XGBoost: {delta_vs_raw_xgb:+.8f}",
        f"Folds improved vs raw XGBoost: {folds_improved_vs_raw}/5",
        "",
        f"CatBoost 3-seed champion OOF AUC: {catboost_auc:.8f}",
        f"Delta vs CatBoost champion: {delta_vs_catboost:+.8f}",
        f"Folds beating CatBoost champion: {folds_beating_catboost}/5",
        "",
        f"Probability correlation vs CatBoost: "
        f"{probability_corr_vs_catboost:.6f}",
        f"Rank correlation vs CatBoost: "
        f"{rank_corr_vs_catboost:.6f}",
        f"Probability correlation vs raw XGBoost: "
        f"{probability_corr_vs_raw_xgb:.6f}",
        f"Rank correlation vs raw XGBoost: "
        f"{rank_corr_vs_raw_xgb:.6f}",
        "",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"raw={row.raw_xgb_auc:.8f} -> "
            f"exact_TE={row.exact_te_xgb_auc:.8f} "
            f"({row.delta_vs_raw_xgb:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "EXACT-TE FEATURE IMPORTANCES",
        ]
    )

    te_importance = importance_summary[
        importance_summary[
            "is_exact_value_te"
        ]
    ]

    for row in te_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "This experiment specifically tests whether manual leakage-safe",
            "Bayesian statistics unlock the exact-value signal for XGBoost",
            "that its native categorical split handling failed to use.",
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
        "XGBOOST EXACT-VALUE BAYESIAN TE EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Raw XGBoost OOF : "
        f"{raw_xgb_auc:.8f}"
    )

    print(
        f"Exact-TE XGB OOF: "
        f"{candidate_auc:.8f}"
    )

    print(
        f"Delta vs raw XGB: "
        f"{delta_vs_raw_xgb:+.8f}"
    )

    print(
        f"Folds improved  : "
        f"{folds_improved_vs_raw}/5"
    )

    print(
        f"CatBoost champ  : "
        f"{catboost_auc:.8f}"
    )

    print(
        f"Delta vs CAT3   : "
        f"{delta_vs_catboost:+.8f}"
    )

    print(
        f"Rank corr vs CAT: "
        f"{rank_corr_vs_catboost:.6f}"
    )

    print(
        f"Decision        : "
        f"{decision}"
    )

    print(
        f"Runtime         : "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Artifacts       : "
        f"{args.output_dir.resolve()}"
    )

    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. feature_importance.csv")
    print("  5. te_diagnostics.csv")


if __name__ == "__main__":
    main()
