"""
xgboost_target_encoding_gpu.py

Phase 5A — Leakage-safe target-encoding experiment.

HYPOTHESIS
----------
Can smoothed target encodings of the strongest single features and interactions
improve our depth-4 XGBoost model beyond the current OOF AUC of ~0.94206547?

IMPORTANT
---------
This script is leakage-safe.

For each OUTER frozen validation fold:
    - validation rows are never used to build target encodings
    - outer-train rows receive OOF target encodings:
        each training row is encoded using the OTHER three training folds
    - outer validation and test rows are encoded using all four outer-train folds

This prevents a row's own target from leaking into its target-encoded features.

BASE MODEL
----------
Same depth-4 GPU XGBoost configuration that currently gives our best OOF score.

ONLY MAJOR CHANGE
-----------------
Add leakage-safe smoothed target-encoding features.

Run:
    python xgboost_target_encoding_gpu.py
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
SMOOTHING = 50.0

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_target_encoding_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe target encoding + depth-4 XGBoost."
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
        "--baseline-oof",
        type=Path,
        default=DEFAULT_BASELINE_OOF,
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
        help="Bayesian smoothing strength for target means.",
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
            f"Expected exactly one train-only target, found {train_only}"
        )

    return train_only[0]


def encode_binary_target(
    y: pd.Series,
) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))

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
        if str(value).strip().lower() in preferred:
            positive = value
            break

    if positive is None:
        positive = y.value_counts().idxmin()

    return (
        (y == positive).astype(np.int8).to_numpy(),
        positive,
    )


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}

    missing = required - set(folds.columns)

    if missing:
        raise ValueError(
            f"Fold file missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Fold row count differs from train.csv."
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
            "Fold row order does not match train.csv."
        )

    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected frozen folds [0,1,2,3,4]."
        )

    extras = [
        c
        for c in folds.columns
        if c not in {"row_index", "fold"}
    ]

    if len(extras) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extras}"
        )

    id_col = extras[0] if extras else None

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
                "Fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(dtype=np.int16),
        id_col,
    )


def detect_categoricals(
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
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            out.append(c)

    return out


def prepare_base_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categoricals: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[features].copy()
    x_test = test[features].copy()

    for c in categoricals:
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
                [train_values, test_values],
                ignore_index=True,
            ).unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        x_train[c] = train_values.astype(dtype)
        x_test[c] = test_values.astype(dtype)

    return x_train, x_test


def load_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Depth-4 baseline OOF not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Baseline OOF has unexpected columns."
        )

    if len(df) != len(train):
        raise ValueError(
            "Baseline OOF row count mismatch."
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            "Baseline OOF row order mismatch."
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Baseline OOF fold assignment mismatch."
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
            "Baseline OOF IDs are misaligned."
        )

    return df["oof_prediction"].to_numpy(
        dtype=np.float64
    )


def quantile_edges(
    values: pd.Series,
    n_bins: int,
) -> np.ndarray:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        raise ValueError(
            f"Cannot bin empty numeric feature {values.name!r}."
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.unique(
        np.nanquantile(
            numeric.to_numpy(dtype=float),
            quantiles,
        )
    )

    if len(edges) < 3:
        raise ValueError(
            f"Not enough unique bin edges for {values.name!r}."
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def apply_bin(
    values: pd.Series,
    edges: np.ndarray,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    binned = pd.cut(
        numeric,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
        labels=False,
    )

    return (
        binned
        .fillna(-1)
        .astype(np.int16)
        .astype(str)
    )


def normalize_key_part(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return (
            series
            .fillna(-999999)
            .astype(str)
        )

    return (
        series
        .fillna("__MISSING__")
        .astype(str)
    )


def make_key_frame(
    frame: pd.DataFrame,
    bin_edges: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Build categorical keys used ONLY for target encoding.

    These are deliberately focused on high-signal features and plausible
    interactions from our EDA/model results.
    """
    keys = pd.DataFrame(
        index=frame.index
    )

    # Raw categorical/discrete signals.
    keys["subsidy"] = normalize_key_part(
        frame["Subsidy_Available"]
    )
    keys["env"] = normalize_key_part(
        frame["Environmental_Concern_Level"]
    )
    keys["range"] = normalize_key_part(
        frame["Range_Anxiety_Level"]
    )
    keys["home_charge"] = normalize_key_part(
        frame["Home_Charging_Possible"]
    )
    keys["city"] = normalize_key_part(
        frame["City_Type"]
    )
    keys["car_type"] = normalize_key_part(
        frame["Current_Car_Type"]
    )

    # Outer-train-derived numeric bins.
    keys["income_bin"] = apply_bin(
        frame["Annual_Income_USD"],
        bin_edges["Annual_Income_USD"],
    )
    keys["commute_bin"] = apply_bin(
        frame["Daily_Commute_km"],
        bin_edges["Daily_Commute_km"],
    )
    keys["age_bin"] = apply_bin(
        frame["Age"],
        bin_edges["Age"],
    )

    # High-value interactions.
    keys["subsidy_env"] = (
        keys["subsidy"]
        + "|"
        + keys["env"]
    )

    keys["subsidy_range"] = (
        keys["subsidy"]
        + "|"
        + keys["range"]
    )

    keys["subsidy_income"] = (
        keys["subsidy"]
        + "|"
        + keys["income_bin"]
    )

    keys["env_income"] = (
        keys["env"]
        + "|"
        + keys["income_bin"]
    )

    keys["env_range"] = (
        keys["env"]
        + "|"
        + keys["range"]
    )

    keys["subsidy_env_range"] = (
        keys["subsidy"]
        + "|"
        + keys["env"]
        + "|"
        + keys["range"]
    )

    keys["subsidy_homecharge"] = (
        keys["subsidy"]
        + "|"
        + keys["home_charge"]
    )

    keys["subsidy_commute"] = (
        keys["subsidy"]
        + "|"
        + keys["commute_bin"]
    )

    keys["env_commute"] = (
        keys["env"]
        + "|"
        + keys["commute_bin"]
    )

    return keys


def fit_te_mapping(
    key: pd.Series,
    y: np.ndarray,
    prior: float,
    smoothing: float,
) -> pd.Series:
    df = pd.DataFrame(
        {
            "key": key.to_numpy(),
            "y": y,
        }
    )

    stats = (
        df.groupby(
            "key",
            observed=True,
        )["y"]
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


def apply_te_mapping(
    key: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> np.ndarray:
    return (
        key.map(mapping)
        .fillna(prior)
        .to_numpy(dtype=np.float32)
    )


def build_te_for_outer_fold(
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
    list[str],
]:
    """
    Returns:
        train_te : TE features for outer-training rows, OOF inside outer train
        valid_te : TE features for outer-validation rows
        test_te  : TE features for test rows

    Outer training rows are encoded using OTHER training folds only.
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

    # Binning edges are learned from the outer-training feature distribution only.
    bin_edges = {
        "Annual_Income_USD": quantile_edges(
            train.loc[
                outer_train_mask,
                "Annual_Income_USD",
            ],
            n_bins=20,
        ),
        "Daily_Commute_km": quantile_edges(
            train.loc[
                outer_train_mask,
                "Daily_Commute_km",
            ],
            n_bins=20,
        ),
        "Age": quantile_edges(
            train.loc[
                outer_train_mask,
                "Age",
            ],
            n_bins=10,
        ),
    }

    outer_train_keys = make_key_frame(
        train.loc[
            outer_train_mask
        ].reset_index(drop=True),
        bin_edges,
    )

    valid_keys = make_key_frame(
        train.loc[
            outer_valid_mask
        ].reset_index(drop=True),
        bin_edges,
    )

    test_keys = make_key_frame(
        test.reset_index(drop=True),
        bin_edges,
    )

    key_names = list(
        outer_train_keys.columns
    )

    te_columns = [
        f"TE__{name}"
        for name in key_names
    ]

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

    outer_train_y = y[
        outer_train_idx
    ]

    outer_train_original_folds = fold_ids[
        outer_train_idx
    ]

    full_prior = float(
        outer_train_y.mean()
    )

    for key_name, te_name in zip(
        key_names,
        te_columns,
    ):
        train_encoded = np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )

        # Inner OOF target encoding:
        # each original training fold is encoded from the other 3 folds
        # that remain after the current outer fold is removed.
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

            mapping = fit_te_mapping(
                key=outer_train_keys.loc[
                    inner_fit_mask,
                    key_name,
                ],
                y=outer_train_y[
                    inner_fit_mask
                ],
                prior=inner_prior,
                smoothing=smoothing,
            )

            train_encoded[
                inner_valid_mask
            ] = apply_te_mapping(
                key=outer_train_keys.loc[
                    inner_valid_mask,
                    key_name,
                ],
                mapping=mapping,
                prior=inner_prior,
            )

        if np.isnan(
            train_encoded
        ).any():
            raise RuntimeError(
                f"Leakage-safe train TE contains NaN for {key_name}."
            )

        full_mapping = fit_te_mapping(
            key=outer_train_keys[
                key_name
            ],
            y=outer_train_y,
            prior=full_prior,
            smoothing=smoothing,
        )

        valid_encoded = apply_te_mapping(
            key=valid_keys[
                key_name
            ],
            mapping=full_mapping,
            prior=full_prior,
        )

        test_encoded = apply_te_mapping(
            key=test_keys[
                key_name
            ],
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

    return (
        train_te,
        valid_te,
        test_te,
        te_columns,
    )


def build_model() -> xgb.XGBClassifier:
    """
    Same settings as our winning depth=4 XGBoost experiment.
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
        args.baseline_oof,
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
        "Loading data, frozen folds, and depth-4 baseline..."
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

    y, positive_label = encode_binary_target(
        train[
            target
        ]
    )

    fold_ids, id_col = validate_folds(
        folds_df,
        train,
    )

    features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    categoricals = detect_categoricals(
        train,
        features,
    )

    X_base, X_test_base = prepare_base_frames(
        train,
        test,
        features,
        categoricals,
    )

    baseline_oof = load_baseline_oof(
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

    print(
        f"XGBoost version: {xgb.__version__}"
    )

    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Depth-4 baseline OOF AUC: "
        f"{baseline_auc:.8f}"
    )

    print(
        f"Target-encoding smoothing: "
        f"{args.smoothing:g}"
    )

    print()
    print(
        "Target-encoding definitions:"
    )

    definitions = [
        "subsidy",
        "env",
        "range",
        "home_charge",
        "city",
        "car_type",
        "income_bin (20 quantile bins)",
        "commute_bin (20 quantile bins)",
        "age_bin (10 quantile bins)",
        "subsidy × env",
        "subsidy × range",
        "subsidy × income_bin",
        "env × income_bin",
        "env × range",
        "subsidy × env × range",
        "subsidy × home_charge",
        "subsidy × commute_bin",
        "env × commute_bin",
    ]

    for item in definitions:
        print(
            f"  - {item}"
        )

    print()
    print(
        "Running frozen 5-fold leakage-safe TE experiment..."
    )
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []

    fold_rows = []
    importance_rows = []

    all_te_columns = None

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        outer_train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )

        outer_valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        te_start = time.perf_counter()

        (
            train_te,
            valid_te,
            test_te,
            te_columns,
        ) = build_te_for_outer_fold(
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

        if all_te_columns is None:
            all_te_columns = te_columns

        if te_columns != all_te_columns:
            raise RuntimeError(
                "Target-encoding columns changed between folds."
            )

        X_train_fold = (
            X_base.iloc[
                outer_train_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_valid_fold = (
            X_base.iloc[
                outer_valid_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_test_fold = (
            X_test_base
            .reset_index(drop=True)
            .copy()
        )

        for c in te_columns:
            X_train_fold[c] = train_te[c].to_numpy(
                dtype=np.float32
            )
            X_valid_fold[c] = valid_te[c].to_numpy(
                dtype=np.float32
            )
            X_test_fold[c] = test_te[c].to_numpy(
                dtype=np.float32
            )

        model = build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train_fold,
            y[
                outer_train_idx
            ],
            eval_set=[
                (
                    X_valid_fold,
                    y[
                        outer_valid_idx
                    ],
                )
            ],
            verbose=False,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        valid_pred = model.predict_proba(
            X_valid_fold
        )[:, 1]

        test_pred = model.predict_proba(
            X_test_fold
        )[:, 1]

        oof[
            outer_valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        fold_auc = float(
            roc_auc_score(
                y[
                    outer_valid_idx
                ],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[
                    outer_valid_idx
                ],
                baseline_oof[
                    outer_valid_idx
                ],
            )
        )

        fold_delta = (
            fold_auc
            - baseline_fold_auc
        )

        feature_names = list(
            X_train_fold.columns
        )

        importances = (
            model.feature_importances_
        )

        for feature, importance in zip(
            feature_names,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature,
                    "is_target_encoding": feature.startswith(
                        "TE__"
                    ),
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_depth4_auc": baseline_fold_auc,
                "target_encoded_auc": fold_auc,
                "auc_delta": fold_delta,
                "best_iteration_zero_based": int(
                    model.best_iteration
                ),
                "te_generation_seconds": float(
                    te_seconds
                ),
                "fit_seconds": float(
                    fit_seconds
                ),
                "total_fold_seconds": float(
                    fold_seconds
                ),
                "n_base_features": len(
                    features
                ),
                "n_te_features": len(
                    te_columns
                ),
                "n_total_features": len(
                    feature_names
                ),
            }
        )

        print(
            f"Fold {outer_fold}: "
            f"baseline={baseline_fold_auc:.6f} | "
            f"TE={fold_auc:.6f} | "
            f"delta={fold_delta:+.6f} | "
            f"TE_build={te_seconds:.1f}s | "
            f"fit={fit_seconds:.1f}s | "
            f"best_iter={model.best_iteration}"
        )

        del (
            model,
            X_train_fold,
            X_valid_fold,
            X_test_fold,
            train_te,
            valid_te,
            test_te,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Target-encoded OOF predictions contain NaNs."
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    te_oof_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta = (
        te_oof_auc
        - baseline_auc
    )

    improved_folds = int(
        (
            fold_metrics[
                "auc_delta"
            ] > 0
        ).sum()
    )

    mean_fold_auc = float(
        fold_metrics[
            "target_encoded_auc"
        ].mean()
    )

    fold_std = float(
        fold_metrics[
            "target_encoded_auc"
        ].std(
            ddof=1
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
                "is_target_encoding",
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
        .reset_index(drop=True)
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

    definitions_df = pd.DataFrame(
        {
            "target_encoding": all_te_columns,
        }
    )

    definitions_df.to_csv(
        args.output_dir
        / "target_encoding_features.csv",
        index=False,
    )

    decision = (
        "KEEP"
        if delta > 0
        and improved_folds >= 3
        else "REJECT_FOR_NOW"
    )

    summary_lines = [
        "EXPERIMENT: XGBOOST_GPU_LEAKAGE_SAFE_TARGET_ENCODING",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Can leakage-safe target encodings of high-signal features and",
        "interactions improve the depth-4 XGBoost baseline?",
        "",
        "LEAKAGE SAFETY",
        "For each outer validation fold:",
        "- validation targets are never used in encoding",
        "- each outer-training row is encoded from the other three training folds",
        "- validation/test mappings are fit from all four outer-training folds",
        "",
        "MODEL CONTROL",
        "Same depth-4 XGBoost settings as the current best baseline.",
        f"Bayesian smoothing strength: {args.smoothing:g}",
        f"Base features: {len(features)}",
        f"Added TE features: {len(all_te_columns)}",
        "",
        "RESULTS",
        f"Depth-4 baseline OOF AUC: {baseline_auc:.8f}",
        f"Target-encoded OOF AUC: {te_oof_auc:.8f}",
        f"OOF AUC delta: {delta:+.8f}",
        f"Mean fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {fold_std:.8f}",
        f"Folds improved: {improved_folds}/5",
        f"Total runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD DELTAS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.baseline_depth4_auc:.8f} -> "
            f"{row.target_encoded_auc:.8f} "
            f"({row.auc_delta:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "TOP TARGET-ENCODING IMPORTANCES",
        ]
    )

    top_te = importance_summary[
        importance_summary[
            "is_target_encoding"
        ]
    ].head(12)

    for row in top_te.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "NEXT DECISION",
            "If this produces a meaningful consistent gain, target encoding",
            "becomes part of the main pipeline and we will ablate/tune the TE set.",
            "",
            "If it does not help, we will not force target encoding just because",
            "public notebooks use it; we move to another model/feature family.",
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
        "LEAKAGE-SAFE TARGET ENCODING EXPERIMENT COMPLETE"
    )
    print("=" * 78)
    print(
        f"Depth-4 baseline : {baseline_auc:.8f}"
    )
    print(
        f"Target encoded   : {te_oof_auc:.8f}"
    )
    print(
        f"Delta            : {delta:+.8f}"
    )
    print(
        f"Folds improved   : {improved_folds}/5"
    )
    print(
        f"Decision         : {decision}"
    )
    print(
        f"Total runtime    : {total_seconds:.2f}s"
    )
    print(
        f"Artifacts        : {args.output_dir.resolve()}"
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
