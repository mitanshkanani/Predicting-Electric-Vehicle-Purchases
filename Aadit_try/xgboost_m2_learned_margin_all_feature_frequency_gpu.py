"""
xgboost_exact_value_bayesian_te_gpu.py

Controlled experiment — extend target-free marginal frequency encoding from income/commute to every remaining raw feature while keeping the current learned-margin XGBoost champion fixed.

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
    m = 2

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
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.special import ndtr, expit

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U xgboost\n"
    ) from exc


SEED = 42
SMOOTHING = 2.0

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
    / "xgboost_m2_learned_margin_all_feature_frequency_gpu"
)


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

DEFAULT_M2_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_smoothing_ultralow_gpu"
    / "oof_predictions_m2.csv"
)

DEFAULT_INCOME_DIGIT_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digit_decomposition_gpu"
    / "oof_predictions.csv"
)

DEFAULT_RECIPE_MARGIN_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digits_recipe_base_margin_gpu"
    / "oof_predictions.csv"
)

DEFAULT_EXACT_FREQUENCY_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digits_recipe_margin_exact_frequency_gpu"
    / "oof_predictions.csv"
)

DEFAULT_LEARNED_MARGIN_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_exact_frequency_learned_logistic_margin_gpu"
    / "oof_predictions.csv"
)


INCOME_DIGIT_SPECS = [
    ("FE__income_digit_ones", 1),
    ("FE__income_digit_tens", 10),
    ("FE__income_digit_hundreds", 100),
    ("FE__income_digit_thousands", 1_000),
    ("FE__income_digit_ten_thousands", 10_000),
    ("FE__income_digit_hundred_thousands", 100_000),
]

INCOME_GROUP_SPECS = [
    ("FE__income_last2", 100),
    ("FE__income_last3", 1_000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost m=2 exact-TE + Annual_Income_USD digit-decomposition experiment."
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
        "--m2-xgb-oof",
        type=Path,
        default=DEFAULT_M2_XGB_OOF,
    )

    parser.add_argument(
        "--income-digit-oof",
        type=Path,
        default=DEFAULT_INCOME_DIGIT_OOF,
    )

    parser.add_argument(
        "--recipe-margin-oof",
        type=Path,
        default=DEFAULT_RECIPE_MARGIN_OOF,
    )

    parser.add_argument(
        "--exact-frequency-oof",
        type=Path,
        default=DEFAULT_EXACT_FREQUENCY_OOF,
    )

    parser.add_argument(
        "--learned-margin-oof",
        type=Path,
        default=DEFAULT_LEARNED_MARGIN_OOF,
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



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_income_digit_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Add decimal decomposition of Annual_Income_USD only.

    This is intentionally target-free feature engineering:
    every added value is a deterministic function of Annual_Income_USD.
    """
    feature = "Annual_Income_USD"

    train_income = np.rint(
        pd.to_numeric(
            train_source[feature],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    ).astype(np.int64)

    test_income = np.rint(
        pd.to_numeric(
            test_source[feature],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    ).astype(np.int64)

    if (train_income < 0).any() or (test_income < 0).any():
        raise ValueError("Annual_Income_USD unexpectedly contains negative values.")

    if train_income.max() >= 1_000_000 or test_income.max() >= 1_000_000:
        raise ValueError(
            "Income >= 1,000,000 found. This experiment assumes six decimal "
            "positions; inspect the data before changing the feature set."
        )

    X_train = X_train.copy()
    X_test = X_test.copy()
    added = []

    for name, divisor in INCOME_DIGIT_SPECS:
        X_train[name] = ((train_income // divisor) % 10).astype(np.int8)
        X_test[name] = ((test_income // divisor) % 10).astype(np.int8)
        added.append(name)

    # Small hierarchical remainder features. These still express the same
    # single hypothesis: decimal structure in the exact income value.
    for name, modulus in INCOME_GROUP_SPECS:
        X_train[name] = (train_income % modulus).astype(np.int16)
        X_test[name] = (test_income % modulus).astype(np.int16)
        added.append(name)

    return X_train, X_test, added




def add_remaining_raw_frequency_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    raw_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Add ONE target-free marginal-density representation for every raw feature
    except Annual_Income_USD and Daily_Commute_km, whose exact-frequency
    count/log/rate features are already in the validated baseline.

    We use log1p(combined train+test exact-value count). No target values are
    involved. This tests whether marginal generator density outside income and
    commute carries additional synthetic signal.

    For categoricals, the exact value is the category string.
    For discrete numerics, the exact value is a stable string representation.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    excluded = {
        "Annual_Income_USD",
        "Daily_Commute_km",
    }

    added: list[str] = []

    for feature in raw_features:
        if feature in excluded:
            continue

        if feature not in train_source.columns or feature not in test_source.columns:
            raise ValueError(
                f"Missing raw feature for frequency encoding: {feature}"
            )

        train_values = (
            train_source[feature]
            .fillna("__MISSING__")
            .astype(str)
        )
        test_values = (
            test_source[feature]
            .fillna("__MISSING__")
            .astype(str)
        )

        combined = pd.concat(
            [
                train_values,
                test_values,
            ],
            ignore_index=True,
        )

        counts = combined.value_counts(
            dropna=False
        )

        train_count = (
            train_values
            .map(counts)
            .to_numpy(dtype=np.float32)
        )
        test_count = (
            test_values
            .map(counts)
            .to_numpy(dtype=np.float32)
        )

        feature_name = (
            f"FE__{feature}__frequency_log1p"
        )

        X_train[feature_name] = np.log1p(
            train_count
        ).astype(np.float32)

        X_test[feature_name] = np.log1p(
            test_count
        ).astype(np.float32)

        added.append(feature_name)

    return X_train, X_test, added


def build_recipe_base_margin(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the public original-dataset recipe as an XGBoost base margin.

    Original-data recipe:
        score =
            1.2 * (income / 100000)
          + 0.6 * environmental concern
          + 2.0 * subsidy_yes
          - 1.0 * range_anxiety_medium
          - 3.0 * range_anxiety_high

        buy if score + Normal(0, 1) > 5.5

    Therefore the implied probability is:
        P(buy) = Phi(score - 5.5)

    XGBoost's binary:logistic base_margin expects raw log-odds, so we convert
    the implied probability to logit space.

    This uses no competition targets.
    """
    required = [
        "Annual_Income_USD",
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing recipe columns: {missing}")

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    env = pd.to_numeric(
        df["Environmental_Concern_Level"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    subsidy = (
        df["Subsidy_Available"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    anxiety = (
        df["Range_Anxiety_Level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not set(subsidy.unique()).issubset({"yes", "no"}):
        raise ValueError(
            "Unexpected Subsidy_Available labels: "
            f"{sorted(set(subsidy.unique()))}"
        )

    if not set(anxiety.unique()).issubset({"low", "medium", "high"}):
        raise ValueError(
            "Unexpected Range_Anxiety_Level labels: "
            f"{sorted(set(anxiety.unique()))}"
        )

    subsidy_yes = (subsidy == "yes").to_numpy(dtype=np.float64)
    medium = (anxiety == "medium").to_numpy(dtype=np.float64)
    high = (anxiety == "high").to_numpy(dtype=np.float64)

    score = (
        1.2 * (income / 100_000.0)
        + 0.6 * env
        + 2.0 * subsidy_yes
        - 1.0 * medium
        - 3.0 * high
    )

    z = score - 5.5
    probability = ndtr(z)

    # Avoid infinite log-odds for extreme tails.
    probability = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    margin = np.log(
        probability / (1.0 - probability)
    ).astype(np.float32)

    return (
        margin,
        score.astype(np.float32),
        probability.astype(np.float32),
    )



def add_exact_frequency_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Add target-free exact-value density features for the two high-cardinality
    numeric variables already known to carry exact-identity signal.

    Frequency tables are computed on train + test together. This is unsupervised
    transductive feature engineering: no target values are used. It matches the
    final deployment setting and mirrors the already-used train+test category
    vocabulary construction.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    feature_specs = [
        ("Annual_Income_USD", "income"),
        ("Daily_Commute_km", "commute"),
    ]

    added: list[str] = []
    total_rows = float(len(train_source) + len(test_source))

    for source_col, short_name in feature_specs:
        if source_col not in train_source.columns or source_col not in test_source.columns:
            raise ValueError(f"Missing source feature: {source_col}")

        if source_col == "Annual_Income_USD":
            train_key = (
                np.rint(
                    pd.to_numeric(
                        train_source[source_col],
                        errors="raise",
                    ).to_numpy(dtype=np.float64)
                )
                .astype(np.int64)
                .astype(str)
            )
            test_key = (
                np.rint(
                    pd.to_numeric(
                        test_source[source_col],
                        errors="raise",
                    ).to_numpy(dtype=np.float64)
                )
                .astype(np.int64)
                .astype(str)
            )
        else:
            train_numeric = pd.to_numeric(
                train_source[source_col],
                errors="raise",
            ).to_numpy(dtype=np.float64)
            test_numeric = pd.to_numeric(
                test_source[source_col],
                errors="raise",
            ).to_numpy(dtype=np.float64)

            train_key = np.array(
                [f"{v:.1f}" for v in train_numeric],
                dtype=object,
            )
            test_key = np.array(
                [f"{v:.1f}" for v in test_numeric],
                dtype=object,
            )

        combined = pd.Series(
            np.concatenate([train_key, test_key]),
            dtype="object",
        )
        counts = combined.value_counts(dropna=False)

        train_count = (
            pd.Series(train_key, dtype="object")
            .map(counts)
            .to_numpy(dtype=np.float32)
        )
        test_count = (
            pd.Series(test_key, dtype="object")
            .map(counts)
            .to_numpy(dtype=np.float32)
        )

        count_name = f"FE__{short_name}_exact_frequency_count"
        log_name = f"FE__{short_name}_exact_frequency_log1p"
        rate_name = f"FE__{short_name}_exact_frequency_rate"

        X_train[count_name] = train_count
        X_test[count_name] = test_count

        X_train[log_name] = np.log1p(train_count).astype(np.float32)
        X_test[log_name] = np.log1p(test_count).astype(np.float32)

        X_train[rate_name] = (train_count / total_rows).astype(np.float32)
        X_test[rate_name] = (test_count / total_rows).astype(np.float32)

        added.extend(
            [
                count_name,
                log_name,
                rate_name,
            ]
        )

    return X_train, X_test, added



def build_logistic_recipe_matrix(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Competition-adapted version of the four-variable original-data recipe.

    Columns:
      1. Annual_Income_USD / 100000
      2. Environmental_Concern_Level
      3. Subsidy_Available == Yes
      4. Range_Anxiety_Level == Medium
      5. Range_Anxiety_Level == High

    Low anxiety is the reference level. No target information is used here.
    """
    required = [
        "Annual_Income_USD",
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing learned-margin columns: {missing}"
        )

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="raise",
    ).to_numpy(dtype=np.float64) / 100_000.0

    env = pd.to_numeric(
        df["Environmental_Concern_Level"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    subsidy = (
        df["Subsidy_Available"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    anxiety = (
        df["Range_Anxiety_Level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    subsidy_values = set(subsidy.unique().tolist())
    anxiety_values = set(anxiety.unique().tolist())

    if not subsidy_values.issubset({"yes", "no"}):
        raise ValueError(
            "Unexpected Subsidy_Available labels: "
            f"{sorted(subsidy_values)}"
        )

    if not anxiety_values.issubset(
        {"low", "medium", "high"}
    ):
        raise ValueError(
            "Unexpected Range_Anxiety_Level labels: "
            f"{sorted(anxiety_values)}"
        )

    return np.column_stack(
        [
            income,
            env,
            (subsidy == "yes").to_numpy(dtype=np.float64),
            (anxiety == "medium").to_numpy(dtype=np.float64),
            (anxiety == "high").to_numpy(dtype=np.float64),
        ]
    )


def new_logistic_prior_model() -> LogisticRegression:
    """
    Near-unregularized logistic regression.

    C is intentionally fixed, not tuned. The experiment is about replacing
    the fixed source-data recipe margin with a competition-learned smooth
    four-variable prior, not about logistic hyperparameter search.
    """
    return LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


def build_learned_logistic_margins_for_outer_fold(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Build leakage-safe XGBoost base margins.

    Outer-training rows:
      nested OOF logistic predictions using only the other frozen folds.

    Outer-validation rows:
      logistic model fit on the full outer-training set.

    Test rows:
      same full outer-training logistic model.

    This prevents the logistic prior from seeing a training row's own target
    when constructing that row's XGBoost base margin.
    """
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    original_folds = fold_ids[
        outer_train_idx
    ]

    train_margin = np.full(
        len(outer_train_idx),
        np.nan,
        dtype=np.float32,
    )

    for inner_valid_fold in sorted(
        np.unique(
            original_folds
        ).tolist()
    ):
        inner_valid_mask = (
            original_folds
            == inner_valid_fold
        )
        inner_fit_mask = ~inner_valid_mask

        model = new_logistic_prior_model()

        model.fit(
            train_matrix[
                outer_train_idx[
                    inner_fit_mask
                ]
            ],
            y[
                outer_train_idx[
                    inner_fit_mask
                ]
            ],
        )

        train_margin[
            inner_valid_mask
        ] = model.decision_function(
            train_matrix[
                outer_train_idx[
                    inner_valid_mask
                ]
            ]
        ).astype(np.float32)

    if np.isnan(train_margin).any():
        raise RuntimeError(
            "Nested logistic training margins contain NaN."
        )

    full_model = new_logistic_prior_model()

    full_model.fit(
        train_matrix[
            outer_train_idx
        ],
        y[
            outer_train_idx
        ],
    )

    valid_margin = full_model.decision_function(
        train_matrix[
            outer_valid_idx
        ]
    ).astype(np.float32)

    test_margin = full_model.decision_function(
        test_matrix
    ).astype(np.float32)

    coefficients = full_model.coef_[0]
    diagnostics = pd.DataFrame(
        [
            {
                "outer_fold": outer_fold,
                "intercept": float(
                    full_model.intercept_[0]
                ),
                "coef_income_100k": float(
                    coefficients[0]
                ),
                "coef_environmental_concern": float(
                    coefficients[1]
                ),
                "coef_subsidy_yes": float(
                    coefficients[2]
                ),
                "coef_range_anxiety_medium": float(
                    coefficients[3]
                ),
                "coef_range_anxiety_high": float(
                    coefficients[4]
                ),
            }
        ]
    )

    return (
        train_margin,
        valid_margin,
        test_margin,
        diagnostics,
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

    if not np.isclose(args.smoothing, 2.0):
        raise ValueError(
            "This is a controlled m=2 experiment. Do not change --smoothing."
        )

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    for path in [
        train_path,
        test_path,
        args.folds_path,
        args.raw_xgb_oof,
        args.catboost_champion_oof,
        args.m2_xgb_oof,
        args.income_digit_oof,
        args.recipe_margin_oof,
        args.exact_frequency_oof,
        args.learned_margin_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    fold_hash = sha256_file(args.folds_path)
    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}\n"
            f"File    : {args.folds_path.resolve()}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("XGBOOST m=2 + LEARNED MARGIN + ALL-FEATURE FREQUENCY EXPERIMENT")
    print("=" * 88)
    print(f"XGBoost version: {xgb.__version__}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(train, test)
    y, positive_label = encode_binary_target(train[target])
    fold_ids, id_col = validate_folds(folds_df, train)

    raw_features = [
        c
        for c in test.columns
        if c in train.columns and c != id_col
    ]

    raw_categoricals = detect_raw_categoricals(
        train,
        raw_features,
    )

    X_base, X_test_base = prepare_base_frames(
        train=train,
        test=test,
        raw_features=raw_features,
        categorical_features=raw_categoricals,
    )

    # Keep the already-validated income-digit representation.
    X_income_base, X_income_test_base, digit_features = (
        add_income_digit_features(
            X_train=X_base,
            X_test=X_test_base,
            train_source=train,
            test_source=test,
        )
    )

    # ONLY candidate feature change in THIS experiment:
    # add exact-value frequency/density representations of income + commute.
    (
        X_exact_frequency_base,
        X_exact_frequency_test_base,
        exact_frequency_features,
    ) = add_exact_frequency_features(
        X_train=X_income_base,
        X_test=X_income_test_base,
        train_source=train,
        test_source=test,
    )

    # ONLY candidate feature change in THIS experiment:
    # extend marginal frequency encoding from income+commute to every
    # remaining raw feature, using one log1p count per feature.
    (
        X_candidate_base,
        X_candidate_test_base,
        all_feature_frequency_features,
    ) = add_remaining_raw_frequency_features(
        X_train=X_exact_frequency_base,
        X_test=X_exact_frequency_test_base,
        train_source=train,
        test_source=test,
        raw_features=raw_features,
    )

    # Keep the validated learned logistic base margin exactly.
    (
        train_recipe_margin,
        train_recipe_score,
        train_recipe_probability,
    ) = build_recipe_base_margin(train)

    (
        test_recipe_margin,
        test_recipe_score,
        test_recipe_probability,
    ) = build_recipe_base_margin(test)

    # Candidate margin uses the SAME four conceptual recipe variables,
    # but learns their coefficients from competition training data in a
    # fully nested leakage-safe manner inside each frozen outer fold.
    logistic_train_matrix = build_logistic_recipe_matrix(
        train
    )
    logistic_test_matrix = build_logistic_recipe_matrix(
        test
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

    m2_xgb_oof = load_oof(
        args.m2_xgb_oof,
        train,
        fold_ids,
        id_col,
    )

    income_digit_oof = load_oof(
        args.income_digit_oof,
        train,
        fold_ids,
        id_col,
    )

    recipe_margin_oof = load_oof(
        args.recipe_margin_oof,
        train,
        fold_ids,
        id_col,
    )

    exact_frequency_oof = load_oof(
        args.exact_frequency_oof,
        train,
        fold_ids,
        id_col,
    )

    learned_margin_oof = load_oof(
        args.learned_margin_oof,
        train,
        fold_ids,
        id_col,
    )

    raw_xgb_auc = float(roc_auc_score(y, raw_xgb_oof))
    catboost_auc = float(roc_auc_score(y, catboost_oof))
    m2_baseline_auc = float(roc_auc_score(y, m2_xgb_oof))
    income_digit_baseline_auc = float(
        roc_auc_score(
            y,
            income_digit_oof,
        )
    )
    recipe_margin_baseline_auc = float(
        roc_auc_score(
            y,
            recipe_margin_oof,
        )
    )
    exact_frequency_baseline_auc = float(
        roc_auc_score(
            y,
            exact_frequency_oof,
        )
    )
    learned_margin_baseline_auc = float(
        roc_auc_score(
            y,
            learned_margin_oof,
        )
    )
    recipe_only_auc = float(
        roc_auc_score(
            y,
            train_recipe_probability,
        )
    )

    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Raw depth-4 XGBoost OOF : {raw_xgb_auc:.8f}")
    print(f"CatBoost 3-seed OOF     : {catboost_auc:.8f}")
    print(f"XGB exact-TE m=2 OOF    : {m2_baseline_auc:.8f}")
    print(f"Income-digit XGB OOF    : {income_digit_baseline_auc:.8f}")
    print(f"Recipe-margin XGB OOF   : {recipe_margin_baseline_auc:.8f}")
    print(f"Exact-frequency XGB OOF : {exact_frequency_baseline_auc:.8f}")
    print(f"Learned-margin XGB OOF  : {learned_margin_baseline_auc:.8f}")
    print(f"Fixed recipe-only AUC   : {recipe_only_auc:.8f}")
    print()
    print("HYPOTHESIS:")
    print(
        "  Exact-frequency density helped for income and commute. Other raw "
        "features may also carry generator-density artifacts in how often exact "
        "values/categories occur, even when their ordinary values are already "
        "present in the model."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add one target-free log1p(train+test exact-frequency) feature for "
        "every remaining raw feature:"
    )
    for feature in all_feature_frequency_features:
        print(f"    - {feature}")
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds")
    print("  - original raw features")
    print("  - validated income digit features")
    print("  - validated income/commute exact-frequency features")
    print("  - validated nested learned logistic base margin")
    print("  - exact-value TE keys: income + commute")
    print("  - leakage-safe nested TE")
    print("  - smoothing m=2")
    print("  - depth-4 XGBoost hyperparameters")
    print("  - seed 42")
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
    logistic_diagnostic_frames = []

    learned_prior_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

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
            smoothing=2.0,
        )

        te_seconds = time.perf_counter() - te_start
        diagnostic_frames.append(diagnostics)

        (
            learned_train_margin,
            learned_valid_margin,
            learned_test_margin,
            logistic_diagnostics,
        ) = build_learned_logistic_margins_for_outer_fold(
            train_matrix=logistic_train_matrix,
            test_matrix=logistic_test_matrix,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        logistic_diagnostic_frames.append(
            logistic_diagnostics
        )

        learned_prior_oof[
            valid_idx
        ] = expit(
            learned_valid_margin
        )

        X_train = (
            X_candidate_base.iloc[train_idx]
            .reset_index(drop=True)
            .copy()
        )

        X_valid = (
            X_candidate_base.iloc[valid_idx]
            .reset_index(drop=True)
            .copy()
        )

        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        te_columns = list(train_te.columns)

        for c in te_columns:
            X_train[c] = train_te[c].to_numpy(dtype=np.float32)
            X_valid[c] = valid_te[c].to_numpy(dtype=np.float32)
            X_test[c] = test_te[c].to_numpy(dtype=np.float32)

        model = build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[train_idx],
            base_margin=learned_train_margin,
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            base_margin_eval_set=[
                learned_valid_margin,
            ],
            verbose=100,
        )

        fit_seconds = time.perf_counter() - fit_start

        infer_start = time.perf_counter()

        valid_pred = model.predict_proba(
            X_valid,
            base_margin=learned_valid_margin,
        )[:, 1]

        test_pred = model.predict_proba(
            X_test,
            base_margin=learned_test_margin,
        )[:, 1]

        inference_seconds = time.perf_counter() - infer_start

        oof[valid_idx] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                learned_margin_oof[valid_idx],
            )
        )

        cat_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                catboost_oof[valid_idx],
            )
        )

        fold_seconds = time.perf_counter() - fold_start

        fold_rows.append(
            {
                "fold": outer_fold,
                "learned_margin_baseline_auc": baseline_fold_auc,
                "digit_candidate_auc": candidate_fold_auc,
                "delta_vs_learned_margin": (
                    candidate_fold_auc
                    - baseline_fold_auc
                ),
                "catboost_auc": cat_fold_auc,
                "delta_vs_catboost": (
                    candidate_fold_auc
                    - cat_fold_auc
                ),
                "best_iteration_zero_based": int(
                    model.best_iteration
                ),
                "te_generation_seconds": float(te_seconds),
                "fit_seconds": float(fit_seconds),
                "inference_seconds": float(inference_seconds),
                "total_fold_seconds": float(fold_seconds),
            }
        )

        for feature, importance in zip(
            X_train.columns,
            model.feature_importances_,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature,
                    "is_income_digit_feature": (
                        feature in digit_features
                    ),
                    "is_exact_frequency_feature": (
                        feature in exact_frequency_features
                    ),
                    "is_all_feature_frequency_extension": (
                        feature in all_feature_frequency_features
                    ),
                    "is_exact_te_feature": (
                        feature in te_columns
                    ),
                    "gain_importance": float(importance),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"learned_margin={baseline_fold_auc:.8f} -> "
            f"+all_frequency={candidate_fold_auc:.8f} "
            f"({candidate_fold_auc - baseline_fold_auc:+.8f}) | "
            f"best_iter={model.best_iteration}"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test,
            train_te,
            valid_te,
            test_te,
        )

    total_seconds = time.perf_counter() - total_start

    if np.isnan(oof).any():
        raise RuntimeError(
            "Candidate OOF predictions contain NaNs."
        )

    fold_metrics = pd.DataFrame(fold_rows)

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    if np.isnan(learned_prior_oof).any():
        raise RuntimeError(
            "Learned logistic prior OOF contains NaN."
        )

    learned_prior_auc = float(
        roc_auc_score(
            y,
            learned_prior_oof,
        )
    )

    delta_vs_learned_margin = (
        candidate_auc
        - learned_margin_baseline_auc
    )
    delta_vs_exact_frequency = (
        candidate_auc
        - exact_frequency_baseline_auc
    )
    delta_vs_recipe_margin = (
        candidate_auc
        - recipe_margin_baseline_auc
    )
    delta_vs_income_digits = (
        candidate_auc
        - income_digit_baseline_auc
    )
    delta_vs_m2 = candidate_auc - m2_baseline_auc
    delta_vs_cat = candidate_auc - catboost_auc

    folds_improved_vs_learned_margin = int(
        (
            fold_metrics[
                "delta_vs_learned_margin"
            ] > 0
        ).sum()
    )

    folds_worse_vs_learned_margin = int(
        (
            fold_metrics[
                "delta_vs_learned_margin"
            ] < 0
        ).sum()
    )

    probability_corr_vs_learned_margin = float(
        np.corrcoef(
            oof,
            learned_margin_oof,
        )[0, 1]
    )

    rank_corr_vs_learned_margin = rank_corr(
        oof,
        learned_margin_oof,
    )

    probability_corr_vs_exact_frequency = float(
        np.corrcoef(
            oof,
            exact_frequency_oof,
        )[0, 1]
    )

    rank_corr_vs_exact_frequency = rank_corr(
        oof,
        exact_frequency_oof,
    )

    probability_corr_vs_recipe_margin = float(
        np.corrcoef(
            oof,
            recipe_margin_oof,
        )[0, 1]
    )

    rank_corr_vs_recipe_margin = rank_corr(
        oof,
        recipe_margin_oof,
    )

    probability_corr_vs_income_digits = float(
        np.corrcoef(
            oof,
            income_digit_oof,
        )[0, 1]
    )

    rank_corr_vs_income_digits = rank_corr(
        oof,
        income_digit_oof,
    )

    probability_corr_vs_m2 = float(
        np.corrcoef(
            oof,
            m2_xgb_oof,
        )[0, 1]
    )

    rank_corr_vs_m2 = rank_corr(
        oof,
        m2_xgb_oof,
    )

    probability_corr_vs_cat = float(
        np.corrcoef(
            oof,
            catboost_oof,
        )[0, 1]
    )

    rank_corr_vs_cat = rank_corr(
        oof,
        catboost_oof,
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
                "is_income_digit_feature",
                "is_exact_frequency_feature",
                "is_all_feature_frequency_extension",
                "is_exact_te_feature",
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

    all_feature_frequency_importance = (
        importance_summary[
            importance_summary[
                "is_all_feature_frequency_extension"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    diagnostics = pd.concat(
        diagnostic_frames,
        ignore_index=True,
    )

    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )

    importance_summary.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        args.output_dir / "feature_importance_by_fold.csv",
        index=False,
    )

    all_feature_frequency_importance.to_csv(
        args.output_dir / "all_feature_frequency_importance.csv",
        index=False,
    )

    diagnostics.to_csv(
        args.output_dir / "te_diagnostics.csv",
        index=False,
    )

    logistic_diagnostics = pd.concat(
        logistic_diagnostic_frames,
        ignore_index=True,
    )

    logistic_diagnostics.to_csv(
        args.output_dir / "learned_logistic_margin_coefficients.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "recipe_score": train_recipe_score,
            "recipe_probability": train_recipe_probability,
            "recipe_base_margin": train_recipe_margin,
        }
    ).to_csv(
        args.output_dir / "recipe_margin_diagnostics.csv",
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

    if (
        delta_vs_learned_margin > 0
        and folds_improved_vs_learned_margin >= 3
    ):
        decision = "KEEP_ALL_FEATURE_FREQUENCY_EXTENSION"
    else:
        decision = "REJECT_ALL_FEATURE_FREQUENCY_EXTENSION"

    summary_lines = [
        "EXPERIMENT: XGBOOST m=2 + LEARNED MARGIN + ALL-FEATURE FREQUENCY",
        "=" * 78,
        "",
        "HYPOTHESIS",
        "Can marginal exact-value frequency for the remaining raw features add",
        "synthetic generator-density signal beyond income/commute frequencies?",
        "",
        "ONLY CHANGE",
        "Added one target-free log1p exact-frequency feature for every remaining raw feature.",
        "",
        "HELD FIXED",
        "- frozen 5 folds",
        f"- frozen fold SHA256: {fold_hash}",
        "- original raw features",
        "- exact-value TE of income + commute",
        "- nested leakage-safe TE",
        "- smoothing m=2",
        "- depth-4 XGBoost hyperparameters",
        "- seed 42",
        "",
        "RESULTS",
        f"XGB m=2 baseline OOF: {m2_baseline_auc:.8f}",
        f"Income-digit baseline OOF: {income_digit_baseline_auc:.8f}",
        f"Recipe-margin baseline OOF: {recipe_margin_baseline_auc:.8f}",
        f"Exact-frequency baseline OOF: {exact_frequency_baseline_auc:.8f}",
        f"Learned-margin baseline OOF: {learned_margin_baseline_auc:.8f}",
        f"Fixed recipe-only AUC: {recipe_only_auc:.8f}",
        f"Learned logistic prior OOF AUC: {learned_prior_auc:.8f}",
        f"Candidate OOF: {candidate_auc:.8f}",
        f"Delta vs learned-margin baseline: {delta_vs_learned_margin:+.8f}",
        f"Delta vs exact-frequency baseline: {delta_vs_exact_frequency:+.8f}",
        f"Delta vs recipe-margin baseline: {delta_vs_recipe_margin:+.8f}",
        f"Delta vs income-digit baseline: {delta_vs_income_digits:+.8f}",
        f"Delta vs plain m=2: {delta_vs_m2:+.8f}",
        f"Folds improved vs learned-margin baseline: {folds_improved_vs_learned_margin}/5",
        f"Folds worse vs learned-margin baseline: {folds_worse_vs_learned_margin}/5",
        "",
        f"CatBoost 3-seed OOF: {catboost_auc:.8f}",
        f"Delta vs CatBoost: {delta_vs_cat:+.8f}",
        "",
        f"Probability corr vs learned-margin XGB: {probability_corr_vs_learned_margin:.6f}",
        f"Rank corr vs learned-margin XGB: {rank_corr_vs_learned_margin:.6f}",
        f"Probability corr vs exact-frequency XGB: {probability_corr_vs_exact_frequency:.6f}",
        f"Rank corr vs exact-frequency XGB: {rank_corr_vs_exact_frequency:.6f}",
        f"Probability corr vs recipe-margin XGB: {probability_corr_vs_recipe_margin:.6f}",
        f"Rank corr vs recipe-margin XGB: {rank_corr_vs_recipe_margin:.6f}",
        f"Probability corr vs income-digit XGB: {probability_corr_vs_income_digits:.6f}",
        f"Rank corr vs income-digit XGB: {rank_corr_vs_income_digits:.6f}",
        f"Probability corr vs m=2 XGB: {probability_corr_vs_m2:.6f}",
        f"Rank corr vs m=2 XGB: {rank_corr_vs_m2:.6f}",
        f"Probability corr vs CatBoost: {probability_corr_vs_cat:.6f}",
        f"Rank corr vs CatBoost: {rank_corr_vs_cat:.6f}",
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
            f"learned_margin={row.learned_margin_baseline_auc:.8f} -> "
            f"+all_frequency={row.digit_candidate_auc:.8f} "
            f"({row.delta_vs_learned_margin:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "ALL-FEATURE FREQUENCY EXTENSION IMPORTANCE",
        ]
    )

    for row in all_feature_frequency_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.6f}"
        )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("LEARNED MARGIN + ALL-FEATURE FREQUENCY EXPERIMENT COMPLETE")
    print("=" * 88)
    print(f"XGB m=2 baseline       : {m2_baseline_auc:.8f}")
    print(f"Income-digit baseline  : {income_digit_baseline_auc:.8f}")
    print(f"Recipe-margin baseline : {recipe_margin_baseline_auc:.8f}")
    print(f"Exact-frequency baseline: {exact_frequency_baseline_auc:.8f}")
    print(f"Learned-margin baseline : {learned_margin_baseline_auc:.8f}")
    print(f"Fixed recipe-only AUC   : {recipe_only_auc:.8f}")
    print(f"Learned prior OOF AUC   : {learned_prior_auc:.8f}")
    print(f"+ all-feature frequency : {candidate_auc:.8f}")
    print(f"Delta vs learned margin : {delta_vs_learned_margin:+.8f}")
    print(f"Delta vs exact freq     : {delta_vs_exact_frequency:+.8f}")
    print(f"Delta vs recipe margin : {delta_vs_recipe_margin:+.8f}")
    print(f"Delta vs income digits : {delta_vs_income_digits:+.8f}")
    print(f"Delta vs plain m=2     : {delta_vs_m2:+.8f}")
    print(f"Folds improved          : {folds_improved_vs_learned_margin}/5")
    print(f"Folds worse             : {folds_worse_vs_learned_margin}/5")
    print(f"Decision         : {decision}")
    print(f"Runtime          : {total_seconds:.2f}s")
    print(f"Artifacts        : {args.output_dir.resolve()}")
    print()
    print("All-feature frequency-extension importances:")
    if len(all_feature_frequency_importance):
        print(
            all_feature_frequency_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(index=False)
        )
    print("=" * 88)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. all_feature_frequency_importance.csv")


if __name__ == "__main__":
    main()
