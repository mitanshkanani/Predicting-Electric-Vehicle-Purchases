"""
zoom_zoom_r_frozen5_lightgbm.py

Kaggle Playground Series S6E9
External-recipe replication: "Zoom Zoom R Edition"

PURPOSE
-------
The uploaded R notebook reports one held-out validation AUC, not full 5-fold OOF.
This script adapts its COMPLETE standalone model view to our frozen 5-fold
validation so we can judge it honestly and measure diversity.

ONE MEANINGFUL CHANGE
---------------------
Introduce one new standalone model recipe:
    Zoom Zoom R feature view + LightGBM.

This does NOT modify the current internal champion.

SOURCE RECIPE REPRODUCED
------------------------
13 raw mapped features
+ 8 decimal-digit features for each raw column
+ normalized frequency encoding for each raw column
+ leakage-safe target encoding for each raw column (smooth=10)
+ Annual_Income_USD q1280 position / central TE / neighbor TE

Total = 146 features.

LightGBM source parameters:
    learning_rate      = 0.02
    max_depth          = 5
    num_leaves         = 32
    min_data_in_leaf   = 10
    feature_fraction   = 0.3
    bagging_fraction   = 0.8
    bagging_freq       = 1
    lambda_l1          = 0.07
    lambda_l2          = 2.0
    max_bin            = 1024
    seed               = 60

VALIDATION ADAPTATION
---------------------
Outer validation:
    exact frozen 5 folds used by this project.

Inside each outer-training partition:
    3-fold StratifiedKFold, shuffle=True, random_state=51
    is used only to create train-side target encodings.

This preserves the source notebook's 3-inner-fold concept while keeping all
outer-validation labels completely out of feature construction.

The source R notebook's custom R fold sampler is not bit-identical to
scikit-learn's sampler, so this is a faithful methodological adaptation rather
than a byte-for-byte R reproduction.

No public leaderboard information is used.
No blend is created.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "\nLightGBM is required.\n"
        "Install it with:\n"
        "    python -m pip install -U lightgbm\n"
    ) from exc


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

INTERNAL_CHAMPION_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "zoom_zoom_r_frozen5_lightgbm"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_INTERNAL_CHAMPION_AUC = 0.94611253
AUC_TOL = 5e-6

INNER_FOLDS = 3
INNER_SEED = 51
MODEL_SEED = 60

Q_BINS = 1280
TE_SMOOTHING = 10.0
Q_KERNEL_SIGMA = 0.8

DIGIT_POWERS = tuple(range(-4, 4))

CATEGORY_MAPS = {
    "Gender": {
        "Female": 0.0,
        "Male": 1.0,
        "Other": 2.0,
    },
    "City_Type": {
        "Rural": 0.0,
        "Suburban": 1.0,
        "Urban": 2.0,
    },
    "Current_Car_Type": {
        "Hatchback": 0.0,
        "Sedan": 1.0,
        "SUV": 2.0,
        "Truck": 3.0,
    },
    "Home_Charging_Possible": {
        "No": 0.0,
        "Yes": 1.0,
    },
    "Subsidy_Available": {
        "No": 0.0,
        "Yes": 1.0,
    },
    "Range_Anxiety_Level": {
        "High": 0.0,
        "Low": 1.0,
        "Medium": 2.0,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    train_only = [
        c for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise RuntimeError(
            f"Expected exactly one train-only column; found {train_only}"
        )

    return train_only[0]


def encode_target(y: pd.Series) -> np.ndarray:
    vals = list(pd.unique(y.dropna()))

    if len(vals) != 2:
        raise RuntimeError(
            f"Expected binary target; found {vals}"
        )

    yes = [
        v for v in vals
        if str(v).strip().lower() == "yes"
    ]

    positive = (
        yes[0]
        if yes
        else y.value_counts().idxmin()
    )

    return (
        y == positive
    ).astype(np.int8).to_numpy()


def canonical_keys(s: pd.Series) -> pd.Series:
    return (
        s.astype("object")
        .where(s.notna(), "__MISSING__")
        .astype(str)
        .reset_index(drop=True)
    )


def map_raw_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(frame)))

    for col in frame.columns:
        if col in CATEGORY_MAPS:
            vals = (
                frame[col]
                .astype(str)
                .map(CATEGORY_MAPS[col])
            )

            # Match R behavior conceptually: unknown categories become NA.
            out[col] = pd.to_numeric(
                vals,
                errors="coerce",
            ).astype(np.float64)
        else:
            out[col] = pd.to_numeric(
                frame[col],
                errors="raise",
            ).astype(np.float64)

    return out


def frequency_encode(
    fit_keys: pd.Series,
    new_keys: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    counts = fit_keys.value_counts(
        dropna=False
    )

    denom = float(len(fit_keys))

    fit_values = (
        fit_keys.map(counts)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        / denom
    )

    new_values = (
        new_keys.map(counts)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        / denom
    )

    return (
        fit_values.astype(np.float32),
        new_values.astype(np.float32),
    )


def fit_target_mapping(
    keys: pd.Series,
    y: np.ndarray,
    smoothing: float,
) -> tuple[pd.Series, float]:
    prior = float(y.mean())

    frame = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "target": y,
        }
    )

    stats = (
        frame.groupby(
            "key",
            observed=True,
        )["target"]
        .agg(["sum", "count"])
    )

    encoded = (
        stats["sum"] + smoothing * prior
    ) / (
        stats["count"] + smoothing
    )

    return encoded, prior


def apply_target_mapping(
    keys: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> np.ndarray:
    return (
        keys.map(mapping)
        .fillna(prior)
        .to_numpy(dtype=np.float32)
    )


def make_inner_splits(
    y: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=INNER_SEED,
    )

    dummy = np.zeros(
        len(y),
        dtype=np.uint8,
    )

    return list(
        splitter.split(
            dummy,
            y,
        )
    )


def exact_target_encode(
    fit_keys: pd.Series,
    fit_y: np.ndarray,
    new_keys: pd.Series,
    inner_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(
        len(fit_y),
        np.nan,
        dtype=np.float32,
    )

    for inner_fit, inner_valid in inner_splits:
        mapping, prior = fit_target_mapping(
            fit_keys.iloc[
                inner_fit
            ].reset_index(drop=True),
            fit_y[inner_fit],
            TE_SMOOTHING,
        )

        oof[
            inner_valid
        ] = apply_target_mapping(
            fit_keys.iloc[
                inner_valid
            ].reset_index(drop=True),
            mapping,
            prior,
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Inner-OOF target encoding contains NaNs."
        )

    mapping, prior = fit_target_mapping(
        fit_keys,
        fit_y,
        TE_SMOOTHING,
    )

    new_encoded = apply_target_mapping(
        new_keys,
        mapping,
        prior,
    )

    return oof, new_encoded


def make_quantile_edges(
    fit_income: np.ndarray,
) -> np.ndarray:
    probs = np.linspace(
        0.0,
        1.0,
        Q_BINS + 1,
        dtype=np.float64,
    )

    edges = np.quantile(
        fit_income,
        probs,
        method="linear",
    )

    edges = np.unique(
        edges.astype(np.float64)
    )

    if len(edges) < 3:
        raise RuntimeError(
            "Too few unique q1280 edges."
        )

    return edges


def assign_q_codes(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    interior = edges[1:-1]

    # R source:
    # findInterval(values, interior, left.open = TRUE) + 1L
    # We retain zero-based codes in Python.
    return np.searchsorted(
        interior,
        values,
        side="left",
    ).astype(np.int32)


def fit_q_maps(
    codes: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    prior = float(y.mean())

    counts = np.bincount(
        codes,
        minlength=n_bins,
    ).astype(np.float64)

    sums = np.bincount(
        codes,
        weights=y.astype(np.float64),
        minlength=n_bins,
    ).astype(np.float64)

    center = (
        sums + TE_SMOOTHING * prior
    ) / (
        counts + TE_SMOOTHING
    )

    offsets = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    kernel = np.exp(
        -0.5
        * (
            offsets / Q_KERNEL_SIGMA
        ) ** 2
    )

    left_sums = np.concatenate(
        ([0.0], sums[:-1])
    )
    right_sums = np.concatenate(
        (sums[1:], [0.0])
    )

    left_counts = np.concatenate(
        ([0.0], counts[:-1])
    )
    right_counts = np.concatenate(
        (counts[1:], [0.0])
    )

    neighbor_sums = (
        kernel[0] * left_sums
        + kernel[1] * sums
        + kernel[2] * right_sums
    )

    neighbor_counts = (
        kernel[0] * left_counts
        + kernel[1] * counts
        + kernel[2] * right_counts
    )

    neighbor = (
        neighbor_sums
        + TE_SMOOTHING
        * float(kernel.sum())
        * prior
    ) / (
        neighbor_counts
        + TE_SMOOTHING
        * float(kernel.sum())
    )

    return center, neighbor


def neighbor_qte(
    fit_income: np.ndarray,
    fit_y: np.ndarray,
    new_income: np.ndarray,
    inner_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    edges = make_quantile_edges(
        fit_income
    )

    fit_codes = assign_q_codes(
        fit_income,
        edges,
    )

    new_codes = assign_q_codes(
        new_income,
        edges,
    )

    # The R source allocates n_bins = length(edges), which is one slot
    # larger than the maximum occupied code when all quantiles are unique.
    # Keeping the same allocation is harmless and reproduces the boundary
    # behavior of its neighbor kernel.
    n_bins = len(edges)

    fit_center = np.full(
        len(fit_y),
        np.nan,
        dtype=np.float32,
    )

    fit_neighbor = np.full(
        len(fit_y),
        np.nan,
        dtype=np.float32,
    )

    for inner_fit, inner_valid in inner_splits:
        center_map, neighbor_map = (
            fit_q_maps(
                fit_codes[inner_fit],
                fit_y[inner_fit],
                n_bins,
            )
        )

        fit_center[
            inner_valid
        ] = center_map[
            fit_codes[
                inner_valid
            ]
        ].astype(np.float32)

        fit_neighbor[
            inner_valid
        ] = neighbor_map[
            fit_codes[
                inner_valid
            ]
        ].astype(np.float32)

    if (
        np.isnan(fit_center).any()
        or np.isnan(fit_neighbor).any()
    ):
        raise RuntimeError(
            "Inner-OOF neighbor QTE contains NaNs."
        )

    center_map, neighbor_map = (
        fit_q_maps(
            fit_codes,
            fit_y,
            n_bins,
        )
    )

    new_center = center_map[
        new_codes
    ].astype(np.float32)

    new_neighbor = neighbor_map[
        new_codes
    ].astype(np.float32)

    fit_position = (
        fit_codes.astype(np.float64)
        / max(Q_BINS - 1, 1)
    ).astype(np.float32)

    new_position = (
        new_codes.astype(np.float64)
        / max(Q_BINS - 1, 1)
    ).astype(np.float32)

    fit_features = np.column_stack(
        [
            fit_position,
            fit_center,
            fit_neighbor,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    new_features = np.column_stack(
        [
            new_position,
            new_center,
            new_neighbor,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    names = [
        "income_q1280_pos",
        "income_q1280_center",
        "income_q1280_neighbor",
    ]

    return (
        fit_features,
        new_features,
        names,
    )


def build_zoom_features(
    fit_raw: pd.DataFrame,
    fit_y: np.ndarray,
    new_raw: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Reproduce strong_local_features() from the R notebook.

    Returns float32 matrices with exactly 146 columns.
    """
    fit_raw = (
        fit_raw
        .reset_index(drop=True)
        .copy()
    )

    new_raw = (
        new_raw
        .reset_index(drop=True)
        .copy()
    )

    mapped_fit = map_raw_numeric(
        fit_raw
    )

    mapped_new = map_raw_numeric(
        new_raw
    )

    fit_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    feature_names: list[str] = []

    # First: 13 mapped raw columns.
    for col in fit_raw.columns:
        fit_parts.append(
            mapped_fit[col]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )

        new_parts.append(
            mapped_new[col]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )

        feature_names.append(col)

    # Then source loop:
    # for each column -> eight digit features + one frequency feature.
    for col in fit_raw.columns:
        fit_num = mapped_fit[
            col
        ].to_numpy(dtype=np.float64)

        new_num = mapped_new[
            col
        ].to_numpy(dtype=np.float64)

        for power in DIGIT_POWERS:
            scale = 10.0 ** power

            fit_digit = (
                np.floor(
                    fit_num / scale
                ) % 10.0
            ).astype(np.float32)

            new_digit = (
                np.floor(
                    new_num / scale
                ) % 10.0
            ).astype(np.float32)

            fit_parts.append(
                fit_digit.reshape(-1, 1)
            )

            new_parts.append(
                new_digit.reshape(-1, 1)
            )

            feature_names.append(
                f"{col}_digit{power}"
            )

        fit_keys = canonical_keys(
            fit_raw[col]
        )

        new_keys = canonical_keys(
            new_raw[col]
        )

        fit_freq, new_freq = (
            frequency_encode(
                fit_keys,
                new_keys,
            )
        )

        fit_parts.append(
            fit_freq.reshape(-1, 1)
        )

        new_parts.append(
            new_freq.reshape(-1, 1)
        )

        feature_names.append(
            f"{col}_freq"
        )

    inner_splits = make_inner_splits(
        fit_y
    )

    # Then one leakage-safe TE for each original raw column.
    for col in fit_raw.columns:
        fit_keys = canonical_keys(
            fit_raw[col]
        )

        new_keys = canonical_keys(
            new_raw[col]
        )

        fit_te, new_te = (
            exact_target_encode(
                fit_keys,
                fit_y,
                new_keys,
                inner_splits,
            )
        )

        fit_parts.append(
            fit_te.reshape(-1, 1)
        )

        new_parts.append(
            new_te.reshape(-1, 1)
        )

        feature_names.append(
            f"{col}_te"
        )

    # Finally q1280 income position / central / neighbor TE.
    fit_income = pd.to_numeric(
        fit_raw[
            "Annual_Income_USD"
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    new_income = pd.to_numeric(
        new_raw[
            "Annual_Income_USD"
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    (
        fit_qte,
        new_qte,
        qte_names,
    ) = neighbor_qte(
        fit_income,
        fit_y,
        new_income,
        inner_splits,
    )

    fit_parts.append(fit_qte)
    new_parts.append(new_qte)
    feature_names.extend(qte_names)

    X_fit = np.column_stack(
        fit_parts
    ).astype(
        np.float32,
        copy=False,
    )

    X_new = np.column_stack(
        new_parts
    ).astype(
        np.float32,
        copy=False,
    )

    if X_fit.shape[1] != 146:
        raise RuntimeError(
            f"Expected 146 source features; "
            f"built {X_fit.shape[1]}."
        )

    if X_new.shape[1] != 146:
        raise RuntimeError(
            f"Expected 146 new-row features; "
            f"built {X_new.shape[1]}."
        )

    if not (
        np.isfinite(
            X_fit[
                np.isfinite(X_fit)
            ]
        ).all()
        and np.isfinite(
            X_new[
                np.isfinite(X_new)
            ]
        ).all()
    ):
        raise RuntimeError(
            "Unexpected non-finite feature values."
        )

    return (
        X_fit,
        X_new,
        feature_names,
    )


def load_internal_champion(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(
        INTERNAL_CHAMPION_OOF
    )

    if len(df) != len(train):
        raise RuntimeError(
            "Internal champion OOF row-count mismatch."
        )

    if (
        "row_index" not in df.columns
        or "fold" not in df.columns
        or "oof_prediction" not in df.columns
    ):
        raise RuntimeError(
            "Internal champion OOF schema mismatch."
        )

    if not np.array_equal(
        df[
            "row_index"
        ].to_numpy(dtype=np.int64),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Internal champion row_index mismatch."
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(dtype=np.int64),
        folds,
    ):
        raise RuntimeError(
            "Internal champion frozen-fold mismatch."
        )

    p = pd.to_numeric(
        df["oof_prediction"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    found_auc = float(
        roc_auc_score(
            encode_target(
                train[
                    detect_target(
                        train,
                        pd.read_csv(
                            TEST_PATH,
                            nrows=1,
                        ),
                    )
                ]
            ),
            p,
        )
    )

    if abs(
        found_auc
        - EXPECTED_INTERNAL_CHAMPION_AUC
    ) > AUC_TOL:
        raise RuntimeError(
            "Internal champion AUC mismatch.\n"
            f"Expected: {EXPECTED_INTERNAL_CHAMPION_AUC:.8f}\n"
            f"Found:    {found_auc:.8f}"
        )

    return p


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
        .to_numpy(dtype=np.float64)
    )

    br = (
        pd.Series(b)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(dtype=np.float64)
    )

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.02,
        max_depth=5,
        num_leaves=32,
        min_child_samples=10,
        colsample_bytree=0.3,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.07,
        reg_lambda=2.0,
        max_bin=1024,
        n_estimators=20000,
        random_state=MODEL_SEED,
        n_jobs=max(
            1,
            os.cpu_count() or 4,
        ),
        verbosity=-1,
    )


def main() -> None:
    total_start = time.perf_counter()

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_CHAMPION_OOF,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(
        TRAIN_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    if len(folds_df) != len(train):
        raise RuntimeError(
            "Frozen fold row-count mismatch."
        )

    if "fold" not in folds_df.columns:
        raise RuntimeError(
            "Frozen fold file has no 'fold' column."
        )

    folds = folds_df[
        "fold"
    ].to_numpy(dtype=np.int64)

    target = detect_target(
        train,
        test,
    )

    y = encode_target(
        train[target]
    )

    if "id" not in train.columns:
        raise RuntimeError(
            "Expected competition ID column 'id'."
        )

    predictors = [
        c
        for c in train.columns
        if c not in {"id", target}
    ]

    if len(predictors) != 13:
        raise RuntimeError(
            f"Zoom Zoom source expects 13 predictors; "
            f"found {len(predictors)}: {predictors}"
        )

    if predictors != [
        c
        for c in test.columns
        if c != "id"
    ]:
        raise RuntimeError(
            "Train/test predictor order mismatch."
        )

    internal_oof = (
        load_internal_champion(
            train,
            folds,
        )
    )

    internal_auc = float(
        roc_auc_score(
            y,
            internal_oof,
        )
    )

    print("=" * 108)
    print("ZOOM ZOOM R EDITION — FROZEN 5-FOLD LIGHTGBM REPLICATION")
    print("=" * 108)
    print(f"LightGBM version        : {lgb.__version__}")
    print(f"Frozen fold SHA256      : {fold_hash}")
    print(f"Rows                    : {len(train):,} train / {len(test):,} test")
    print(f"Predictors              : {len(predictors)}")
    print(f"Source feature count    : 146")
    print(f"Internal champion ref   : {internal_auc:.8f}")
    print()
    print("MODEL VIEW:")
    print("  13 mapped raw features")
    print("  + 8 digit features per raw column")
    print("  + normalized frequency encoding per raw column")
    print("  + leakage-safe TE per raw column")
    print("  + q1280 income position/center/neighbor TE")
    print("  + source LightGBM parameters")
    print()
    print("VALIDATION:")
    print("  frozen project outer folds")
    print("  3-fold leakage-safe inner TE construction")
    print("  no public-LB information")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_predictions: list[
        np.ndarray
    ] = []

    fold_rows: list[dict] = []
    importance_rows: list[dict] = []

    raw_train = train[
        predictors
    ].copy()

    raw_test = test[
        predictors
    ].copy()

    for outer_fold in sorted(
        np.unique(folds).tolist()
    ):
        fold_start = (
            time.perf_counter()
        )

        train_idx = np.flatnonzero(
            folds != outer_fold
        )

        valid_idx = np.flatnonzero(
            folds == outer_fold
        )

        # Transform outer validation and test together so both see exactly the
        # same mappings fitted from outer-training only.
        new_raw = pd.concat(
            [
                raw_train.iloc[
                    valid_idx
                ],
                raw_test,
            ],
            axis=0,
            ignore_index=True,
        )

        feature_start = (
            time.perf_counter()
        )

        (
            X_fit,
            X_new,
            feature_names,
        ) = build_zoom_features(
            fit_raw=raw_train.iloc[
                train_idx
            ],
            fit_y=y[train_idx],
            new_raw=new_raw,
        )

        feature_seconds = (
            time.perf_counter()
            - feature_start
        )

        n_valid = len(valid_idx)

        X_valid = X_new[
            :n_valid
        ]

        X_test = X_new[
            n_valid:
        ]

        if (
            X_test.shape[0]
            != len(test)
        ):
            raise RuntimeError(
                "Test feature row-count mismatch."
            )

        model = build_model()

        fit_start = (
            time.perf_counter()
        )

        model.fit(
            X_fit,
            y[train_idx],
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    verbose=False,
                )
            ],
            feature_name=feature_names,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        best_iteration = int(
            model.best_iteration_
        )

        valid_pred = (
            model.predict_proba(
                X_valid,
                num_iteration=best_iteration,
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test,
                num_iteration=best_iteration,
            )[:, 1]
        )

        oof[
            valid_idx
        ] = valid_pred

        test_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        internal_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                internal_oof[
                    valid_idx
                ],
            )
        )

        delta = (
            candidate_fold_auc
            - internal_fold_auc
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": int(
                    outer_fold
                ),
                "internal_champion_auc":
                    internal_fold_auc,
                "zoom_zoom_auc":
                    candidate_fold_auc,
                "delta_vs_internal":
                    delta,
                "best_iteration":
                    best_iteration,
                "feature_seconds":
                    feature_seconds,
                "fit_seconds":
                    fit_seconds,
                "total_fold_seconds":
                    fold_seconds,
            }
        )

        importances = (
            model.booster_
            .feature_importance(
                importance_type="gain"
            )
        )

        for feature, importance in zip(
            feature_names,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": int(
                        outer_fold
                    ),
                    "feature": feature,
                    "gain_importance":
                        float(
                            importance
                        ),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"internal={internal_fold_auc:.8f} | "
            f"zoom={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={best_iteration} | "
            f"features={feature_seconds:.1f}s | "
            f"fit={fit_seconds:.1f}s"
        )

        del (
            X_fit,
            X_new,
            X_valid,
            X_test,
            new_raw,
            model,
        )

        gc.collect()

    if np.isnan(oof).any():
        raise RuntimeError(
            "Zoom Zoom OOF contains NaNs."
        )

    zoom_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_internal = (
        zoom_auc
        - internal_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    folds_better = int(
        (
            fold_metrics[
                "delta_vs_internal"
            ] > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "delta_vs_internal"
            ] < 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            internal_oof,
        )[0, 1]
    )

    rank_correlation = rank_corr(
        oof,
        internal_oof,
    )

    test_prediction = np.mean(
        np.vstack(
            test_predictions
        ),
        axis=0,
    )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    importance_by_fold = pd.DataFrame(
        importance_rows
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_gain=(
                "gain_importance",
                "mean",
            ),
            std_gain=(
                "gain_importance",
                "std",
            ),
        )
        .sort_values(
            "mean_gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    importance_summary.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "id": train[
                "id"
            ].to_numpy(),
            "fold": folds,
            "target_encoded": y,
            "oof_prediction":
                oof.astype(
                    np.float32
                ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "id": test[
                "id"
            ].to_numpy(),
            "prediction":
                test_prediction.astype(
                    np.float32
                ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    if (
        zoom_auc >= 0.94610
        and rank_correlation <= 0.9985
    ):
        ensemble_priority = (
            "HIGH_DIVERSITY_AUDIT_PRIORITY"
        )
    elif zoom_auc >= 0.94600:
        ensemble_priority = (
            "POSSIBLE_DIVERSITY_AUDIT"
        )
    else:
        ensemble_priority = (
            "LOW_PRIORITY"
        )

    pd.DataFrame(
        [
            {
                "internal_champion_auc":
                    internal_auc,
                "zoom_zoom_oof_auc":
                    zoom_auc,
                "delta_vs_internal":
                    delta_vs_internal,
                "folds_better":
                    folds_better,
                "folds_worse":
                    folds_worse,
                "probability_corr_vs_internal":
                    probability_corr,
                "rank_corr_vs_internal":
                    rank_correlation,
                "ensemble_priority":
                    ensemble_priority,
                "runtime_seconds":
                    total_seconds,
            }
        ]
    ).to_csv(
        OUTPUT_DIR
        / "audit_metrics.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: ZOOM ZOOM R EDITION — FROZEN 5-FOLD REPLICATION",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target external-recipe replication.",
        "",
        "HYPOTHESIS",
        (
            "Does the complete 146-feature Zoom Zoom R LightGBM view "
            "remain strong under our full frozen 5-fold OOF validation?"
        ),
        "",
        "ONE MEANINGFUL CHANGE",
        (
            "Introduce the uploaded Zoom Zoom R recipe as one new standalone "
            "model view. Existing champion predictions are unchanged."
        ),
        "",
        "VALIDATION",
        f"- frozen fold SHA256: {fold_hash}",
        "- 5 frozen outer folds",
        "- 3-fold leakage-safe inner target encodings",
        "- no public leaderboard information",
        "",
        "SOURCE RECIPE",
        "- 13 mapped raw features",
        "- 8 digit features per raw column",
        "- normalized frequency encoding per raw column",
        "- target encoding per raw column, smoothing=10",
        "- q1280 income position/center/neighbor target encoding",
        "- LightGBM depth=5, leaves=32, lr=0.02, max_bin=1024",
        "",
        "RESULT",
        f"Internal champion reference OOF: {internal_auc:.8f}",
        f"Zoom Zoom frozen-5 OOF: {zoom_auc:.8f}",
        f"Delta vs internal champion: {delta_vs_internal:+.8f}",
        f"Folds better than internal: {folds_better}/5",
        f"Folds worse than internal: {folds_worse}/5",
        f"Probability corr vs internal: {probability_corr:.6f}",
        f"Rank corr vs internal: {rank_correlation:.6f}",
        f"Ensemble priority: {ensemble_priority}",
        f"Runtime: {total_seconds:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"internal={r.internal_champion_auc:.8f}, "
            f"zoom={r.zoom_zoom_auc:.8f}, "
            f"delta={r.delta_vs_internal:+.8f}, "
            f"best_iter={r.best_iteration}"
        )

    summary.extend(
        [
            "",
            "TOP 20 FEATURES BY MEAN GAIN",
        ]
    )

    for r in (
        importance_summary
        .head(20)
        .itertuples()
    ):
        summary.append(
            f"{r.feature}: "
            f"{r.mean_gain:.6f}"
        )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("ZOOM ZOOM FROZEN-5 REPLICATION COMPLETE")
    print("=" * 108)
    print(f"Internal champion OOF : {internal_auc:.8f}")
    print(f"Zoom Zoom OOF         : {zoom_auc:.8f}")
    print(f"Delta vs internal     : {delta_vs_internal:+.8f}")
    print(f"Folds better          : {folds_better}/5")
    print(f"Folds worse           : {folds_worse}/5")
    print(f"Probability corr      : {probability_corr:.6f}")
    print(f"Rank corr             : {rank_correlation:.6f}")
    print(f"Ensemble priority     : {ensemble_priority}")
    print(f"Runtime               : {total_seconds:.2f}s")
    print(f"Artifacts             : {OUTPUT_DIR}")
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. audit_metrics.csv")
    print("  5. feature_importance.csv")
    print("=" * 108)


if __name__ == "__main__":
    main()
