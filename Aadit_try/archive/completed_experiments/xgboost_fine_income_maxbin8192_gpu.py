"""
xgboost_fine_income_maxbin8192_gpu.py

Kaggle Playground Series S6E9

CONTROLLED HIGH-RESOLUTION HISTOGRAM EXPERIMENT
-----------------------------------------------
Current baseline:
    fine-income XGBoost OOF = 0.94606664

HYPOTHESIS
----------
Annual_Income_USD has very high cardinality and income resolution has repeatedly
been one of the strongest signals in this competition.

Our current XGBoost uses:
    max_bin = 256

A much larger histogram budget may preserve more of the fine income structure
that is otherwise compressed before tree construction.

ONLY CHANGE
-----------
    max_bin: 256 -> 8192

HELD FIXED
----------
- complete current fine-income XGBoost representation
- raw features
- income digits
- exact income/commute frequency features
- exact income/commute nested Bayesian TE
- hierarchical income TE at $1k/$10k/$100k
- hierarchical commute TE at 1/5/10 km
- fine income TE at $50/$250
- nested learned logistic base margin
- smoothing m=2
- max_depth=4
- min_child_weight
- learning_rate
- subsample
- colsample_bytree
- regularization
- seed 42
- CUDA
- frozen 5 folds + SHA256
- early stopping
- no public leaderboard optimization

NO interaction constraints.
NO feature changes.
NO parameter grid.

This is intentionally one controlled structural change.

Run:
    python xgboost_fine_income_maxbin8192_gpu.py
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

try:
    import lightgbm_engineered_learned_margin_cpu as base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import lightgbm_engineered_learned_margin_cpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc

try:
    import xgboost_hierarchical_income_te_gpu as income_hte
except ImportError as exc:
    raise SystemExit(
        "\nCould not import xgboost_hierarchical_income_te_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc


SEED = 42
SMOOTHING = 2.0

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94606664
AUC_CHECK_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_fine_income_te_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_fine_income_maxbin8192_gpu"
)

COMMUTE_BUCKET_SPECS = [
    ("HTE__commute_1km_bucket", 1.0),
    ("HTE__commute_5km_bucket", 5.0),
    ("HTE__commute_10km_bucket", 10.0),
]

FINE_INCOME_BUCKET_SPECS = [
    ("HTE__income_50_bucket", 50.0),
    ("HTE__income_250_bucket", 250.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Current hierarchical-income + commute XGB + fine $50/$250 income TE."
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


def make_commute_bucket_key(
    series: pd.Series,
    width_km: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numeric).all():
        raise ValueError(
            "Daily_Commute_km contains non-finite values."
        )

    if (numeric < 0).any():
        raise ValueError(
            "Daily_Commute_km unexpectedly contains negative values."
        )

    bucket = np.floor(
        numeric / width_km
    ).astype(np.int64)

    return pd.Series(
        bucket.astype(str),
        index=series.index,
        dtype="object",
    )


def make_income_bucket_key(
    series: pd.Series,
    width_usd: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numeric).all():
        raise ValueError(
            "Annual_Income_USD contains non-finite values."
        )

    if (numeric < 0).any():
        raise ValueError(
            "Annual_Income_USD unexpectedly contains negative values."
        )

    bucket = np.floor(
        numeric / width_usd
    ).astype(np.int64)

    return pd.Series(
        bucket.astype(str),
        index=series.index,
        dtype="object",
    )


def fit_bayesian_mapping(
    keys: pd.Series,
    y: np.ndarray,
    prior: float,
    smoothing: float,
) -> pd.DataFrame:
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
        .agg(["sum", "count"])
    )

    stats["encoded"] = (
        stats["sum"]
        + smoothing * prior
    ) / (
        stats["count"]
        + smoothing
    )

    return stats


def apply_mapping(
    keys: pd.Series,
    mapping: pd.DataFrame,
    prior: float,
) -> tuple[np.ndarray, float]:
    encoded = keys.map(
        mapping["encoded"]
    )

    unseen_rate = float(
        encoded.isna().mean()
    )

    return (
        encoded
        .fillna(prior)
        .to_numpy(dtype=np.float32),
        unseen_rate,
    )


def build_hierarchical_commute_te_for_outer_fold(
    *,
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
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    outer_train_folds = fold_ids[
        outer_train_idx
    ]

    train_out = pd.DataFrame(
        index=np.arange(
            len(outer_train_idx)
        )
    )

    valid_out = pd.DataFrame(
        index=np.arange(
            len(outer_valid_idx)
        )
    )

    test_out = pd.DataFrame(
        index=np.arange(
            len(test)
        )
    )

    diagnostics = []

    for feature_name, width_km in COMMUTE_BUCKET_SPECS:
        full_train_keys = make_commute_bucket_key(
            train["Daily_Commute_km"],
            width_km,
        )

        test_keys = make_commute_bucket_key(
            test["Daily_Commute_km"],
            width_km,
        )

        outer_train_keys = (
            full_train_keys.iloc[
                outer_train_idx
            ]
            .reset_index(drop=True)
        )

        outer_valid_keys = (
            full_train_keys.iloc[
                outer_valid_idx
            ]
            .reset_index(drop=True)
        )

        outer_train_y = y[
            outer_train_idx
        ]

        prior = float(
            outer_train_y.mean()
        )

        inner_oof = np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )

        # Use the remaining frozen folds as the inner OOF structure.
        # This is deterministic and keeps each outer-training row's own target
        # out of its commute target encoding.
        for inner_fold in sorted(
            np.unique(
                outer_train_folds
            ).tolist()
        ):
            inner_valid_mask = (
                outer_train_folds
                == inner_fold
            )

            inner_fit_mask = (
                ~inner_valid_mask
            )

            inner_prior = float(
                outer_train_y[
                    inner_fit_mask
                ].mean()
            )

            mapping = fit_bayesian_mapping(
                keys=outer_train_keys.loc[
                    inner_fit_mask
                ].reset_index(drop=True),
                y=outer_train_y[
                    inner_fit_mask
                ],
                prior=inner_prior,
                smoothing=smoothing,
            )

            encoded, _ = apply_mapping(
                keys=outer_train_keys.loc[
                    inner_valid_mask
                ].reset_index(drop=True),
                mapping=mapping,
                prior=inner_prior,
            )

            inner_oof[
                inner_valid_mask
            ] = encoded

        if np.isnan(
            inner_oof
        ).any():
            raise RuntimeError(
                f"Inner-OOF commute TE contains NaNs for {feature_name}, "
                f"outer fold {outer_fold}."
            )

        outer_mapping = fit_bayesian_mapping(
            keys=outer_train_keys,
            y=outer_train_y,
            prior=prior,
            smoothing=smoothing,
        )

        valid_encoded, valid_unseen = apply_mapping(
            keys=outer_valid_keys,
            mapping=outer_mapping,
            prior=prior,
        )

        test_encoded, test_unseen = apply_mapping(
            keys=test_keys,
            mapping=outer_mapping,
            prior=prior,
        )

        train_out[
            feature_name
        ] = inner_oof

        valid_out[
            feature_name
        ] = valid_encoded

        test_out[
            feature_name
        ] = test_encoded

        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "feature": feature_name,
                "bucket_width_km": width_km,
                "smoothing": smoothing,
                "outer_train_prior": prior,
                "outer_train_unique_keys": int(
                    len(
                        outer_mapping
                    )
                ),
                "outer_train_min_group_count": int(
                    outer_mapping[
                        "count"
                    ].min()
                ),
                "outer_train_median_group_count": float(
                    outer_mapping[
                        "count"
                    ].median()
                ),
                "outer_train_max_group_count": int(
                    outer_mapping[
                        "count"
                    ].max()
                ),
                "validation_unseen_rate": valid_unseen,
                "test_unseen_rate": test_unseen,
            }
        )

    return (
        train_out,
        valid_out,
        test_out,
        pd.DataFrame(
            diagnostics
        ),
    )



def build_fine_income_te_for_outer_fold(
    *,
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
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    outer_train_folds = fold_ids[
        outer_train_idx
    ]

    train_out = pd.DataFrame(
        index=np.arange(
            len(outer_train_idx)
        )
    )

    valid_out = pd.DataFrame(
        index=np.arange(
            len(outer_valid_idx)
        )
    )

    test_out = pd.DataFrame(
        index=np.arange(
            len(test)
        )
    )

    diagnostics = []

    for feature_name, width_usd in FINE_INCOME_BUCKET_SPECS:
        full_train_keys = make_income_bucket_key(
            train["Annual_Income_USD"],
            width_usd,
        )

        test_keys = make_income_bucket_key(
            test["Annual_Income_USD"],
            width_usd,
        )

        outer_train_keys = (
            full_train_keys.iloc[
                outer_train_idx
            ]
            .reset_index(drop=True)
        )

        outer_valid_keys = (
            full_train_keys.iloc[
                outer_valid_idx
            ]
            .reset_index(drop=True)
        )

        outer_train_y = y[
            outer_train_idx
        ]

        prior = float(
            outer_train_y.mean()
        )

        inner_oof = np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )

        # Reuse the remaining frozen folds as the inner OOF structure.
        # Each outer-training row's own target is excluded from the mapping
        # used to encode that row.
        for inner_fold in sorted(
            np.unique(
                outer_train_folds
            ).tolist()
        ):
            inner_valid_mask = (
                outer_train_folds
                == inner_fold
            )

            inner_fit_mask = (
                ~inner_valid_mask
            )

            inner_prior = float(
                outer_train_y[
                    inner_fit_mask
                ].mean()
            )

            mapping = fit_bayesian_mapping(
                keys=outer_train_keys.loc[
                    inner_fit_mask
                ].reset_index(drop=True),
                y=outer_train_y[
                    inner_fit_mask
                ],
                prior=inner_prior,
                smoothing=smoothing,
            )

            encoded, _ = apply_mapping(
                keys=outer_train_keys.loc[
                    inner_valid_mask
                ].reset_index(drop=True),
                mapping=mapping,
                prior=inner_prior,
            )

            inner_oof[
                inner_valid_mask
            ] = encoded

        if np.isnan(
            inner_oof
        ).any():
            raise RuntimeError(
                f"Inner-OOF fine-income TE contains NaNs for {feature_name}, "
                f"outer fold {outer_fold}."
            )

        outer_mapping = fit_bayesian_mapping(
            keys=outer_train_keys,
            y=outer_train_y,
            prior=prior,
            smoothing=smoothing,
        )

        valid_encoded, valid_unseen = apply_mapping(
            keys=outer_valid_keys,
            mapping=outer_mapping,
            prior=prior,
        )

        test_encoded, test_unseen = apply_mapping(
            keys=test_keys,
            mapping=outer_mapping,
            prior=prior,
        )

        train_out[
            feature_name
        ] = inner_oof

        valid_out[
            feature_name
        ] = valid_encoded

        test_out[
            feature_name
        ] = test_encoded

        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "feature": feature_name,
                "bucket_width_usd": width_usd,
                "smoothing": smoothing,
                "outer_train_prior": prior,
                "outer_train_unique_keys": int(
                    len(
                        outer_mapping
                    )
                ),
                "outer_train_min_group_count": int(
                    outer_mapping[
                        "count"
                    ].min()
                ),
                "outer_train_median_group_count": float(
                    outer_mapping[
                        "count"
                    ].median()
                ),
                "outer_train_max_group_count": int(
                    outer_mapping[
                        "count"
                    ].max()
                ),
                "validation_unseen_rate": valid_unseen,
                "test_unseen_rate": test_unseen,
            }
        )

    return (
        train_out,
        valid_out,
        test_out,
        pd.DataFrame(
            diagnostics
        ),
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(
            a
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    br = (
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
            ar,
            br,
        )[0, 1]
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

    required_paths = [
        train_path,
        test_path,
        args.folds_path,
        args.baseline_oof,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = base.sha256_file(
        args.folds_path
    )

    if (
        fold_hash
        != EXPECTED_FOLDS_SHA256
    ):
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

    target = base.detect_target(
        train,
        test,
    )

    y, positive_label = (
        base.encode_binary_target(
            train[
                target
            ]
        )
    )

    fold_ids, id_col = (
        base.validate_folds(
            folds_df,
            train,
        )
    )

    baseline_oof = base.load_oof(
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
            "Loaded baseline OOF does not match the current fine-income XGB baseline.\n"
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
        base.detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    X_base, X_test_base = (
        base.prepare_base_frames(
            train=train,
            test=test,
            raw_features=raw_features,
            categorical_features=raw_categoricals,
        )
    )

    (
        X_income_base,
        X_income_test_base,
        digit_features,
    ) = base.add_income_digit_features(
        X_train=X_base,
        X_test=X_test_base,
        train_source=train,
        test_source=test,
    )

    (
        X_candidate_base,
        X_candidate_test_base,
        exact_frequency_features,
    ) = base.add_exact_frequency_features(
        X_train=X_income_base,
        X_test=X_income_test_base,
        train_source=train,
        test_source=test,
    )

    logistic_train_matrix = (
        base.build_logistic_recipe_matrix(
            train
        )
    )

    logistic_test_matrix = (
        base.build_logistic_recipe_matrix(
            test
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("XGBOOST FINE-INCOME + MAX_BIN 8192")
    print("=" * 100)
    print(f"XGBoost version: {xgb.__version__}")
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Current fine-income XGB baseline: "
        f"{baseline_auc:.8f}"
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  A much larger histogram budget may preserve useful high-cardinality "
        "income resolution that max_bin=256 compresses."
    )
    print()
    print("ONLY CHANGE:")
    print("  max_bin: 256 -> 8192")
    print()
    print("HELD FIXED:")
    print("  - complete current fine-income representation")
    print("  - frozen folds + SHA256")
    print("  - raw features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - hierarchical income TE at $1k/$10k/$100k")
    print("  - hierarchical commute TE at 1km/5km/10km")
    print("  - fine income TE at $50/$250")
    print("  - smoothing m=2")
    print("  - nested learned logistic base margin")
    print("  - all other XGBoost parameters")
    print("  - seed 42")
    print("  - CUDA GPU")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    exact_diag_frames: list[pd.DataFrame] = []
    income_diag_frames: list[pd.DataFrame] = []
    commute_diag_frames: list[pd.DataFrame] = []
    fine_income_diag_frames: list[pd.DataFrame] = []
    logistic_diag_frames: list[pd.DataFrame] = []

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        (
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            exact_diag,
        ) = base.build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        exact_diag_frames.append(
            exact_diag
        )

        (
            train_income_hte,
            valid_income_hte,
            test_income_hte,
            income_diag,
        ) = income_hte.build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        income_diag_frames.append(
            income_diag
        )

        (
            train_commute_hte,
            valid_commute_hte,
            test_commute_hte,
            commute_diag,
        ) = build_hierarchical_commute_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        commute_diag_frames.append(
            commute_diag
        )

        (
            train_fine_income_te,
            valid_fine_income_te,
            test_fine_income_te,
            fine_income_diag,
        ) = build_fine_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        fine_income_diag_frames.append(
            fine_income_diag
        )

        (
            learned_train_margin,
            learned_valid_margin,
            learned_test_margin,
            logistic_diag,
        ) = base.build_learned_logistic_margins_for_outer_fold(
            train_matrix=logistic_train_matrix,
            test_matrix=logistic_test_matrix,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        logistic_diag_frames.append(
            logistic_diag
        )

        X_train = (
            X_candidate_base.iloc[
                train_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_valid = (
            X_candidate_base.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        exact_te_columns = list(
            train_exact_te.columns
        )

        for c in exact_te_columns:
            X_train[c] = train_exact_te[
                c
            ].to_numpy(dtype=np.float32)

            X_valid[c] = valid_exact_te[
                c
            ].to_numpy(dtype=np.float32)

            X_test[c] = test_exact_te[
                c
            ].to_numpy(dtype=np.float32)

        income_hte_columns = list(
            train_income_hte.columns
        )

        for c in income_hte_columns:
            X_train[c] = train_income_hte[
                c
            ].to_numpy(dtype=np.float32)

            X_valid[c] = valid_income_hte[
                c
            ].to_numpy(dtype=np.float32)

            X_test[c] = test_income_hte[
                c
            ].to_numpy(dtype=np.float32)

        commute_hte_columns = list(
            train_commute_hte.columns
        )

        for c in commute_hte_columns:
            X_train[c] = train_commute_hte[
                c
            ].to_numpy(dtype=np.float32)

            X_valid[c] = valid_commute_hte[
                c
            ].to_numpy(dtype=np.float32)

            X_test[c] = test_commute_hte[
                c
            ].to_numpy(dtype=np.float32)

        fine_income_te_columns = list(
            train_fine_income_te.columns
        )

        for c in fine_income_te_columns:
            X_train[c] = train_fine_income_te[
                c
            ].to_numpy(dtype=np.float32)

            X_valid[c] = valid_fine_income_te[
                c
            ].to_numpy(dtype=np.float32)

            X_test[c] = test_fine_income_te[
                c
            ].to_numpy(dtype=np.float32)

        model = income_hte.build_model()

        # ONLY MODEL CHANGE IN THIS EXPERIMENT.
        model.set_params(
            max_bin=8192
        )

        if outer_fold == 0:
            print("Candidate max_bin: 8192")

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[
                train_idx
            ],
            base_margin=learned_train_margin,
            eval_set=[
                (
                    X_valid,
                    y[
                        valid_idx
                    ],
                )
            ],
            base_margin_eval_set=[
                learned_valid_margin
            ],
            verbose=False,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        if model.best_iteration is None:
            iteration_range = None
            best_iteration = -1
        else:
            best_iteration = int(
                model.best_iteration
            )

            iteration_range = (
                0,
                best_iteration + 1,
            )

        valid_pred = (
            model.predict_proba(
                X_valid,
                base_margin=learned_valid_margin,
                iteration_range=iteration_range,
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test,
                base_margin=learned_test_margin,
                iteration_range=iteration_range,
            )[:, 1]
        )

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

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": best_iteration,
                "fit_seconds": fit_seconds,
                "total_fold_seconds": fold_seconds,
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
                    "is_commute_hte": (
                        feature
                        in commute_hte_columns
                    ),
                    "is_income_hte": (
                        feature
                        in income_hte_columns
                    ),
                    "is_fine_income_te": (
                        feature
                        in fine_income_te_columns
                    ),
                    "is_exact_te": (
                        feature
                        in exact_te_columns
                    ),
                    "is_income_digit": (
                        feature
                        in digit_features
                    ),
                    "is_exact_frequency": (
                        feature
                        in exact_frequency_features
                    ),
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={best_iteration}"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test,
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            train_income_hte,
            valid_income_hte,
            test_income_hte,
            train_commute_hte,
            valid_commute_hte,
            test_commute_hte,
            train_fine_income_te,
            valid_fine_income_te,
            test_fine_income_te,
        )

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

    rank_correlation = rank_corr(
        oof,
        baseline_oof,
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
                "is_commute_hte",
                "is_income_hte",
                "is_fine_income_te",
                "is_exact_te",
                "is_income_digit",
                "is_exact_frequency",
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

    commute_importance = (
        importance_summary[
            importance_summary[
                "is_commute_hte"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    fine_income_importance = (
        importance_summary[
            importance_summary[
                "is_fine_income_te"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    if (
        delta_vs_baseline >= 0.00003
        and folds_improved >= 4
    ):
        decision = (
            "POSITIVE_MAXBIN8192_SIGNAL"
        )
    elif (
        delta_vs_baseline < 0
        or folds_worse >= 4
    ):
        decision = (
            "NEGATIVE_MAXBIN8192_SIGNAL"
        )
    else:
        decision = (
            "WEAK_OR_INCONSISTENT_MAXBIN8192_SIGNAL"
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

    commute_importance.to_csv(
        args.output_dir
        / "hierarchical_commute_te_importance.csv",
        index=False,
    )

    fine_income_importance.to_csv(
        args.output_dir
        / "fine_income_te_importance.csv",
        index=False,
    )

    pd.concat(
        exact_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "exact_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        income_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "hierarchical_income_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        commute_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "hierarchical_commute_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        fine_income_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "fine_income_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        logistic_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "learned_logistic_margin_coefficients.csv",
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
        "EXPERIMENT: XGBOOST FINE-INCOME + MAX_BIN 8192",
        "=" * 84,
        "",
        "HYPOTHESIS",
        "Can $50/$250 income neighbourhoods add useful local structure between",
        "exact income identity and the existing $1k income hierarchy?",
        "",
        "ONLY CHANGE",
        "max_bin: 256 -> 8192",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- complete current fine-income XGB representation",
        "- exact income + commute TE",
        "- hierarchical income TE at 1k/10k/100k",
        "- hierarchical commute TE at 1km/5km/10km",
        "- income digits",
        "- exact income + commute frequencies",
        "- learned nested logistic base margin",
        "- smoothing m=2",
        "- depth-4 XGBoost configuration",
        "- seed 42",
        "- CUDA GPU",
        "",
        "ONLY MODEL CHANGE",
        "- max_bin=8192",
        "",
        "RESULTS",
        f"Current fine-income XGB baseline OOF: {baseline_auc:.8f}",
        f"max_bin=8192 candidate OOF: {candidate_auc:.8f}",
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
            "FINE INCOME TE IMPORTANCE",
        ]
    )

    for row in fine_income_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.8f}"
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
    print("=" * 96)
    print("MAX_BIN 8192 EXPERIMENT COMPLETE")
    print("=" * 96)
    print(f"Current fine XGB     : {baseline_auc:.8f}")
    print(f"Candidate            : {candidate_auc:.8f}")
    print(f"Delta                : {delta_vs_baseline:+.8f}")
    print(f"Folds improved       : {folds_improved}/5")
    print(f"Folds worse          : {folds_worse}/5")
    print(f"Probability corr     : {probability_corr:.6f}")
    print(f"Rank corr            : {rank_correlation:.6f}")
    print(f"Decision             : {decision}")
    print(f"Runtime              : {total_seconds:.2f}s")
    print(f"Artifacts            : {args.output_dir.resolve()}")
    print()
    print("Fine income TE importances (representation held fixed):")
    if len(
        fine_income_importance
    ):
        print(
            fine_income_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(
                index=False
            )
        )
    print("=" * 96)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. fine_income_te_importance.csv")
    print("  5. fine_income_te_diagnostics.csv")


if __name__ == "__main__":
    main()
