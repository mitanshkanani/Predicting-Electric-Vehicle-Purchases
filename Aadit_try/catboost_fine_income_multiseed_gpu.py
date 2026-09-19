"""
catboost_fine_income_multiseed_gpu.py

Kaggle Playground Series S6E9

Controlled structural experiment:
Transfer the validated fine-grained income representation to the CURRENT
hierarchical-income+commute CatBoost model.

HYPOTHESIS
----------
The current CatBoost already benefits from:
- exact Annual_Income_USD categorical identity
- hierarchical income categorical buckets
- exact Daily_Commute_km categorical identity
- hierarchical commute categorical buckets

XGBoost showed a large, stable gain from adding $50/$250 local income
neighbourhoods. Test whether CatBoost can exploit the same target-free local
income structure through its categorical machinery.

ONLY CHANGE
-----------
Add two target-free categorical income bucket copies:

    FE__income_50_bucket_cat
    FE__income_250_bucket_cat

where:
    bucket = floor(Annual_Income_USD / width)

All three predetermined CatBoost seeds (42, 7, 2026) are retrained because the
feature representation changed. The old seed-42 predictions cannot be reused.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw 13 features
- raw income + commute numeric columns
- exact income + commute categorical IDs
- hierarchical income categorical buckets
- hierarchical commute categorical buckets
- seeds = 42, 7, 2026
- iterations=2000
- learning_rate=0.05
- depth=6
- loss_function="Logloss"
- eval_metric="AUC"
- task_type="GPU"
- devices="0"
- early_stopping_rounds=150
- all unspecified CatBoost parameters remain defaults
- no public leaderboard optimization

PRIMARY BASELINE
----------------
Current hierarchical-income+commute CatBoost 3-seed rank average:
    OOF AUC = 0.94558572

Run:
    python catboost_fine_income_multiseed_gpu.py
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

try:
    import catboost_hierarchical_income_buckets_gpu as income_base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import catboost_hierarchical_income_buckets_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc

try:
    import catboost_hierarchical_income_commute_buckets_gpu as commute_base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import catboost_hierarchical_income_commute_buckets_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_CURRENT_SEED42_AUC = 0.94552424
EXPECTED_CURRENT_CAT3_AUC = 0.94558572
AUC_CHECK_TOLERANCE = 2e-5

ALL_SEEDS = [42, 7, 2026]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_CURRENT_SEED42_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_buckets_gpu"
    / "oof_predictions.csv"
)

DEFAULT_CURRENT_CAT3_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_fine_income_multiseed_gpu"
)

FINE_INCOME_BUCKET_SPECS = [
    ("FE__income_50_bucket_cat", 50.0),
    ("FE__income_250_bucket_cat", 250.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Current hierarchical-income+commute CatBoost "
            "+ target-free $50/$250 income categorical buckets."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)
    p.add_argument(
        "--current-seed42-oof",
        type=Path,
        default=DEFAULT_CURRENT_SEED42_OOF,
    )
    p.add_argument(
        "--current-cat3-oof",
        type=Path,
        default=DEFAULT_CURRENT_CAT3_OOF,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_prediction_column(
    df: pd.DataFrame,
    path: Path,
) -> str:
    preferred = [
        "oof_prediction",
        "prediction",
        "rank_average_prediction",
        "best_average_prediction",
    ]

    for c in preferred:
        if c in df.columns:
            return c

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
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) == 1:
        return numeric[0]

    raise ValueError(
        f"Could not safely identify OOF prediction column in:\n"
        f"{path.resolve()}\n"
        f"Columns: {list(df.columns)}"
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

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch in {path}"
        )

    if "row_index" not in df.columns:
        raise ValueError(
            f"OOF row_index missing in {path}"
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            f"OOF row order mismatch in {path}"
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"OOF fold missing in {path}"
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF fold mismatch in {path}"
        )

    if id_col is not None and id_col in df.columns:
        if not np.array_equal(
            df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                f"OOF ID mismatch in {path}"
            )

    pred_col = pick_prediction_column(
        df,
        path,
    )

    pred = df[pred_col].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"Non-finite OOF predictions in {path}"
        )

    return pred


def verify_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    score = float(
        roc_auc_score(
            y,
            pred,
        )
    )

    if abs(
        score - expected
    ) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            f"{name} OOF AUC mismatch.\n"
            f"Expected approximately: {expected:.8f}\n"
            f"Loaded artifact AUC   : {score:.8f}"
        )

    return score


def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(dtype=np.float64)
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            percentile_rank(a),
            percentile_rank(b),
        )[0, 1]
    )


def build_model(
    seed: int,
) -> CatBoostClassifier:
    # Exact model recipe from the validated current CatBoost script.
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
    )


def make_income_bucket(
    series: pd.Series,
    width: float,
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
        numeric / width
    ).astype(
        np.int64
    )

    return pd.Series(
        bucket.astype(str),
        index=series.index,
        dtype="object",
    )


def add_fine_income_categories(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    columns = []
    diagnostics = []

    for feature_name, width in FINE_INCOME_BUCKET_SPECS:
        train_bucket = make_income_bucket(
            train_source["Annual_Income_USD"],
            width,
        )
        test_bucket = make_income_bucket(
            test_source["Annual_Income_USD"],
            width,
        )

        X_train[feature_name] = train_bucket
        X_test[feature_name] = test_bucket
        columns.append(feature_name)

        train_unique = set(
            train_bucket.unique().tolist()
        )

        test_seen_rate = float(
            test_bucket.isin(
                train_unique
            ).mean()
        )

        train_counts = (
            train_bucket
            .value_counts()
        )

        diagnostics.append(
            {
                "feature": feature_name,
                "bucket_width_usd": width,
                "train_unique_buckets": int(
                    train_bucket.nunique()
                ),
                "test_unique_buckets": int(
                    test_bucket.nunique()
                ),
                "train_min_count": int(
                    train_counts.min()
                ),
                "train_median_count": float(
                    train_counts.median()
                ),
                "train_mean_count": float(
                    train_counts.mean()
                ),
                "train_max_count": int(
                    train_counts.max()
                ),
                "test_seen_rate": test_seen_rate,
            }
        )

    return (
        X_train,
        X_test,
        columns,
        pd.DataFrame(
            diagnostics
        ),
    )


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = (
        income_base.detect_raw_categoricals(
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
    ) = income_base.add_exact_value_ids(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    (
        X_train,
        X_test,
        income_bucket_columns,
    ) = income_base.add_hierarchical_income_categories(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    (
        X_train,
        X_test,
        commute_bucket_columns,
    ) = commute_base.add_hierarchical_commute_categories(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    (
        X_train,
        X_test,
        fine_income_columns,
        fine_income_diagnostics,
    ) = add_fine_income_categories(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    categorical_columns = (
        raw_categoricals
        + exact_id_columns
        + income_bucket_columns
        + commute_bucket_columns
        + fine_income_columns
    )

    if (
        len(
            set(
                categorical_columns
            )
        )
        != len(
            categorical_columns
        )
    ):
        raise ValueError(
            "Duplicate categorical feature names detected."
        )

    return (
        X_train,
        X_test,
        categorical_columns,
        fine_income_diagnostics,
    )


def save_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    y: np.ndarray,
    pred: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": pred.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        df.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    df.to_csv(
        path,
        index=False,
    )


def save_test(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
    pred: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "prediction": pred.astype(
                np.float32
            )
        }
    )

    if id_col is not None:
        df.insert(
            0,
            id_col,
            test[
                id_col
            ].to_numpy(),
        )

    df.to_csv(
        path,
        index=False,
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
        args.current_seed42_oof,
        args.current_cat3_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n"
                f"{path.resolve()}"
            )

    fold_hash = sha256_file(
        args.folds_path
    )

    if (
        fold_hash
        != EXPECTED_FOLDS_SHA256
    ):
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}"
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

    target = income_base.detect_target(
        train,
        test,
    )

    y, positive_label = (
        income_base.encode_binary_target(
            train[
                target
            ]
        )
    )

    fold_ids, id_col = (
        income_base.validate_folds(
            folds_df,
            train,
        )
    )

    current_seed42_oof = load_oof(
        args.current_seed42_oof,
        train,
        fold_ids,
        id_col,
    )

    current_seed42_auc = verify_auc(
        "Current income+commute CatBoost seed42",
        y,
        current_seed42_oof,
        EXPECTED_CURRENT_SEED42_AUC,
    )

    current_cat3_oof = load_oof(
        args.current_cat3_oof,
        train,
        fold_ids,
        id_col,
    )

    current_cat3_auc = verify_auc(
        "Current income+commute CatBoost 3-seed",
        y,
        current_cat3_oof,
        EXPECTED_CURRENT_CAT3_AUC,
    )

    (
        X_train,
        X_test,
        categorical_columns,
        fine_income_diagnostics,
    ) = prepare_frames(
        train,
        test,
        id_col,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fine_income_diagnostics.to_csv(
        args.output_dir
        / "fine_income_category_diagnostics.csv",
        index=False,
    )

    print("=" * 100)
    print(
        "CATBOOST + FINE-GRAINED INCOME CATEGORIES "
        "(3-SEED)"
    )
    print("=" * 100)
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Current seed42 baseline : "
        f"{current_seed42_auc:.8f}"
    )
    print(
        f"Current Cat3 baseline   : "
        f"{current_cat3_auc:.8f}"
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  The $50/$250 income neighbourhood signal that strongly improved XGBoost "
        "may also help CatBoost through target-free categorical buckets."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add FE__income_50_bucket_cat and FE__income_250_bucket_cat."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen folds + SHA256")
    print("  - exact income/commute categorical IDs")
    print("  - existing income hierarchy")
    print("  - existing commute hierarchy")
    print("  - seeds 42 / 7 / 2026")
    print("  - iterations=2000")
    print("  - learning_rate=0.05")
    print("  - depth=6")
    print("  - GPU / early stopping=150")
    print()
    print("Fine-income category diagnostics:")
    print(
        fine_income_diagnostics.to_string(
            index=False,
        )
    )
    print()

    seed_oof: dict[int, np.ndarray] = {}
    seed_test: dict[int, np.ndarray] = {}
    seed_auc: dict[int, float] = {}
    fold_rows: list[dict] = []

    total_start = time.perf_counter()

    for seed in ALL_SEEDS:
        print("=" * 100)
        print(
            f"TRAINING CANDIDATE SEED {seed}"
        )
        print("=" * 100)

        oof = np.full(
            len(train),
            np.nan,
            dtype=np.float64,
        )

        test_fold_predictions: list[
            np.ndarray
        ] = []

        for fold in range(5):
            fold_start = (
                time.perf_counter()
            )

            train_idx = np.flatnonzero(
                fold_ids != fold
            )

            valid_idx = np.flatnonzero(
                fold_ids == fold
            )

            model = build_model(
                seed
            )

            model.fit(
                X_train.iloc[
                    train_idx
                ],
                y[
                    train_idx
                ],
                cat_features=(
                    categorical_columns
                ),
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

            valid_pred = (
                model.predict_proba(
                    X_train.iloc[
                        valid_idx
                    ]
                )[:, 1]
            )

            test_pred = (
                model.predict_proba(
                    X_test
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

            fold_auc = float(
                roc_auc_score(
                    y[
                        valid_idx
                    ],
                    valid_pred,
                )
            )

            current_seed42_fold_auc = (
                float(
                    roc_auc_score(
                        y[
                            valid_idx
                        ],
                        current_seed42_oof[
                            valid_idx
                        ],
                    )
                )
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
                    "seed": seed,
                    "fold": fold,
                    "current_seed42_auc": (
                        current_seed42_fold_auc
                    ),
                    "candidate_seed_auc": (
                        fold_auc
                    ),
                    "delta_vs_current_seed42": (
                        fold_auc
                        - current_seed42_fold_auc
                    ),
                    "best_iteration": (
                        best_iteration
                    ),
                    "total_fold_seconds": (
                        fold_seconds
                    ),
                }
            )

            print(
                f"Seed {seed} fold {fold}: "
                f"current_seed42="
                f"{current_seed42_fold_auc:.8f} -> "
                f"candidate="
                f"{fold_auc:.8f} "
                f"({fold_auc - current_seed42_fold_auc:+.8f}) | "
                f"best_iter="
                f"{best_iteration}"
            )

            del model

        if np.isnan(
            oof
        ).any():
            raise RuntimeError(
                f"Seed {seed} OOF contains NaNs."
            )

        test_prediction = np.mean(
            np.vstack(
                test_fold_predictions
            ),
            axis=0,
        )

        score = float(
            roc_auc_score(
                y,
                oof,
            )
        )

        seed_oof[
            seed
        ] = oof

        seed_test[
            seed
        ] = test_prediction

        seed_auc[
            seed
        ] = score

        save_oof(
            args.output_dir
            / f"oof_predictions_seed{seed}.csv",
            train,
            fold_ids,
            id_col,
            y,
            oof,
        )

        save_test(
            args.output_dir
            / f"test_predictions_seed{seed}.csv",
            test,
            id_col,
            test_prediction,
        )

        print()
        print(
            f"Candidate seed {seed} overall OOF AUC: "
            f"{score:.8f}"
        )
        print()

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    oof_stack = np.vstack(
        [
            seed_oof[
                s
            ]
            for s in ALL_SEEDS
        ]
    )

    test_stack = np.vstack(
        [
            seed_test[
                s
            ]
            for s in ALL_SEEDS
        ]
    )

    probability_average_oof = np.mean(
        oof_stack,
        axis=0,
    )

    probability_average_test = np.mean(
        test_stack,
        axis=0,
    )

    probability_average_auc = float(
        roc_auc_score(
            y,
            probability_average_oof,
        )
    )

    rank_oof_stack = np.vstack(
        [
            percentile_rank(
                seed_oof[
                    s
                ]
            )
            for s in ALL_SEEDS
        ]
    )

    rank_test_stack = np.vstack(
        [
            percentile_rank(
                seed_test[
                    s
                ]
            )
            for s in ALL_SEEDS
        ]
    )

    rank_average_oof = np.mean(
        rank_oof_stack,
        axis=0,
    )

    rank_average_test = np.mean(
        rank_test_stack,
        axis=0,
    )

    rank_average_auc = float(
        roc_auc_score(
            y,
            rank_average_oof,
        )
    )

    if (
        rank_average_auc
        >= probability_average_auc
    ):
        best_method = (
            "rank_average"
        )
        best_oof = (
            rank_average_oof
        )
        best_test = (
            rank_average_test
        )
        best_auc = (
            rank_average_auc
        )
    else:
        best_method = (
            "probability_average"
        )
        best_oof = (
            probability_average_oof
        )
        best_test = (
            probability_average_test
        )
        best_auc = (
            probability_average_auc
        )

    delta_vs_current_cat3 = (
        best_auc
        - current_cat3_auc
    )

    seed42_delta = (
        seed_auc[
            42
        ]
        - current_seed42_auc
    )

    current_cat3_rank_corr = rank_corr(
        best_oof,
        current_cat3_oof,
    )

    ensemble_fold_rows = []
    folds_improved_vs_current_cat3 = 0
    folds_worse_vs_current_cat3 = 0

    for fold in range(5):
        mask = (
            fold_ids
            == fold
        )

        current_fold_auc = float(
            roc_auc_score(
                y[
                    mask
                ],
                current_cat3_oof[
                    mask
                ],
            )
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[
                    mask
                ],
                best_oof[
                    mask
                ],
            )
        )

        delta = (
            candidate_fold_auc
            - current_fold_auc
        )

        if delta > 0:
            folds_improved_vs_current_cat3 += 1
        elif delta < 0:
            folds_worse_vs_current_cat3 += 1

        ensemble_fold_rows.append(
            {
                "fold": fold,
                "current_cat3_auc": (
                    current_fold_auc
                ),
                "candidate_cat3_auc": (
                    candidate_fold_auc
                ),
                "delta_vs_current_cat3": (
                    delta
                ),
            }
        )

    ensemble_fold_metrics = (
        pd.DataFrame(
            ensemble_fold_rows
        )
    )

    pair_rows = []

    for i, seed_a in enumerate(
        ALL_SEEDS
    ):
        for seed_b in (
            ALL_SEEDS[
                i + 1:
            ]
        ):
            pair_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "probability_corr": float(
                        np.corrcoef(
                            seed_oof[
                                seed_a
                            ],
                            seed_oof[
                                seed_b
                            ],
                        )[0, 1]
                    ),
                    "rank_corr": rank_corr(
                        seed_oof[
                            seed_a
                        ],
                        seed_oof[
                            seed_b
                        ],
                    ),
                }
            )

    seed_pair_correlations = (
        pd.DataFrame(
            pair_rows
        )
    )

    if (
        delta_vs_current_cat3 > 0
        and folds_improved_vs_current_cat3 >= 4
    ):
        primitive = (
            "POSITIVE_FINE_INCOME_CATBOOST_SIGNAL"
        )
    elif (
        delta_vs_current_cat3 < 0
    ):
        primitive = (
            "NEGATIVE_FINE_INCOME_CATBOOST_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_FINE_INCOME_CATBOOST_SIGNAL"
        )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        args.output_dir
        / "seed_fold_metrics.csv",
        index=False,
    )

    ensemble_fold_metrics.to_csv(
        args.output_dir
        / "ensemble_fold_metrics.csv",
        index=False,
    )

    seed_pair_correlations.to_csv(
        args.output_dir
        / "seed_pair_correlations.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "model": (
                    f"candidate_seed_{s}"
                ),
                "oof_auc": (
                    seed_auc[
                        s
                    ]
                ),
            }
            for s in ALL_SEEDS
        ]
        + [
            {
                "model": (
                    "probability_average"
                ),
                "oof_auc": (
                    probability_average_auc
                ),
            },
            {
                "model": (
                    "rank_average"
                ),
                "oof_auc": (
                    rank_average_auc
                ),
            },
        ]
    ).to_csv(
        args.output_dir
        / "seed_scores.csv",
        index=False,
    )

    save_oof(
        args.output_dir
        / "probability_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        probability_average_oof,
    )

    save_test(
        args.output_dir
        / "probability_average_test_predictions.csv",
        test,
        id_col,
        probability_average_test,
    )

    save_oof(
        args.output_dir
        / "rank_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        rank_average_oof,
    )

    save_test(
        args.output_dir
        / "rank_average_test_predictions.csv",
        test,
        id_col,
        rank_average_test,
    )

    save_oof(
        args.output_dir
        / "best_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        best_oof,
    )

    save_test(
        args.output_dir
        / "best_average_test_predictions.csv",
        test,
        id_col,
        best_test,
    )

    summary_lines = [
        "EXPERIMENT: CATBOOST + FINE-GRAINED INCOME CATEGORIES (3-SEED)",
        "=" * 92,
        "",
        "HYPOTHESIS",
        "Can target-free $50/$250 income categorical buckets transfer the",
        "validated local-income signal from XGBoost into CatBoost?",
        "",
        "ONLY CHANGE",
        "Add FE__income_50_bucket_cat and FE__income_250_bucket_cat.",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- current exact income/commute categorical IDs",
        "- current hierarchical income buckets",
        "- current hierarchical commute buckets",
        "- seeds 42/7/2026",
        "- iterations=2000",
        "- learning_rate=0.05",
        "- depth=6",
        "- GPU / early stopping=150",
        "",
        "BASELINES",
        f"Current seed42: {current_seed42_auc:.8f}",
        f"Current Cat3: {current_cat3_auc:.8f}",
        "",
        "CANDIDATE SINGLE-SEED RESULTS",
        f"Seed 42: {seed_auc[42]:.8f}",
        f"Seed 7: {seed_auc[7]:.8f}",
        f"Seed 2026: {seed_auc[2026]:.8f}",
        f"Seed42 delta vs current seed42: {seed42_delta:+.8f}",
        "",
        "CANDIDATE 3-SEED RESULTS",
        f"Probability average: {probability_average_auc:.8f}",
        f"Rank average: {rank_average_auc:.8f}",
        f"Best averaging method: {best_method}",
        f"Best candidate Cat3 AUC: {best_auc:.8f}",
        f"Delta vs current Cat3: {delta_vs_current_cat3:+.8f}",
        f"Folds improved vs current Cat3: {folds_improved_vs_current_cat3}/5",
        f"Folds worse vs current Cat3: {folds_worse_vs_current_cat3}/5",
        f"Rank corr vs current Cat3: {current_cat3_rank_corr:.6f}",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"PRIMITIVE: {primitive}",
        "",
        "FOLD RESULTS VS CURRENT CAT3",
    ]

    for row in (
        ensemble_fold_metrics.itertuples()
    ):
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.current_cat3_auc:.8f} -> "
            f"{row.candidate_cat3_auc:.8f} "
            f"({row.delta_vs_current_cat3:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "FINE INCOME CATEGORY DIAGNOSTICS",
        ]
    )

    for row in (
        fine_income_diagnostics.itertuples()
    ):
        summary_lines.append(
            f"{row.feature}: "
            f"width=${row.bucket_width_usd:.0f}, "
            f"train_unique={row.train_unique_buckets}, "
            f"median_count={row.train_median_count:.1f}, "
            f"test_seen_rate={row.test_seen_rate:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "SEED PAIR CORRELATIONS",
        ]
    )

    for row in (
        seed_pair_correlations.itertuples()
    ):
        summary_lines.append(
            f"{row.seed_a} vs {row.seed_b}: "
            f"prob_corr={row.probability_corr:.6f}, "
            f"rank_corr={row.rank_corr:.6f}"
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
    print("=" * 100)
    print(
        "FINE-INCOME CATBOOST 3-SEED EXPERIMENT COMPLETE"
    )
    print("=" * 100)
    print(
        f"Current seed42      : "
        f"{current_seed42_auc:.8f}"
    )
    print(
        f"Candidate seed42    : "
        f"{seed_auc[42]:.8f}"
    )
    print(
        f"Seed42 delta        : "
        f"{seed42_delta:+.8f}"
    )
    print()
    print(
        f"Current Cat3        : "
        f"{current_cat3_auc:.8f}"
    )
    print(
        f"Candidate prob avg  : "
        f"{probability_average_auc:.8f}"
    )
    print(
        f"Candidate rank avg  : "
        f"{rank_average_auc:.8f}"
    )
    print(
        f"Best method         : "
        f"{best_method}"
    )
    print(
        f"Best candidate Cat3 : "
        f"{best_auc:.8f}"
    )
    print(
        f"Delta vs current    : "
        f"{delta_vs_current_cat3:+.8f}"
    )
    print(
        f"Folds improved      : "
        f"{folds_improved_vs_current_cat3}/5"
    )
    print(
        f"Folds worse         : "
        f"{folds_worse_vs_current_cat3}/5"
    )
    print(
        f"Rank corr vs current: "
        f"{current_cat3_rank_corr:.6f}"
    )
    print(
        f"Primitive           : "
        f"{primitive}"
    )
    print(
        f"Runtime             : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts           : "
        f"{args.output_dir.resolve()}"
    )
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. seed_scores.csv")
    print("  4. ensemble_fold_metrics.csv")
    print("  5. fine_income_category_diagnostics.csv")


if __name__ == "__main__":
    main()
