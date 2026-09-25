"""
catboost_hierarchical_income_multiseed_gpu.py

Kaggle Playground Series S6E9

Controlled experiment:
Test whether the newly validated hierarchical-income CatBoost representation
benefits from multi-seed averaging.

HYPOTHESIS
----------
The seed-42 hierarchical-income CatBoost improved the exact-value-ID baseline:

    0.94536428 -> 0.94546676
    delta = +0.00010248
    folds improved = 5/5

The feature representation is therefore already validated.

This experiment changes ONLY random seed diversity:
- reuse existing seed 42 OOF/test predictions
- train seed 7
- train seed 2026
- evaluate 3-seed probability average
- evaluate 3-seed rank average

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw 13 features
- raw income + commute numeric columns
- exact income + commute categorical IDs
- hierarchical income categorical buckets:
      CAT__income_1k_bucket
      CAT__income_10k_bucket
      CAT__income_100k_bucket
- iterations=2000
- learning_rate=0.05
- depth=6
- loss_function="Logloss"
- eval_metric="AUC"
- task_type="GPU"
- devices="0"
- early_stopping_rounds=150
- all unspecified CatBoost parameters remain library defaults
- no public leaderboard optimization

IMPORTANT
---------
Seed 42 is NOT retrained. It is loaded from:

    artifacts/experiments/catboost_hierarchical_income_buckets_gpu/
        oof_predictions.csv
        test_predictions.csv

Only seeds 7 and 2026 are trained.

Run:
    python catboost_hierarchical_income_multiseed_gpu.py
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
    import catboost_hierarchical_income_buckets_gpu as base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import catboost_hierarchical_income_buckets_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_SEED42_AUC = 0.94546676
EXPECTED_OLD_CAT3_AUC = 0.94543956
AUC_CHECK_TOLERANCE = 2e-5

TRAIN_SEEDS = [7, 2026]
ALL_SEEDS = [42, 7, 2026]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_SEED42_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_buckets_gpu"
    / "oof_predictions.csv"
)

DEFAULT_SEED42_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_buckets_gpu"
    / "test_predictions.csv"
)

DEFAULT_OLD_CAT3_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_multiseed_gpu"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-seed audit for validated hierarchical-income CatBoost."
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
        "--seed42-oof",
        type=Path,
        default=DEFAULT_SEED42_OOF,
    )

    p.add_argument(
        "--seed42-test",
        type=Path,
        default=DEFAULT_SEED42_TEST,
    )

    p.add_argument(
        "--old-cat3-oof",
        type=Path,
        default=DEFAULT_OLD_CAT3_OOF,
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


def choose_prediction_column(
    df: pd.DataFrame,
    *,
    path: Path,
    kind: str,
) -> str:
    if kind == "oof":
        preferred = [
            "oof_prediction",
            "prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {
            "row_index",
            "fold",
            "target",
            "target_encoded",
            "id",
        }
    elif kind == "test":
        preferred = [
            "prediction",
            "test_prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {
            "row_index",
            "fold",
            "id",
        }
    else:
        raise ValueError(
            f"Unknown prediction kind: {kind}"
        )

    for c in preferred:
        if c in df.columns:
            return c

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
        return numeric[0]

    raise ValueError(
        f"Could not safely identify {kind} prediction column in:\n"
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
            f"OOF fold missing in {path}"
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

    pred_col = (
        choose_prediction_column(
            df,
            path=path,
            kind="oof",
        )
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
            f"Non-finite OOF predictions in {path}"
        )

    return pred


def load_test_predictions(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Test prediction file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(
        path
    )

    if len(df) != len(test):
        raise ValueError(
            f"Test row count mismatch in {path}"
        )

    if (
        id_col is not None
        and id_col in df.columns
    ):
        if id_col not in test.columns:
            raise ValueError(
                f"Test data missing ID column {id_col!r}"
            )

        if not np.array_equal(
            df[
                id_col
            ].to_numpy(),
            test[
                id_col
            ].to_numpy(),
        ):
            raise ValueError(
                f"Test ID mismatch in {path}"
            )

    pred_col = (
        choose_prediction_column(
            df,
            path=path,
            kind="test",
        )
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
            f"Non-finite test predictions in {path}"
        )

    return pred


def verify_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    auc = float(
        roc_auc_score(
            y,
            pred,
        )
    )

    if abs(
        auc - expected
    ) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            f"{name} OOF AUC mismatch.\n"
            f"Expected approximately: {expected:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}"
        )

    return auc


def percentile_rank(
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


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            percentile_rank(
                a
            ),
            percentile_rank(
                b
            ),
        )[0, 1]
    )


def build_model(
    seed: int,
) -> CatBoostClassifier:
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


def prepare_model_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
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

    X_train = train[
        raw_features
    ].copy()

    X_test = test[
        raw_features
    ].copy()

    for c in raw_categoricals:
        X_train[
            c
        ] = (
            X_train[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        X_test[
            c
        ] = (
            X_test[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    (
        X_train,
        X_test,
        exact_id_columns,
    ) = base.add_exact_value_ids(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    (
        X_train,
        X_test,
        bucket_columns,
    ) = base.add_hierarchical_income_categories(
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
        bucket_columns,
    )


def save_seed_oof(
    *,
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


def save_test_predictions(
    *,
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

    required_paths = [
        train_path,
        test_path,
        args.folds_path,
        args.seed42_oof,
        args.seed42_test,
        args.old_cat3_oof,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
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

    seed42_oof = load_oof(
        args.seed42_oof,
        train,
        fold_ids,
        id_col,
    )

    seed42_test = (
        load_test_predictions(
            args.seed42_test,
            test,
            id_col,
        )
    )

    seed42_auc = verify_auc(
        "Hierarchical CatBoost seed 42",
        y,
        seed42_oof,
        EXPECTED_SEED42_AUC,
    )

    old_cat3_oof = load_oof(
        args.old_cat3_oof,
        train,
        fold_ids,
        id_col,
    )

    old_cat3_auc = verify_auc(
        "Previous exact-value CatBoost 3-seed ensemble",
        y,
        old_cat3_oof,
        EXPECTED_OLD_CAT3_AUC,
    )

    (
        X_train,
        X_test,
        categorical_columns,
        bucket_columns,
    ) = prepare_model_frames(
        train,
        test,
        id_col,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 98)
    print("CATBOOST HIERARCHICAL-INCOME MULTI-SEED AUDIT")
    print("=" * 98)
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Hierarchical seed-42 OOF: "
        f"{seed42_auc:.8f}"
    )
    print(
        f"Previous exact-value Cat 3-seed: "
        f"{old_cat3_auc:.8f}"
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  The hierarchical-income CatBoost representation is already validated. "
        "Averaging independent CatBoost seeds may reduce seed-specific ranking "
        "noise while preserving the new representation."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add seed diversity: reuse seed 42, train seeds 7 and 2026, "
        "then compare probability and rank averages."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds + SHA256")
    print("  - exact income + commute categorical IDs")
    print("  - hierarchical income categorical buckets")
    print("  - raw numeric income + commute")
    print("  - iterations=2000")
    print("  - learning_rate=0.05")
    print("  - depth=6")
    print("  - GPU device 0")
    print("  - early stopping 150")
    print()
    print(
        f"Training only seeds: "
        f"{TRAIN_SEEDS}"
    )
    print()

    seed_oof: dict[int, np.ndarray] = {
        42: seed42_oof,
    }

    seed_test: dict[int, np.ndarray] = {
        42: seed42_test,
    }

    seed_auc: dict[int, float] = {
        42: seed42_auc,
    }

    fold_rows = []
    total_start = time.perf_counter()

    for seed in TRAIN_SEEDS:
        print(
            "=" * 98
        )
        print(
            f"TRAINING SEED {seed}"
        )
        print(
            "=" * 98
        )

        oof = np.full(
            len(train),
            np.nan,
            dtype=np.float64,
        )

        test_fold_predictions = []

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

            seed42_fold_auc = float(
                roc_auc_score(
                    y[
                        valid_idx
                    ],
                    seed42_oof[
                        valid_idx
                    ],
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
                    "seed42_auc": seed42_fold_auc,
                    "seed_auc": fold_auc,
                    "delta_vs_seed42": (
                        fold_auc
                        - seed42_fold_auc
                    ),
                    "best_iteration": best_iteration,
                    "total_fold_seconds": fold_seconds,
                }
            )

            print(
                f"Seed {seed} fold {fold}: "
                f"seed42={seed42_fold_auc:.8f} -> "
                f"seed{seed}={fold_auc:.8f} "
                f"({fold_auc - seed42_fold_auc:+.8f}) | "
                f"best_iter={best_iteration}"
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

        auc = float(
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
        ] = auc

        save_seed_oof(
            path=(
                args.output_dir
                / f"oof_predictions_seed{seed}.csv"
            ),
            train=train,
            fold_ids=fold_ids,
            id_col=id_col,
            y=y,
            pred=oof,
        )

        save_test_predictions(
            path=(
                args.output_dir
                / f"test_predictions_seed{seed}.csv"
            ),
            test=test,
            id_col=id_col,
            pred=test_prediction,
        )

        print()
        print(
            f"Seed {seed} overall OOF AUC: "
            f"{auc:.8f}"
        )
        print()

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    oof_stack = np.vstack(
        [
            seed_oof[
                seed
            ]
            for seed in ALL_SEEDS
        ]
    )

    test_stack = np.vstack(
        [
            seed_test[
                seed
            ]
            for seed in ALL_SEEDS
        ]
    )

    probability_average_oof = (
        np.mean(
            oof_stack,
            axis=0,
        )
    )

    probability_average_test = (
        np.mean(
            test_stack,
            axis=0,
        )
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
                    seed
                ]
            )
            for seed in ALL_SEEDS
        ]
    )

    rank_test_stack = np.vstack(
        [
            percentile_rank(
                seed_test[
                    seed
                ]
            )
            for seed in ALL_SEEDS
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

    delta_vs_seed42 = (
        best_auc
        - seed42_auc
    )

    delta_vs_old_cat3 = (
        best_auc
        - old_cat3_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    folds_best_ensemble_vs_seed42 = 0
    ensemble_fold_rows = []

    for fold in range(5):
        mask = (
            fold_ids
            == fold
        )

        seed42_fold_auc = float(
            roc_auc_score(
                y[
                    mask
                ],
                seed42_oof[
                    mask
                ],
            )
        )

        best_fold_auc = float(
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
            best_fold_auc
            - seed42_fold_auc
        )

        if delta > 0:
            folds_best_ensemble_vs_seed42 += 1

        ensemble_fold_rows.append(
            {
                "fold": fold,
                "seed42_auc": seed42_fold_auc,
                "best_ensemble_auc": best_fold_auc,
                "delta_vs_seed42": delta,
            }
        )

    ensemble_fold_metrics = pd.DataFrame(
        ensemble_fold_rows
    )

    pair_rows = []

    for i, seed_a in enumerate(
        ALL_SEEDS
    ):
        for seed_b in ALL_SEEDS[
            i + 1:
        ]:
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

    seed_pair_correlations = pd.DataFrame(
        pair_rows
    )

    if (
        delta_vs_seed42 > 0
        and folds_best_ensemble_vs_seed42 >= 3
    ):
        decision = (
            "KEEP_HIERARCHICAL_CATBOOST_3SEED"
        )
    else:
        decision = (
            "REJECT_HIERARCHICAL_CATBOOST_3SEED"
        )

    fold_metrics.to_csv(
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

    seed_score_rows = [
        {
            "model": f"seed_{seed}",
            "oof_auc": seed_auc[
                seed
            ],
        }
        for seed in ALL_SEEDS
    ]

    seed_score_rows.extend(
        [
            {
                "model": "probability_average",
                "oof_auc": probability_average_auc,
            },
            {
                "model": "rank_average",
                "oof_auc": rank_average_auc,
            },
        ]
    )

    pd.DataFrame(
        seed_score_rows
    ).to_csv(
        args.output_dir
        / "seed_scores.csv",
        index=False,
    )

    save_seed_oof(
        path=(
            args.output_dir
            / "probability_average_oof_predictions.csv"
        ),
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        y=y,
        pred=probability_average_oof,
    )

    save_test_predictions(
        path=(
            args.output_dir
            / "probability_average_test_predictions.csv"
        ),
        test=test,
        id_col=id_col,
        pred=probability_average_test,
    )

    save_seed_oof(
        path=(
            args.output_dir
            / "rank_average_oof_predictions.csv"
        ),
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        y=y,
        pred=rank_average_oof,
    )

    save_test_predictions(
        path=(
            args.output_dir
            / "rank_average_test_predictions.csv"
        ),
        test=test,
        id_col=id_col,
        pred=rank_average_test,
    )

    save_seed_oof(
        path=(
            args.output_dir
            / "best_average_oof_predictions.csv"
        ),
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        y=y,
        pred=best_oof,
    )

    save_test_predictions(
        path=(
            args.output_dir
            / "best_average_test_predictions.csv"
        ),
        test=test,
        id_col=id_col,
        pred=best_test,
    )

    summary_lines = [
        "EXPERIMENT: HIERARCHICAL-INCOME CATBOOST MULTI-SEED AUDIT",
        "=" * 86,
        "",
        "HYPOTHESIS",
        "Does seed averaging improve the already-validated hierarchical-income",
        "CatBoost representation?",
        "",
        "ONLY CHANGE",
        "Random-seed diversity. Reuse seed 42, train seeds 7 and 2026.",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- exact income + commute categorical IDs",
        "- hierarchical income categorical buckets",
        "- raw numeric income + commute",
        "- iterations=2000",
        "- learning_rate=0.05",
        "- depth=6",
        "- GPU device 0",
        "- early stopping 150",
        "",
        "SINGLE-SEED RESULTS",
        f"Seed 42: {seed_auc[42]:.8f}",
        f"Seed 7: {seed_auc[7]:.8f}",
        f"Seed 2026: {seed_auc[2026]:.8f}",
        "",
        "ENSEMBLE RESULTS",
        f"3-seed probability average: {probability_average_auc:.8f}",
        f"3-seed rank average: {rank_average_auc:.8f}",
        f"Best averaging method: {best_method}",
        f"Best 3-seed AUC: {best_auc:.8f}",
        f"Delta vs hierarchical seed42: {delta_vs_seed42:+.8f}",
        f"Delta vs previous exact-value Cat3: {delta_vs_old_cat3:+.8f}",
        f"Best ensemble folds improved vs seed42: {folds_best_ensemble_vs_seed42}/5",
        f"Runtime for newly trained seeds: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "BEST-ENSEMBLE FOLD RESULTS",
    ]

    for row in ensemble_fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"seed42={row.seed42_auc:.8f} -> "
            f"best_ensemble={row.best_ensemble_auc:.8f} "
            f"({row.delta_vs_seed42:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "SEED PAIR CORRELATIONS",
        ]
    )

    for row in seed_pair_correlations.itertuples():
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
    print("=" * 98)
    print("HIERARCHICAL CATBOOST MULTI-SEED AUDIT COMPLETE")
    print("=" * 98)
    print(f"Seed 42              : {seed_auc[42]:.8f}")
    print(f"Seed 7               : {seed_auc[7]:.8f}")
    print(f"Seed 2026            : {seed_auc[2026]:.8f}")
    print(f"Probability average  : {probability_average_auc:.8f}")
    print(f"Rank average         : {rank_average_auc:.8f}")
    print(f"Best method          : {best_method}")
    print(f"Best 3-seed AUC      : {best_auc:.8f}")
    print(f"Delta vs seed42      : {delta_vs_seed42:+.8f}")
    print(f"Delta vs old Cat3    : {delta_vs_old_cat3:+.8f}")
    print(
        f"Folds improved vs seed42: "
        f"{folds_best_ensemble_vs_seed42}/5"
    )
    print(f"Decision             : {decision}")
    print(f"Artifacts            : {args.output_dir.resolve()}")
    print("=" * 98)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. seed_scores.csv")
    print("  4. ensemble_fold_metrics.csv")
    print("  5. seed_pair_correlations.csv")


if __name__ == "__main__":
    main()
