"""
catboost_value_ids_multiseed_gpu.py

Phase 5F — Multi-seed averaging of the exact-value CatBoost champion.

CURRENT CHAMPION
----------------
CatBoost GPU + exact value identities:
    Local OOF AUC = 0.94536428
    Public LB     = 0.94552

HYPOTHESIS
----------
GPU CatBoost has stochastic variation. Averaging independently seeded models
may reduce ranking noise and improve ROC-AUC without changing the feature
representation.

SEEDS
-----
Existing saved champion:
    seed = 42

New models trained by this script:
    seed = 7
    seed = 2026

CONTROLLED EXPERIMENT
---------------------
SAME:
- frozen 5 folds
- exact same 15-feature representation
- same CatBoost hyperparameters
- same GPU device
- same early stopping

ONLY CHANGE:
- random_seed

The script evaluates:
1. seed 42 alone (saved champion)
2. seed 7 alone
3. seed 2026 alone
4. equal probability average of all 3 seeds
5. equal rank average of all 3 seeds

No Kaggle submission is created.

Run:
    python catboost_value_ids_multiseed_gpu.py
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


BASE_SEED = 42
NEW_SEEDS = [7, 2026]
ALL_SEEDS = [42, 7, 2026]

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_CHAMPION_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "oof_predictions.csv"
)

DEFAULT_CHAMPION_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "test_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed exact-value CatBoost experiment."
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
        "--champion-oof",
        type=Path,
        default=DEFAULT_CHAMPION_OOF,
    )

    parser.add_argument(
        "--champion-test",
        type=Path,
        default=DEFAULT_CHAMPION_TEST,
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
            f"Expected exactly one train-only target column, found {train_only}"
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
                "Frozen fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(dtype=np.int16),
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
            or isinstance(dtype, pd.CategoricalDtype)
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
            numeric.to_numpy(dtype=np.float64)
        ).astype(np.int64)

        return pd.Series(
            arr.astype(str),
            index=series.index,
            name=f"{feature}_id",
        )

    if feature == "Daily_Commute_km":
        arr = numeric.to_numpy(dtype=np.float64)

        return pd.Series(
            [f"{x:.1f}" for x in arr],
            index=series.index,
            name=f"{feature}_id",
        )

    return numeric.map(
        lambda x: format(float(x), ".12g")
    ).rename(
        f"{feature}_id"
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
]:
    x_train = train[raw_features].copy()
    x_test = test[raw_features].copy()

    for c in raw_categoricals:
        x_train[c] = (
            x_train[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        x_test[c] = (
            x_test[c]
            .fillna("__MISSING__")
            .astype(str)
        )

    value_id_columns = []

    for feature in VALUE_ID_FEATURES:
        id_col = f"{feature}_id"

        x_train[id_col] = stable_value_id(
            train[feature],
            feature,
        )

        x_test[id_col] = stable_value_id(
            test[feature],
            feature,
        )

        value_id_columns.append(id_col)

    categorical_features = (
        raw_categoricals
        + value_id_columns
    )

    return (
        x_train,
        x_test,
        categorical_features,
    )


def load_champion_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Champion OOF file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Champion OOF file has unexpected columns."
        )

    if len(df) != len(train):
        raise ValueError(
            "Champion OOF row count mismatch."
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            "Champion OOF row order mismatch."
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Champion OOF folds mismatch."
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
            "Champion OOF IDs are misaligned."
        )

    return df["oof_prediction"].to_numpy(
        dtype=np.float64
    )


def load_champion_test(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Champion test prediction file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    if "prediction" not in df.columns:
        raise ValueError(
            "Champion test file missing 'prediction' column."
        )

    if len(df) != len(test):
        raise ValueError(
            "Champion test prediction row count mismatch."
        )

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(
            df[id_col].to_numpy(),
            test[id_col].to_numpy(),
        )
    ):
        raise ValueError(
            "Champion test IDs are misaligned."
        )

    return df["prediction"].to_numpy(
        dtype=np.float64
    )


def rank01(
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
            rank01(a),
            rank01(b),
        )[0, 1]
    )


def build_model(
    seed: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=2000,
        learning_rate=0.05,
        depth=6,

        random_seed=seed,

        task_type="GPU",
        devices="0",

        allow_writing_files=False,
    )


def train_seed(
    seed: int,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    categorical_features: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    oof = np.full(
        len(X),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    rows = []

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=list(X.columns),
    )

    print()
    print("=" * 78)
    print(
        f"TRAINING NEW SEED {seed}"
    )
    print("=" * 78)

    for fold in range(5):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        train_pool = Pool(
            X.iloc[train_idx],
            label=y[train_idx],
            cat_features=categorical_features,
            feature_names=list(X.columns),
        )

        valid_pool = Pool(
            X.iloc[valid_idx],
            label=y[valid_idx],
            cat_features=categorical_features,
            feature_names=list(X.columns),
        )

        model = build_model(
            seed=seed
        )

        start = time.perf_counter()

        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=200,
        )

        fit_seconds = (
            time.perf_counter()
            - start
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

        oof[valid_idx] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration_zero_based": int(
                    model.get_best_iteration()
                ),
                "tree_count": int(
                    model.tree_count_
                ),
                "fit_seconds": float(
                    fit_seconds
                ),
            }
        )

        print()
        print(
            f"Seed {seed} | Fold {fold}: "
            f"AUC={fold_auc:.6f} | "
            f"best_iter={model.get_best_iteration()} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            train_pool,
            valid_pool,
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            f"Seed {seed} OOF contains NaN predictions."
        )

    test_average = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    return (
        oof,
        test_average,
        pd.DataFrame(rows),
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
        args.champion_oof,
        args.champion_test,
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
        "Loading competition data and existing seed-42 champion..."
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(
        train,
        test,
    )

    y, positive_label = encode_binary_target(
        train[target]
    )

    fold_ids, id_col = validate_folds(
        folds_df,
        train,
    )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = detect_raw_categoricals(
        train,
        raw_features,
    )

    (
        X,
        X_test,
        categorical_features,
    ) = prepare_frames(
        train,
        test,
        raw_features,
        raw_categoricals,
    )

    seed42_oof = load_champion_oof(
        args.champion_oof,
        train,
        fold_ids,
        id_col,
    )

    seed42_test = load_champion_test(
        args.champion_test,
        test,
        id_col,
    )

    seed42_auc = float(
        roc_auc_score(
            y,
            seed42_oof,
        )
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Existing seed-42 champion OOF: "
        f"{seed42_auc:.8f}"
    )

    print(
        f"New seeds to train: {NEW_SEEDS}"
    )

    print(
        f"Total features: {len(X.columns)}"
    )

    print()
    print(
        "No feature or hyperparameter changes."
    )
    print(
        "Only random_seed changes."
    )

    seed_oof = {
        42: seed42_oof,
    }

    seed_test = {
        42: seed42_test,
    }

    all_fold_metrics = []

    total_start = (
        time.perf_counter()
    )

    for seed in NEW_SEEDS:
        (
            oof,
            test_pred,
            fold_metrics,
        ) = train_seed(
            seed=seed,
            X=X,
            X_test=X_test,
            y=y,
            fold_ids=fold_ids,
            categorical_features=categorical_features,
        )

        seed_oof[
            seed
        ] = oof

        seed_test[
            seed
        ] = test_pred

        all_fold_metrics.append(
            fold_metrics
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    # ---------------------------------------------------------
    # Single-seed metrics
    # ---------------------------------------------------------

    result_rows = []

    for seed in ALL_SEEDS:
        score = float(
            roc_auc_score(
                y,
                seed_oof[
                    seed
                ],
            )
        )

        result_rows.append(
            {
                "candidate": f"seed_{seed}",
                "oof_auc": score,
                "delta_vs_seed42": (
                    score
                    - seed42_auc
                ),
                "kind": "single_seed",
            }
        )

    # ---------------------------------------------------------
    # Equal probability average
    # ---------------------------------------------------------

    oof_matrix = np.vstack(
        [
            seed_oof[
                seed
            ]
            for seed in ALL_SEEDS
        ]
    )

    test_matrix = np.vstack(
        [
            seed_test[
                seed
            ]
            for seed in ALL_SEEDS
        ]
    )

    prob_avg_oof = np.mean(
        oof_matrix,
        axis=0,
    )

    prob_avg_test = np.mean(
        test_matrix,
        axis=0,
    )

    prob_avg_auc = float(
        roc_auc_score(
            y,
            prob_avg_oof,
        )
    )

    result_rows.append(
        {
            "candidate": "avg3_probability",
            "oof_auc": prob_avg_auc,
            "delta_vs_seed42": (
                prob_avg_auc
                - seed42_auc
            ),
            "kind": "equal_average",
        }
    )

    # ---------------------------------------------------------
    # Equal rank average
    # ---------------------------------------------------------

    oof_rank_matrix = np.vstack(
        [
            rank01(
                seed_oof[
                    seed
                ]
            )
            for seed in ALL_SEEDS
        ]
    )

    test_rank_matrix = np.vstack(
        [
            rank01(
                seed_test[
                    seed
                ]
            )
            for seed in ALL_SEEDS
        ]
    )

    rank_avg_oof = np.mean(
        oof_rank_matrix,
        axis=0,
    )

    rank_avg_test = np.mean(
        test_rank_matrix,
        axis=0,
    )

    rank_avg_auc = float(
        roc_auc_score(
            y,
            rank_avg_oof,
        )
    )

    result_rows.append(
        {
            "candidate": "avg3_rank",
            "oof_auc": rank_avg_auc,
            "delta_vs_seed42": (
                rank_avg_auc
                - seed42_auc
            ),
            "kind": "equal_average",
        }
    )

    results = (
        pd.DataFrame(
            result_rows
        )
        .sort_values(
            "oof_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # Pairwise seed diversity
    # ---------------------------------------------------------

    corr_rows = []

    for i, seed_a in enumerate(
        ALL_SEEDS
    ):
        for seed_b in ALL_SEEDS[
            i + 1:
        ]:
            corr_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,

                    "probability_correlation": float(
                        np.corrcoef(
                            seed_oof[
                                seed_a
                            ],
                            seed_oof[
                                seed_b
                            ],
                        )[0, 1]
                    ),

                    "rank_correlation": rank_corr(
                        seed_oof[
                            seed_a
                        ],
                        seed_oof[
                            seed_b
                        ],
                    ),
                }
            )

    correlations = pd.DataFrame(
        corr_rows
    )

    # ---------------------------------------------------------
    # Select best equal-average candidate
    # ---------------------------------------------------------

    average_results = results[
        results[
            "kind"
        ] == "equal_average"
    ].copy()

    best_average_row = (
        average_results
        .sort_values(
            "oof_auc",
            ascending=False,
        )
        .iloc[0]
    )

    best_average_name = str(
        best_average_row[
            "candidate"
        ]
    )

    best_average_auc = float(
        best_average_row[
            "oof_auc"
        ]
    )

    best_delta = (
        best_average_auc
        - seed42_auc
    )

    if (
        best_average_name
        == "avg3_probability"
    ):
        best_oof = prob_avg_oof
        best_test = prob_avg_test
    elif (
        best_average_name
        == "avg3_rank"
    ):
        best_oof = rank_avg_oof
        best_test = rank_avg_test
    else:
        raise RuntimeError(
            "Unexpected best average candidate."
        )

    # ---------------------------------------------------------
    # Fold-level ensemble comparison
    # ---------------------------------------------------------

    ensemble_fold_rows = []

    folds_improved = 0

    for fold in range(5):
        mask = (
            fold_ids == fold
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[
                    mask
                ],
                seed42_oof[
                    mask
                ],
            )
        )

        ensemble_fold_auc = float(
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
            ensemble_fold_auc
            - baseline_fold_auc
        )

        if delta > 0:
            folds_improved += 1

        ensemble_fold_rows.append(
            {
                "fold": fold,
                "seed42_auc": baseline_fold_auc,
                "best_average_auc": ensemble_fold_auc,
                "delta": delta,
            }
        )

    ensemble_folds = pd.DataFrame(
        ensemble_fold_rows
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    results.to_csv(
        args.output_dir
        / "candidate_summary.csv",
        index=False,
    )

    correlations.to_csv(
        args.output_dir
        / "seed_correlations.csv",
        index=False,
    )

    ensemble_folds.to_csv(
        args.output_dir
        / "ensemble_fold_metrics.csv",
        index=False,
    )

    if all_fold_metrics:
        pd.concat(
            all_fold_metrics,
            ignore_index=True,
        ).to_csv(
            args.output_dir
            / "new_seed_fold_metrics.csv",
            index=False,
        )

    prediction_frame = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "seed42": seed_oof[
                42
            ].astype(
                np.float32
            ),
            "seed7": seed_oof[
                7
            ].astype(
                np.float32
            ),
            "seed2026": seed_oof[
                2026
            ].astype(
                np.float32
            ),
            "avg3_probability": prob_avg_oof.astype(
                np.float32
            ),
            "avg3_rank": rank_avg_oof.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        prediction_frame.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    prediction_frame.to_csv(
        args.output_dir
        / "oof_predictions.csv",
        index=False,
    )

    test_frame = pd.DataFrame(
        {
            "seed42": seed_test[
                42
            ].astype(
                np.float32
            ),
            "seed7": seed_test[
                7
            ].astype(
                np.float32
            ),
            "seed2026": seed_test[
                2026
            ].astype(
                np.float32
            ),
            "avg3_probability": prob_avg_test.astype(
                np.float32
            ),
            "avg3_rank": rank_avg_test.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        test_frame.insert(
            0,
            id_col,
            test[
                id_col
            ].to_numpy(),
        )

    test_frame.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    best_oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": best_oof.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        best_oof_output.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    best_oof_output.to_csv(
        args.output_dir
        / "best_average_oof_predictions.csv",
        index=False,
    )

    best_test_output = pd.DataFrame(
        {
            "prediction": best_test.astype(
                np.float32
            )
        }
    )

    if id_col is not None:
        best_test_output.insert(
            0,
            id_col,
            test[
                id_col
            ].to_numpy(),
        )

    best_test_output.to_csv(
        args.output_dir
        / "best_average_test_predictions.csv",
        index=False,
    )

    decision = (
        "KEEP"
        if (
            best_delta > 0
            and
            folds_improved >= 3
        )
        else
        "REJECT_FOR_NOW"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    lines = [
        "EXPERIMENT: CATBOOST_VALUE_IDS_MULTI_SEED_GPU",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Does equal averaging across seeds 42, 7, and 2026 reduce",
        "GPU CatBoost ranking noise and improve the exact-value champion?",
        "",
        "CONTROL",
        "Same frozen folds, exact same value-ID feature representation,",
        "same model hyperparameters. Only random_seed changes.",
        "",
        "RESULTS",
    ]

    for row in results.itertuples():
        lines.append(
            f"- {row.candidate}: "
            f"OOF={row.oof_auc:.8f}, "
            f"delta_vs_seed42={row.delta_vs_seed42:+.8f}"
        )

    lines.extend(
        [
            "",
            "BEST EQUAL AVERAGE",
            f"Candidate: {best_average_name}",
            f"OOF AUC: {best_average_auc:.8f}",
            f"Delta vs seed42: {best_delta:+.8f}",
            f"Folds improved: {folds_improved}/5",
            f"Training runtime for new seeds only: {total_seconds:.2f} seconds",
            "",
            f"DECISION: {decision}",
            "",
            "SEED CORRELATIONS",
        ]
    )

    for row in correlations.itertuples():
        lines.append(
            f"- {row.seed_a} vs {row.seed_b}: "
            f"prob_corr={row.probability_correlation:.6f}, "
            f"rank_corr={row.rank_correlation:.6f}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "Averaging is retained only if the equal-weight ensemble beats",
            "the saved seed-42 champion and improves at least 3/5 folds.",
            "",
            "No blend weights were fitted on OOF labels.",
        ]
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CATBOOST VALUE-ID MULTI-SEED EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Seed-42 champion : {seed42_auc:.8f}"
    )

    print(
        f"Best avg method  : {best_average_name}"
    )

    print(
        f"Best avg OOF     : {best_average_auc:.8f}"
    )

    print(
        f"Delta            : {best_delta:+.8f}"
    )

    print(
        f"Folds improved   : {folds_improved}/5"
    )

    print(
        f"Decision         : {decision}"
    )

    print(
        f"New-seed runtime : {total_seconds:.2f}s"
    )

    print(
        f"Artifacts        : {args.output_dir.resolve()}"
    )

    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. candidate_summary.csv")
    print("  4. ensemble_fold_metrics.csv")
    print("  5. seed_correlations.csv")


if __name__ == "__main__":
    main()
