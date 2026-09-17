"""
catboost_value_ids_depth_sweep_gpu.py

Phase 5H — CatBoost depth sweep on the exact-value identity representation.

CURRENT SINGLE-SEED CHAMPION
----------------------------
CatBoost GPU + exact value identities, seed=42, depth=6:
    OOF AUC = 0.94536428

CURRENT MULTI-SEED CHAMPION
--------------------------
3-seed rank average:
    OOF AUC = 0.94543956

HYPOTHESIS
----------
The exact-value identity representation changed the structure of the problem.
Depth=6 was inherited from our raw CatBoost baseline, so it may no longer be
the best tree complexity.

CONTROLLED EXPERIMENT
---------------------
SAME:
- exact frozen 5 folds
- 13 raw features
- Annual_Income_USD_id
- Daily_Commute_km_id
- seed=42
- GPU
- iterations=2000
- learning_rate=0.05
- early_stopping_rounds=150

ONLY CHANGE:
- depth

We reuse the already-saved depth=6 predictions and only train:
    depth=5
    depth=7

This keeps runtime reasonable while testing the two nearest alternatives.

Run:
    python catboost_value_ids_depth_sweep_gpu.py
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
DEPTHS_TO_TRAIN = [5, 7]

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_DEPTH6_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "oof_predictions.csv"
)

DEFAULT_DEPTH6_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "test_predictions.csv"
)

DEFAULT_MULTI_SEED_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_depth_sweep_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth sweep for exact-value CatBoost GPU."
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
        "--depth6-oof",
        type=Path,
        default=DEFAULT_DEPTH6_OOF,
    )

    parser.add_argument(
        "--depth6-test",
        type=Path,
        default=DEFAULT_DEPTH6_TEST,
    )

    parser.add_argument(
        "--multiseed-oof",
        type=Path,
        default=DEFAULT_MULTI_SEED_OOF,
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
        c for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            f"Expected one train-only target column, found {train_only}"
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
]:
    x_train = train[
        raw_features
    ].copy()

    x_test = test[
        raw_features
    ].copy()

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
        id_col = (
            f"{feature}_id"
        )

        x_train[id_col] = stable_value_id(
            train[feature],
            feature,
        )

        x_test[id_col] = stable_value_id(
            test[feature],
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

    return (
        df[
            "oof_prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


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

    if "prediction" not in df.columns:
        raise ValueError(
            "Expected a 'prediction' column."
        )

    if len(df) != len(test):
        raise ValueError(
            "Test prediction row count mismatch."
        )

    if (
        id_col is not None
        and
        id_col in df.columns
        and not np.array_equal(
            df[
                id_col
            ].to_numpy(),
            test[
                id_col
            ].to_numpy(),
        )
    ):
        raise ValueError(
            "Test prediction IDs are misaligned."
        )

    return (
        df[
            "prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
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


def build_model(
    depth: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=2000,
        learning_rate=0.05,
        depth=depth,

        random_seed=SEED,

        task_type="GPU",
        devices="0",

        allow_writing_files=False,
    )


def train_depth(
    depth: int,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    categorical_features: list[str],
    depth6_oof: np.ndarray,
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

    test_preds = []
    rows = []

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=list(
            X.columns
        ),
    )

    print()
    print("=" * 78)
    print(
        f"TESTING DEPTH {depth}"
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

        model = build_model(
            depth=depth
        )

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

        oof[
            valid_idx
        ] = valid_pred

        test_preds.append(
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

        depth6_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                depth6_oof[
                    valid_idx
                ],
            )
        )

        rows.append(
            {
                "depth": depth,
                "fold": fold,

                "depth6_auc": (
                    depth6_fold_auc
                ),

                "candidate_auc": (
                    fold_auc
                ),

                "delta_vs_depth6": (
                    fold_auc
                    - depth6_fold_auc
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
            }
        )

        print()
        print(
            f"Depth {depth} | Fold {fold}: "
            f"depth6={depth6_fold_auc:.6f} | "
            f"candidate={fold_auc:.6f} | "
            f"delta={fold_auc - depth6_fold_auc:+.6f} | "
            f"best_iter={model.get_best_iteration()} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            train_pool,
            valid_pool,
        )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            f"Depth {depth} OOF contains NaN values."
        )

    test_average = np.mean(
        np.vstack(
            test_preds
        ),
        axis=0,
    )

    return (
        oof,
        test_average,
        pd.DataFrame(
            rows
        ),
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
        args.depth6_oof,
        args.depth6_test,
        args.multiseed_oof,
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
        "Loading data, frozen folds, and saved champions..."
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
    ) = prepare_frames(
        train,
        test,
        raw_features,
        raw_categoricals,
    )

    depth6_oof = load_oof(
        args.depth6_oof,
        train,
        fold_ids,
        id_col,
    )

    depth6_test = load_test_predictions(
        args.depth6_test,
        test,
        id_col,
    )

    multiseed_oof = load_oof(
        args.multiseed_oof,
        train,
        fold_ids,
        id_col,
    )

    depth6_auc = float(
        roc_auc_score(
            y,
            depth6_oof,
        )
    )

    multiseed_auc = float(
        roc_auc_score(
            y,
            multiseed_oof,
        )
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Saved depth-6 seed-42 OOF: "
        f"{depth6_auc:.8f}"
    )

    print(
        f"Current 3-seed rank champion: "
        f"{multiseed_auc:.8f}"
    )

    print()
    print(
        "Depths to train: "
        f"{DEPTHS_TO_TRAIN}"
    )

    print()
    print(
        "Only tree depth changes."
    )
    print()

    all_oof = {
        6: depth6_oof,
    }

    all_test = {
        6: depth6_test,
    }

    all_fold_metrics = []

    total_start = (
        time.perf_counter()
    )

    for depth in DEPTHS_TO_TRAIN:
        (
            depth_oof,
            depth_test,
            fold_metrics,
        ) = train_depth(
            depth=depth,
            X=X,
            X_test=X_test,
            y=y,
            fold_ids=fold_ids,
            categorical_features=categorical_features,
            depth6_oof=depth6_oof,
        )

        all_oof[
            depth
        ] = depth_oof

        all_test[
            depth
        ] = depth_test

        all_fold_metrics.append(
            fold_metrics
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    summary_rows = []

    for depth in [
        5,
        6,
        7,
    ]:
        score = float(
            roc_auc_score(
                y,
                all_oof[
                    depth
                ],
            )
        )

        if depth == 6:
            folds_won = 0
        else:
            subset = (
                fold_metrics[
                    fold_metrics[
                        "depth"
                    ] == depth
                ]
            )

            folds_won = int(
                (
                    subset[
                        "delta_vs_depth6"
                    ] > 0
                ).sum()
            )

        summary_rows.append(
            {
                "depth": depth,

                "oof_auc": score,

                "delta_vs_depth6": (
                    score
                    - depth6_auc
                ),

                "folds_beating_depth6": (
                    folds_won
                ),

                "rank_corr_vs_depth6": (
                    1.0
                    if depth == 6
                    else rank_corr(
                        all_oof[
                            depth
                        ],
                        depth6_oof,
                    )
                ),
            }
        )

    depth_summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "oof_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best_row = (
        depth_summary
        .iloc[0]
    )

    best_depth = int(
        best_row[
            "depth"
        ]
    )

    best_auc = float(
        best_row[
            "oof_auc"
        ]
    )

    best_delta = (
        best_auc
        - depth6_auc
    )

    if best_depth == 6:
        decision = (
            "KEEP_DEPTH_6"
        )
    else:
        best_folds_won = int(
            best_row[
                "folds_beating_depth6"
            ]
        )

        decision = (
            "PROMOTE_NEW_DEPTH"
            if (
                best_delta > 0
                and
                best_folds_won >= 3
            )
            else
            "KEEP_DEPTH_6"
        )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    fold_metrics.to_csv(
        args.output_dir
        / "fold_metrics.csv",
        index=False,
    )

    depth_summary.to_csv(
        args.output_dir
        / "depth_summary.csv",
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

            "oof_depth_5": (
                all_oof[
                    5
                ].astype(
                    np.float32
                )
            ),

            "oof_depth_6": (
                all_oof[
                    6
                ].astype(
                    np.float32
                )
            ),

            "oof_depth_7": (
                all_oof[
                    7
                ].astype(
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
        / "oof_predictions_by_depth.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction_depth_5": (
                all_test[
                    5
                ].astype(
                    np.float32
                )
            ),

            "prediction_depth_6": (
                all_test[
                    6
                ].astype(
                    np.float32
                )
            ),

            "prediction_depth_7": (
                all_test[
                    7
                ].astype(
                    np.float32
                )
            ),
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
        / "test_predictions_by_depth.csv",
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

            "oof_prediction": (
                all_oof[
                    best_depth
                ].astype(
                    np.float32
                )
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
        / "best_oof_predictions.csv",
        index=False,
    )

    best_test_output = pd.DataFrame(
        {
            "prediction": (
                all_test[
                    best_depth
                ].astype(
                    np.float32
                )
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
        / "best_test_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    lines = [
        "EXPERIMENT: CATBOOST_VALUE_IDS_DEPTH_SWEEP_GPU",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Does the exact-value CatBoost representation prefer a different",
        "tree depth than the inherited depth=6 raw baseline?",
        "",
        "CONTROL",
        "Same frozen folds, features, seed, learning rate, iterations, GPU,",
        "and early stopping. Only depth changes.",
        "",
        f"Depth-6 single-seed reference: {depth6_auc:.8f}",
        f"3-seed rank champion context: {multiseed_auc:.8f}",
        "",
        "RESULTS",
    ]

    for row in (
        depth_summary
        .itertuples()
    ):
        lines.append(
            f"- depth={row.depth}: "
            f"OOF={row.oof_auc:.8f}, "
            f"delta_vs_depth6={row.delta_vs_depth6:+.8f}, "
            f"folds_beating_depth6={row.folds_beating_depth6}/5, "
            f"rank_corr={row.rank_corr_vs_depth6:.6f}"
        )

    lines.extend(
        [
            "",
            "BEST DEPTH",
            f"Depth: {best_depth}",
            f"OOF AUC: {best_auc:.8f}",
            f"Delta vs depth 6: {best_delta:+.8f}",
            f"Decision: {decision}",
            f"Runtime for newly trained depths: {total_seconds:.2f} seconds",
            "",
            "NEXT",
            "If depth 5 or 7 clearly wins, that depth becomes the new",
            "single-seed base before any further seed averaging.",
            "",
            "If depth 6 remains best, we freeze depth and move on rather",
            "than repeatedly tuning the same axis.",
        ]
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CATBOOST VALUE-ID DEPTH SWEEP COMPLETE"
    )
    print("=" * 78)

    print(
        f"Depth-6 reference : "
        f"{depth6_auc:.8f}"
    )

    print(
        f"Best depth        : "
        f"{best_depth}"
    )

    print(
        f"Best OOF          : "
        f"{best_auc:.8f}"
    )

    print(
        f"Delta             : "
        f"{best_delta:+.8f}"
    )

    print(
        f"3-seed champion   : "
        f"{multiseed_auc:.8f}"
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
    print("  3. depth_summary.csv")
    print("  4. fold_metrics.csv")


if __name__ == "__main__":
    main()
