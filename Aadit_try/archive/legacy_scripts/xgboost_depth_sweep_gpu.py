"""
xgboost_depth_sweep_gpu.py

Phase 4C — Controlled XGBoost tree-depth sweep on frozen folds.

HYPOTHESIS
----------
Our current XGBoost baseline uses max_depth=6.
Test whether shallower or deeper trees improve ROC-AUC while keeping every
other modeling choice fixed.

WHY THIS IS A GOOD NEXT EXPERIMENT
----------------------------------
XGBoost is our current best model:
    OOF AUC = 0.94189926

A depth sweep directly tests model complexity:
- too shallow -> underfits interactions
- too deep    -> overfits/noisy splits
- best depth  -> better bias/variance trade-off

CONTROLLED VARIABLES
--------------------
SAME:
- frozen 5 folds
- 13 raw features
- categorical handling
- GPU training
- learning_rate=0.03
- n_estimators=5000
- min_child_weight=8
- subsample=0.90
- colsample_bytree=0.90
- reg_lambda=2
- early stopping=200

ONLY CHANGE:
- max_depth in [4, 5, 6, 7, 8]

No Kaggle submission is created.

Run:
    python xgboost_depth_sweep_gpu.py
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
DEPTHS = [4, 5, 6, 7, 8]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
    required = {
        "row_index",
        "fold",
    }

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

    if sorted(
        folds["fold"].unique().tolist()
    ) != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected folds [0,1,2,3,4]."
        )

    extra = [
        c
        for c in folds.columns
        if c not in {"row_index", "fold"}
    ]

    if len(extra) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extra}"
        )

    id_col = (
        extra[0]
        if extra
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} missing from train."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                "Fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(
            dtype=np.int16
        ),
        id_col,
    )


def detect_categoricals(
    train: pd.DataFrame,
    features: list[str],
) -> list[str]:
    categorical = []

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
            categorical.append(c)

    return categorical


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categoricals: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[
        features
    ].copy()

    x_test = test[
        features
    ].copy()

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

    return x_train, x_test


def load_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline OOF not found: {path.resolve()}"
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
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Baseline OOF row order mismatch."
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Baseline OOF folds differ."
        )

    if (
        id_col is not None
        and id_col in df.columns
    ):
        if not np.array_equal(
            df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                "Baseline OOF IDs differ."
            )

    return df[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )


def build_model(
    depth: int,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",

        n_estimators=5000,
        learning_rate=0.03,

        max_depth=depth,
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


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(
            pct=True,
            method="average",
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            pct=True,
            method="average",
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
        "Loading data, frozen folds, and XGBoost baseline..."
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

    X, X_test = prepare_frames(
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
        f"Official XGBoost baseline OOF: "
        f"{baseline_auc:.8f}"
    )

    print(
        f"Depths to test: {DEPTHS}"
    )

    print()
    print(
        "Only max_depth changes. "
        "Everything else is frozen."
    )
    print()

    fold_rows = []

    oof_by_depth = {}
    test_by_depth = {}

    total_start = (
        time.perf_counter()
    )

    for depth in DEPTHS:
        print()
        print(
            "=" * 78
        )

        print(
            f"TESTING max_depth={depth}"
        )

        print(
            "=" * 78
        )

        depth_oof = np.full(
            len(train),
            np.nan,
            dtype=np.float64,
        )

        depth_test_preds = []

        depth_start = (
            time.perf_counter()
        )

        for fold in range(5):
            train_idx = np.flatnonzero(
                fold_ids != fold
            )

            valid_idx = np.flatnonzero(
                fold_ids == fold
            )

            model = build_model(
                depth=depth
            )

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
                verbose=False,
            )

            fit_seconds = (
                time.perf_counter()
                - fit_start
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

            depth_oof[
                valid_idx
            ] = valid_pred

            depth_test_preds.append(
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

            fold_rows.append(
                {
                    "max_depth": depth,
                    "fold": fold,
                    "auc": fold_auc,
                    "baseline_depth6_auc": baseline_fold_auc,
                    "delta_vs_official_baseline": (
                        fold_auc
                        - baseline_fold_auc
                    ),
                    "best_iteration_zero_based": int(
                        model.best_iteration
                    ),
                    "fit_seconds": float(
                        fit_seconds
                    ),
                }
            )

            print(
                f"Depth {depth} | "
                f"Fold {fold}: "
                f"AUC={fold_auc:.6f} | "
                f"baseline={baseline_fold_auc:.6f} | "
                f"delta="
                f"{fold_auc - baseline_fold_auc:+.6f} | "
                f"best_iter={model.best_iteration} | "
                f"fit={fit_seconds:.1f}s"
            )

            del model

        if np.isnan(
            depth_oof
        ).any():
            raise RuntimeError(
                f"Depth {depth} OOF contains NaNs."
            )

        depth_auc = float(
            roc_auc_score(
                y,
                depth_oof,
            )
        )

        depth_seconds = (
            time.perf_counter()
            - depth_start
        )

        oof_by_depth[
            depth
        ] = depth_oof

        test_by_depth[
            depth
        ] = np.mean(
            np.vstack(
                depth_test_preds
            ),
            axis=0,
        )

        print()
        print(
            f"Depth {depth} COMPLETE | "
            f"OOF={depth_auc:.8f} | "
            f"delta_vs_baseline="
            f"{depth_auc - baseline_auc:+.8f} | "
            f"time={depth_seconds:.1f}s"
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    summary_rows = []

    for depth in DEPTHS:
        subset = fold_metrics[
            fold_metrics[
                "max_depth"
            ] == depth
        ]

        depth_oof = oof_by_depth[
            depth
        ]

        depth_auc = float(
            roc_auc_score(
                y,
                depth_oof,
            )
        )

        summary_rows.append(
            {
                "max_depth": depth,
                "oof_auc": depth_auc,
                "delta_vs_official_depth6_baseline": (
                    depth_auc
                    - baseline_auc
                ),
                "mean_fold_auc": float(
                    subset[
                        "auc"
                    ].mean()
                ),
                "fold_auc_std": float(
                    subset[
                        "auc"
                    ].std(
                        ddof=1
                    )
                ),
                "folds_beating_official_baseline": int(
                    (
                        subset[
                            "delta_vs_official_baseline"
                        ] > 0
                    ).sum()
                ),
                "mean_best_iteration": float(
                    subset[
                        "best_iteration_zero_based"
                    ].mean()
                ),
                "mean_fit_seconds": float(
                    subset[
                        "fit_seconds"
                    ].mean()
                ),
                "rank_corr_vs_official_baseline": rank_corr(
                    depth_oof,
                    baseline_oof,
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

    best_depth = int(
        depth_summary.iloc[
            0
        ][
            "max_depth"
        ]
    )

    best_auc = float(
        depth_summary.iloc[
            0
        ][
            "oof_auc"
        ]
    )

    best_delta = (
        best_auc
        - baseline_auc
    )

    # ----------------------------------------------------
    # Save compact prediction matrices
    # ----------------------------------------------------

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
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

    for depth in DEPTHS:
        oof_output[
            f"oof_depth_{depth}"
        ] = (
            oof_by_depth[
                depth
            ].astype(
                np.float32
            )
        )

    oof_output.to_csv(
        args.output_dir
        / "oof_predictions_by_depth.csv",
        index=False,
    )

    test_output = pd.DataFrame()

    if id_col is not None:
        test_output[
            id_col
        ] = test[
            id_col
        ].to_numpy()

    for depth in DEPTHS:
        test_output[
            f"prediction_depth_{depth}"
        ] = (
            test_by_depth[
                depth
            ].astype(
                np.float32
            )
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions_by_depth.csv",
        index=False,
    )

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

    # Also save the best depth in the standard OOF/test format,
    # so later experiments can use it directly.

    best_oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": (
                oof_by_depth[
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
                test_by_depth[
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

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    lines = [
        "EXPERIMENT: XGBOOST_GPU_MAX_DEPTH_SWEEP",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Does changing XGBoost tree depth improve the bias/variance trade-off?",
        "",
        "CONTROL",
        "Everything except max_depth was held fixed.",
        f"Official depth=6 baseline OOF AUC: {baseline_auc:.8f}",
        "",
        "DEPTH RESULTS",
    ]

    for row in depth_summary.itertuples():
        lines.append(
            f"- depth={row.max_depth}: "
            f"OOF={row.oof_auc:.8f}, "
            f"delta={row.delta_vs_official_depth6_baseline:+.8f}, "
            f"folds_won={row.folds_beating_official_baseline}/5, "
            f"fold_std={row.fold_auc_std:.8f}, "
            f"mean_best_iter={row.mean_best_iteration:.1f}"
        )

    lines.extend(
        [
            "",
            "BEST RESULT",
            f"Best depth: {best_depth}",
            f"Best OOF AUC: {best_auc:.8f}",
            f"Delta vs official baseline: {best_delta:+.8f}",
            f"Total experiment runtime: {total_seconds:.2f} seconds",
            "",
            "DECISION RULE",
            "If a different depth produces a consistent meaningful gain,",
            "that depth becomes the base for the next tuning experiment.",
            "",
            "If depth=6 remains best, we freeze depth and tune another",
            "hyperparameter family instead of repeatedly retesting it.",
        ]
    )

    summary_text = "\n".join(
        lines
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(
        "=" * 78
    )

    print(
        "XGBOOST GPU DEPTH SWEEP COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Official baseline : "
        f"{baseline_auc:.8f}"
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
        f"Total runtime     : "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Artifacts         : "
        f"{args.output_dir.resolve()}"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Send me:"
    )

    print(
        "  1. terminal output"
    )

    print(
        "  2. summary.txt"
    )

    print(
        "  3. depth_summary.csv"
    )

    print(
        "  4. fold_metrics.csv"
    )


if __name__ == "__main__":
    main()
