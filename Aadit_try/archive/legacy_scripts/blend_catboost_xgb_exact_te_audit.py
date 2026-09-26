"""
blend_catboost_xgb_exact_te_audit.py

Phase 5K — Leakage-safe CatBoost + XGBoost blend audit.

MODELS
------
A) Current CatBoost champion
   3-seed rank average + exact value IDs
   OOF ≈ 0.94543956

B) XGBoost exact-value Bayesian TE
   OOF ≈ 0.94441818

WHY THIS EXPERIMENT
-------------------
XGBoost is now strong enough to matter and its probability correlation with
CatBoost is much lower than the old raw models. We therefore test whether
their OOF rankings complement each other.

IMPORTANT
---------
We do NOT choose a blend weight on the full OOF and then report that same OOF
score as proof.

Instead, for each held-out frozen fold:

    1. choose the blend weight using the OTHER four folds only
    2. apply that selected weight to the held-out fold
    3. concatenate the five held-out blended predictions
    4. compute one meta-OOF AUC

This gives a leakage-safe estimate of whether weight selection generalizes.

PRIMARY BLEND
-------------
Weighted rank blend:

    blend = w_cat * rank(catboost) + (1 - w_cat) * rank(xgboost)

Weight grid:
    0.50, 0.51, ..., 1.00

We also report several fixed weights for context.

NO KAGGLE SUBMISSION IS CREATED.

Run:
    python blend_catboost_xgb_exact_te_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_CAT_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_test_predictions.csv"
)

DEFAULT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_gpu"
    / "oof_predictions.csv"
)

DEFAULT_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_gpu"
    / "test_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_xgb_exact_te_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe CatBoost/XGBoost weighted-rank blend audit."
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
        "--cat-oof",
        type=Path,
        default=DEFAULT_CAT_OOF,
    )

    parser.add_argument(
        "--cat-test",
        type=Path,
        default=DEFAULT_CAT_TEST,
    )

    parser.add_argument(
        "--xgb-oof",
        type=Path,
        default=DEFAULT_XGB_OOF,
    )

    parser.add_argument(
        "--xgb-test",
        type=Path,
        default=DEFAULT_XGB_TEST,
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
            "Frozen fold row count mismatch."
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
            "Frozen fold row order mismatch."
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
                f"Fold ID {id_col!r} missing from train."
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
                "Fold IDs do not align with train.csv."
            )

    return (
        folds[
            "fold"
        ].to_numpy(
            dtype=np.int16
        ),
        id_col,
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
            f"OOF folds mismatch in {path}"
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


def load_test(
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
            f"Expected 'prediction' column in {path}"
        )

    if len(df) != len(test):
        raise ValueError(
            f"Test row count mismatch in {path}"
        )

    if (
        id_col is not None
        and id_col in df.columns
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
            f"Test ID mismatch in {path}"
        )

    return (
        df[
            "prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


def rank01(
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


def blend_ranked(
    cat_rank: np.ndarray,
    xgb_rank: np.ndarray,
    cat_weight: float,
) -> np.ndarray:
    return (
        cat_weight * cat_rank
        + (1.0 - cat_weight) * xgb_rank
    )


def choose_weight(
    y: np.ndarray,
    cat_rank: np.ndarray,
    xgb_rank: np.ndarray,
    mask: np.ndarray,
    weight_grid: np.ndarray,
) -> tuple[
    float,
    float,
]:
    best_weight = None
    best_auc = -np.inf

    for w in weight_grid:
        pred = blend_ranked(
            cat_rank[
                mask
            ],
            xgb_rank[
                mask
            ],
            float(w),
        )

        auc = float(
            roc_auc_score(
                y[
                    mask
                ],
                pred,
            )
        )

        if (
            auc > best_auc + 1e-15
            or (
                abs(
                    auc - best_auc
                ) <= 1e-15
                and (
                    best_weight is None
                    or abs(
                        float(w) - 1.0
                    )
                    < abs(
                        best_weight - 1.0
                    )
                )
            )
        ):
            best_auc = auc
            best_weight = float(w)

    assert best_weight is not None

    return (
        best_weight,
        best_auc,
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
        args.cat_oof,
        args.cat_test,
        args.xgb_oof,
        args.xgb_test,
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
        "Loading aligned OOF/test predictions..."
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    folds = pd.read_csv(
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
            folds,
            train,
        )
    )

    cat_oof = load_oof(
        args.cat_oof,
        train,
        fold_ids,
        id_col,
    )

    xgb_oof = load_oof(
        args.xgb_oof,
        train,
        fold_ids,
        id_col,
    )

    cat_test = load_test(
        args.cat_test,
        test,
        id_col,
    )

    xgb_test = load_test(
        args.xgb_test,
        test,
        id_col,
    )

    cat_auc = float(
        roc_auc_score(
            y,
            cat_oof,
        )
    )

    xgb_auc = float(
        roc_auc_score(
            y,
            xgb_oof,
        )
    )

    cat_rank = rank01(
        cat_oof
    )

    xgb_rank = rank01(
        xgb_oof
    )

    cat_test_rank = rank01(
        cat_test
    )

    xgb_test_rank = rank01(
        xgb_test
    )

    probability_corr = float(
        np.corrcoef(
            cat_oof,
            xgb_oof,
        )[0, 1]
    )

    rank_corr = float(
        np.corrcoef(
            cat_rank,
            xgb_rank,
        )[0, 1]
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"CatBoost champion OOF: "
        f"{cat_auc:.8f}"
    )

    print(
        f"XGBoost exact-TE OOF: "
        f"{xgb_auc:.8f}"
    )

    print(
        f"OOF probability correlation: "
        f"{probability_corr:.6f}"
    )

    print(
        f"OOF rank correlation: "
        f"{rank_corr:.6f}"
    )

    print()
    print(
        "Primary audit: meta-CV weighted rank blend"
    )

    weight_grid = np.round(
        np.arange(
            0.50,
            1.0001,
            0.01,
        ),
        2,
    )

    meta_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    meta_rows = []

    selected_weights = []

    for held_fold in range(5):
        fit_mask = (
            fold_ids != held_fold
        )

        held_mask = (
            fold_ids == held_fold
        )

        (
            selected_weight,
            fit_auc,
        ) = choose_weight(
            y=y,
            cat_rank=cat_rank,
            xgb_rank=xgb_rank,
            mask=fit_mask,
            weight_grid=weight_grid,
        )

        held_pred = blend_ranked(
            cat_rank[
                held_mask
            ],
            xgb_rank[
                held_mask
            ],
            selected_weight,
        )

        held_blend_auc = float(
            roc_auc_score(
                y[
                    held_mask
                ],
                held_pred,
            )
        )

        held_cat_auc = float(
            roc_auc_score(
                y[
                    held_mask
                ],
                cat_oof[
                    held_mask
                ],
            )
        )

        held_xgb_auc = float(
            roc_auc_score(
                y[
                    held_mask
                ],
                xgb_oof[
                    held_mask
                ],
            )
        )

        meta_oof[
            held_mask
        ] = held_pred

        selected_weights.append(
            selected_weight
        )

        meta_rows.append(
            {
                "held_fold": held_fold,

                "selected_cat_weight": (
                    selected_weight
                ),

                "training_four_folds_auc": (
                    fit_auc
                ),

                "held_cat_auc": (
                    held_cat_auc
                ),

                "held_xgb_auc": (
                    held_xgb_auc
                ),

                "held_blend_auc": (
                    held_blend_auc
                ),

                "delta_blend_vs_cat": (
                    held_blend_auc
                    - held_cat_auc
                ),
            }
        )

        print(
            f"Fold {held_fold}: "
            f"selected w_cat={selected_weight:.2f} | "
            f"CAT={held_cat_auc:.6f} | "
            f"XGB={held_xgb_auc:.6f} | "
            f"blend={held_blend_auc:.6f} | "
            f"delta_CAT={held_blend_auc - held_cat_auc:+.6f}"
        )

    if np.isnan(
        meta_oof
    ).any():
        raise RuntimeError(
            "Meta-CV OOF contains NaNs."
        )

    meta_fold_metrics = pd.DataFrame(
        meta_rows
    )

    meta_auc = float(
        roc_auc_score(
            y,
            meta_oof,
        )
    )

    meta_delta = (
        meta_auc
        - cat_auc
    )

    meta_fold_wins = int(
        (
            meta_fold_metrics[
                "delta_blend_vs_cat"
            ] > 0
        ).sum()
    )

    selected_weights = np.array(
        selected_weights,
        dtype=np.float64,
    )

    mean_selected_weight = float(
        selected_weights.mean()
    )

    std_selected_weight = float(
        selected_weights.std(
            ddof=1
        )
    )

    # ---------------------------------------------------------
    # Fixed-weight context
    # ---------------------------------------------------------

    fixed_rows = []

    fixed_weights = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]

    for w in fixed_weights:
        pred = blend_ranked(
            cat_rank,
            xgb_rank,
            w,
        )

        auc = float(
            roc_auc_score(
                y,
                pred,
            )
        )

        fold_deltas = []

        for fold in range(5):
            mask = (
                fold_ids == fold
            )

            blend_fold_auc = float(
                roc_auc_score(
                    y[
                        mask
                    ],
                    pred[
                        mask
                    ],
                )
            )

            cat_fold_auc = float(
                roc_auc_score(
                    y[
                        mask
                    ],
                    cat_oof[
                        mask
                    ],
                )
            )

            fold_deltas.append(
                blend_fold_auc
                - cat_fold_auc
            )

        fixed_rows.append(
            {
                "cat_weight": w,
                "xgb_weight": 1.0 - w,
                "full_oof_auc_diagnostic_only": auc,
                "delta_vs_cat_diagnostic_only": (
                    auc
                    - cat_auc
                ),
                "folds_beating_cat": int(
                    sum(
                        d > 0
                        for d in fold_deltas
                    )
                ),
                "minimum_fold_delta_vs_cat": float(
                    min(
                        fold_deltas
                    )
                ),
                "mean_fold_delta_vs_cat": float(
                    np.mean(
                        fold_deltas
                    )
                ),
            }
        )

    fixed_summary = (
        pd.DataFrame(
            fixed_rows
        )
        .sort_values(
            "full_oof_auc_diagnostic_only",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # Full-OOF selected weight is diagnostic only.
    # It is NOT used as evidence for promotion.
    # ---------------------------------------------------------

    (
        full_oof_best_weight,
        full_oof_best_auc,
    ) = choose_weight(
        y=y,
        cat_rank=cat_rank,
        xgb_rank=xgb_rank,
        mask=np.ones(
            len(train),
            dtype=bool,
        ),
        weight_grid=weight_grid,
    )

    # Conservative deployment weight:
    # use the mean of five meta-selected weights.
    deployment_weight = float(
        np.round(
            mean_selected_weight,
            2,
        )
    )

    deployment_weight = float(
        np.clip(
            deployment_weight,
            0.50,
            1.00,
        )
    )

    deployment_oof = blend_ranked(
        cat_rank,
        xgb_rank,
        deployment_weight,
    )

    deployment_auc = float(
        roc_auc_score(
            y,
            deployment_oof,
        )
    )

    deployment_test = blend_ranked(
        cat_test_rank,
        xgb_test_rank,
        deployment_weight,
    )

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    decision = (
        "KEEP"
        if (
            meta_delta > 0
            and
            meta_fold_wins >= 3
            and
            mean_selected_weight < 0.995
        )
        else
        "REJECT_FOR_NOW"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    meta_fold_metrics.to_csv(
        args.output_dir
        / "meta_fold_metrics.csv",
        index=False,
    )

    fixed_summary.to_csv(
        args.output_dir
        / "fixed_weight_summary.csv",
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

            "catboost_oof": (
                cat_oof.astype(
                    np.float32
                )
            ),

            "xgboost_oof": (
                xgb_oof.astype(
                    np.float32
                )
            ),

            "meta_cv_blend_oof": (
                meta_oof.astype(
                    np.float32
                )
            ),

            "deployment_rank_blend_oof": (
                deployment_oof.astype(
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
                deployment_test.astype(
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
        "EXPERIMENT: CATBOOST_XGBOOST_EXACT_TE_BLEND_AUDIT",
        "=" * 76,
        "",
        "QUESTION",
        "Does the strong XGBoost exact-value Bayesian-TE model add useful",
        "OOF ranking diversity to the current CatBoost champion?",
        "",
        "BASE MODELS",
        f"CatBoost champion OOF AUC: {cat_auc:.8f}",
        f"XGBoost exact-TE OOF AUC: {xgb_auc:.8f}",
        f"Probability correlation: {probability_corr:.6f}",
        f"Rank correlation: {rank_corr:.6f}",
        "",
        "PRIMARY EVIDENCE: META-CV WEIGHT SELECTION",
        "Each held fold uses a blend weight chosen only on the other 4 folds.",
        f"Meta-CV blended OOF AUC: {meta_auc:.8f}",
        f"Delta vs CatBoost champion: {meta_delta:+.8f}",
        f"Held folds beating CatBoost: {meta_fold_wins}/5",
        f"Mean selected CatBoost weight: {mean_selected_weight:.4f}",
        f"Std selected CatBoost weight: {std_selected_weight:.4f}",
        f"Selected weights: {selected_weights.tolist()}",
        "",
        "DEPLOYMENT WEIGHT",
        f"Rounded mean meta-selected CatBoost weight: {deployment_weight:.2f}",
        f"XGBoost weight: {1.0 - deployment_weight:.2f}",
        f"Deployment-weight full OOF AUC (diagnostic): {deployment_auc:.8f}",
        "",
        "FULL-OOF OPTIMUM — DIAGNOSTIC ONLY",
        f"Best full-OOF CatBoost weight: {full_oof_best_weight:.2f}",
        f"Best full-OOF AUC: {full_oof_best_auc:.8f}",
        "This full-OOF optimum is NOT used as promotion evidence.",
        "",
        f"DECISION: {decision}",
        "",
        "META-FOLD RESULTS",
    ]

    for row in meta_fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.held_fold}: "
            f"w_cat={row.selected_cat_weight:.2f}, "
            f"CAT={row.held_cat_auc:.8f}, "
            f"blend={row.held_blend_auc:.8f}, "
            f"delta={row.delta_blend_vs_cat:+.8f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "A blend is kept only if out-of-fold weight selection generalizes",
            "to held folds. Full-OOF weight optimization is reported only as",
            "a diagnostic and cannot by itself justify a submission.",
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
        "CATBOOST + XGBOOST BLEND AUDIT COMPLETE"
    )
    print("=" * 78)

    print(
        f"CatBoost OOF      : "
        f"{cat_auc:.8f}"
    )

    print(
        f"XGBoost OOF       : "
        f"{xgb_auc:.8f}"
    )

    print(
        f"Meta-CV blend OOF : "
        f"{meta_auc:.8f}"
    )

    print(
        f"Delta vs CatBoost : "
        f"{meta_delta:+.8f}"
    )

    print(
        f"Fold wins         : "
        f"{meta_fold_wins}/5"
    )

    print(
        f"Mean selected CAT : "
        f"{mean_selected_weight:.4f}"
    )

    print(
        f"Deployment weight : "
        f"CAT {deployment_weight:.2f} / "
        f"XGB {1.0 - deployment_weight:.2f}"
    )

    print(
        f"Decision          : "
        f"{decision}"
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
    print("  3. meta_fold_metrics.csv")
    print("  4. fixed_weight_summary.csv")


if __name__ == "__main__":
    main()
