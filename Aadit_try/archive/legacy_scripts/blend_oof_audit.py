"""
blend_oof_audit.py

Phase 4B — Leakage-aware OOF blend audit.

QUESTION
--------
Do our saved Logistic, CatBoost-GPU, and XGBoost-GPU OOF predictions contain
enough complementary information to improve ROC-AUC by blending?

WHY THIS SCRIPT EXISTS
----------------------
XGBoost is currently best, but CatBoost and Logistic make slightly different
ranking mistakes. Before spending time tuning models, we can cheaply test
whether those differences are useful.

IMPORTANT LEAKAGE-SAFETY IDEA
-----------------------------
A naive approach would:
    1. search the best blend weight on ALL OOF rows
    2. report AUC on those SAME rows

That can overfit the blend weight.

Instead, this script performs META-CV:
    - hold out one frozen fold
    - search blend weights using the other four folds' OOF predictions
    - evaluate the chosen blend on the held-out fold
    - repeat for all five folds

This gives us a much more trustworthy estimate of whether blending really
generalizes.

Models used:
    Logistic raw baseline
    CatBoost raw GPU baseline
    XGBoost raw GPU baseline

It tests:
    1. Probability blend: XGBoost + CatBoost
    2. Rank blend: XGBoost + CatBoost
    3. Probability blend: XGBoost + CatBoost + Logistic
    4. Rank blend: XGBoost + CatBoost + Logistic

No model training occurs in this file.

Run:
    python blend_oof_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_LOGISTIC_OOF = (
    Path("artifacts")
    / "experiments"
    / "logistic_raw"
    / "oof_predictions.csv"
)

DEFAULT_CATBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_XGBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_oof_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware meta-CV blend audit."
    )

    parser.add_argument(
        "--logistic-oof",
        type=Path,
        default=DEFAULT_LOGISTIC_OOF,
    )

    parser.add_argument(
        "--catboost-oof",
        type=Path,
        default=DEFAULT_CATBOOST_OOF,
    )

    parser.add_argument(
        "--xgboost-oof",
        type=Path,
        default=DEFAULT_XGBOOST_OOF,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--two-model-step",
        type=float,
        default=0.01,
        help="Weight step for XGBoost/CatBoost search.",
    )

    parser.add_argument(
        "--three-model-step",
        type=float,
        default=0.05,
        help="Simplex weight step for 3-model search.",
    )

    return parser.parse_args()


def load_and_validate_oof(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} OOF file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "target_encoded",
        "oof_prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} OOF file missing columns: {sorted(missing)}"
        )

    if df["oof_prediction"].isna().any():
        raise ValueError(
            f"{model_name} OOF predictions contain NaN values."
        )

    return df


def validate_alignment(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    other_name: str,
) -> None:
    if len(reference) != len(other):
        raise ValueError(
            f"{other_name} row count differs from reference."
        )

    for col in [
        "row_index",
        "fold",
        "target_encoded",
    ]:
        if not np.array_equal(
            reference[col].to_numpy(),
            other[col].to_numpy(),
        ):
            raise ValueError(
                f"{other_name} is misaligned on column {col!r}."
            )

    common_id_cols = [
        c
        for c in reference.columns
        if c not in {
            "row_index",
            "fold",
            "target_encoded",
            "oof_prediction",
        }
        and c in other.columns
    ]

    for col in common_id_cols:
        if not np.array_equal(
            reference[col].to_numpy(),
            other[col].to_numpy(),
        ):
            raise ValueError(
                f"{other_name} is misaligned on ID-like column {col!r}."
            )


def rank01(values: np.ndarray) -> np.ndarray:
    """
    Convert predictions to normalized ranks in [0, 1].

    ROC-AUC depends on ordering, so rank blending often works well when
    individual models have different probability calibration.
    """
    ranks = (
        pd.Series(values)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    return ranks


def auc(
    y: np.ndarray,
    pred: np.ndarray,
) -> float:
    return float(
        roc_auc_score(
            y,
            pred,
        )
    )


def search_two_model_weight(
    y: np.ndarray,
    xgb_pred: np.ndarray,
    cat_pred: np.ndarray,
    step: float,
) -> tuple[float, float]:
    """
    Blend:
        prediction = w_xgb * XGB + (1 - w_xgb) * CatBoost
    """
    if not (
        0 < step <= 1
    ):
        raise ValueError(
            "--two-model-step must be in (0, 1]."
        )

    weights = np.arange(
        0.0,
        1.0 + step / 2.0,
        step,
    )

    best_auc = -np.inf
    best_w_xgb = None

    for w_xgb in weights:
        blended = (
            w_xgb * xgb_pred
            + (1.0 - w_xgb) * cat_pred
        )

        score = auc(
            y,
            blended,
        )

        if score > best_auc:
            best_auc = score
            best_w_xgb = float(
                w_xgb
            )

    assert best_w_xgb is not None

    return (
        best_w_xgb,
        float(best_auc),
    )


def simplex_weights(
    step: float,
):
    """
    Generate 3-model non-negative weights that sum to 1.
    """
    if not (
        0 < step <= 1
    ):
        raise ValueError(
            "--three-model-step must be in (0, 1]."
        )

    n = round(
        1.0 / step
    )

    if not np.isclose(
        n * step,
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "--three-model-step must divide 1 exactly, "
            "for example 0.1, 0.05, 0.02."
        )

    for i in range(
        n + 1
    ):
        for j in range(
            n - i + 1
        ):
            k = (
                n
                - i
                - j
            )

            yield (
                i / n,
                j / n,
                k / n,
            )


def search_three_model_weights(
    y: np.ndarray,
    xgb_pred: np.ndarray,
    cat_pred: np.ndarray,
    log_pred: np.ndarray,
    step: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Search:
        w_xgb + w_cat + w_log = 1
        all weights >= 0
    """
    best_auc = -np.inf
    best_weights = None

    for (
        w_xgb,
        w_cat,
        w_log,
    ) in simplex_weights(
        step
    ):
        blended = (
            w_xgb * xgb_pred
            + w_cat * cat_pred
            + w_log * log_pred
        )

        score = auc(
            y,
            blended,
        )

        if score > best_auc:
            best_auc = score
            best_weights = (
                float(w_xgb),
                float(w_cat),
                float(w_log),
            )

    assert best_weights is not None

    return (
        best_weights[0],
        best_weights[1],
        best_weights[2],
        float(best_auc),
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading saved OOF predictions..."
    )

    logistic = load_and_validate_oof(
        args.logistic_oof,
        "Logistic",
    )

    catboost = load_and_validate_oof(
        args.catboost_oof,
        "CatBoost GPU",
    )

    xgboost = load_and_validate_oof(
        args.xgboost_oof,
        "XGBoost GPU",
    )

    validate_alignment(
        xgboost,
        catboost,
        "CatBoost GPU",
    )

    validate_alignment(
        xgboost,
        logistic,
        "Logistic",
    )

    y = xgboost[
        "target_encoded"
    ].to_numpy(
        dtype=np.int8
    )

    folds = xgboost[
        "fold"
    ].to_numpy(
        dtype=np.int16
    )

    pred_xgb = xgboost[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )

    pred_cat = catboost[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )

    pred_log = logistic[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )

    rank_xgb = rank01(
        pred_xgb
    )

    rank_cat = rank01(
        pred_cat
    )

    rank_log = rank01(
        pred_log
    )

    unique_folds = sorted(
        np.unique(
            folds
        ).tolist()
    )

    if unique_folds != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise ValueError(
            f"Expected folds [0,1,2,3,4], found {unique_folds}"
        )

    xgb_auc = auc(
        y,
        pred_xgb,
    )

    cat_auc = auc(
        y,
        pred_cat,
    )

    log_auc = auc(
        y,
        pred_log,
    )

    print()
    print(
        f"XGBoost OOF AUC : {xgb_auc:.8f}"
    )

    print(
        f"CatBoost OOF AUC: {cat_auc:.8f}"
    )

    print(
        f"Logistic OOF AUC: {log_auc:.8f}"
    )

    print()
    print(
        "Running leakage-aware meta-CV blend search..."
    )
    print()

    result_rows = []

    meta_oof = {
        "prob_xgb_cat": np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        ),
        "rank_xgb_cat": np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        ),
        "prob_three": np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        ),
        "rank_three": np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        ),
    }

    for held_fold in unique_folds:
        train_mask = (
            folds != held_fold
        )

        valid_mask = (
            folds == held_fold
        )

        # --------------------------------------------------
        # 2-model probability blend
        # --------------------------------------------------

        (
            w_xgb_prob_2,
            train_auc_prob_2,
        ) = search_two_model_weight(
            y=y[
                train_mask
            ],
            xgb_pred=pred_xgb[
                train_mask
            ],
            cat_pred=pred_cat[
                train_mask
            ],
            step=args.two_model_step,
        )

        pred_prob_2 = (
            w_xgb_prob_2
            * pred_xgb[
                valid_mask
            ]
            +
            (
                1.0
                - w_xgb_prob_2
            )
            * pred_cat[
                valid_mask
            ]
        )

        meta_oof[
            "prob_xgb_cat"
        ][
            valid_mask
        ] = pred_prob_2

        held_auc_prob_2 = auc(
            y[
                valid_mask
            ],
            pred_prob_2,
        )

        # --------------------------------------------------
        # 2-model rank blend
        # --------------------------------------------------

        (
            w_xgb_rank_2,
            train_auc_rank_2,
        ) = search_two_model_weight(
            y=y[
                train_mask
            ],
            xgb_pred=rank_xgb[
                train_mask
            ],
            cat_pred=rank_cat[
                train_mask
            ],
            step=args.two_model_step,
        )

        pred_rank_2 = (
            w_xgb_rank_2
            * rank_xgb[
                valid_mask
            ]
            +
            (
                1.0
                - w_xgb_rank_2
            )
            * rank_cat[
                valid_mask
            ]
        )

        meta_oof[
            "rank_xgb_cat"
        ][
            valid_mask
        ] = pred_rank_2

        held_auc_rank_2 = auc(
            y[
                valid_mask
            ],
            pred_rank_2,
        )

        # --------------------------------------------------
        # 3-model probability blend
        # --------------------------------------------------

        (
            w_xgb_prob_3,
            w_cat_prob_3,
            w_log_prob_3,
            train_auc_prob_3,
        ) = search_three_model_weights(
            y=y[
                train_mask
            ],
            xgb_pred=pred_xgb[
                train_mask
            ],
            cat_pred=pred_cat[
                train_mask
            ],
            log_pred=pred_log[
                train_mask
            ],
            step=args.three_model_step,
        )

        pred_prob_3 = (
            w_xgb_prob_3
            * pred_xgb[
                valid_mask
            ]
            +
            w_cat_prob_3
            * pred_cat[
                valid_mask
            ]
            +
            w_log_prob_3
            * pred_log[
                valid_mask
            ]
        )

        meta_oof[
            "prob_three"
        ][
            valid_mask
        ] = pred_prob_3

        held_auc_prob_3 = auc(
            y[
                valid_mask
            ],
            pred_prob_3,
        )

        # --------------------------------------------------
        # 3-model rank blend
        # --------------------------------------------------

        (
            w_xgb_rank_3,
            w_cat_rank_3,
            w_log_rank_3,
            train_auc_rank_3,
        ) = search_three_model_weights(
            y=y[
                train_mask
            ],
            xgb_pred=rank_xgb[
                train_mask
            ],
            cat_pred=rank_cat[
                train_mask
            ],
            log_pred=rank_log[
                train_mask
            ],
            step=args.three_model_step,
        )

        pred_rank_3 = (
            w_xgb_rank_3
            * rank_xgb[
                valid_mask
            ]
            +
            w_cat_rank_3
            * rank_cat[
                valid_mask
            ]
            +
            w_log_rank_3
            * rank_log[
                valid_mask
            ]
        )

        meta_oof[
            "rank_three"
        ][
            valid_mask
        ] = pred_rank_3

        held_auc_rank_3 = auc(
            y[
                valid_mask
            ],
            pred_rank_3,
        )

        xgb_fold_auc = auc(
            y[
                valid_mask
            ],
            pred_xgb[
                valid_mask
            ],
        )

        result_rows.extend(
            [
                {
                    "held_fold": held_fold,
                    "blend": "prob_xgb_cat",
                    "w_xgb": w_xgb_prob_2,
                    "w_cat": (
                        1.0
                        - w_xgb_prob_2
                    ),
                    "w_log": 0.0,
                    "train_search_auc": train_auc_prob_2,
                    "held_fold_auc": held_auc_prob_2,
                    "held_fold_xgb_auc": xgb_fold_auc,
                    "delta_vs_xgb": (
                        held_auc_prob_2
                        - xgb_fold_auc
                    ),
                },
                {
                    "held_fold": held_fold,
                    "blend": "rank_xgb_cat",
                    "w_xgb": w_xgb_rank_2,
                    "w_cat": (
                        1.0
                        - w_xgb_rank_2
                    ),
                    "w_log": 0.0,
                    "train_search_auc": train_auc_rank_2,
                    "held_fold_auc": held_auc_rank_2,
                    "held_fold_xgb_auc": xgb_fold_auc,
                    "delta_vs_xgb": (
                        held_auc_rank_2
                        - xgb_fold_auc
                    ),
                },
                {
                    "held_fold": held_fold,
                    "blend": "prob_three",
                    "w_xgb": w_xgb_prob_3,
                    "w_cat": w_cat_prob_3,
                    "w_log": w_log_prob_3,
                    "train_search_auc": train_auc_prob_3,
                    "held_fold_auc": held_auc_prob_3,
                    "held_fold_xgb_auc": xgb_fold_auc,
                    "delta_vs_xgb": (
                        held_auc_prob_3
                        - xgb_fold_auc
                    ),
                },
                {
                    "held_fold": held_fold,
                    "blend": "rank_three",
                    "w_xgb": w_xgb_rank_3,
                    "w_cat": w_cat_rank_3,
                    "w_log": w_log_rank_3,
                    "train_search_auc": train_auc_rank_3,
                    "held_fold_auc": held_auc_rank_3,
                    "held_fold_xgb_auc": xgb_fold_auc,
                    "delta_vs_xgb": (
                        held_auc_rank_3
                        - xgb_fold_auc
                    ),
                },
            ]
        )

        print(
            f"Held fold {held_fold}: "
            f"XGB={xgb_fold_auc:.6f} | "
            f"prob2={held_auc_prob_2:.6f} | "
            f"rank2={held_auc_rank_2:.6f} | "
            f"prob3={held_auc_prob_3:.6f} | "
            f"rank3={held_auc_rank_3:.6f}"
        )

    for name, values in meta_oof.items():
        if np.isnan(
            values
        ).any():
            raise RuntimeError(
                f"Meta-OOF blend {name} contains missing predictions."
            )

    results = pd.DataFrame(
        result_rows
    )

    summary_rows = []

    for name, values in meta_oof.items():
        score = auc(
            y,
            values,
        )

        sub = results[
            results[
                "blend"
            ] == name
        ]

        summary_rows.append(
            {
                "blend": name,
                "meta_oof_auc": score,
                "delta_vs_xgboost": (
                    score
                    - xgb_auc
                ),
                "folds_beating_xgboost": int(
                    (
                        sub[
                            "delta_vs_xgb"
                        ] > 0
                    ).sum()
                ),
                "mean_w_xgb": float(
                    sub[
                        "w_xgb"
                    ].mean()
                ),
                "mean_w_cat": float(
                    sub[
                        "w_cat"
                    ].mean()
                ),
                "mean_w_log": float(
                    sub[
                        "w_log"
                    ].mean()
                ),
                "std_w_xgb": float(
                    sub[
                        "w_xgb"
                    ].std(
                        ddof=1
                    )
                ),
                "std_w_cat": float(
                    sub[
                        "w_cat"
                    ].std(
                        ddof=1
                    )
                ),
                "std_w_log": float(
                    sub[
                        "w_log"
                    ].std(
                        ddof=1
                    )
                ),
            }
        )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "meta_oof_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------
    # Full-OOF search ONLY for a provisional deployment
    # weight suggestion.
    #
    # We do NOT use this score as our trustworthy blend CV
    # estimate. The meta-CV score above is the key result.
    # ------------------------------------------------------

    (
        final_prob2_xgb,
        final_prob2_search_auc,
    ) = search_two_model_weight(
        y=y,
        xgb_pred=pred_xgb,
        cat_pred=pred_cat,
        step=args.two_model_step,
    )

    (
        final_rank2_xgb,
        final_rank2_search_auc,
    ) = search_two_model_weight(
        y=y,
        xgb_pred=rank_xgb,
        cat_pred=rank_cat,
        step=args.two_model_step,
    )

    (
        final_prob3_xgb,
        final_prob3_cat,
        final_prob3_log,
        final_prob3_search_auc,
    ) = search_three_model_weights(
        y=y,
        xgb_pred=pred_xgb,
        cat_pred=pred_cat,
        log_pred=pred_log,
        step=args.three_model_step,
    )

    (
        final_rank3_xgb,
        final_rank3_cat,
        final_rank3_log,
        final_rank3_search_auc,
    ) = search_three_model_weights(
        y=y,
        xgb_pred=rank_xgb,
        cat_pred=rank_cat,
        log_pred=rank_log,
        step=args.three_model_step,
    )

    final_weight_rows = [
        {
            "blend": "prob_xgb_cat",
            "w_xgb": final_prob2_xgb,
            "w_cat": (
                1.0
                - final_prob2_xgb
            ),
            "w_log": 0.0,
            "full_oof_search_auc": final_prob2_search_auc,
        },
        {
            "blend": "rank_xgb_cat",
            "w_xgb": final_rank2_xgb,
            "w_cat": (
                1.0
                - final_rank2_xgb
            ),
            "w_log": 0.0,
            "full_oof_search_auc": final_rank2_search_auc,
        },
        {
            "blend": "prob_three",
            "w_xgb": final_prob3_xgb,
            "w_cat": final_prob3_cat,
            "w_log": final_prob3_log,
            "full_oof_search_auc": final_prob3_search_auc,
        },
        {
            "blend": "rank_three",
            "w_xgb": final_rank3_xgb,
            "w_cat": final_rank3_cat,
            "w_log": final_rank3_log,
            "full_oof_search_auc": final_rank3_search_auc,
        },
    ]

    final_weights = pd.DataFrame(
        final_weight_rows
    )

    # ------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------

    results.to_csv(
        args.output_dir
        / "meta_fold_results.csv",
        index=False,
    )

    summary_df.to_csv(
        args.output_dir
        / "blend_summary.csv",
        index=False,
    )

    final_weights.to_csv(
        args.output_dir
        / "provisional_full_oof_weights.csv",
        index=False,
    )

    meta_output = pd.DataFrame(
        {
            "row_index": xgboost[
                "row_index"
            ].to_numpy(),

            "fold": folds,

            "target_encoded": y,

            "xgboost_oof": pred_xgb.astype(
                np.float32
            ),

            "catboost_oof": pred_cat.astype(
                np.float32
            ),

            "logistic_oof": pred_log.astype(
                np.float32
            ),

            "meta_prob_xgb_cat": meta_oof[
                "prob_xgb_cat"
            ].astype(
                np.float32
            ),

            "meta_rank_xgb_cat": meta_oof[
                "rank_xgb_cat"
            ].astype(
                np.float32
            ),

            "meta_prob_three": meta_oof[
                "prob_three"
            ].astype(
                np.float32
            ),

            "meta_rank_three": meta_oof[
                "rank_three"
            ].astype(
                np.float32
            ),
        }
    )

    meta_output.to_csv(
        args.output_dir
        / "meta_oof_predictions.csv",
        index=False,
    )

    best = summary_df.iloc[
        0
    ]

    decision = (
        "BLENDING_LOOKS_USEFUL"
        if (
            best[
                "delta_vs_xgboost"
            ] > 0
            and
            best[
                "folds_beating_xgboost"
            ] >= 3
        )
        else "NO_ROBUST_BLEND_GAIN_YET"
    )

    summary_lines = [
        "EXPERIMENT: LEAKAGE_AWARE_OOF_BLEND_AUDIT",
        "=" * 76,
        "",
        "BASE MODELS",
        f"XGBoost OOF AUC: {xgb_auc:.8f}",
        f"CatBoost OOF AUC: {cat_auc:.8f}",
        f"Logistic OOF AUC: {log_auc:.8f}",
        "",
        "METHOD",
        "For each held-out frozen fold, blend weights were selected using",
        "the other four folds only, then evaluated on the held-out fold.",
        "",
        "This meta-CV result is more trustworthy than searching weights on",
        "all OOF rows and reporting the score on those same rows.",
        "",
        "META-CV BLEND RESULTS",
    ]

    for row in summary_df.itertuples():
        summary_lines.append(
            f"- {row.blend}: "
            f"AUC={row.meta_oof_auc:.8f}, "
            f"delta_vs_XGB={row.delta_vs_xgboost:+.8f}, "
            f"folds_won={row.folds_beating_xgboost}/5, "
            f"mean_weights="
            f"(XGB={row.mean_w_xgb:.3f}, "
            f"CAT={row.mean_w_cat:.3f}, "
            f"LOG={row.mean_w_log:.3f})"
        )

    summary_lines.extend(
        [
            "",
            f"DECISION: {decision}",
            "",
            "PROVISIONAL FULL-OOF WEIGHTS",
            "These weights are useful later for generating a test blend,",
            "but their full-OOF search AUC is optimistic because those weights",
            "were selected and scored on the same OOF rows.",
        ]
    )

    for row in final_weights.itertuples():
        summary_lines.append(
            f"- {row.blend}: "
            f"XGB={row.w_xgb:.3f}, "
            f"CAT={row.w_cat:.3f}, "
            f"LOG={row.w_log:.3f}, "
            f"search_AUC={row.full_oof_search_auc:.8f}"
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
        "LEAKAGE-AWARE BLEND AUDIT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Best single XGBoost OOF : "
        f"{xgb_auc:.8f}"
    )

    print(
        f"Best meta-CV blend       : "
        f"{best['blend']}"
    )

    print(
        f"Best meta-CV blend AUC   : "
        f"{best['meta_oof_auc']:.8f}"
    )

    print(
        f"Delta vs XGBoost         : "
        f"{best['delta_vs_xgboost']:+.8f}"
    )

    print(
        f"Folds beating XGBoost    : "
        f"{int(best['folds_beating_xgboost'])}/5"
    )

    print(
        f"Decision                 : "
        f"{decision}"
    )

    print(
        f"Artifacts                : "
        f"{args.output_dir.resolve()}"
    )

    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. blend_summary.csv")
    print("  4. meta_fold_results.csv")


if __name__ == "__main__":
    main()
