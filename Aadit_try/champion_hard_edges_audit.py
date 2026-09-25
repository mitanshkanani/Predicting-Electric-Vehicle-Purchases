"""
champion_hard_edges_audit.py

Kaggle Playground Series S6E9
NO TRAINING | NO SUBMISSION GENERATION

Purpose
-------
Test whether publicly discussed "hard edges" can improve our current validated
champion ranking BEFORE touching the Kaggle submission.

Current champion OOF:
    0.94611253

We separate two kinds of evidence:

A) CLEANER CROSS-FITTED HARD-EDGE TESTS
   For every held fold, discover the edge using only the other four folds:

   1. High-income hard-positive edge
      threshold = maximum income observed among NEGATIVE outer-training rows.
      Held rows above that threshold are forced to the top of that held-fold ranking.

   2. Long-commute hard-negative edge
      threshold = maximum commute observed among POSITIVE outer-training rows.
      Held rows above that threshold are forced to the bottom.

   3. Combination of the two.

   These are the main tests.

B) PUBLIC-RULE DESCRIPTIVE TESTS
   Fixed public observations:
      income > 169,972
      income > 170,537
      income < 31,004 + subsidy
      no subsidy
      exact 5.0 km commute cluster

   These are useful diagnostics, BUT the thresholds were publicly discovered
   from the same competition train labels, so treat their OOF deltas as
   descriptive rather than pristine confirmatory validation.

No public leaderboard is used.
No model is retrained.
No blend weights are searched.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"

FOLDS_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "candidate_folds.csv"
)

CHAMPION_OOF_PATH = (
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
    / "champion_hard_edges_audit"
)

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_CHAMPION_AUC = 0.94611253
AUC_TOL = 5e-6

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

PUBLIC_HIGH_169972 = 169_972.0
PUBLIC_HIGH_170537 = 170_537.0
PUBLIC_POOR_THRESHOLD = 31_004.0
PUBLIC_COMMUTE_CLUSTER = 5.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def prediction_column(df: pd.DataFrame) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
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

    candidates = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not identify champion prediction column: {list(df.columns)}"
        )

    return candidates[0]


def load_champion_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    if not CHAMPION_OOF_PATH.exists():
        raise FileNotFoundError(CHAMPION_OOF_PATH)

    df = pd.read_csv(CHAMPION_OOF_PATH)

    if len(df) != len(train):
        raise RuntimeError("Champion OOF row-count mismatch.")

    if "row_index" in df.columns:
        expected = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError("Champion OOF row_index mismatch.")

    if "fold" in df.columns:
        if not np.array_equal(
            df["fold"].to_numpy(dtype=int),
            folds,
        ):
            raise RuntimeError("Champion OOF fold mismatch.")

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise RuntimeError("Champion OOF id mismatch.")

    pred = pd.to_numeric(
        df[prediction_column(df)],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError("Champion OOF contains non-finite values.")

    score = auc(
        (
            train[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(POSITIVE_LABEL.lower())
            .astype(np.int8)
            .to_numpy()
        ),
        pred,
    )

    if abs(score - EXPECTED_CHAMPION_AUC) > AUC_TOL:
        raise RuntimeError(
            "Champion OOF AUC mismatch.\n"
            f"Expected: {EXPECTED_CHAMPION_AUC:.8f}\n"
            f"Loaded:   {score:.8f}"
        )

    return pred


def pct_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def lexicographic_group_rank(
    scores: np.ndarray,
    group_level: np.ndarray,
) -> np.ndarray:
    """
    group_level:
        lower values rank lower,
        higher values rank higher.

    Within the same group level, preserve champion-score ordering.
    """
    scores = np.asarray(scores, dtype=np.float64)
    group_level = np.asarray(group_level)

    order = np.lexsort(
        (
            scores,
            group_level,
        )
    )

    ranked = np.empty(
        len(scores),
        dtype=np.float64,
    )

    ranked[order] = (
        np.arange(
            1,
            len(scores) + 1,
            dtype=np.float64,
        )
        / len(scores)
    )

    return ranked


def force_group_top(
    scores: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    levels = mask.astype(np.int8)
    return lexicographic_group_rank(
        scores,
        levels,
    )


def force_group_bottom(
    scores: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    # masked rows -> level 0 -> bottom
    # unmasked rows -> level 1
    levels = (~mask).astype(np.int8)
    return lexicographic_group_rank(
        scores,
        levels,
    )


def hard_positive_and_negative_rank(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> np.ndarray:
    """
    0 = hard negative
    1 = ordinary
    2 = hard positive
    """
    levels = np.ones(
        len(scores),
        dtype=np.int8,
    )

    levels[negative_mask] = 0
    levels[positive_mask] = 2

    return lexicographic_group_rank(
        scores,
        levels,
    )


def reorder_within_subset(
    scores: np.ndarray,
    subset_mask: np.ndarray,
    preferred_mask: np.ndarray,
) -> np.ndarray:
    """
    Reassign only the score positions already occupied by subset rows.

    Within subset:
      preferred=False receives lower subset score positions,
      preferred=True receives higher subset score positions.

    The multiset of subset score values stays unchanged.
    """
    out = np.asarray(
        scores,
        dtype=np.float64,
    ).copy()

    idx = np.flatnonzero(
        subset_mask
    )

    if len(idx) <= 1:
        return pct_rank(out)

    available_scores = np.sort(
        out[idx]
    )

    preferred = preferred_mask[
        idx
    ].astype(np.int8)

    # Within each preference group preserve original champion ordering.
    order_inside = np.lexsort(
        (
            out[idx],
            preferred,
        )
    )

    assignment_idx = idx[
        order_inside
    ]

    out[
        assignment_idx
    ] = available_scores

    return pct_rank(out)


def group_stats(
    name: str,
    mask: np.ndarray,
    y: np.ndarray,
    champion: np.ndarray,
) -> dict:
    n = int(mask.sum())

    if n == 0:
        return {
            "group": name,
            "n": 0,
            "positive_rate": np.nan,
            "champion_mean": np.nan,
            "champion_min": np.nan,
            "champion_max": np.nan,
        }

    return {
        "group": name,
        "n": n,
        "positive_rate": float(
            y[mask].mean()
        ),
        "champion_mean": float(
            champion[mask].mean()
        ),
        "champion_min": float(
            champion[mask].min()
        ),
        "champion_max": float(
            champion[mask].max()
        ),
    }


def main() -> None:
    print("=" * 104)
    print("CHAMPION HARD-EDGE AUDIT")
    print("NO TRAINING | NO SUBMISSION | NO BLEND-WEIGHT SEARCH")
    print("=" * 104)

    for path in [
        TRAIN_PATH,
        FOLDS_PATH,
        CHAMPION_OOF_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_FOLD_SHA:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(
        TRAIN_PATH
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    if len(train) != len(folds_df):
        raise RuntimeError(
            "Train/folds row-count mismatch."
        )

    folds = folds_df[
        "fold"
    ].to_numpy(dtype=int)

    if sorted(
        np.unique(folds).tolist()
    ) != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected folds: {sorted(np.unique(folds).tolist())}"
        )

    y = (
        train[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(POSITIVE_LABEL.lower())
        .astype(np.int8)
        .to_numpy()
    )

    champion = load_champion_oof(
        train,
        folds,
    )

    income = pd.to_numeric(
        train[
            "Annual_Income_USD"
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    commute = pd.to_numeric(
        train[
            "Daily_Commute_km"
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    subsidy = (
        train[
            "Subsidy_Available"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    if not set(
        np.unique(subsidy)
    ).issubset(
        {"yes", "no"}
    ):
        raise RuntimeError(
            f"Unexpected subsidy labels: {sorted(set(np.unique(subsidy)))}"
        )

    print(
        f"\nFrozen fold SHA256 verified: {fold_hash}"
    )
    print(
        f"Champion OOF verified       : {EXPECTED_CHAMPION_AUC:.8f}"
    )

    # ------------------------------------------------------------------
    # Public-rule descriptive statistics.
    # ------------------------------------------------------------------

    public_masks = {
        "income_gt_169972": (
            income > PUBLIC_HIGH_169972
        ),
        "income_gt_170537": (
            income > PUBLIC_HIGH_170537
        ),
        "income_lt_31004": (
            income < PUBLIC_POOR_THRESHOLD
        ),
        "poor_plus_subsidy": (
            (income < PUBLIC_POOR_THRESHOLD)
            & (subsidy == "yes")
        ),
        "poor_no_subsidy": (
            (income < PUBLIC_POOR_THRESHOLD)
            & (subsidy == "no")
        ),
        "subsidy_yes": (
            subsidy == "yes"
        ),
        "subsidy_no": (
            subsidy == "no"
        ),
        "commute_exact_5km": (
            np.isclose(
                commute,
                PUBLIC_COMMUTE_CLUSTER,
                rtol=0.0,
                atol=1e-12,
            )
        ),
    }

    public_stats = pd.DataFrame(
        [
            group_stats(
                name,
                mask,
                y,
                champion,
            )
            for name, mask
            in public_masks.items()
        ]
    )

    print()
    print("--- Public-rule group statistics in OUR train ---")
    print(
        public_stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    subsidy_yes_rate = float(
        y[
            subsidy == "yes"
        ].mean()
    )

    subsidy_no_rate = float(
        y[
            subsidy == "no"
        ].mean()
    )

    multiplier = (
        subsidy_yes_rate
        / subsidy_no_rate
        if subsidy_no_rate > 0
        else np.inf
    )

    poor_mask = (
        income
        < PUBLIC_POOR_THRESHOLD
    )

    poor_buy_mask = (
        poor_mask
        & (y == 1)
    )

    poor_sub_mask = (
        poor_mask
        & (subsidy == "yes")
    )

    poor_no_sub_mask = (
        poor_mask
        & (subsidy == "no")
    )

    p_buy_given_poor = float(
        y[
            poor_mask
        ].mean()
    )

    p_sub_given_buy_poor = float(
        (
            subsidy[
                poor_buy_mask
            ] == "yes"
        ).mean()
    ) if poor_buy_mask.any() else np.nan

    p_buy_given_poor_sub = float(
        y[
            poor_sub_mask
        ].mean()
    ) if poor_sub_mask.any() else np.nan

    p_buy_given_poor_no_sub = float(
        y[
            poor_no_sub_mask
        ].mean()
    ) if poor_no_sub_mask.any() else np.nan

    print()
    print("--- Conditional-probability sanity check ---")
    print(
        f"P(Buy | Subsidy=Yes)            = {subsidy_yes_rate:.6f}"
    )
    print(
        f"P(Buy | Subsidy=No)             = {subsidy_no_rate:.6f}"
    )
    print(
        f"Observed subsidy buy-rate ratio = {multiplier:.3f}x"
    )
    print()
    print(
        f"P(Buy | income<31,004)                 = {p_buy_given_poor:.6f}"
    )
    print(
        f"P(Subsidy=Yes | Buy, income<31,004)    = {p_sub_given_buy_poor:.6f}"
    )
    print(
        f"P(Buy | income<31,004, Subsidy=Yes)    = {p_buy_given_poor_sub:.6f}"
    )
    print(
        f"P(Buy | income<31,004, Subsidy=No)     = {p_buy_given_poor_no_sub:.6f}"
    )

    # ------------------------------------------------------------------
    # Cross-fitted hard-edge discovery.
    # ------------------------------------------------------------------

    baseline_meta = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    high_meta = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    long_meta = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    combined_meta = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    crossfit_rows = []

    print()
    print("--- CROSS-FITTED hard-edge test (main evidence) ---")

    for fold in range(5):
        tr = (
            folds != fold
        )

        va = (
            folds == fold
        )

        negative_train_income = income[
            tr & (y == 0)
        ]

        positive_train_commute = commute[
            tr & (y == 1)
        ]

        if len(
            negative_train_income
        ) == 0:
            raise RuntimeError(
                f"No negative outer-training rows for fold {fold}"
            )

        if len(
            positive_train_commute
        ) == 0:
            raise RuntimeError(
                f"No positive outer-training rows for fold {fold}"
            )

        # Hard-positive boundary learned ONLY from outer-training labels:
        # no observed negative row lies above this.
        high_threshold = float(
            negative_train_income.max()
        )

        # Hard-negative long-commute boundary learned ONLY from outer-training:
        # no observed positive row lies above this.
        long_threshold = float(
            positive_train_commute.max()
        )

        val_scores = champion[
            va
        ]

        val_y = y[
            va
        ]

        val_income = income[
            va
        ]

        val_commute = commute[
            va
        ]

        high_mask = (
            val_income
            > high_threshold
        )

        long_mask = (
            val_commute
            > long_threshold
        )

        base_rank = pct_rank(
            val_scores
        )

        high_rank = force_group_top(
            val_scores,
            high_mask,
        )

        long_rank = force_group_bottom(
            val_scores,
            long_mask,
        )

        combined_rank = (
            hard_positive_and_negative_rank(
                val_scores,
                positive_mask=high_mask,
                negative_mask=long_mask,
            )
        )

        baseline_meta[
            va
        ] = base_rank

        high_meta[
            va
        ] = high_rank

        long_meta[
            va
        ] = long_rank

        combined_meta[
            va
        ] = combined_rank

        base_auc = auc(
            val_y,
            base_rank,
        )

        high_auc = auc(
            val_y,
            high_rank,
        )

        long_auc = auc(
            val_y,
            long_rank,
        )

        combined_auc = auc(
            val_y,
            combined_rank,
        )

        high_n = int(
            high_mask.sum()
        )

        high_pos = int(
            val_y[
                high_mask
            ].sum()
        ) if high_n else 0

        long_n = int(
            long_mask.sum()
        )

        long_pos = int(
            val_y[
                long_mask
            ].sum()
        ) if long_n else 0

        crossfit_rows.append(
            {
                "fold": fold,
                "high_income_threshold_from_outer_train": (
                    high_threshold
                ),
                "high_edge_validation_n": high_n,
                "high_edge_validation_positives": high_pos,
                "high_edge_validation_positive_rate": (
                    high_pos / high_n
                    if high_n
                    else np.nan
                ),
                "long_commute_threshold_from_outer_train": (
                    long_threshold
                ),
                "long_edge_validation_n": long_n,
                "long_edge_validation_positives": long_pos,
                "long_edge_validation_positive_rate": (
                    long_pos / long_n
                    if long_n
                    else np.nan
                ),
                "baseline_auc": base_auc,
                "high_edge_auc": high_auc,
                "high_edge_delta": (
                    high_auc
                    - base_auc
                ),
                "long_edge_auc": long_auc,
                "long_edge_delta": (
                    long_auc
                    - base_auc
                ),
                "combined_auc": combined_auc,
                "combined_delta": (
                    combined_auc
                    - base_auc
                ),
            }
        )

        print(
            f"Fold {fold}: "
            f"high_thr={high_threshold:.3f} "
            f"(val n={high_n}, pos={high_pos}) | "
            f"long_thr={long_threshold:.3f} "
            f"(val n={long_n}, pos={long_pos})"
        )
        print(
            f"         baseline={base_auc:.8f} | "
            f"high={high_auc:.8f} "
            f"({high_auc-base_auc:+.8f}) | "
            f"long={long_auc:.8f} "
            f"({long_auc-base_auc:+.8f}) | "
            f"combined={combined_auc:.8f} "
            f"({combined_auc-base_auc:+.8f})"
        )

    crossfit_df = pd.DataFrame(
        crossfit_rows
    )

    crossfit_results = {
        "baseline": auc(
            y,
            baseline_meta,
        ),
        "high_edge": auc(
            y,
            high_meta,
        ),
        "long_edge": auc(
            y,
            long_meta,
        ),
        "combined": auc(
            y,
            combined_meta,
        ),
    }

    # ------------------------------------------------------------------
    # Public fixed-rule descriptive ranking tests.
    # ------------------------------------------------------------------

    public_test_specs = {
        "public_income_gt_169972_force_top": (
            "top",
            public_masks[
                "income_gt_169972"
            ],
        ),
        "public_income_gt_170537_force_top": (
            "top",
            public_masks[
                "income_gt_170537"
            ],
        ),
        "public_no_subsidy_force_bottom": (
            "bottom",
            public_masks[
                "subsidy_no"
            ],
        ),
    }

    public_rule_rows = []

    print()
    print(
        "--- PUBLIC fixed-rule ranking tests "
        "(DESCRIPTIVE; thresholds came from public same-train EDA) ---"
    )

    for name, (
        mode,
        full_mask,
    ) in public_test_specs.items():
        meta = np.full(
            len(train),
            np.nan,
            dtype=np.float64,
        )

        fold_deltas = []

        for fold in range(5):
            va = (
                folds == fold
            )

            val_scores = champion[
                va
            ]

            val_mask = full_mask[
                va
            ]

            base_rank = pct_rank(
                val_scores
            )

            if mode == "top":
                candidate_rank = (
                    force_group_top(
                        val_scores,
                        val_mask,
                    )
                )
            elif mode == "bottom":
                candidate_rank = (
                    force_group_bottom(
                        val_scores,
                        val_mask,
                    )
                )
            else:
                raise RuntimeError(
                    f"Unknown mode: {mode}"
                )

            meta[
                va
            ] = candidate_rank

            fold_deltas.append(
                auc(
                    y[
                        va
                    ],
                    candidate_rank,
                )
                - auc(
                    y[
                        va
                    ],
                    base_rank,
                )
            )

        score = auc(
            y,
            meta,
        )

        delta = (
            score
            - crossfit_results[
                "baseline"
            ]
        )

        row = {
            "rule": name,
            "meta_auc": score,
            "delta_vs_baseline": delta,
            "folds_improved": int(
                np.sum(
                    np.asarray(
                        fold_deltas
                    ) > 0
                )
            ),
            "folds_worse": int(
                np.sum(
                    np.asarray(
                        fold_deltas
                    ) < 0
                )
            ),
            "fold0_delta": fold_deltas[0],
            "fold1_delta": fold_deltas[1],
            "fold2_delta": fold_deltas[2],
            "fold3_delta": fold_deltas[3],
            "fold4_delta": fold_deltas[4],
        }

        public_rule_rows.append(
            row
        )

        print(
            f"{name}: "
            f"{crossfit_results['baseline']:.8f} -> "
            f"{score:.8f} "
            f"({delta:+.8f}) | "
            f"folds +={row['folds_improved']}/5 "
            f"-={row['folds_worse']}/5"
        )

    # Poor+subsidy: only reorder rows already inside low-income subset.
    poor_meta = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    poor_fold_deltas = []

    for fold in range(5):
        va = (
            folds == fold
        )

        val_scores = champion[
            va
        ]

        val_poor = (
            income[
                va
            ]
            < PUBLIC_POOR_THRESHOLD
        )

        val_preferred = (
            val_poor
            & (
                subsidy[
                    va
                ] == "yes"
            )
        )

        base_rank = pct_rank(
            val_scores
        )

        candidate_rank = (
            reorder_within_subset(
                val_scores,
                subset_mask=val_poor,
                preferred_mask=val_preferred,
            )
        )

        poor_meta[
            va
        ] = candidate_rank

        poor_fold_deltas.append(
            auc(
                y[
                    va
                ],
                candidate_rank,
            )
            - auc(
                y[
                    va
                ],
                base_rank,
            )
        )

    poor_auc = auc(
        y,
        poor_meta,
    )

    poor_delta = (
        poor_auc
        - crossfit_results[
            "baseline"
        ]
    )

    poor_row = {
        "rule": (
            "public_poor_subsidy_rank_above_poor_no_subsidy"
        ),
        "meta_auc": poor_auc,
        "delta_vs_baseline": poor_delta,
        "folds_improved": int(
            np.sum(
                np.asarray(
                    poor_fold_deltas
                ) > 0
            )
        ),
        "folds_worse": int(
            np.sum(
                np.asarray(
                    poor_fold_deltas
                ) < 0
            )
        ),
        "fold0_delta": poor_fold_deltas[0],
        "fold1_delta": poor_fold_deltas[1],
        "fold2_delta": poor_fold_deltas[2],
        "fold3_delta": poor_fold_deltas[3],
        "fold4_delta": poor_fold_deltas[4],
    }

    public_rule_rows.append(
        poor_row
    )

    print(
        "public_poor_subsidy_rank_above_poor_no_subsidy: "
        f"{crossfit_results['baseline']:.8f} -> "
        f"{poor_auc:.8f} "
        f"({poor_delta:+.8f}) | "
        f"folds +={poor_row['folds_improved']}/5 "
        f"-={poor_row['folds_worse']}/5"
    )

    public_rule_df = pd.DataFrame(
        public_rule_rows
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------

    baseline_meta_auc = (
        crossfit_results[
            "baseline"
        ]
    )

    high_delta = (
        crossfit_results[
            "high_edge"
        ]
        - baseline_meta_auc
    )

    long_delta = (
        crossfit_results[
            "long_edge"
        ]
        - baseline_meta_auc
    )

    combined_delta = (
        crossfit_results[
            "combined"
        ]
        - baseline_meta_auc
    )

    high_folds_improved = int(
        (
            crossfit_df[
                "high_edge_delta"
            ] > 0
        ).sum()
    )

    long_folds_improved = int(
        (
            crossfit_df[
                "long_edge_delta"
            ] > 0
        ).sum()
    )

    combined_folds_improved = int(
        (
            crossfit_df[
                "combined_delta"
            ] > 0
        ).sum()
    )

    print()
    print("=" * 104)
    print("MAIN CROSS-FITTED RESULT")
    print("=" * 104)
    print(
        f"Champion fold-rank baseline : "
        f"{baseline_meta_auc:.8f}"
    )
    print(
        f"High-income hard edge       : "
        f"{crossfit_results['high_edge']:.8f} "
        f"({high_delta:+.8f}) | "
        f"folds improved={high_folds_improved}/5"
    )
    print(
        f"Long-commute hard edge      : "
        f"{crossfit_results['long_edge']:.8f} "
        f"({long_delta:+.8f}) | "
        f"folds improved={long_folds_improved}/5"
    )
    print(
        f"Combined hard edges         : "
        f"{crossfit_results['combined']:.8f} "
        f"({combined_delta:+.8f}) | "
        f"folds improved={combined_folds_improved}/5"
    )

    if (
        combined_delta >= 2e-5
        and combined_folds_improved >= 4
    ):
        primitive = (
            "PROMISING_CROSSFIT_HARD_EDGE_SIGNAL"
        )
    elif (
        max(
            high_delta,
            long_delta,
            combined_delta,
        ) <= 5e-6
    ):
        primitive = (
            "NO_MEANINGFUL_CROSSFIT_HARD_EDGE_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_MIXED_CROSSFIT_HARD_EDGE_SIGNAL"
        )

    print(
        f"Primitive                    : "
        f"{primitive}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_stats.to_csv(
        OUTPUT_DIR
        / "public_rule_group_stats.csv",
        index=False,
    )

    crossfit_df.to_csv(
        OUTPUT_DIR
        / "crossfit_hard_edge_folds.csv",
        index=False,
    )

    public_rule_df.to_csv(
        OUTPUT_DIR
        / "public_rule_descriptive_tests.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": folds,
            "target_encoded": y,
            "champion_fold_rank": (
                baseline_meta
            ),
            "crossfit_high_edge_rank": (
                high_meta
            ),
            "crossfit_long_edge_rank": (
                long_meta
            ),
            "crossfit_combined_rank": (
                combined_meta
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_rank_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: CHAMPION HARD-EDGE AUDIT",
        "=" * 88,
        "",
        "NO TRAINING",
        "NO SUBMISSION GENERATION",
        "NO BLEND-WEIGHT SEARCH",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"Raw champion OOF verified: {EXPECTED_CHAMPION_AUC:.8f}",
        "",
        "MAIN CROSS-FITTED TESTS",
        (
            "High-income threshold per fold = "
            "max income among NEGATIVE outer-training rows."
        ),
        (
            "Long-commute threshold per fold = "
            "max commute among POSITIVE outer-training rows."
        ),
        "",
        f"Champion fold-rank baseline: {baseline_meta_auc:.8f}",
        (
            f"High-income hard edge: {crossfit_results['high_edge']:.8f} "
            f"({high_delta:+.8f}), folds improved {high_folds_improved}/5"
        ),
        (
            f"Long-commute hard edge: {crossfit_results['long_edge']:.8f} "
            f"({long_delta:+.8f}), folds improved {long_folds_improved}/5"
        ),
        (
            f"Combined hard edges: {crossfit_results['combined']:.8f} "
            f"({combined_delta:+.8f}), folds improved {combined_folds_improved}/5"
        ),
        f"Primitive: {primitive}",
        "",
        "PUBLIC OBSERVATION SANITY CHECKS",
        f"P(Buy | Subsidy=Yes): {subsidy_yes_rate:.8f}",
        f"P(Buy | Subsidy=No): {subsidy_no_rate:.8f}",
        f"Observed buy-rate ratio: {multiplier:.4f}x",
        f"P(Buy | income<31,004): {p_buy_given_poor:.8f}",
        (
            "P(Subsidy=Yes | Buy, income<31,004): "
            f"{p_sub_given_buy_poor:.8f}"
        ),
        (
            "P(Buy | income<31,004, Subsidy=Yes): "
            f"{p_buy_given_poor_sub:.8f}"
        ),
        (
            "P(Buy | income<31,004, Subsidy=No): "
            f"{p_buy_given_poor_no_sub:.8f}"
        ),
        "",
        "IMPORTANT",
        (
            "Public fixed-threshold tests are descriptive because those rules "
            "were discovered from the same competition training labels."
        ),
        (
            "Do not modify the Kaggle submission unless the cross-fitted "
            "hard-edge tests show a meaningful, fold-stable gain."
        ),
    ]

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Artifacts: "
        f"{OUTPUT_DIR.relative_to(ROOT)}"
    )
    print("=" * 104)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. crossfit_hard_edge_folds.csv")
    print("  4. public_rule_descriptive_tests.csv")
    print("  5. public_rule_group_stats.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 104)
        print("AUDIT FAILED")
        print("=" * 104)
        print(str(exc))
        sys.exit(1)
