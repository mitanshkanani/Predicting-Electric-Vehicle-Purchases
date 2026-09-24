from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =============================================================================
# EXACT LEXICOGRAPHIC TIE-BREAK AUDIT
#
# NO TRAINING.
# NO BLEND-WEIGHT TUNING.
# NO QUANTIZATION / NEAR-TIE THRESHOLD.
#
# Primary score:
#   current validated champion OOF = 0.94611253
#
# Question:
#   Does the current champion leave enough EXACT prediction ties for an
#   already-validated secondary model to improve AUC by ordering only rows
#   that the champion considers exactly tied?
#
# Selection:
#   Leave-one-fold-out over a SMALL FIXED set of existing models.
#   "NONE" is included, so the meta-procedure can choose not to break ties.
#
# IMPORTANT:
#   Non-tied primary ordering is NEVER changed.
# =============================================================================


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

EXPECTED = {
    "champion": 0.94611253,
    "fine_xgb": 0.94606664,
    "old_cat": 0.94558572,
    "fine_cat": 0.94578633,
    "lgbm": 0.94578042,
    "realmlp": 0.94329901,
}

AUC_TOL = 5e-6

OOF_PATHS = {
    "champion": (
        ROOT
        / "artifacts"
        / "experiments"
        / "blend_fine_income_xgb_validated_submission"
        / "oof_predictions.csv"
    ),
    "fine_xgb": (
        ROOT
        / "artifacts"
        / "experiments"
        / "xgboost_fine_income_te_gpu"
        / "oof_predictions.csv"
    ),
    "old_cat": (
        ROOT
        / "artifacts"
        / "experiments"
        / "catboost_hierarchical_income_commute_multiseed_gpu"
        / "best_average_oof_predictions.csv"
    ),
    "fine_cat": (
        ROOT
        / "artifacts"
        / "experiments"
        / "catboost_fine_income_multiseed_gpu"
        / "best_average_oof_predictions.csv"
    ),
    "lgbm": (
        ROOT
        / "artifacts"
        / "experiments"
        / "lightgbm_engineered_learned_margin_cpu"
        / "oof_predictions.csv"
    ),
    "realmlp": (
        ROOT
        / "artifacts"
        / "experiments"
        / "realmlp_frozen5_vectorized_gpu"
        / "oof_predictions.csv"
    ),
}

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lexsort_exact_tie_break_audit"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def prediction_column(df: pd.DataFrame, path: Path) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "meta_oof_prediction",
        "prediction",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    excluded = {
        "id",
        "row_index",
        "fold",
        "target",
        "target_encoded",
    }

    candidates = [
        c
        for c in df.columns
        if (
            c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify OOF prediction column in {path}.\n"
            f"Candidates={candidates}\n"
            f"Columns={list(df.columns)}"
        )

    return candidates[0]


def load_oof(
    path: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise RuntimeError(
            f"Row count mismatch for {path}: "
            f"{len(df):,} vs {len(train):,}"
        )

    if "row_index" in df.columns:
        expected = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError(f"row_index mismatch in {path}")

    if "fold" in df.columns:
        if not np.array_equal(
            df["fold"].to_numpy(dtype=int),
            folds,
        ):
            raise RuntimeError(f"fold mismatch in {path}")

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise RuntimeError(f"id mismatch in {path}")

    col = prediction_column(df, path)
    pred = pd.to_numeric(
        df[col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite OOF values in {path}")

    return pred


def average_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def lexicographic_average_rank(
    primary: np.ndarray,
    secondary: np.ndarray,
) -> np.ndarray:
    """
    Rank ascending by primary, then secondary.

    Rows with exactly equal (primary, secondary) pairs remain tied.
    Rows with different primary values can NEVER swap order.
    """
    n = len(primary)

    order = np.lexsort(
        (
            secondary,
            primary,
        )
    )

    p = primary[order]
    s = secondary[order]

    new_group = np.ones(
        n,
        dtype=bool,
    )

    if n > 1:
        new_group[1:] = (
            (p[1:] != p[:-1])
            | (s[1:] != s[:-1])
        )

    starts = np.flatnonzero(new_group)
    ends = np.r_[
        starts[1:],
        n,
    ]

    sorted_scores = np.empty(
        n,
        dtype=np.float64,
    )

    # Match pandas pct-rank convention:
    # ranks are 1..n and equal values receive the average position.
    for start, end in zip(starts, ends):
        avg_position = (
            (start + 1)
            + end
        ) / 2.0

        sorted_scores[
            start:end
        ] = avg_position / n

    out = np.empty(
        n,
        dtype=np.float64,
    )
    out[order] = sorted_scores

    return out


def foldwise_primary_rank(
    primary: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    out = np.empty(
        len(primary),
        dtype=np.float64,
    )

    for fold in range(5):
        mask = folds == fold
        out[mask] = average_rank(
            primary[mask]
        )

    return out


def foldwise_lex_rank(
    primary: np.ndarray,
    secondary: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    out = np.empty(
        len(primary),
        dtype=np.float64,
    )

    for fold in range(5):
        mask = folds == fold

        out[mask] = lexicographic_average_rank(
            primary[mask],
            secondary[mask],
        )

    return out


def tie_diagnostics(
    primary: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "score": primary,
            "y": y,
        }
    )

    grouped = (
        frame
        .groupby(
            "score",
            sort=False,
            observed=True,
        )["y"]
        .agg(
            n="size",
            positives="sum",
        )
        .reset_index(drop=True)
    )

    grouped["negatives"] = (
        grouped["n"]
        - grouped["positives"]
    )

    tie_groups = grouped[
        grouped["n"] > 1
    ]

    rows_in_ties = int(
        tie_groups["n"].sum()
    )

    pos_neg_tied_pairs = int(
        (
            grouped["positives"]
            * grouped["negatives"]
        ).sum()
    )

    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)

    total_pos_neg_pairs = (
        total_pos
        * total_neg
    )

    # Baseline AUC awards 0.5 for tied positive-negative pairs.
    # Perfect tie-breaking could at most convert all such pairs to wins.
    theoretical_max_auc_gain = (
        0.5
        * pos_neg_tied_pairs
        / total_pos_neg_pairs
        if total_pos_neg_pairs > 0
        else 0.0
    )

    return {
        "rows": len(primary),
        "unique_primary_scores": int(
            grouped.shape[0]
        ),
        "tie_groups": int(
            len(tie_groups)
        ),
        "rows_in_tie_groups": rows_in_ties,
        "fraction_rows_in_ties": (
            rows_in_ties / len(primary)
        ),
        "max_tie_group_size": (
            int(tie_groups["n"].max())
            if len(tie_groups)
            else 1
        ),
        "positive_negative_tied_pairs": (
            pos_neg_tied_pairs
        ),
        "total_positive_negative_pairs": (
            total_pos_neg_pairs
        ),
        "theoretical_perfect_tie_break_auc_gain": (
            theoretical_max_auc_gain
        ),
    }


def main() -> None:
    print("=" * 100)
    print("EXACT LEXICOGRAPHIC TIE-BREAK AUDIT")
    print("NO TRAINING | NO WEIGHT SEARCH | NO NEAR-TIE THRESHOLD")
    print("=" * 100)

    for path in [
        TRAIN_PATH,
        FOLDS_PATH,
        *OOF_PATHS.values(),
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
            "Train/fold row mismatch."
        )

    if "fold" not in folds_df.columns:
        raise RuntimeError(
            "Frozen fold file has no 'fold' column."
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

    preds: dict[str, np.ndarray] = {}

    print(
        f"\nFrozen fold SHA256 verified: {fold_hash}"
    )
    print("\n--- OOF integrity ---")

    for name, path in OOF_PATHS.items():
        pred = load_oof(
            path,
            train,
            folds,
        )

        score = auc(
            y,
            pred,
        )

        if abs(
            score
            - EXPECTED[name]
        ) > AUC_TOL:
            raise RuntimeError(
                f"{name} AUC mismatch: "
                f"{score:.8f} vs expected {EXPECTED[name]:.8f}"
            )

        preds[name] = pred

        print(
            f"{name:9s}: "
            f"{score:.8f}"
        )

    primary = preds[
        "champion"
    ]

    raw_ties = tie_diagnostics(
        primary,
        y,
    )

    print("\n--- Exact tie burden in current champion ---")
    print(
        f"Unique champion scores       : "
        f"{raw_ties['unique_primary_scores']:,} / {len(primary):,}"
    )
    print(
        f"Tie groups                   : "
        f"{raw_ties['tie_groups']:,}"
    )
    print(
        f"Rows in tie groups           : "
        f"{raw_ties['rows_in_tie_groups']:,} "
        f"({100.0 * raw_ties['fraction_rows_in_ties']:.4f}%)"
    )
    print(
        f"Largest exact tie group      : "
        f"{raw_ties['max_tie_group_size']:,}"
    )
    print(
        f"Positive-negative tied pairs : "
        f"{raw_ties['positive_negative_tied_pairs']:,}"
    )
    print(
        f"PERFECT tie-break upper bound: "
        f"+{raw_ties['theoretical_perfect_tie_break_auc_gain']:.8f} AUC"
    )

    baseline_meta = foldwise_primary_rank(
        primary,
        folds,
    )

    baseline_meta_auc = auc(
        y,
        baseline_meta,
    )

    candidate_names = [
        "fine_xgb",
        "old_cat",
        "fine_cat",
        "lgbm",
        "realmlp",
    ]

    lex_scores = {
        name: foldwise_lex_rank(
            primary,
            preds[name],
            folds,
        )
        for name in candidate_names
    }

    descriptive_rows = []

    print("\n--- Fixed secondary descriptive results ---")

    for name in candidate_names:
        score = auc(
            y,
            lex_scores[name],
        )

        delta = (
            score
            - baseline_meta_auc
        )

        descriptive_rows.append(
            {
                "secondary": name,
                "auc": score,
                "delta_vs_primary": delta,
            }
        )

        print(
            f"{name:9s}: "
            f"{baseline_meta_auc:.8f} -> "
            f"{score:.8f} "
            f"({delta:+.8f})"
        )

    # Leave-one-fold-out secondary selection.
    # NONE is allowed and corresponds to the untouched champion.
    meta_prediction = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []

    print(
        "\n--- Leave-one-fold-out secondary selection ---"
    )

    for held_fold in range(5):
        fit_mask = (
            folds != held_fold
        )

        held_mask = (
            folds == held_fold
        )

        choices = {
            "NONE": baseline_meta,
            **lex_scores,
        }

        fit_scores = {
            name: auc(
                y[fit_mask],
                score_vector[fit_mask],
            )
            for name, score_vector
            in choices.items()
        }

        # Deterministic tie-break:
        # prefer NONE if equal, then the order in candidate_names.
        ordered_names = [
            "NONE",
            *candidate_names,
        ]

        selected = max(
            ordered_names,
            key=lambda name: (
                fit_scores[name],
                -ordered_names.index(name),
            ),
        )

        selected_scores = choices[
            selected
        ]

        meta_prediction[
            held_mask
        ] = selected_scores[
            held_mask
        ]

        held_base_auc = auc(
            y[held_mask],
            baseline_meta[
                held_mask
            ],
        )

        held_candidate_auc = auc(
            y[held_mask],
            selected_scores[
                held_mask
            ],
        )

        delta = (
            held_candidate_auc
            - held_base_auc
        )

        fold_rows.append(
            {
                "held_fold": held_fold,
                "selected_secondary": selected,
                "fit_auc_selected": (
                    fit_scores[
                        selected
                    ]
                ),
                "fit_auc_none": (
                    fit_scores[
                        "NONE"
                    ]
                ),
                "held_primary_auc": (
                    held_base_auc
                ),
                "held_lexsort_auc": (
                    held_candidate_auc
                ),
                "delta": delta,
            }
        )

        print(
            f"Fold {held_fold}: "
            f"secondary={selected:9s} | "
            f"{held_base_auc:.8f} -> "
            f"{held_candidate_auc:.8f} "
            f"({delta:+.8f})"
        )

    if np.isnan(
        meta_prediction
    ).any():
        raise RuntimeError(
            "Meta OOF contains NaNs."
        )

    fold_results = pd.DataFrame(
        fold_rows
    )

    candidate_meta_auc = auc(
        y,
        meta_prediction,
    )

    meta_delta = (
        candidate_meta_auc
        - baseline_meta_auc
    )

    improved = int(
        (
            fold_results[
                "delta"
            ] > 0
        ).sum()
    )

    worse = int(
        (
            fold_results[
                "delta"
            ] < 0
        ).sum()
    )

    unchanged = int(
        (
            fold_results[
                "delta"
            ] == 0
        ).sum()
    )

    selected_counts = (
        fold_results[
            "selected_secondary"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        meta_delta > 0
        and improved >= 4
    ):
        primitive = (
            "POSITIVE_EXACT_TIE_BREAK_SIGNAL"
        )
    elif meta_delta < 0:
        primitive = (
            "NEGATIVE_EXACT_TIE_BREAK_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_EXACT_TIE_BREAK_SIGNAL"
        )

    print("\n" + "=" * 100)
    print("FINAL EXACT TIE-BREAK RESULT")
    print("=" * 100)
    print(
        f"Primary meta AUC       : "
        f"{baseline_meta_auc:.8f}"
    )
    print(
        f"Lexsort meta-CV AUC    : "
        f"{candidate_meta_auc:.8f}"
    )
    print(
        f"Delta                  : "
        f"{meta_delta:+.8f}"
    )
    print(
        f"Folds improved         : "
        f"{improved}/5"
    )
    print(
        f"Folds worse            : "
        f"{worse}/5"
    )
    print(
        f"Folds unchanged        : "
        f"{unchanged}/5"
    )
    print(
        f"Selected secondaries   : "
        f"{selected_counts}"
    )
    print(
        f"Perfect upper bound    : "
        f"+{raw_ties['theoretical_perfect_tie_break_auc_gain']:.8f}"
    )
    print(
        f"Primitive              : "
        f"{primitive}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [raw_ties]
    ).to_csv(
        OUTPUT_DIR
        / "tie_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        descriptive_rows
    ).to_csv(
        OUTPUT_DIR
        / "fixed_secondary_results.csv",
        index=False,
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "meta_fold_metrics.csv",
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
            "primary_meta_rank": (
                baseline_meta
            ),
            "lexsort_meta_prediction": (
                meta_prediction
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: EXACT LEXICOGRAPHIC TIE-BREAK AUDIT",
        "=" * 86,
        "",
        "NO TRAINING",
        "NO WEIGHT SEARCH",
        "NO NEAR-TIE THRESHOLD",
        "",
        f"Raw champion OOF: {EXPECTED['champion']:.8f}",
        f"Primary meta AUC: {baseline_meta_auc:.8f}",
        "",
        "EXACT TIE BURDEN",
        f"unique champion scores: {raw_ties['unique_primary_scores']}",
        f"tie groups: {raw_ties['tie_groups']}",
        f"rows in tie groups: {raw_ties['rows_in_tie_groups']}",
        f"fraction rows in ties: {raw_ties['fraction_rows_in_ties']:.8f}",
        f"max tie group size: {raw_ties['max_tie_group_size']}",
        f"positive-negative tied pairs: {raw_ties['positive_negative_tied_pairs']}",
        (
            "perfect tie-break AUC upper bound: "
            f"+{raw_ties['theoretical_perfect_tie_break_auc_gain']:.8f}"
        ),
        "",
        "META-CV RESULT",
        f"Lexsort meta-CV AUC: {candidate_meta_auc:.8f}",
        f"Delta: {meta_delta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        f"Folds unchanged: {unchanged}/5",
        f"Selected secondaries: {selected_counts}",
        f"Primitive: {primitive}",
        "",
        "HELD-FOLD RESULTS",
    ]

    for row in (
        fold_results
        .itertuples()
    ):
        summary.append(
            f"Fold {row.held_fold}: "
            f"secondary={row.selected_secondary} | "
            f"{row.held_primary_auc:.8f} -> "
            f"{row.held_lexsort_auc:.8f} "
            f"({row.delta:+.8f})"
        )

    summary.extend(
        [
            "",
            "INTERPRETATION",
            (
                "This experiment changes ordering ONLY inside exact primary-score ties."
            ),
            (
                "If the perfect upper bound is tiny, exact lexsort cannot materially "
                "improve this champion regardless of secondary model."
            ),
        ]
    )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print(
        f"Artifacts: "
        f"{OUTPUT_DIR.relative_to(ROOT)}"
    )
    print(
        "Done. No models were trained."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "\n" + "=" * 100
        )
        print("AUDIT FAILED")
        print("=" * 100)
        print(str(exc))
        sys.exit(1)
