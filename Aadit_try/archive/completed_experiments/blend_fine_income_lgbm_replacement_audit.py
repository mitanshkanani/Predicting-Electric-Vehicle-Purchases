"""
blend_fine_income_lgbm_replacement_audit.py

Kaggle Playground Series S6E9

NO TRAINING.
NO WEIGHT SEARCH.
NO LEADERBOARD USAGE.

Question
--------
Does the newly validated fine-income LightGBM survive when it replaces ONLY
the old engineered LightGBM inside the current validated champion?

Current champion components:
- CatBoost hierarchical income+commute 3-seed
- fine-income XGBoost
- OLD engineered LightGBM

Current frozen fold weights:
fold 0: CAT=.25 XGB=.50 LGBM=.25
fold 1: CAT=.25 XGB=.50 LGBM=.25
fold 2: CAT=.20 XGB=.55 LGBM=.25
fold 3: CAT=.20 XGB=.55 LGBM=.25
fold 4: CAT=.20 XGB=.55 LGBM=.25

ONLY CHANGE
-----------
OLD engineered LightGBM -> fine-income LightGBM.

Everything else is frozen.

Run:
    python blend_fine_income_lgbm_replacement_audit.py
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

CAT_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

FINE_XGB_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "xgboost_fine_income_te_gpu"
    / "oof_predictions.csv"
)

OLD_LGBM_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

FINE_LGBM_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_fine_income_te_cpu"
    / "oof_predictions.csv"
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
    / "blend_fine_income_lgbm_replacement_audit"
)

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED = {
    "cat": 0.94558572,
    "fine_xgb": 0.94606664,
    "old_lgbm": 0.94578042,
    "fine_lgbm": 0.94595207,
    "champion": 0.94611253,
}

AUC_TOL = 5e-6

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

FOLD_WEIGHTS = {
    0: (0.25, 0.50, 0.25),
    1: (0.25, 0.50, 0.25),
    2: (0.20, 0.55, 0.25),
    3: (0.20, 0.55, 0.25),
    4: (0.20, 0.55, 0.25),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = (
        pd.Series(a)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    br = (
        pd.Series(b)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    return float(np.corrcoef(ar, br)[0, 1])


def pred_col(df: pd.DataFrame) -> str:
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
            f"Could not identify prediction column. "
            f"Candidates={candidates}, columns={list(df.columns)}"
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
            f"Row-count mismatch in {path}: "
            f"{len(df)} vs {len(train)}"
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

    pred = pd.to_numeric(
        df[pred_col(df)],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError(
            f"Non-finite predictions in {path}"
        )

    return pred


def fold_rank(
    values: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    out = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    for fold in range(5):
        mask = folds == fold

        out[mask] = (
            pd.Series(values[mask])
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64)
        )

    if np.isnan(out).any():
        raise RuntimeError(
            "fold_rank created NaNs."
        )

    return out


def build_blend(
    cat: np.ndarray,
    xgb: np.ndarray,
    lgbm: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    cat_r = fold_rank(cat, folds)
    xgb_r = fold_rank(xgb, folds)
    lgbm_r = fold_rank(lgbm, folds)

    out = np.full(
        len(cat),
        np.nan,
        dtype=np.float64,
    )

    for fold in range(5):
        mask = folds == fold

        wc, wx, wl = FOLD_WEIGHTS[
            fold
        ]

        out[mask] = (
            wc * cat_r[mask]
            + wx * xgb_r[mask]
            + wl * lgbm_r[mask]
        )

    if np.isnan(out).any():
        raise RuntimeError(
            "Blend created NaNs."
        )

    return out


def main() -> None:
    print("=" * 100)
    print("FINE-INCOME LIGHTGBM -> CURRENT CHAMPION REPLACEMENT AUDIT")
    print("NO TRAINING | NO WEIGHT SEARCH | SAME FROZEN FOLD WEIGHTS")
    print("=" * 100)

    for path in [
        TRAIN_PATH,
        FOLDS_PATH,
        CAT_OOF_PATH,
        FINE_XGB_OOF_PATH,
        OLD_LGBM_OOF_PATH,
        FINE_LGBM_OOF_PATH,
        CHAMPION_OOF_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_FOLD_SHA:
        raise RuntimeError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(
        TRAIN_PATH
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    if "fold" not in folds_df.columns:
        raise RuntimeError(
            "Frozen fold file has no 'fold' column."
        )

    folds = folds_df[
        "fold"
    ].to_numpy(dtype=int)

    y = (
        train[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(POSITIVE_LABEL.lower())
        .astype(np.int8)
        .to_numpy()
    )

    members = {
        "cat": load_oof(
            CAT_OOF_PATH,
            train,
            folds,
        ),
        "fine_xgb": load_oof(
            FINE_XGB_OOF_PATH,
            train,
            folds,
        ),
        "old_lgbm": load_oof(
            OLD_LGBM_OOF_PATH,
            train,
            folds,
        ),
        "fine_lgbm": load_oof(
            FINE_LGBM_OOF_PATH,
            train,
            folds,
        ),
        "champion": load_oof(
            CHAMPION_OOF_PATH,
            train,
            folds,
        ),
    }

    print(
        f"\nFrozen fold SHA256 verified: "
        f"{fold_hash}"
    )

    print("\n--- Member integrity ---")

    for name, pred in members.items():
        score = auc(
            y,
            pred,
        )

        expected = EXPECTED[
            name
        ]

        if abs(
            score
            - expected
        ) > AUC_TOL:
            raise RuntimeError(
                f"{name} AUC mismatch: "
                f"{score:.8f} vs expected {expected:.8f}"
            )

        print(
            f"{name:9s}: "
            f"{score:.8f}"
        )

    control = build_blend(
        members["cat"],
        members["fine_xgb"],
        members["old_lgbm"],
        folds,
    )

    candidate = build_blend(
        members["cat"],
        members["fine_xgb"],
        members["fine_lgbm"],
        folds,
    )

    control_auc = auc(
        y,
        control,
    )

    candidate_auc = auc(
        y,
        candidate,
    )

    raw_champion_auc = auc(
        y,
        members["champion"],
    )

    if abs(
        control_auc
        - raw_champion_auc
    ) > AUC_TOL:
        raise RuntimeError(
            "Reconstructed champion does not match validated champion.\n"
            f"Reconstructed: {control_auc:.8f}\n"
            f"Artifact:      {raw_champion_auc:.8f}\n"
            "Do not continue until the blend protocol/path mismatch is resolved."
        )

    print()
    print("--- Frozen replacement test ---")
    print(
        "Only change: OLD engineered LGBM -> fine-income LGBM"
    )

    fold_rows = []

    for fold in range(5):
        mask = folds == fold

        control_fold_auc = auc(
            y[mask],
            control[mask],
        )

        candidate_fold_auc = auc(
            y[mask],
            candidate[mask],
        )

        delta = (
            candidate_fold_auc
            - control_fold_auc
        )

        wc, wx, wl = FOLD_WEIGHTS[
            fold
        ]

        fold_rows.append(
            {
                "fold": fold,
                "cat_weight": wc,
                "xgb_weight": wx,
                "lgbm_weight": wl,
                "control_auc": (
                    control_fold_auc
                ),
                "candidate_auc": (
                    candidate_fold_auc
                ),
                "delta": delta,
            }
        )

        print(
            f"Fold {fold}: "
            f"{control_fold_auc:.8f} -> "
            f"{candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"weights={wc:.2f}/{wx:.2f}/{wl:.2f}"
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    delta = (
        candidate_auc
        - control_auc
    )

    improved = int(
        (
            fold_metrics[
                "delta"
            ] > 0
        ).sum()
    )

    worse = int(
        (
            fold_metrics[
                "delta"
            ] < 0
        ).sum()
    )

    unchanged = 5 - improved - worse

    prob_corr = float(
        np.corrcoef(
            candidate,
            control,
        )[0, 1]
    )

    rank_correlation = rank_corr(
        candidate,
        control,
    )

    if (
        delta >= 2e-5
        and improved >= 4
    ):
        primitive = (
            "POSITIVE_FINE_LGBM_REPLACEMENT_SIGNAL"
        )
    elif (
        delta < 0
        or worse >= 4
    ):
        primitive = (
            "NEGATIVE_FINE_LGBM_REPLACEMENT_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_FINE_LGBM_REPLACEMENT_SIGNAL"
        )

    print()
    print("=" * 100)
    print("FINAL FINE-LGBM REPLACEMENT AUDIT")
    print("=" * 100)
    print(
        f"Current champion      : "
        f"{control_auc:.8f}"
    )
    print(
        f"Fine-LGBM replacement : "
        f"{candidate_auc:.8f}"
    )
    print(
        f"Delta                 : "
        f"{delta:+.8f}"
    )
    print(
        f"Folds improved        : "
        f"{improved}/5"
    )
    print(
        f"Folds worse           : "
        f"{worse}/5"
    )
    print(
        f"Folds unchanged       : "
        f"{unchanged}/5"
    )
    print(
        f"Prob corr vs champion : "
        f"{prob_corr:.6f}"
    )
    print(
        f"Rank corr vs champion : "
        f"{rank_correlation:.6f}"
    )
    print(
        f"Primitive             : "
        f"{primitive}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
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
            "control_oof_prediction": (
                control.astype(
                    np.float32
                )
            ),
            "candidate_oof_prediction": (
                candidate.astype(
                    np.float32
                )
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: FINE-INCOME LIGHTGBM -> CURRENT CHAMPION REPLACEMENT AUDIT",
        "=" * 88,
        "",
        "NO TRAINING",
        "NO WEIGHT SEARCH",
        "NO LEADERBOARD USAGE",
        "",
        "ONLY CHANGE",
        "OLD engineered LightGBM -> fine-income LightGBM",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        "",
        "FROZEN WEIGHTS",
        "Fold 0: CAT=.25 XGB=.50 LGBM=.25",
        "Fold 1: CAT=.25 XGB=.50 LGBM=.25",
        "Fold 2: CAT=.20 XGB=.55 LGBM=.25",
        "Fold 3: CAT=.20 XGB=.55 LGBM=.25",
        "Fold 4: CAT=.20 XGB=.55 LGBM=.25",
        "",
        "RESULT",
        f"Current champion: {control_auc:.8f}",
        f"Fine-LGBM replacement: {candidate_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        f"Folds unchanged: {unchanged}/5",
        f"Probability corr vs champion: {prob_corr:.6f}",
        f"Rank corr vs champion: {rank_correlation:.6f}",
        f"Primitive: {primitive}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary.append(
            f"Fold {row.fold}: "
            f"{row.control_auc:.8f} -> "
            f"{row.candidate_auc:.8f} "
            f"({row.delta:+.8f})"
        )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary
        ),
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
        print()
        print("=" * 100)
        print("AUDIT FAILED")
        print("=" * 100)
        print(str(exc))
        sys.exit(1)
