"""
screen_remaining_external_diversity.py

Diagnostic only.

Goal:
- inspect remaining strong external OOF candidates
- recompute standalone AUC
- measure rank/probability correlation versus the current 2-way champion
- report per-fold standalone AUC
- DO NOT blend
- DO NOT tune weights
- DO NOT create a submission
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

INTERNAL_OOF_PATH = (
    ROOT / "artifacts" / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

XGB10_OOF_PATH = (
    ROOT / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_oof.csv"
)

EXTERNAL_DIR = ROOT / "external_oof_strong"

OUTPUT_DIR = (
    ROOT / "artifacts" / "experiments"
    / "remaining_external_diversity_screen"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_TWO_WAY_AUC = 0.94632348
AUC_TOL = 5e-6

EXCLUDE_FILES = {
    "XGBoost_Triple_TE_10folds_oof.csv",
    "Pure LGBM_V5_oof.csv",
    "01_blend_oof.csv",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise RuntimeError(
            f"Expected one train-only target column, found {train_only}"
        )
    return train_only[0]


def encode_target(y: pd.Series) -> np.ndarray:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise RuntimeError(f"Expected binary target, found {values}")

    yes = [v for v in values if str(v).strip().lower() == "yes"]
    positive = yes[0] if yes else y.value_counts().idxmin()
    return (y == positive).astype(np.int8).to_numpy()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def rank_pct(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(x, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def prediction_column(df: pd.DataFrame, id_col: str | None) -> str | None:
    preferred = [
        "OOF_Pred",
        "oof_prediction",
        "prediction",
        "pred",
        "probability",
        "prob",
        "Will_Buy_EV",
    ]

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "label",
        "y",
    }
    if id_col:
        excluded.add(id_col)

    valid = []

    for c in preferred + list(df.columns):
        if c not in df.columns or c in excluded or c in valid:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            continue

        mn = float(vals.min())
        mx = float(vals.max())

        if mn >= -1e-12 and mx <= 1.0 + 1e-12:
            valid.append(c)

    for c in preferred:
        if c in valid:
            return c

    if len(valid) == 1:
        return valid[0]

    return None


def load_internal_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(INTERNAL_OOF_PATH)

    if len(df) != len(train):
        raise RuntimeError("Internal OOF row-count mismatch.")

    if "row_index" not in df.columns or "fold" not in df.columns:
        raise RuntimeError(
            "Internal OOF must contain row_index and fold."
        )

    expected_index = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(
        df["row_index"].to_numpy(dtype=np.int64),
        expected_index,
    ):
        raise RuntimeError("Internal OOF row_index mismatch.")

    if not np.array_equal(
        df["fold"].to_numpy(dtype=np.int64),
        folds,
    ):
        raise RuntimeError("Internal OOF fold mismatch.")

    col = prediction_column(df, id_col=None)
    if col is None:
        raise RuntimeError("Could not identify internal OOF prediction column.")

    return pd.to_numeric(
        df[col],
        errors="raise",
    ).to_numpy(dtype=np.float64)


def load_external_oof(
    path: Path,
    train: pd.DataFrame,
) -> tuple[np.ndarray | None, str | None, str]:
    df = pd.read_csv(path)

    if len(df) != len(train):
        return None, None, "ROW_COUNT_MISMATCH"

    if "id" not in df.columns:
        return None, None, "NO_ID"

    if not np.array_equal(
        df["id"].to_numpy(),
        train["id"].to_numpy(),
    ):
        return None, None, "ID_MISMATCH"

    col = prediction_column(df, id_col="id")

    if col is None:
        return None, None, "NO_UNAMBIGUOUS_PREDICTION_COLUMN"

    p = pd.to_numeric(
        df[col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(p).all():
        return None, col, "NONFINITE"

    return p, col, "OK"


def main() -> None:
    for path in [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_OOF_PATH,
        XGB10_OOF_PATH,
        EXTERNAL_DIR,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(FOLDS_PATH)
    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    fold_df = pd.read_csv(FOLDS_PATH)

    if len(fold_df) != len(train):
        raise RuntimeError("Frozen fold row-count mismatch.")

    folds = fold_df["fold"].to_numpy(dtype=np.int64)

    target = detect_target(train, test)
    y = encode_target(train[target])

    internal_oof = load_internal_oof(train, folds)

    xgb10_df = pd.read_csv(XGB10_OOF_PATH)

    if len(xgb10_df) != len(train):
        raise RuntimeError("XGB10 row-count mismatch.")

    if "id" not in xgb10_df.columns:
        raise RuntimeError("XGB10 has no ID column.")

    if not np.array_equal(
        xgb10_df["id"].to_numpy(),
        train["id"].to_numpy(),
    ):
        raise RuntimeError("XGB10 ID mismatch.")

    xgb10_col = prediction_column(xgb10_df, id_col="id")
    if xgb10_col is None:
        raise RuntimeError("Could not identify XGB10 prediction column.")

    xgb10_oof = pd.to_numeric(
        xgb10_df[xgb10_col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    current_two_way = (
        rank_pct(internal_oof) + rank_pct(xgb10_oof)
    ) / 2.0

    two_way_auc = auc(y, current_two_way)

    if abs(two_way_auc - EXPECTED_TWO_WAY_AUC) > AUC_TOL:
        raise RuntimeError(
            "Current 2-way champion mismatch.\n"
            f"Expected: {EXPECTED_TWO_WAY_AUC:.8f}\n"
            f"Found:    {two_way_auc:.8f}"
        )

    current_rank = rank_pct(current_two_way)

    rows = []

    oof_files = sorted(EXTERNAL_DIR.glob("*_oof.csv"))

    for path in oof_files:
        if path.name in EXCLUDE_FILES:
            continue

        p, col, status = load_external_oof(path, train)

        row = {
            "file": path.name,
            "status": status,
            "prediction_column": col,
            "standalone_auc": np.nan,
            "prob_corr_vs_two_way": np.nan,
            "rank_corr_vs_two_way": np.nan,
            "rank_distance_1_minus_corr": np.nan,
        }

        for fold in range(5):
            row[f"fold_{fold}_auc"] = np.nan

        if status == "OK" and p is not None:
            p_rank = rank_pct(p)

            row["standalone_auc"] = auc(y, p)
            row["prob_corr_vs_two_way"] = float(
                np.corrcoef(p, current_two_way)[0, 1]
            )
            row["rank_corr_vs_two_way"] = float(
                np.corrcoef(p_rank, current_rank)[0, 1]
            )
            row["rank_distance_1_minus_corr"] = (
                1.0 - row["rank_corr_vs_two_way"]
            )

            for fold in range(5):
                m = folds == fold
                row[f"fold_{fold}_auc"] = auc(y[m], p[m])

        rows.append(row)

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["status", "standalone_auc"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_DIR / "diversity_screen.csv",
        index=False,
    )

    summary = [
        "DIAGNOSTIC: REMAINING STRONG EXTERNAL OOF DIVERSITY SCREEN",
        "=" * 92,
        "",
        "TYPE",
        "Diagnostic only. No blending. No weight search. No submission.",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"Current 2-way champion OOF: {two_way_auc:.8f}",
        "",
        "EXCLUDED BECAUSE ALREADY TESTED",
        "- XGBoost_Triple_TE_10folds_oof.csv",
        "- Pure LGBM_V5_oof.csv",
        "- 01_blend_oof.csv",
        "",
        "RESULTS",
    ]

    for r in result.itertuples():
        if r.status == "OK":
            summary.append(
                f"{r.file}: "
                f"AUC={r.standalone_auc:.8f}, "
                f"rank_corr_vs_two_way={r.rank_corr_vs_two_way:.6f}, "
                f"rank_distance={r.rank_distance_1_minus_corr:.6f}"
            )
        else:
            summary.append(
                f"{r.file}: status={r.status}"
            )

    summary.extend(
        [
            "",
            "INTERPRETATION",
            (
                "This file does not test ensemble performance. It only identifies "
                "which remaining member combines standalone strength with genuine "
                "ranking diversity versus the current champion."
            ),
            "",
            "NEXT STEP",
            (
                "Choose exactly one candidate for the next fixed-weight ensemble "
                "experiment after manual review."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("=" * 104)
    print("REMAINING EXTERNAL OOF DIVERSITY SCREEN")
    print("=" * 104)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Current 2-way OOF  : {two_way_auc:.8f}")
    print()

    if result.empty:
        print("No remaining OOF candidates found.")
    else:
        cols = [
            "file",
            "status",
            "standalone_auc",
            "rank_corr_vs_two_way",
            "rank_distance_1_minus_corr",
        ]
        print(result[cols].to_string(index=False))

    print()
    print(f"Artifacts: {OUTPUT_DIR}")
    print("=" * 104)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. diversity_screen.csv")


if __name__ == "__main__":
    main()
