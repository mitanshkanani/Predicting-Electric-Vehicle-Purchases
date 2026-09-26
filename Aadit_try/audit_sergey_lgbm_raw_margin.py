"""
audit_sergey_lgbm_raw_margin.py

Diagnostic only.

Sergey_LGBM_oof.csv stores raw model margins/logits in column Will_Buy_EV.
ROC-AUC is invariant to monotonic transforms, so the raw margin can be scored
directly and ranked directly.

No blending.
No weight search.
No submission.
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

SERGEY_OOF_PATH = (
    ROOT / "external_oof_strong"
    / "Sergey_LGBM_oof.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "experiments"
    / "sergey_lgbm_raw_margin_audit"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_TWO_WAY_AUC = 0.94632348
AUC_TOL = 5e-6


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


def load_internal_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(INTERNAL_OOF_PATH)

    if len(df) != len(train):
        raise RuntimeError("Internal champion row-count mismatch.")

    if "row_index" not in df.columns or "fold" not in df.columns:
        raise RuntimeError(
            "Internal champion OOF must expose row_index and fold."
        )

    expected_index = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(
        df["row_index"].to_numpy(dtype=np.int64),
        expected_index,
    ):
        raise RuntimeError("Internal champion row_index mismatch.")

    if not np.array_equal(
        df["fold"].to_numpy(dtype=np.int64),
        folds,
    ):
        raise RuntimeError("Internal champion fold mismatch.")

    if "oof_prediction" not in df.columns:
        raise RuntimeError(
            f"Expected oof_prediction in internal artifact; got {list(df.columns)}"
        )

    p = pd.to_numeric(
        df["oof_prediction"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite internal predictions.")

    return p


def load_external_by_id(
    path: Path,
    train: pd.DataFrame,
    prediction_col: str,
) -> np.ndarray:
    df = pd.read_csv(path)

    if len(df) != len(train):
        raise RuntimeError(
            f"{path.name} row-count mismatch: {len(df)} vs {len(train)}"
        )

    if "id" not in df.columns:
        raise RuntimeError(f"{path.name} missing id column.")

    if not np.array_equal(
        df["id"].to_numpy(),
        train["id"].to_numpy(),
    ):
        raise RuntimeError(f"{path.name} ID order mismatch.")

    if prediction_col not in df.columns:
        raise RuntimeError(
            f"{path.name} missing {prediction_col!r}; "
            f"columns={list(df.columns)}"
        )

    p = pd.to_numeric(
        df[prediction_col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(p).all():
        raise RuntimeError(f"Non-finite predictions in {path.name}")

    return p


def main() -> None:
    for path in [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_OOF_PATH,
        XGB10_OOF_PATH,
        SERGEY_OOF_PATH,
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

    if "id" not in train.columns:
        raise RuntimeError("Expected competition id column.")

    internal = load_internal_oof(train, folds)

    xgb10 = load_external_by_id(
        XGB10_OOF_PATH,
        train,
        "OOF_Pred",
    )

    sergey_raw = load_external_by_id(
        SERGEY_OOF_PATH,
        train,
        "Will_Buy_EV",
    )

    current_two_way = (
        rank_pct(internal) + rank_pct(xgb10)
    ) / 2.0

    two_way_auc = auc(y, current_two_way)

    if abs(two_way_auc - EXPECTED_TWO_WAY_AUC) > AUC_TOL:
        raise RuntimeError(
            "Current 2-way champion mismatch.\n"
            f"Expected: {EXPECTED_TWO_WAY_AUC:.8f}\n"
            f"Found:    {two_way_auc:.8f}"
        )

    sergey_auc = auc(y, sergey_raw)

    sergey_rank = rank_pct(sergey_raw)
    champion_rank = rank_pct(current_two_way)

    rank_corr = float(
        np.corrcoef(sergey_rank, champion_rank)[0, 1]
    )

    prob_like_corr = float(
        np.corrcoef(sergey_raw, current_two_way)[0, 1]
    )

    fold_rows = []
    for fold in range(5):
        m = folds == fold
        fold_rows.append(
            {
                "fold": fold,
                "current_two_way_auc": auc(y[m], current_two_way[m]),
                "sergey_raw_margin_auc": auc(y[m], sergey_raw[m]),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "current_two_way_auc": two_way_auc,
                "sergey_raw_margin_auc": sergey_auc,
                "rank_corr_vs_two_way": rank_corr,
                "raw_score_corr_vs_two_way": prob_like_corr,
                "rank_distance_1_minus_corr": 1.0 - rank_corr,
                "raw_margin_min": float(np.min(sergey_raw)),
                "raw_margin_max": float(np.max(sergey_raw)),
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "audit_metrics.csv",
        index=False,
    )

    summary = [
        "DIAGNOSTIC: SERGEY LGBM RAW-MARGIN OOF",
        "=" * 92,
        "",
        "TYPE",
        "Diagnostic only. No blending. No tuning. No submission.",
        "",
        "INTERPRETATION",
        (
            "Sergey_LGBM_oof.csv stores raw model margins/logits in "
            "Will_Buy_EV. ROC-AUC is scored directly because any monotonic "
            "sigmoid transform preserves ranking."
        ),
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"Current 2-way champion OOF: {two_way_auc:.8f}",
        f"Sergey raw-margin OOF: {sergey_auc:.8f}",
        f"Rank corr vs 2-way: {rank_corr:.6f}",
        f"Rank distance: {1.0 - rank_corr:.6f}",
        f"Raw-score corr vs 2-way: {prob_like_corr:.6f}",
        f"Raw margin range: [{np.min(sergey_raw):.6f}, {np.max(sergey_raw):.6f}]",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"two-way={r.current_two_way_auc:.8f}, "
            f"Sergey={r.sergey_raw_margin_auc:.8f}"
        )

    summary.extend(
        [
            "",
            "NEXT STEP",
            (
                "Manual review: only if standalone strength and diversity are "
                "credible should Sergey receive one fixed-weight ensemble test."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("=" * 104)
    print("SERGEY LGBM RAW-MARGIN DIAGNOSTIC")
    print("=" * 104)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Current 2-way OOF  : {two_way_auc:.8f}")
    print(f"Sergey OOF         : {sergey_auc:.8f}")
    print(f"Rank corr vs 2-way : {rank_corr:.6f}")
    print(f"Rank distance      : {1.0 - rank_corr:.6f}")
    print(f"Raw-score corr     : {prob_like_corr:.6f}")
    print(
        f"Raw margin range   : "
        f"[{np.min(sergey_raw):.6f}, {np.max(sergey_raw):.6f}]"
    )
    print()

    for r in fold_metrics.itertuples():
        print(
            f"Fold {r.fold}: "
            f"two-way={r.current_two_way_auc:.8f} | "
            f"Sergey={r.sergey_raw_margin_auc:.8f}"
        )

    print()
    print(f"Artifacts: {OUTPUT_DIR}")
    print("=" * 104)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. audit_metrics.csv")


if __name__ == "__main__":
    main()
