"""
audit_external_lgbmv5_threeway_fixed_rank_blend.py

S6E9 controlled ensemble audit.

BASELINE:
    fixed 50/50 rank blend:
      - current internal champion
      - external XGBoost Triple-TE 10-fold

CANDIDATE:
    fixed equal-weight 3-way rank blend:
      - 1/3 current internal champion
      - 1/3 external XGBoost Triple-TE 10-fold
      - 1/3 external Pure LGBM V5

No weight search.
No training.
No leaderboard optimization.
"""

from __future__ import annotations

import hashlib
import time
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

INTERNAL_TEST_PATH = (
    ROOT / "artifacts" / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "submission.csv"
)

XGB10_OOF_PATH = (
    ROOT / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_oof.csv"
)

XGB10_TEST_PATH = (
    ROOT / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_test.csv"
)

LGBMV5_OOF_PATH = (
    ROOT / "external_oof_strong"
    / "Pure LGBM_V5_oof.csv"
)

LGBMV5_TEST_PATH = (
    ROOT / "external_oof_strong"
    / "Pure LGBM_V5_test.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "experiments"
    / "external_lgbmv5_threeway_fixed_rank_blend_audit"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_INTERNAL_AUC = 0.94611253
EXPECTED_XGB10_AUC = 0.94624283
EXPECTED_LGBMV5_AUC = 0.94616964
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


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank_pct(a), rank_pct(b))[0, 1])


def prediction_column(df: pd.DataFrame, id_col: str | None) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
        "OOF_Pred",
        "Will_Buy_EV",
    ]

    excluded = {"row_index", "fold", "target", "target_encoded"}
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

    if not valid:
        raise RuntimeError(
            f"No probability-like prediction column in {list(df.columns)}"
        )

    for c in preferred:
        if c in valid:
            return c

    if len(valid) != 1:
        raise RuntimeError(f"Ambiguous prediction columns: {valid}")

    return valid[0]


def load_internal_oof(
    path: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[np.ndarray, str]:
    df = pd.read_csv(path)

    if len(df) != len(train):
        raise RuntimeError(
            f"{path.name} row count mismatch: {len(df)} vs {len(train)}"
        )

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

    col = prediction_column(df, id_col=None)
    pred = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite internal OOF predictions.")

    return pred, col


def load_id_aligned(
    path: Path,
    reference: pd.DataFrame,
    id_col: str,
) -> tuple[np.ndarray, str]:
    df = pd.read_csv(path)

    if len(df) != len(reference):
        raise RuntimeError(
            f"{path.name} row count mismatch: {len(df)} vs {len(reference)}"
        )

    if id_col not in df.columns:
        raise RuntimeError(
            f"{path.name} does not contain required ID column {id_col!r}"
        )

    if not np.array_equal(
        df[id_col].to_numpy(),
        reference[id_col].to_numpy(),
    ):
        raise RuntimeError(f"{path.name} ID alignment mismatch.")

    col = prediction_column(df, id_col=id_col)
    pred = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite predictions in {path.name}")

    return pred, col


def main() -> None:
    start = time.perf_counter()

    required = [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_OOF_PATH,
        INTERNAL_TEST_PATH,
        XGB10_OOF_PATH,
        XGB10_TEST_PATH,
        LGBMV5_OOF_PATH,
        LGBMV5_TEST_PATH,
    ]

    for path in required:
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
        raise RuntimeError("Frozen fold row count mismatch.")

    folds = fold_df["fold"].to_numpy(dtype=np.int64)

    target = detect_target(train, test)
    y = encode_target(train[target])

    if "id" not in train.columns or "id" not in test.columns:
        raise RuntimeError("Expected competition ID column 'id'.")

    id_col = "id"

    internal_oof, internal_oof_col = load_internal_oof(
        INTERNAL_OOF_PATH,
        train,
        folds,
    )

    internal_test, internal_test_col = load_id_aligned(
        INTERNAL_TEST_PATH,
        test,
        id_col,
    )

    xgb10_oof, xgb10_oof_col = load_id_aligned(
        XGB10_OOF_PATH,
        train,
        id_col,
    )

    xgb10_test, xgb10_test_col = load_id_aligned(
        XGB10_TEST_PATH,
        test,
        id_col,
    )

    lgbm_oof, lgbm_oof_col = load_id_aligned(
        LGBMV5_OOF_PATH,
        train,
        id_col,
    )

    lgbm_test, lgbm_test_col = load_id_aligned(
        LGBMV5_TEST_PATH,
        test,
        id_col,
    )

    internal_auc = auc(y, internal_oof)
    xgb10_auc = auc(y, xgb10_oof)
    lgbm_auc = auc(y, lgbm_oof)

    for label, found, expected in [
        ("internal", internal_auc, EXPECTED_INTERNAL_AUC),
        ("external_xgb10", xgb10_auc, EXPECTED_XGB10_AUC),
        ("external_lgbmv5", lgbm_auc, EXPECTED_LGBMV5_AUC),
    ]:
        if abs(found - expected) > AUC_TOL:
            raise RuntimeError(
                f"{label} AUC mismatch: expected {expected:.8f}, "
                f"found {found:.8f}"
            )

    r_internal = rank_pct(internal_oof)
    r_xgb10 = rank_pct(xgb10_oof)
    r_lgbm = rank_pct(lgbm_oof)

    baseline_oof = 0.50 * r_internal + 0.50 * r_xgb10
    candidate_oof = (
        r_internal + r_xgb10 + r_lgbm
    ) / 3.0

    baseline_auc = auc(y, baseline_oof)
    candidate_auc = auc(y, candidate_oof)
    delta = candidate_auc - baseline_auc

    if abs(baseline_auc - EXPECTED_TWO_WAY_AUC) > AUC_TOL:
        raise RuntimeError(
            "Recomputed two-way champion mismatch.\n"
            f"Expected: {EXPECTED_TWO_WAY_AUC:.8f}\n"
            f"Found:    {baseline_auc:.8f}"
        )

    corr_lgbm_baseline = rank_corr(lgbm_oof, baseline_oof)
    corr_lgbm_internal = rank_corr(lgbm_oof, internal_oof)
    corr_lgbm_xgb10 = rank_corr(lgbm_oof, xgb10_oof)

    fold_rows = []

    for fold in range(5):
        m = folds == fold

        baseline_fold = auc(y[m], baseline_oof[m])
        candidate_fold = auc(y[m], candidate_oof[m])
        lgbm_fold = auc(y[m], lgbm_oof[m])

        fold_rows.append(
            {
                "fold": fold,
                "two_way_champion_auc": baseline_fold,
                "lgbmv5_auc": lgbm_fold,
                "three_way_auc": candidate_fold,
                "delta_three_way_vs_two_way": (
                    candidate_fold - baseline_fold
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)

    folds_improved = int(
        (fold_metrics["delta_three_way_vs_two_way"] > 0).sum()
    )
    folds_worse = int(
        (fold_metrics["delta_three_way_vs_two_way"] < 0).sum()
    )

    if delta >= 3e-5 and folds_improved >= 4:
        primitive = "POSITIVE_THREE_WAY_SIGNAL"
    elif delta <= 0 or folds_worse >= 4:
        primitive = "NEGATIVE_THREE_WAY_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_THREE_WAY_SIGNAL"

    # Exact deployment counterpart.
    rt_internal = rank_pct(internal_test)
    rt_xgb10 = rank_pct(xgb10_test)
    rt_lgbm = rank_pct(lgbm_test)

    candidate_test = (
        rt_internal + rt_xgb10 + rt_lgbm
    ) / 3.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "internal_auc": internal_auc,
                "external_xgb10_auc": xgb10_auc,
                "external_lgbmv5_auc": lgbm_auc,
                "two_way_champion_auc": baseline_auc,
                "three_way_auc": candidate_auc,
                "delta_vs_two_way_champion": delta,
                "rank_corr_lgbmv5_vs_two_way": corr_lgbm_baseline,
                "rank_corr_lgbmv5_vs_internal": corr_lgbm_internal,
                "rank_corr_lgbmv5_vs_xgb10": corr_lgbm_xgb10,
                "folds_improved": folds_improved,
                "folds_worse": folds_worse,
                "primitive": primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "audit_metrics.csv",
        index=False,
    )

    submission = pd.DataFrame(
        {
            id_col: test[id_col].to_numpy(),
            target: candidate_test,
        }
    )

    submission.to_csv(
        OUTPUT_DIR / "candidate_submission.csv",
        index=False,
    )

    runtime = time.perf_counter() - start

    summary = [
        "EXPERIMENT: FIXED EQUAL-WEIGHT THREE-WAY RANK BLEND",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target ensemble audit.",
        "",
        "HYPOTHESIS",
        (
            "Can external Pure LGBM V5 add complementary ranking signal "
            "beyond the current fixed internal+XGB10 champion?"
        ),
        "",
        "ONLY CHANGE",
        (
            "Current 50/50 internal+XGB10 rank average -> "
            "fixed equal 1/3 internal + 1/3 XGB10 + 1/3 LGBM V5."
        ),
        "",
        "NO WEIGHT SEARCH",
        "No alpha grid. No meta-model. No leaderboard optimization.",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {fold_hash}",
        "",
        "INPUT COLUMNS",
        f"Internal OOF: {internal_oof_col}",
        f"Internal test: {internal_test_col}",
        f"XGB10 OOF: {xgb10_oof_col}",
        f"XGB10 test: {xgb10_test_col}",
        f"LGBM V5 OOF: {lgbm_oof_col}",
        f"LGBM V5 test: {lgbm_test_col}",
        "",
        "STANDALONE OOF",
        f"Internal champion: {internal_auc:.8f}",
        f"External XGB10: {xgb10_auc:.8f}",
        f"External LGBM V5: {lgbm_auc:.8f}",
        "",
        "ENSEMBLE RESULT",
        f"Current 2-way champion: {baseline_auc:.8f}",
        f"Fixed 3-way candidate: {candidate_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Rank corr LGBM V5 vs 2-way: {corr_lgbm_baseline:.6f}",
        f"Rank corr LGBM V5 vs internal: {corr_lgbm_internal:.6f}",
        f"Rank corr LGBM V5 vs XGB10: {corr_lgbm_xgb10:.6f}",
        f"Primitive: {primitive}",
        f"Runtime: {runtime:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"two-way={r.two_way_champion_auc:.8f}, "
            f"LGBM={r.lgbmv5_auc:.8f}, "
            f"three-way={r.three_way_auc:.8f}, "
            f"delta={r.delta_three_way_vs_two_way:+.8f}"
        )

    summary.extend(
        [
            "",
            "DEPLOYMENT ARTIFACT",
            "candidate_submission.csv",
            "",
            "MANUAL REVIEW REQUIRED",
            (
                "Do not submit automatically. Promote only if OOF magnitude "
                "and fold consistency are meaningful."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("=" * 108)
    print("FIXED THREE-WAY RANK-BLEND AUDIT")
    print("=" * 108)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Internal OOF       : {internal_auc:.8f}")
    print(f"External XGB10 OOF : {xgb10_auc:.8f}")
    print(f"External LGBM V5   : {lgbm_auc:.8f}")
    print()
    print(f"Current 2-way      : {baseline_auc:.8f}")
    print(f"Fixed 3-way        : {candidate_auc:.8f}")
    print(f"Delta              : {delta:+.8f}")
    print()
    print(f"LGBM rank corr vs 2-way : {corr_lgbm_baseline:.6f}")
    print(f"LGBM rank corr vs int   : {corr_lgbm_internal:.6f}")
    print(f"LGBM rank corr vs XGB10 : {corr_lgbm_xgb10:.6f}")
    print()

    for r in fold_metrics.itertuples():
        print(
            f"Fold {r.fold}: "
            f"{r.two_way_champion_auc:.8f} -> "
            f"{r.three_way_auc:.8f} "
            f"({r.delta_three_way_vs_two_way:+.8f}) | "
            f"LGBM={r.lgbmv5_auc:.8f}"
        )

    print()
    print("=" * 108)
    print("RESULT")
    print("=" * 108)
    print(f"2-way champion : {baseline_auc:.8f}")
    print(f"3-way candidate: {candidate_auc:.8f}")
    print(f"Delta          : {delta:+.8f}")
    print(f"Folds improved : {folds_improved}/5")
    print(f"Folds worse    : {folds_worse}/5")
    print(f"Primitive      : {primitive}")
    print(f"Runtime        : {runtime:.2f}s")
    print(f"Artifacts      : {OUTPUT_DIR}")
    print("=" * 108)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. audit_metrics.csv")


if __name__ == "__main__":
    main()
