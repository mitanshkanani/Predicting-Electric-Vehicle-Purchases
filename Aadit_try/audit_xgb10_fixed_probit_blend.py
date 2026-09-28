"""
audit_xgb10_fixed_probit_blend.py

S6E9 controlled ensemble audit.

BASELINE
--------
Fixed 50/50 percentile-rank blend:
    50% internal champion
    50% external XGB Triple-TE 10-fold

CANDIDATE
---------
Same two members, same 50/50 weights, but each prediction vector is transformed:

    average_rank -> rank / (n + 1) -> scipy.stats.norm.ppf

The two probit-rank vectors are then averaged 50/50.

For test submission, the averaged probit score is passed through norm.cdf to
return values in [0, 1]. This final monotonic transform does not alter ranking.

No training.
No weight search.
No leaderboard optimization.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
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

OUTPUT_DIR = (
    ROOT / "artifacts" / "experiments"
    / "xgb10_fixed_probit_blend_audit"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_INTERNAL_AUC = 0.94611253
EXPECTED_XGB10_AUC = 0.94624283
EXPECTED_BASELINE_AUC = 0.94632348
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


def percentile_rank(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(x, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def probit_rank(x: np.ndarray) -> np.ndarray:
    """
    Reproduce the EV Grand Prix rank -> probit geometry:

        average rank in [1, n]
        rank / (n + 1)
        norm.ppf(...)
    """
    ranks = (
        pd.Series(np.asarray(x, dtype=np.float64))
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )

    u = ranks / (len(ranks) + 1.0)

    if not ((u > 0).all() and (u < 1).all()):
        raise RuntimeError("Probit rank transform produced invalid quantiles.")

    z = norm.ppf(u)

    if not np.isfinite(z).all():
        raise RuntimeError("Probit transform produced non-finite values.")

    return z


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = percentile_rank(a)
    br = percentile_rank(b)
    return float(np.corrcoef(ar, br)[0, 1])


def load_internal_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(INTERNAL_OOF_PATH)

    if len(df) != len(train):
        raise RuntimeError("Internal OOF row-count mismatch.")

    if "row_index" not in df.columns or "fold" not in df.columns:
        raise RuntimeError(
            "Internal champion OOF must expose row_index and fold."
        )

    if not np.array_equal(
        df["row_index"].to_numpy(dtype=np.int64),
        np.arange(len(train), dtype=np.int64),
    ):
        raise RuntimeError("Internal champion row_index mismatch.")

    if not np.array_equal(
        df["fold"].to_numpy(dtype=np.int64),
        folds,
    ):
        raise RuntimeError("Internal champion fold mismatch.")

    if "oof_prediction" not in df.columns:
        raise RuntimeError(
            f"Expected oof_prediction; columns={list(df.columns)}"
        )

    p = pd.to_numeric(
        df["oof_prediction"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite internal OOF predictions.")

    return p


def load_id_aligned(
    path: Path,
    reference: pd.DataFrame,
    prediction_col: str,
) -> np.ndarray:
    df = pd.read_csv(path)

    if len(df) != len(reference):
        raise RuntimeError(
            f"{path.name} row-count mismatch: {len(df)} vs {len(reference)}"
        )

    if "id" not in df.columns:
        raise RuntimeError(f"{path.name} missing id column.")

    if not np.array_equal(
        df["id"].to_numpy(),
        reference["id"].to_numpy(),
    ):
        raise RuntimeError(f"{path.name} ID alignment mismatch.")

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
    start = time.perf_counter()

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_OOF_PATH,
        INTERNAL_TEST_PATH,
        XGB10_OOF_PATH,
        XGB10_TEST_PATH,
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

    if "id" not in train.columns or "id" not in test.columns:
        raise RuntimeError("Expected competition id column.")

    internal_oof = load_internal_oof(
        train,
        folds,
    )

    internal_test = load_id_aligned(
        INTERNAL_TEST_PATH,
        test,
        "Will_Buy_EV",
    )

    xgb10_oof = load_id_aligned(
        XGB10_OOF_PATH,
        train,
        "OOF_Pred",
    )

    xgb10_test = load_id_aligned(
        XGB10_TEST_PATH,
        test,
        "Will_Buy_EV",
    )

    internal_auc = auc(y, internal_oof)
    xgb10_auc = auc(y, xgb10_oof)

    for label, found, expected in [
        ("internal", internal_auc, EXPECTED_INTERNAL_AUC),
        ("xgb10", xgb10_auc, EXPECTED_XGB10_AUC),
    ]:
        if abs(found - expected) > AUC_TOL:
            raise RuntimeError(
                f"{label} AUC mismatch: expected {expected:.8f}, "
                f"found {found:.8f}"
            )

    # Existing champion geometry.
    baseline_oof = (
        0.50 * percentile_rank(internal_oof)
        + 0.50 * percentile_rank(xgb10_oof)
    )

    baseline_auc = auc(y, baseline_oof)

    if abs(baseline_auc - EXPECTED_BASELINE_AUC) > AUC_TOL:
        raise RuntimeError(
            "Recomputed baseline champion mismatch.\n"
            f"Expected: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Found:    {baseline_auc:.8f}"
        )

    # Only experiment change: percentile-rank geometry -> probit-rank geometry.
    z_internal = probit_rank(internal_oof)
    z_xgb10 = probit_rank(xgb10_oof)

    candidate_oof = (
        0.50 * z_internal
        + 0.50 * z_xgb10
    )

    candidate_auc = auc(y, candidate_oof)
    delta = candidate_auc - baseline_auc

    fold_rows = []

    for fold in range(5):
        m = folds == fold

        baseline_fold = auc(
            y[m],
            baseline_oof[m],
        )

        candidate_fold = auc(
            y[m],
            candidate_oof[m],
        )

        fold_rows.append(
            {
                "fold": fold,
                "percentile_rank_blend_auc": baseline_fold,
                "probit_rank_blend_auc": candidate_fold,
                "delta_vs_percentile_rank": (
                    candidate_fold - baseline_fold
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)

    folds_improved = int(
        (fold_metrics["delta_vs_percentile_rank"] > 0).sum()
    )

    folds_worse = int(
        (fold_metrics["delta_vs_percentile_rank"] < 0).sum()
    )

    corr_candidate_baseline = rank_corr(
        candidate_oof,
        baseline_oof,
    )

    if delta >= 3e-5 and folds_improved >= 4:
        primitive = "POSITIVE_PROBIT_BLEND_SIGNAL"
    elif delta <= 0 or folds_worse >= 4:
        primitive = "NEGATIVE_PROBIT_BLEND_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_PROBIT_BLEND_SIGNAL"

    # Exact deployment counterpart.
    z_internal_test = probit_rank(internal_test)
    z_xgb10_test = probit_rank(xgb10_test)

    candidate_test_z = (
        0.50 * z_internal_test
        + 0.50 * z_xgb10_test
    )

    # Return bounded values while preserving candidate ranking.
    candidate_test = norm.cdf(candidate_test_z)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "internal_auc": internal_auc,
                "xgb10_auc": xgb10_auc,
                "percentile_rank_baseline_auc": baseline_auc,
                "probit_rank_candidate_auc": candidate_auc,
                "delta_vs_baseline": delta,
                "folds_improved": folds_improved,
                "folds_worse": folds_worse,
                "rank_corr_candidate_vs_baseline": corr_candidate_baseline,
                "primitive": primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "audit_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            target: candidate_test,
        }
    ).to_csv(
        OUTPUT_DIR / "candidate_submission.csv",
        index=False,
    )

    runtime = time.perf_counter() - start

    summary = [
        "EXPERIMENT: FIXED 50/50 PROBIT-RANK BLEND",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target ensemble audit.",
        "",
        "HYPOTHESIS",
        (
            "Can Gaussianizing member ranks before fixed 50/50 averaging "
            "improve the current external-assisted champion?"
        ),
        "",
        "ONLY CHANGE",
        (
            "Plain percentile ranks -> rank/(n+1) -> norm.ppf, "
            "with the same two members and same 50/50 weights."
        ),
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- internal champion predictions",
        "- external XGB10 predictions",
        "- fixed 50/50 weights",
        "- no weight fitting",
        "- no model training",
        "- no leaderboard optimization",
        "",
        "RESULT",
        f"Percentile-rank baseline: {baseline_auc:.8f}",
        f"Probit-rank candidate: {candidate_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Rank corr candidate vs baseline: {corr_candidate_baseline:.6f}",
        f"Primitive: {primitive}",
        f"Runtime: {runtime:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"percentile={r.percentile_rank_blend_auc:.8f} -> "
            f"probit={r.probit_rank_blend_auc:.8f} "
            f"({r.delta_vs_percentile_rank:+.8f})"
        )

    summary.extend(
        [
            "",
            "DEPLOYMENT ARTIFACT",
            "candidate_submission.csv",
            "",
            "MANUAL REVIEW REQUIRED",
            "Do not submit automatically.",
        ]
    )

    (
        OUTPUT_DIR / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("=" * 108)
    print("FIXED 50/50 PROBIT-RANK BLEND AUDIT")
    print("=" * 108)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Internal OOF       : {internal_auc:.8f}")
    print(f"External XGB10 OOF : {xgb10_auc:.8f}")
    print()
    print(f"Rank baseline      : {baseline_auc:.8f}")
    print(f"Probit candidate   : {candidate_auc:.8f}")
    print(f"Delta              : {delta:+.8f}")
    print(f"Rank corr vs base  : {corr_candidate_baseline:.6f}")
    print()

    for r in fold_metrics.itertuples():
        print(
            f"Fold {r.fold}: "
            f"{r.percentile_rank_blend_auc:.8f} -> "
            f"{r.probit_rank_blend_auc:.8f} "
            f"({r.delta_vs_percentile_rank:+.8f})"
        )

    print()
    print("=" * 108)
    print("RESULT")
    print("=" * 108)
    print(f"Baseline       : {baseline_auc:.8f}")
    print(f"Candidate      : {candidate_auc:.8f}")
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
