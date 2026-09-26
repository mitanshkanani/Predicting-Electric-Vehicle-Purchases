"""
audit_external_xgb10_fixed_rank_blend.py

Kaggle Playground Series S6E9

CONTROLLED FIXED EXTERNAL-BLEND AUDIT
-------------------------------------
Candidate external member:
    external_oof_strong/XGBoost_Triple_TE_10folds_oof.csv
    recomputed OOF reference ~= 0.94624283

Current champion:
    artifacts/experiments/blend_fine_income_xgb_validated_submission/oof_predictions.csv
    expected OOF ~= 0.94611253

HYPOTHESIS
----------
A genuinely strong independently-produced external XGBoost OOF member may
improve ranking when combined with our current champion.

ONLY CHANGE
-----------
Add the external 10-fold XGBoost member with a PREDECLARED FIXED 50/50
rank average:

    candidate = 0.50 * rank(champion) + 0.50 * rank(external_xgb10)

WHY FIXED?
----------
The external file does not expose its generation fold vector. Therefore this
experiment performs NO OOF weight fitting, NO alpha search and NO meta-model.
The 50/50 weight is declared before looking at the blend result.

HELD FIXED
----------
- our frozen validation fold file + SHA256
- current champion OOF/test predictions
- train/test row order and IDs
- external OOF/test files
- no model training
- no public leaderboard optimization
- no weight tuning

The frozen 5 folds are used ONLY to report paired consistency of the fixed
candidate versus champion; they are not used to select any parameter.

Run:
    python audit_external_xgb10_fixed_rank_blend.py
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

FOLDS_PATH = (
    ROOT / "artifacts" / "validation" / "candidate_folds.csv"
)

CHAMPION_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

CHAMPION_TEST_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "submission.csv"
)

CHAMPION_TEST_FALLBACK_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "test_predictions.csv"
)

EXTERNAL_OOF_PATH = (
    ROOT
    / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_oof.csv"
)

EXTERNAL_TEST_PATH = (
    ROOT
    / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_test.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "external_xgb10_fixed_rank_blend_audit"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)
EXPECTED_CHAMPION_AUC = 0.94611253
EXPECTED_EXTERNAL_AUC = 0.94624283
AUC_TOL = 5e-6

EXTERNAL_ALPHA = 0.50


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


def choose_prediction_column(
    df: pd.DataFrame,
    *,
    id_col: str | None,
    target_col: str | None = None,
) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
        "OOF_Pred",
        "Will_Buy_EV",
    ]

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
    }
    if id_col:
        excluded.add(id_col)
    if target_col:
        excluded.add(target_col)

    # First prefer known prediction names, but only if numeric probabilities.
    ordered = preferred + [
        c
        for c in df.columns
        if c not in preferred
    ]

    candidates = []
    for c in ordered:
        if c not in df.columns or c in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            continue

        mn = float(vals.min())
        mx = float(vals.max())
        if mn >= -1e-12 and mx <= 1.0 + 1e-12:
            candidates.append(c)

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 0:
        raise RuntimeError(
            "Could not identify numeric probability prediction column. "
            f"Columns={list(df.columns)}"
        )

    # Known prediction names win deterministically.
    for c in preferred:
        if c in candidates:
            return c

    if len(candidates) != 1:
        raise RuntimeError(
            "Ambiguous prediction columns: "
            f"{candidates}. Refuse to guess."
        )

    return candidates[0]


def load_prediction_csv(
    path: Path,
    reference: pd.DataFrame,
    *,
    id_col: str | None,
    folds: np.ndarray | None = None,
    target_col: str | None = None,
    require_id: bool = False,
) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != len(reference):
        raise RuntimeError(
            f"Row-count mismatch for {path.name}: "
            f"{len(df)} vs {len(reference)}"
        )

    row_index_verified = False
    fold_verified = False
    id_verified = False

    if "row_index" in df.columns:
        expected = np.arange(len(reference), dtype=np.int64)
        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError(
                f"row_index mismatch in {path.name}"
            )
        row_index_verified = True

    if folds is not None and "fold" in df.columns:
        found = df["fold"].to_numpy(dtype=np.int64)
        if not np.array_equal(found, folds):
            raise RuntimeError(
                f"fold mismatch in {path.name}"
            )
        fold_verified = True

    if id_col and id_col in reference.columns and id_col in df.columns:
        if not np.array_equal(
            df[id_col].to_numpy(),
            reference[id_col].to_numpy(),
        ):
            raise RuntimeError(
                f"ID order mismatch in {path.name}"
            )
        id_verified = True

    if require_id and not id_verified:
        raise RuntimeError(
            f"{path.name} must expose exact {id_col!r} alignment "
            "for this test-side artifact."
        )

    # OOF artifacts from our validated pipeline may intentionally omit ID but
    # include exact row_index + fold columns. That is sufficient positional
    # evidence and is stronger than merely assuming file order.
    if folds is not None and not id_verified:
        if not (row_index_verified and fold_verified):
            raise RuntimeError(
                f"{path.name} has no ID column and does not provide both "
                "verified row_index and frozen-fold alignment."
            )

    pred_col = choose_prediction_column(
        df,
        id_col=id_col if id_col in df.columns else None,
        target_col=target_col,
    )

    pred = pd.to_numeric(
        df[pred_col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError(
            f"Non-finite predictions in {path.name}"
        )

    return pred, pred_col


def main() -> None:
    start = time.perf_counter()

    champion_test_path = (
        CHAMPION_TEST_PATH
        if CHAMPION_TEST_PATH.exists()
        else CHAMPION_TEST_FALLBACK_PATH
    )

    required = [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        CHAMPION_OOF_PATH,
        champion_test_path,
        EXTERNAL_OOF_PATH,
        EXTERNAL_TEST_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(FOLDS_PATH)
    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)

    if len(folds_df) != len(train):
        raise RuntimeError(
            "candidate_folds.csv row count does not match train.csv."
        )

    if "row_index" in folds_df.columns:
        expected = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(
            folds_df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError(
                "candidate_folds.csv row order mismatch."
            )

    folds = folds_df["fold"].to_numpy(dtype=np.int64)

    target = detect_target(train, test)
    y = encode_target(train[target])

    id_col = (
        "id"
        if "id" in train.columns and "id" in test.columns
        else None
    )

    champion_oof, champion_oof_col = load_prediction_csv(
        CHAMPION_OOF_PATH,
        train,
        id_col=id_col,
        folds=folds,
        target_col=target,
    )

    champion_test, champion_test_col = load_prediction_csv(
        champion_test_path,
        test,
        id_col=id_col,
        require_id=True,
    )

    external_oof, external_oof_col = load_prediction_csv(
        EXTERNAL_OOF_PATH,
        train,
        id_col=id_col,
        target_col=target,
    )

    external_test, external_test_col = load_prediction_csv(
        EXTERNAL_TEST_PATH,
        test,
        id_col=id_col,
        require_id=True,
    )

    champion_auc = auc(y, champion_oof)
    external_auc = auc(y, external_oof)

    if abs(champion_auc - EXPECTED_CHAMPION_AUC) > AUC_TOL:
        raise RuntimeError(
            "Champion OOF AUC mismatch.\n"
            f"Expected: {EXPECTED_CHAMPION_AUC:.8f}\n"
            f"Loaded:   {champion_auc:.8f}"
        )

    if abs(external_auc - EXPECTED_EXTERNAL_AUC) > AUC_TOL:
        raise RuntimeError(
            "External XGB10 OOF AUC mismatch.\n"
            f"Expected: {EXPECTED_EXTERNAL_AUC:.8f}\n"
            f"Loaded:   {external_auc:.8f}"
        )

    champion_rank = rank_pct(champion_oof)
    external_rank = rank_pct(external_oof)

    candidate_oof = (
        (1.0 - EXTERNAL_ALPHA) * champion_rank
        + EXTERNAL_ALPHA * external_rank
    )

    candidate_auc = auc(y, candidate_oof)
    delta = candidate_auc - champion_auc

    prob_corr = float(
        np.corrcoef(champion_oof, external_oof)[0, 1]
    )
    rcorr = rank_corr(champion_oof, external_oof)

    fold_rows = []
    for fold in range(5):
        mask = folds == fold

        baseline_fold_auc = auc(
            y[mask],
            champion_oof[mask],
        )
        external_fold_auc = auc(
            y[mask],
            external_oof[mask],
        )
        candidate_fold_auc = auc(
            y[mask],
            candidate_oof[mask],
        )

        fold_rows.append(
            {
                "fold": fold,
                "champion_auc": baseline_fold_auc,
                "external_xgb10_auc": external_fold_auc,
                "fixed_rank_blend_auc": candidate_fold_auc,
                "delta_blend_vs_champion": (
                    candidate_fold_auc - baseline_fold_auc
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)

    folds_improved = int(
        (fold_metrics["delta_blend_vs_champion"] > 0).sum()
    )
    folds_worse = int(
        (fold_metrics["delta_blend_vs_champion"] < 0).sum()
    )

    if delta >= 3e-5 and folds_improved >= 4:
        primitive = "POSITIVE_FIXED_EXTERNAL_XGB10_BLEND_SIGNAL"
    elif delta <= 0 or folds_worse >= 4:
        primitive = "NEGATIVE_FIXED_EXTERNAL_XGB10_BLEND_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_FIXED_EXTERNAL_XGB10_BLEND_SIGNAL"

    # Deployment counterpart uses the exact same fixed 50/50 rank rule.
    champion_test_rank = rank_pct(champion_test)
    external_test_rank = rank_pct(external_test)

    candidate_test = (
        (1.0 - EXTERNAL_ALPHA) * champion_test_rank
        + EXTERNAL_ALPHA * external_test_rank
    )

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
                "champion_auc": champion_auc,
                "external_xgb10_auc": external_auc,
                "fixed_rank_blend_auc": candidate_auc,
                "delta_vs_champion": delta,
                "probability_corr": prob_corr,
                "rank_corr": rcorr,
                "external_alpha": EXTERNAL_ALPHA,
                "folds_improved": folds_improved,
                "folds_worse": folds_worse,
                "primitive": primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "audit_metrics.csv",
        index=False,
    )

    submission = pd.DataFrame()
    if id_col is not None:
        submission[id_col] = test[id_col].to_numpy()
    submission[target] = candidate_test

    submission.to_csv(
        OUTPUT_DIR / "candidate_submission.csv",
        index=False,
    )

    runtime = time.perf_counter() - start

    summary = [
        "EXPERIMENT: FIXED 50/50 RANK BLEND — CHAMPION + EXTERNAL XGB10",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target ensemble audit.",
        "",
        "HYPOTHESIS",
        (
            "Can the independently-produced stronger external 10-fold "
            "XGBoost OOF improve our current champion via a fixed equal "
            "rank average?"
        ),
        "",
        "ONLY CHANGE",
        "Add external XGB10 at fixed alpha=0.50 in rank space.",
        "",
        "NO WEIGHT SEARCH",
        "No alpha grid. No meta-model. No public leaderboard optimization.",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {fold_hash}",
        (
            "External generation fold vector: not exposed; therefore no "
            "meta-weight fitting is performed."
        ),
        "",
        "INPUT COLUMNS",
        f"Champion OOF column: {champion_oof_col}",
        f"Champion test column: {champion_test_col}",
        f"External OOF column: {external_oof_col}",
        f"External test column: {external_test_col}",
        "",
        "RESULT",
        f"Champion OOF: {champion_auc:.8f}",
        f"External XGB10 OOF: {external_auc:.8f}",
        f"Fixed 50/50 rank blend OOF: {candidate_auc:.8f}",
        f"Delta vs champion: {delta:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr: {prob_corr:.6f}",
        f"Rank corr: {rcorr:.6f}",
        f"Primitive: {primitive}",
        f"Runtime: {runtime:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"champion={r.champion_auc:.8f}, "
            f"external={r.external_xgb10_auc:.8f}, "
            f"blend={r.fixed_rank_blend_auc:.8f}, "
            f"delta={r.delta_blend_vs_champion:+.8f}"
        )

    summary.extend(
        [
            "",
            "DEPLOYMENT ARTIFACT",
            "candidate_submission.csv",
            "",
            "MANUAL REVIEW REQUIRED",
            (
                "Do not submit automatically. Promote only after reviewing "
                "OOF magnitude, fold consistency and external provenance."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("=" * 104)
    print("FIXED CHAMPION + EXTERNAL XGB10 RANK-BLEND AUDIT")
    print("=" * 104)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Champion OOF       : {champion_auc:.8f}")
    print(f"External XGB10 OOF : {external_auc:.8f}")
    print(f"Fixed blend weight : champion=0.50 external=0.50")
    print(f"Probability corr   : {prob_corr:.6f}")
    print(f"Rank corr          : {rcorr:.6f}")
    print()

    for r in fold_metrics.itertuples():
        print(
            f"Fold {r.fold}: "
            f"{r.champion_auc:.8f} -> "
            f"{r.fixed_rank_blend_auc:.8f} "
            f"({r.delta_blend_vs_champion:+.8f}) | "
            f"external={r.external_xgb10_auc:.8f}"
        )

    print()
    print("=" * 104)
    print("RESULT")
    print("=" * 104)
    print(f"Champion OOF       : {champion_auc:.8f}")
    print(f"External XGB10 OOF : {external_auc:.8f}")
    print(f"Fixed rank blend   : {candidate_auc:.8f}")
    print(f"Delta              : {delta:+.8f}")
    print(f"Folds improved     : {folds_improved}/5")
    print(f"Folds worse        : {folds_worse}/5")
    print(f"Primitive          : {primitive}")
    print(f"Runtime            : {runtime:.2f}s")
    print(f"Artifacts          : {OUTPUT_DIR}")
    print("=" * 104)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. audit_metrics.csv")


if __name__ == "__main__":
    main()
