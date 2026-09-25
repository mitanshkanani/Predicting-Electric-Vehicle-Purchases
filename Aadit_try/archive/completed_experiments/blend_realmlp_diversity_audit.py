from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# REALMLP DIVERSITY BLEND AUDIT
#
# NO MODEL TRAINING.
# NO FINE WEIGHT TUNING.
#
# Question:
# Does the materially less-correlated RealMLP prediction stream add
# ensemble value to the current validated champion?
#
# Method:
# - exact frozen folds
# - foldwise percentile ranks
# - coarse RealMLP weights only: 0.00, 0.05, ..., 0.30
# - leave-one-fold-out weight selection:
#     choose alpha on other 4 folds, evaluate on held fold
# - compare meta-OOF AUC to current champion
# ============================================================

ROOT = Path(__file__).resolve().parent

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_CHAMPION_AUC = 0.94611253
EXPECTED_REALMLP_AUC = 0.94329901
TOL = 5e-6

TRAIN_PATH = ROOT / "data" / "train.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

CHAMPION_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

REALMLP_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "realmlp_frozen5_vectorized_gpu"
    / "oof_predictions.csv"
)

OUT = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_realmlp_diversity_audit"
)

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

# Coarse only. Do not change to 0.01/0.02 steps.
ALPHAS = np.arange(0.00, 0.3001, 0.05)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pred_col(df: pd.DataFrame) -> str:
    for c in [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
    ]:
        if c in df.columns:
            return c

    exclude = {"id", "row_index", "fold", "target", "target_encoded"}
    nums = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(nums) != 1:
        raise ValueError(
            f"Cannot identify prediction column. Columns={list(df.columns)}"
        )

    return nums[0]


def load_oof(
    path: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise ValueError(f"row mismatch: {path}")

    if "row_index" in df.columns:
        expected = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(df["row_index"].to_numpy(), expected):
            raise ValueError(f"row order mismatch: {path}")

    if "fold" in df.columns:
        if not np.array_equal(df["fold"].to_numpy(dtype=int), folds):
            raise ValueError(f"fold mismatch: {path}")

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise ValueError(f"id mismatch: {path}")

    p = df[pred_col(df)].to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise ValueError(f"non-finite predictions: {path}")

    return p


def fold_rank(v: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(v), dtype=float)

    for fold in range(5):
        mask = folds == fold
        out[mask] = (
            pd.Series(v[mask])
            .rank(method="average", pct=True)
            .to_numpy(dtype=float)
        )

    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def blend(
    champion_rank: np.ndarray,
    realmlp_rank: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (
        (1.0 - alpha) * champion_rank
        + alpha * realmlp_rank
    )


def main() -> None:
    print("=" * 96)
    print("REALMLP DIVERSITY -> CURRENT CHAMPION BLEND AUDIT")
    print("NO MODEL TRAINING | COARSE 0.05 WEIGHT GRID ONLY")
    print("=" * 96)

    for path in [
        TRAIN_PATH,
        FOLDS_PATH,
        CHAMPION_OOF,
        REALMLP_OOF,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256(FOLDS_PATH)
    if fold_hash != EXPECTED_FOLD_SHA:
        raise ValueError(
            f"Frozen fold SHA mismatch:\n"
            f"expected={EXPECTED_FOLD_SHA}\n"
            f"found={fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    fold_df = pd.read_csv(FOLDS_PATH)

    if len(fold_df) != len(train):
        raise ValueError("fold row count mismatch")

    folds = fold_df["fold"].to_numpy(dtype=int)

    y = (
        train[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(POSITIVE_LABEL.lower())
        .astype(np.int8)
        .to_numpy()
    )

    champion = load_oof(
        CHAMPION_OOF,
        train,
        folds,
    )
    realmlp = load_oof(
        REALMLP_OOF,
        train,
        folds,
    )

    champion_auc = auc(y, champion)
    realmlp_auc = auc(y, realmlp)

    if abs(champion_auc - EXPECTED_CHAMPION_AUC) > TOL:
        raise ValueError(
            f"Champion AUC mismatch: "
            f"{champion_auc:.8f} vs {EXPECTED_CHAMPION_AUC:.8f}"
        )

    if abs(realmlp_auc - EXPECTED_REALMLP_AUC) > TOL:
        raise ValueError(
            f"RealMLP AUC mismatch: "
            f"{realmlp_auc:.8f} vs {EXPECTED_REALMLP_AUC:.8f}"
        )

    champion_r = fold_rank(champion, folds)
    realmlp_r = fold_rank(realmlp, folds)

    raw_rank_corr = float(
        np.corrcoef(
            pd.Series(champion).rank(pct=True),
            pd.Series(realmlp).rank(pct=True),
        )[0, 1]
    )

    print(f"\nFrozen fold SHA256: {fold_hash}")
    print("[PASS] Frozen validation verified.")
    print()
    print("--- Member integrity ---")
    print(f"Champion OOF : {champion_auc:.8f}")
    print(f"RealMLP OOF  : {realmlp_auc:.8f}")
    print(f"Global rank corr: {raw_rank_corr:.6f}")
    print()
    print("Coarse RealMLP alpha grid:")
    print("  " + ", ".join(f"{a:.2f}" for a in ALPHAS))

    # Full-OOF diagnostic only (NOT used as final evidence).
    full_rows = []

    for alpha in ALPHAS:
        p = blend(
            champion_r,
            realmlp_r,
            float(alpha),
        )
        full_rows.append(
            {
                "realmlp_alpha": float(alpha),
                "champion_alpha": float(1.0 - alpha),
                "full_oof_auc_diagnostic": auc(y, p),
            }
        )

    full_df = pd.DataFrame(full_rows)

    print()
    print("--- Full-OOF diagnostic grid (descriptive only) ---")
    for r in full_df.itertuples():
        print(
            f"RealMLP={r.realmlp_alpha:.2f} | "
            f"AUC={r.full_oof_auc_diagnostic:.8f} | "
            f"delta={r.full_oof_auc_diagnostic - champion_auc:+.8f}"
        )

    # Leave-one-fold-out meta-CV.
    meta_pred = np.full(len(train), np.nan, dtype=float)
    fold_rows = []

    print()
    print("--- Leave-one-fold-out coarse blend selection ---")

    for held in range(5):
        fit = folds != held
        val = folds == held

        best_fit_auc = -np.inf
        best_alpha = None

        for alpha in ALPHAS:
            fit_pred = blend(
                champion_r[fit],
                realmlp_r[fit],
                float(alpha),
            )
            fit_auc = auc(
                y[fit],
                fit_pred,
            )

            if fit_auc > best_fit_auc + 1e-12:
                best_fit_auc = fit_auc
                best_alpha = float(alpha)

        held_pred = blend(
            champion_r[val],
            realmlp_r[val],
            best_alpha,
        )

        base_auc = auc(
            y[val],
            champion_r[val],
        )
        held_auc = auc(
            y[val],
            held_pred,
        )
        delta = held_auc - base_auc

        meta_pred[val] = held_pred

        fold_rows.append(
            {
                "held_fold": held,
                "selected_realmlp_alpha": best_alpha,
                "selected_champion_alpha": 1.0 - best_alpha,
                "fit_auc": best_fit_auc,
                "held_champion_auc": base_auc,
                "held_blend_auc": held_auc,
                "delta": delta,
            }
        )

        print(
            f"Fold {held}: "
            f"alpha={best_alpha:.2f} | "
            f"{base_auc:.8f} -> {held_auc:.8f} "
            f"({delta:+.8f})"
        )

    if np.isnan(meta_pred).any():
        raise RuntimeError("meta prediction contains NaNs")

    fold_results = pd.DataFrame(fold_rows)

    # Compare against the same fold-ranked champion representation.
    champion_meta_auc = auc(
        y,
        champion_r,
    )
    candidate_meta_auc = auc(
        y,
        meta_pred,
    )
    delta_meta = candidate_meta_auc - champion_meta_auc

    improved = int((fold_results["delta"] > 0).sum())
    worse = int((fold_results["delta"] < 0).sum())

    mean_alpha = float(
        fold_results["selected_realmlp_alpha"].mean()
    )

    print()
    print("=" * 96)
    print("FINAL REALMLP DIVERSITY AUDIT")
    print("=" * 96)
    print(f"Champion meta representation : {champion_meta_auc:.8f}")
    print(f"RealMLP blend meta-CV        : {candidate_meta_auc:.8f}")
    print(f"Delta                        : {delta_meta:+.8f}")
    print(f"Folds improved               : {improved}/5")
    print(f"Folds worse                  : {worse}/5")
    print(f"Mean selected RealMLP alpha  : {mean_alpha:.3f}")

    if delta_meta > 0 and improved >= 4:
        primitive = "POSITIVE_REALMLP_ENSEMBLE_SIGNAL"
    elif delta_meta < 0:
        primitive = "NEGATIVE_REALMLP_ENSEMBLE_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_REALMLP_ENSEMBLE_SIGNAL"

    print(f"Primitive                    : {primitive}")

    OUT.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(
        OUT / "coarse_full_oof_grid.csv",
        index=False,
    )

    fold_results.to_csv(
        OUT / "meta_fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": folds,
            "target_encoded": y,
            "champion_fold_rank": champion_r,
            "realmlp_fold_rank": realmlp_r,
            "meta_oof_prediction": meta_pred,
        }
    ).to_csv(
        OUT / "oof_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: REALMLP DIVERSITY BLEND AUDIT",
        "=" * 80,
        "",
        "NO TRAINING",
        "COARSE 0.05 WEIGHT GRID ONLY",
        "",
        f"Champion raw OOF: {champion_auc:.8f}",
        f"RealMLP raw OOF: {realmlp_auc:.8f}",
        f"Global rank corr: {raw_rank_corr:.6f}",
        "",
        f"Champion meta representation: {champion_meta_auc:.8f}",
        f"RealMLP blend meta-CV: {candidate_meta_auc:.8f}",
        f"Delta: {delta_meta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        f"Mean selected RealMLP alpha: {mean_alpha:.3f}",
        f"Primitive: {primitive}",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_results.itertuples():
        summary.append(
            f"Fold {r.held_fold}: "
            f"alpha={r.selected_realmlp_alpha:.2f} | "
            f"{r.held_champion_auc:.8f} -> {r.held_blend_auc:.8f} "
            f"({r.delta:+.8f})"
        )

    (OUT / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print(f"Saved outputs to: {OUT.relative_to(ROOT)}")
    print("Done. No models were trained.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 96)
        print("AUDIT FAILED")
        print("=" * 96)
        print(str(exc))
        sys.exit(1)
