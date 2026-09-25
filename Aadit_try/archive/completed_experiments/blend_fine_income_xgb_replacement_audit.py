from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# Fine-income XGB -> current champion replacement audit
# NO MODEL TRAINING. NO BLEND-WEIGHT TUNING.
# ============================================================

ROOT = Path(__file__).resolve().parent

FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"
EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

TRAIN_PATH = ROOT / "data" / "train.csv"

CAT_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "rank_average_oof_predictions.csv"
)

OLD_XGB_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu"
    / "oof_predictions.csv"
)

FINE_XGB_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "xgboost_fine_income_te_gpu"
    / "oof_predictions.csv"
)

LGBM_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

CHAMPION_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_income_commute_catboost_commute_xgb_lgbm_rank_audit"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_replacement_audit"
)

EXPECTED_AUC = {
    "CAT": 0.94558572,
    "OLD_XGB": 0.94591178,
    "FINE_XGB": 0.94606664,
    "LGBM": 0.94578042,
    "CHAMPION": 0.94596441,
}

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"
EXPECTED_N = 668_665
AUC_TOL = 5e-6

# We are reconstructing an already-saved blend artifact, not fitting a new
# predictive stacker. Require a close numeric reconstruction before trusting
# the replacement comparison.
MAX_RECON_RMSE = 2e-4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def detect_fold_column(df: pd.DataFrame) -> str:
    exact = [c for c in df.columns if c.lower() == "fold"]
    if len(exact) == 1:
        return exact[0]

    fuzzy = [c for c in df.columns if "fold" in c.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise RuntimeError(
        f"Could not uniquely identify fold column. Columns={list(df.columns)}"
    )


def align_folds(train: pd.DataFrame, folds_df: pd.DataFrame) -> np.ndarray:
    fold_col = detect_fold_column(folds_df)

    if "id" in train.columns and "id" in folds_df.columns:
        if train["id"].duplicated().any() or folds_df["id"].duplicated().any():
            raise RuntimeError("Duplicate ids while aligning frozen folds.")
        mapping = folds_df.set_index("id")[fold_col]
        out = train["id"].map(mapping)
        if out.isna().any():
            raise RuntimeError("Frozen fold file does not cover all train ids.")
        return out.to_numpy(dtype=int)

    if len(train) != len(folds_df):
        raise RuntimeError("Frozen folds cannot be aligned by row order.")
    return folds_df[fold_col].to_numpy(dtype=int)


def detect_prediction_column(
    df: pd.DataFrame,
    preferred: list[str],
    path: Path,
) -> str:
    for c in preferred:
        if c in df.columns:
            return c

    numeric_candidates = []
    for c in df.columns:
        if c.lower() in {"id", "row_index", "fold", "target", "target_encoded"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().all() and s.nunique() > 100:
            v = s.to_numpy(dtype=float)
            if np.isfinite(v).all() and v.min() >= -1e-9 and v.max() <= 1 + 1e-9:
                numeric_candidates.append(c)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify prediction column in {path}. "
        f"Candidates={numeric_candidates}, columns={list(df.columns)}"
    )


def load_prediction(
    path: Path,
    train: pd.DataFrame,
    preferred_columns: list[str],
) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if len(df) != len(train):
        raise RuntimeError(
            f"Row count mismatch for {path}: {len(df):,} vs {len(train):,}"
        )

    pred_col = detect_prediction_column(df, preferred_columns, path)

    if "id" in train.columns and "id" in df.columns:
        if train["id"].duplicated().any() or df["id"].duplicated().any():
            raise RuntimeError(f"Duplicate ids in {path}")

        aligned = train[["id"]].merge(
            df[["id", pred_col]],
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned[pred_col].isna().any():
            raise RuntimeError(f"Could not align all train ids in {path}")
        p = aligned[pred_col].to_numpy(dtype=float)
    else:
        p = pd.to_numeric(df[pred_col], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise RuntimeError(f"Non-finite predictions in {path}")

    return p, pred_col


def global_rank(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(x)
        .rank(method="average", pct=True)
        .to_numpy(dtype=float)
    )


def foldwise_rank(x: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=float)
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        out[mask] = (
            pd.Series(x[mask])
            .rank(method="average", pct=True)
            .to_numpy(dtype=float)
        )
    return out


def solve_sum1_weights(
    cat: np.ndarray,
    xgb: np.ndarray,
    lgbm: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve target ~= wc*cat + wx*xgb + wl*lgbm subject to wc+wx+wl=1.

    This is artifact reconstruction only. No labels are used.
    """
    A = np.column_stack([cat - lgbm, xgb - lgbm])
    b = target - lgbm

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    wc, wx = coef
    wl = 1.0 - wc - wx

    weights = np.array([wc, wx, wl], dtype=float)
    recon = wc * cat + wx * xgb + wl * lgbm
    return weights, recon


def reconstruct_mode(
    mode_name: str,
    cat: np.ndarray,
    old_xgb: np.ndarray,
    fine_xgb: np.ndarray,
    lgbm: np.ndarray,
    champion: np.ndarray,
    folds: np.ndarray,
) -> dict:
    if mode_name == "global_rank":
        cat_t = global_rank(cat)
        old_xgb_t = global_rank(old_xgb)
        fine_xgb_t = global_rank(fine_xgb)
        lgbm_t = global_rank(lgbm)

    elif mode_name == "foldwise_rank":
        cat_t = foldwise_rank(cat, folds)
        old_xgb_t = foldwise_rank(old_xgb, folds)
        fine_xgb_t = foldwise_rank(fine_xgb, folds)
        lgbm_t = foldwise_rank(lgbm, folds)

    elif mode_name == "raw":
        cat_t = cat
        old_xgb_t = old_xgb
        fine_xgb_t = fine_xgb
        lgbm_t = lgbm

    else:
        raise ValueError(mode_name)

    reconstruction = np.full(len(folds), np.nan, dtype=float)
    replacement = np.full(len(folds), np.nan, dtype=float)
    weight_rows = []

    for fold in sorted(np.unique(folds)):
        mask = folds == fold

        w, recon = solve_sum1_weights(
            cat_t[mask],
            old_xgb_t[mask],
            lgbm_t[mask],
            champion[mask],
        )

        replacement_fold = (
            w[0] * cat_t[mask]
            + w[1] * fine_xgb_t[mask]
            + w[2] * lgbm_t[mask]
        )

        reconstruction[mask] = recon
        replacement[mask] = replacement_fold

        rmse = float(np.sqrt(np.mean((recon - champion[mask]) ** 2)))
        mae = float(np.mean(np.abs(recon - champion[mask])))

        weight_rows.append(
            {
                "mode": mode_name,
                "fold": int(fold),
                "cat_weight": float(w[0]),
                "xgb_weight": float(w[1]),
                "lgbm_weight": float(w[2]),
                "reconstruction_rmse": rmse,
                "reconstruction_mae": mae,
            }
        )

    overall_rmse = float(
        np.sqrt(np.mean((reconstruction - champion) ** 2))
    )
    overall_mae = float(np.mean(np.abs(reconstruction - champion)))

    return {
        "mode": mode_name,
        "cat": cat_t,
        "old_xgb": old_xgb_t,
        "fine_xgb": fine_xgb_t,
        "lgbm": lgbm_t,
        "reconstruction": reconstruction,
        "replacement": replacement,
        "weights": pd.DataFrame(weight_rows),
        "rmse": overall_rmse,
        "mae": overall_mae,
    }


def main() -> None:
    print("=" * 92)
    print("FINE-INCOME XGB -> CURRENT CHAMPION REPLACEMENT AUDIT")
    print("NO MODEL TRAINING | NO BLEND-WEIGHT TUNING")
    print("=" * 92)

    required = [
        TRAIN_PATH,
        FOLDS_PATH,
        CAT_OOF,
        OLD_XGB_OOF,
        FINE_XGB_OOF,
        LGBM_OOF,
        CHAMPION_OOF,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    fold_hash = sha256_file(FOLDS_PATH)
    print(f"\nFrozen fold SHA256: {fold_hash}")
    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen fold hash mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Actual:   {fold_hash}"
        )
    print("[PASS] Frozen validation hash verified.")

    train = pd.read_csv(TRAIN_PATH)
    if len(train) != EXPECTED_N:
        raise RuntimeError(
            f"Unexpected train size {len(train):,}; expected {EXPECTED_N:,}"
        )
    if TARGET not in train.columns:
        raise RuntimeError(f"Missing target {TARGET}")

    y = (
        train[TARGET]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(np.int8)
        .to_numpy()
    )

    folds_df = pd.read_csv(FOLDS_PATH)
    folds = align_folds(train, folds_df)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected folds: {sorted(np.unique(folds).tolist())}"
        )

    specs = {
        "CAT": (
            CAT_OOF,
            ["oof_prediction", "rank_average_oof_prediction"],
        ),
        "OLD_XGB": (
            OLD_XGB_OOF,
            ["oof_prediction"],
        ),
        "FINE_XGB": (
            FINE_XGB_OOF,
            ["oof_prediction"],
        ),
        "LGBM": (
            LGBM_OOF,
            ["oof_prediction"],
        ),
        "CHAMPION": (
            CHAMPION_OOF,
            [
                "candidate_meta_oof_prediction",
                "oof_prediction",
                "prediction",
            ],
        ),
    }

    pred = {}
    loaded_rows = []

    print("\n--- OOF integrity ---")
    for name, (path, cols) in specs.items():
        p, col = load_prediction(path, train, cols)
        score = auc(y, p)

        if abs(score - EXPECTED_AUC[name]) > AUC_TOL:
            raise RuntimeError(
                f"{name} OOF integrity mismatch: "
                f"got {score:.8f}, expected {EXPECTED_AUC[name]:.8f}"
            )

        pred[name] = p
        loaded_rows.append(
            {
                "model": name,
                "auc": score,
                "expected_auc": EXPECTED_AUC[name],
                "path": str(path.relative_to(ROOT)),
                "column": col,
            }
        )
        print(
            f"{name:9s} AUC={score:.8f} | "
            f"{path.relative_to(ROOT)} | {col}"
        )

    print("[PASS] All required OOF artifacts match validated scores.")

    # ------------------------------------------------------------
    # Reconstruct the existing champion WITHOUT labels.
    # Try the plausible representations used by the historical rank audit.
    # The representation with the smallest saved-artifact reconstruction
    # error is used. This is NOT model/weight optimization.
    # ------------------------------------------------------------
    modes = []
    for mode_name in ("global_rank", "foldwise_rank", "raw"):
        result = reconstruct_mode(
            mode_name,
            pred["CAT"],
            pred["OLD_XGB"],
            pred["FINE_XGB"],
            pred["LGBM"],
            pred["CHAMPION"],
            folds,
        )
        modes.append(result)

    modes.sort(key=lambda d: d["rmse"])
    chosen = modes[0]

    print("\n--- Champion artifact reconstruction ---")
    for m in modes:
        print(
            f"{m['mode']:14s} "
            f"RMSE={m['rmse']:.10f} "
            f"MAE={m['mae']:.10f}"
        )

    print(f"\nChosen reconstruction mode: {chosen['mode']}")

    if chosen["rmse"] > MAX_RECON_RMSE:
        raise RuntimeError(
            "Could not reconstruct the current champion closely enough from "
            "the three confirmed members.\n"
            f"Best mode: {chosen['mode']}\n"
            f"RMSE: {chosen['rmse']:.10f}\n"
            f"Allowed: {MAX_RECON_RMSE:.10f}\n"
            "Do NOT trust a member-replacement result from a mismatched blend."
        )

    print("[PASS] Current champion blend reconstructed closely enough.")

    weights = chosen["weights"].copy()
    print("\nRecovered fold-specific weights:")
    print(
        weights[
            [
                "fold",
                "cat_weight",
                "xgb_weight",
                "lgbm_weight",
                "reconstruction_rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print(
        "\nMean recovered weights: "
        f"CAT={weights['cat_weight'].mean():.4f}, "
        f"XGB={weights['xgb_weight'].mean():.4f}, "
        f"LGBM={weights['lgbm_weight'].mean():.4f}"
    )

    reconstructed_auc = auc(y, chosen["reconstruction"])
    saved_champion_auc = auc(y, pred["CHAMPION"])
    candidate_auc = auc(y, chosen["replacement"])

    if abs(reconstructed_auc - saved_champion_auc) > 5e-5:
        raise RuntimeError(
            "Reconstructed champion AUC is too far from saved champion AUC.\n"
            f"Saved:         {saved_champion_auc:.8f}\n"
            f"Reconstructed:{reconstructed_auc:.8f}"
        )

    # ------------------------------------------------------------
    # Fold-by-fold member replacement:
    # same CAT, same LGBM, same recovered fold weight;
    # ONLY old XGB rank/probability stream -> fine-income XGB stream.
    # ------------------------------------------------------------
    fold_rows = []

    print("\n--- Same-weight XGB replacement by fold ---")
    for fold in sorted(np.unique(folds)):
        mask = folds == fold

        baseline_fold_auc = auc(
            y[mask],
            pred["CHAMPION"][mask],
        )
        candidate_fold_auc = auc(
            y[mask],
            chosen["replacement"][mask],
        )
        delta = candidate_fold_auc - baseline_fold_auc

        fold_rows.append(
            {
                "fold": int(fold),
                "baseline_champion_auc": baseline_fold_auc,
                "fine_xgb_replacement_auc": candidate_fold_auc,
                "delta": delta,
            }
        )

        print(
            f"Fold {fold}: "
            f"{baseline_fold_auc:.8f} -> "
            f"{candidate_fold_auc:.8f} "
            f"({delta:+.8f})"
        )

    fold_results = pd.DataFrame(fold_rows)

    delta = candidate_auc - saved_champion_auc
    folds_improved = int((fold_results["delta"] > 0).sum())
    folds_worse = int((fold_results["delta"] < 0).sum())

    probability_corr = float(
        np.corrcoef(
            chosen["replacement"],
            pred["CHAMPION"],
        )[0, 1]
    )

    rank_corr = float(
        np.corrcoef(
            global_rank(chosen["replacement"]),
            global_rank(pred["CHAMPION"]),
        )[0, 1]
    )

    print("\n" + "=" * 92)
    print("FINAL MEMBER-REPLACEMENT RESULT")
    print("=" * 92)
    print(f"Saved current champion : {saved_champion_auc:.8f}")
    print(f"Fine-XGB replacement   : {candidate_auc:.8f}")
    print(f"Delta                  : {delta:+.8f}")
    print(f"Folds improved         : {folds_improved}/5")
    print(f"Folds worse            : {folds_worse}/5")
    print(f"Probability corr       : {probability_corr:.6f}")
    print(f"Rank corr              : {rank_corr:.6f}")

    if delta > 0 and folds_improved >= 4:
        primitive = "POSITIVE_REPLACEMENT_SIGNAL"
    elif delta < 0:
        primitive = "NEGATIVE_REPLACEMENT_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_REPLACEMENT_SIGNAL"

    print(f"Primitive              : {primitive}")
    print(
        "\nNOTE: The primitive is not the final KEEP/REJECT judgment. "
        "Magnitude and fold consistency still require manual review."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(loaded_rows).to_csv(
        OUTPUT_DIR / "loaded_oof_artifacts.csv",
        index=False,
    )

    weights.to_csv(
        OUTPUT_DIR / "recovered_fold_weights.csv",
        index=False,
    )

    fold_results.to_csv(
        OUTPUT_DIR / "fold_results.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "reconstruction_mode": chosen["mode"],
                "reconstruction_rmse": chosen["rmse"],
                "reconstruction_mae": chosen["mae"],
                "saved_champion_auc": saved_champion_auc,
                "reconstructed_champion_auc": reconstructed_auc,
                "fine_xgb_replacement_auc": candidate_auc,
                "delta": delta,
                "folds_improved": folds_improved,
                "folds_worse": folds_worse,
                "probability_corr": probability_corr,
                "rank_corr": rank_corr,
                "primitive": primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": folds,
            "target": y,
            "saved_champion_prediction": pred["CHAMPION"],
            "reconstructed_champion_prediction": chosen["reconstruction"],
            "fine_xgb_replacement_prediction": chosen["replacement"],
        }
    ).to_csv(
        OUTPUT_DIR / "replacement_oof_predictions.csv",
        index=False,
    )

    print(
        f"\nSaved outputs to: {OUTPUT_DIR.relative_to(ROOT)}"
    )
    print("Done. No models were trained and no blend weights were tuned.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 92)
        print("AUDIT FAILED")
        print("=" * 92)
        print(str(exc))
        sys.exit(1)
