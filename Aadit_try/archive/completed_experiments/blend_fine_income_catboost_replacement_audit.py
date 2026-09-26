from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# Fine-income CatBoost -> current fine-XGB champion replacement audit
#
# NO MODEL TRAINING.
# NO WEIGHT SEARCH.
#
# Current validated champion:
#   old CatBoost + fine-income XGB + old diverse LightGBM
#   OOF AUC = 0.94611253
#
# Candidate:
#   fine-income CatBoost + SAME fine-income XGB + SAME LightGBM
#   using the SAME already-validated fold-specific weights.
# ============================================================

ROOT = Path(__file__).resolve().parent

FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"
EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

TRAIN_PATH = ROOT / "data" / "train.csv"

OLD_CAT_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

FINE_CAT_OOF = (
    ROOT
    / "artifacts"
    / "experiments"
    / "catboost_fine_income_multiseed_gpu"
    / "best_average_oof_predictions.csv"
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

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_catboost_replacement_audit"
)

EXPECTED_AUC = {
    "OLD_CAT": 0.94558572,
    "FINE_CAT": 0.94578633,
    "FINE_XGB": 0.94606664,
    "LGBM": 0.94578042,
}

EXPECTED_CURRENT_CHAMPION_AUC = 0.94611253

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"
EXPECTED_N = 668_665
AUC_TOL = 5e-6
CHAMPION_TOL = 2e-6

# Exact fold-specific weights already validated for the current fine-XGB champion.
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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def fold_column(df: pd.DataFrame) -> str:
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
    fc = fold_column(folds_df)

    if "id" in train.columns and "id" in folds_df.columns:
        if train["id"].duplicated().any() or folds_df["id"].duplicated().any():
            raise RuntimeError("Duplicate id encountered while aligning folds.")
        mapping = folds_df.set_index("id")[fc]
        out = train["id"].map(mapping)
        if out.isna().any():
            raise RuntimeError("Frozen folds do not cover every train id.")
        return out.to_numpy(dtype=int)

    if len(folds_df) != len(train):
        raise RuntimeError("Frozen folds row count does not match train.")
    return folds_df[fc].to_numpy(dtype=int)


def detect_prediction_column(
    df: pd.DataFrame,
    preferred: list[str],
    path: Path,
) -> str:
    for c in preferred:
        if c in df.columns:
            return c

    exclude = {"id", "row_index", "fold", "target", "target_encoded"}
    candidates = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify prediction column in {path}. "
        f"Candidates={candidates}, columns={list(df.columns)}"
    )


def load_prediction(
    path: Path,
    train: pd.DataFrame,
    preferred: list[str],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise RuntimeError(
            f"Row mismatch for {path}: {len(df):,} vs {len(train):,}"
        )

    col = detect_prediction_column(df, preferred, path)

    if "id" in train.columns and "id" in df.columns:
        aligned = train[["id"]].merge(
            df[["id", col]],
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned[col].isna().any():
            raise RuntimeError(f"Could not align all ids in {path}")
        p = aligned[col].to_numpy(dtype=float)
    else:
        if "row_index" in df.columns:
            expected = np.arange(len(train), dtype=np.int64)
            if not np.array_equal(df["row_index"].to_numpy(), expected):
                raise RuntimeError(f"row_index mismatch in {path}")
        p = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise RuntimeError(f"Non-finite predictions in {path}")

    return p


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


def global_rank(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(x)
        .rank(method="average", pct=True)
        .to_numpy(dtype=float)
    )


def main() -> None:
    print("=" * 96)
    print("FINE-INCOME CATBOOST -> CURRENT CHAMPION REPLACEMENT AUDIT")
    print("NO TRAINING | NO WEIGHT SEARCH")
    print("=" * 96)

    if sha256_file(FOLDS_PATH) != EXPECTED_FOLDS_SHA256:
        raise RuntimeError("Frozen-fold SHA256 mismatch.")

    print(f"\nFrozen fold SHA256 verified: {EXPECTED_FOLDS_SHA256}")

    train = pd.read_csv(TRAIN_PATH)
    if len(train) != EXPECTED_N:
        raise RuntimeError(
            f"Unexpected train rows: {len(train):,}; expected {EXPECTED_N:,}"
        )

    if TARGET not in train.columns:
        raise RuntimeError(f"Missing target column {TARGET}")

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
        raise RuntimeError(f"Unexpected frozen folds: {sorted(np.unique(folds))}")

    specs = {
        "OLD_CAT": (
            OLD_CAT_OOF,
            ["oof_prediction", "best_average_prediction", "prediction"],
        ),
        "FINE_CAT": (
            FINE_CAT_OOF,
            ["oof_prediction", "best_average_prediction", "prediction"],
        ),
        "FINE_XGB": (
            FINE_XGB_OOF,
            ["oof_prediction", "prediction"],
        ),
        "LGBM": (
            LGBM_OOF,
            ["oof_prediction", "prediction"],
        ),
    }

    pred = {}

    print("\n--- OOF integrity ---")
    for name, (path, preferred) in specs.items():
        p = load_prediction(path, train, preferred)
        score = auc(y, p)

        if abs(score - EXPECTED_AUC[name]) > AUC_TOL:
            raise RuntimeError(
                f"{name} AUC mismatch: got {score:.8f}, "
                f"expected {EXPECTED_AUC[name]:.8f}"
            )

        pred[name] = p
        print(f"{name:9s} AUC={score:.8f}")

    print("[PASS] All four validated members loaded.")

    old_cat_r = foldwise_rank(pred["OLD_CAT"], folds)
    fine_cat_r = foldwise_rank(pred["FINE_CAT"], folds)
    xgb_r = foldwise_rank(pred["FINE_XGB"], folds)
    lgbm_r = foldwise_rank(pred["LGBM"], folds)

    control = np.empty(len(train), dtype=float)
    candidate = np.empty(len(train), dtype=float)

    fold_rows = []

    print("\n--- Same-weight CatBoost replacement by fold ---")
    for fold in range(5):
        mask = folds == fold
        wc, wx, wl = FOLD_WEIGHTS[fold]

        control[mask] = (
            wc * old_cat_r[mask]
            + wx * xgb_r[mask]
            + wl * lgbm_r[mask]
        )

        candidate[mask] = (
            wc * fine_cat_r[mask]
            + wx * xgb_r[mask]
            + wl * lgbm_r[mask]
        )

        control_auc = auc(y[mask], control[mask])
        candidate_auc = auc(y[mask], candidate[mask])
        delta = candidate_auc - control_auc

        fold_rows.append(
            {
                "fold": fold,
                "cat_weight": wc,
                "xgb_weight": wx,
                "lgbm_weight": wl,
                "control_auc": control_auc,
                "candidate_auc": candidate_auc,
                "delta": delta,
            }
        )

        print(
            f"Fold {fold}: "
            f"{control_auc:.8f} -> {candidate_auc:.8f} "
            f"({delta:+.8f}) | "
            f"weights={wc:.2f}/{wx:.2f}/{wl:.2f}"
        )

    fold_results = pd.DataFrame(fold_rows)

    control_auc = auc(y, control)
    candidate_auc = auc(y, candidate)
    delta = candidate_auc - control_auc

    if abs(control_auc - EXPECTED_CURRENT_CHAMPION_AUC) > CHAMPION_TOL:
        raise RuntimeError(
            "Current champion reproduction mismatch.\n"
            f"Expected: {EXPECTED_CURRENT_CHAMPION_AUC:.8f}\n"
            f"Found:    {control_auc:.8f}"
        )

    improved = int((fold_results["delta"] > 0).sum())
    worse = int((fold_results["delta"] < 0).sum())

    rank_corr_vs_control = float(
        np.corrcoef(global_rank(control), global_rank(candidate))[0, 1]
    )

    print("\n" + "=" * 96)
    print("FINAL REPLACEMENT RESULT")
    print("=" * 96)
    print(f"Current champion        : {control_auc:.8f}")
    print(f"Fine-Cat replacement    : {candidate_auc:.8f}")
    print(f"Delta                   : {delta:+.8f}")
    print(f"Folds improved          : {improved}/5")
    print(f"Folds worse             : {worse}/5")
    print(f"Rank corr vs champion   : {rank_corr_vs_control:.6f}")

    if delta > 0 and improved >= 4:
        primitive = "POSITIVE_FINE_CAT_REPLACEMENT_SIGNAL"
    elif delta < 0:
        primitive = "NEGATIVE_FINE_CAT_REPLACEMENT_SIGNAL"
    else:
        primitive = "WEAK_OR_INCONSISTENT_FINE_CAT_REPLACEMENT_SIGNAL"

    print(f"Primitive               : {primitive}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_results.to_csv(
        OUTPUT_DIR / "fold_results.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": folds,
            "target_encoded": y,
            "control_oof_prediction": control,
            "candidate_oof_prediction": candidate,
        }
    ).to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "control_auc": control_auc,
                "candidate_auc": candidate_auc,
                "delta": delta,
                "folds_improved": improved,
                "folds_worse": worse,
                "rank_corr_vs_control": rank_corr_vs_control,
                "primitive": primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    print(
        f"\nSaved outputs to: "
        f"{OUTPUT_DIR.relative_to(ROOT)}"
    )
    print("Done. No models were trained and no blend weights were tuned.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 96)
        print("AUDIT FAILED")
        print("=" * 96)
        print(str(exc))
        sys.exit(1)
