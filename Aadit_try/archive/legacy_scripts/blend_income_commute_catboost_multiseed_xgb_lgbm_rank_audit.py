"""
blend_income_commute_catboost_multiseed_xgb_lgbm_rank_audit.py

Controlled ensemble audit.

CONTROL
-------
hierarchical-income+commute CatBoost 3-seed
+ hierarchical-income+commute XGBoost seed 42
+ old engineered LightGBM

CANDIDATE
---------
hierarchical-income+commute CatBoost 3-seed
+ hierarchical-income+commute XGBoost 3-seed
+ old engineered LightGBM

ONLY CHANGE
-----------
XGBoost seed42 -> XGBoost 3-seed rank average.

Everything else stays fixed:
- frozen folds + SHA256
- CatBoost member
- LightGBM member
- fold-local percentile-rank blending
- coarse 0.05 simplex grid
- weights selected on 4 folds, evaluated on held fold
- no model retraining
- no leaderboard optimization

Current control meta-CV:
    0.94596441

Run:
    python blend_income_commute_catboost_multiseed_xgb_lgbm_rank_audit.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_HASH = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_CAT_AUC = 0.94558572
EXPECTED_OLD_XGB_AUC = 0.94591178
EXPECTED_NEW_XGB_AUC = 0.94592339
EXPECTED_LGBM_AUC = 0.94578042

CONTROL_META_REFERENCE = 0.94596441

AUC_TOL = 2e-5
CONTROL_TOL = 2e-5
STEP = 0.05

FOLDS_PATH = Path("artifacts/validation/candidate_folds.csv")

CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)
CAT_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_test_predictions.csv"
)

OLD_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu"
    / "oof_predictions.csv"
)
OLD_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu"
    / "test_predictions.csv"
)

NEW_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)
NEW_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_test_predictions.csv"
)

LGBM_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)
LGBM_TEST = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "test_predictions.csv"
)

OUT = (
    Path("artifacts")
    / "experiments"
    / "blend_income_commute_catboost_multiseed_xgb_lgbm_rank_audit"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    cols = [c for c in train.columns if c not in test.columns]
    if len(cols) != 1:
        raise ValueError(f"Target detection failed: {cols}")
    return cols[0]


def encode_target(y: pd.Series) -> np.ndarray:
    vals = y.astype(str).str.strip().str.lower()
    if set(vals.unique()) == {"yes", "no"}:
        return (vals == "yes").astype(np.int8).to_numpy()
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("Expected binary target.")
    positive = counts.idxmin()
    return (y == positive).astype(np.int8).to_numpy()


def pick_prediction_column(df: pd.DataFrame, kind: str) -> str:
    preferred = (
        ["oof_prediction", "prediction", "rank_average_prediction", "best_average_prediction"]
        if kind == "oof"
        else ["prediction", "test_prediction", "rank_average_prediction", "best_average_prediction"]
    )

    for c in preferred:
        if c in df.columns:
            return c

    excluded = {"row_index", "fold", "target", "target_encoded", "id"}
    numeric = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) != 1:
        raise ValueError(
            f"Could not safely identify {kind} prediction column: {list(df.columns)}"
        )

    return numeric[0]


def load_oof(
    path: Path,
    n_rows: int,
    fold_ids: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path.resolve())

    df = pd.read_csv(path)

    if len(df) != n_rows:
        raise ValueError(f"OOF row mismatch: {path}")

    if "row_index" not in df.columns:
        raise ValueError(f"Missing row_index: {path}")

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(n_rows, dtype=np.int64),
    ):
        raise ValueError(f"OOF row-order mismatch: {path}")

    if "fold" not in df.columns:
        raise ValueError(f"Missing fold: {path}")

    if not np.array_equal(df["fold"].to_numpy(), fold_ids):
        raise ValueError(f"OOF fold mismatch: {path}")

    pred = df[pick_prediction_column(df, "oof")].to_numpy(np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite OOF predictions: {path}")

    return pred


def load_test(path: Path, n_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path.resolve())

    df = pd.read_csv(path)

    if len(df) != n_rows:
        raise ValueError(f"Test row mismatch: {path}")

    pred = df[pick_prediction_column(df, "test")].to_numpy(np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite test predictions: {path}")

    return pred


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(np.float64)
    )


def foldwise_rank(
    values: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)

    for fold in range(5):
        mask = fold_ids == fold
        out[mask] = percentile_rank(values[mask])

    return out


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.corrcoef(
            percentile_rank(a),
            percentile_rank(b),
        )[0, 1]
    )


def simplex_grid() -> list[tuple[float, float, float]]:
    units = int(round(1.0 / STEP))
    grid = []

    for lgb_units in range(units + 1):
        for cat_units in range(units - lgb_units + 1):
            xgb_units = units - lgb_units - cat_units

            grid.append(
                (
                    cat_units / units,
                    xgb_units / units,
                    lgb_units / units,
                )
            )

    return grid


def choose_weights(
    y: np.ndarray,
    cat_rank: np.ndarray,
    xgb_rank: np.ndarray,
    lgbm_rank: np.ndarray,
    fit_mask: np.ndarray,
    grid: list[tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    best_auc = -np.inf
    best_weights = (0.0, 1.0, 0.0)

    for wc, wx, wl in grid:
        pred = (
            wc * cat_rank[fit_mask]
            + wx * xgb_rank[fit_mask]
            + wl * lgbm_rank[fit_mask]
        )

        auc = float(
            roc_auc_score(
                y[fit_mask],
                pred,
            )
        )

        if auc > best_auc + 1e-12:
            best_auc = auc
            best_weights = (wc, wx, wl)

    return (*best_weights, best_auc)


def verify_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    auc = float(roc_auc_score(y, pred))

    if abs(auc - expected) > AUC_TOL:
        raise ValueError(
            f"{name} AUC mismatch: loaded={auc:.8f}, expected~={expected:.8f}"
        )

    return auc


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    if sha256_file(FOLDS_PATH) != EXPECTED_HASH:
        raise ValueError("Frozen fold SHA256 mismatch.")

    fold_df = pd.read_csv(FOLDS_PATH)

    if len(fold_df) != len(train):
        raise ValueError("Frozen fold row count mismatch.")

    fold_ids = fold_df["fold"].to_numpy(np.int16)

    target = detect_target(train, test)
    y = encode_target(train[target])

    cat_oof = load_oof(
        CAT_OOF,
        len(train),
        fold_ids,
    )
    old_xgb_oof = load_oof(
        OLD_XGB_OOF,
        len(train),
        fold_ids,
    )
    new_xgb_oof = load_oof(
        NEW_XGB_OOF,
        len(train),
        fold_ids,
    )
    lgbm_oof = load_oof(
        LGBM_OOF,
        len(train),
        fold_ids,
    )

    cat_auc = verify_auc(
        "CatBoost",
        y,
        cat_oof,
        EXPECTED_CAT_AUC,
    )
    old_xgb_auc = verify_auc(
        "Seed42 XGB",
        y,
        old_xgb_oof,
        EXPECTED_OLD_XGB_AUC,
    )
    new_xgb_auc = verify_auc(
        "3-seed XGB",
        y,
        new_xgb_oof,
        EXPECTED_NEW_XGB_AUC,
    )
    lgbm_auc = verify_auc(
        "LightGBM",
        y,
        lgbm_oof,
        EXPECTED_LGBM_AUC,
    )

    cat_rank = foldwise_rank(cat_oof, fold_ids)
    old_xgb_rank = foldwise_rank(old_xgb_oof, fold_ids)
    new_xgb_rank = foldwise_rank(new_xgb_oof, fold_ids)
    lgbm_rank = foldwise_rank(lgbm_oof, fold_ids)

    grid = simplex_grid()

    print("=" * 102)
    print("ENSEMBLE AUDIT: XGBOOST SEED42 -> XGBOOST 3-SEED")
    print("=" * 102)
    print(f"Frozen fold SHA256 verified: {EXPECTED_HASH}")
    print(f"CatBoost            : {cat_auc:.8f}")
    print(f"XGB seed42          : {old_xgb_auc:.8f}")
    print(f"XGB 3-seed          : {new_xgb_auc:.8f}")
    print(f"LightGBM            : {lgbm_auc:.8f}")
    print(
        f"old XGB vs new XGB rank corr: "
        f"{rank_corr(old_xgb_oof, new_xgb_oof):.6f}"
    )
    print()

    control_oof = np.full(len(train), np.nan)
    candidate_oof = np.full(len(train), np.nan)

    rows = []
    start = time.perf_counter()

    for held_fold in range(5):
        fit_mask = fold_ids != held_fold
        held_mask = fold_ids == held_fold

        (
            cc,
            cx,
            cl,
            control_fit_auc,
        ) = choose_weights(
            y,
            cat_rank,
            old_xgb_rank,
            lgbm_rank,
            fit_mask,
            grid,
        )

        (
            nc,
            nx,
            nl,
            candidate_fit_auc,
        ) = choose_weights(
            y,
            cat_rank,
            new_xgb_rank,
            lgbm_rank,
            fit_mask,
            grid,
        )

        control_pred = (
            cc * cat_rank[held_mask]
            + cx * old_xgb_rank[held_mask]
            + cl * lgbm_rank[held_mask]
        )

        candidate_pred = (
            nc * cat_rank[held_mask]
            + nx * new_xgb_rank[held_mask]
            + nl * lgbm_rank[held_mask]
        )

        control_auc = float(
            roc_auc_score(
                y[held_mask],
                control_pred,
            )
        )

        candidate_auc = float(
            roc_auc_score(
                y[held_mask],
                candidate_pred,
            )
        )

        delta = candidate_auc - control_auc

        control_oof[held_mask] = control_pred
        candidate_oof[held_mask] = candidate_pred

        rows.append(
            {
                "held_fold": held_fold,
                "control_cat_weight": cc,
                "control_old_xgb_weight": cx,
                "control_lgbm_weight": cl,
                "control_fit_auc": control_fit_auc,
                "control_held_auc": control_auc,
                "candidate_cat_weight": nc,
                "candidate_new_xgb_weight": nx,
                "candidate_lgbm_weight": nl,
                "candidate_fit_auc": candidate_fit_auc,
                "candidate_held_auc": candidate_auc,
                "candidate_delta_vs_control": delta,
            }
        )

        print(
            f"Fold {held_fold}: "
            f"control={control_auc:.8f} "
            f"({cc:.2f}/{cx:.2f}/{cl:.2f}) -> "
            f"candidate={candidate_auc:.8f} "
            f"({nc:.2f}/{nx:.2f}/{nl:.2f}) "
            f"{delta:+.8f}"
        )

    runtime = time.perf_counter() - start

    fm = pd.DataFrame(rows)

    control_meta = float(
        roc_auc_score(
            y,
            control_oof,
        )
    )

    candidate_meta = float(
        roc_auc_score(
            y,
            candidate_oof,
        )
    )

    if abs(
        control_meta
        - CONTROL_META_REFERENCE
    ) > CONTROL_TOL:
        raise ValueError(
            f"Control reproduction mismatch: "
            f"{control_meta:.8f} vs {CONTROL_META_REFERENCE:.8f}"
        )

    delta_meta = candidate_meta - control_meta

    improved = int(
        (
            fm[
                "candidate_delta_vs_control"
            ] > 0
        ).sum()
    )

    worse = int(
        (
            fm[
                "candidate_delta_vs_control"
            ] < 0
        ).sum()
    )

    candidate_weights = np.array(
        [
            fm["candidate_cat_weight"].mean(),
            fm["candidate_new_xgb_weight"].mean(),
            fm["candidate_lgbm_weight"].mean(),
        ],
        dtype=np.float64,
    )

    candidate_weights /= candidate_weights.sum()

    decision = (
        "KEEP_XGB_3SEED_IN_3MODEL_ENSEMBLE"
        if (
            delta_meta > 0
            and improved >= 4
        )
        else
        "REJECT_XGB_3SEED_ENSEMBLE_UPDATE"
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fm.to_csv(
        OUT / "meta_fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "control_meta_oof_prediction": control_oof,
            "candidate_meta_oof_prediction": candidate_oof,
        }
    ).to_csv(
        OUT / "oof_predictions.csv",
        index=False,
    )

    cat_test = percentile_rank(
        load_test(
            CAT_TEST,
            len(test),
        )
    )
    new_xgb_test = percentile_rank(
        load_test(
            NEW_XGB_TEST,
            len(test),
        )
    )
    lgbm_test = percentile_rank(
        load_test(
            LGBM_TEST,
            len(test),
        )
    )

    test_pred = (
        candidate_weights[0] * cat_test
        + candidate_weights[1] * new_xgb_test
        + candidate_weights[2] * lgbm_test
    )

    test_out = pd.DataFrame(
        {
            "prediction": test_pred.astype(
                np.float32
            )
        }
    )

    if "id" in test.columns:
        test_out.insert(
            0,
            "id",
            test["id"].to_numpy(),
        )

    test_out.to_csv(
        OUT / "test_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: XGBOOST 3-SEED ENSEMBLE UPDATE",
        "=" * 80,
        f"Control meta-CV: {control_meta:.8f}",
        f"Candidate meta-CV: {candidate_meta:.8f}",
        f"Delta: {delta_meta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        (
            "Mean weights: "
            f"CAT={candidate_weights[0]:.4f}, "
            f"XGB={candidate_weights[1]:.4f}, "
            f"LGBM={candidate_weights[2]:.4f}"
        ),
        f"Runtime: {runtime:.2f}s",
        f"Decision: {decision}",
        "",
        "FOLD DETAILS",
    ]

    for row in fm.itertuples():
        summary.append(
            f"Fold {row.held_fold}: "
            f"{row.control_held_auc:.8f} -> "
            f"{row.candidate_held_auc:.8f} "
            f"({row.candidate_delta_vs_control:+.8f})"
        )

    (
        OUT / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 102)
    print("XGBOOST 3-SEED ENSEMBLE AUDIT COMPLETE")
    print("=" * 102)
    print(f"Control current 3-model : {control_meta:.8f}")
    print(f"Candidate updated model : {candidate_meta:.8f}")
    print(f"Delta vs control        : {delta_meta:+.8f}")
    print(f"Folds improved/worse    : {improved}/{worse}")
    print(
        "Mean weights            : "
        f"CAT={candidate_weights[0]:.4f}, "
        f"XGB={candidate_weights[1]:.4f}, "
        f"LGBM={candidate_weights[2]:.4f}"
    )
    print(f"Decision                : {decision}")
    print(f"Artifacts               : {OUT.resolve()}")
    print("=" * 102)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. meta_fold_metrics.csv")


if __name__ == "__main__":
    main()
