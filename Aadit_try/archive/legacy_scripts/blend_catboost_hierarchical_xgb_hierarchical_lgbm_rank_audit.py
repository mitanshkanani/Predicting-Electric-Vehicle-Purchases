"""
blend_catboost_hierarchical_xgb_hierarchical_lgbm_rank_audit.py

Kaggle Playground Series S6E9

Controlled ensemble update:
Replace ONLY the old engineered LightGBM member in the current validated
3-model ensemble with the new hierarchical-income LightGBM.

CURRENT CONTROL
---------------
CatBoost + hierarchical-income XGBoost + OLD engineered LightGBM

CANDIDATE
---------
CatBoost + hierarchical-income XGBoost + NEW hierarchical-income LightGBM

HYPOTHESIS
----------
Hierarchical income TE improved LightGBM from 0.94578042 -> 0.94582702 and
improved all 5 frozen folds. The question is whether that standalone gain
survives once LightGBM is used as the diversity member of the current
3-model ensemble.

ONLY CHANGE
-----------
Old LightGBM ensemble member -> hierarchical-income LightGBM ensemble member.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- CatBoost artifact
- hierarchical-income XGBoost artifact
- fold-local rank blending
- coarse 0.05 simplex grid
- held-fold meta-CV protocol
- no model retraining
- no public leaderboard optimization

META-CV
-------
For each held fold:
1. Select weights using only the other 4 frozen folds.
2. Evaluate those selected weights on the untouched held fold.
3. Compare CONTROL vs CANDIDATE.

This is a structural ensemble-member audit, not blend-weight microtuning.

Run:
    python blend_catboost_hierarchical_xgb_hierarchical_lgbm_rank_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_CAT_AUC = 0.94543956
EXPECTED_XGB_AUC = 0.94587366
EXPECTED_OLD_LGBM_AUC = 0.94578042
EXPECTED_NEW_LGBM_AUC = 0.94582702

REFERENCE_CURRENT_3MODEL_META_AUC = 0.94591016

GRID_STEP = 0.05
AUC_CHECK_TOLERANCE = 2e-5
CONTROL_REPRO_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = (
    Path("artifacts") / "validation" / "candidate_folds.csv"
)

DEFAULT_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)
DEFAULT_CAT_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_test_predictions.csv"
)

DEFAULT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_te_gpu"
    / "oof_predictions.csv"
)
DEFAULT_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_te_gpu"
    / "test_predictions.csv"
)

DEFAULT_OLD_LGBM_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)
DEFAULT_OLD_LGBM_TEST = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "test_predictions.csv"
)

DEFAULT_NEW_LGBM_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_hierarchical_income_te_cpu"
    / "oof_predictions.csv"
)
DEFAULT_NEW_LGBM_TEST = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_hierarchical_income_te_cpu"
    / "test_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_hierarchical_xgb_hierarchical_lgbm_rank_audit"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Meta-CV audit replacing old LightGBM with hierarchical-income "
            "LightGBM inside the current CatBoost/XGB/LightGBM ensemble."
        )
    )

    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)

    p.add_argument("--cat-oof", type=Path, default=DEFAULT_CAT_OOF)
    p.add_argument("--cat-test", type=Path, default=DEFAULT_CAT_TEST)

    p.add_argument("--xgb-oof", type=Path, default=DEFAULT_XGB_OOF)
    p.add_argument("--xgb-test", type=Path, default=DEFAULT_XGB_TEST)

    p.add_argument("--old-lgbm-oof", type=Path, default=DEFAULT_OLD_LGBM_OOF)
    p.add_argument("--old-lgbm-test", type=Path, default=DEFAULT_OLD_LGBM_TEST)

    p.add_argument("--new-lgbm-oof", type=Path, default=DEFAULT_NEW_LGBM_OOF)
    p.add_argument("--new-lgbm-test", type=Path, default=DEFAULT_NEW_LGBM_TEST)

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise ValueError(
            f"Expected exactly one train-only target column, found {train_only}"
        )
    return train_only[0]


def encode_binary_target(y: pd.Series) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise ValueError(f"Expected binary target, found {values}")

    preferred = {"yes", "true", "1", "positive", "buy", "will_buy"}
    positive = None

    for value in values:
        if str(value).strip().lower() in preferred:
            positive = value
            break

    if positive is None:
        positive = y.value_counts().idxmin()

    return (y == positive).astype(np.int8).to_numpy(), positive


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}
    missing = required - set(folds.columns)

    if missing:
        raise ValueError(f"Fold file missing columns: {sorted(missing)}")

    if len(folds) != len(train):
        raise ValueError("Frozen fold row count does not match train.csv.")

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError("Frozen fold row order does not match train.csv.")

    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected frozen folds [0,1,2,3,4].")

    extras = [c for c in folds.columns if c not in {"row_index", "fold"}]

    if len(extras) > 1:
        raise ValueError(f"Unexpected extra fold columns: {extras}")

    id_col = extras[0] if extras else None

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(f"Fold ID {id_col!r} missing from train.csv.")

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError("Frozen fold IDs do not align with train.csv.")

    return folds["fold"].to_numpy(dtype=np.int16), id_col


def choose_prediction_column(
    df: pd.DataFrame,
    path: Path,
    kind: str,
) -> str:
    if kind == "oof":
        preferred = [
            "oof_prediction",
            "prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {
            "row_index",
            "fold",
            "target",
            "target_encoded",
            "id",
        }
    elif kind == "test":
        preferred = [
            "prediction",
            "test_prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {"row_index", "fold", "id"}
    else:
        raise ValueError(f"Unknown kind: {kind}")

    for c in preferred:
        if c in df.columns:
            return c

    numeric = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) == 1:
        return numeric[0]

    raise ValueError(
        f"Could not safely identify {kind} prediction column in:\n"
        f"{path.resolve()}\n"
        f"Columns: {list(df.columns)}"
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"OOF file not found:\n{path.resolve()}")

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise ValueError(f"OOF row count mismatch in {path}")

    if "row_index" not in df.columns:
        raise ValueError(f"OOF row_index missing in {path}")

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(f"OOF row order mismatch in {path}")

    if "fold" not in df.columns:
        raise ValueError(f"OOF fold missing in {path}")

    if not np.array_equal(df["fold"].to_numpy(), fold_ids):
        raise ValueError(f"OOF fold mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if not np.array_equal(
            df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(f"OOF ID mismatch in {path}")

    pred_col = choose_prediction_column(df, path, "oof")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite OOF predictions in {path}")

    return pred


def load_test(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Test prediction file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    if len(df) != len(test):
        raise ValueError(f"Test row count mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if id_col not in test.columns:
            raise ValueError(f"Test data missing ID column {id_col!r}")

        if not np.array_equal(
            df[id_col].to_numpy(),
            test[id_col].to_numpy(),
        ):
            raise ValueError(f"Test ID mismatch in {path}")

    pred_col = choose_prediction_column(df, path, "test")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite test predictions in {path}")

    return pred


def verify_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    auc = float(roc_auc_score(y, pred))

    if abs(auc - expected) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            f"{name} OOF AUC mismatch.\n"
            f"Expected approximately: {expected:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}\n"
            "Stop and inspect artifact alignment before blending."
        )

    return auc


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def foldwise_percentile_rank(
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
    units = int(round(1.0 / GRID_STEP))
    grid = []

    # Enumerate LightGBM from zero upward so exact ties conservatively prefer
    # less LightGBM rather than inventing diversity from a tie.
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
    lgb_rank: np.ndarray,
    fit_mask: np.ndarray,
    grid: list[tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    best_auc = -np.inf
    best_weights = (0.0, 1.0, 0.0)

    y_fit = y[fit_mask]
    cat_fit = cat_rank[fit_mask]
    xgb_fit = xgb_rank[fit_mask]
    lgb_fit = lgb_rank[fit_mask]

    for cat_w, xgb_w, lgb_w in grid:
        score = (
            cat_w * cat_fit
            + xgb_w * xgb_fit
            + lgb_w * lgb_fit
        )

        auc = float(roc_auc_score(y_fit, score))

        if auc > best_auc + 1e-12:
            best_auc = auc
            best_weights = (cat_w, xgb_w, lgb_w)

    return (*best_weights, best_auc)


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    required_paths = [
        train_path,
        test_path,
        args.folds_path,
        args.cat_oof,
        args.cat_test,
        args.xgb_oof,
        args.xgb_test,
        args.old_lgbm_oof,
        args.old_lgbm_test,
        args.new_lgbm_oof,
        args.new_lgbm_test,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = sha256_file(args.folds_path)

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds = pd.read_csv(args.folds_path)

    target = detect_target(train, test)
    y, positive_label = encode_binary_target(train[target])
    fold_ids, id_col = validate_folds(folds, train)

    cat_oof = load_oof(
        args.cat_oof,
        train,
        fold_ids,
        id_col,
    )
    xgb_oof = load_oof(
        args.xgb_oof,
        train,
        fold_ids,
        id_col,
    )
    old_lgbm_oof = load_oof(
        args.old_lgbm_oof,
        train,
        fold_ids,
        id_col,
    )
    new_lgbm_oof = load_oof(
        args.new_lgbm_oof,
        train,
        fold_ids,
        id_col,
    )

    cat_auc = verify_auc(
        "CatBoost",
        y,
        cat_oof,
        EXPECTED_CAT_AUC,
    )
    xgb_auc = verify_auc(
        "Hierarchical-income XGBoost",
        y,
        xgb_oof,
        EXPECTED_XGB_AUC,
    )
    old_lgbm_auc = verify_auc(
        "Old engineered LightGBM",
        y,
        old_lgbm_oof,
        EXPECTED_OLD_LGBM_AUC,
    )
    new_lgbm_auc = verify_auc(
        "Hierarchical-income LightGBM",
        y,
        new_lgbm_oof,
        EXPECTED_NEW_LGBM_AUC,
    )

    cat_test = load_test(
        args.cat_test,
        test,
        id_col,
    )
    xgb_test = load_test(
        args.xgb_test,
        test,
        id_col,
    )
    old_lgbm_test = load_test(
        args.old_lgbm_test,
        test,
        id_col,
    )
    new_lgbm_test = load_test(
        args.new_lgbm_test,
        test,
        id_col,
    )

    cat_rank = foldwise_percentile_rank(
        cat_oof,
        fold_ids,
    )
    xgb_rank = foldwise_percentile_rank(
        xgb_oof,
        fold_ids,
    )
    old_lgbm_rank = foldwise_percentile_rank(
        old_lgbm_oof,
        fold_ids,
    )
    new_lgbm_rank = foldwise_percentile_rank(
        new_lgbm_oof,
        fold_ids,
    )

    grid = simplex_grid()

    print("=" * 98)
    print("3-MODEL ENSEMBLE UPDATE: OLD LGBM -> HIERARCHICAL-INCOME LGBM")
    print("=" * 98)
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Simplex grid step: {GRID_STEP:.2f} | grid size: {len(grid)}")
    print()
    print("SINGLE-MODEL REFERENCES")
    print(f"  CatBoost                     : {cat_auc:.8f}")
    print(f"  Hierarchical-income XGB      : {xgb_auc:.8f}")
    print(f"  Old engineered LightGBM      : {old_lgbm_auc:.8f}")
    print(f"  Hierarchical-income LightGBM : {new_lgbm_auc:.8f}")
    print()
    print("RANK-CORRELATION DIAGNOSTICS")
    print(
        f"  old LGBM vs new LGBM : "
        f"{rank_corr(old_lgbm_oof, new_lgbm_oof):.6f}"
    )
    print(
        f"  XGB vs new LGBM      : "
        f"{rank_corr(xgb_oof, new_lgbm_oof):.6f}"
    )
    print(
        f"  CAT vs new LGBM      : "
        f"{rank_corr(cat_oof, new_lgbm_oof):.6f}"
    )
    print()
    print("META-CV: weights selected on 4 folds, evaluated on held fold")
    print("-" * 98)

    control_meta_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )
    candidate_meta_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []
    total_start = time.perf_counter()

    for held_fold in range(5):
        fit_mask = fold_ids != held_fold
        held_mask = fold_ids == held_fold

        (
            ctrl_cat_w,
            ctrl_xgb_w,
            ctrl_lgb_w,
            ctrl_fit_auc,
        ) = choose_weights(
            y=y,
            cat_rank=cat_rank,
            xgb_rank=xgb_rank,
            lgb_rank=old_lgbm_rank,
            fit_mask=fit_mask,
            grid=grid,
        )

        control_held_pred = (
            ctrl_cat_w * cat_rank[held_mask]
            + ctrl_xgb_w * xgb_rank[held_mask]
            + ctrl_lgb_w * old_lgbm_rank[held_mask]
        )

        control_held_auc = float(
            roc_auc_score(
                y[held_mask],
                control_held_pred,
            )
        )

        control_meta_oof[held_mask] = control_held_pred

        (
            cand_cat_w,
            cand_xgb_w,
            cand_lgb_w,
            cand_fit_auc,
        ) = choose_weights(
            y=y,
            cat_rank=cat_rank,
            xgb_rank=xgb_rank,
            lgb_rank=new_lgbm_rank,
            fit_mask=fit_mask,
            grid=grid,
        )

        candidate_held_pred = (
            cand_cat_w * cat_rank[held_mask]
            + cand_xgb_w * xgb_rank[held_mask]
            + cand_lgb_w * new_lgbm_rank[held_mask]
        )

        candidate_held_auc = float(
            roc_auc_score(
                y[held_mask],
                candidate_held_pred,
            )
        )

        candidate_meta_oof[held_mask] = candidate_held_pred

        delta = candidate_held_auc - control_held_auc

        fold_rows.append(
            {
                "held_fold": held_fold,
                "control_cat_weight": ctrl_cat_w,
                "control_xgb_weight": ctrl_xgb_w,
                "control_old_lgbm_weight": ctrl_lgb_w,
                "control_fit_auc": ctrl_fit_auc,
                "control_held_auc": control_held_auc,
                "candidate_cat_weight": cand_cat_w,
                "candidate_xgb_weight": cand_xgb_w,
                "candidate_new_lgbm_weight": cand_lgb_w,
                "candidate_fit_auc": cand_fit_auc,
                "candidate_held_auc": candidate_held_auc,
                "candidate_delta_vs_control": delta,
            }
        )

        print(
            f"Fold {held_fold}: "
            f"control CAT/XGB/oldLGBM="
            f"{ctrl_cat_w:.2f}/{ctrl_xgb_w:.2f}/{ctrl_lgb_w:.2f} "
            f"AUC={control_held_auc:.8f} | "
            f"candidate CAT/XGB/newLGBM="
            f"{cand_cat_w:.2f}/{cand_xgb_w:.2f}/{cand_lgb_w:.2f} "
            f"AUC={candidate_held_auc:.8f} "
            f"({delta:+.8f})"
        )

    total_seconds = time.perf_counter() - total_start

    if (
        np.isnan(control_meta_oof).any()
        or np.isnan(candidate_meta_oof).any()
    ):
        raise RuntimeError("Meta-CV predictions contain NaNs.")

    fold_metrics = pd.DataFrame(fold_rows)

    control_meta_auc = float(
        roc_auc_score(
            y,
            control_meta_oof,
        )
    )
    candidate_meta_auc = float(
        roc_auc_score(
            y,
            candidate_meta_oof,
        )
    )

    delta_meta = candidate_meta_auc - control_meta_auc

    if abs(
        control_meta_auc - REFERENCE_CURRENT_3MODEL_META_AUC
    ) > CONTROL_REPRO_TOLERANCE:
        raise ValueError(
            "Control did not reproduce the current 3-model champion closely enough.\n"
            f"Expected approximately: {REFERENCE_CURRENT_3MODEL_META_AUC:.8f}\n"
            f"Reproduced control    : {control_meta_auc:.8f}\n"
            "Stop and inspect artifact alignment before trusting the candidate."
        )

    folds_improved = int(
        (fold_metrics["candidate_delta_vs_control"] > 0).sum()
    )
    folds_worse = int(
        (fold_metrics["candidate_delta_vs_control"] < 0).sum()
    )
    folds_tied = 5 - folds_improved - folds_worse

    control_mean_weights = np.array(
        [
            fold_metrics["control_cat_weight"].mean(),
            fold_metrics["control_xgb_weight"].mean(),
            fold_metrics["control_old_lgbm_weight"].mean(),
        ],
        dtype=np.float64,
    )
    control_mean_weights /= control_mean_weights.sum()

    candidate_mean_weights = np.array(
        [
            fold_metrics["candidate_cat_weight"].mean(),
            fold_metrics["candidate_xgb_weight"].mean(),
            fold_metrics["candidate_new_lgbm_weight"].mean(),
        ],
        dtype=np.float64,
    )
    candidate_mean_weights /= candidate_mean_weights.sum()

    control_full_oof = (
        control_mean_weights[0] * cat_rank
        + control_mean_weights[1] * xgb_rank
        + control_mean_weights[2] * old_lgbm_rank
    )

    candidate_full_oof = (
        candidate_mean_weights[0] * cat_rank
        + candidate_mean_weights[1] * xgb_rank
        + candidate_mean_weights[2] * new_lgbm_rank
    )

    control_full_auc = float(
        roc_auc_score(y, control_full_oof)
    )
    candidate_full_auc = float(
        roc_auc_score(y, candidate_full_oof)
    )

    # Deployment prediction using mean held-fold-selected weights.
    cat_test_rank = percentile_rank(cat_test)
    xgb_test_rank = percentile_rank(xgb_test)
    new_lgbm_test_rank = percentile_rank(new_lgbm_test)

    candidate_test_prediction = (
        candidate_mean_weights[0] * cat_test_rank
        + candidate_mean_weights[1] * xgb_test_rank
        + candidate_mean_weights[2] * new_lgbm_test_rank
    )

    if (
        delta_meta > 0
        and folds_improved >= 3
    ):
        decision = "KEEP_NEW_LGBM_IN_3MODEL_ENSEMBLE"
    else:
        decision = "REJECT_NEW_LGBM_ENSEMBLE_UPDATE"

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        args.output_dir / "meta_fold_metrics.csv",
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "control_meta_oof_prediction": control_meta_oof,
            "candidate_meta_oof_prediction": candidate_meta_oof,
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[id_col].to_numpy(),
        )

    oof_output.to_csv(
        args.output_dir / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": candidate_test_prediction.astype(np.float32)
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: 3-MODEL UPDATE WITH HIERARCHICAL-INCOME LIGHTGBM",
        "=" * 84,
        "",
        "HYPOTHESIS",
        "Does the standalone LightGBM gain from hierarchical-income TE survive",
        "when LightGBM is used as the diversity member in the current ensemble?",
        "",
        "ONLY CHANGE",
        "Old engineered LightGBM -> hierarchical-income LightGBM.",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {fold_hash}",
        f"Simplex grid step: {GRID_STEP:.2f}",
        "Weights selected on 4 folds and evaluated on held fold.",
        "No public leaderboard optimization.",
        "",
        "SINGLE MODEL REFERENCES",
        f"CatBoost: {cat_auc:.8f}",
        f"Hierarchical-income XGB: {xgb_auc:.8f}",
        f"Old engineered LightGBM: {old_lgbm_auc:.8f}",
        f"Hierarchical-income LightGBM: {new_lgbm_auc:.8f}",
        "",
        "META-CV RESULTS",
        f"Control current 3-model meta-CV AUC: {control_meta_auc:.8f}",
        f"Candidate updated 3-model meta-CV AUC: {candidate_meta_auc:.8f}",
        f"Delta vs control: {delta_meta:+.8f}",
        f"Held folds improved: {folds_improved}/5",
        f"Held folds worse: {folds_worse}/5",
        f"Held folds tied: {folds_tied}/5",
        "",
        "MEAN CANDIDATE DEPLOYMENT WEIGHTS",
        f"CatBoost: {candidate_mean_weights[0]:.4f}",
        f"Hierarchical-income XGB: {candidate_mean_weights[1]:.4f}",
        f"Hierarchical-income LightGBM: {candidate_mean_weights[2]:.4f}",
        "",
        "FULL-OOF DIAGNOSTICS USING MEAN WEIGHTS",
        f"Control full-OOF diagnostic AUC: {control_full_auc:.8f}",
        f"Candidate full-OOF diagnostic AUC: {candidate_full_auc:.8f}",
        "",
        f"Runtime: {total_seconds:.2f} seconds",
        f"DECISION: {decision}",
        "",
        "FOLD DETAILS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.held_fold}: "
            f"control={row.control_held_auc:.8f} "
            f"(CAT/XGB/oldLGBM="
            f"{row.control_cat_weight:.2f}/"
            f"{row.control_xgb_weight:.2f}/"
            f"{row.control_old_lgbm_weight:.2f}) -> "
            f"candidate={row.candidate_held_auc:.8f} "
            f"(CAT/XGB/newLGBM="
            f"{row.candidate_cat_weight:.2f}/"
            f"{row.candidate_xgb_weight:.2f}/"
            f"{row.candidate_new_lgbm_weight:.2f}) "
            f"delta={row.candidate_delta_vs_control:+.8f}"
        )

    (
        args.output_dir / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 98)
    print("ENSEMBLE LIGHTGBM UPDATE AUDIT COMPLETE")
    print("=" * 98)
    print(f"Control current 3-model : {control_meta_auc:.8f}")
    print(f"Candidate updated model : {candidate_meta_auc:.8f}")
    print(f"Delta vs control        : {delta_meta:+.8f}")
    print(
        f"Folds improved/worse/tied: "
        f"{folds_improved}/{folds_worse}/{folds_tied}"
    )
    print(
        "Mean candidate weights  : "
        f"CAT={candidate_mean_weights[0]:.4f}, "
        f"XGB={candidate_mean_weights[1]:.4f}, "
        f"newLGBM={candidate_mean_weights[2]:.4f}"
    )
    print(f"Decision                : {decision}")
    print(f"Runtime                 : {total_seconds:.2f}s")
    print(f"Artifacts               : {args.output_dir.resolve()}")
    print("=" * 98)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. meta_fold_metrics.csv")


if __name__ == "__main__":
    main()
