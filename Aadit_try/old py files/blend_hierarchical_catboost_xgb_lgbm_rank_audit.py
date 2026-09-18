"""
blend_hierarchical_catboost_xgb_lgbm_rank_audit.py

Kaggle Playground Series S6E9

Controlled ensemble update:
Replace ONLY the CatBoost member in the current validated 3-model ensemble.

CONTROL
-------
OLD exact-value CatBoost 3-seed
+ hierarchical-income XGBoost
+ old engineered LightGBM

CANDIDATE
---------
NEW hierarchical-income CatBoost 3-seed
+ hierarchical-income XGBoost
+ old engineered LightGBM

HYPOTHESIS
----------
The new hierarchical CatBoost 3-seed rank average improved the old CatBoost
3-seed from 0.94543956 -> 0.94553699, with 5/5 folds improved versus the
hierarchical seed-42 model.

The question now is whether that stronger CatBoost member improves the
validated 3-model ensemble.

ONLY CHANGE
-----------
Old CatBoost 3-seed -> hierarchical-income CatBoost 3-seed.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- hierarchical-income XGBoost artifact
- old engineered LightGBM artifact
- fold-local percentile-rank blending
- coarse 0.05 simplex weight grid
- weights selected on 4 folds, evaluated on held fold
- no model retraining
- no public leaderboard optimization

Run:
    python blend_hierarchical_catboost_xgb_lgbm_rank_audit.py
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

EXPECTED_OLD_CAT_AUC = 0.94543956
EXPECTED_NEW_CAT_AUC = 0.94553699
EXPECTED_XGB_AUC = 0.94587366
EXPECTED_LGBM_AUC = 0.94578042

REFERENCE_CURRENT_ENSEMBLE_META_AUC = 0.94591016

GRID_STEP = 0.05
AUC_CHECK_TOLERANCE = 2e-5
CONTROL_REPRO_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_OLD_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OLD_CAT_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_test_predictions.csv"
)

DEFAULT_NEW_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_NEW_CAT_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_multiseed_gpu"
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

DEFAULT_LGBM_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

DEFAULT_LGBM_TEST = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "test_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_hierarchical_catboost_xgb_lgbm_rank_audit"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Meta-CV audit replacing old CatBoost 3-seed with the new "
            "hierarchical-income CatBoost 3-seed."
        )
    )

    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)

    p.add_argument("--old-cat-oof", type=Path, default=DEFAULT_OLD_CAT_OOF)
    p.add_argument("--old-cat-test", type=Path, default=DEFAULT_OLD_CAT_TEST)

    p.add_argument("--new-cat-oof", type=Path, default=DEFAULT_NEW_CAT_OOF)
    p.add_argument("--new-cat-test", type=Path, default=DEFAULT_NEW_CAT_TEST)

    p.add_argument("--xgb-oof", type=Path, default=DEFAULT_XGB_OOF)
    p.add_argument("--xgb-test", type=Path, default=DEFAULT_XGB_TEST)

    p.add_argument("--lgbm-oof", type=Path, default=DEFAULT_LGBM_OOF)
    p.add_argument("--lgbm-test", type=Path, default=DEFAULT_LGBM_TEST)

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

    expected_rows = np.arange(len(train), dtype=np.int64)

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        expected_rows,
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


def pick_prediction_column(
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
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
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

    pred_col = pick_prediction_column(df, path, "oof")
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

    pred_col = pick_prediction_column(df, path, "test")
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
            f"Loaded artifact AUC   : {auc:.8f}"
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
    grid: list[tuple[float, float, float]] = []

    # Enumerate LGBM from zero upward so exact ties conservatively prefer
    # less LGBM rather than injecting extra diversity from a tie.
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
        args.old_cat_oof,
        args.old_cat_test,
        args.new_cat_oof,
        args.new_cat_test,
        args.xgb_oof,
        args.xgb_test,
        args.lgbm_oof,
        args.lgbm_test,
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

    old_cat_oof = load_oof(
        args.old_cat_oof,
        train,
        fold_ids,
        id_col,
    )
    new_cat_oof = load_oof(
        args.new_cat_oof,
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
    lgbm_oof = load_oof(
        args.lgbm_oof,
        train,
        fold_ids,
        id_col,
    )

    old_cat_auc = verify_auc(
        "Old CatBoost 3-seed",
        y,
        old_cat_oof,
        EXPECTED_OLD_CAT_AUC,
    )
    new_cat_auc = verify_auc(
        "Hierarchical CatBoost 3-seed",
        y,
        new_cat_oof,
        EXPECTED_NEW_CAT_AUC,
    )
    xgb_auc = verify_auc(
        "Hierarchical-income XGBoost",
        y,
        xgb_oof,
        EXPECTED_XGB_AUC,
    )
    lgbm_auc = verify_auc(
        "Engineered LightGBM",
        y,
        lgbm_oof,
        EXPECTED_LGBM_AUC,
    )

    old_cat_test = load_test(
        args.old_cat_test,
        test,
        id_col,
    )
    new_cat_test = load_test(
        args.new_cat_test,
        test,
        id_col,
    )
    xgb_test = load_test(
        args.xgb_test,
        test,
        id_col,
    )
    lgbm_test = load_test(
        args.lgbm_test,
        test,
        id_col,
    )

    old_cat_rank = foldwise_percentile_rank(
        old_cat_oof,
        fold_ids,
    )
    new_cat_rank = foldwise_percentile_rank(
        new_cat_oof,
        fold_ids,
    )
    xgb_rank = foldwise_percentile_rank(
        xgb_oof,
        fold_ids,
    )
    lgbm_rank = foldwise_percentile_rank(
        lgbm_oof,
        fold_ids,
    )

    grid = simplex_grid()

    print("=" * 100)
    print("3-MODEL ENSEMBLE UPDATE: OLD CATBOOST -> HIERARCHICAL CATBOOST")
    print("=" * 100)
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Simplex grid step: {GRID_STEP:.2f} | grid size: {len(grid)}")
    print()
    print("SINGLE-MODEL REFERENCES")
    print(f"  Old CatBoost 3-seed          : {old_cat_auc:.8f}")
    print(f"  Hierarchical CatBoost 3-seed : {new_cat_auc:.8f}")
    print(f"  Hierarchical-income XGB       : {xgb_auc:.8f}")
    print(f"  Old engineered LightGBM       : {lgbm_auc:.8f}")
    print()
    print("RANK-CORRELATION DIAGNOSTICS")
    print(
        f"  old CAT vs new CAT : "
        f"{rank_corr(old_cat_oof, new_cat_oof):.6f}"
    )
    print(
        f"  new CAT vs XGB     : "
        f"{rank_corr(new_cat_oof, xgb_oof):.6f}"
    )
    print(
        f"  new CAT vs LGBM    : "
        f"{rank_corr(new_cat_oof, lgbm_oof):.6f}"
    )
    print()
    print("META-CV: weights selected on 4 folds, evaluated on held fold")
    print("-" * 100)

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

    fold_rows: list[dict] = []
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
            cat_rank=old_cat_rank,
            xgb_rank=xgb_rank,
            lgb_rank=lgbm_rank,
            fit_mask=fit_mask,
            grid=grid,
        )

        control_held_pred = (
            ctrl_cat_w * old_cat_rank[held_mask]
            + ctrl_xgb_w * xgb_rank[held_mask]
            + ctrl_lgb_w * lgbm_rank[held_mask]
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
            cat_rank=new_cat_rank,
            xgb_rank=xgb_rank,
            lgb_rank=lgbm_rank,
            fit_mask=fit_mask,
            grid=grid,
        )

        candidate_held_pred = (
            cand_cat_w * new_cat_rank[held_mask]
            + cand_xgb_w * xgb_rank[held_mask]
            + cand_lgb_w * lgbm_rank[held_mask]
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
                "control_old_cat_weight": ctrl_cat_w,
                "control_xgb_weight": ctrl_xgb_w,
                "control_lgbm_weight": ctrl_lgb_w,
                "control_fit_auc": ctrl_fit_auc,
                "control_held_auc": control_held_auc,
                "candidate_new_cat_weight": cand_cat_w,
                "candidate_xgb_weight": cand_xgb_w,
                "candidate_lgbm_weight": cand_lgb_w,
                "candidate_fit_auc": cand_fit_auc,
                "candidate_held_auc": candidate_held_auc,
                "candidate_delta_vs_control": delta,
            }
        )

        print(
            f"Fold {held_fold}: "
            f"control oldCAT/XGB/LGBM="
            f"{ctrl_cat_w:.2f}/{ctrl_xgb_w:.2f}/{ctrl_lgb_w:.2f} "
            f"AUC={control_held_auc:.8f} | "
            f"candidate newCAT/XGB/LGBM="
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
        control_meta_auc
        - REFERENCE_CURRENT_ENSEMBLE_META_AUC
    ) > CONTROL_REPRO_TOLERANCE:
        raise ValueError(
            "Control did not reproduce the current ensemble champion closely enough.\n"
            f"Expected approximately: {REFERENCE_CURRENT_ENSEMBLE_META_AUC:.8f}\n"
            f"Reproduced control    : {control_meta_auc:.8f}\n"
            "Stop and inspect artifact alignment before trusting the candidate."
        )

    folds_improved = int(
        (
            fold_metrics[
                "candidate_delta_vs_control"
            ] > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "candidate_delta_vs_control"
            ] < 0
        ).sum()
    )

    folds_tied = 5 - folds_improved - folds_worse

    control_mean_weights = np.array(
        [
            fold_metrics[
                "control_old_cat_weight"
            ].mean(),
            fold_metrics[
                "control_xgb_weight"
            ].mean(),
            fold_metrics[
                "control_lgbm_weight"
            ].mean(),
        ],
        dtype=np.float64,
    )

    control_mean_weights /= control_mean_weights.sum()

    candidate_mean_weights = np.array(
        [
            fold_metrics[
                "candidate_new_cat_weight"
            ].mean(),
            fold_metrics[
                "candidate_xgb_weight"
            ].mean(),
            fold_metrics[
                "candidate_lgbm_weight"
            ].mean(),
        ],
        dtype=np.float64,
    )

    candidate_mean_weights /= candidate_mean_weights.sum()

    control_full_oof = (
        control_mean_weights[0] * old_cat_rank
        + control_mean_weights[1] * xgb_rank
        + control_mean_weights[2] * lgbm_rank
    )

    candidate_full_oof = (
        candidate_mean_weights[0] * new_cat_rank
        + candidate_mean_weights[1] * xgb_rank
        + candidate_mean_weights[2] * lgbm_rank
    )

    control_full_auc = float(
        roc_auc_score(
            y,
            control_full_oof,
        )
    )

    candidate_full_auc = float(
        roc_auc_score(
            y,
            candidate_full_oof,
        )
    )

    new_cat_test_rank = percentile_rank(new_cat_test)
    xgb_test_rank = percentile_rank(xgb_test)
    lgbm_test_rank = percentile_rank(lgbm_test)

    candidate_test_prediction = (
        candidate_mean_weights[0] * new_cat_test_rank
        + candidate_mean_weights[1] * xgb_test_rank
        + candidate_mean_weights[2] * lgbm_test_rank
    )

    # Conservative decision rule:
    # - positive overall held-fold meta-CV delta
    # - at least 4/5 held folds improve
    #
    # If the overall delta is microscopic, we will still review it manually
    # rather than blindly trusting this label.
    if (
        delta_meta > 0
        and folds_improved >= 4
    ):
        decision = "KEEP_HIERARCHICAL_CATBOOST_IN_3MODEL_ENSEMBLE"
    else:
        decision = "REJECT_HIERARCHICAL_CATBOOST_ENSEMBLE_UPDATE"

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        args.output_dir
        / "meta_fold_metrics.csv",
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
        args.output_dir
        / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": candidate_test_prediction.astype(
                np.float32
            )
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: 3-MODEL UPDATE WITH HIERARCHICAL CATBOOST 3-SEED",
        "=" * 86,
        "",
        "HYPOTHESIS",
        "Does replacing the old CatBoost 3-seed member with the stronger",
        "hierarchical-income CatBoost 3-seed improve the current ensemble?",
        "",
        "ONLY CHANGE",
        "Old CatBoost 3-seed -> hierarchical-income CatBoost 3-seed.",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {fold_hash}",
        f"Simplex grid step: {GRID_STEP:.2f}",
        "Weights selected on 4 folds and evaluated on held fold.",
        "No public leaderboard optimization.",
        "",
        "SINGLE MODEL REFERENCES",
        f"Old CatBoost 3-seed: {old_cat_auc:.8f}",
        f"Hierarchical CatBoost 3-seed: {new_cat_auc:.8f}",
        f"Hierarchical-income XGBoost: {xgb_auc:.8f}",
        f"Old engineered LightGBM: {lgbm_auc:.8f}",
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
        f"Hierarchical CatBoost: {candidate_mean_weights[0]:.4f}",
        f"Hierarchical-income XGB: {candidate_mean_weights[1]:.4f}",
        f"Old engineered LightGBM: {candidate_mean_weights[2]:.4f}",
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
            f"(oldCAT/XGB/LGBM="
            f"{row.control_old_cat_weight:.2f}/"
            f"{row.control_xgb_weight:.2f}/"
            f"{row.control_lgbm_weight:.2f}) -> "
            f"candidate={row.candidate_held_auc:.8f} "
            f"(newCAT/XGB/LGBM="
            f"{row.candidate_new_cat_weight:.2f}/"
            f"{row.candidate_xgb_weight:.2f}/"
            f"{row.candidate_lgbm_weight:.2f}) "
            f"delta={row.candidate_delta_vs_control:+.8f}"
        )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("HIERARCHICAL CATBOOST ENSEMBLE UPDATE AUDIT COMPLETE")
    print("=" * 100)
    print(f"Control current 3-model : {control_meta_auc:.8f}")
    print(f"Candidate updated model : {candidate_meta_auc:.8f}")
    print(f"Delta vs control        : {delta_meta:+.8f}")
    print(
        f"Folds improved/worse/tied: "
        f"{folds_improved}/{folds_worse}/{folds_tied}"
    )
    print(
        "Mean candidate weights  : "
        f"newCAT={candidate_mean_weights[0]:.4f}, "
        f"XGB={candidate_mean_weights[1]:.4f}, "
        f"LGBM={candidate_mean_weights[2]:.4f}"
    )
    print(f"Decision                : {decision}")
    print(f"Runtime                 : {total_seconds:.2f}s")
    print(f"Artifacts               : {args.output_dir.resolve()}")
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. meta_fold_metrics.csv")


if __name__ == "__main__":
    main()
