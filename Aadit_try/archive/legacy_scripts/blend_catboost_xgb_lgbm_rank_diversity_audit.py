"""
blend_catboost_xgb_lgbm_rank_diversity_audit.py

Controlled diversity audit for Kaggle Playground Series S6E9.

HYPOTHESIS
----------
Engineered LightGBM is slightly weaker than the current XGBoost champion, but
its residual ranking differences may still add useful ensemble diversity.

ONLY CHANGE
-----------
Add engineered LightGBM as a third member to the current CatBoost + XGBoost
rank-blend family.

HELD FIXED
----------
- frozen 5-fold validation assignment + SHA256
- CatBoost 3-seed exact-value rank ensemble
- current learned-margin XGBoost champion
- engineered LightGBM from the immediately preceding experiment
- no model retraining
- no public-LB optimization
- rank blending
- one fixed coarse 0.05 simplex grid

META-CV
-------
For each held fold:
1. Use the other 4 frozen folds to choose blend weights.
2. Evaluate those weights only on the untouched held fold.
3. Compare:
      CONTROL   = CatBoost + XGBoost
      CANDIDATE = CatBoost + XGBoost + LightGBM

Ranks are computed separately inside each frozen OOF fold, without targets.
The held-fold target is never used for weight selection.

The 0.05 grid is intentionally coarse. This experiment asks whether LightGBM
adds a real structural diversity benefit, not whether tiny blend-weight
micro-tuning can manufacture a gain.

Run:
    python blend_catboost_xgb_lgbm_rank_diversity_audit.py
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
EXPECTED_XGB_AUC = 0.94583075
EXPECTED_LGBM_AUC = 0.94578042
REFERENCE_OLD_BEST_BLEND_AUC = 0.94583385

GRID_STEP = 0.05
AUC_CHECK_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"

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
    / "xgboost_m2_exact_frequency_learned_logistic_margin_gpu"
    / "oof_predictions.csv"
)
DEFAULT_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_exact_frequency_learned_logistic_margin_gpu"
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
    / "blend_catboost_xgb_lgbm_rank_diversity_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe CatBoost + XGBoost + LightGBM rank-blend diversity audit."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)

    parser.add_argument("--cat-oof", type=Path, default=DEFAULT_CAT_OOF)
    parser.add_argument("--cat-test", type=Path, default=DEFAULT_CAT_TEST)

    parser.add_argument("--xgb-oof", type=Path, default=DEFAULT_XGB_OOF)
    parser.add_argument("--xgb-test", type=Path, default=DEFAULT_XGB_TEST)

    parser.add_argument("--lgbm-oof", type=Path, default=DEFAULT_LGBM_OOF)
    parser.add_argument("--lgbm-test", type=Path, default=DEFAULT_LGBM_TEST)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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
            f"Expected exactly one train-only target column, found: {train_only}"
        )
    return train_only[0]


def encode_binary_target(y: pd.Series) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise ValueError(f"Expected binary target, found: {values}")

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
    if not np.array_equal(folds["row_index"].to_numpy(), expected_rows):
        raise ValueError("Frozen fold row order does not match train.csv.")

    unique_folds = sorted(folds["fold"].unique().tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected frozen folds [0,1,2,3,4], found {unique_folds}")

    extras = [c for c in folds.columns if c not in {"row_index", "fold"}]
    if len(extras) > 1:
        raise ValueError(f"Unexpected extra fold columns: {extras}")

    id_col = extras[0] if extras else None

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(f"Fold ID column {id_col!r} missing from train.csv.")
        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError("Frozen fold IDs do not align with train.csv.")

    return folds["fold"].to_numpy(dtype=np.int16), id_col


def _pick_prediction_column(
    df: pd.DataFrame,
    *,
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
    elif kind == "test":
        preferred = [
            "prediction",
            "test_prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
    else:
        raise ValueError(f"Unknown prediction kind: {kind}")

    present = [c for c in preferred if c in df.columns]
    if present:
        return present[0]

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "id",
    }
    numeric = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) == 1:
        return numeric[0]

    raise ValueError(
        f"Could not safely identify the {kind} prediction column in:\n"
        f"{path.resolve()}\n"
        f"Columns: {list(df.columns)}\n"
        "No prediction column was guessed because that could silently blend the "
        "wrong artifact."
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
        raise ValueError(f"OOF file missing row_index: {path}")

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(f"OOF row order mismatch in {path}")

    if "fold" not in df.columns:
        raise ValueError(f"OOF file missing fold: {path}")

    if not np.array_equal(df["fold"].to_numpy(), fold_ids):
        raise ValueError(f"OOF fold assignment mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if not np.array_equal(df[id_col].to_numpy(), train[id_col].to_numpy()):
            raise ValueError(f"OOF ID mismatch in {path}")

    pred_col = _pick_prediction_column(df, path=path, kind="oof")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite OOF predictions in {path}")

    return pred


def load_test_predictions(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Test prediction file not found:\n{path.resolve()}")

    df = pd.read_csv(path)

    if len(df) != len(test):
        raise ValueError(f"Test prediction row count mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if id_col not in test.columns:
            raise ValueError(f"Test data missing ID column {id_col!r}")
        if not np.array_equal(df[id_col].to_numpy(), test[id_col].to_numpy()):
            raise ValueError(f"Test prediction ID mismatch in {path}")

    pred_col = _pick_prediction_column(df, path=path, kind="test")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite test predictions in {path}")

    return pred


def assert_expected_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    auc = float(roc_auc_score(y, pred))
    if abs(auc - expected) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            f"{name} OOF AUC does not match the documented artifact.\n"
            f"Expected approximately: {expected:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}\n"
            "Stop here and inspect the prediction column/path rather than blending "
            "a potentially wrong artifact."
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
    """
    Rank each frozen OOF fold independently.

    This prevents the held fold's score distribution from changing the rank
    scaling of the four folds used to choose blend weights.
    """
    out = np.empty(len(values), dtype=np.float64)

    for fold in range(5):
        mask = fold_ids == fold
        out[mask] = percentile_rank(values[mask])

    return out


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = percentile_rank(a)
    br = percentile_rank(b)
    return float(np.corrcoef(ar, br)[0, 1])


def control_weight_grid() -> list[tuple[float, float]]:
    units = int(round(1.0 / GRID_STEP))
    return [
        (cat_units / units, 1.0 - cat_units / units)
        for cat_units in range(units + 1)
    ]


def three_model_simplex_grid() -> list[tuple[float, float, float]]:
    units = int(round(1.0 / GRID_STEP))
    grid: list[tuple[float, float, float]] = []

    # Enumerate LightGBM from zero upward so exact AUC ties are resolved
    # conservatively in favor of less LightGBM.
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


def choose_best_control_weights(
    y: np.ndarray,
    cat_rank: np.ndarray,
    xgb_rank: np.ndarray,
    mask: np.ndarray,
    grid: list[tuple[float, float]],
) -> tuple[float, float, float]:
    best_auc = -np.inf
    best_cat = 0.0
    best_xgb = 1.0

    y_fit = y[mask]
    cat_fit = cat_rank[mask]
    xgb_fit = xgb_rank[mask]

    for cat_w, xgb_w in grid:
        score = cat_w * cat_fit + xgb_w * xgb_fit
        auc = float(roc_auc_score(y_fit, score))

        if auc > best_auc + 1e-12:
            best_auc = auc
            best_cat = cat_w
            best_xgb = xgb_w

    return best_cat, best_xgb, best_auc


def choose_best_three_model_weights(
    y: np.ndarray,
    cat_rank: np.ndarray,
    xgb_rank: np.ndarray,
    lgb_rank: np.ndarray,
    mask: np.ndarray,
    grid: list[tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    best_auc = -np.inf
    best_cat = 0.0
    best_xgb = 1.0
    best_lgb = 0.0

    y_fit = y[mask]
    cat_fit = cat_rank[mask]
    xgb_fit = xgb_rank[mask]
    lgb_fit = lgb_rank[mask]

    for cat_w, xgb_w, lgb_w in grid:
        score = cat_w * cat_fit + xgb_w * xgb_fit + lgb_w * lgb_fit
        auc = float(roc_auc_score(y_fit, score))

        if auc > best_auc + 1e-12:
            best_auc = auc
            best_cat = cat_w
            best_xgb = xgb_w
            best_lgb = lgb_w

    return best_cat, best_xgb, best_lgb, best_auc


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
        args.lgbm_oof,
        args.lgbm_test,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path.resolve()}")

    fold_hash = sha256_file(args.folds_path)
    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}\n"
            f"File    : {args.folds_path.resolve()}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds = pd.read_csv(args.folds_path)

    target = detect_target(train, test)
    y, positive_label = encode_binary_target(train[target])
    fold_ids, id_col = validate_folds(folds, train)

    print("=" * 92)
    print("CATBOOST + XGBOOST + LIGHTGBM RANK-BLEND DIVERSITY AUDIT")
    print("=" * 92)
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Fixed simplex grid step: {GRID_STEP:.2f}")
    print()

    cat_oof = load_oof(args.cat_oof, train, fold_ids, id_col)
    xgb_oof = load_oof(args.xgb_oof, train, fold_ids, id_col)
    lgb_oof = load_oof(args.lgbm_oof, train, fold_ids, id_col)

    cat_test = load_test_predictions(args.cat_test, test, id_col)
    xgb_test = load_test_predictions(args.xgb_test, test, id_col)
    lgb_test = load_test_predictions(args.lgbm_test, test, id_col)

    cat_auc = assert_expected_auc(
        "CatBoost 3-seed",
        y,
        cat_oof,
        EXPECTED_CAT_AUC,
    )
    xgb_auc = assert_expected_auc(
        "Current learned-margin XGBoost",
        y,
        xgb_oof,
        EXPECTED_XGB_AUC,
    )
    lgb_auc = assert_expected_auc(
        "Engineered LightGBM",
        y,
        lgb_oof,
        EXPECTED_LGBM_AUC,
    )

    print("VALIDATED SINGLE-MODEL REFERENCES")
    print(f"  CatBoost 3-seed exact-value : {cat_auc:.8f}")
    print(f"  Current XGB champion        : {xgb_auc:.8f}")
    print(f"  Engineered LightGBM         : {lgb_auc:.8f}")
    print(f"  Previous documented blend   : {REFERENCE_OLD_BEST_BLEND_AUC:.8f}")
    print("    (reference only; it used the older exact-frequency XGB)")
    print()

    print("PAIRWISE RANK CORRELATIONS")
    print(f"  CAT vs XGB : {rank_corr(cat_oof, xgb_oof):.6f}")
    print(f"  CAT vs LGBM: {rank_corr(cat_oof, lgb_oof):.6f}")
    print(f"  XGB vs LGBM: {rank_corr(xgb_oof, lgb_oof):.6f}")
    print()

    # Use fold-local percentile ranks so a held fold does not influence the
    # rank scaling used on the four folds that choose weights.
    cat_rank = foldwise_percentile_rank(cat_oof, fold_ids)
    xgb_rank = foldwise_percentile_rank(xgb_oof, fold_ids)
    lgb_rank = foldwise_percentile_rank(lgb_oof, fold_ids)

    control_grid = control_weight_grid()
    candidate_grid = three_model_simplex_grid()

    print(f"Control grid size  : {len(control_grid)}")
    print(f"3-model grid size  : {len(candidate_grid)}")
    print()
    print("META-CV: weights selected on 4 folds, scored on held fold")
    print("-" * 92)

    control_meta_oof = np.full(len(train), np.nan, dtype=np.float64)
    candidate_meta_oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_rows = []

    total_start = time.perf_counter()

    for held_fold in range(5):
        fit_mask = fold_ids != held_fold
        held_mask = fold_ids == held_fold

        control_cat_w, control_xgb_w, control_fit_auc = choose_best_control_weights(
            y=y,
            cat_rank=cat_rank,
            xgb_rank=xgb_rank,
            mask=fit_mask,
            grid=control_grid,
        )

        control_held_score = (
            control_cat_w * cat_rank[held_mask]
            + control_xgb_w * xgb_rank[held_mask]
        )
        control_held_auc = float(
            roc_auc_score(y[held_mask], control_held_score)
        )
        control_meta_oof[held_mask] = control_held_score

        (
            candidate_cat_w,
            candidate_xgb_w,
            candidate_lgb_w,
            candidate_fit_auc,
        ) = choose_best_three_model_weights(
            y=y,
            cat_rank=cat_rank,
            xgb_rank=xgb_rank,
            lgb_rank=lgb_rank,
            mask=fit_mask,
            grid=candidate_grid,
        )

        candidate_held_score = (
            candidate_cat_w * cat_rank[held_mask]
            + candidate_xgb_w * xgb_rank[held_mask]
            + candidate_lgb_w * lgb_rank[held_mask]
        )
        candidate_held_auc = float(
            roc_auc_score(y[held_mask], candidate_held_score)
        )
        candidate_meta_oof[held_mask] = candidate_held_score

        delta = candidate_held_auc - control_held_auc

        fold_rows.append(
            {
                "held_fold": held_fold,
                "control_cat_weight": control_cat_w,
                "control_xgb_weight": control_xgb_w,
                "control_fit_auc": control_fit_auc,
                "control_held_auc": control_held_auc,
                "candidate_cat_weight": candidate_cat_w,
                "candidate_xgb_weight": candidate_xgb_w,
                "candidate_lgbm_weight": candidate_lgb_w,
                "candidate_fit_auc": candidate_fit_auc,
                "candidate_held_auc": candidate_held_auc,
                "candidate_delta_vs_control": delta,
            }
        )

        print(
            f"Fold {held_fold}: "
            f"control CAT/XGB={control_cat_w:.2f}/{control_xgb_w:.2f} "
            f"AUC={control_held_auc:.8f} | "
            f"candidate CAT/XGB/LGBM="
            f"{candidate_cat_w:.2f}/{candidate_xgb_w:.2f}/{candidate_lgb_w:.2f} "
            f"AUC={candidate_held_auc:.8f} "
            f"({delta:+.8f})"
        )

    total_seconds = time.perf_counter() - total_start

    if np.isnan(control_meta_oof).any() or np.isnan(candidate_meta_oof).any():
        raise RuntimeError("Meta-CV OOF predictions contain NaNs.")

    fold_metrics = pd.DataFrame(fold_rows)

    control_meta_auc = float(roc_auc_score(y, control_meta_oof))
    candidate_meta_auc = float(roc_auc_score(y, candidate_meta_oof))
    meta_gain = candidate_meta_auc - control_meta_auc

    folds_improved = int(
        (fold_metrics["candidate_delta_vs_control"] > 0).sum()
    )
    folds_worse = int(
        (fold_metrics["candidate_delta_vs_control"] < 0).sum()
    )
    folds_tied = 5 - folds_improved - folds_worse

    lgb_nonzero_folds = int(
        (fold_metrics["candidate_lgbm_weight"] > 0).sum()
    )

    control_mean_weights = np.array(
        [
            fold_metrics["control_cat_weight"].mean(),
            fold_metrics["control_xgb_weight"].mean(),
        ],
        dtype=np.float64,
    )
    control_mean_weights /= control_mean_weights.sum()

    candidate_mean_weights = np.array(
        [
            fold_metrics["candidate_cat_weight"].mean(),
            fold_metrics["candidate_xgb_weight"].mean(),
            fold_metrics["candidate_lgbm_weight"].mean(),
        ],
        dtype=np.float64,
    )
    candidate_mean_weights /= candidate_mean_weights.sum()

    control_full_oof_diagnostic = (
        control_mean_weights[0] * cat_rank
        + control_mean_weights[1] * xgb_rank
    )
    candidate_full_oof_diagnostic = (
        candidate_mean_weights[0] * cat_rank
        + candidate_mean_weights[1] * xgb_rank
        + candidate_mean_weights[2] * lgb_rank
    )

    control_full_auc = float(
        roc_auc_score(y, control_full_oof_diagnostic)
    )
    candidate_full_auc = float(
        roc_auc_score(y, candidate_full_oof_diagnostic)
    )

    # Deployment ranking: each model is ranked across the competition test set.
    # No labels are involved.
    cat_test_rank = percentile_rank(cat_test)
    xgb_test_rank = percentile_rank(xgb_test)
    lgb_test_rank = percentile_rank(lgb_test)

    candidate_test_prediction = (
        candidate_mean_weights[0] * cat_test_rank
        + candidate_mean_weights[1] * xgb_test_rank
        + candidate_mean_weights[2] * lgb_test_rank
    )

    if meta_gain > 0 and folds_improved >= 3 and lgb_nonzero_folds >= 3:
        decision = "KEEP_LIGHTGBM_FOR_3MODEL_ENSEMBLE"
    else:
        decision = "REJECT_LIGHTGBM_FOR_ENSEMBLE"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(
        args.output_dir / "meta_fold_metrics.csv",
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold_ids,
            "target_encoded": y,
            "cat_rank": cat_rank,
            "xgb_rank": xgb_rank,
            "lgbm_rank": lgb_rank,
            "control_meta_oof_prediction": control_meta_oof,
            "candidate_meta_oof_prediction": candidate_meta_oof,
        }
    )
    if id_col is not None:
        oof_output.insert(1, id_col, train[id_col].to_numpy())

    oof_output.to_csv(
        args.output_dir / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {"prediction": candidate_test_prediction.astype(np.float32)}
    )
    if id_col is not None:
        test_output.insert(0, id_col, test[id_col].to_numpy())

    test_output.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: CATBOOST + XGBOOST + LIGHTGBM RANK-BLEND DIVERSITY AUDIT",
        "=" * 82,
        "",
        "HYPOTHESIS",
        "Can engineered LightGBM add held-fold ensemble value despite being",
        "slightly weaker than the current XGBoost champion?",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {fold_hash}",
        f"Rank-blend simplex grid step: {GRID_STEP:.2f}",
        "Weights are selected on 4 frozen folds and evaluated on the held fold.",
        "No public leaderboard information is used.",
        "",
        "SINGLE-MODEL REFERENCES",
        f"CatBoost 3-seed OOF: {cat_auc:.8f}",
        f"Current XGB champion OOF: {xgb_auc:.8f}",
        f"Engineered LightGBM OOF: {lgb_auc:.8f}",
        f"Previous documented 2-model blend reference: {REFERENCE_OLD_BEST_BLEND_AUC:.8f}",
        "",
        "META-CV RESULTS",
        f"Control CAT+XGB meta-CV AUC: {control_meta_auc:.8f}",
        f"Candidate CAT+XGB+LGBM meta-CV AUC: {candidate_meta_auc:.8f}",
        f"Candidate delta vs control: {meta_gain:+.8f}",
        f"Held folds improved: {folds_improved}/5",
        f"Held folds worse: {folds_worse}/5",
        f"Held folds tied: {folds_tied}/5",
        f"Candidate selected nonzero LGBM weight: {lgb_nonzero_folds}/5 folds",
        "",
        "MEAN HELD-FOLD-SELECTED DEPLOYMENT WEIGHTS",
        f"Control CAT: {control_mean_weights[0]:.4f}",
        f"Control XGB: {control_mean_weights[1]:.4f}",
        f"Candidate CAT: {candidate_mean_weights[0]:.4f}",
        f"Candidate XGB: {candidate_mean_weights[1]:.4f}",
        f"Candidate LGBM: {candidate_mean_weights[2]:.4f}",
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
            f"(CAT/XGB={row.control_cat_weight:.2f}/{row.control_xgb_weight:.2f}) -> "
            f"candidate={row.candidate_held_auc:.8f} "
            f"(CAT/XGB/LGBM="
            f"{row.candidate_cat_weight:.2f}/"
            f"{row.candidate_xgb_weight:.2f}/"
            f"{row.candidate_lgbm_weight:.2f}) "
            f"delta={row.candidate_delta_vs_control:+.8f}"
        )

    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("DIVERSITY AUDIT COMPLETE")
    print("=" * 92)
    print(f"Control CAT+XGB meta-CV     : {control_meta_auc:.8f}")
    print(f"Candidate CAT+XGB+LGBM      : {candidate_meta_auc:.8f}")
    print(f"Delta vs control            : {meta_gain:+.8f}")
    print(f"Folds improved/worse/tied   : {folds_improved}/{folds_worse}/{folds_tied}")
    print(f"LGBM nonzero selected folds : {lgb_nonzero_folds}/5")
    print()
    print(
        "Mean candidate weights      : "
        f"CAT={candidate_mean_weights[0]:.4f}, "
        f"XGB={candidate_mean_weights[1]:.4f}, "
        f"LGBM={candidate_mean_weights[2]:.4f}"
    )
    print(f"Decision                    : {decision}")
    print(f"Artifacts                   : {args.output_dir.resolve()}")
    print("=" * 92)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. meta_fold_metrics.csv")


if __name__ == "__main__":
    main()
