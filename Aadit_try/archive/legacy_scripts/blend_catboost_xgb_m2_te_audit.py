"""
Controlled blend audit: CatBoost 3-seed rank ensemble + XGBoost exact-TE m=2.

HYPOTHESIS
----------
The new XGBoost exact-value Bayesian TE model with smoothing m=2 is materially
stronger than the old m=100 XGBoost component. Re-running the SAME leakage-safe
rank-blend audit may improve the overall ensemble.

IMPORTANT CONTROL
-----------------
Before evaluating m=2, this script first reproduces the PREVIOUS blend audit:

    CatBoost 3-seed + XGBoost exact-TE m=100
    expected meta-CV OOF ≈ 0.94549841
    expected held-fold CAT weights = [0.81, 0.82, 0.82, 0.79, 0.82]

If that control does not reproduce, the script STOPS. This prevents us from
silently changing the blend methodology.

NEW EXPERIMENT
--------------
Only the XGBoost component changes:

OLD:
    XGBoost exact-value Bayesian TE, m=100

NEW:
    XGBoost exact-value Bayesian TE, m=2

Everything else stays fixed:
- frozen 5 folds
- CatBoost 3-seed predictions
- rank blending
- CAT weight grid 0.50 -> 1.00 by 0.01
- for each held fold, weight is chosen using the OTHER FOUR folds only
- no Kaggle leaderboard information is used

Run:
    python blend_catboost_xgb_m2_te_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_OLD_META_AUC = 0.94549841
EXPECTED_OLD_WEIGHTS = [0.81, 0.82, 0.82, 0.79, 0.82]

WEIGHT_GRID = np.round(np.arange(0.50, 1.0001, 0.01), 2)

DEFAULT_FOLDS = Path("artifacts") / "validation" / "candidate_folds.csv"

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

DEFAULT_OLD_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_gpu"
    / "oof_predictions.csv"
)

DEFAULT_NEW_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_smoothing_ultralow_gpu"
    / "oof_predictions_m2.csv"
)

DEFAULT_NEW_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_exact_value_bayesian_te_smoothing_ultralow_gpu"
    / "best_test_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_xgb_m2_te_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe rank-blend audit using the new XGBoost m=2 model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cat-oof", type=Path, default=DEFAULT_CAT_OOF)
    parser.add_argument("--cat-test", type=Path, default=DEFAULT_CAT_TEST)
    parser.add_argument("--old-xgb-oof", type=Path, default=DEFAULT_OLD_XGB_OOF)
    parser.add_argument("--new-xgb-oof", type=Path, default=DEFAULT_NEW_XGB_OOF)
    parser.add_argument("--new-xgb-test", type=Path, default=DEFAULT_NEW_XGB_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def encode_target(y: pd.Series) -> np.ndarray:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise ValueError(f"Expected binary target, found {values}")

    for value in values:
        if str(value).strip().lower() in {
            "yes", "true", "1", "positive", "buy", "will_buy"
        }:
            positive = value
            break
    else:
        positive = y.value_counts().idxmin()

    return (y == positive).astype(np.int8).to_numpy()


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    if not {"row_index", "fold"}.issubset(folds.columns):
        raise ValueError("Frozen fold file must contain row_index and fold.")

    if len(folds) != len(train):
        raise ValueError("Frozen fold row count does not match train.csv.")

    expected_rows = np.arange(len(train), dtype=np.int64)
    if not np.array_equal(folds["row_index"].to_numpy(), expected_rows):
        raise ValueError("Frozen fold row order does not match train.csv.")

    fold_ids = folds["fold"].to_numpy(dtype=np.int16)
    if sorted(np.unique(fold_ids).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected frozen folds [0,1,2,3,4].")

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

    return fold_ids, id_col


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"OOF file not found: {path.resolve()}")

    df = pd.read_csv(path)
    required = {"row_index", "fold", "oof_prediction"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} must contain {sorted(required)}. Found {list(df.columns)}"
        )

    if len(df) != len(train):
        raise ValueError(f"OOF row count mismatch in {path}")

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(f"OOF row order mismatch in {path}")

    if not np.array_equal(df["fold"].to_numpy(), fold_ids):
        raise ValueError(f"OOF fold mismatch in {path}")

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(df[id_col].to_numpy(), train[id_col].to_numpy())
    ):
        raise ValueError(f"OOF ID mismatch in {path}")

    pred = df["oof_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite OOF predictions in {path}")

    return pred


def load_test_prediction(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Test prediction file not found: {path.resolve()}")

    df = pd.read_csv(path)
    if len(df) != len(test):
        raise ValueError(f"Test prediction row count mismatch in {path}")

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(df[id_col].to_numpy(), test[id_col].to_numpy())
    ):
        raise ValueError(f"Test prediction IDs do not align in {path}")

    preferred = [
        "prediction",
        "test_prediction",
        "oof_prediction",
        "pred",
        "probability",
    ]
    pred_col = next((c for c in preferred if c in df.columns), None)

    if pred_col is None:
        excluded = {id_col} if id_col is not None else set()
        candidates = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify prediction column in {path}. "
                f"Columns: {list(df.columns)}"
            )
        pred_col = candidates[0]

    pred = df[pred_col].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite test predictions in {path}")

    return pred


def audit_rank_blend(
    *,
    y: np.ndarray,
    fold_ids: np.ndarray,
    cat_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    # Rank transform uses predictions only, not targets.
    cat_rank = rank01(cat_pred)
    xgb_rank = rank01(xgb_pred)

    held_meta_pred = np.full(len(y), np.nan, dtype=np.float64)
    rows = []

    for held_fold in range(5):
        selection_mask = fold_ids != held_fold
        held_mask = fold_ids == held_fold

        selection_scores = []
        for cat_weight in WEIGHT_GRID:
            blend = (
                cat_weight * cat_rank[selection_mask]
                + (1.0 - cat_weight) * xgb_rank[selection_mask]
            )
            auc = float(roc_auc_score(y[selection_mask], blend))
            selection_scores.append(auc)

        best_idx = int(np.argmax(selection_scores))
        selected_cat_weight = float(WEIGHT_GRID[best_idx])
        selected_xgb_weight = 1.0 - selected_cat_weight

        held_blend = (
            selected_cat_weight * cat_rank[held_mask]
            + selected_xgb_weight * xgb_rank[held_mask]
        )
        held_meta_pred[held_mask] = held_blend

        held_y = y[held_mask]
        cat_auc = float(roc_auc_score(held_y, cat_rank[held_mask]))
        xgb_auc = float(roc_auc_score(held_y, xgb_rank[held_mask]))
        blend_auc = float(roc_auc_score(held_y, held_blend))

        rows.append(
            {
                "held_fold": held_fold,
                "selected_cat_weight": selected_cat_weight,
                "selected_xgb_weight": selected_xgb_weight,
                "selection_auc_other_4_folds": float(selection_scores[best_idx]),
                "held_cat_auc": cat_auc,
                "held_xgb_auc": xgb_auc,
                "held_blend_auc": blend_auc,
                "blend_delta_vs_cat": blend_auc - cat_auc,
                "blend_delta_vs_xgb": blend_auc - xgb_auc,
            }
        )

    if np.isnan(held_meta_pred).any():
        raise RuntimeError("Meta-CV blend predictions contain NaNs.")

    meta_auc = float(roc_auc_score(y, held_meta_pred))
    return pd.DataFrame(rows), held_meta_pred, meta_auc


def full_oof_rank_blend_auc(
    y: np.ndarray,
    cat_pred: np.ndarray,
    xgb_pred: np.ndarray,
    cat_weight: float,
) -> float:
    cat_rank = rank01(cat_pred)
    xgb_rank = rank01(xgb_pred)
    blend = cat_weight * cat_rank + (1.0 - cat_weight) * xgb_rank
    return float(roc_auc_score(y, blend))


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
        args.old_xgb_oof,
        args.new_xgb_oof,
        args.new_xgb_test,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

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

    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise ValueError(f"Expected one train-only target column, found {train_only}")
    target = train_only[0]
    y = encode_target(train[target])

    fold_ids, id_col = validate_folds(folds, train)

    cat_oof = load_oof(args.cat_oof, train, fold_ids, id_col)
    old_xgb_oof = load_oof(args.old_xgb_oof, train, fold_ids, id_col)
    new_xgb_oof = load_oof(args.new_xgb_oof, train, fold_ids, id_col)

    cat_auc = float(roc_auc_score(y, cat_oof))
    old_xgb_auc = float(roc_auc_score(y, old_xgb_oof))
    new_xgb_auc = float(roc_auc_score(y, new_xgb_oof))

    print("=" * 86)
    print("CATBOOST + XGBOOST m=2 LEAKAGE-SAFE RANK-BLEND AUDIT")
    print("=" * 86)
    print(f"Frozen fold SHA256 verified : {fold_hash}")
    print(f"CatBoost 3-seed OOF         : {cat_auc:.8f}")
    print(f"Old XGB m=100 OOF           : {old_xgb_auc:.8f}")
    print(f"New XGB m=2 OOF             : {new_xgb_auc:.8f}")
    print(f"New XGB delta vs CatBoost   : {new_xgb_auc - cat_auc:+.8f}")
    print()

    # ---------------------------------------------------------
    # CONTROL: reproduce the old blend methodology first.
    # ---------------------------------------------------------
    print("CONTROL: reproducing previous CatBoost + XGB m=100 blend audit...")
    old_folds, old_meta_pred, old_meta_auc = audit_rank_blend(
        y=y,
        fold_ids=fold_ids,
        cat_pred=cat_oof,
        xgb_pred=old_xgb_oof,
    )

    old_weights = old_folds["selected_cat_weight"].round(2).tolist()

    print(f"Reproduced old meta-CV OOF : {old_meta_auc:.8f}")
    print(f"Expected old meta-CV OOF   : {EXPECTED_OLD_META_AUC:.8f}")
    print(f"Reproduced old weights     : {old_weights}")
    print(f"Expected old weights       : {EXPECTED_OLD_WEIGHTS}")
    print()

    score_ok = abs(old_meta_auc - EXPECTED_OLD_META_AUC) <= 1e-7
    weights_ok = old_weights == EXPECTED_OLD_WEIGHTS

    if not (score_ok and weights_ok):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        old_folds.to_csv(
            args.output_dir / "old_control_fold_metrics.csv",
            index=False,
        )
        raise RuntimeError(
            "STOP: previous blend audit did not reproduce exactly enough. "
            "Do not trust the new blend result yet. Send this terminal output "
            "and old_control_fold_metrics.csv so the SAME file can be fixed."
        )

    print("CONTROL PASSED.")
    print()

    # ---------------------------------------------------------
    # NEW EXPERIMENT: swap only m=100 -> m=2.
    # ---------------------------------------------------------
    print("NEW AUDIT: CatBoost + XGB m=2...")
    new_folds, new_meta_pred, new_meta_auc = audit_rank_blend(
        y=y,
        fold_ids=fold_ids,
        cat_pred=cat_oof,
        xgb_pred=new_xgb_oof,
    )

    selected_weights = new_folds["selected_cat_weight"].to_numpy(dtype=float)
    mean_cat_weight = float(selected_weights.mean())
    std_cat_weight = float(selected_weights.std(ddof=0))

    # Match the previous deployment convention: round the mean selected
    # weight to the same 0.01 grid.
    deployment_cat_weight = float(
        WEIGHT_GRID[np.argmin(np.abs(WEIGHT_GRID - mean_cat_weight))]
    )
    deployment_xgb_weight = 1.0 - deployment_cat_weight

    full_oof_auc = full_oof_rank_blend_auc(
        y=y,
        cat_pred=cat_oof,
        xgb_pred=new_xgb_oof,
        cat_weight=deployment_cat_weight,
    )

    folds_beating_cat = int((new_folds["blend_delta_vs_cat"] > 0).sum())
    folds_beating_xgb = int((new_folds["blend_delta_vs_xgb"] > 0).sum())

    delta_vs_old_meta = new_meta_auc - old_meta_auc
    delta_vs_new_xgb = new_meta_auc - new_xgb_auc

    print(new_folds.to_string(index=False))
    print()
    print(f"New meta-CV blended OOF     : {new_meta_auc:.8f}")
    print(f"Previous meta-CV blended OOF: {old_meta_auc:.8f}")
    print(f"Delta vs previous blend     : {delta_vs_old_meta:+.8f}")
    print(f"Delta vs XGB m=2 single     : {delta_vs_new_xgb:+.8f}")
    print(f"Folds blend > CatBoost      : {folds_beating_cat}/5")
    print(f"Folds blend > XGB m=2       : {folds_beating_xgb}/5")
    print(f"Selected CAT weights        : {selected_weights.round(2).tolist()}")
    print(f"Mean CAT weight             : {mean_cat_weight:.4f}")
    print(f"Std CAT weight              : {std_cat_weight:.4f}")
    print(
        f"Deployment rank blend       : "
        f"{deployment_cat_weight:.2f} CAT / {deployment_xgb_weight:.2f} XGB"
    )
    print(f"Full-OOF deployment diagnostic: {full_oof_auc:.8f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_folds.to_csv(
        args.output_dir / "old_control_fold_metrics.csv",
        index=False,
    )
    new_folds.to_csv(
        args.output_dir / "new_m2_fold_metrics.csv",
        index=False,
    )

    meta_oof = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold_ids,
            "target_encoded": y,
            "meta_oof_prediction": new_meta_pred.astype(np.float32),
        }
    )
    if id_col is not None:
        meta_oof.insert(1, id_col, train[id_col].to_numpy())
    meta_oof.to_csv(
        args.output_dir / "meta_oof_predictions.csv",
        index=False,
    )

    # Package test predictions for later submission creation, but DO NOT create
    # a Kaggle submission here.
    cat_test = load_test_prediction(args.cat_test, test, id_col)
    new_xgb_test = load_test_prediction(args.new_xgb_test, test, id_col)

    cat_test_rank = rank01(cat_test)
    xgb_test_rank = rank01(new_xgb_test)
    blended_test = (
        deployment_cat_weight * cat_test_rank
        + deployment_xgb_weight * xgb_test_rank
    )

    test_output = pd.DataFrame(
        {"prediction": blended_test.astype(np.float32)}
    )
    if id_col is not None:
        test_output.insert(0, id_col, test[id_col].to_numpy())
    test_output.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    if (
        new_meta_auc > old_meta_auc
        and new_meta_auc > new_xgb_auc
        and folds_beating_xgb >= 3
    ):
        decision = "KEEP_NEW_BLEND"
    elif new_xgb_auc >= new_meta_auc:
        decision = "KEEP_XGB_M2_SINGLE_FOR_NOW"
    else:
        decision = "NO_CLEAR_IMPROVEMENT"

    summary = [
        "EXPERIMENT: CATBOOST + XGBOOST m=2 RANK-BLEND AUDIT",
        "=" * 78,
        "",
        "HYPOTHESIS",
        "Does replacing the old m=100 XGBoost component with the stronger m=2",
        "exact-TE XGBoost improve the leakage-safe CatBoost/XGBoost ensemble?",
        "",
        "CONTROL REPRODUCTION",
        f"Old meta-CV expected: {EXPECTED_OLD_META_AUC:.8f}",
        f"Old meta-CV reproduced: {old_meta_auc:.8f}",
        f"Old weights reproduced: {old_weights}",
        "CONTROL: PASS",
        "",
        "SINGLE MODELS",
        f"CatBoost 3-seed OOF: {cat_auc:.8f}",
        f"XGB m=100 OOF: {old_xgb_auc:.8f}",
        f"XGB m=2 OOF: {new_xgb_auc:.8f}",
        f"XGB m=2 delta vs CatBoost: {new_xgb_auc - cat_auc:+.8f}",
        "",
        "NEW META-CV BLEND",
        f"Meta-CV OOF: {new_meta_auc:.8f}",
        f"Delta vs previous meta-CV blend: {delta_vs_old_meta:+.8f}",
        f"Delta vs XGB m=2 single: {delta_vs_new_xgb:+.8f}",
        f"Folds blend > CatBoost: {folds_beating_cat}/5",
        f"Folds blend > XGB m=2: {folds_beating_xgb}/5",
        f"Selected CAT weights: {selected_weights.round(2).tolist()}",
        f"Mean CAT weight: {mean_cat_weight:.4f}",
        f"Std CAT weight: {std_cat_weight:.4f}",
        (
            f"Deployment blend: {deployment_cat_weight:.2f} CAT / "
            f"{deployment_xgb_weight:.2f} XGB"
        ),
        f"Full-OOF deployment diagnostic: {full_oof_auc:.8f}",
        "",
        f"DECISION: {decision}",
        "",
        "Do not submit to Kaggle yet. Analyze these local results first.",
    ]

    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print(f"Decision                   : {decision}")
    print(f"Artifacts                  : {args.output_dir.resolve()}")
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. old_control_fold_metrics.csv")
    print("  4. new_m2_fold_metrics.csv")


if __name__ == "__main__":
    main()
