"""
Controlled blend audit: CatBoost 3-seed rank ensemble + recipe-base-margin XGBoost.

HYPOTHESIS
----------
The recipe-base-margin XGBoost is slightly stronger than the validated
income-digit XGBoost and is also a little less rank-correlated with CatBoost.
Replacing the income-digit XGBoost component may therefore improve the
leakage-safe CatBoost/XGBoost blend.

CONTROL
-------
Before testing the new component, reproduce the current locally validated
CatBoost + income-digit XGBoost blend:

    meta-CV OOF = 0.94577587
    held-fold CAT weights = [0.30, 0.32, 0.30, 0.29, 0.34]

If this does not reproduce, STOP.

NEW EXPERIMENT
--------------
Only the XGBoost component changes:

OLD:
    XGBoost m=2 + income digit decomposition

NEW:
    XGBoost m=2 + income digit decomposition
    + original-recipe base margin

Everything else stays fixed:
- frozen 5 folds
- CatBoost 3-seed predictions
- rank blending
- CAT weight grid 0.00 -> 1.00 by 0.01
- for each held fold, weight is selected on the OTHER FOUR folds only
- deployment convention = mean selected CAT weight rounded to the 0.01 grid
- no leaderboard information is used

Run:
    python blend_catboost_recipe_margin_xgb_audit.py
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

EXPECTED_RESTRICTED_META_AUC = 0.94577587
EXPECTED_RESTRICTED_WEIGHTS = [0.30, 0.32, 0.30, 0.29, 0.34]

RESTRICTED_GRID = np.round(np.arange(0.00, 1.0001, 0.01), 2)
EXPANDED_GRID = np.round(np.arange(0.00, 1.0001, 0.01), 2)

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

DEFAULT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digits_recipe_base_margin_gpu"
    / "oof_predictions.csv"
)

DEFAULT_XGB_TEST = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digits_recipe_base_margin_gpu"
    / "test_predictions.csv"
)

DEFAULT_CONTROL_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_income_digit_decomposition_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_recipe_margin_xgb_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe expanded rank-blend weight audit for "
            "CatBoost + XGBoost exact-TE m=2."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cat-oof", type=Path, default=DEFAULT_CAT_OOF)
    parser.add_argument("--cat-test", type=Path, default=DEFAULT_CAT_TEST)
    parser.add_argument("--xgb-oof", type=Path, default=DEFAULT_XGB_OOF)
    parser.add_argument("--xgb-test", type=Path, default=DEFAULT_XGB_TEST)
    parser.add_argument(
        "--control-xgb-oof",
        type=Path,
        default=DEFAULT_CONTROL_XGB_OOF,
    )
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

    positive = None
    for value in values:
        if str(value).strip().lower() in {
            "yes",
            "true",
            "1",
            "positive",
            "buy",
            "will_buy",
        }:
            positive = value
            break

    if positive is None:
        positive = y.value_counts().idxmin()

    return (y == positive).astype(np.int8).to_numpy()


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}
    if not required.issubset(folds.columns):
        raise ValueError(
            f"Frozen fold file missing columns: {sorted(required - set(folds.columns))}"
        )

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
        numeric_candidates = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

        if len(numeric_candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify prediction column in {path}. "
                f"Columns: {list(df.columns)}"
            )

        pred_col = numeric_candidates[0]

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
    weight_grid: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    cat_rank = rank01(cat_pred)
    xgb_rank = rank01(xgb_pred)

    meta_pred = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict] = []

    for held_fold in range(5):
        selection_mask = fold_ids != held_fold
        held_mask = fold_ids == held_fold

        selection_scores = np.empty(len(weight_grid), dtype=np.float64)

        for i, cat_weight in enumerate(weight_grid):
            blend = (
                cat_weight * cat_rank[selection_mask]
                + (1.0 - cat_weight) * xgb_rank[selection_mask]
            )
            selection_scores[i] = roc_auc_score(
                y[selection_mask],
                blend,
            )

        best_idx = int(np.argmax(selection_scores))
        cat_weight = float(weight_grid[best_idx])
        xgb_weight = 1.0 - cat_weight

        held_blend = (
            cat_weight * cat_rank[held_mask]
            + xgb_weight * xgb_rank[held_mask]
        )
        meta_pred[held_mask] = held_blend

        held_y = y[held_mask]

        cat_auc = float(
            roc_auc_score(
                held_y,
                cat_rank[held_mask],
            )
        )
        xgb_auc = float(
            roc_auc_score(
                held_y,
                xgb_rank[held_mask],
            )
        )
        blend_auc = float(
            roc_auc_score(
                held_y,
                held_blend,
            )
        )

        rows.append(
            {
                "held_fold": held_fold,
                "selected_cat_weight": cat_weight,
                "selected_xgb_weight": xgb_weight,
                "selection_auc_other_4_folds": float(
                    selection_scores[best_idx]
                ),
                "held_cat_auc": cat_auc,
                "held_xgb_auc": xgb_auc,
                "held_blend_auc": blend_auc,
                "blend_delta_vs_cat": blend_auc - cat_auc,
                "blend_delta_vs_xgb": blend_auc - xgb_auc,
            }
        )

    if np.isnan(meta_pred).any():
        raise RuntimeError("Meta-CV predictions contain NaNs.")

    meta_auc = float(
        roc_auc_score(
            y,
            meta_pred,
        )
    )

    return (
        pd.DataFrame(rows),
        meta_pred,
        meta_auc,
    )


def full_oof_rank_blend_auc(
    *,
    y: np.ndarray,
    cat_pred: np.ndarray,
    xgb_pred: np.ndarray,
    cat_weight: float,
) -> float:
    cat_rank = rank01(cat_pred)
    xgb_rank = rank01(xgb_pred)

    blend = (
        cat_weight * cat_rank
        + (1.0 - cat_weight) * xgb_rank
    )

    return float(
        roc_auc_score(
            y,
            blend,
        )
    )


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
        args.control_xgb_oof,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
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

    train_only = [
        c
        for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            f"Expected exactly one train-only target column, found {train_only}"
        )

    target = train_only[0]
    y = encode_target(train[target])

    fold_ids, id_col = validate_folds(
        folds,
        train,
    )

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

    control_xgb_oof = load_oof(
        args.control_xgb_oof,
        train,
        fold_ids,
        id_col,
    )

    cat_auc = float(
        roc_auc_score(
            y,
            cat_oof,
        )
    )

    xgb_auc = float(
        roc_auc_score(
            y,
            xgb_oof,
        )
    )

    print("=" * 88)
    print("CATBOOST + RECIPE-BASE-MARGIN XGBOOST LEAKAGE-SAFE RANK-BLEND AUDIT")
    print("=" * 88)
    print(f"Frozen fold SHA256 verified : {fold_hash}")
    print(f"CatBoost 3-seed OOF         : {cat_auc:.8f}")
    print(f"Recipe-margin XGBoost OOF   : {xgb_auc:.8f}")
    print()

    # ---------------------------------------------------------
    # CONTROL: reproduce the previous 0.50 -> 1.00 audit first.
    # ---------------------------------------------------------
    print("CONTROL: reproducing current CatBoost + income-digit XGBoost blend...")

    (
        restricted_folds,
        restricted_meta_pred,
        restricted_meta_auc,
    ) = audit_rank_blend(
        y=y,
        fold_ids=fold_ids,
        cat_pred=cat_oof,
        xgb_pred=control_xgb_oof,
        weight_grid=RESTRICTED_GRID,
    )

    restricted_weights = (
        restricted_folds[
            "selected_cat_weight"
        ]
        .round(2)
        .tolist()
    )

    print(
        f"Reproduced control meta-CV : "
        f"{restricted_meta_auc:.8f}"
    )
    print(
        f"Expected control meta-CV   : "
        f"{EXPECTED_RESTRICTED_META_AUC:.8f}"
    )
    print(
        f"Reproduced control weights : "
        f"{restricted_weights}"
    )
    print(
        f"Expected control weights   : "
        f"{EXPECTED_RESTRICTED_WEIGHTS}"
    )
    print()

    score_ok = (
        abs(
            restricted_meta_auc
            - EXPECTED_RESTRICTED_META_AUC
        )
        <= 1e-7
    )

    weights_ok = (
        restricted_weights
        == EXPECTED_RESTRICTED_WEIGHTS
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    restricted_folds.to_csv(
        args.output_dir
        / "income_digit_control_fold_metrics.csv",
        index=False,
    )

    if not (
        score_ok
        and weights_ok
    ):
        raise RuntimeError(
            "STOP: current income-digit blend control did not reproduce. "
            "Do not trust the expanded audit. Send the terminal output "
            "and m2_control_fold_metrics.csv so this SAME file "
            "can be fixed."
        )

    print("CONTROL PASSED.")
    print()

    # ---------------------------------------------------------
    # EXPERIMENT: expand CAT weights down to zero.
    # ---------------------------------------------------------
    print("NEW AUDIT: CatBoost + recipe-margin XGBoost, CAT grid 0.00 -> 1.00...")

    (
        expanded_folds,
        expanded_meta_pred,
        expanded_meta_auc,
    ) = audit_rank_blend(
        y=y,
        fold_ids=fold_ids,
        cat_pred=cat_oof,
        xgb_pred=xgb_oof,
        weight_grid=EXPANDED_GRID,
    )

    expanded_weights = (
        expanded_folds[
            "selected_cat_weight"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mean_cat_weight = float(
        expanded_weights.mean()
    )

    std_cat_weight = float(
        expanded_weights.std(
            ddof=0
        )
    )

    deployment_cat_weight = float(
        EXPANDED_GRID[
            np.argmin(
                np.abs(
                    EXPANDED_GRID
                    - mean_cat_weight
                )
            )
        ]
    )

    deployment_xgb_weight = (
        1.0
        - deployment_cat_weight
    )

    full_oof_deployment_auc = (
        full_oof_rank_blend_auc(
            y=y,
            cat_pred=cat_oof,
            xgb_pred=xgb_oof,
            cat_weight=deployment_cat_weight,
        )
    )

    folds_blend_beats_cat = int(
        (
            expanded_folds[
                "blend_delta_vs_cat"
            ] > 0
        ).sum()
    )

    folds_blend_beats_xgb = int(
        (
            expanded_folds[
                "blend_delta_vs_xgb"
            ] > 0
        ).sum()
    )

    delta_vs_restricted = (
        expanded_meta_auc
        - restricted_meta_auc
    )

    delta_vs_xgb = (
        expanded_meta_auc
        - xgb_auc
    )

    print(
        expanded_folds.to_string(
            index=False
        )
    )
    print()
    print(
        f"New meta-CV OOF         : "
        f"{expanded_meta_auc:.8f}"
    )
    print(
        f"Control m=2 blend OOF       : "
        f"{restricted_meta_auc:.8f}"
    )
    print(
        f"Delta vs income-digit blend   : "
        f"{delta_vs_restricted:+.8f}"
    )
    print(
        f"Delta vs recipe-margin XGB   : "
        f"{delta_vs_xgb:+.8f}"
    )
    print(
        f"Folds blend > CatBoost       : "
        f"{folds_blend_beats_cat}/5"
    )
    print(
        f"Folds blend > recipe-margin XGB       : "
        f"{folds_blend_beats_xgb}/5"
    )
    print(
        f"Selected CAT weights         : "
        f"{expanded_weights.round(2).tolist()}"
    )
    print(
        f"Mean CAT weight              : "
        f"{mean_cat_weight:.4f}"
    )
    print(
        f"Std CAT weight               : "
        f"{std_cat_weight:.4f}"
    )
    print(
        f"Deployment rank blend        : "
        f"{deployment_cat_weight:.2f} CAT / "
        f"{deployment_xgb_weight:.2f} XGB"
    )
    print(
        f"Full-OOF deployment diagnostic: "
        f"{full_oof_deployment_auc:.8f}"
    )

    expanded_folds.to_csv(
        args.output_dir
        / "recipe_margin_blend_fold_metrics.csv",
        index=False,
    )

    meta_oof = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "meta_oof_prediction": (
                expanded_meta_pred.astype(
                    np.float32
                )
            ),
        }
    )

    if id_col is not None:
        meta_oof.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    meta_oof.to_csv(
        args.output_dir
        / "meta_oof_predictions.csv",
        index=False,
    )

    # Save test predictions for a possible later submission,
    # but do NOT create a Kaggle submission CSV here.
    cat_test = load_test_prediction(
        args.cat_test,
        test,
        id_col,
    )

    xgb_test = load_test_prediction(
        args.xgb_test,
        test,
        id_col,
    )

    cat_test_rank = rank01(
        cat_test
    )

    xgb_test_rank = rank01(
        xgb_test
    )

    blended_test = (
        deployment_cat_weight
        * cat_test_rank
        + deployment_xgb_weight
        * xgb_test_rank
    )

    test_output = pd.DataFrame(
        {
            "prediction": (
                blended_test.astype(
                    np.float32
                )
            )
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[
                id_col
            ].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    # Conservative interpretation.
    if (
        expanded_meta_auc
        > restricted_meta_auc
        and expanded_meta_auc
        > xgb_auc
        and folds_blend_beats_xgb >= 3
    ):
        decision = "KEEP_RECIPE_MARGIN_BLEND"
    elif (
        abs(
            expanded_meta_auc
            - xgb_auc
        )
        <= 0.00002
    ):
        decision = "NO_CLEAR_BLEND_GAIN_OVER_RECIPE_MARGIN_XGB"
    elif (
        expanded_meta_auc
        <= xgb_auc
    ):
        decision = "KEEP_RECIPE_MARGIN_XGB_SINGLE_FOR_NOW"
    else:
        decision = "NO_CLEAR_IMPROVEMENT"

    boundary_note = (
        "YES"
        if (
            np.isclose(
                expanded_weights.min(),
                EXPANDED_GRID.min(),
            )
            or np.isclose(
                expanded_weights.max(),
                EXPANDED_GRID.max(),
            )
        )
        else "NO"
    )

    summary = [
        "EXPERIMENT: CATBOOST + RECIPE-BASE-MARGIN XGBOOST RANK-BLEND AUDIT",
        "=" * 78,
        "",
        "HYPOTHESIS",
        "Does replacing the income-digit XGBoost component with the slightly",
        "stronger recipe-base-margin XGBoost improve the leakage-safe blend?",
        "",
        "ONLY CHANGE",
        "XGBoost component: income digits -> income digits + recipe base margin",
        "",
        "CONTROL",
        f"Income-digit control meta-CV expected: {EXPECTED_RESTRICTED_META_AUC:.8f}",
        f"Income-digit control meta-CV reproduced: {restricted_meta_auc:.8f}",
        f"Income-digit control weights reproduced: {restricted_weights}",
        "CONTROL: PASS",
        "",
        "SINGLE MODELS",
        f"CatBoost 3-seed OOF: {cat_auc:.8f}",
        f"Recipe-margin XGBoost OOF: {xgb_auc:.8f}",
        "",
        "NEW META-CV BLEND",
        f"Meta-CV OOF: {expanded_meta_auc:.8f}",
        f"Delta vs current income-digit blend: {delta_vs_restricted:+.8f}",
        f"Delta vs recipe-margin XGB single: {delta_vs_xgb:+.8f}",
        f"Folds blend > CatBoost: {folds_blend_beats_cat}/5",
        f"Folds blend > recipe-margin XGB: {folds_blend_beats_xgb}/5",
        f"Selected CAT weights: {expanded_weights.round(2).tolist()}",
        f"Mean CAT weight: {mean_cat_weight:.4f}",
        f"Std CAT weight: {std_cat_weight:.4f}",
        (
            f"Deployment blend: "
            f"{deployment_cat_weight:.2f} CAT / "
            f"{deployment_xgb_weight:.2f} XGB"
        ),
        f"Full-OOF deployment diagnostic: {full_oof_deployment_auc:.8f}",
        f"Selected weights hit a search boundary: {boundary_note}",
        "",
        f"DECISION: {decision}",
        "",
        "Do not submit to Kaggle yet. Analyze the local result first.",
    ]

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Decision                    : "
        f"{decision}"
    )
    print(
        f"Selection hit boundary      : "
        f"{boundary_note}"
    )
    print(
        f"Artifacts                   : "
        f"{args.output_dir.resolve()}"
    )
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. income_digit_control_fold_metrics.csv")
    print("  4. recipe_margin_blend_fold_metrics.csv")


if __name__ == "__main__":
    main()
