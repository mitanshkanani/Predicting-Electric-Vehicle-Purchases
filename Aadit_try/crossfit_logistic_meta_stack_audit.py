"""
crossfit_logistic_meta_stack_audit.py

Kaggle Playground Series S6E9

NO BASE-MODEL TRAINING.
NO BLEND-WEIGHT GRID.
NO LEADERBOARD USAGE.

Question
--------
Can a cross-fitted linear meta-model combine our already-generated model views
better than the current validated rank-blend champion?

Current champion:
    0.94611253 OOF

META INPUTS
-----------
1. old CatBoost income+commute 3-seed
2. fine-income CatBoost 3-seed
3. fine-income XGBoost
4. old engineered LightGBM
5. fine-income LightGBM
6. RealMLP

The old + fine variants are both included intentionally:
- old versions may preserve diversity
- fine versions may contribute stronger local-income signal

LEAKAGE SAFETY
--------------
For every held frozen fold:
- each base input is already OOF for every row
- each base prediction is converted to a percentile rank WITHIN its original fold
- LogisticRegression is fit only on the other four frozen folds
- the held fold is scored once
- no held-fold target enters meta fitting

No meta hyperparameter search is performed.

Fixed meta learner:
    LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000
    )

Run:
    python crossfit_logistic_meta_stack_audit.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
FOLDS_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "candidate_folds.csv"
)

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

EXPECTED = {
    "old_cat": 0.94558572,
    "fine_cat": 0.94578633,
    "fine_xgb": 0.94606664,
    "old_lgbm": 0.94578042,
    "fine_lgbm": 0.94595207,
    "realmlp": 0.94329901,
    "champion": 0.94611253,
}

AUC_TOL = 2e-5

OOF_PATHS = {
    "old_cat": (
        ROOT
        / "artifacts"
        / "experiments"
        / "catboost_hierarchical_income_commute_multiseed_gpu"
        / "best_average_oof_predictions.csv"
    ),
    "fine_cat": (
        ROOT
        / "artifacts"
        / "experiments"
        / "catboost_fine_income_multiseed_gpu"
        / "best_average_oof_predictions.csv"
    ),
    "fine_xgb": (
        ROOT
        / "artifacts"
        / "experiments"
        / "xgboost_fine_income_te_gpu"
        / "oof_predictions.csv"
    ),
    "old_lgbm": (
        ROOT
        / "artifacts"
        / "experiments"
        / "lightgbm_engineered_learned_margin_cpu"
        / "oof_predictions.csv"
    ),
    "fine_lgbm": (
        ROOT
        / "artifacts"
        / "experiments"
        / "lightgbm_fine_income_te_cpu"
        / "oof_predictions.csv"
    ),
    "realmlp": (
        ROOT
        / "artifacts"
        / "experiments"
        / "realmlp_frozen5_vectorized_gpu"
        / "oof_predictions.csv"
    ),
    "champion": (
        ROOT
        / "artifacts"
        / "experiments"
        / "blend_fine_income_xgb_validated_submission"
        / "oof_predictions.csv"
    ),
}

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "crossfit_logistic_meta_stack_audit"
)

META_FEATURES = [
    "old_cat",
    "fine_cat",
    "fine_xgb",
    "old_lgbm",
    "fine_lgbm",
    "realmlp",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred))


def prediction_column(df: pd.DataFrame, path: Path) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "meta_oof_prediction",
        "prediction",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    excluded = {
        "id",
        "row_index",
        "fold",
        "target",
        "target_encoded",
    }

    candidates = [
        c
        for c in df.columns
        if (
            c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify prediction column in {path}\n"
            f"Candidates={candidates}\n"
            f"Columns={list(df.columns)}"
        )

    return candidates[0]


def load_oof(
    path: Path,
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise RuntimeError(
            f"Row-count mismatch in {path}: "
            f"{len(df):,} vs {len(train):,}"
        )

    if "row_index" in df.columns:
        expected = np.arange(len(train), dtype=np.int64)

        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError(
                f"row_index mismatch in {path}"
            )

    if "fold" in df.columns:
        if not np.array_equal(
            df["fold"].to_numpy(dtype=int),
            folds,
        ):
            raise RuntimeError(
                f"fold mismatch in {path}"
            )

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise RuntimeError(
                f"id mismatch in {path}"
            )

    pred = pd.to_numeric(
        df[prediction_column(df, path)],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError(
            f"Non-finite OOF values in {path}"
        )

    return pred


def foldwise_rank(
    values: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    out = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    for fold in range(5):
        mask = folds == fold

        out[mask] = (
            pd.Series(values[mask])
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64)
        )

    if np.isnan(out).any():
        raise RuntimeError(
            "Foldwise rank produced NaNs."
        )

    return out


def main() -> None:
    print("=" * 104)
    print("CROSS-FITTED LOGISTIC META-STACK AUDIT")
    print("NO BASE-MODEL TRAINING | NO WEIGHT GRID | NO LEADERBOARD USAGE")
    print("=" * 104)

    for path in [
        TRAIN_PATH,
        FOLDS_PATH,
        *OOF_PATHS.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(FOLDS_PATH)

    if fold_hash != EXPECTED_FOLD_SHA:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)

    if "fold" not in folds_df.columns:
        raise RuntimeError(
            "Frozen fold file has no 'fold' column."
        )

    if len(train) != len(folds_df):
        raise RuntimeError(
            "Train/fold row-count mismatch."
        )

    folds = folds_df[
        "fold"
    ].to_numpy(dtype=int)

    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected fold IDs: "
            f"{sorted(np.unique(folds).tolist())}"
        )

    y = (
        train[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(POSITIVE_LABEL.lower())
        .astype(np.int8)
        .to_numpy()
    )

    predictions: dict[str, np.ndarray] = {}

    print(
        f"\nFrozen fold SHA256 verified: {fold_hash}"
    )
    print("\n--- Base OOF integrity ---")

    for name, path in OOF_PATHS.items():
        pred = load_oof(
            path,
            train,
            folds,
        )

        score = auc(
            y,
            pred,
        )

        expected = EXPECTED[
            name
        ]

        if abs(
            score
            - expected
        ) > AUC_TOL:
            raise RuntimeError(
                f"{name} AUC mismatch: "
                f"{score:.8f} vs expected {expected:.8f}"
            )

        predictions[
            name
        ] = pred

        print(
            f"{name:10s}: "
            f"{score:.8f}"
        )

    # Normalize each model within each frozen fold.
    ranked = {
        name: foldwise_rank(
            predictions[name],
            folds,
        )
        for name in META_FEATURES
    }

    X = np.column_stack(
        [
            ranked[name]
            for name in META_FEATURES
        ]
    )

    champion_meta = foldwise_rank(
        predictions["champion"],
        folds,
    )

    champion_meta_auc = auc(
        y,
        champion_meta,
    )

    meta_oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    coefficient_rows = []
    fold_rows = []

    print()
    print("--- Held-fold meta evaluation ---")
    print(
        "Meta features: "
        + ", ".join(META_FEATURES)
    )

    for held_fold in range(5):
        fit_mask = (
            folds != held_fold
        )

        held_mask = (
            folds == held_fold
        )

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )

        model.fit(
            X[fit_mask],
            y[fit_mask],
        )

        held_score = model.decision_function(
            X[held_mask]
        )

        meta_oof[
            held_mask
        ] = held_score

        baseline_fold_auc = auc(
            y[held_mask],
            champion_meta[
                held_mask
            ],
        )

        stack_fold_auc = auc(
            y[held_mask],
            held_score,
        )

        delta = (
            stack_fold_auc
            - baseline_fold_auc
        )

        fold_rows.append(
            {
                "held_fold": held_fold,
                "champion_auc": (
                    baseline_fold_auc
                ),
                "stack_auc": (
                    stack_fold_auc
                ),
                "delta": delta,
                "intercept": float(
                    model.intercept_[0]
                ),
            }
        )

        for feature, coef in zip(
            META_FEATURES,
            model.coef_[0],
        ):
            coefficient_rows.append(
                {
                    "held_fold": held_fold,
                    "feature": feature,
                    "coefficient": float(
                        coef
                    ),
                }
            )

        coef_text = ", ".join(
            f"{name}={coef:+.4f}"
            for name, coef
            in zip(
                META_FEATURES,
                model.coef_[0],
            )
        )

        print(
            f"Fold {held_fold}: "
            f"{baseline_fold_auc:.8f} -> "
            f"{stack_fold_auc:.8f} "
            f"({delta:+.8f})"
        )
        print(
            f"          {coef_text}"
        )

    if np.isnan(meta_oof).any():
        raise RuntimeError(
            "Meta OOF contains NaNs."
        )

    stack_auc = auc(
        y,
        meta_oof,
    )

    delta = (
        stack_auc
        - champion_meta_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    coef_by_fold = pd.DataFrame(
        coefficient_rows
    )

    coef_summary = (
        coef_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_coefficient=(
                "coefficient",
                "mean",
            ),
            std_coefficient=(
                "coefficient",
                "std",
            ),
            min_coefficient=(
                "coefficient",
                "min",
            ),
            max_coefficient=(
                "coefficient",
                "max",
            ),
            positive_folds=(
                "coefficient",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            negative_folds=(
                "coefficient",
                lambda x: int(
                    (x < 0).sum()
                ),
            ),
        )
        .sort_values(
            "mean_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    improved = int(
        (
            fold_metrics[
                "delta"
            ] > 0
        ).sum()
    )

    worse = int(
        (
            fold_metrics[
                "delta"
            ] < 0
        ).sum()
    )

    prob_corr = float(
        np.corrcoef(
            meta_oof,
            champion_meta,
        )[0, 1]
    )

    rank_corr = float(
        np.corrcoef(
            pd.Series(meta_oof)
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64),
            pd.Series(champion_meta)
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64),
        )[0, 1]
    )

    if (
        delta >= 0.00002
        and improved >= 4
    ):
        primitive = (
            "POSITIVE_META_STACK_SIGNAL"
        )
    elif (
        delta < 0
        or worse >= 4
    ):
        primitive = (
            "NEGATIVE_META_STACK_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_META_STACK_SIGNAL"
        )

    print()
    print("=" * 104)
    print("META-STACK RESULT")
    print("=" * 104)
    print(
        f"Champion meta AUC  : "
        f"{champion_meta_auc:.8f}"
    )
    print(
        f"Logistic stack AUC : "
        f"{stack_auc:.8f}"
    )
    print(
        f"Delta              : "
        f"{delta:+.8f}"
    )
    print(
        f"Folds improved     : "
        f"{improved}/5"
    )
    print(
        f"Folds worse        : "
        f"{worse}/5"
    )
    print(
        f"Prob corr champion : "
        f"{prob_corr:.6f}"
    )
    print(
        f"Rank corr champion : "
        f"{rank_corr:.6f}"
    )
    print(
        f"Primitive          : "
        f"{primitive}"
    )
    print()
    print("Coefficient stability:")
    print(
        coef_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    coef_by_fold.to_csv(
        OUTPUT_DIR
        / "coefficients_by_fold.csv",
        index=False,
    )

    coef_summary.to_csv(
        OUTPUT_DIR
        / "coefficient_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": folds,
            "target_encoded": y,
            "champion_meta_rank": (
                champion_meta
            ),
            "meta_stack_score": (
                meta_oof
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: CROSS-FITTED LOGISTIC META-STACK",
        "=" * 88,
        "",
        "NO BASE-MODEL TRAINING",
        "NO BLEND-WEIGHT GRID",
        "NO LEADERBOARD USAGE",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        "",
        "META FEATURES",
        *[
            f"- {name}: {EXPECTED[name]:.8f}"
            for name in META_FEATURES
        ],
        "",
        "META LEARNER",
        "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000)",
        "",
        "VALIDATION",
        "Train meta learner on 4 frozen folds, evaluate once on held fold.",
        "Each base prediction is percentile-ranked within its original frozen fold.",
        "",
        "RESULT",
        f"Champion meta AUC: {champion_meta_auc:.8f}",
        f"Logistic stack AUC: {stack_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        f"Probability corr vs champion: {prob_corr:.6f}",
        f"Rank corr vs champion: {rank_corr:.6f}",
        f"Primitive: {primitive}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary.append(
            f"Fold {row.held_fold}: "
            f"{row.champion_auc:.8f} -> "
            f"{row.stack_auc:.8f} "
            f"({row.delta:+.8f})"
        )

    summary.extend(
        [
            "",
            "COEFFICIENT STABILITY",
        ]
    )

    for row in coef_summary.itertuples():
        summary.append(
            f"{row.feature}: "
            f"mean={row.mean_coefficient:+.6f}, "
            f"std={row.std_coefficient:.6f}, "
            f"positive_folds={row.positive_folds}/5, "
            f"negative_folds={row.negative_folds}/5"
        )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print(
        f"Artifacts: "
        f"{OUTPUT_DIR.relative_to(ROOT)}"
    )
    print(
        "Done. No base models were trained."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 104)
        print("AUDIT FAILED")
        print("=" * 104)
        print(str(exc))
        sys.exit(1)
