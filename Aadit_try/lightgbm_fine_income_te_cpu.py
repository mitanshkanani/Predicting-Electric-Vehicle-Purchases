"""
lightgbm_fine_income_te_cpu.py

Kaggle Playground Series S6E9

CONTROLLED CROSS-MODEL TRANSFER EXPERIMENT
------------------------------------------
The $50 / $250 leakage-safe income target encodings strongly improved our
current XGBoost. This experiment transfers ONLY those two fine-income features
into the OLD engineered LightGBM member that is still useful in our ensemble.

Why the old LightGBM baseline?
------------------------------
Later hierarchical-income / hierarchical-commute LightGBM variants improved
standalone AUC, but became more redundant with XGBoost and hurt the ensemble.

So this test deliberately starts from the old diverse LightGBM:

    0.94578042

and adds ONLY:

    HTE__income_50_bucket
    HTE__income_250_bucket

Everything else remains identical to the old engineered LightGBM.

NO hierarchical income TE.
NO hierarchical commute TE.
NO parameter tuning.
NO blend tuning.

The goal is to ask whether the strongest newly-discovered local-income signal
can improve the diversity member without turning it into another XGBoost clone.

Validation:
    frozen 5 folds
    SHA256:
    55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee

Run:
    python lightgbm_fine_income_te_cpu.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "\nLightGBM is not installed.\n"
        "Install it with:\n"
        "    python -m pip install -U lightgbm\n"
    ) from exc

try:
    import lightgbm_engineered_learned_margin_cpu as base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import lightgbm_engineered_learned_margin_cpu.py.\n"
        "Place this experiment file in the same repo root as that validated script.\n"
    ) from exc

try:
    import xgboost_fine_income_te_gpu as fine_xgb
except ImportError as exc:
    raise SystemExit(
        "\nCould not import xgboost_fine_income_te_gpu.py.\n"
        "Place this experiment file in the same repo root as that validated script.\n"
    ) from exc


ROOT = Path(__file__).resolve().parent

EXPECTED_FOLD_SHA = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94578042
EXPECTED_FINE_XGB_AUC = 0.94606664
EXPECTED_CHAMPION_AUC = 0.94611253

AUC_TOL = 5e-6
SMOOTHING = 2.0
SEED = 42

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

FOLDS_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "candidate_folds.csv"
)

BASELINE_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

FINE_XGB_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "xgboost_fine_income_te_gpu"
    / "oof_predictions.csv"
)

CHAMPION_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_fine_income_te_cpu"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = (
        pd.Series(a)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    br = (
        pd.Series(b)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    return float(np.corrcoef(ar, br)[0, 1])


def prediction_column(df: pd.DataFrame) -> str:
    preferred = [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "id",
    }

    candidates = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not identify prediction column. Columns={list(df.columns)}"
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
            f"Row-count mismatch for {path}: {len(df)} vs {len(train)}"
        )

    if "row_index" in df.columns:
        expected = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError(f"row_index mismatch in {path}")

    if "fold" in df.columns:
        if not np.array_equal(
            df["fold"].to_numpy(dtype=int),
            folds,
        ):
            raise RuntimeError(f"fold mismatch in {path}")

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise RuntimeError(f"id mismatch in {path}")

    pred = (
        pd.to_numeric(
            df[prediction_column(df)],
            errors="raise",
        )
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite predictions in {path}")

    return pred


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def main() -> None:
    print("=" * 100)
    print("LIGHTGBM + FINE $50/$250 INCOME TE")
    print("CONTROLLED TRANSFER INTO OLD DIVERSE LIGHTGBM")
    print("=" * 100)

    required = [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        BASELINE_OOF_PATH,
    ]

    for path in required:
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
    test = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)

    target = base.detect_target(train, test)

    y, positive_label = base.encode_binary_target(
        train[target]
    )

    fold_ids, id_col = base.validate_folds(
        folds_df,
        train,
    )

    baseline_oof = load_oof(
        BASELINE_OOF_PATH,
        train,
        fold_ids,
    )

    baseline_auc = auc(y, baseline_oof)

    if abs(
        baseline_auc
        - EXPECTED_BASELINE_AUC
    ) > AUC_TOL:
        raise RuntimeError(
            "Old engineered LightGBM baseline mismatch.\n"
            f"Expected: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Loaded:   {baseline_auc:.8f}\n"
            f"Path:     {BASELINE_OOF_PATH}"
        )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = base.detect_raw_categoricals(
        train,
        raw_features,
    )

    X_base, X_test_base = base.prepare_base_frames(
        train=train,
        test=test,
        raw_features=raw_features,
        categorical_features=raw_categoricals,
    )

    (
        X_income_base,
        X_income_test_base,
        digit_features,
    ) = base.add_income_digit_features(
        X_train=X_base,
        X_test=X_test_base,
        train_source=train,
        test_source=test,
    )

    (
        X_candidate_base,
        X_candidate_test_base,
        exact_frequency_features,
    ) = base.add_exact_frequency_features(
        X_train=X_income_base,
        X_test=X_income_test_base,
        train_source=train,
        test_source=test,
    )

    logistic_train_matrix = (
        base.build_logistic_recipe_matrix(
            train
        )
    )

    logistic_test_matrix = (
        base.build_logistic_recipe_matrix(
            test
        )
    )

    print(f"LightGBM version: {lgb.__version__}")
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Old engineered LightGBM baseline: {baseline_auc:.8f}")
    print()
    print("HYPOTHESIS:")
    print(
        "  The $50/$250 local-income signal can improve the old diverse "
        "LightGBM without adding the hierarchical income/commute features "
        "that previously made LightGBM less useful in the ensemble."
    )
    print()
    print("ONLY CHANGE:")
    print("  Add nested leakage-safe:")
    print("    HTE__income_50_bucket")
    print("    HTE__income_250_bucket")
    print()
    print("HELD FIXED:")
    print("  - old engineered LightGBM representation")
    print("  - raw 13 features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - learned nested logistic base margin")
    print("  - smoothing m=2")
    print("  - fixed LightGBM parameters")
    print("  - seed 42")
    print("  - CPU device")
    print("  - frozen competition folds")
    print()
    print("INTENTIONALLY NOT ADDED:")
    print("  - hierarchical income 1k/10k/100k TE")
    print("  - hierarchical commute 1/5/10km TE")
    print("  - any parameter search")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    fine_diag_frames: list[pd.DataFrame] = []

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )
        valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        (
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            exact_diag,
        ) = base.build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        (
            train_fine_te,
            valid_fine_te,
            test_fine_te,
            fine_diag,
        ) = fine_xgb.build_fine_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        fine_diag_frames.append(
            fine_diag
        )

        (
            learned_train_margin,
            learned_valid_margin,
            learned_test_margin,
            logistic_diag,
        ) = base.build_learned_logistic_margins_for_outer_fold(
            train_matrix=logistic_train_matrix,
            test_matrix=logistic_test_matrix,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        X_train = (
            X_candidate_base.iloc[
                train_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_valid = (
            X_candidate_base.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_test_fold = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        exact_te_columns = list(
            train_exact_te.columns
        )

        for c in exact_te_columns:
            X_train[c] = (
                train_exact_te[c]
                .to_numpy(dtype=np.float32)
            )
            X_valid[c] = (
                valid_exact_te[c]
                .to_numpy(dtype=np.float32)
            )
            X_test_fold[c] = (
                test_exact_te[c]
                .to_numpy(dtype=np.float32)
            )

        fine_te_columns = list(
            train_fine_te.columns
        )

        for c in fine_te_columns:
            X_train[c] = (
                train_fine_te[c]
                .to_numpy(dtype=np.float32)
            )
            X_valid[c] = (
                valid_fine_te[c]
                .to_numpy(dtype=np.float32)
            )
            X_test_fold[c] = (
                test_fine_te[c]
                .to_numpy(dtype=np.float32)
            )

        model = base.build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[train_idx],
            init_score=learned_train_margin,
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            eval_init_score=[
                learned_valid_margin,
            ],
            eval_metric="auc",
            categorical_feature="auto",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=200,
                    verbose=False,
                ),
                lgb.log_evaluation(
                    period=0,
                ),
            ],
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        valid_tree_raw = model.predict(
            X_valid,
            raw_score=True,
            num_iteration=model.best_iteration_,
        )

        test_tree_raw = model.predict(
            X_test_fold,
            raw_score=True,
            num_iteration=model.best_iteration_,
        )

        valid_pred = expit(
            np.asarray(
                valid_tree_raw,
                dtype=np.float64,
            )
            + learned_valid_margin
        )

        test_pred = expit(
            np.asarray(
                test_tree_raw,
                dtype=np.float64,
            )
            + learned_test_margin
        )

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        baseline_fold_auc = auc(
            y[valid_idx],
            baseline_oof[valid_idx],
        )

        candidate_fold_auc = auc(
            y[valid_idx],
            valid_pred,
        )

        delta = (
            candidate_fold_auc
            - baseline_fold_auc
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": int(
                    model.best_iteration_
                ),
                "fit_seconds": fit_seconds,
                "total_fold_seconds": fold_seconds,
            }
        )

        for feature, importance in zip(
            X_train.columns,
            model.feature_importances_,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature,
                    "is_fine_income_te": (
                        feature
                        in fine_te_columns
                    ),
                    "is_exact_te": (
                        feature
                        in exact_te_columns
                    ),
                    "is_income_digit": (
                        feature
                        in digit_features
                    ),
                    "is_exact_frequency": (
                        feature
                        in exact_frequency_features
                    ),
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"{baseline_fold_auc:.8f} -> "
            f"{candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={int(model.best_iteration_)} | "
            f"{fold_seconds:.2f}s"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test_fold,
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            train_fine_te,
            valid_fine_te,
            test_fine_te,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Candidate OOF contains NaNs."
        )

    candidate_auc = auc(
        y,
        oof,
    )

    delta_vs_baseline = (
        candidate_auc
        - baseline_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    folds_improved = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] < 0
        ).sum()
    )

    prob_corr_vs_baseline = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    rank_corr_vs_baseline = rank_corr(
        oof,
        baseline_oof,
    )

    reference_rows = []

    for name, path, expected in [
        (
            "fine_xgb",
            FINE_XGB_OOF_PATH,
            EXPECTED_FINE_XGB_AUC,
        ),
        (
            "champion",
            CHAMPION_OOF_PATH,
            EXPECTED_CHAMPION_AUC,
        ),
    ]:
        if not path.exists():
            continue

        ref = load_oof(
            path,
            train,
            fold_ids,
        )

        ref_auc = auc(
            y,
            ref,
        )

        if abs(
            ref_auc
            - expected
        ) > 5e-5:
            print(
                f"[WARN] Skipping {name} correlation: "
                f"AUC {ref_auc:.8f} != expected {expected:.8f}"
            )
            continue

        reference_rows.append(
            {
                "reference": name,
                "reference_auc": ref_auc,
                "probability_corr": float(
                    np.corrcoef(
                        oof,
                        ref,
                    )[0, 1]
                ),
                "rank_corr": rank_corr(
                    oof,
                    ref,
                ),
            }
        )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    importance_by_fold = pd.DataFrame(
        importance_rows
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            [
                "feature",
                "is_fine_income_te",
                "is_exact_te",
                "is_income_digit",
                "is_exact_frequency",
            ],
            as_index=False,
        )
        .agg(
            mean_gain_importance=(
                "gain_importance",
                "mean",
            ),
            std_gain_importance=(
                "gain_importance",
                "std",
            ),
        )
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    fine_importance = (
        importance_summary[
            importance_summary[
                "is_fine_income_te"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
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

    importance_summary.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    fine_importance.to_csv(
        OUTPUT_DIR
        / "fine_income_te_importance.csv",
        index=False,
    )

    pd.concat(
        fine_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "fine_income_te_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        reference_rows
    ).to_csv(
        OUTPUT_DIR
        / "diversity_correlations.csv",
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
            "oof_prediction": (
                oof.astype(np.float32)
            ),
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[
                id_col
            ].to_numpy(),
        )

    oof_output.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": (
                test_prediction.astype(
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
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    if (
        delta_vs_baseline >= 3e-5
        and folds_improved >= 4
    ):
        primitive = (
            "POSITIVE_FINE_INCOME_LGBM_SIGNAL"
        )
    elif (
        delta_vs_baseline <= -2e-5
        or folds_worse >= 4
    ):
        primitive = (
            "NEGATIVE_FINE_INCOME_LGBM_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_FINE_INCOME_LGBM_SIGNAL"
        )

    print()
    print("=" * 100)
    print("FINE-INCOME LIGHTGBM RESULT")
    print("=" * 100)
    print(
        f"Old engineered LGBM : "
        f"{baseline_auc:.8f}"
    )
    print(
        f"Fine-income LGBM     : "
        f"{candidate_auc:.8f}"
    )
    print(
        f"Delta                : "
        f"{delta_vs_baseline:+.8f}"
    )
    print(
        f"Folds improved       : "
        f"{folds_improved}/5"
    )
    print(
        f"Folds worse          : "
        f"{folds_worse}/5"
    )
    print(
        f"Prob corr vs old LGBM: "
        f"{prob_corr_vs_baseline:.6f}"
    )
    print(
        f"Rank corr vs old LGBM: "
        f"{rank_corr_vs_baseline:.6f}"
    )

    for row in reference_rows:
        print(
            f"Rank corr vs {row['reference']:8s}: "
            f"{row['rank_corr']:.6f}"
        )

    print(
        f"Primitive            : "
        f"{primitive}"
    )
    print(
        f"Runtime              : "
        f"{total_seconds:.2f}s"
    )
    print()
    print("Fine-income TE importance:")

    if len(fine_importance):
        print(
            fine_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(
                index=False
            )
        )

    summary = [
        "EXPERIMENT: LIGHTGBM + FINE $50/$250 INCOME TE",
        "=" * 84,
        "",
        "HYPOTHESIS",
        (
            "Can the strongest local-income signal improve the old diverse "
            "LightGBM without adding the hierarchical features that previously "
            "reduced ensemble value?"
        ),
        "",
        "ONLY CHANGE",
        "Add nested leakage-safe HTE__income_50_bucket and HTE__income_250_bucket.",
        "",
        "INTENTIONALLY NOT ADDED",
        "- hierarchical income TE",
        "- hierarchical commute TE",
        "- parameter changes",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"Old engineered LightGBM: {baseline_auc:.8f}",
        f"Fine-income LightGBM: {candidate_auc:.8f}",
        f"Delta: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs old LGBM: {prob_corr_vs_baseline:.6f}",
        f"Rank corr vs old LGBM: {rank_corr_vs_baseline:.6f}",
        f"Primitive: {primitive}",
        f"Runtime: {total_seconds:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary.append(
            f"Fold {row.fold}: "
            f"{row.baseline_auc:.8f} -> "
            f"{row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    summary.extend(
        [
            "",
            "DIVERSITY CORRELATIONS",
        ]
    )

    for row in reference_rows:
        summary.append(
            f"{row['reference']}: "
            f"prob_corr={row['probability_corr']:.6f}, "
            f"rank_corr={row['rank_corr']:.6f}"
        )

    summary.extend(
        [
            "",
            "FINE INCOME TE IMPORTANCE",
        ]
    )

    for row in fine_importance.itertuples():
        summary.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.8f}"
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
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. diversity_correlations.csv")
    print("  5. fine_income_te_importance.csv")


if __name__ == "__main__":
    main()
