"""
lightgbm_hierarchical_income_te_cpu.py

Kaggle Playground Series S6E9

Controlled structural experiment:
Add the newly validated hierarchical-income target encodings to the existing
engineered LightGBM model.

HYPOTHESIS
----------
Hierarchical income TE improved the current XGBoost champion on all 5 frozen
folds. If the signal is genuinely representation-level rather than specific to
XGBoost, the same leakage-safe income hierarchy should also improve LightGBM.

ONLY CHANGE
-----------
Add three hierarchical Annual_Income_USD Bayesian target encodings:

    HTE__income_1k_bucket
    HTE__income_10k_bucket
    HTE__income_100k_bucket

HELD FIXED
----------
- frozen 5 folds + SHA256
- raw features
- income digit decomposition
- exact income/commute frequency features
- exact income/commute nested Bayesian TE
- smoothing m=2
- nested learned logistic base margin
- LightGBM model configuration
- seed 42
- CPU device

LEAKAGE SAFETY
--------------
The hierarchical TEs use exactly the same nested protocol as the validated XGB
experiment:
- outer-training rows get inner-OOF encodings
- outer-validation rows use mappings fit only on outer-training rows
- test rows use mappings fit only on outer-training rows

This file imports the already validated utilities from:
    lightgbm_engineered_learned_margin_cpu.py
    xgboost_hierarchical_income_te_gpu.py

Both files must remain in the repo root.

Run:
    python lightgbm_hierarchical_income_te_cpu.py
"""

from __future__ import annotations

import argparse
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
        "Place this script in the same repo root as that validated file.\n"
    ) from exc

try:
    import xgboost_hierarchical_income_te_gpu as hte
except ImportError as exc:
    raise SystemExit(
        "\nCould not import xgboost_hierarchical_income_te_gpu.py.\n"
        "Place this script in the same repo root as that validated file.\n"
    ) from exc


SEED = 42
SMOOTHING = 2.0

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_LGBM_BASELINE_AUC = 0.94578042
EXPECTED_NEW_XGB_AUC = 0.94587366
EXPECTED_CAT_AUC = 0.94543956
AUC_CHECK_TOLERANCE = 2e-5

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_LGBM_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

DEFAULT_NEW_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_te_gpu"
    / "oof_predictions.csv"
)

DEFAULT_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_hierarchical_income_te_cpu"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Engineered LightGBM + leakage-safe hierarchical income TE."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)
    p.add_argument(
        "--lgbm-baseline-oof",
        type=Path,
        default=DEFAULT_LGBM_BASELINE_OOF,
    )
    p.add_argument(
        "--new-xgb-oof",
        type=Path,
        default=DEFAULT_NEW_XGB_OOF,
    )
    p.add_argument(
        "--cat-oof",
        type=Path,
        default=DEFAULT_CAT_OOF,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return p.parse_args()


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
            "Stop and inspect artifact alignment before continuing."
        )

    return auc


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(method="average", pct=True)
        .to_numpy()
    )
    br = (
        pd.Series(b)
        .rank(method="average", pct=True)
        .to_numpy()
    )
    return float(np.corrcoef(ar, br)[0, 1])


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    required_paths = [
        train_path,
        test_path,
        args.folds_path,
        args.lgbm_baseline_oof,
        args.new_xgb_oof,
        args.cat_oof,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = base.sha256_file(args.folds_path)

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}\n"
            f"File    : {args.folds_path.resolve()}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = base.detect_target(train, test)
    y, positive_label = base.encode_binary_target(train[target])
    fold_ids, id_col = base.validate_folds(folds_df, train)

    lgbm_baseline_oof = base.load_oof(
        args.lgbm_baseline_oof,
        train,
        fold_ids,
        id_col,
    )
    new_xgb_oof = base.load_oof(
        args.new_xgb_oof,
        train,
        fold_ids,
        id_col,
    )
    cat_oof = base.load_oof(
        args.cat_oof,
        train,
        fold_ids,
        id_col,
    )

    lgbm_baseline_auc = verify_auc(
        "Engineered LightGBM baseline",
        y,
        lgbm_baseline_oof,
        EXPECTED_LGBM_BASELINE_AUC,
    )
    new_xgb_auc = verify_auc(
        "Hierarchical-income XGBoost",
        y,
        new_xgb_oof,
        EXPECTED_NEW_XGB_AUC,
    )
    cat_auc = verify_auc(
        "CatBoost 3-seed",
        y,
        cat_oof,
        EXPECTED_CAT_AUC,
    )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns and c != id_col
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

    logistic_train_matrix = base.build_logistic_recipe_matrix(train)
    logistic_test_matrix = base.build_logistic_recipe_matrix(test)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 94)
    print("LIGHTGBM + HIERARCHICAL INCOME TARGET ENCODING")
    print("=" * 94)
    print(f"LightGBM version: {lgb.__version__}")
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Engineered LightGBM baseline: {lgbm_baseline_auc:.8f}")
    print(f"Hierarchical-income XGB ref : {new_xgb_auc:.8f}")
    print()
    print("HYPOTHESIS:")
    print(
        "  Hierarchical income TE improved XGBoost on all 5 folds. "
        "If this is a representation-level signal, it should also improve "
        "the engineered LightGBM model."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add the same 3 nested leakage-safe income TEs: "
        "$1k / $10k / $100k buckets."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen folds + SHA256")
    print("  - raw features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - smoothing m=2")
    print("  - nested learned logistic base margin")
    print("  - fixed LightGBM configuration")
    print("  - seed 42")
    print("  - CPU device")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    exact_te_diag_frames: list[pd.DataFrame] = []
    hierarchical_diag_frames: list[pd.DataFrame] = []
    logistic_diag_frames: list[pd.DataFrame] = []

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
        exact_te_diag_frames.append(exact_diag)

        (
            train_hte,
            valid_hte,
            test_hte,
            hte_diag,
        ) = hte.build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )
        hierarchical_diag_frames.append(hte_diag)

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
        logistic_diag_frames.append(logistic_diag)

        X_train = (
            X_candidate_base.iloc[train_idx]
            .reset_index(drop=True)
            .copy()
        )
        X_valid = (
            X_candidate_base.iloc[valid_idx]
            .reset_index(drop=True)
            .copy()
        )
        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        exact_te_columns = list(train_exact_te.columns)
        for c in exact_te_columns:
            X_train[c] = train_exact_te[c].to_numpy(dtype=np.float32)
            X_valid[c] = valid_exact_te[c].to_numpy(dtype=np.float32)
            X_test[c] = test_exact_te[c].to_numpy(dtype=np.float32)

        hte_columns = list(train_hte.columns)
        for c in hte_columns:
            X_train[c] = train_hte[c].to_numpy(dtype=np.float32)
            X_valid[c] = valid_hte[c].to_numpy(dtype=np.float32)
            X_test[c] = test_hte[c].to_numpy(dtype=np.float32)

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
                    period=100,
                ),
            ],
        )

        fit_seconds = time.perf_counter() - fit_start

        valid_tree_raw = model.predict(
            X_valid,
            raw_score=True,
            num_iteration=model.best_iteration_,
        )
        test_tree_raw = model.predict(
            X_test,
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

        oof[valid_idx] = valid_pred
        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                lgbm_baseline_oof[valid_idx],
            )
        )
        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )
        delta = candidate_fold_auc - baseline_fold_auc

        fold_seconds = time.perf_counter() - fold_start

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_lgbm_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": int(model.best_iteration_),
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
                    "is_hierarchical_income_te": feature in hte_columns,
                    "is_exact_te": feature in exact_te_columns,
                    "is_income_digit": feature in digit_features,
                    "is_exact_frequency": (
                        feature in exact_frequency_features
                    ),
                    "gain_importance": float(importance),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={model.best_iteration_}"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test,
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            train_hte,
            valid_hte,
            test_hte,
        )

    total_seconds = time.perf_counter() - total_start

    if np.isnan(oof).any():
        raise RuntimeError("Candidate OOF contains NaNs.")

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_baseline = candidate_auc - lgbm_baseline_auc

    fold_metrics = pd.DataFrame(fold_rows)

    folds_improved = int(
        (fold_metrics["delta_vs_baseline"] > 0).sum()
    )
    folds_worse = int(
        (fold_metrics["delta_vs_baseline"] < 0).sum()
    )

    probability_corr_vs_old_lgbm = float(
        np.corrcoef(
            oof,
            lgbm_baseline_oof,
        )[0, 1]
    )
    rank_corr_vs_old_lgbm = rank_corr(
        oof,
        lgbm_baseline_oof,
    )

    probability_corr_vs_new_xgb = float(
        np.corrcoef(
            oof,
            new_xgb_oof,
        )[0, 1]
    )
    rank_corr_vs_new_xgb = rank_corr(
        oof,
        new_xgb_oof,
    )

    probability_corr_vs_cat = float(
        np.corrcoef(
            oof,
            cat_oof,
        )[0, 1]
    )
    rank_corr_vs_cat = rank_corr(
        oof,
        cat_oof,
    )

    test_prediction = np.mean(
        np.vstack(test_fold_predictions),
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
                "is_hierarchical_income_te",
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

    hte_importance = (
        importance_summary[
            importance_summary[
                "is_hierarchical_income_te"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    if (
        delta_vs_baseline > 0
        and folds_improved >= 3
    ):
        decision = "KEEP_HIERARCHICAL_INCOME_TE_FOR_LIGHTGBM"
    else:
        decision = "REJECT_HIERARCHICAL_INCOME_TE_FOR_LIGHTGBM"

    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )
    importance_summary.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
    )
    importance_by_fold.to_csv(
        args.output_dir / "feature_importance_by_fold.csv",
        index=False,
    )
    hte_importance.to_csv(
        args.output_dir / "hierarchical_income_te_importance.csv",
        index=False,
    )

    pd.concat(
        exact_te_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir / "exact_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        hierarchical_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir / "hierarchical_income_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        logistic_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir / "learned_logistic_margin_coefficients.csv",
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
            "oof_prediction": oof.astype(np.float32),
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
            "prediction": test_prediction.astype(np.float32)
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
        "EXPERIMENT: LIGHTGBM + HIERARCHICAL INCOME TARGET ENCODING",
        "=" * 82,
        "",
        "HYPOTHESIS",
        "Does the hierarchical-income representation that improved XGBoost",
        "also improve the engineered LightGBM model?",
        "",
        "ONLY CHANGE",
        "Add nested leakage-safe income TE at $1k / $10k / $100k scales.",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- existing engineered LightGBM representation",
        "- exact income + commute TE",
        "- smoothing m=2",
        "- income digits",
        "- exact income + commute frequencies",
        "- learned nested logistic base margin",
        "- fixed LightGBM config",
        "- seed 42",
        "- CPU device",
        "",
        "RESULTS",
        f"Engineered LightGBM baseline OOF: {lgbm_baseline_auc:.8f}",
        f"Candidate OOF: {candidate_auc:.8f}",
        f"Delta vs baseline: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Hierarchical-income XGB reference: {new_xgb_auc:.8f}",
        f"CatBoost reference: {cat_auc:.8f}",
        "",
        "DIVERSITY DIAGNOSTICS",
        f"Probability corr vs old LightGBM: {probability_corr_vs_old_lgbm:.6f}",
        f"Rank corr vs old LightGBM: {rank_corr_vs_old_lgbm:.6f}",
        f"Probability corr vs new XGB: {probability_corr_vs_new_xgb:.6f}",
        f"Rank corr vs new XGB: {rank_corr_vs_new_xgb:.6f}",
        f"Probability corr vs CatBoost: {probability_corr_vs_cat:.6f}",
        f"Rank corr vs CatBoost: {rank_corr_vs_cat:.6f}",
        "",
        f"Runtime: {total_seconds:.2f} seconds",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"baseline={row.baseline_lgbm_auc:.8f} -> "
            f"candidate={row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "HIERARCHICAL INCOME TE IMPORTANCE",
        ]
    )

    for row in hte_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.8f}"
        )

    (
        args.output_dir / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("LIGHTGBM HIERARCHICAL-INCOME EXPERIMENT COMPLETE")
    print("=" * 94)
    print(f"Old engineered LightGBM : {lgbm_baseline_auc:.8f}")
    print(f"Candidate               : {candidate_auc:.8f}")
    print(f"Delta                   : {delta_vs_baseline:+.8f}")
    print(f"Folds improved          : {folds_improved}/5")
    print(f"Folds worse             : {folds_worse}/5")
    print(f"Rank corr vs new XGB    : {rank_corr_vs_new_xgb:.6f}")
    print(f"Rank corr vs CatBoost   : {rank_corr_vs_cat:.6f}")
    print(f"Decision                : {decision}")
    print(f"Runtime                 : {total_seconds:.2f}s")
    print(f"Artifacts               : {args.output_dir.resolve()}")
    print()
    print("Hierarchical income TE importances:")
    if len(hte_importance):
        print(
            hte_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(index=False)
        )
    print("=" * 94)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. hierarchical_income_te_importance.csv")


if __name__ == "__main__":
    main()
