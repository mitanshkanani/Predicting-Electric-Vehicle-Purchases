"""
lightgbm_hierarchical_income_commute_te_cpu.py

Kaggle Playground Series S6E9

Controlled structural experiment:
Add hierarchical Daily_Commute_km target encodings to the validated
hierarchical-income LightGBM model.

HYPOTHESIS
----------
Hierarchical commute structure improved:
- XGBoost: 0.94587366 -> 0.94591178, 5/5 folds
- CatBoost seed42: 0.94546676 -> 0.94552424, 5/5 folds

The current hierarchical-income LightGBM does NOT contain hierarchical commute
TE. If commute hierarchy is genuinely representation-level, it should also
improve LightGBM.

ONLY CHANGE
-----------
Add three nested leakage-safe commute TEs:

    HTE__commute_1km_bucket
    HTE__commute_5km_bucket
    HTE__commute_10km_bucket

All use Bayesian smoothing m=2.

HELD FIXED
----------
- frozen 5 folds + SHA256
- raw features
- income digit decomposition
- exact income/commute frequency features
- exact income/commute nested Bayesian TE
- hierarchical income TE at 1k/10k/100k
- nested learned logistic base margin
- LightGBM model configuration
- seed 42
- CPU device
- early stopping
- no public leaderboard optimization

LEAKAGE SAFETY
--------------
For every outer fold:
- outer-training rows receive inner-OOF commute TEs
- outer-validation rows use mappings fit only on outer-training rows
- test rows use mappings fit only on outer-training rows

BASELINE
--------
Validated hierarchical-income LightGBM:
    OOF AUC = 0.94582702

Run:
    python lightgbm_hierarchical_income_commute_te_cpu.py
"""

from __future__ import annotations

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
    import xgboost_hierarchical_income_te_gpu as income_hte
    import xgboost_hierarchical_commute_te_gpu as commute_hte
except ImportError as exc:
    raise SystemExit(
        "\nCould not import one of the validated helper scripts.\n"
        "Keep this file in the same repo root as:\n"
        "  lightgbm_engineered_learned_margin_cpu.py\n"
        "  xgboost_hierarchical_income_te_gpu.py\n"
        "  xgboost_hierarchical_commute_te_gpu.py\n"
    ) from exc


SMOOTHING = 2.0

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94582702
AUC_TOLERANCE = 2e-5

FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_hierarchical_income_te_cpu"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_hierarchical_income_commute_te_cpu"
)


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

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def main() -> None:
    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")

    for path in [
        train_path,
        test_path,
        FOLDS_PATH,
        BASELINE_OOF,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = base.sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}"
        )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    target = base.detect_target(
        train,
        test,
    )

    y, positive_label = (
        base.encode_binary_target(
            train[target]
        )
    )

    fold_ids, id_col = (
        base.validate_folds(
            folds_df,
            train,
        )
    )

    baseline_oof = base.load_oof(
        BASELINE_OOF,
        train,
        fold_ids,
        id_col,
    )

    baseline_auc = float(
        roc_auc_score(
            y,
            baseline_oof,
        )
    )

    if abs(
        baseline_auc
        - EXPECTED_BASELINE_AUC
    ) > AUC_TOLERANCE:
        raise ValueError(
            "Hierarchical-income LightGBM baseline mismatch.\n"
            f"Expected approximately: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Loaded artifact AUC   : {baseline_auc:.8f}"
        )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = (
        base.detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    X_base, X_test_base = (
        base.prepare_base_frames(
            train=train,
            test=test,
            raw_features=raw_features,
            categorical_features=raw_categoricals,
        )
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 98)
    print("LIGHTGBM + HIERARCHICAL INCOME + COMMUTE TARGET ENCODING")
    print("=" * 98)
    print(f"LightGBM version: {lgb.__version__}")
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Hierarchical-income LightGBM baseline: "
        f"{baseline_auc:.8f}"
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  Commute hierarchy improved XGBoost and CatBoost on all 5 folds. "
        "If the signal is model-agnostic, adding the same leakage-safe hierarchy "
        "should also improve LightGBM."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add hierarchical commute TE at 1km / 5km / 10km scales."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen folds + SHA256")
    print("  - raw features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - hierarchical income TE")
    print("  - smoothing m=2")
    print("  - learned logistic base margin")
    print("  - fixed LightGBM configuration")
    print("  - seed 42")
    print("  - CPU device")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    fold_rows = []
    importance_rows = []

    exact_diag_frames = []
    income_diag_frames = []
    commute_diag_frames = []
    logistic_diag_frames = []

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

        exact_diag_frames.append(
            exact_diag
        )

        (
            train_income_hte,
            valid_income_hte,
            test_income_hte,
            income_diag,
        ) = income_hte.build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        income_diag_frames.append(
            income_diag
        )

        (
            train_commute_hte,
            valid_commute_hte,
            test_commute_hte,
            commute_diag,
        ) = commute_hte.build_hierarchical_commute_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        commute_diag_frames.append(
            commute_diag
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

        logistic_diag_frames.append(
            logistic_diag
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

        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        exact_columns = list(
            train_exact_te.columns
        )

        for c in exact_columns:
            X_train[c] = train_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

        income_columns = list(
            train_income_hte.columns
        )

        for c in income_columns:
            X_train[c] = train_income_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_income_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_income_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

        commute_columns = list(
            train_commute_hte.columns
        )

        for c in commute_columns:
            X_train[c] = train_commute_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_commute_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_commute_hte[
                c
            ].to_numpy(
                dtype=np.float32
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
                learned_valid_margin
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

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                baseline_oof[valid_idx],
            )
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
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
                    "is_commute_hte": (
                        feature in commute_columns
                    ),
                    "is_income_hte": (
                        feature in income_columns
                    ),
                    "is_exact_te": (
                        feature in exact_columns
                    ),
                    "is_income_digit": (
                        feature in digit_features
                    ),
                    "is_exact_frequency": (
                        feature in exact_frequency_features
                    ),
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={model.best_iteration_}"
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Candidate OOF contains NaNs."
        )

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
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

    probability_corr = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    rank_correlation = rank_corr(
        oof,
        baseline_oof,
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
                "is_commute_hte",
                "is_income_hte",
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

    commute_importance = (
        importance_summary[
            importance_summary[
                "is_commute_hte"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    decision = (
        "KEEP_HIERARCHICAL_COMMUTE_TE_FOR_LIGHTGBM"
        if (
            delta_vs_baseline > 0
            and folds_improved >= 3
        )
        else
        "REJECT_HIERARCHICAL_COMMUTE_TE_FOR_LIGHTGBM"
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

    importance_by_fold.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    commute_importance.to_csv(
        OUTPUT_DIR
        / "hierarchical_commute_te_importance.csv",
        index=False,
    )

    pd.concat(
        exact_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "exact_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        income_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "hierarchical_income_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        commute_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "hierarchical_commute_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        logistic_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "learned_logistic_margin_coefficients.csv",
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
            "oof_prediction": oof.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[id_col].to_numpy(),
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
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: LIGHTGBM + HIERARCHICAL INCOME + COMMUTE TE",
        "=" * 84,
        f"Baseline hierarchical-income LightGBM: {baseline_auc:.8f}",
        f"Candidate: {candidate_auc:.8f}",
        f"Delta: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs baseline: {probability_corr:.6f}",
        f"Rank corr vs baseline: {rank_correlation:.6f}",
        f"Runtime: {total_seconds:.2f}s",
        f"Decision: {decision}",
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
            "COMMUTE HTE IMPORTANCE",
        ]
    )

    for row in commute_importance.itertuples():
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
    print("=" * 98)
    print("LIGHTGBM HIERARCHICAL-COMMUTE EXPERIMENT COMPLETE")
    print("=" * 98)
    print(
        f"Baseline  : {baseline_auc:.8f}"
    )
    print(
        f"Candidate : {candidate_auc:.8f}"
    )
    print(
        f"Delta     : {delta_vs_baseline:+.8f}"
    )
    print(
        f"Folds improved/worse: "
        f"{folds_improved}/{folds_worse}"
    )
    print(
        f"Rank corr : {rank_correlation:.6f}"
    )
    print(
        f"Decision  : {decision}"
    )
    print(
        f"Runtime   : {total_seconds:.2f}s"
    )
    print(
        f"Artifacts : {OUTPUT_DIR.resolve()}"
    )
    print()
    print("Hierarchical commute TE importances:")
    if len(commute_importance):
        print(
            commute_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(
                index=False
            )
        )
    print("=" * 98)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. hierarchical_commute_te_importance.csv")
    print("  5. hierarchical_commute_te_diagnostics.csv")


if __name__ == "__main__":
    main()
