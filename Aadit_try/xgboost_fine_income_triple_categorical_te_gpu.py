"""
xgboost_fine_income_triple_categorical_te_gpu.py

Kaggle Playground Series S6E9

CONTROLLED STRUCTURAL EXPERIMENT
--------------------------------
Baseline:
    current best internal fine-income XGBoost
    OOF ~= 0.94606664

Only change:
    add leakage-safe multi-smoothing categorical target encodings for:
      - Gender
      - City_Type
      - Current_Car_Type
      - Range_Anxiety_Level

    at smoothing:
      - 1.0
      - 5.0
      - 20.0

This creates 12 new numeric features.

No dynamic pruning.
No hyperparameter search.
No blend search.
No leaderboard optimization.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "XGBoost is required. Install/update with:\n"
        "python -m pip install -U xgboost"
    ) from exc

try:
    import xgboost_fine_income_te_gpu as fine
except ImportError as exc:
    raise SystemExit(
        "Could not import xgboost_fine_income_te_gpu.py.\n"
        "Place this experiment in the same repo root."
    ) from exc


base = fine.base
income_hte = fine.income_hte

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94606664
AUC_TOLERANCE = 2e-5

FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

BASELINE_OOF_PATH = (
    Path("artifacts")
    / "experiments"
    / "xgboost_fine_income_te_gpu"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_fine_income_triple_categorical_te_gpu"
)

TE_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Range_Anxiety_Level",
]

TE_SMOOTHING_LEVELS = [
    1.0,
    5.0,
    20.0,
]


def canonical_key(series: pd.Series) -> pd.Series:
    """Stable categorical key with explicit missing-value token."""
    return (
        series
        .astype("object")
        .where(series.notna(), "__MISSING__")
        .astype(str)
    )


def fit_mapping(
    keys: pd.Series,
    y: np.ndarray,
    prior: float,
    smoothing: float,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "target": y,
        }
    )

    stats = (
        frame
        .groupby("key", observed=True)["target"]
        .agg(["sum", "count"])
    )

    encoded = (
        stats["sum"] + smoothing * prior
    ) / (
        stats["count"] + smoothing
    )

    return encoded


def apply_mapping(
    keys: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> tuple[np.ndarray, float]:
    encoded = keys.map(mapping)
    unseen_rate = float(encoded.isna().mean())

    values = (
        encoded
        .fillna(prior)
        .to_numpy(dtype=np.float32)
    )

    return values, unseen_rate


def build_triple_te_for_outer_fold(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build leakage-safe Triple-TE features.

    For outer-training rows:
        encode each row from a mapping fit without that row's frozen fold.

    For outer-validation and test rows:
        encode using the full outer-training mapping.

    Therefore no outer-validation target participates in its encoding.
    """
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    outer_train_y = y[outer_train_idx]
    outer_train_folds = fold_ids[outer_train_idx]

    train_out = pd.DataFrame(
        index=np.arange(len(outer_train_idx))
    )
    valid_out = pd.DataFrame(
        index=np.arange(len(outer_valid_idx))
    )
    test_out = pd.DataFrame(
        index=np.arange(len(test))
    )

    diagnostics: list[dict] = []

    for column in TE_COLUMNS:
        if column not in train.columns or column not in test.columns:
            raise RuntimeError(
                f"Required Triple-TE column missing: {column}"
            )

        full_train_keys = canonical_key(train[column])
        test_keys = canonical_key(test[column])

        outer_train_keys = (
            full_train_keys
            .iloc[outer_train_idx]
            .reset_index(drop=True)
        )

        outer_valid_keys = (
            full_train_keys
            .iloc[outer_valid_idx]
            .reset_index(drop=True)
        )

        for smoothing in TE_SMOOTHING_LEVELS:
            feature_name = (
                f"TTE__{column}__m"
                f"{str(smoothing).replace('.', 'p')}"
            )

            inner_oof = np.full(
                len(outer_train_idx),
                np.nan,
                dtype=np.float32,
            )

            # Reuse the other frozen outer folds as deterministic inner folds.
            for inner_fold in sorted(
                np.unique(outer_train_folds).tolist()
            ):
                inner_valid_mask = (
                    outer_train_folds == inner_fold
                )
                inner_fit_mask = ~inner_valid_mask

                inner_prior = float(
                    outer_train_y[inner_fit_mask].mean()
                )

                mapping = fit_mapping(
                    keys=outer_train_keys.loc[
                        inner_fit_mask
                    ].reset_index(drop=True),
                    y=outer_train_y[inner_fit_mask],
                    prior=inner_prior,
                    smoothing=smoothing,
                )

                encoded, _ = apply_mapping(
                    keys=outer_train_keys.loc[
                        inner_valid_mask
                    ].reset_index(drop=True),
                    mapping=mapping,
                    prior=inner_prior,
                )

                inner_oof[inner_valid_mask] = encoded

            if np.isnan(inner_oof).any():
                raise RuntimeError(
                    f"Inner OOF Triple-TE contains NaNs: "
                    f"{feature_name}, outer fold {outer_fold}"
                )

            outer_prior = float(outer_train_y.mean())

            outer_mapping = fit_mapping(
                keys=outer_train_keys,
                y=outer_train_y,
                prior=outer_prior,
                smoothing=smoothing,
            )

            valid_encoded, valid_unseen = apply_mapping(
                keys=outer_valid_keys,
                mapping=outer_mapping,
                prior=outer_prior,
            )

            test_encoded, test_unseen = apply_mapping(
                keys=test_keys,
                mapping=outer_mapping,
                prior=outer_prior,
            )

            train_out[feature_name] = inner_oof
            valid_out[feature_name] = valid_encoded
            test_out[feature_name] = test_encoded

            counts = outer_train_keys.value_counts(
                dropna=False
            )

            diagnostics.append(
                {
                    "outer_fold": outer_fold,
                    "source_column": column,
                    "feature": feature_name,
                    "smoothing": smoothing,
                    "outer_train_prior": outer_prior,
                    "unique_keys": int(counts.size),
                    "min_count": int(counts.min()),
                    "median_count": float(counts.median()),
                    "max_count": int(counts.max()),
                    "validation_unseen_rate": valid_unseen,
                    "test_unseen_rate": test_unseen,
                }
            )

    return (
        train_out,
        valid_out,
        test_out,
        pd.DataFrame(diagnostics),
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
    return float(np.corrcoef(ar, br)[0, 1])


def main() -> None:
    train_path = Path("data") / "train.csv"
    test_path = Path("data") / "test.csv"

    for path in [
        train_path,
        test_path,
        FOLDS_PATH,
        BASELINE_OOF_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = base.sha256_file(FOLDS_PATH)

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(FOLDS_PATH)

    target = base.detect_target(train, test)

    y, positive_label = base.encode_binary_target(
        train[target]
    )

    fold_ids, id_col = base.validate_folds(
        folds_df,
        train,
    )

    baseline_oof = base.load_oof(
        BASELINE_OOF_PATH,
        train,
        fold_ids,
        id_col,
    )

    baseline_auc = float(
        roc_auc_score(y, baseline_oof)
    )

    if abs(
        baseline_auc - EXPECTED_BASELINE_AUC
    ) > AUC_TOLERANCE:
        raise RuntimeError(
            "Fine-income baseline OOF mismatch.\n"
            f"Expected about: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Found:          {baseline_auc:.8f}"
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

    logistic_train_matrix = (
        base.build_logistic_recipe_matrix(train)
    )
    logistic_test_matrix = (
        base.build_logistic_recipe_matrix(test)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 104)
    print("XGBOOST FINE-INCOME + TRIPLE CATEGORICAL TARGET ENCODING")
    print("=" * 104)
    print(f"XGBoost version     : {xgb.__version__}")
    print(f"Target              : {target!r}")
    print(f"Positive label      : {positive_label!r}")
    print(f"Frozen fold SHA256  : {fold_hash}")
    print(f"Fine-income baseline: {baseline_auc:.8f}")
    print()
    print("ONLY CHANGE:")
    print(
        "  Add 4 categorical columns x 3 smoothing levels "
        "(1, 5, 20) = 12 leakage-safe Triple-TE features."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds")
    print("  - all existing fine-income XGB features")
    print("  - exact income/commute TE")
    print("  - income hierarchy + commute hierarchy")
    print("  - $50/$250 fine-income TE")
    print("  - learned logistic base margin")
    print("  - XGBoost model configuration")
    print("  - seed / CUDA / early stopping")
    print("  - no dynamic pruning")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []

    exact_diag_frames: list[pd.DataFrame] = []
    income_diag_frames: list[pd.DataFrame] = []
    commute_diag_frames: list[pd.DataFrame] = []
    fine_income_diag_frames: list[pd.DataFrame] = []
    triple_te_diag_frames: list[pd.DataFrame] = []
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
            smoothing=fine.SMOOTHING,
        )
        exact_diag_frames.append(exact_diag)

        (
            train_income_hte,
            valid_income_hte,
            test_income_hte,
            income_diag,
        ) = (
            income_hte
            .build_hierarchical_income_te_for_outer_fold(
                train=train,
                test=test,
                y=y,
                fold_ids=fold_ids,
                outer_fold=outer_fold,
                smoothing=fine.SMOOTHING,
            )
        )
        income_diag_frames.append(income_diag)

        (
            train_commute_hte,
            valid_commute_hte,
            test_commute_hte,
            commute_diag,
        ) = fine.build_hierarchical_commute_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=fine.SMOOTHING,
        )
        commute_diag_frames.append(commute_diag)

        (
            train_fine_income_te,
            valid_fine_income_te,
            test_fine_income_te,
            fine_income_diag,
        ) = fine.build_fine_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=fine.SMOOTHING,
        )
        fine_income_diag_frames.append(
            fine_income_diag
        )

        (
            train_triple_te,
            valid_triple_te,
            test_triple_te,
            triple_diag,
        ) = build_triple_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )
        triple_te_diag_frames.append(triple_diag)

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
            X_candidate_base
            .iloc[train_idx]
            .reset_index(drop=True)
            .copy()
        )
        X_valid = (
            X_candidate_base
            .iloc[valid_idx]
            .reset_index(drop=True)
            .copy()
        )
        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        feature_groups = [
            (
                train_exact_te,
                valid_exact_te,
                test_exact_te,
            ),
            (
                train_income_hte,
                valid_income_hte,
                test_income_hte,
            ),
            (
                train_commute_hte,
                valid_commute_hte,
                test_commute_hte,
            ),
            (
                train_fine_income_te,
                valid_fine_income_te,
                test_fine_income_te,
            ),
            (
                train_triple_te,
                valid_triple_te,
                test_triple_te,
            ),
        ]

        for tr_feat, va_feat, te_feat in feature_groups:
            for c in tr_feat.columns:
                X_train[c] = tr_feat[c].to_numpy(
                    dtype=np.float32
                )
                X_valid[c] = va_feat[c].to_numpy(
                    dtype=np.float32
                )
                X_test[c] = te_feat[c].to_numpy(
                    dtype=np.float32
                )

        triple_te_columns = list(
            train_triple_te.columns
        )

        model = income_hte.build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[train_idx],
            base_margin=learned_train_margin,
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            base_margin_eval_set=[
                learned_valid_margin
            ],
            verbose=False,
        )

        fit_seconds = (
            time.perf_counter() - fit_start
        )

        if model.best_iteration is None:
            best_iteration = -1
            iteration_range = None
        else:
            best_iteration = int(
                model.best_iteration
            )
            iteration_range = (
                0,
                best_iteration + 1,
            )

        valid_pred = model.predict_proba(
            X_valid,
            base_margin=learned_valid_margin,
            iteration_range=iteration_range,
        )[:, 1]

        test_pred = model.predict_proba(
            X_test,
            base_margin=learned_test_margin,
            iteration_range=iteration_range,
        )[:, 1]

        oof[valid_idx] = valid_pred
        test_fold_predictions.append(
            test_pred.astype(np.float32)
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

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": best_iteration,
                "fit_seconds": fit_seconds,
                "total_fold_seconds": (
                    time.perf_counter()
                    - fold_start
                ),
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
                    "is_triple_te": (
                        feature in triple_te_columns
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
            f"best_iter={best_iteration}"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test,
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
        roc_auc_score(y, oof)
    )

    delta_vs_baseline = (
        candidate_auc - baseline_auc
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

    triple_importance = (
        importance_by_fold[
            importance_by_fold["is_triple_te"]
        ]
        .groupby(
            "feature",
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

    if (
        delta_vs_baseline >= 3e-5
        and folds_improved >= 4
    ):
        decision = "STRONG_KEEP"
    elif (
        delta_vs_baseline > 0
        and folds_improved >= 3
    ):
        decision = "KEEP_BUT_WEAK"
    elif (
        delta_vs_baseline <= 0
        or folds_worse >= 4
    ):
        decision = "REJECT"
    else:
        decision = "INCONCLUSIVE"

    fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    triple_importance.to_csv(
        OUTPUT_DIR
        / "triple_te_importance.csv",
        index=False,
    )

    pd.concat(
        triple_te_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "triple_te_diagnostics.csv",
        index=False,
    )

    # Preserve the candidate OOF/test artifacts for later ensemble audit
    # only if the scientific result warrants it.
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
        OUTPUT_DIR / "oof_predictions.csv",
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
        OUTPUT_DIR / "test_predictions.csv",
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
        fine_income_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "fine_income_te_diagnostics.csv",
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

    summary = [
        "EXPERIMENT: FINE-INCOME XGB + TRIPLE CATEGORICAL TARGET ENCODING",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target model experiment.",
        "",
        "HYPOTHESIS",
        (
            "Can multi-smoothing categorical target encoding add useful "
            "signal beyond the current fine-income XGB representation?"
        ),
        "",
        "ONLY CHANGE",
        (
            "Add 12 leakage-safe categorical target encodings: "
            "4 columns x smoothing {1,5,20}."
        ),
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- current fine-income XGB feature stack",
        "- learned logistic base margin",
        "- XGBoost configuration / seed / CUDA / early stopping",
        "- no dynamic pruning",
        "- no HPO",
        "- no leaderboard optimization",
        "",
        "RESULT",
        f"Baseline OOF: {baseline_auc:.8f}",
        f"Candidate OOF: {candidate_auc:.8f}",
        f"Delta: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr: {probability_corr:.6f}",
        f"Rank corr: {rank_correlation:.6f}",
        f"Decision: {decision}",
        f"Runtime: {total_seconds:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"baseline={r.baseline_auc:.8f} -> "
            f"candidate={r.candidate_auc:.8f} "
            f"({r.delta_vs_baseline:+.8f})"
        )

    summary.extend(
        [
            "",
            "TRIPLE-TE IMPORTANCE",
        ]
    )

    for r in triple_importance.itertuples():
        summary.append(
            f"{r.feature}: "
            f"{r.mean_gain_importance:.8f}"
        )

    (
        OUTPUT_DIR / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("TRIPLE-TE EXPERIMENT COMPLETE")
    print("=" * 104)
    print(f"Baseline OOF     : {baseline_auc:.8f}")
    print(f"Candidate OOF    : {candidate_auc:.8f}")
    print(f"Delta            : {delta_vs_baseline:+.8f}")
    print(f"Folds improved   : {folds_improved}/5")
    print(f"Folds worse      : {folds_worse}/5")
    print(f"Probability corr : {probability_corr:.6f}")
    print(f"Rank corr        : {rank_correlation:.6f}")
    print(f"Decision         : {decision}")
    print(f"Runtime          : {total_seconds:.2f}s")
    print(f"Artifacts        : {OUTPUT_DIR.resolve()}")
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. triple_te_importance.csv")
    print("  5. triple_te_diagnostics.csv")
    print("=" * 104)


if __name__ == "__main__":
    main()
