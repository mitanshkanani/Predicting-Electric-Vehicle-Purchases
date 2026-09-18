"""
xgboost_hierarchical_income_te_gpu.py

Kaggle Playground Series S6E9
Controlled structural experiment: add leakage-safe hierarchical income target encoding
to the CURRENT validated XGBoost champion.

HYPOTHESIS
----------
Our strongest local discoveries all point to Annual_Income_USD carrying signal at
multiple representations:

- raw numeric income
- exact-value identity
- exact-value Bayesian TE
- decimal digit decomposition
- exact-value frequency

The current exact-value TE treats each exact income independently. This experiment
asks whether nearby income values can share useful target information through a
small decimal hierarchy.

CANDIDATE FEATURES
------------------
Add exactly three leakage-safe Bayesian target encodings:

    HTE__income_1k_bucket
    HTE__income_10k_bucket
    HTE__income_100k_bucket

where bucket = floor(Annual_Income_USD / scale).

This is one coherent mechanism: hierarchical / multi-scale income identity.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw features
- raw categorical treatment
- income digit decomposition
- exact income + commute frequency features
- exact income + commute nested Bayesian TE
- smoothing m=2
- learned nested logistic base margin
- XGBoost depth-4 model family/config
- seed 42
- GPU training
- early stopping
- no leaderboard optimization

LEAKAGE SAFETY
--------------
For every outer validation fold:

- candidate training rows receive INNER-OOF hierarchical TE
- candidate validation rows are encoded from outer-training rows only
- competition test rows are encoded from outer-training rows only

Therefore no validation row contributes its target to its own encoded feature.

IMPORTANT
---------
This script imports feature-building utilities from the immediately preceding
validated file:

    lightgbm_engineered_learned_margin_cpu.py

That file must remain in the repo root. It is imported only for shared,
already-validated preprocessing / TE / learned-margin utilities. Its main()
function is protected by an if __name__ == "__main__" guard and will not run.

Run:
    python xgboost_hierarchical_income_te_gpu.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U xgboost\n"
    ) from exc

try:
    import lightgbm_engineered_learned_margin_cpu as base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import lightgbm_engineered_learned_margin_cpu.py.\n"
        "Place this experiment file in the same repo directory as that validated script.\n"
    ) from exc


SEED = 42
SMOOTHING = 2.0

EXPECTED_XGB_CHAMPION_AUC = 0.94583075
AUC_CHECK_TOLERANCE = 2e-5

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

HIERARCHICAL_INCOME_SCALES = [
    ("HTE__income_1k_bucket", 1_000),
    ("HTE__income_10k_bucket", 10_000),
    ("HTE__income_100k_bucket", 100_000),
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_XGB_CHAMPION_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_m2_exact_frequency_learned_logistic_margin_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_te_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Current XGBoost champion + leakage-safe hierarchical "
            "Annual_Income_USD target encoding."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
    )
    parser.add_argument(
        "--xgb-champion-oof",
        type=Path,
        default=DEFAULT_XGB_CHAMPION_OOF,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def build_model() -> xgb.XGBClassifier:
    """
    Frozen current XGBoost geometry/config.

    This experiment is NOT a hyperparameter experiment.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        device="cuda",
        max_bin=256,
        enable_categorical=True,
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=200,
    )


def income_bucket_key(
    series: pd.Series,
    divisor: int,
) -> pd.Series:
    income = np.rint(
        pd.to_numeric(
            series,
            errors="raise",
        ).to_numpy(dtype=np.float64)
    ).astype(np.int64)

    if (income < 0).any():
        raise ValueError("Annual_Income_USD unexpectedly contains negative values.")

    bucket = income // divisor

    return pd.Series(
        bucket.astype(str),
        index=series.index,
        dtype="object",
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
        frame.groupby(
            "key",
            observed=True,
        )["target"]
        .agg(["sum", "count"])
    )

    encoded = (
        stats["sum"]
        + smoothing * prior
    ) / (
        stats["count"]
        + smoothing
    )

    return encoded


def apply_mapping(
    keys: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> np.ndarray:
    return (
        keys.map(mapping)
        .fillna(prior)
        .to_numpy(dtype=np.float32)
    )


def build_hierarchical_income_te_for_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
    smoothing: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build 3 leakage-safe hierarchical income TEs.

    Outer-training rows:
        inner-OOF encoding across the remaining 4 frozen folds.

    Outer-validation + test rows:
        mapping fitted on the full outer-training population only.
    """
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    outer_train_original_folds = fold_ids[
        outer_train_idx
    ]
    outer_train_y = y[
        outer_train_idx
    ]

    train_te = pd.DataFrame(
        index=np.arange(len(outer_train_idx))
    )
    valid_te = pd.DataFrame(
        index=np.arange(len(outer_valid_idx))
    )
    test_te = pd.DataFrame(
        index=np.arange(len(test))
    )

    diagnostics: list[dict] = []

    for feature_name, divisor in HIERARCHICAL_INCOME_SCALES:
        outer_train_keys = income_bucket_key(
            train.iloc[outer_train_idx][
                "Annual_Income_USD"
            ].reset_index(drop=True),
            divisor,
        )

        valid_keys = income_bucket_key(
            train.iloc[outer_valid_idx][
                "Annual_Income_USD"
            ].reset_index(drop=True),
            divisor,
        )

        test_keys = income_bucket_key(
            test["Annual_Income_USD"].reset_index(drop=True),
            divisor,
        )

        train_encoded = np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )

        # Inner-OOF encoding for outer-training rows.
        for inner_valid_fold in sorted(
            np.unique(
                outer_train_original_folds
            ).tolist()
        ):
            inner_valid_mask = (
                outer_train_original_folds
                == inner_valid_fold
            )
            inner_fit_mask = ~inner_valid_mask

            inner_prior = float(
                outer_train_y[
                    inner_fit_mask
                ].mean()
            )

            mapping = fit_mapping(
                keys=outer_train_keys[
                    inner_fit_mask
                ],
                y=outer_train_y[
                    inner_fit_mask
                ],
                prior=inner_prior,
                smoothing=smoothing,
            )

            train_encoded[
                inner_valid_mask
            ] = apply_mapping(
                keys=outer_train_keys[
                    inner_valid_mask
                ],
                mapping=mapping,
                prior=inner_prior,
            )

        if np.isnan(train_encoded).any():
            raise RuntimeError(
                f"Hierarchical train TE contains NaNs for {feature_name}."
            )

        full_prior = float(
            outer_train_y.mean()
        )

        full_mapping = fit_mapping(
            keys=outer_train_keys,
            y=outer_train_y,
            prior=full_prior,
            smoothing=smoothing,
        )

        valid_encoded = apply_mapping(
            keys=valid_keys,
            mapping=full_mapping,
            prior=full_prior,
        )

        test_encoded = apply_mapping(
            keys=test_keys,
            mapping=full_mapping,
            prior=full_prior,
        )

        train_te[feature_name] = train_encoded
        valid_te[feature_name] = valid_encoded
        test_te[feature_name] = test_encoded

        known_keys = set(full_mapping.index)

        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "feature": feature_name,
                "divisor": divisor,
                "smoothing": smoothing,
                "outer_train_unique_buckets": int(
                    outer_train_keys.nunique()
                ),
                "valid_unseen_rate": float(
                    (~valid_keys.isin(known_keys)).mean()
                ),
                "test_unseen_rate": float(
                    (~test_keys.isin(known_keys)).mean()
                ),
                "outer_train_prior": full_prior,
            }
        )

    return (
        train_te,
        valid_te,
        test_te,
        pd.DataFrame(diagnostics),
    )


def load_and_verify_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    pred = base.load_oof(
        path,
        train,
        fold_ids,
        id_col,
    )

    auc = float(
        roc_auc_score(
            y,
            pred,
        )
    )

    if abs(
        auc - EXPECTED_XGB_CHAMPION_AUC
    ) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            "Loaded current-XGB OOF does not match the documented champion.\n"
            f"Expected approximately: {EXPECTED_XGB_CHAMPION_AUC:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}\n"
            f"File                  : {path.resolve()}\n"
            "Stop and inspect the artifact instead of comparing against the wrong baseline."
        )

    return pred, auc


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    required = [
        train_path,
        test_path,
        args.folds_path,
        args.xgb_champion_oof,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = base.sha256_file(
        args.folds_path
    )

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

    target = base.detect_target(
        train,
        test,
    )
    y, positive_label = base.encode_binary_target(
        train[target]
    )
    fold_ids, id_col = base.validate_folds(
        folds_df,
        train,
    )

    baseline_oof, baseline_auc = (
        load_and_verify_baseline_oof(
            path=args.xgb_champion_oof,
            train=train,
            fold_ids=fold_ids,
            id_col=id_col,
            y=y,
        )
    )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns and c != id_col
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

    # Current validated deterministic representation.
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

    # Current validated learned logistic prior representation.
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 94)
    print("XGBOOST + HIERARCHICAL INCOME TARGET ENCODING")
    print("=" * 94)
    print(f"XGBoost version: {xgb.__version__}")
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Current XGB champion OOF: {baseline_auc:.8f}")
    print()
    print("HYPOTHESIS:")
    print(
        "  Exact income identity is already valuable, but exact-value TE cannot "
        "share target information between nearby income values. A small decimal "
        "hierarchy may expose stable local income neighborhoods."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add 3 nested leakage-safe income TEs at $1k, $10k and $100k scales."
    )
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds + hash")
    print("  - current raw feature representation")
    print("  - income digit decomposition")
    print("  - income/commute exact frequency features")
    print("  - exact income/commute TE")
    print("  - all TE smoothing m=2")
    print("  - nested learned logistic base margin")
    print("  - depth-4 XGBoost config")
    print("  - seed 42")
    print("  - CUDA GPU")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    exact_te_diagnostics: list[pd.DataFrame] = []
    hierarchical_diagnostics: list[pd.DataFrame] = []
    logistic_diagnostics: list[pd.DataFrame] = []

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )
        valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        # Existing exact income + commute TE.
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

        exact_te_diagnostics.append(
            exact_diag
        )

        # Candidate-only hierarchical income TE.
        (
            train_hte,
            valid_hte,
            test_hte,
            hte_diag,
        ) = build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        hierarchical_diagnostics.append(
            hte_diag
        )

        # Existing nested learned logistic base margin.
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

        logistic_diagnostics.append(
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

        exact_te_columns = list(
            train_exact_te.columns
        )

        for c in exact_te_columns:
            X_train[c] = train_exact_te[c].to_numpy(
                dtype=np.float32
            )
            X_valid[c] = valid_exact_te[c].to_numpy(
                dtype=np.float32
            )
            X_test[c] = test_exact_te[c].to_numpy(
                dtype=np.float32
            )

        hte_columns = list(
            train_hte.columns
        )

        for c in hte_columns:
            X_train[c] = train_hte[c].to_numpy(
                dtype=np.float32
            )
            X_valid[c] = valid_hte[c].to_numpy(
                dtype=np.float32
            )
            X_test[c] = test_hte[c].to_numpy(
                dtype=np.float32
            )

        model = build_model()

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
            time.perf_counter()
            - fit_start
        )

        if model.best_iteration is None:
            iteration_range = None
        else:
            iteration_range = (
                0,
                int(model.best_iteration) + 1,
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

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                baseline_oof[valid_idx],
            )
        )

        delta = (
            candidate_fold_auc
            - baseline_fold_auc
        )

        best_iteration = (
            int(model.best_iteration)
            if model.best_iteration is not None
            else -1
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_xgb_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": best_iteration,
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
                    "is_hierarchical_income_te": (
                        feature in hte_columns
                    ),
                    "is_exact_te": (
                        feature in exact_te_columns
                    ),
                    "is_income_digit": (
                        feature in digit_features
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
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={best_iteration}"
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

    exact_diag_df = pd.concat(
        exact_te_diagnostics,
        ignore_index=True,
    )

    hte_diag_df = pd.concat(
        hierarchical_diagnostics,
        ignore_index=True,
    )

    logistic_diag_df = pd.concat(
        logistic_diagnostics,
        ignore_index=True,
    )

    if (
        delta_vs_baseline > 0
        and folds_improved >= 3
    ):
        decision = "KEEP_HIERARCHICAL_INCOME_TE"
    else:
        decision = "REJECT_HIERARCHICAL_INCOME_TE"

    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )

    importance_summary.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        args.output_dir
        / "feature_importance_by_fold.csv",
        index=False,
    )

    hte_importance.to_csv(
        args.output_dir
        / "hierarchical_income_te_importance.csv",
        index=False,
    )

    exact_diag_df.to_csv(
        args.output_dir
        / "exact_te_diagnostics.csv",
        index=False,
    )

    hte_diag_df.to_csv(
        args.output_dir
        / "hierarchical_income_te_diagnostics.csv",
        index=False,
    )

    logistic_diag_df.to_csv(
        args.output_dir
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
        args.output_dir
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
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: XGBOOST + HIERARCHICAL INCOME TARGET ENCODING",
        "=" * 82,
        "",
        "HYPOTHESIS",
        "Can nearby income values share useful target information beyond the",
        "already-validated exact income identity / exact TE / digit representations?",
        "",
        "ONLY CHANGE",
        "Add 3 nested leakage-safe Annual_Income_USD target encodings:",
        "- floor(income / 1,000)",
        "- floor(income / 10,000)",
        "- floor(income / 100,000)",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- exact income + commute TE",
        "- TE smoothing m=2",
        "- income digit decomposition",
        "- exact income + commute frequency features",
        "- learned nested logistic base margin",
        "- depth-4 XGBoost configuration",
        "- seed 42",
        "- CUDA GPU",
        "",
        "RESULTS",
        f"Current XGB champion OOF: {baseline_auc:.8f}",
        f"Hierarchical-income candidate OOF: {candidate_auc:.8f}",
        f"Delta vs champion: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs champion: {probability_corr:.6f}",
        f"Rank corr vs champion: {rank_correlation:.6f}",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"baseline={row.baseline_xgb_auc:.8f} -> "
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
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("HIERARCHICAL INCOME TE EXPERIMENT COMPLETE")
    print("=" * 94)
    print(f"Current XGB champion : {baseline_auc:.8f}")
    print(f"Candidate            : {candidate_auc:.8f}")
    print(f"Delta                : {delta_vs_baseline:+.8f}")
    print(f"Folds improved       : {folds_improved}/5")
    print(f"Folds worse          : {folds_worse}/5")
    print(f"Probability corr     : {probability_corr:.6f}")
    print(f"Rank corr            : {rank_correlation:.6f}")
    print(f"Decision             : {decision}")
    print(f"Runtime              : {total_seconds:.2f}s")
    print(f"Artifacts            : {args.output_dir.resolve()}")
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
