"""
xgboost_income_commute_joint_te_gpu.py

Kaggle Playground Series S6E9

Controlled structural experiment:
Add ONE leakage-safe joint target-encoding feature to the current
hierarchical-income+commute XGBoost champion.

HYPOTHESIS
----------
Income hierarchy and commute hierarchy both improved XGBoost independently.

The strongest useful income hierarchy was around the $10k scale, while the
strongest commute hierarchy was the 1 km scale.

A joint target encoding of:

    floor(Annual_Income_USD / 10000)
    x
    floor(Daily_Commute_km / 1)

may capture an affordability x usage-demand interaction that separate
marginal encodings cannot represent.

ONLY CHANGE
-----------
Add exactly one feature:

    JTE__income10k_x_commute1km

It is a nested leakage-safe Bayesian target encoding with smoothing m=2.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw features
- income digit decomposition
- exact income/commute frequency features
- exact income/commute nested Bayesian TE
- hierarchical income TE at 1k / 10k / 100k
- hierarchical commute TE at 1km / 5km / 10km
- learned nested logistic base margin
- smoothing m=2
- depth-4 XGBoost configuration
- seed 42
- CUDA GPU
- early stopping
- no public leaderboard optimization

LEAKAGE SAFETY
--------------
For each outer fold:
- outer-training rows get INNER-OOF joint TE
- outer-validation mapping is fit on outer-training only
- test mapping is fit on outer-training only

BASELINE
--------
Current hierarchical-income+commute XGBoost:
    0.94591178

Run:
    python xgboost_income_commute_joint_te_gpu.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm_engineered_learned_margin_cpu as feat
import xgboost_hierarchical_income_te_gpu as income_hte
import xgboost_hierarchical_commute_te_gpu as commute_hte


EXPECTED_HASH = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94591178
AUC_TOL = 2e-5
SMOOTHING = 2.0

FOLDS_PATH = Path("artifacts/validation/candidate_folds.csv")

BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_income_commute_joint_te_gpu"
)

JOINT_FEATURE = "JTE__income10k_x_commute1km"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(np.float64)
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            percentile_rank(a),
            percentile_rank(b),
        )[0, 1]
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(path)

    if len(df) != len(train):
        raise ValueError(
            f"OOF row mismatch: {path}"
        )

    if "row_index" not in df.columns:
        raise ValueError(
            f"Missing row_index: {path}"
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            f"OOF row-order mismatch: {path}"
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"Missing fold column: {path}"
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF fold mismatch: {path}"
        )

    for candidate in [
        "oof_prediction",
        "prediction",
    ]:
        if candidate in df.columns:
            pred = df[
                candidate
            ].to_numpy(
                np.float64
            )
            break
    else:
        raise ValueError(
            f"Could not find prediction column in {path}"
        )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"Non-finite baseline predictions: {path}"
        )

    return pred


def make_joint_key(
    frame: pd.DataFrame,
) -> pd.Series:
    income = pd.to_numeric(
        frame[
            "Annual_Income_USD"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    commute = pd.to_numeric(
        frame[
            "Daily_Commute_km"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if (
        not np.isfinite(income).all()
        or not np.isfinite(commute).all()
    ):
        raise ValueError(
            "Income or commute contains non-finite values."
        )

    income_bucket = np.floor(
        income / 10000.0
    ).astype(
        np.int64
    )

    commute_bucket = np.floor(
        commute / 1.0
    ).astype(
        np.int64
    )

    return pd.Series(
        income_bucket.astype(str)
        + "|"
        + commute_bucket.astype(str),
        index=frame.index,
        dtype="string",
    )


def fit_te_mapping(
    keys: pd.Series,
    y: np.ndarray,
    smoothing: float,
) -> tuple[
    pd.Series,
    float,
]:
    prior = float(
        np.mean(y)
    )

    tmp = pd.DataFrame(
        {
            "key": keys.astype(str).to_numpy(),
            "target": y,
        }
    )

    stats = (
        tmp.groupby(
            "key",
            sort=False,
        )["target"]
        .agg(
            ["sum", "count"]
        )
    )

    encoded = (
        stats["sum"]
        + smoothing
        * prior
    ) / (
        stats["count"]
        + smoothing
    )

    return (
        encoded,
        prior,
    )


def apply_te_mapping(
    keys: pd.Series,
    mapping: pd.Series,
    prior: float,
) -> np.ndarray:
    values = (
        keys.astype(str)
        .map(mapping)
        .fillna(prior)
        .to_numpy(
            dtype=np.float32
        )
    )

    return values


def build_joint_te_for_outer_fold(
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
    train_keys = make_joint_key(
        train
    )

    test_keys = make_joint_key(
        test
    )

    outer_train_mask = (
        fold_ids
        != outer_fold
    )

    outer_valid_mask = (
        fold_ids
        == outer_fold
    )

    outer_train_idx = np.flatnonzero(
        outer_train_mask
    )

    outer_valid_idx = np.flatnonzero(
        outer_valid_mask
    )

    inner_oof_full = np.full(
        len(train),
        np.nan,
        dtype=np.float32,
    )

    inner_rows = []

    # Inner-OOF TE for the rows that will train the outer-fold model.
    for inner_fold in range(5):
        if inner_fold == outer_fold:
            continue

        inner_valid_mask = (
            fold_ids
            == inner_fold
        )

        inner_fit_mask = (
            (fold_ids != outer_fold)
            & (fold_ids != inner_fold)
        )

        mapping, prior = fit_te_mapping(
            train_keys[
                inner_fit_mask
            ],
            y[
                inner_fit_mask
            ],
            smoothing,
        )

        encoded = apply_te_mapping(
            train_keys[
                inner_valid_mask
            ],
            mapping,
            prior,
        )

        inner_oof_full[
            inner_valid_mask
        ] = encoded

        unseen_rate = float(
            (
                ~train_keys[
                    inner_valid_mask
                ]
                .astype(str)
                .isin(
                    mapping.index
                )
            ).mean()
        )

        inner_rows.append(
            {
                "outer_fold": outer_fold,
                "split": f"inner_fold_{inner_fold}",
                "fit_rows": int(
                    inner_fit_mask.sum()
                ),
                "apply_rows": int(
                    inner_valid_mask.sum()
                ),
                "unique_fit_groups": int(
                    len(mapping)
                ),
                "unseen_rate": unseen_rate,
                "prior": prior,
            }
        )

    if np.isnan(
        inner_oof_full[
            outer_train_idx
        ]
    ).any():
        raise RuntimeError(
            "Inner-OOF joint TE contains NaNs."
        )

    # Outer-training mapping for validation and test.
    outer_mapping, outer_prior = (
        fit_te_mapping(
            train_keys[
                outer_train_mask
            ],
            y[
                outer_train_mask
            ],
            smoothing,
        )
    )

    valid_values = apply_te_mapping(
        train_keys[
            outer_valid_mask
        ],
        outer_mapping,
        outer_prior,
    )

    test_values = apply_te_mapping(
        test_keys,
        outer_mapping,
        outer_prior,
    )

    valid_unseen_rate = float(
        (
            ~train_keys[
                outer_valid_mask
            ]
            .astype(str)
            .isin(
                outer_mapping.index
            )
        ).mean()
    )

    test_unseen_rate = float(
        (
            ~test_keys
            .astype(str)
            .isin(
                outer_mapping.index
            )
        ).mean()
    )

    inner_rows.extend(
        [
            {
                "outer_fold": outer_fold,
                "split": "outer_validation",
                "fit_rows": int(
                    outer_train_mask.sum()
                ),
                "apply_rows": int(
                    outer_valid_mask.sum()
                ),
                "unique_fit_groups": int(
                    len(outer_mapping)
                ),
                "unseen_rate": valid_unseen_rate,
                "prior": outer_prior,
            },
            {
                "outer_fold": outer_fold,
                "split": "test",
                "fit_rows": int(
                    outer_train_mask.sum()
                ),
                "apply_rows": int(
                    len(test)
                ),
                "unique_fit_groups": int(
                    len(outer_mapping)
                ),
                "unseen_rate": test_unseen_rate,
                "prior": outer_prior,
            },
        ]
    )

    train_te = pd.DataFrame(
        {
            JOINT_FEATURE: (
                inner_oof_full[
                    outer_train_idx
                ]
            )
        }
    )

    valid_te = pd.DataFrame(
        {
            JOINT_FEATURE: (
                valid_values
            )
        }
    )

    test_te = pd.DataFrame(
        {
            JOINT_FEATURE: (
                test_values
            )
        }
    )

    diagnostics = pd.DataFrame(
        inner_rows
    )

    return (
        train_te,
        valid_te,
        test_te,
        diagnostics,
    )


def main() -> None:
    train_path = Path(
        "data/train.csv"
    )

    test_path = Path(
        "data/test.csv"
    )

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

    fold_hash = sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_HASH:
        raise ValueError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_HASH}\n"
            f"Found   : {fold_hash}"
        )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    fold_df = pd.read_csv(
        FOLDS_PATH
    )

    target = feat.detect_target(
        train,
        test,
    )

    y, positive_label = (
        feat.encode_binary_target(
            train[target]
        )
    )

    fold_ids, id_col = (
        feat.validate_folds(
            fold_df,
            train,
        )
    )

    baseline_oof = load_oof(
        BASELINE_OOF,
        train,
        fold_ids,
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
    ) > AUC_TOL:
        raise ValueError(
            "Baseline XGBoost artifact mismatch.\n"
            f"Expected ~{EXPECTED_BASELINE_AUC:.8f}\n"
            f"Loaded    {baseline_auc:.8f}"
        )

    raw_features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and c != id_col
        )
    ]

    raw_cats = (
        feat.detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    X0, X0_test = (
        feat.prepare_base_frames(
            train=train,
            test=test,
            raw_features=raw_features,
            categorical_features=raw_cats,
        )
    )

    (
        X1,
        X1_test,
        _,
    ) = feat.add_income_digit_features(
        X_train=X0,
        X_test=X0_test,
        train_source=train,
        test_source=test,
    )

    (
        X_base,
        X_base_test,
        _,
    ) = feat.add_exact_frequency_features(
        X_train=X1,
        X_test=X1_test,
        train_source=train,
        test_source=test,
    )

    logistic_train_matrix = (
        feat.build_logistic_recipe_matrix(
            train
        )
    )

    logistic_test_matrix = (
        feat.build_logistic_recipe_matrix(
            test
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("XGBOOST + JOINT INCOME10K x COMMUTE1KM TARGET ENCODING")
    print("=" * 100)
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Current XGB baseline: "
        f"{baseline_auc:.8f}"
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  Separate income and commute hierarchies are validated. "
        "A joint $10k-income x 1km-commute TE may capture their interaction."
    )
    print()
    print("ONLY CHANGE:")
    print(
        f"  Add one nested leakage-safe feature: {JOINT_FEATURE}"
    )
    print()
    print("HELD FIXED:")
    print("  - frozen folds + SHA256")
    print("  - all current raw/engineered features")
    print("  - exact TE")
    print("  - income hierarchy TE")
    print("  - commute hierarchy TE")
    print("  - smoothing m=2")
    print("  - learned logistic base margin")
    print("  - depth-4 XGBoost")
    print("  - seed 42")
    print("  - CUDA")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    fold_rows = []
    diagnostics_frames = []
    importance_rows = []

    total_start = (
        time.perf_counter()
    )

    for outer_fold in range(5):
        fold_start = (
            time.perf_counter()
        )

        train_idx = np.flatnonzero(
            fold_ids
            != outer_fold
        )

        valid_idx = np.flatnonzero(
            fold_ids
            == outer_fold
        )

        (
            train_exact,
            valid_exact,
            test_exact,
            _,
        ) = feat.build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        (
            train_income,
            valid_income,
            test_income,
            _,
        ) = income_hte.build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        (
            train_commute,
            valid_commute,
            test_commute,
            _,
        ) = commute_hte.build_hierarchical_commute_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        (
            train_joint,
            valid_joint,
            test_joint,
            joint_diag,
        ) = build_joint_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        diagnostics_frames.append(
            joint_diag
        )

        (
            bm_train,
            bm_valid,
            bm_test,
            _,
        ) = feat.build_learned_logistic_margins_for_outer_fold(
            train_matrix=logistic_train_matrix,
            test_matrix=logistic_test_matrix,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        X_train = (
            X_base.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        X_valid = (
            X_base.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        X_test = (
            X_base_test
            .reset_index(
                drop=True
            )
            .copy()
        )

        for (
            train_frame,
            valid_frame,
            test_frame,
        ) in [
            (
                train_exact,
                valid_exact,
                test_exact,
            ),
            (
                train_income,
                valid_income,
                test_income,
            ),
            (
                train_commute,
                valid_commute,
                test_commute,
            ),
            (
                train_joint,
                valid_joint,
                test_joint,
            ),
        ]:
            for c in (
                train_frame.columns
            ):
                X_train[c] = (
                    train_frame[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                X_valid[c] = (
                    valid_frame[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                X_test[c] = (
                    test_frame[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

        model = (
            income_hte.build_model()
        )

        model.fit(
            X_train,
            y[train_idx],
            base_margin=bm_train,
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            base_margin_eval_set=[
                bm_valid
            ],
            verbose=False,
        )

        if (
            model.best_iteration
            is None
        ):
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

        valid_pred = (
            model.predict_proba(
                X_valid,
                base_margin=bm_valid,
                iteration_range=iteration_range,
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test,
                base_margin=bm_test,
                iteration_range=iteration_range,
            )[:, 1]
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
                baseline_oof[
                    valid_idx
                ],
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
                "fold_seconds": (
                    time.perf_counter()
                    - fold_start
                ),
            }
        )

        booster = (
            model.get_booster()
        )

        gain_dict = (
            booster.get_score(
                importance_type="gain"
            )
        )

        importance_rows.append(
            {
                "fold": outer_fold,
                "feature": JOINT_FEATURE,
                "gain_importance": float(
                    gain_dict.get(
                        JOINT_FEATURE,
                        0.0,
                    )
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

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
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

    fold_metrics = (
        pd.DataFrame(
            fold_rows
        )
    )

    folds_improved = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ]
            > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ]
            < 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            baseline_oof,
            oof,
        )[0, 1]
    )

    rank_correlation = (
        rank_corr(
            baseline_oof,
            oof,
        )
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    diagnostics = pd.concat(
        diagnostics_frames,
        ignore_index=True,
    )

    importance_df = pd.DataFrame(
        importance_rows
    )

    mean_importance = float(
        importance_df[
            "gain_importance"
        ].mean()
    )

    decision = (
        "KEEP_JOINT_INCOME_COMMUTE_TE"
        if (
            delta_vs_baseline > 0
            and folds_improved >= 3
        )
        else
        "REJECT_JOINT_INCOME_COMMUTE_TE"
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    diagnostics.to_csv(
        OUTPUT_DIR
        / "joint_te_diagnostics.csv",
        index=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR
        / "joint_te_importance_by_fold.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": (
                oof.astype(
                    np.float32
                )
            ),
        }
    ).to_csv(
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

    if "id" in test.columns:
        test_output.insert(
            0,
            "id",
            test["id"].to_numpy(),
        )

    test_output.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: XGBOOST JOINT INCOME10K x COMMUTE1KM TE",
        "=" * 82,
        f"Baseline: {baseline_auc:.8f}",
        f"Candidate: {candidate_auc:.8f}",
        f"Delta: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr: {probability_corr:.6f}",
        f"Rank corr: {rank_correlation:.6f}",
        f"Mean joint-TE gain importance: {mean_importance:.8f}",
        f"Runtime: {total_seconds:.2f}s",
        f"Decision: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in (
        fold_metrics.itertuples()
    ):
        summary.append(
            f"Fold {row.fold}: "
            f"{row.baseline_auc:.8f} -> "
            f"{row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("JOINT INCOME x COMMUTE TE EXPERIMENT COMPLETE")
    print("=" * 100)
    print(
        f"Baseline      : "
        f"{baseline_auc:.8f}"
    )
    print(
        f"Candidate     : "
        f"{candidate_auc:.8f}"
    )
    print(
        f"Delta         : "
        f"{delta_vs_baseline:+.8f}"
    )
    print(
        f"Folds improved/worse: "
        f"{folds_improved}/{folds_worse}"
    )
    print(
        f"Probability corr: "
        f"{probability_corr:.6f}"
    )
    print(
        f"Rank corr       : "
        f"{rank_correlation:.6f}"
    )
    print(
        f"Mean JTE gain   : "
        f"{mean_importance:.8f}"
    )
    print(
        f"Decision        : "
        f"{decision}"
    )
    print(
        f"Runtime         : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts       : "
        f"{OUTPUT_DIR.resolve()}"
    )
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. joint_te_diagnostics.csv")
    print("  5. joint_te_importance_by_fold.csv")


if __name__ == "__main__":
    main()
