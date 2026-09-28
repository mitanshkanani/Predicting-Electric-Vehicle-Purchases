"""
xgboost_fine_income_q1280_neighbor_te_gpu.py

Kaggle Playground Series S6E9

CONTROLLED STRUCTURAL EXPERIMENT
--------------------------------
Baseline:
    current best internal fine-income XGBoost
    OOF ~= 0.94606664

Only change:
    add three leakage-safe Annual_Income_USD features inspired by the
    Zoom Zoom R Edition notebook:

      1. NQTE__income_q1280_position
      2. NQTE__income_q1280_center
      3. NQTE__income_q1280_neighbor

The representation uses:
    - 1280 adaptive income quantile bins
    - Bayesian target smoothing = 10
    - adjacent-bin Gaussian kernel with sigma = 0.8
    - offsets [-1, 0, +1]

No Triple-TE.
No model-parameter changes.
No blending.
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
        "XGBoost is required.\n"
        "Install/update with:\n"
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

Q_BINS = 1280
Q_SMOOTHING = 10.0
KERNEL_SIGMA = 0.8

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
    / "xgboost_fine_income_q1280_neighbor_te_gpu"
)


def make_quantile_edges(
    values: np.ndarray,
    q_bins: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Annual_Income_USD contains non-finite values."
        )

    if (values < 0).any():
        raise RuntimeError(
            "Annual_Income_USD unexpectedly contains negative values."
        )

    probs = np.linspace(
        0.0,
        1.0,
        q_bins + 1,
        dtype=np.float64,
    )

    edges = np.quantile(
        values,
        probs,
        method="linear",
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        raise RuntimeError(
            "Too few unique quantile edges for neighbor QTE."
        )

    return edges


def assign_bins(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Return zero-based bin codes.

    R source notebook used findInterval(..., left.open=TRUE).
    np.searchsorted(..., side='left') is the corresponding boundary choice
    for this continuous-valued income feature.
    """
    values = np.asarray(values, dtype=np.float64)
    interior = edges[1:-1]

    codes = np.searchsorted(
        interior,
        values,
        side="left",
    ).astype(np.int32)

    return codes


def aggregate_bins(
    codes: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(
        codes,
        minlength=n_bins,
    ).astype(np.float64)

    sums = np.bincount(
        codes,
        weights=y.astype(np.float64),
        minlength=n_bins,
    ).astype(np.float64)

    return sums, counts


def fit_center_neighbor(
    codes: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    prior = float(np.mean(y))

    sums, counts = aggregate_bins(
        codes,
        y,
        n_bins,
    )

    center = (
        sums + smoothing * prior
    ) / (
        counts + smoothing
    )

    offsets = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    kernel = np.exp(
        -0.5
        * (offsets / KERNEL_SIGMA) ** 2
    )

    left_sums = np.concatenate(
        ([0.0], sums[:-1])
    )
    right_sums = np.concatenate(
        (sums[1:], [0.0])
    )

    left_counts = np.concatenate(
        ([0.0], counts[:-1])
    )
    right_counts = np.concatenate(
        (counts[1:], [0.0])
    )

    neighbor_sums = (
        kernel[0] * left_sums
        + kernel[1] * sums
        + kernel[2] * right_sums
    )

    neighbor_counts = (
        kernel[0] * left_counts
        + kernel[1] * counts
        + kernel[2] * right_counts
    )

    kernel_mass = float(kernel.sum())

    neighbor = (
        neighbor_sums
        + smoothing * kernel_mass * prior
    ) / (
        neighbor_counts
        + smoothing * kernel_mass
    )

    return center, neighbor, prior


def position_from_codes(
    codes: np.ndarray,
    q_bins: int,
) -> np.ndarray:
    return (
        codes.astype(np.float64)
        / max(q_bins - 1, 1)
    ).astype(np.float32)


def build_neighbor_qte_for_outer_fold(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Strict leakage-safe outer-fold implementation.

    Quantile edges for an outer fold are learned only from outer-training
    Annual_Income_USD values.

    Outer-training target-encoded features are inner-OOF:
      each row's TE statistics exclude its own frozen inner fold.

    Outer-validation/test mappings use the complete outer-training labels.
    """
    outer_train_idx = np.flatnonzero(
        fold_ids != outer_fold
    )
    outer_valid_idx = np.flatnonzero(
        fold_ids == outer_fold
    )

    outer_train_income = pd.to_numeric(
        train.loc[
            outer_train_idx,
            "Annual_Income_USD",
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    outer_valid_income = pd.to_numeric(
        train.loc[
            outer_valid_idx,
            "Annual_Income_USD",
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    test_income = pd.to_numeric(
        test["Annual_Income_USD"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    outer_train_y = y[outer_train_idx]
    outer_train_folds = fold_ids[
        outer_train_idx
    ]

    # Important: edges see outer-training X only, never outer-validation X.
    edges = make_quantile_edges(
        outer_train_income,
        Q_BINS,
    )

    n_bins = len(edges) - 1

    train_codes = assign_bins(
        outer_train_income,
        edges,
    )
    valid_codes = assign_bins(
        outer_valid_income,
        edges,
    )
    test_codes = assign_bins(
        test_income,
        edges,
    )

    if (
        train_codes.min() < 0
        or train_codes.max() >= n_bins
        or valid_codes.min() < 0
        or valid_codes.max() >= n_bins
        or test_codes.min() < 0
        or test_codes.max() >= n_bins
    ):
        raise RuntimeError(
            "Neighbor-QTE bin assignment out of range."
        )

    train_center = np.full(
        len(outer_train_idx),
        np.nan,
        dtype=np.float32,
    )
    train_neighbor = np.full(
        len(outer_train_idx),
        np.nan,
        dtype=np.float32,
    )

    # Use the remaining frozen folds as deterministic inner folds.
    for inner_fold in sorted(
        np.unique(outer_train_folds).tolist()
    ):
        inner_valid_mask = (
            outer_train_folds == inner_fold
        )
        inner_fit_mask = ~inner_valid_mask

        center_map, neighbor_map, _ = (
            fit_center_neighbor(
                codes=train_codes[
                    inner_fit_mask
                ],
                y=outer_train_y[
                    inner_fit_mask
                ],
                n_bins=n_bins,
                smoothing=Q_SMOOTHING,
            )
        )

        train_center[
            inner_valid_mask
        ] = center_map[
            train_codes[
                inner_valid_mask
            ]
        ].astype(np.float32)

        train_neighbor[
            inner_valid_mask
        ] = neighbor_map[
            train_codes[
                inner_valid_mask
            ]
        ].astype(np.float32)

    if (
        np.isnan(train_center).any()
        or np.isnan(train_neighbor).any()
    ):
        raise RuntimeError(
            f"Inner-OOF neighbor QTE contains NaNs "
            f"for outer fold {outer_fold}."
        )

    center_map, neighbor_map, outer_prior = (
        fit_center_neighbor(
            codes=train_codes,
            y=outer_train_y,
            n_bins=n_bins,
            smoothing=Q_SMOOTHING,
        )
    )

    valid_center = center_map[
        valid_codes
    ].astype(np.float32)

    valid_neighbor = neighbor_map[
        valid_codes
    ].astype(np.float32)

    test_center = center_map[
        test_codes
    ].astype(np.float32)

    test_neighbor = neighbor_map[
        test_codes
    ].astype(np.float32)

    train_out = pd.DataFrame(
        {
            "NQTE__income_q1280_position":
                position_from_codes(
                    train_codes,
                    Q_BINS,
                ),
            "NQTE__income_q1280_center":
                train_center,
            "NQTE__income_q1280_neighbor":
                train_neighbor,
        }
    )

    valid_out = pd.DataFrame(
        {
            "NQTE__income_q1280_position":
                position_from_codes(
                    valid_codes,
                    Q_BINS,
                ),
            "NQTE__income_q1280_center":
                valid_center,
            "NQTE__income_q1280_neighbor":
                valid_neighbor,
        }
    )

    test_out = pd.DataFrame(
        {
            "NQTE__income_q1280_position":
                position_from_codes(
                    test_codes,
                    Q_BINS,
                ),
            "NQTE__income_q1280_center":
                test_center,
            "NQTE__income_q1280_neighbor":
                test_neighbor,
        }
    )

    train_counts = np.bincount(
        train_codes,
        minlength=n_bins,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "outer_fold": outer_fold,
                "requested_quantile_bins": Q_BINS,
                "actual_bins": n_bins,
                "smoothing": Q_SMOOTHING,
                "kernel_sigma": KERNEL_SIGMA,
                "outer_train_prior": outer_prior,
                "min_bin_count": int(
                    train_counts.min()
                ),
                "median_bin_count": float(
                    np.median(train_counts)
                ),
                "max_bin_count": int(
                    train_counts.max()
                ),
                "empty_bins": int(
                    (train_counts == 0).sum()
                ),
                "edge_min": float(edges[0]),
                "edge_max": float(edges[-1]),
            }
        ]
    )

    return (
        train_out,
        valid_out,
        test_out,
        diagnostics,
    )


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
    train_path = Path("data") / "train.csv"
    test_path = Path("data") / "test.csv"

    required = [
        train_path,
        test_path,
        FOLDS_PATH,
        BASELINE_OOF_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = base.sha256_file(
        FOLDS_PATH
    )

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
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
        BASELINE_OOF_PATH,
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
        raise RuntimeError(
            "Fine-income baseline OOF mismatch.\n"
            f"Expected approximately: "
            f"{EXPECTED_BASELINE_AUC:.8f}\n"
            f"Found: {baseline_auc:.8f}"
        )

    raw_features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and c != id_col
        )
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

    print("=" * 104)
    print(
        "XGBOOST FINE-INCOME + "
        "Q1280 NEIGHBOR INCOME TARGET ENCODING"
    )
    print("=" * 104)
    print(
        f"XGBoost version      : "
        f"{xgb.__version__}"
    )
    print(
        f"Target               : "
        f"{target!r}"
    )
    print(
        f"Positive label       : "
        f"{positive_label!r}"
    )
    print(
        f"Frozen fold SHA256   : "
        f"{fold_hash}"
    )
    print(
        f"Fine-income baseline : "
        f"{baseline_auc:.8f}"
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add q1280 income position, "
        "central TE, and adjacent-bin "
        "Gaussian-smoothed TE."
    )
    print()
    print("SOURCE-FIXED PARAMETERS:")
    print(
        f"  quantile bins={Q_BINS}, "
        f"smoothing={Q_SMOOTHING:g}, "
        f"kernel sigma={KERNEL_SIGMA:g}"
    )
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds + SHA256")
    print("  - complete fine-income XGB representation")
    print("  - exact income + commute TE")
    print("  - hierarchical income TE")
    print("  - hierarchical commute TE")
    print("  - $50/$250 fine-income TE")
    print("  - income digits")
    print("  - exact income/commute frequencies")
    print("  - learned logistic base margin")
    print("  - XGBoost configuration")
    print("  - seed 42 / CUDA / early stopping")
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[
        np.ndarray
    ] = []

    fold_rows: list[dict] = []
    importance_rows: list[dict] = []

    qte_diag_frames: list[
        pd.DataFrame
    ] = []

    total_start = (
        time.perf_counter()
    )

    for outer_fold in range(5):
        fold_start = (
            time.perf_counter()
        )

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
            _,
        ) = base.build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=fine.SMOOTHING,
        )

        (
            train_income_hte,
            valid_income_hte,
            test_income_hte,
            _,
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

        (
            train_commute_hte,
            valid_commute_hte,
            test_commute_hte,
            _,
        ) = (
            fine
            .build_hierarchical_commute_te_for_outer_fold(
                train=train,
                test=test,
                y=y,
                fold_ids=fold_ids,
                outer_fold=outer_fold,
                smoothing=fine.SMOOTHING,
            )
        )

        (
            train_fine_income_te,
            valid_fine_income_te,
            test_fine_income_te,
            _,
        ) = (
            fine
            .build_fine_income_te_for_outer_fold(
                train=train,
                test=test,
                y=y,
                fold_ids=fold_ids,
                outer_fold=outer_fold,
                smoothing=fine.SMOOTHING,
            )
        )

        (
            train_neighbor_qte,
            valid_neighbor_qte,
            test_neighbor_qte,
            qte_diag,
        ) = build_neighbor_qte_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        qte_diag_frames.append(
            qte_diag
        )

        (
            learned_train_margin,
            learned_valid_margin,
            learned_test_margin,
            _,
        ) = (
            base
            .build_learned_logistic_margins_for_outer_fold(
                train_matrix=logistic_train_matrix,
                test_matrix=logistic_test_matrix,
                y=y,
                fold_ids=fold_ids,
                outer_fold=outer_fold,
            )
        )

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
                train_neighbor_qte,
                valid_neighbor_qte,
                test_neighbor_qte,
            ),
        ]

        for (
            train_features,
            valid_features,
            test_features,
        ) in feature_groups:
            for c in train_features.columns:
                X_train[c] = (
                    train_features[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                X_valid[c] = (
                    valid_features[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                X_test[c] = (
                    test_features[c]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

        qte_columns = list(
            train_neighbor_qte.columns
        )

        model = (
            income_hte.build_model()
        )

        fit_start = (
            time.perf_counter()
        )

        model.fit(
            X_train,
            y[train_idx],
            base_margin=(
                learned_train_margin
            ),
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
                base_margin=(
                    learned_valid_margin
                ),
                iteration_range=(
                    iteration_range
                ),
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test,
                base_margin=(
                    learned_test_margin
                ),
                iteration_range=(
                    iteration_range
                ),
            )[:, 1]
        )

        oof[valid_idx] = valid_pred

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

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_auc":
                    baseline_fold_auc,
                "candidate_auc":
                    candidate_fold_auc,
                "delta_vs_baseline":
                    delta,
                "best_iteration":
                    best_iteration,
                "fit_seconds":
                    fit_seconds,
                "total_fold_seconds":
                    (
                        time.perf_counter()
                        - fold_start
                    ),
            }
        )

        for (
            feature,
            importance,
        ) in zip(
            X_train.columns,
            model.feature_importances_,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature,
                    "is_neighbor_qte": (
                        feature
                        in qte_columns
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

    rank_correlation = (
        rank_corr(
            oof,
            baseline_oof,
        )
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    importance_by_fold = (
        pd.DataFrame(
            importance_rows
        )
    )

    qte_importance = (
        importance_by_fold[
            importance_by_fold[
                "is_neighbor_qte"
            ]
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
        delta_vs_baseline
        >= 3e-5
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
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    qte_importance.to_csv(
        OUTPUT_DIR
        / "neighbor_qte_importance.csv",
        index=False,
    )

    pd.concat(
        qte_diag_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "neighbor_qte_diagnostics.csv",
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
                oof.astype(
                    np.float32
                )
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

    summary = [
        (
            "EXPERIMENT: FINE-INCOME XGB "
            "+ Q1280 NEIGHBOR INCOME TE"
        ),
        "=" * 92,
        "",
        "TYPE",
        (
            "Competition-target model "
            "experiment."
        ),
        "",
        "HYPOTHESIS",
        (
            "Can adaptive equal-frequency "
            "income neighborhoods capture "
            "local purchase structure beyond "
            "the existing fixed-width income "
            "encodings?"
        ),
        "",
        "ONLY CHANGE",
        (
            "Add q1280 position, central "
            "target encoding, and adjacent-bin "
            "Gaussian-smoothed target encoding "
            "for Annual_Income_USD."
        ),
        "",
        "SOURCE-FIXED PARAMETERS",
        f"- requested quantile bins: {Q_BINS}",
        f"- target smoothing: {Q_SMOOTHING:g}",
        f"- kernel sigma: {KERNEL_SIGMA:g}",
        "",
        "HELD FIXED",
        (
            f"- frozen fold SHA256: "
            f"{fold_hash}"
        ),
        (
            "- complete current fine-income "
            "XGB feature stack"
        ),
        "- learned logistic base margin",
        (
            "- XGBoost configuration / seed "
            "/ CUDA / early stopping"
        ),
        "- no HPO",
        "- no blending",
        "- no leaderboard optimization",
        "",
        "RESULT",
        (
            f"Baseline OOF: "
            f"{baseline_auc:.8f}"
        ),
        (
            f"Candidate OOF: "
            f"{candidate_auc:.8f}"
        ),
        (
            f"Delta: "
            f"{delta_vs_baseline:+.8f}"
        ),
        (
            f"Folds improved: "
            f"{folds_improved}/5"
        ),
        (
            f"Folds worse: "
            f"{folds_worse}/5"
        ),
        (
            f"Probability corr: "
            f"{probability_corr:.6f}"
        ),
        (
            f"Rank corr: "
            f"{rank_correlation:.6f}"
        ),
        f"Decision: {decision}",
        (
            f"Runtime: "
            f"{total_seconds:.2f}s"
        ),
        "",
        "FOLD RESULTS",
    ]

    for r in fold_metrics.itertuples():
        summary.append(
            f"Fold {r.fold}: "
            f"baseline="
            f"{r.baseline_auc:.8f} -> "
            f"candidate="
            f"{r.candidate_auc:.8f} "
            f"({r.delta_vs_baseline:+.8f})"
        )

    summary.extend(
        [
            "",
            "NEIGHBOR-QTE IMPORTANCE",
        ]
    )

    for r in qte_importance.itertuples():
        summary.append(
            f"{r.feature}: "
            f"{r.mean_gain_importance:.8f}"
        )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print(
        "Q1280 NEIGHBOR INCOME "
        "TE EXPERIMENT COMPLETE"
    )
    print("=" * 104)
    print(
        f"Baseline OOF     : "
        f"{baseline_auc:.8f}"
    )
    print(
        f"Candidate OOF    : "
        f"{candidate_auc:.8f}"
    )
    print(
        f"Delta            : "
        f"{delta_vs_baseline:+.8f}"
    )
    print(
        f"Folds improved   : "
        f"{folds_improved}/5"
    )
    print(
        f"Folds worse      : "
        f"{folds_worse}/5"
    )
    print(
        f"Probability corr : "
        f"{probability_corr:.6f}"
    )
    print(
        f"Rank corr        : "
        f"{rank_correlation:.6f}"
    )
    print(
        f"Decision         : "
        f"{decision}"
    )
    print(
        f"Runtime          : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts        : "
        f"{OUTPUT_DIR.resolve()}"
    )
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. neighbor_qte_importance.csv")
    print("  5. neighbor_qte_diagnostics.csv")
    print("=" * 104)


if __name__ == "__main__":
    main()
