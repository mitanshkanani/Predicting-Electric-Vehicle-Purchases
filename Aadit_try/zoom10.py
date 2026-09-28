"""
audit_zoom10_threeway_fixed_rank.py

S6E9 controlled ensemble audit.

BASELINE
--------
Current external-assisted champion:
    50% internal champion rank
    50% external XGB10 rank

CANDIDATE
---------
Fixed conservative diversification:
    45% internal champion rank
    45% external XGB10 rank
    10% Zoom Zoom frozen-5 LightGBM rank

Only change:
    allocate 10% of total ensemble weight to the newly reproduced Zoom Zoom
    standalone view, reducing the two incumbent members proportionally.

No training.
No weight sweep.
No meta-fitting.
No leaderboard optimization.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

FOLDS_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "candidate_folds.csv"
)

INTERNAL_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

INTERNAL_TEST_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "submission.csv"
)

XGB10_OOF_PATH = (
    ROOT
    / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_oof.csv"
)

XGB10_TEST_PATH = (
    ROOT
    / "external_oof_strong"
    / "XGBoost_Triple_TE_10folds_test.csv"
)

ZOOM_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "zoom_zoom_r_frozen5_lightgbm"
    / "oof_predictions.csv"
)

ZOOM_TEST_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "zoom_zoom_r_frozen5_lightgbm"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "zoom10_threeway_fixed_rank_audit"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_INTERNAL_AUC = 0.94611253
EXPECTED_XGB10_AUC = 0.94624283
EXPECTED_ZOOM_AUC = 0.94586490
EXPECTED_BASELINE_AUC = 0.94632348

AUC_TOL = 5e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    cols = [
        c for c in train.columns
        if c not in test.columns
    ]

    if len(cols) != 1:
        raise RuntimeError(
            f"Expected one train-only target column, found {cols}"
        )

    return cols[0]


def encode_target(
    y: pd.Series,
) -> np.ndarray:
    vals = list(
        pd.unique(
            y.dropna()
        )
    )

    yes = [
        v for v in vals
        if str(v).strip().lower() == "yes"
    ]

    if len(vals) != 2:
        raise RuntimeError(
            f"Expected binary target, found {vals}"
        )

    positive = (
        yes[0]
        if yes
        else y.value_counts().idxmin()
    )

    return (
        y == positive
    ).astype(
        np.int8
    ).to_numpy()


def auc(
    y: np.ndarray,
    p: np.ndarray,
) -> float:
    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def rank01(
    p: np.ndarray,
) -> np.ndarray:
    return (
        pd.Series(
            np.asarray(
                p,
                dtype=np.float64,
            )
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            rank01(a),
            rank01(b),
        )[0, 1]
    )


def load_internal_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(
        INTERNAL_OOF_PATH
    )

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(
        df.columns
    ):
        raise RuntimeError(
            "Internal OOF schema mismatch."
        )

    if len(df) != len(train):
        raise RuntimeError(
            "Internal OOF row-count mismatch."
        )

    if not np.array_equal(
        df[
            "row_index"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Internal OOF row alignment mismatch."
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(
            dtype=np.int64
        ),
        folds,
    ):
        raise RuntimeError(
            "Internal OOF fold alignment mismatch."
        )

    return pd.to_numeric(
        df[
            "oof_prediction"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )


def load_zoom_oof(
    train: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(
        ZOOM_OOF_PATH
    )

    required = {
        "row_index",
        "id",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(
        df.columns
    ):
        raise RuntimeError(
            "Zoom OOF schema mismatch."
        )

    if len(df) != len(train):
        raise RuntimeError(
            "Zoom OOF row-count mismatch."
        )

    if not np.array_equal(
        df[
            "row_index"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Zoom row_index mismatch."
        )

    if not np.array_equal(
        df[
            "id"
        ].to_numpy(),
        train[
            "id"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Zoom train ID mismatch."
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(
            dtype=np.int64
        ),
        folds,
    ):
        raise RuntimeError(
            "Zoom frozen-fold mismatch."
        )

    return pd.to_numeric(
        df[
            "oof_prediction"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )


def load_id_prediction(
    path: Path,
    reference: pd.DataFrame,
    prediction_col: str,
) -> np.ndarray:
    df = pd.read_csv(path)

    if len(df) != len(reference):
        raise RuntimeError(
            f"{path.name}: row-count mismatch."
        )

    if "id" not in df.columns:
        raise RuntimeError(
            f"{path.name}: missing id column."
        )

    if not np.array_equal(
        df[
            "id"
        ].to_numpy(),
        reference[
            "id"
        ].to_numpy(),
    ):
        raise RuntimeError(
            f"{path.name}: ID alignment mismatch."
        )

    if prediction_col not in df.columns:
        raise RuntimeError(
            f"{path.name}: missing {prediction_col!r}; "
            f"columns={list(df.columns)}"
        )

    p = pd.to_numeric(
        df[
            prediction_col
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        p
    ).all():
        raise RuntimeError(
            f"{path.name}: non-finite predictions."
        )

    return p


def main() -> None:
    start = (
        time.perf_counter()
    )

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        INTERNAL_OOF_PATH,
        INTERNAL_TEST_PATH,
        XGB10_OOF_PATH,
        XGB10_TEST_PATH,
        ZOOM_OOF_PATH,
        ZOOM_TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    fold_hash = (
        sha256_file(
            FOLDS_PATH
        )
    )

    if (
        fold_hash
        != EXPECTED_FOLD_SHA256
    ):
        raise RuntimeError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(
        TRAIN_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    fold_df = pd.read_csv(
        FOLDS_PATH
    )

    folds = fold_df[
        "fold"
    ].to_numpy(
        dtype=np.int64
    )

    target = detect_target(
        train,
        test,
    )

    y = encode_target(
        train[target]
    )

    internal_oof = (
        load_internal_oof(
            train,
            folds,
        )
    )

    xgb10_oof = (
        load_id_prediction(
            XGB10_OOF_PATH,
            train,
            "OOF_Pred",
        )
    )

    zoom_oof = (
        load_zoom_oof(
            train,
            folds,
        )
    )

    internal_auc = auc(
        y,
        internal_oof,
    )

    xgb10_auc = auc(
        y,
        xgb10_oof,
    )

    zoom_auc = auc(
        y,
        zoom_oof,
    )

    for (
        label,
        found,
        expected,
    ) in [
        (
            "internal",
            internal_auc,
            EXPECTED_INTERNAL_AUC,
        ),
        (
            "xgb10",
            xgb10_auc,
            EXPECTED_XGB10_AUC,
        ),
        (
            "zoom",
            zoom_auc,
            EXPECTED_ZOOM_AUC,
        ),
    ]:
        if abs(
            found
            - expected
        ) > AUC_TOL:
            raise RuntimeError(
                f"{label} AUC mismatch: "
                f"expected {expected:.8f}, "
                f"found {found:.8f}"
            )

    r_internal = rank01(
        internal_oof
    )

    r_xgb10 = rank01(
        xgb10_oof
    )

    r_zoom = rank01(
        zoom_oof
    )

    baseline_oof = (
        0.50
        * r_internal
        + 0.50
        * r_xgb10
    )

    baseline_auc = auc(
        y,
        baseline_oof,
    )

    if abs(
        baseline_auc
        - EXPECTED_BASELINE_AUC
    ) > AUC_TOL:
        raise RuntimeError(
            "Baseline champion mismatch.\n"
            f"Expected: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Found:    {baseline_auc:.8f}"
        )

    candidate_oof = (
        0.45
        * r_internal
        + 0.45
        * r_xgb10
        + 0.10
        * r_zoom
    )

    candidate_auc = auc(
        y,
        candidate_oof,
    )

    delta = (
        candidate_auc
        - baseline_auc
    )

    fold_rows = []

    for fold in range(5):
        mask = (
            folds
            == fold
        )

        base_fold = auc(
            y[mask],
            baseline_oof[
                mask
            ],
        )

        cand_fold = auc(
            y[mask],
            candidate_oof[
                mask
            ],
        )

        fold_rows.append(
            {
                "fold": fold,
                "baseline_auc":
                    base_fold,
                "candidate_auc":
                    cand_fold,
                "delta_vs_baseline":
                    (
                        cand_fold
                        - base_fold
                    ),
            }
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

    corr_zoom_baseline = (
        rank_corr(
            zoom_oof,
            baseline_oof,
        )
    )

    corr_candidate_baseline = (
        rank_corr(
            candidate_oof,
            baseline_oof,
        )
    )

    if (
        delta >= 3e-5
        and folds_improved >= 4
    ):
        primitive = (
            "POSITIVE_ZOOM10_"
            "THREEWAY_SIGNAL"
        )
    elif (
        delta <= 0
        or folds_worse >= 4
    ):
        primitive = (
            "NEGATIVE_ZOOM10_"
            "THREEWAY_SIGNAL"
        )
    else:
        primitive = (
            "WEAK_OR_INCONSISTENT_"
            "ZOOM10_THREEWAY_SIGNAL"
        )

    internal_test = (
        load_id_prediction(
            INTERNAL_TEST_PATH,
            test,
            "Will_Buy_EV",
        )
    )

    xgb10_test = (
        load_id_prediction(
            XGB10_TEST_PATH,
            test,
            "Will_Buy_EV",
        )
    )

    zoom_test = (
        load_id_prediction(
            ZOOM_TEST_PATH,
            test,
            "prediction",
        )
    )

    candidate_test = (
        0.45
        * rank01(
            internal_test
        )
        + 0.45
        * rank01(
            xgb10_test
        )
        + 0.10
        * rank01(
            zoom_test
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

    pd.DataFrame(
        [
            {
                "internal_auc":
                    internal_auc,
                "xgb10_auc":
                    xgb10_auc,
                "zoom_auc":
                    zoom_auc,
                "baseline_auc":
                    baseline_auc,
                "candidate_auc":
                    candidate_auc,
                "delta_vs_baseline":
                    delta,
                "folds_improved":
                    folds_improved,
                "folds_worse":
                    folds_worse,
                "zoom_rank_corr_vs_baseline":
                    corr_zoom_baseline,
                "candidate_rank_corr_vs_baseline":
                    corr_candidate_baseline,
                "primitive":
                    primitive,
            }
        ]
    ).to_csv(
        OUTPUT_DIR
        / "audit_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "id": test[
                "id"
            ].to_numpy(),
            target:
                candidate_test,
        }
    ).to_csv(
        OUTPUT_DIR
        / "candidate_submission.csv",
        index=False,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    summary = [
        "EXPERIMENT: FIXED 10% ZOOM ZOOM THREE-WAY RANK BLEND",
        "=" * 92,
        "",
        "TYPE",
        "Competition-target ensemble audit.",
        "",
        "HYPOTHESIS",
        (
            "Can a small fixed allocation to the weaker but structurally "
            "different Zoom Zoom model add marginal ensemble value?"
        ),
        "",
        "ONLY CHANGE",
        (
            "50% internal + 50% XGB10 -> "
            "45% internal + 45% XGB10 + 10% Zoom Zoom."
        ),
        "",
        "NO WEIGHT SEARCH",
        "The 10% diversification allocation was declared before evaluation.",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- plain percentile-rank geometry",
        "- all existing predictions",
        "- no training",
        "- no meta-fitting",
        "- no leaderboard optimization",
        "",
        "STANDALONE OOF",
        f"Internal: {internal_auc:.8f}",
        f"XGB10: {xgb10_auc:.8f}",
        f"Zoom Zoom: {zoom_auc:.8f}",
        "",
        "RESULT",
        f"Baseline champion: {baseline_auc:.8f}",
        f"10% Zoom candidate: {candidate_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Zoom rank corr vs baseline: {corr_zoom_baseline:.6f}",
        (
            "Candidate rank corr vs baseline: "
            f"{corr_candidate_baseline:.6f}"
        ),
        f"Primitive: {primitive}",
        f"Runtime: {runtime:.2f}s",
        "",
        "FOLD RESULTS",
    ]

    for row in (
        fold_metrics.itertuples()
    ):
        summary.append(
            f"Fold {row.fold}: "
            f"baseline={row.baseline_auc:.8f} -> "
            f"candidate={row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    summary.extend(
        [
            "",
            "DEPLOYMENT ARTIFACT",
            "candidate_submission.csv",
            "",
            "MANUAL REVIEW REQUIRED",
            "Do not submit automatically.",
        ]
    )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary
        ),
        encoding="utf-8",
    )

    print("=" * 108)
    print("FIXED 10% ZOOM ZOOM THREE-WAY RANK BLEND AUDIT")
    print("=" * 108)
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"Internal OOF       : {internal_auc:.8f}")
    print(f"XGB10 OOF          : {xgb10_auc:.8f}")
    print(f"Zoom Zoom OOF      : {zoom_auc:.8f}")
    print()
    print(f"Baseline           : {baseline_auc:.8f}")
    print(f"10% Zoom candidate : {candidate_auc:.8f}")
    print(f"Delta              : {delta:+.8f}")
    print(
        f"Zoom vs baseline rank corr : "
        f"{corr_zoom_baseline:.6f}"
    )
    print()

    for row in (
        fold_metrics.itertuples()
    ):
        print(
            f"Fold {row.fold}: "
            f"{row.baseline_auc:.8f} -> "
            f"{row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    print()
    print("=" * 108)
    print("RESULT")
    print("=" * 108)
    print(f"Baseline       : {baseline_auc:.8f}")
    print(f"Candidate      : {candidate_auc:.8f}")
    print(f"Delta          : {delta:+.8f}")
    print(f"Folds improved : {folds_improved}/5")
    print(f"Folds worse    : {folds_worse}/5")
    print(f"Primitive      : {primitive}")
    print(f"Runtime        : {runtime:.2f}s")
    print(f"Artifacts      : {OUTPUT_DIR}")
    print("=" * 108)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. audit_metrics.csv")


if __name__ == "__main__":
    main()
