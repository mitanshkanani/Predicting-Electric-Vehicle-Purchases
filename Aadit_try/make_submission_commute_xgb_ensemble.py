"""
make_submission_commute_xgb_ensemble.py

Create a Kaggle submission from the CURRENT validated ensemble champion:

    hierarchical-income CatBoost 3-seed
    + hierarchical-income + hierarchical-commute XGBoost
    + old engineered LightGBM

Validated held-fold meta-CV:
    0.94595446

Mean deployment weights from the validated audit:
    CatBoost = 0.18
    XGBoost  = 0.60
    LightGBM = 0.22

IMPORTANT
---------
This script does NOT retrain models and does NOT recompute blend weights.

It uses the already-generated test predictions from:

    artifacts/experiments/
    blend_hierarchical_catboost_commute_xgb_lgbm_rank_audit/
    test_predictions.csv

Then it aligns those predictions to data/sample_submission.csv and writes:

    submissions/submission_current_best_094595446.csv

Run:
    python make_submission_commute_xgb_ensemble.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


VALIDATED_META_CV = 0.94595446

SAMPLE_SUBMISSION = (
    Path("data")
    / "sample_submission.csv"
)

PREDICTIONS = (
    Path("artifacts")
    / "experiments"
    / "blend_hierarchical_catboost_commute_xgb_lgbm_rank_audit"
    / "test_predictions.csv"
)

OUTPUT = (
    Path("submissions")
    / "submission_current_best_094595446.csv"
)


def choose_prediction_column(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> str:
    preferred = [
        "prediction",
        "test_prediction",
        target_col,
    ]

    for c in preferred:
        if c in df.columns:
            return c

    numeric_cols = [
        c
        for c in df.columns
        if (
            c != id_col
            and pd.api.types.is_numeric_dtype(
                df[c]
            )
        )
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Could not uniquely identify the prediction column.\n"
        f"Columns found: {list(df.columns)}"
    )


def main() -> None:
    if not SAMPLE_SUBMISSION.exists():
        raise FileNotFoundError(
            "Missing sample submission:\n"
            f"{SAMPLE_SUBMISSION.resolve()}"
        )

    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            "Missing validated ensemble predictions:\n"
            f"{PREDICTIONS.resolve()}"
        )

    sample = pd.read_csv(
        SAMPLE_SUBMISSION
    )

    pred_df = pd.read_csv(
        PREDICTIONS
    )

    if len(sample) != len(pred_df):
        raise ValueError(
            "Row-count mismatch.\n"
            f"sample_submission: {len(sample):,}\n"
            f"predictions      : {len(pred_df):,}"
        )

    if sample.shape[1] < 2:
        raise ValueError(
            "sample_submission.csv should contain "
            "an ID column and a target column."
        )

    id_col = sample.columns[0]

    target_cols = [
        c
        for c in sample.columns
        if c != id_col
    ]

    if len(target_cols) != 1:
        raise ValueError(
            "Expected exactly one submission target column.\n"
            f"Columns found: {list(sample.columns)}"
        )

    target_col = target_cols[0]

    if id_col in pred_df.columns:
        if not np.array_equal(
            sample[id_col].to_numpy(),
            pred_df[id_col].to_numpy(),
        ):
            raise ValueError(
                "Prediction IDs do not match "
                "sample_submission IDs."
            )

    prediction_col = (
        choose_prediction_column(
            pred_df,
            target_col,
            id_col,
        )
    )

    predictions = (
        pred_df[
            prediction_col
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    if (
        (predictions < 0)
        | (predictions > 1)
    ).any():
        raise ValueError(
            "Predictions are outside [0, 1]."
        )

    submission = sample.copy()

    submission[
        target_col
    ] = predictions

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT,
        index=False,
    )

    # Read it back once so we know the written file is structurally valid.
    check = pd.read_csv(
        OUTPUT
    )

    if len(check) != len(sample):
        raise RuntimeError(
            "Written submission row count changed unexpectedly."
        )

    if list(check.columns) != list(sample.columns):
        raise RuntimeError(
            "Written submission columns do not match sample_submission."
        )

    print("=" * 92)
    print("CURRENT BEST KAGGLE SUBMISSION CREATED")
    print("=" * 92)
    print(f"Validated ensemble meta-CV : {VALIDATED_META_CV:.8f}")
    print("Validated mean weights      : CAT=0.18 / XGB=0.60 / LGBM=0.22")
    print(f"Rows                        : {len(submission):,}")
    print(f"ID column                   : {id_col}")
    print(f"Target column               : {target_col}")
    print(f"Prediction column loaded    : {prediction_col}")
    print(f"Prediction min              : {predictions.min():.8f}")
    print(f"Prediction max              : {predictions.max():.8f}")
    print(f"Prediction mean             : {predictions.mean():.8f}")
    print(f"Prediction source           : {PREDICTIONS.resolve()}")
    print(f"Submission                  : {OUTPUT.resolve()}")
    print("=" * 92)


if __name__ == "__main__":
    main()
