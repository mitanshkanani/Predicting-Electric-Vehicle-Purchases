"""
make_submission_current_best.py

Create a Kaggle submission from the CURRENT validated ensemble champion:

    hierarchical CatBoost 3-seed
    + hierarchical-income XGBoost
    + old engineered LightGBM

Validated meta-CV:
    0.94592600

The predictions are already stored at:
    artifacts/experiments/blend_hierarchical_catboost_xgb_lgbm_rank_audit/
        test_predictions.csv

This script does NOT retrain or reblend anything.

It:
1. loads data/sample_submission.csv
2. loads the validated ensemble test_predictions.csv
3. verifies row count and ID alignment
4. writes the prediction into the sample submission target column
5. saves:
       submissions/submission_current_best_094592600.csv

Run:
    python make_submission_current_best.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"

PREDICTIONS = (
    Path("artifacts")
    / "experiments"
    / "blend_hierarchical_catboost_xgb_lgbm_rank_audit"
    / "test_predictions.csv"
)

OUTPUT = (
    Path("submissions")
    / "submission_current_best_094592600.csv"
)


def main() -> None:
    if not SAMPLE_SUBMISSION.exists():
        raise FileNotFoundError(
            f"Missing sample submission:\n{SAMPLE_SUBMISSION.resolve()}"
        )

    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing validated ensemble predictions:\n{PREDICTIONS.resolve()}"
        )

    sample = pd.read_csv(SAMPLE_SUBMISSION)
    pred_df = pd.read_csv(PREDICTIONS)

    if len(sample) != len(pred_df):
        raise ValueError(
            "Row count mismatch.\n"
            f"sample_submission: {len(sample):,}\n"
            f"predictions      : {len(pred_df):,}"
        )

    if sample.shape[1] < 2:
        raise ValueError(
            "sample_submission.csv should contain an ID column and target column."
        )

    id_col = sample.columns[0]

    target_cols = [
        c for c in sample.columns
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
                "Prediction IDs do not match sample_submission IDs."
            )

    prediction_candidates = [
        "prediction",
        "test_prediction",
        target_col,
    ]

    prediction_col = None

    for c in prediction_candidates:
        if c in pred_df.columns:
            prediction_col = c
            break

    if prediction_col is None:
        numeric_cols = [
            c
            for c in pred_df.columns
            if c != id_col
            and pd.api.types.is_numeric_dtype(pred_df[c])
        ]

        if len(numeric_cols) == 1:
            prediction_col = numeric_cols[0]
        else:
            raise ValueError(
                "Could not uniquely identify the prediction column.\n"
                f"Columns: {list(pred_df.columns)}"
            )

    predictions = pred_df[prediction_col].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError(
            "Predictions are outside [0, 1]."
        )

    submission = sample.copy()

    submission[target_col] = predictions

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 88)
    print("CURRENT BEST SUBMISSION CREATED")
    print("=" * 88)
    print("Validated ensemble meta-CV : 0.94592600")
    print(f"Rows                       : {len(submission):,}")
    print(f"ID column                  : {id_col}")
    print(f"Target column              : {target_col}")
    print(f"Prediction source          : {PREDICTIONS.resolve()}")
    print(f"Prediction min             : {predictions.min():.8f}")
    print(f"Prediction max             : {predictions.max():.8f}")
    print(f"Prediction mean            : {predictions.mean():.8f}")
    print(f"Submission                 : {OUTPUT.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
