"""
Create Kaggle submission for the current locally validated champion:

    38% CatBoost 3-seed rank
    62% XGBoost exact-value Bayesian TE (m=2) rank

LOCAL EVIDENCE
--------------
Leakage-safe expanded-weight meta-CV OOF:
    0.94566717

Selected CAT weights by held fold:
    [0.37, 0.39, 0.38, 0.36, 0.41]

Mean selected CAT weight:
    0.3820

Deployment weight:
    0.38 CAT / 0.62 XGB

IMPORTANT
---------
This script does NOT train or tune anything.
It only packages the already-produced test predictions into Kaggle format.

Run:
    python make_submission_catboost_xgb_m2_blend_38_62.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Will_Buy_EV"
ID_COL = "id"

TEST_PATH = Path("data") / "test.csv"

PREDICTION_PATH = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_xgb_m2_expanded_weight_audit"
    / "test_predictions.csv"
)

OUTPUT_DIR = Path("submissions")

OUTPUT_PATH = (
    OUTPUT_DIR
    / "catboost_xgb_m2_rank_blend_38_62.csv"
)


def detect_prediction_column(
    df: pd.DataFrame,
) -> str:
    preferred = [
        "prediction",
        "test_prediction",
        "probability",
        "pred",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    candidates = [
        c
        for c in df.columns
        if c != ID_COL
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify prediction column. "
            f"Found columns: {list(df.columns)}"
        )

    return candidates[0]


def main() -> None:
    for path in [
        TEST_PATH,
        PREDICTION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    test = pd.read_csv(
        TEST_PATH
    )

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    if ID_COL not in test.columns:
        raise ValueError(
            f"{TEST_PATH} does not contain required ID column {ID_COL!r}."
        )

    if len(predictions) != len(test):
        raise ValueError(
            "Prediction row count does not match test.csv: "
            f"{len(predictions)} vs {len(test)}"
        )

    if (
        ID_COL in predictions.columns
        and not np.array_equal(
            predictions[
                ID_COL
            ].to_numpy(),
            test[
                ID_COL
            ].to_numpy(),
        )
    ):
        raise ValueError(
            "Prediction IDs do not align exactly with test.csv."
        )

    prediction_col = (
        detect_prediction_column(
            predictions
        )
    )

    pred = predictions[
        prediction_col
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    if (
        pred.min() < 0.0
        or pred.max() > 1.0
    ):
        raise ValueError(
            "Predictions must be in [0, 1]. "
            f"Observed range: [{pred.min()}, {pred.max()}]"
        )

    if np.unique(
        pred
    ).size < 100:
        raise ValueError(
            "Suspiciously few unique prediction values. "
            "Stopping instead of creating submission."
        )

    submission = pd.DataFrame(
        {
            ID_COL: test[
                ID_COL
            ].to_numpy(),
            TARGET: pred,
        }
    )

    if submission[
        ID_COL
    ].duplicated().any():
        raise ValueError(
            "Submission contains duplicate IDs."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 78)
    print("KAGGLE SUBMISSION CREATED")
    print("=" * 78)
    print(
        "Model: 38% CatBoost 3-seed rank "
        "+ 62% XGBoost exact-TE m=2 rank"
    )
    print(
        "Local leakage-safe meta-CV OOF: "
        "0.94566717"
    )
    print(
        f"Rows: {len(submission):,}"
    )
    print(
        f"Prediction source column: "
        f"{prediction_col!r}"
    )
    print(
        f"Prediction min: "
        f"{pred.min():.10f}"
    )
    print(
        f"Prediction max: "
        f"{pred.max():.10f}"
    )
    print(
        f"Prediction mean: "
        f"{pred.mean():.10f}"
    )
    print(
        f"Unique predictions: "
        f"{np.unique(pred).size:,}"
    )
    print(
        f"Output: "
        f"{OUTPUT_PATH.resolve()}"
    )
    print("=" * 78)
    print()
    print(
        "Upload this CSV to Kaggle, then send me:"
    )
    print(
        "  Public score:"
    )
    print(
        "  Rank:"
    )


if __name__ == "__main__":
    main()
