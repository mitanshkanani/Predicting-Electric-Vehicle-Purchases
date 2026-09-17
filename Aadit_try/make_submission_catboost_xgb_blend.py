"""
make_submission_catboost_xgb_blend.py

Create a Kaggle submission from the leakage-safe CatBoost/XGBoost blend.

Model:
    81% CatBoost rank
    19% XGBoost rank

Evidence:
    Meta-CV OOF AUC ≈ 0.94549841
    Improvement vs CatBoost champion ≈ +0.00005886
    Held-fold wins: 5/5

This script DOES NOT retrain anything.

Reads:
    data/train.csv
    data/test.csv
    data/sample_submission.csv
    artifacts/experiments/blend_catboost_xgb_exact_te_audit/test_predictions.csv

Writes:
    submissions/catboost_xgb_rank_blend_81_19.csv

Run:
    python make_submission_catboost_xgb_blend.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = (
    Path("artifacts")
    / "experiments"
    / "blend_catboost_xgb_exact_te_audit"
    / "test_predictions.csv"
)

DEFAULT_OUTPUT = (
    Path("submissions")
    / "catboost_xgb_rank_blend_81_19.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Kaggle submission from 81/19 CAT/XGB rank blend."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def detect_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    train_only = [
        c
        for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            f"Expected exactly one train-only target column, found {train_only}"
        )

    return train_only[0]


def detect_id(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
) -> str | None:
    for c in sample.columns:
        if (
            c in test.columns
            and c != target
            and c.lower()
            in {
                "id",
                "row_id",
                "rowid",
                "index",
            }
        ):
            return c

    candidates = [
        c
        for c in sample.columns
        if c != target
        and c in test.columns
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def main() -> None:
    args = parse_args()

    train_path = (
        args.data_dir
        / "train.csv"
    )

    test_path = (
        args.data_dir
        / "test.csv"
    )

    sample_path = (
        args.data_dir
        / "sample_submission.csv"
    )

    for path in [
        train_path,
        test_path,
        sample_path,
        args.predictions,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    print(
        "Loading competition files and saved blend predictions..."
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    sample = pd.read_csv(
        sample_path
    )

    predictions = pd.read_csv(
        args.predictions
    )

    target = detect_target(
        train,
        test,
    )

    if target not in sample.columns:
        raise ValueError(
            f"sample_submission.csv does not contain target {target!r}."
        )

    id_col = detect_id(
        test,
        sample,
        target,
    )

    if len(sample) != len(test):
        raise ValueError(
            f"Sample submission has {len(sample):,} rows "
            f"but test.csv has {len(test):,}."
        )

    if len(predictions) != len(test):
        raise ValueError(
            f"Prediction file has {len(predictions):,} rows "
            f"but test.csv has {len(test):,}."
        )

    if "prediction" not in predictions.columns:
        raise ValueError(
            "Blend prediction file must contain a 'prediction' column."
        )

    # ---------------------------------------------------------
    # ID alignment
    # ---------------------------------------------------------

    if id_col is not None:
        if id_col not in predictions.columns:
            raise ValueError(
                f"Prediction file is missing ID column {id_col!r}."
            )

        test_ids = test[
            id_col
        ].to_numpy()

        sample_ids = sample[
            id_col
        ].to_numpy()

        prediction_ids = predictions[
            id_col
        ].to_numpy()

        if not np.array_equal(
            test_ids,
            sample_ids,
        ):
            raise ValueError(
                "sample_submission.csv ID order differs from test.csv."
            )

        if not np.array_equal(
            test_ids,
            prediction_ids,
        ):
            raise ValueError(
                "Saved blend prediction ID order differs from test.csv."
            )

    # ---------------------------------------------------------
    # Prediction validation
    # ---------------------------------------------------------

    pred = pd.to_numeric(
        predictions[
            "prediction"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if np.isnan(
        pred
    ).any():
        raise ValueError(
            "Blend predictions contain NaN values."
        )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "Blend predictions contain non-finite values."
        )

    # Rank blends naturally fall in roughly [0,1].
    if (
        (pred < 0).any()
        or
        (pred > 1).any()
    ):
        raise ValueError(
            "Blend predictions should lie in [0, 1]."
        )

    if np.std(
        pred
    ) == 0:
        raise ValueError(
            "All predictions are identical."
        )

    # ---------------------------------------------------------
    # Build submission
    # ---------------------------------------------------------

    submission = sample.copy()

    submission[
        target
    ] = pred

    if (
        submission.columns.tolist()
        != sample.columns.tolist()
    ):
        raise RuntimeError(
            "Submission column order changed unexpectedly."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        args.output,
        index=False,
    )

    # ---------------------------------------------------------
    # Read-back audit
    # ---------------------------------------------------------

    check = pd.read_csv(
        args.output
    )

    if len(check) != len(test):
        raise RuntimeError(
            "Saved submission has incorrect row count."
        )

    if (
        check.columns.tolist()
        != sample.columns.tolist()
    ):
        raise RuntimeError(
            "Saved submission schema does not match Kaggle sample."
        )

    if check[
        target
    ].isna().any():
        raise RuntimeError(
            "Saved submission contains missing predictions."
        )

    print()
    print("=" * 78)
    print(
        "CATBOOST + XGBOOST BLEND SUBMISSION CREATED"
    )
    print("=" * 78)

    print(
        "Blend                : 81% CatBoost rank / 19% XGBoost rank"
    )

    print(
        "Meta-CV OOF AUC      : 0.94549841"
    )

    print(
        "CatBoost champion OOF: 0.94543956"
    )

    print(
        "Meta-CV gain         : +0.00005886"
    )

    print(
        "Held-fold wins       : 5/5"
    )

    print(
        f"Rows                 : {len(submission):,}"
    )

    print(
        f"Target column        : {target!r}"
    )

    print(
        f"ID column            : {id_col!r}"
    )

    print(
        f"Prediction min       : {pred.min():.8f}"
    )

    print(
        f"Prediction max       : {pred.max():.8f}"
    )

    print(
        f"Prediction mean      : {pred.mean():.8f}"
    )

    print(
        f"Prediction std       : {pred.std():.8f}"
    )

    print(
        f"Submission path      : {args.output.resolve()}"
    )

    print("=" * 78)
    print()
    print(
        "Upload this CSV to Kaggle."
    )
    print()
    print(
        "Then send me:"
    )
    print(
        "  Public score:"
    )
    print(
        "  Rank:"
    )


if __name__ == "__main__":
    main()
