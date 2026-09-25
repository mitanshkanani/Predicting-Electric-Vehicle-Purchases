"""
make_submission_xgb_depth4.py

Phase 4D — First Kaggle submission.

PURPOSE
-------
Create our first clean Kaggle submission from the best XGBoost depth-sweep
model (max_depth=4, local OOF AUC ≈ 0.94206547).

This script does NOT retrain anything.

It:
1. Reads Kaggle's sample_submission.csv as the schema source of truth.
2. Reads the saved test predictions from the depth sweep.
3. Verifies row count, ID alignment, finite predictions, and prediction range.
4. Writes a submission CSV with exactly the same columns/order as Kaggle's
   sample submission.

Expected inputs:
    data/sample_submission.csv
    data/test.csv
    artifacts/experiments/xgboost_depth_sweep_gpu/best_test_predictions.csv

Output:
    submissions/xgb_depth4_first_submission.csv

Run:
    python make_submission_xgb_depth4.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_test_predictions.csv"
)

DEFAULT_OUTPUT = (
    Path("submissions")
    / "xgb_depth4_first_submission.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create first Kaggle submission from depth-4 XGBoost predictions."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing train.csv, test.csv, and sample_submission.csv.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Saved depth-sweep best_test_predictions.csv.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Submission CSV to create.",
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
            "Could not safely infer target. "
            f"Expected one train-only column, found: {train_only}"
        )

    return train_only[0]


def detect_id_column(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
) -> str | None:
    # Prefer conventional ID names shared by test and sample.
    for c in sample.columns:
        if (
            c in test.columns
            and c != target
            and c.lower() in {"id", "row_id", "rowid", "index"}
        ):
            return c

    # Otherwise, if sample has exactly one non-target column that also exists
    # in test.csv, use it as the submission ID.
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

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"

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

    print("Loading competition files and saved predictions...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    predictions = pd.read_csv(args.predictions)

    target = detect_target(
        train,
        test,
    )

    if target not in sample.columns:
        raise ValueError(
            f"Kaggle sample_submission.csv does not contain target "
            f"column {target!r}.\n"
            f"Sample columns: {sample.columns.tolist()}"
        )

    id_col = detect_id_column(
        test,
        sample,
        target,
    )

    if len(sample) != len(test):
        raise ValueError(
            f"Sample submission has {len(sample):,} rows but "
            f"test.csv has {len(test):,} rows."
        )

    if len(predictions) != len(test):
        raise ValueError(
            f"Prediction file has {len(predictions):,} rows but "
            f"test.csv has {len(test):,} rows."
        )

    if "prediction" not in predictions.columns:
        raise ValueError(
            "Expected a 'prediction' column in "
            "best_test_predictions.csv.\n"
            f"Found columns: {predictions.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # Verify ID alignment
    # ---------------------------------------------------------

    if id_col is not None:
        if id_col not in predictions.columns:
            raise ValueError(
                f"Submission ID column {id_col!r} exists in sample/test "
                "but is missing from the prediction file."
            )

        test_ids = test[id_col].to_numpy()
        sample_ids = sample[id_col].to_numpy()
        pred_ids = predictions[id_col].to_numpy()

        if not np.array_equal(
            test_ids,
            sample_ids,
        ):
            raise ValueError(
                "sample_submission.csv IDs are not in the exact same order "
                "as test.csv."
            )

        if not np.array_equal(
            test_ids,
            pred_ids,
        ):
            raise ValueError(
                "Saved prediction IDs are not in the exact same order "
                "as test.csv."
            )

    # ---------------------------------------------------------
    # Validate predictions
    # ---------------------------------------------------------

    pred = pd.to_numeric(
        predictions["prediction"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if np.isnan(pred).any():
        raise ValueError(
            "Predictions contain NaN values."
        )

    if not np.isfinite(pred).all():
        raise ValueError(
            "Predictions contain inf/-inf values."
        )

    if (
        (pred < 0).any()
        or
        (pred > 1).any()
    ):
        raise ValueError(
            "Expected probability predictions in [0, 1]. "
            f"Observed min={pred.min():.8f}, max={pred.max():.8f}"
        )

    if np.std(pred) == 0:
        raise ValueError(
            "All predictions are identical."
        )

    # ---------------------------------------------------------
    # Construct submission using Kaggle's sample as schema
    # ---------------------------------------------------------

    submission = sample.copy()

    submission[target] = pred

    # Strict schema check.
    if submission.columns.tolist() != sample.columns.tolist():
        raise RuntimeError(
            "Submission column order changed unexpectedly."
        )

    if len(submission) != len(sample):
        raise RuntimeError(
            "Submission row count changed unexpectedly."
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
    # Final audit
    # ---------------------------------------------------------

    reread = pd.read_csv(
        args.output
    )

    if reread.columns.tolist() != sample.columns.tolist():
        raise RuntimeError(
            "Saved submission does not match sample schema."
        )

    if len(reread) != len(sample):
        raise RuntimeError(
            "Saved submission row count is incorrect."
        )

    if reread[target].isna().any():
        raise RuntimeError(
            "Saved submission contains missing predictions."
        )

    print()
    print("=" * 78)
    print("FIRST KAGGLE SUBMISSION CREATED")
    print("=" * 78)
    print(f"Model                : XGBoost GPU, max_depth=4")
    print(f"Local OOF AUC        : 0.94206547")
    print(f"Rows                 : {len(submission):,}")
    print(f"Target column        : {target!r}")
    print(f"ID column            : {id_col!r}")
    print(f"Prediction min       : {pred.min():.8f}")
    print(f"Prediction max       : {pred.max():.8f}")
    print(f"Prediction mean      : {pred.mean():.8f}")
    print(f"Prediction std       : {pred.std():.8f}")
    print(f"Submission path      : {args.output.resolve()}")
    print("=" * 78)
    print()
    print("Upload this CSV to Kaggle, then send me:")
    print("  1. the public leaderboard score")
    print("  2. your leaderboard rank")
    print()
    print("Do not make additional submissions yet.")
    print("We first want to learn how local CV maps to the leaderboard.")


if __name__ == "__main__":
    main()
