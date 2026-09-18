"""
make_submission_current_champion.py

Creates the Kaggle submission for the CURRENT validated champion:

Meta-CV: 0.94596441

Members:
- hierarchical income+commute CatBoost 3-seed
- hierarchical income+commute XGBoost seed 42
- old engineered LightGBM

This script DOES NOT retrain anything.
It only converts the already-saved champion test predictions into the
competition submission format.

Expected champion predictions:
artifacts/experiments/blend_income_commute_catboost_commute_xgb_lgbm_rank_audit/test_predictions.csv

Output:
submissions/submission_champion_094596441.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PREDICTION_PATH = (
    Path("artifacts")
    / "experiments"
    / "blend_income_commute_catboost_commute_xgb_lgbm_rank_audit"
    / "test_predictions.csv"
)

TEST_PATH = Path("data/test.csv")
SAMPLE_CANDIDATES = [
    Path("data/sample_submission.csv"),
    Path("sample_submission.csv"),
]
OUTPUT_DIR = Path("submissions")
OUTPUT_PATH = OUTPUT_DIR / "submission_champion_094596441.csv"


def find_sample_submission():
    for path in SAMPLE_CANDIDATES:
        if path.exists():
            return path
    return None


def find_prediction_column(df):
    for col in ["prediction", "test_prediction", "Will_Buy_EV"]:
        if col in df.columns:
            return col

    excluded = {"id", "row_index", "fold"}
    numeric = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) != 1:
        raise ValueError(
            "Could not safely identify prediction column.\n"
            f"Columns: {list(df.columns)}"
        )

    return numeric[0]


def main():
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "Current champion predictions were not found:\n"
            f"{PREDICTION_PATH.resolve()}\n\n"
            "Do not substitute another experiment's predictions."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test.csv:\n{TEST_PATH.resolve()}")

    pred_df = pd.read_csv(PREDICTION_PATH)
    test = pd.read_csv(TEST_PATH)

    if len(pred_df) != len(test):
        raise ValueError(
            "Prediction row count does not match test.csv.\n"
            f"Predictions: {len(pred_df):,}\n"
            f"Test rows:   {len(test):,}"
        )

    pred_col = find_prediction_column(pred_df)
    pred = pred_df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain NaN or infinity.")

    if pred.min() < 0 or pred.max() > 1:
        raise ValueError(
            "Predictions are outside [0, 1].\n"
            f"min={pred.min():.8f}, max={pred.max():.8f}"
        )

    sample_path = find_sample_submission()

    if sample_path is not None:
        sample = pd.read_csv(sample_path)

        if len(sample) != len(test):
            raise ValueError(
                "sample_submission row count does not match test.csv."
            )

        non_id_cols = [c for c in sample.columns if c.lower() != "id"]

        if len(non_id_cols) != 1:
            raise ValueError(
                "Could not safely determine target column from "
                "sample_submission.csv.\n"
                f"Columns: {list(sample.columns)}"
            )

        target_col = non_id_cols[0]
        submission = sample.copy()

        if "id" in submission.columns:
            if "id" not in test.columns:
                raise ValueError(
                    "sample_submission contains id but test.csv does not."
                )

            if not np.array_equal(
                submission["id"].to_numpy(),
                test["id"].to_numpy(),
            ):
                raise ValueError(
                    "sample_submission IDs do not align with test.csv."
                )

        submission[target_col] = pred

    else:
        if "id" not in test.columns:
            raise FileNotFoundError(
                "sample_submission.csv was not found and test.csv "
                "does not contain an id column."
            )

        target_col = "Will_Buy_EV"
        submission = pd.DataFrame({
            "id": test["id"].to_numpy(),
            target_col: pred,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print("=" * 90)
    print("CURRENT CHAMPION KAGGLE SUBMISSION CREATED")
    print("=" * 90)
    print("Validated meta-CV : 0.94596441")
    print(f"Prediction source : {PREDICTION_PATH}")
    print(f"Rows              : {len(submission):,}")
    print(f"Target column     : {target_col}")
    print(f"Prediction min    : {pred.min():.8f}")
    print(f"Prediction max    : {pred.max():.8f}")
    print(f"Prediction mean   : {pred.mean():.8f}")
    print(f"Output            : {OUTPUT_PATH.resolve()}")
    print("=" * 90)
    print()
    print("Upload THIS file to Kaggle:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
