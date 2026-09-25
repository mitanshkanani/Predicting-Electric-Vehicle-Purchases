"""
validation_setup.py

Phase 2A — Validation strategy audit + candidate fold creation
for Kaggle Playground Series S6E9: Predicting Electric Vehicle Purchases.

This script does NOT train any predictive model.

Goals:
1. Verify whether the sequential ID axis shows meaningful target drift.
2. Confirm that ordinary stratified random CV is reasonable.
3. Create one reproducible candidate 5-fold assignment.
4. Save diagnostics so we can inspect them before freezing the folds.

Run:
    python validation_setup.py

Optional:
    python validation_setup.py --data-dir data --output-dir artifacts/validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
DEFAULT_ID_BINS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit validation strategy and create candidate stratified folds."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "validation",
        help="Folder where validation artifacts will be written.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of candidate StratifiedKFold folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for fold creation.",
    )
    parser.add_argument(
        "--id-bins",
        type=int,
        default=DEFAULT_ID_BINS,
        help="Number of contiguous ID bins used for stationarity diagnostics.",
    )
    return parser.parse_args()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise ValueError(
            "Could not safely infer target. Expected exactly one train-only column, "
            f"found: {train_only}"
        )
    return train_only[0]


def detect_id(train: pd.DataFrame, test: pd.DataFrame, target: str) -> str | None:
    common = [c for c in train.columns if c in test.columns and c != target]

    exact = [c for c in common if c.lower() in {"id", "index", "row_id", "rowid"}]
    if exact:
        return exact[0]

    candidates = []
    for c in common:
        if train[c].isna().any() or test[c].isna().any():
            continue

        train_unique = train[c].nunique(dropna=False) == len(train)
        test_unique = test[c].nunique(dropna=False) == len(test)

        if train_unique and test_unique:
            candidates.append(c)

    return candidates[0] if len(candidates) == 1 else None


def encode_binary_target(y: pd.Series) -> tuple[pd.Series, object]:
    values = list(pd.unique(y.dropna()))

    if len(values) != 2:
        raise ValueError(
            f"Expected a binary target, but found {len(values)} unique values: {values}"
        )

    positive_candidates = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive = None
    for value in values:
        if str(value).strip().lower() in positive_candidates:
            positive = value
            break

    if positive is None:
        positive = y.value_counts().idxmin()

    encoded = (y == positive).astype(np.int8)
    return encoded, positive


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()

    if n == 0:
        return float("nan"), float("nan")

    rows, cols = table.shape
    denom = n * max(min(rows - 1, cols - 1), 1)
    v = np.sqrt(chi2 / denom)
    return float(v), float(p_value)


def build_id_stationarity_table(
    train: pd.DataFrame,
    id_col: str,
    y_bin: pd.Series,
    n_bins: int,
) -> pd.DataFrame:
    order = np.argsort(train[id_col].to_numpy(), kind="mergesort")
    n = len(train)

    bin_number = np.floor(np.arange(n) * n_bins / n).astype(int)
    bin_number = np.minimum(bin_number, n_bins - 1)

    sorted_ids = train[id_col].to_numpy()[order]
    sorted_y = y_bin.to_numpy()[order]

    tmp = pd.DataFrame(
        {
            "id_bin": bin_number,
            "id_value": sorted_ids,
            "target": sorted_y,
        }
    )

    return (
        tmp.groupby("id_bin", observed=True)
        .agg(
            row_count=("target", "size"),
            positive_count=("target", "sum"),
            positive_rate=("target", "mean"),
            id_min=("id_value", "min"),
            id_max=("id_value", "max"),
        )
        .reset_index()
    )


def create_candidate_folds(
    train: pd.DataFrame,
    y_bin: pd.Series,
    id_col: str | None,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    dummy_x = np.zeros(len(train), dtype=np.int8)

    for fold, (_, valid_idx) in enumerate(splitter.split(dummy_x, y_bin)):
        fold_assignment[valid_idx] = fold

    folds = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold_assignment,
        }
    )

    if id_col is not None:
        folds.insert(1, id_col, train[id_col].to_numpy())

    fold_report = (
        pd.DataFrame(
            {
                "fold": fold_assignment,
                "target": y_bin.to_numpy(),
            }
        )
        .groupby("fold")
        .agg(
            row_count=("target", "size"),
            positive_count=("target", "sum"),
            positive_rate=("target", "mean"),
        )
        .reset_index()
    )

    return folds, fold_report


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path.resolve()}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path.resolve()}")

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.id_bins < 4:
        raise ValueError("--id-bins should be at least 4.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train/test...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = detect_target(train, test)
    id_col = detect_id(train, test, target)
    y_bin, positive_label = encode_binary_target(train[target])

    overall_rate = float(y_bin.mean())

    summary_lines = [
        "VALIDATION STRATEGY AUDIT",
        "=" * 72,
        f"Train shape: {train.shape}",
        f"Test shape: {test.shape}",
        f"Target: {target!r}",
        f"Positive label used for diagnostics: {positive_label!r}",
        f"Overall positive rate: {overall_rate:.6f}",
        f"Detected ID: {id_col!r}",
        "",
    ]

    id_stationarity = None
    id_auc = float("nan")
    id_auc_strength = float("nan")
    id_rate_range = float("nan")
    id_rate_std = float("nan")
    id_cramers_v = float("nan")
    id_chi2_p = float("nan")
    first_10_rate = float("nan")
    last_10_rate = float("nan")
    first_last_gap = float("nan")
    sequential_test_ids = False

    if id_col is not None:
        print("Checking target stationarity along the ID axis...")

        id_values = pd.to_numeric(train[id_col], errors="coerce")
        test_id_values = pd.to_numeric(test[id_col], errors="coerce")

        if id_values.notna().all() and test_id_values.notna().all():
            raw_auc = roc_auc_score(y_bin, id_values)
            id_auc = float(raw_auc)
            id_auc_strength = float(max(raw_auc, 1.0 - raw_auc))

            id_stationarity = build_id_stationarity_table(
                train=train,
                id_col=id_col,
                y_bin=y_bin,
                n_bins=args.id_bins,
            )

            id_rate_range = float(
                id_stationarity["positive_rate"].max()
                - id_stationarity["positive_rate"].min()
            )
            id_rate_std = float(
                id_stationarity["positive_rate"].std(ddof=0)
            )

            # Create the same contiguous ID-bin labels for every original row.
            # These labels are then cross-tabulated against the target.
            sorted_idx = np.argsort(
                train[id_col].to_numpy(),
                kind="mergesort",
            )

            labels_sorted = np.floor(
                np.arange(len(train)) * args.id_bins / len(train)
            ).astype(int)
            labels_sorted = np.minimum(
                labels_sorted,
                args.id_bins - 1,
            )

            labels_original = np.empty(len(train), dtype=int)
            labels_original[sorted_idx] = labels_sorted

            chi_table = pd.crosstab(labels_original, y_bin)
            id_cramers_v, id_chi2_p = cramers_v(chi_table)

            sorted_y = y_bin.to_numpy()[sorted_idx]
            ten_pct = max(int(len(sorted_y) * 0.10), 1)

            first_10_rate = float(sorted_y[:ten_pct].mean())
            last_10_rate = float(sorted_y[-ten_pct:].mean())
            first_last_gap = float(last_10_rate - first_10_rate)

            train_max = float(id_values.max())
            test_min = float(test_id_values.min())
            train_min = float(id_values.min())
            test_max = float(test_id_values.max())

            sequential_test_ids = test_min > train_max

            summary_lines.extend(
                [
                    "ID / ORDER DIAGNOSTICS",
                    f"Train ID range: {train_min:g} → {train_max:g}",
                    f"Test ID range: {test_min:g} → {test_max:g}",
                    f"Test IDs begin after train IDs: {sequential_test_ids}",
                    f"ID univariate ROC-AUC: {id_auc:.6f}",
                    f"ID AUC strength: {id_auc_strength:.6f}",
                    f"Positive-rate range across {args.id_bins} contiguous ID bins: "
                    f"{id_rate_range:.6f}",
                    f"Std. dev. of ID-bin positive rates: {id_rate_std:.6f}",
                    f"First 10% positive rate: {first_10_rate:.6f}",
                    f"Last 10% positive rate: {last_10_rate:.6f}",
                    f"Last-minus-first 10% rate gap: {first_last_gap:+.6f}",
                    f"Cramer's V(target, ID-bin): {id_cramers_v:.6f}",
                    f"Chi-square p-value: {id_chi2_p:.6g}",
                    "",
                ]
            )

            id_stationarity.to_csv(
                args.output_dir / "id_stationarity.csv",
                index=False,
            )

        else:
            summary_lines.extend(
                [
                    "ID / ORDER DIAGNOSTICS",
                    "ID was detected but is not fully numeric, so numeric/order "
                    "stationarity diagnostics were skipped.",
                    "",
                ]
            )

    else:
        summary_lines.extend(
            [
                "ID / ORDER DIAGNOSTICS",
                "No unique ID column could be detected safely.",
                "",
            ]
        )

    print("Creating candidate stratified folds...")

    folds, fold_report = create_candidate_folds(
        train=train,
        y_bin=y_bin,
        id_col=id_col,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    folds_path = args.output_dir / "candidate_folds.csv"
    fold_report_path = args.output_dir / "fold_balance.csv"

    folds.to_csv(folds_path, index=False)
    fold_report.to_csv(fold_report_path, index=False)

    fold_rate_range = float(
        fold_report["positive_rate"].max()
        - fold_report["positive_rate"].min()
    )

    summary_lines.extend(
        [
            "CANDIDATE CV",
            f"Splitter: StratifiedKFold(n_splits={args.n_splits}, "
            f"shuffle=True, random_state={args.seed})",
            f"Fold positive-rate range: {fold_rate_range:.8f}",
            "",
        ]
    )

    warnings = []

    if np.isfinite(id_auc_strength) and id_auc_strength > 0.52:
        warnings.append(
            f"ID has non-trivial univariate target ordering "
            f"(AUC strength={id_auc_strength:.4f})."
        )

    if np.isfinite(id_rate_range) and id_rate_range > 0.02:
        warnings.append(
            f"Target prevalence varies by more than 2 percentage points across "
            f"contiguous ID bins (range={id_rate_range:.4f})."
        )

    if np.isfinite(first_last_gap) and abs(first_last_gap) > 0.015:
        warnings.append(
            f"First-vs-last 10% target-rate gap exceeds 1.5 percentage points "
            f"({first_last_gap:+.4f})."
        )

    if np.isfinite(id_cramers_v) and id_cramers_v > 0.02:
        warnings.append(
            f"ID-bin association has Cramer's V={id_cramers_v:.4f}, "
            "which deserves investigation."
        )

    summary_lines.append("INTERPRETATION")

    if warnings:
        summary_lines.append(
            "STATUS: REVIEW REQUIRED — do NOT freeze the candidate folds yet."
        )
        for warning in warnings:
            summary_lines.append(f"- {warning}")
    else:
        summary_lines.append(
            "STATUS: NO PRACTICALLY LARGE ID-ORDER TARGET DRIFT DETECTED."
        )
        summary_lines.append(
            "The candidate stratified folds are plausible, but we will inspect "
            "this report before declaring them frozen."
        )

    summary_lines.extend(
        [
            "",
            "IMPORTANT:",
            "- This script does not train a model.",
            "- candidate_folds.csv is not considered frozen until we review these results.",
            "- A tiny chi-square p-value alone is not evidence of useful drift when N is huge;",
            "  effect size and practical target-rate differences matter more here.",
        ]
    )

    summary_text = "\n".join(summary_lines)

    (args.output_dir / "validation_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(summary_text)
    print("=" * 78)
    print(f"Artifacts written to: {args.output_dir.resolve()}")
    print("Send me validation_summary.txt and id_stationarity.csv before modeling.")


if __name__ == "__main__":
    main()
