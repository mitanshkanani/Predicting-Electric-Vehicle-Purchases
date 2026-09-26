"""
catboost_source_augmentation.py

Phase 3D — Source-data augmentation experiment.

Experiment question:
    Does adding the 10,000 labeled rows from the original
    "EV Adoption Behavior and Range Anxiety" dataset improve CatBoost OOF AUC
    on untouched competition validation rows?

Experimental-control rule:
    - SAME frozen competition folds.
    - SAME 13 competition features.
    - SAME feature-type policy.
    - SAME CatBoost hyperparameters.
    - ONLY change: append all original/source rows to each training fold.
    - Validation rows remain competition-only.

This is the leakage-safe way to test source augmentation.

Run:
    python catboost_source_augmentation.py

Requires:
    python -m pip install catboost kagglehub
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise SystemExit(
        "\nCatBoost is not installed.\n"
        "Install it with:\n"
        "    python -m pip install catboost\n"
    ) from exc


SEED = 42

SOURCE_HANDLE = "itzzomkar/ev-adoption-behavior-and-range-anxiety"
SOURCE_FILENAME = "EV_Adoption_and_Range_Anxiety_Dataset.csv"

DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"
DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw"
    / "oof_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_source_augmentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test source-data augmentation with frozen CatBoost CV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing competition train.csv and test.csv.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit source CSV path. If omitted, common paths are "
            "checked and kagglehub is used if necessary."
        ),
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
        help="Frozen competition fold assignment.",
    )
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=DEFAULT_BASELINE_OOF,
        help="Raw CatBoost baseline OOF predictions for direct comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for experiment artifacts.",
    )
    return parser.parse_args()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]

    if len(train_only) != 1:
        raise ValueError(
            "Expected exactly one train-only target column, "
            f"found: {train_only}"
        )

    return train_only[0]


def detect_competition_id(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> str | None:
    common = [
        c for c in test.columns
        if c in train.columns and c != target
    ]

    for c in common:
        if c.lower() in {"id", "row_id", "rowid", "index"}:
            return c

    return None


def detect_source_id(
    source: pd.DataFrame,
    target: str,
) -> str | None:
    for c in source.columns:
        if (
            c != target
            and c.lower()
            in {
                "buyer_id",
                "customer_id",
                "person_id",
                "id",
                "row_id",
                "rowid",
            }
        ):
            return c

    return None


def encode_binary_target(
    y: pd.Series,
) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))

    if len(values) != 2:
        raise ValueError(
            f"Expected binary target; found {values}"
        )

    preferred = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive = next(
        (
            v
            for v in values
            if str(v).strip().lower() in preferred
        ),
        y.value_counts().idxmin(),
    )

    encoded = (
        y == positive
    ).astype(np.int8).to_numpy()

    return encoded, positive


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def locate_source(
    explicit_path: Path | None,
    data_dir: Path,
) -> Path:
    candidates = []

    if explicit_path is not None:
        candidates.append(explicit_path)

    candidates.extend(
        [
            data_dir / "original" / SOURCE_FILENAME,
            data_dir / SOURCE_FILENAME,
            Path(SOURCE_FILENAME),
        ]
    )

    for path in candidates:
        if path.exists():
            print(
                f"Using local source dataset: "
                f"{path.resolve()}"
            )
            return path

    print(
        "Source dataset not found in the project; "
        "checking/downloading via kagglehub..."
    )

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "\nkagglehub is not installed.\n"
            "Install it with:\n"
            "    python -m pip install kagglehub\n"
        ) from exc

    root = Path(
        kagglehub.dataset_download(SOURCE_HANDLE)
    )

    matches = list(
        root.rglob(SOURCE_FILENAME)
    )

    if not matches:
        csvs = list(root.rglob("*.csv"))

        if len(csvs) == 1:
            matches = csvs

    if not matches:
        raise FileNotFoundError(
            f"Could not find the source CSV under {root}"
        )

    print(
        f"Using source dataset: "
        f"{matches[0].resolve()}"
    )

    return matches[0]


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {"row_index", "fold"}

    missing = required - set(folds.columns)

    if missing:
        raise ValueError(
            f"Fold file missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen folds do not match train row count."
        )

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        np.arange(len(train)),
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    unique_folds = sorted(
        folds["fold"].unique().tolist()
    )

    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"Expected folds 0-4, found {unique_folds}"
        )

    extra = [
        c
        for c in folds.columns
        if c not in {"row_index", "fold"}
    ]

    if len(extra) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extra}"
        )

    id_col = extra[0] if extra else None

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} missing from train."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                "Frozen fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(dtype=np.int16),
        id_col,
    )


def detect_categoricals(
    train: pd.DataFrame,
    features: list[str],
) -> list[str]:
    out = []

    for c in features:
        dtype = train[c].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(
                dtype,
                pd.CategoricalDtype,
            )
        ):
            out.append(c)

    return out


def align_source_types(
    source: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    categoricals: list[str],
) -> pd.DataFrame:
    aligned = source[features].copy()

    for c in features:
        if c in categoricals:
            # CatBoost categorical values should not be NaN.
            # The audit showed source missingness only in numeric fields,
            # but this makes the experiment robust anyway.
            aligned[c] = (
                aligned[c]
                .fillna("__MISSING__")
                .astype(str)
            )
        else:
            aligned[c] = pd.to_numeric(
                aligned[c],
                errors="coerce",
            )

    # Keep competition categorical representations consistent too.
    return aligned


def prepare_competition_types(
    frame: pd.DataFrame,
    features: list[str],
    categoricals: list[str],
) -> pd.DataFrame:
    out = frame[features].copy()

    for c in categoricals:
        out[c] = (
            out[c]
            .fillna("__MISSING__")
            .astype(str)
        )

    return out


def build_model() -> CatBoostClassifier:
    # EXACTLY the same baseline settings as baseline_catboost.py.
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        thread_count=-1,
        allow_writing_files=False,
    )


def safe_rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = pd.Series(a).rank(
        method="average"
    ).to_numpy()

    br = pd.Series(b).rank(
        method="average"
    ).to_numpy()

    return float(
        np.corrcoef(ar, br)[0, 1]
    )


def load_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            "Baseline CatBoost OOF file not found:\n"
            f"{path.resolve()}\n"
            "Do not continue without the baseline comparison."
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Baseline OOF file has unexpected columns."
        )

    if len(df) != len(train):
        raise ValueError(
            "Baseline OOF row count does not match train."
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train)),
    ):
        raise ValueError(
            "Baseline OOF row order is misaligned."
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Baseline OOF used different folds."
        )

    if (
        id_col is not None
        and id_col in df.columns
        and not np.array_equal(
            df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        )
    ):
        raise ValueError(
            "Baseline OOF IDs are misaligned."
        )

    return df["oof_prediction"].to_numpy(
        dtype=np.float64
    )


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    for path in (
        train_path,
        test_path,
        args.folds_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading competition data and frozen folds...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(
        train,
        test,
    )

    comp_y, positive_label = encode_binary_target(
        train[target]
    )

    fold_ids, fold_id_col = validate_folds(
        folds_df,
        train,
    )

    comp_id = detect_competition_id(
        train,
        test,
        target,
    )

    if (
        fold_id_col is not None
        and comp_id is not None
        and fold_id_col != comp_id
    ):
        raise ValueError(
            "Competition ID detection disagrees with frozen fold ID."
        )

    id_col = fold_id_col or comp_id

    source_path = locate_source(
        args.source_path,
        args.data_dir,
    )

    print("Loading source data...")
    source = pd.read_csv(source_path)

    if target not in source.columns:
        raise ValueError(
            f"Source data does not contain target {target!r}."
        )

    source_id = detect_source_id(
        source,
        target,
    )

    source_y, source_positive_label = encode_binary_target(
        source[target]
    )

    if str(source_positive_label) != str(positive_label):
        raise ValueError(
            "Source and competition positive labels disagree."
        )

    features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    missing_source_features = [
        c for c in features
        if c not in source.columns
    ]

    if missing_source_features:
        raise ValueError(
            "Source data is missing model features: "
            f"{missing_source_features}"
        )

    categoricals = detect_categoricals(
        train,
        features,
    )

    numerics = [
        c for c in features
        if c not in categoricals
    ]

    X = prepare_competition_types(
        train,
        features,
        categoricals,
    )

    X_test = prepare_competition_types(
        test,
        features,
        categoricals,
    )

    X_source = align_source_types(
        source,
        train,
        features,
        categoricals,
    )

    baseline_oof = load_baseline_oof(
        args.baseline_oof,
        train,
        fold_ids,
        id_col,
    )

    baseline_auc = float(
        roc_auc_score(
            comp_y,
            baseline_oof,
        )
    )

    print(
        f"Target: {target!r} | positive label: {positive_label!r}"
    )
    print(f"Competition ID excluded: {id_col!r}")
    print(f"Source ID excluded: {source_id!r}")
    print(f"Competition training rows: {len(train):,}")
    print(f"Source rows added per fold: {len(source):,}")
    print(f"Model features: {len(features)}")
    print(f"  numeric: {len(numerics)}")
    print(f"  native categorical/binary: {len(categoricals)}")
    print()
    print(
        f"Raw CatBoost baseline OOF AUC: "
        f"{baseline_auc:.8f}"
    )
    print()
    print(
        "Running source-augmented frozen 5-fold CatBoost CV..."
    )

    test_pool = Pool(
        X_test,
        cat_features=categoricals,
        feature_names=features,
    )

    source_pool_data = X_source

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    metric_rows = []
    importance_rows = []

    total_start = time.perf_counter()

    for fold in range(5):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        # IMPORTANT:
        # Source rows are appended ONLY to the training side.
        # The validation fold stays 100% competition data.
        X_fold_train = pd.concat(
            [
                X.iloc[train_idx],
                source_pool_data,
            ],
            axis=0,
            ignore_index=True,
        )

        y_fold_train = np.concatenate(
            [
                comp_y[train_idx],
                source_y,
            ]
        )

        train_pool = Pool(
            X_fold_train,
            label=y_fold_train,
            cat_features=categoricals,
            feature_names=features,
        )

        valid_pool = Pool(
            X.iloc[valid_idx],
            label=comp_y[valid_idx],
            cat_features=categoricals,
            feature_names=features,
        )

        model = build_model()

        fit_start = time.perf_counter()

        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=200,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        infer_start = time.perf_counter()

        valid_pred = model.predict_proba(
            valid_pool
        )[:, 1]

        test_pred = model.predict_proba(
            test_pool
        )[:, 1]

        inference_seconds = (
            time.perf_counter()
            - infer_start
        )

        oof[valid_idx] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        fold_auc = float(
            roc_auc_score(
                comp_y[valid_idx],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                comp_y[valid_idx],
                baseline_oof[valid_idx],
            )
        )

        delta = (
            fold_auc
            - baseline_fold_auc
        )

        metric_rows.append(
            {
                "fold": fold,
                "competition_train_rows": len(train_idx),
                "source_rows_added": len(source),
                "total_train_rows": len(y_fold_train),
                "valid_rows": len(valid_idx),
                "baseline_auc": baseline_fold_auc,
                "augmented_auc": fold_auc,
                "auc_delta": delta,
                "best_iteration_zero_based": int(
                    model.get_best_iteration()
                ),
                "tree_count": int(
                    model.tree_count_
                ),
                "fit_seconds": float(fit_seconds),
                "inference_seconds_valid_plus_test": float(
                    inference_seconds
                ),
            }
        )

        importances = model.get_feature_importance(
            type="PredictionValuesChange"
        )

        for feature, importance in zip(
            features,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {fold}: "
            f"baseline={baseline_fold_auc:.6f} | "
            f"augmented={fold_auc:.6f} | "
            f"delta={delta:+.6f} | "
            f"best_iter={model.get_best_iteration()} | "
            f"fit={fit_seconds:.1f}s"
        )

        del (
            model,
            train_pool,
            valid_pool,
            X_fold_train,
            y_fold_train,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "OOF predictions contain missing values."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    augmented_oof_auc = float(
        roc_auc_score(
            comp_y,
            oof,
        )
    )

    auc_delta = (
        augmented_oof_auc
        - baseline_auc
    )

    mean_fold_auc = float(
        fold_metrics["augmented_auc"].mean()
    )

    std_fold_auc = float(
        fold_metrics["augmented_auc"].std(
            ddof=1
        )
    )

    improved_folds = int(
        (fold_metrics["auc_delta"] > 0).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    rank_corr = safe_rank_correlation(
        oof,
        baseline_oof,
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    importance_by_fold = pd.DataFrame(
        importance_rows
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),
            std_importance=(
                "importance",
                "std",
            ),
        )
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------
    # Save artifacts
    # -------------------------
    fold_metrics.to_csv(
        args.output_dir
        / "fold_metrics.csv",
        index=False,
    )

    importance_summary.to_csv(
        args.output_dir
        / "feature_importance.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        args.output_dir
        / "feature_importance_by_fold.csv",
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": comp_y,
            "baseline_oof_prediction": baseline_oof.astype(
                np.float32
            ),
            "augmented_oof_prediction": oof.astype(
                np.float32
            ),
        }
    )

    if id_col is not None:
        oof_output.insert(
            1,
            id_col,
            train[id_col].to_numpy(),
        )

    oof_output.to_csv(
        args.output_dir
        / "oof_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame(
        {
            "prediction": test_prediction.astype(
                np.float32
            )
        }
    )

    if id_col is not None:
        test_output.insert(
            0,
            id_col,
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    # -------------------------
    # Summary
    # -------------------------
    decision = (
        "KEEP"
        if auc_delta > 0
        and improved_folds >= 3
        else "REJECT_FOR_NOW"
    )

    lines = [
        "EXPERIMENT: CATBOOST_SOURCE_AUGMENTATION",
        "=" * 76,
        "",
        "QUESTION",
        "Does appending all 10,000 original labeled rows to each CatBoost",
        "training fold improve AUC on untouched competition validation rows?",
        "",
        "EXPERIMENTAL CONTROL",
        "Same frozen competition folds.",
        "Same 13 model features.",
        "Same feature-type policy.",
        "Same CatBoost hyperparameters.",
        "Only source-row augmentation changed.",
        "",
        "LEAKAGE SAFETY",
        "Source rows are used only for fold training.",
        "Every validation fold contains only competition rows.",
        "",
        "DATA",
        f"Competition rows: {len(train)}",
        f"Source rows: {len(source)}",
        f"Source positive rate: {source_y.mean():.6f}",
        f"Competition positive rate: {comp_y.mean():.6f}",
        "",
        "VALIDATION",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "RESULTS",
        f"Raw CatBoost OOF AUC: {baseline_auc:.8f}",
        f"Source-augmented OOF AUC: {augmented_oof_auc:.8f}",
        f"OOF AUC delta: {auc_delta:+.8f}",
        f"Mean augmented fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {std_fold_auc:.8f}",
        f"Folds improved: {improved_folds}/5",
        f"OOF probability correlation vs baseline: {probability_corr:.6f}",
        f"OOF rank correlation vs baseline: {rank_corr:.6f}",
        f"Total runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD DELTAS",
    ]

    for row in fold_metrics.itertuples():
        lines.append(
            f"Fold {row.fold}: "
            f"{row.baseline_auc:.8f} -> "
            f"{row.augmented_auc:.8f} "
            f"({row.auc_delta:+.8f})"
        )

    lines.extend(
        [
            "",
            "NOTE",
            "KEEP/REJECT is based only on this controlled CV experiment.",
            "We will inspect the size and consistency of the delta before deciding",
            "whether to keep full-weight source augmentation or test downweighting.",
        ]
    )

    summary = "\n".join(lines)

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CATBOOST SOURCE AUGMENTATION COMPLETE"
    )
    print("=" * 78)
    print(
        f"Baseline OOF  : {baseline_auc:.8f}"
    )
    print(
        f"Augmented OOF : {augmented_oof_auc:.8f}"
    )
    print(
        f"Delta         : {auc_delta:+.8f}"
    )
    print(
        f"Folds improved: {improved_folds}/5"
    )
    print(
        f"Decision      : {decision}"
    )
    print(
        f"Total runtime : {total_seconds:.2f}s"
    )
    print(
        f"Artifacts     : "
        f"{args.output_dir.resolve()}"
    )
    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. feature_importance.csv")


if __name__ == "__main__":
    main()
