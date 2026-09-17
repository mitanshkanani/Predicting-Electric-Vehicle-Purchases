"""
catboost_source_augmentation_gpu.py

Phase 3D-GPU — Source-data augmentation experiment.

EXPERIMENT QUESTION
-------------------
Does adding the 10,000 labeled rows from the original
"EV Adoption Behavior and Range Anxiety" dataset improve CatBoost ROC-AUC
on untouched competition validation rows?

CONTROLLED EXPERIMENT
---------------------
Compared with baseline_catboost_gpu.py:

SAME:
- frozen competition folds
- 13 raw modeling features
- feature-type policy
- CatBoost hyperparameters
- GPU training
- early stopping
- validation metric

ONLY CHANGE:
- add every source/original row to the TRAINING side of each fold

Validation remains 100% competition data.

This is important because it means any OOF improvement is measured on
competition rows the model did not train on.

Run:
    python catboost_source_augmentation_gpu.py

Requirements:
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

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_source_augmentation_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test source-data augmentation using the frozen GPU CatBoost baseline."
        )
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
            "Optional explicit path to "
            "EV_Adoption_and_Range_Anxiety_Dataset.csv. "
            "If omitted, local paths are checked and kagglehub is used."
        ),
    )

    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
        help="Frozen fold assignment created by validation_setup.py.",
    )

    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=DEFAULT_BASELINE_OOF,
        help="OOF predictions from baseline_catboost_gpu.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for experiment artifacts.",
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
        c
        for c in test.columns
        if c in train.columns
        and c != target
    ]

    for c in common:
        if c.lower() in {
            "id",
            "row_id",
            "rowid",
            "index",
        }:
            return c

    return None


def detect_source_id(
    source: pd.DataFrame,
    target: str,
) -> str | None:
    candidates = {
        "buyer_id",
        "customer_id",
        "person_id",
        "id",
        "row_id",
        "rowid",
    }

    for c in source.columns:
        if (
            c != target
            and c.lower() in candidates
        ):
            return c

    return None


def encode_binary_target(
    y: pd.Series,
) -> tuple[np.ndarray, object]:
    values = list(
        pd.unique(
            y.dropna()
        )
    )

    if len(values) != 2:
        raise ValueError(
            f"Expected binary target; found: {values}"
        )

    preferred_positive = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive = None

    for value in values:
        if (
            str(value)
            .strip()
            .lower()
            in preferred_positive
        ):
            positive = value
            break

    if positive is None:
        positive = (
            y.value_counts()
            .idxmin()
        )

    encoded = (
        y == positive
    ).astype(
        np.int8
    ).to_numpy()

    return encoded, positive


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def locate_source(
    explicit_path: Path | None,
    data_dir: Path,
) -> Path:
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(
            explicit_path
        )

    candidates.extend(
        [
            data_dir
            / "original"
            / SOURCE_FILENAME,

            data_dir
            / SOURCE_FILENAME,

            Path(
                SOURCE_FILENAME
            ),
        ]
    )

    for path in candidates:
        if path.exists():
            print(
                "Using local source dataset: "
                f"{path.resolve()}"
            )
            return path

    print(
        "Source dataset not found in project folder; "
        "checking kagglehub cache/downloading..."
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
        kagglehub.dataset_download(
            SOURCE_HANDLE
        )
    )

    matches = list(
        root.rglob(
            SOURCE_FILENAME
        )
    )

    if not matches:
        csvs = list(
            root.rglob("*.csv")
        )

        if len(csvs) == 1:
            matches = csvs

    if not matches:
        raise FileNotFoundError(
            "Could not locate source CSV "
            f"under {root}"
        )

    source_path = matches[0]

    print(
        "Using source dataset: "
        f"{source_path.resolve()}"
    )

    return source_path


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {
        "row_index",
        "fold",
    }

    missing = (
        required
        - set(
            folds.columns
        )
    )

    if missing:
        raise ValueError(
            "Fold file missing required columns: "
            f"{sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen fold file row count "
            "does not match train.csv."
        )

    expected_index = np.arange(
        len(train),
        dtype=np.int64,
    )

    if not np.array_equal(
        folds[
            "row_index"
        ].to_numpy(),
        expected_index,
    ):
        raise ValueError(
            "Frozen fold row order does not "
            "match train.csv."
        )

    unique_folds = sorted(
        folds[
            "fold"
        ].unique().tolist()
    )

    if unique_folds != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise ValueError(
            "Expected frozen folds "
            f"[0,1,2,3,4], found {unique_folds}"
        )

    extra_columns = [
        c
        for c in folds.columns
        if c
        not in {
            "row_index",
            "fold",
        }
    ]

    if len(
        extra_columns
    ) > 1:
        raise ValueError(
            "Unexpected extra fold columns: "
            f"{extra_columns}"
        )

    id_col = (
        extra_columns[0]
        if extra_columns
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} "
                "is missing from train.csv."
            )

        if not np.array_equal(
            folds[
                id_col
            ].to_numpy(),
            train[
                id_col
            ].to_numpy(),
        ):
            raise ValueError(
                "Fold IDs do not align "
                "with train.csv."
            )

    return (
        folds[
            "fold"
        ].to_numpy(
            dtype=np.int16
        ),
        id_col,
    )


def detect_categorical_features(
    train: pd.DataFrame,
    features: list[str],
) -> list[str]:
    categorical = []

    for c in features:
        dtype = train[
            c
        ].dtype

        if (
            pd.api.types.is_object_dtype(
                dtype
            )
            or
            pd.api.types.is_string_dtype(
                dtype
            )
            or
            pd.api.types.is_bool_dtype(
                dtype
            )
            or
            isinstance(
                dtype,
                pd.CategoricalDtype,
            )
        ):
            categorical.append(
                c
            )

    return categorical


def prepare_competition_frame(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    out = frame[
        features
    ].copy()

    for c in categorical_features:
        out[c] = (
            out[c]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    return out


def prepare_source_frame(
    source: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    out = source[
        features
    ].copy()

    for c in features:
        if (
            c
            in categorical_features
        ):
            out[c] = (
                out[c]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )
        else:
            # CatBoost can natively handle NaN
            # in numeric features.
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce",
            )

    return out


def load_gpu_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            "\nGPU CatBoost baseline OOF "
            "file was not found:\n"
            f"{path.resolve()}\n\n"
            "Run baseline_catboost_gpu.py first."
        )

    df = pd.read_csv(
        path
    )

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(
        df.columns
    ):
        raise ValueError(
            "GPU baseline OOF file "
            "has unexpected columns."
        )

    if len(df) != len(train):
        raise ValueError(
            "GPU baseline OOF row count "
            "does not match train.csv."
        )

    if not np.array_equal(
        df[
            "row_index"
        ].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "GPU baseline OOF row order "
            "is misaligned."
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "GPU baseline OOF was generated "
            "using different folds."
        )

    if (
        id_col is not None
        and
        id_col in df.columns
    ):
        if not np.array_equal(
            df[
                id_col
            ].to_numpy(),
            train[
                id_col
            ].to_numpy(),
        ):
            raise ValueError(
                "GPU baseline OOF IDs "
                "are misaligned."
            )

    return df[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )


def safe_rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a_rank = (
        pd.Series(a)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    b_rank = (
        pd.Series(b)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            a_rank,
            b_rank,
        )[0, 1]
    )


def build_model() -> CatBoostClassifier:
    """
    EXACT same model configuration as baseline_catboost_gpu.py.
    """

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=2000,
        learning_rate=0.05,
        depth=6,

        random_seed=SEED,

        task_type="GPU",
        devices="0",

        allow_writing_files=False,
    )


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

    for path in [
        train_path,
        test_path,
        args.folds_path,
        args.baseline_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required file: "
                f"{path.resolve()}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading competition data, "
        "frozen folds, and GPU baseline..."
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    folds_df = pd.read_csv(
        args.folds_path
    )

    target = detect_target(
        train,
        test,
    )

    competition_y, positive_label = (
        encode_binary_target(
            train[
                target
            ]
        )
    )

    fold_ids, fold_id_col = (
        validate_folds(
            folds_df,
            train,
        )
    )

    competition_id = (
        detect_competition_id(
            train,
            test,
            target,
        )
    )

    if (
        competition_id is not None
        and
        fold_id_col is not None
        and
        competition_id
        != fold_id_col
    ):
        raise ValueError(
            "Detected ID differs from "
            "fold-file ID."
        )

    id_col = (
        fold_id_col
        or competition_id
    )

    baseline_oof = (
        load_gpu_baseline_oof(
            args.baseline_oof,
            train,
            fold_ids,
            id_col,
        )
    )

    baseline_auc = float(
        roc_auc_score(
            competition_y,
            baseline_oof,
        )
    )

    source_path = locate_source(
        args.source_path,
        args.data_dir,
    )

    print(
        "Loading original/source dataset..."
    )

    source = pd.read_csv(
        source_path
    )

    if target not in source.columns:
        raise ValueError(
            "Source dataset does not contain "
            f"target {target!r}."
        )

    source_id = detect_source_id(
        source,
        target,
    )

    source_y, source_positive_label = (
        encode_binary_target(
            source[
                target
            ]
        )
    )

    if (
        str(
            source_positive_label
        )
        !=
        str(
            positive_label
        )
    ):
        raise ValueError(
            "Competition and source "
            "positive labels disagree."
        )

    features = [
        c
        for c in test.columns
        if (
            c in train.columns
            and
            c != id_col
        )
    ]

    missing_features = [
        c
        for c in features
        if c not in source.columns
    ]

    if missing_features:
        raise ValueError(
            "Source dataset is missing "
            "competition model features: "
            f"{missing_features}"
        )

    categorical_features = (
        detect_categorical_features(
            train,
            features,
        )
    )

    numeric_features = [
        c
        for c in features
        if c
        not in categorical_features
    ]

    X = prepare_competition_frame(
        train,
        features,
        categorical_features,
    )

    X_test = prepare_competition_frame(
        test,
        features,
        categorical_features,
    )

    X_source = prepare_source_frame(
        source,
        features,
        categorical_features,
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Competition ID excluded: {id_col!r}"
    )
    print(
        f"Source ID excluded: {source_id!r}"
    )
    print(
        f"Competition rows: {len(train):,}"
    )
    print(
        f"Source rows added to each training fold: "
        f"{len(source):,}"
    )
    print(
        f"Competition positive rate: "
        f"{competition_y.mean():.6f}"
    )
    print(
        f"Source positive rate: "
        f"{source_y.mean():.6f}"
    )
    print(
        f"GPU baseline OOF AUC: "
        f"{baseline_auc:.8f}"
    )
    print()
    print(
        f"Features: {len(features)} "
        f"({len(numeric_features)} numeric, "
        f"{len(categorical_features)} categorical)"
    )
    print()
    print(
        "Running frozen 5-fold GPU "
        "source-augmentation CV..."
    )
    print()

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=features,
    )

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []

    metric_rows = []
    importance_rows = []

    total_start = (
        time.perf_counter()
    )

    for fold in range(5):
        competition_train_idx = (
            np.flatnonzero(
                fold_ids != fold
            )
        )

        valid_idx = (
            np.flatnonzero(
                fold_ids == fold
            )
        )

        # =====================================================
        # IMPORTANT LEAKAGE-SAFETY RULE
        #
        # Source rows enter ONLY training.
        # Validation remains 100% competition rows.
        # =====================================================

        X_augmented = pd.concat(
            [
                X.iloc[
                    competition_train_idx
                ],
                X_source,
            ],
            axis=0,
            ignore_index=True,
        )

        y_augmented = np.concatenate(
            [
                competition_y[
                    competition_train_idx
                ],
                source_y,
            ]
        )

        train_pool = Pool(
            X_augmented,
            label=y_augmented,
            cat_features=categorical_features,
            feature_names=features,
        )

        valid_pool = Pool(
            X.iloc[
                valid_idx
            ],
            label=competition_y[
                valid_idx
            ],
            cat_features=categorical_features,
            feature_names=features,
        )

        model = build_model()

        fit_start = (
            time.perf_counter()
        )

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

        infer_start = (
            time.perf_counter()
        )

        valid_pred = (
            model.predict_proba(
                valid_pool
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                test_pool
            )[:, 1]
        )

        inference_seconds = (
            time.perf_counter()
            - infer_start
        )

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(
                np.float32
            )
        )

        augmented_fold_auc = float(
            roc_auc_score(
                competition_y[
                    valid_idx
                ],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                competition_y[
                    valid_idx
                ],
                baseline_oof[
                    valid_idx
                ],
            )
        )

        fold_delta = (
            augmented_fold_auc
            - baseline_fold_auc
        )

        best_iteration = int(
            model.get_best_iteration()
        )

        tree_count = int(
            model.tree_count_
        )

        metric_rows.append(
            {
                "fold": fold,

                "competition_train_rows": len(
                    competition_train_idx
                ),

                "source_rows_added": len(
                    source
                ),

                "total_train_rows": len(
                    y_augmented
                ),

                "valid_rows": len(
                    valid_idx
                ),

                "baseline_gpu_auc": (
                    baseline_fold_auc
                ),

                "augmented_gpu_auc": (
                    augmented_fold_auc
                ),

                "auc_delta": (
                    fold_delta
                ),

                "best_iteration_zero_based": (
                    best_iteration
                ),

                "tree_count": (
                    tree_count
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "inference_seconds_valid_plus_test": (
                    float(
                        inference_seconds
                    )
                ),
            }
        )

        importances = (
            model.get_feature_importance(
                type="PredictionValuesChange"
            )
        )

        for (
            feature,
            importance,
        ) in zip(
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

        print()
        print(
            f"Fold {fold}: "
            f"baseline={baseline_fold_auc:.6f} | "
            f"augmented={augmented_fold_auc:.6f} | "
            f"delta={fold_delta:+.6f} | "
            f"best_iter={best_iteration} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            train_pool,
            valid_pool,
            X_augmented,
            y_augmented,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "OOF predictions contain "
            "missing values."
        )

    fold_metrics = (
        pd.DataFrame(
            metric_rows
        )
    )

    augmented_oof_auc = float(
        roc_auc_score(
            competition_y,
            oof,
        )
    )

    oof_delta = (
        augmented_oof_auc
        - baseline_auc
    )

    mean_fold_auc = float(
        fold_metrics[
            "augmented_gpu_auc"
        ].mean()
    )

    std_fold_auc = float(
        fold_metrics[
            "augmented_gpu_auc"
        ].std(
            ddof=1
        )
    )

    improved_folds = int(
        (
            fold_metrics[
                "auc_delta"
            ] > 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    rank_corr = (
        safe_rank_correlation(
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
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================

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

            "fold": (
                fold_ids
            ),

            "target_encoded": (
                competition_y
            ),

            "baseline_gpu_oof_prediction": (
                baseline_oof.astype(
                    np.float32
                )
            ),

            "augmented_gpu_oof_prediction": (
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
        args.output_dir
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
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    # =====================================================
    # DECISION
    # =====================================================

    if (
        oof_delta > 0
        and
        improved_folds >= 3
    ):
        decision = "KEEP"
    else:
        decision = "REJECT_FOR_NOW"

    summary_lines = [
        "EXPERIMENT: CATBOOST_SOURCE_AUGMENTATION_GPU",
        "=" * 76,
        "",
        "QUESTION",
        "Does appending all 10,000 original labeled rows to every CatBoost",
        "training fold improve ROC-AUC on untouched competition validation rows?",
        "",
        "EXPERIMENTAL CONTROL",
        "Same frozen folds.",
        "Same 13 model features.",
        "Same feature-type policy.",
        "Same CatBoost hyperparameters.",
        "Same GPU training mode.",
        "Only source-data augmentation changed.",
        "",
        "LEAKAGE SAFETY",
        "Source rows are appended only to fold training data.",
        "Every validation fold contains competition rows only.",
        "",
        "DATA",
        f"Competition rows: {len(train)}",
        f"Source rows added per fold: {len(source)}",
        f"Competition positive rate: {competition_y.mean():.6f}",
        f"Source positive rate: {source_y.mean():.6f}",
        "",
        "VALIDATION",
        "Frozen competition 5-fold split.",
        f"Fold-file SHA256: {sha256_file(args.folds_path)}",
        "",
        "MODEL",
        "CatBoostClassifier",
        "task_type='GPU'",
        "devices='0'",
        "iterations=2000",
        "learning_rate=0.05",
        "depth=6",
        "early_stopping_rounds=150",
        "",
        "RESULTS",
        f"GPU baseline OOF AUC: {baseline_auc:.8f}",
        f"Source-augmented GPU OOF AUC: {augmented_oof_auc:.8f}",
        f"OOF AUC delta: {oof_delta:+.8f}",
        f"Mean augmented fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {std_fold_auc:.8f}",
        f"Folds improved: {improved_folds}/5",
        f"OOF probability correlation vs baseline: {probability_corr:.6f}",
        f"OOF rank correlation vs baseline: {rank_corr:.6f}",
        f"Total CV + test inference time: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD DELTAS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.baseline_gpu_auc:.8f} -> "
            f"{row.augmented_gpu_auc:.8f} "
            f"({row.auc_delta:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "TOP MEAN FEATURE IMPORTANCES",
        ]
    )

    for row in (
        importance_summary
        .head(10)
        .itertuples()
    ):
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "A positive score alone is not enough; we also care whether the gain",
            "is consistent across folds.",
            "",
            "If full-weight source augmentation helps, we keep it provisionally.",
            "If it hurts, we reject it or later test a lower source sample weight.",
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CATBOOST GPU SOURCE AUGMENTATION COMPLETE"
    )
    print("=" * 78)
    print(
        f"GPU baseline OOF : "
        f"{baseline_auc:.8f}"
    )
    print(
        f"Augmented OOF    : "
        f"{augmented_oof_auc:.8f}"
    )
    print(
        f"Delta            : "
        f"{oof_delta:+.8f}"
    )
    print(
        f"Folds improved   : "
        f"{improved_folds}/5"
    )
    print(
        f"Decision         : "
        f"{decision}"
    )
    print(
        f"Total runtime    : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts        : "
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
