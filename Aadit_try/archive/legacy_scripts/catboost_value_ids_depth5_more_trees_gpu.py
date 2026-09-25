"""
catboost_value_ids_depth5_more_trees_gpu.py

Phase 5I — Does depth=5 need a larger tree budget?

CURRENT RESULTS
---------------
Depth=6, seed=42:
    OOF AUC = 0.94536428

Depth=5, seed=42, max 2000 trees:
    OOF AUC = 0.94539185

3-seed rank champion (depth=6):
    OOF AUC = 0.94543956

WHY THIS EXPERIMENT
-------------------
In the depth=5 run, several folds selected best iterations very close to the
2000-tree ceiling. That means the depth=5 candidate may have been limited by
the iteration budget rather than by generalization.

CONTROLLED EXPERIMENT
---------------------
SAME:
- exact frozen 5 folds
- 13 raw features
- Annual_Income_USD_id
- Daily_Commute_km_id
- depth=5
- learning_rate=0.05
- random_seed=42
- GPU
- early_stopping_rounds=150

ONLY CHANGE:
- iterations: 2000 -> 3500

The script reuses the saved depth=5 2000-tree OOF/test predictions for the
comparison and retrains only the new 3500-tree candidate.

Run:
    python catboost_value_ids_depth5_more_trees_gpu.py
"""

from __future__ import annotations

import argparse
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
DEPTH = 5
BASE_ITERATIONS = 2000
NEW_ITERATIONS = 3500

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_DEPTH_SWEEP_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_depth_sweep_gpu"
)

DEFAULT_MULTI_SEED_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_depth5_more_trees_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a larger iteration budget for depth-5 value-ID CatBoost."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--folds-path",
        type=Path,
        default=DEFAULT_FOLDS_PATH,
    )

    parser.add_argument(
        "--depth-sweep-dir",
        type=Path,
        default=DEFAULT_DEPTH_SWEEP_DIR,
    )

    parser.add_argument(
        "--multiseed-oof",
        type=Path,
        default=DEFAULT_MULTI_SEED_OOF,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def detect_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    train_only = [
        c for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            f"Expected one train-only target column, found {train_only}"
        )

    return train_only[0]


def encode_binary_target(
    y: pd.Series,
) -> tuple[np.ndarray, object]:
    values = list(pd.unique(y.dropna()))

    if len(values) != 2:
        raise ValueError(
            f"Expected binary target, found {values}"
        )

    preferred = {
        "yes",
        "true",
        "1",
        "positive",
        "buy",
        "will_buy",
    }

    positive = None

    for value in values:
        if str(value).strip().lower() in preferred:
            positive = value
            break

    if positive is None:
        positive = y.value_counts().idxmin()

    encoded = (
        y == positive
    ).astype(np.int8).to_numpy()

    return encoded, positive


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
            "Frozen fold row count does not match train.csv."
        )

    expected = np.arange(
        len(train),
        dtype=np.int64,
    )

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        expected,
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    if sorted(
        folds["fold"].unique().tolist()
    ) != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected frozen folds [0,1,2,3,4]."
        )

    extras = [
        c
        for c in folds.columns
        if c not in {"row_index", "fold"}
    ]

    if len(extras) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extras}"
        )

    id_col = extras[0] if extras else None

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} missing from train.csv."
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


def detect_raw_categoricals(
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
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            out.append(c)

    return out


def stable_value_id(
    series: pd.Series,
    feature: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{feature!r} contains missing/non-numeric values."
        )

    if feature == "Annual_Income_USD":
        arr = np.rint(
            numeric.to_numpy(dtype=np.float64)
        ).astype(np.int64)

        return pd.Series(
            arr.astype(str),
            index=series.index,
            name=f"{feature}_id",
        )

    if feature == "Daily_Commute_km":
        arr = numeric.to_numpy(dtype=np.float64)

        return pd.Series(
            [f"{x:.1f}" for x in arr],
            index=series.index,
            name=f"{feature}_id",
        )

    return numeric.map(
        lambda x: format(float(x), ".12g")
    ).rename(
        f"{feature}_id"
    )


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_features: list[str],
    raw_categoricals: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    x_train = train[raw_features].copy()
    x_test = test[raw_features].copy()

    for c in raw_categoricals:
        x_train[c] = (
            x_train[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        x_test[c] = (
            x_test[c]
            .fillna("__MISSING__")
            .astype(str)
        )

    value_id_columns = []

    for feature in VALUE_ID_FEATURES:
        id_col = f"{feature}_id"

        x_train[id_col] = stable_value_id(
            train[feature],
            feature,
        )

        x_test[id_col] = stable_value_id(
            test[feature],
            feature,
        )

        value_id_columns.append(id_col)

    categorical_features = (
        raw_categoricals
        + value_id_columns
    )

    return (
        x_train,
        x_test,
        categorical_features,
    )


def load_depth5_baseline(
    depth_sweep_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame | None,
]:
    oof_path = (
        depth_sweep_dir
        / "oof_predictions_by_depth.csv"
    )

    test_path = (
        depth_sweep_dir
        / "test_predictions_by_depth.csv"
    )

    fold_metrics_path = (
        depth_sweep_dir
        / "fold_metrics.csv"
    )

    for path in [
        oof_path,
        test_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required depth-sweep artifact missing:\n{path.resolve()}"
            )

    oof_df = pd.read_csv(
        oof_path
    )

    test_df = pd.read_csv(
        test_path
    )

    required_oof = {
        "row_index",
        "fold",
        "oof_depth_5",
    }

    if not required_oof.issubset(
        oof_df.columns
    ):
        raise ValueError(
            "Depth-sweep OOF file is missing required depth-5 columns."
        )

    if len(oof_df) != len(train):
        raise ValueError(
            "Depth-5 OOF row count mismatch."
        )

    if not np.array_equal(
        oof_df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            "Depth-5 OOF row order mismatch."
        )

    if not np.array_equal(
        oof_df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Depth-5 OOF folds mismatch."
        )

    if (
        id_col is not None
        and id_col in oof_df.columns
        and not np.array_equal(
            oof_df[id_col].to_numpy(),
            train[id_col].to_numpy(),
        )
    ):
        raise ValueError(
            "Depth-5 OOF IDs are misaligned."
        )

    if "prediction_depth_5" not in test_df.columns:
        raise ValueError(
            "Depth-sweep test file lacks prediction_depth_5."
        )

    if len(test_df) != len(test):
        raise ValueError(
            "Depth-5 test prediction row count mismatch."
        )

    if (
        id_col is not None
        and id_col in test_df.columns
        and not np.array_equal(
            test_df[id_col].to_numpy(),
            test[id_col].to_numpy(),
        )
    ):
        raise ValueError(
            "Depth-5 test IDs are misaligned."
        )

    baseline_fold_metrics = None

    if fold_metrics_path.exists():
        fm = pd.read_csv(
            fold_metrics_path
        )

        if {
            "depth",
            "fold",
            "best_iteration_zero_based",
        }.issubset(
            fm.columns
        ):
            baseline_fold_metrics = (
                fm[
                    fm["depth"] == DEPTH
                ]
                .copy()
                .sort_values("fold")
                .reset_index(drop=True)
            )

    return (
        oof_df["oof_depth_5"].to_numpy(
            dtype=np.float64
        ),
        test_df["prediction_depth_5"].to_numpy(
            dtype=np.float64
        ),
        baseline_fold_metrics,
    )


def load_multiseed_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Multi-seed champion OOF not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Multi-seed OOF has unexpected columns."
        )

    if len(df) != len(train):
        raise ValueError(
            "Multi-seed OOF row count mismatch."
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            "Multi-seed OOF row order mismatch."
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            "Multi-seed OOF fold assignment mismatch."
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
            "Multi-seed OOF IDs are misaligned."
        )

    return df["oof_prediction"].to_numpy(
        dtype=np.float64
    )


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=NEW_ITERATIONS,
        learning_rate=0.05,
        depth=DEPTH,

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
        args.multiseed_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.resolve()}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading data, frozen folds, and saved depth-5 baseline..."
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = detect_target(
        train,
        test,
    )

    y, positive_label = encode_binary_target(
        train[target]
    )

    fold_ids, id_col = validate_folds(
        folds_df,
        train,
    )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = detect_raw_categoricals(
        train,
        raw_features,
    )

    (
        X,
        X_test,
        categorical_features,
    ) = prepare_frames(
        train,
        test,
        raw_features,
        raw_categoricals,
    )

    (
        depth5_2000_oof,
        depth5_2000_test,
        baseline_fold_metrics,
    ) = load_depth5_baseline(
        args.depth_sweep_dir,
        train,
        test,
        fold_ids,
        id_col,
    )

    multiseed_oof = load_multiseed_oof(
        args.multiseed_oof,
        train,
        fold_ids,
        id_col,
    )

    baseline_auc = float(
        roc_auc_score(
            y,
            depth5_2000_oof,
        )
    )

    multiseed_auc = float(
        roc_auc_score(
            y,
            multiseed_oof,
        )
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Depth-5 / 2000-tree OOF: "
        f"{baseline_auc:.8f}"
    )

    print(
        f"Current 3-seed champion OOF: "
        f"{multiseed_auc:.8f}"
    )

    print()
    print(
        f"Testing depth={DEPTH} with "
        f"iterations={NEW_ITERATIONS}"
    )

    print(
        "Everything else is unchanged."
    )

    if (
        baseline_fold_metrics is not None
        and
        len(baseline_fold_metrics) == 5
    ):
        print()
        print(
            "Previous depth-5 best iterations:"
        )

        for row in baseline_fold_metrics.itertuples():
            print(
                f"  Fold {row.fold}: "
                f"{int(row.best_iteration_zero_based)}"
            )

    print()
    print(
        "Running frozen 5-fold CV..."
    )
    print()

    test_pool = Pool(
        X_test,
        cat_features=categorical_features,
        feature_names=list(X.columns),
    )

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    fold_rows = []

    total_start = time.perf_counter()

    for fold in range(5):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        train_pool = Pool(
            X.iloc[train_idx],
            label=y[train_idx],
            cat_features=categorical_features,
            feature_names=list(X.columns),
        )

        valid_pool = Pool(
            X.iloc[valid_idx],
            label=y[valid_idx],
            cat_features=categorical_features,
            feature_names=list(X.columns),
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

        oof[valid_idx] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                depth5_2000_oof[valid_idx],
            )
        )

        multiseed_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                multiseed_oof[valid_idx],
            )
        )

        best_iteration = int(
            model.get_best_iteration()
        )

        fold_rows.append(
            {
                "fold": fold,

                "depth5_2000_auc": (
                    baseline_fold_auc
                ),

                "depth5_3500_auc": (
                    candidate_fold_auc
                ),

                "delta_vs_depth5_2000": (
                    candidate_fold_auc
                    - baseline_fold_auc
                ),

                "multiseed_champion_auc": (
                    multiseed_fold_auc
                ),

                "delta_vs_multiseed_champion": (
                    candidate_fold_auc
                    - multiseed_fold_auc
                ),

                "best_iteration_zero_based": (
                    best_iteration
                ),

                "tree_count": int(
                    model.tree_count_
                ),

                "fit_seconds": float(
                    fit_seconds
                ),

                "stopped_before_budget": (
                    best_iteration
                    < NEW_ITERATIONS - 151
                ),
            }
        )

        print()
        print(
            f"Fold {fold}: "
            f"2000={baseline_fold_auc:.6f} | "
            f"3500={candidate_fold_auc:.6f} | "
            f"delta={candidate_fold_auc - baseline_fold_auc:+.6f} | "
            f"best_iter={best_iteration} | "
            f"fit={fit_seconds:.1f}s"
        )
        print()

        del (
            model,
            train_pool,
            valid_pool,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Candidate OOF contains NaN predictions."
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_2000 = (
        candidate_auc
        - baseline_auc
    )

    delta_vs_multiseed = (
        candidate_auc
        - multiseed_auc
    )

    folds_improved = int(
        (
            fold_metrics[
                "delta_vs_depth5_2000"
            ] > 0
        ).sum()
    )

    folds_beating_multiseed = int(
        (
            fold_metrics[
                "delta_vs_multiseed_champion"
            ] > 0
        ).sum()
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    fold_metrics.to_csv(
        args.output_dir
        / "fold_metrics.csv",
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": oof.astype(
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
            test[id_col].to_numpy(),
        )

    test_output.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    decision = (
        "KEEP"
        if (
            delta_vs_2000 > 0
            and
            folds_improved >= 3
        )
        else
        "REJECT_FOR_NOW"
    )

    hit_new_ceiling = int(
        (
            fold_metrics[
                "best_iteration_zero_based"
            ] >= NEW_ITERATIONS - 25
        ).sum()
    )

    lines = [
        "EXPERIMENT: CATBOOST_VALUE_IDS_DEPTH5_MORE_TREES_GPU",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Was depth=5 artificially limited by the 2000-tree iteration budget?",
        "",
        "CONTROL",
        "Same frozen folds, same exact-value features, same seed=42,",
        "same depth=5, learning_rate=0.05, GPU, and early stopping.",
        f"Only iterations changed: {BASE_ITERATIONS} -> {NEW_ITERATIONS}.",
        "",
        "RESULTS",
        f"Depth-5 / 2000-tree OOF AUC: {baseline_auc:.8f}",
        f"Depth-5 / 3500-tree OOF AUC: {candidate_auc:.8f}",
        f"Delta vs 2000-tree depth 5: {delta_vs_2000:+.8f}",
        f"Folds improved: {folds_improved}/5",
        "",
        f"3-seed depth-6 rank champion OOF AUC: {multiseed_auc:.8f}",
        f"Delta vs current champion: {delta_vs_multiseed:+.8f}",
        f"Folds beating current champion: {folds_beating_multiseed}/5",
        "",
        f"Folds whose best iteration hit the new 3500-tree ceiling: "
        f"{hit_new_ceiling}/5",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        lines.append(
            f"Fold {row.fold}: "
            f"{row.depth5_2000_auc:.8f} -> "
            f"{row.depth5_3500_auc:.8f} "
            f"({row.delta_vs_depth5_2000:+.8f}), "
            f"best_iter={row.best_iteration_zero_based}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "If the extra tree budget improves depth=5 consistently, the",
            "3500-tree configuration becomes the depth-5 base for any later",
            "multi-seed experiment.",
            "",
            "If the gain disappears or remains negligible, we stop spending",
            "compute on the depth/iteration axis and move to a new idea.",
        ]
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "DEPTH-5 LARGER TREE-BUDGET EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Depth5 / 2000 : {baseline_auc:.8f}"
    )

    print(
        f"Depth5 / 3500 : {candidate_auc:.8f}"
    )

    print(
        f"Delta         : {delta_vs_2000:+.8f}"
    )

    print(
        f"Folds improved: {folds_improved}/5"
    )

    print(
        f"3-seed champ  : {multiseed_auc:.8f}"
    )

    print(
        f"Vs champion   : {delta_vs_multiseed:+.8f}"
    )

    print(
        f"Decision      : {decision}"
    )

    print(
        f"Runtime       : {total_seconds:.2f}s"
    )

    print(
        f"Artifacts     : {args.output_dir.resolve()}"
    )

    print("=" * 78)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")


if __name__ == "__main__":
    main()
