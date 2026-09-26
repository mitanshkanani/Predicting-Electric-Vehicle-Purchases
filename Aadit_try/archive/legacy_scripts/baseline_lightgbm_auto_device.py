"""
baseline_lightgbm_auto_device.py

Phase 5B — LightGBM baseline on the exact frozen 5-fold split.

EXPERIMENT QUESTION
-------------------
Can LightGBM capture useful structure that differs from our current best
depth-4 XGBoost model?

CURRENT REFERENCE
-----------------
Depth-4 XGBoost OOF AUC ≈ 0.94206547.

WHY LIGHTGBM
------------
LightGBM grows trees leaf-wise rather than level-wise like our current
XGBoost setup. Even if its standalone score is similar or slightly lower,
different OOF rankings can make it valuable later in an ensemble.

DEVICE BEHAVIOR
---------------
This script first tries LightGBM's Windows-compatible GPU backend:
    device_type="gpu"

If the installed LightGBM build does not contain GPU support, it automatically
falls back to:
    device_type="cpu"

That lets us test the model family without requiring a custom LightGBM build.

NO TUNING YET
-------------
This is intentionally a sensible baseline, not a hyperparameter search.

Run:
    python baseline_lightgbm_auto_device.py

Install if needed:
    python -m pip install -U lightgbm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "\nLightGBM is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U lightgbm\n"
        "Then rerun this script.\n"
    ) from exc


SEED = 42

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_oof_predictions.csv"
)

DEFAULT_CATBOOST_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_raw_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_raw"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LightGBM baseline on frozen folds."
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
        "--xgb-oof",
        type=Path,
        default=DEFAULT_XGB_OOF,
    )

    parser.add_argument(
        "--catboost-oof",
        type=Path,
        default=DEFAULT_CATBOOST_OOF,
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
        c
        for c in train.columns
        if c not in test.columns
    ]

    if len(train_only) != 1:
        raise ValueError(
            "Expected exactly one train-only target column; "
            f"found: {train_only}"
        )

    return train_only[0]


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

    positive = None

    for value in values:
        if (
            str(value)
            .strip()
            .lower()
            in preferred
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
            f"Fold file missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen fold row count differs from train.csv."
        )

    if not np.array_equal(
        folds[
            "row_index"
        ].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
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
            f"Expected folds [0,1,2,3,4], found {unique_folds}"
        )

    extras = [
        c
        for c in folds.columns
        if c
        not in {
            "row_index",
            "fold",
        }
    ]

    if len(extras) > 1:
        raise ValueError(
            f"Unexpected extra fold columns: {extras}"
        )

    id_col = (
        extras[0]
        if extras
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Fold ID {id_col!r} missing from train.csv."
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
                "Frozen fold IDs do not align with train.csv."
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


def prepare_lightgbm_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Give train/test exactly the same pandas categorical vocabularies.
    """
    x_train = train[
        features
    ].copy()

    x_test = test[
        features
    ].copy()

    for c in categorical_features:
        train_values = (
            x_train[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        test_values = (
            x_test[
                c
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

        categories = pd.Index(
            pd.concat(
                [
                    train_values,
                    test_values,
                ],
                axis=0,
                ignore_index=True,
            ).unique()
        )

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        x_train[
            c
        ] = train_values.astype(
            dtype
        )

        x_test[
            c
        ] = test_values.astype(
            dtype
        )

    return (
        x_train,
        x_test,
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    required: bool,
) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required OOF file not found:\n{path.resolve()}"
            )
        return None

    df = pd.read_csv(
        path
    )

    required_cols = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required_cols.issubset(
        df.columns
    ):
        raise ValueError(
            f"Unexpected OOF columns in {path}"
        )

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch in {path}"
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
            f"OOF row order mismatch in {path}"
        )

    if not np.array_equal(
        df[
            "fold"
        ].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF folds mismatch in {path}"
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
                f"OOF IDs mismatch in {path}"
            )

    return df[
        "oof_prediction"
    ].to_numpy(
        dtype=np.float64
    )


def rank_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ar = (
        pd.Series(a)
        .rank(
            pct=True,
            method="average",
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            pct=True,
            method="average",
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def detect_device() -> tuple[str, str]:
    """
    Try LightGBM's OpenCL GPU learner.

    A normal Windows pip build may be CPU-only. If GPU training is not
    available, return CPU without failing the experiment.
    """
    rng = np.random.default_rng(
        SEED
    )

    x = rng.normal(
        size=(
            2000,
            5,
        )
    )

    y = (
        x[
            :,
            0
        ]
        + 0.2
        * x[
            :,
            1
        ]
        > 0
    ).astype(
        np.int8
    )

    try:
        probe = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5,
            num_leaves=7,
            learning_rate=0.1,
            min_child_samples=10,
            max_bin=63,
            device_type="gpu",
            verbosity=-1,
            random_state=SEED,
        )

        probe.fit(
            x,
            y,
        )

        return (
            "gpu",
            "GPU probe succeeded.",
        )

    except Exception as exc:
        message = (
            str(exc)
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        if len(message) > 300:
            message = (
                message[
                    :300
                ]
                + "..."
            )

        return (
            "cpu",
            "GPU probe failed; using CPU. "
            f"Reason: {message}",
        )


def build_model(
    device_type: str,
) -> lgb.LGBMClassifier:
    """
    Sensible raw LightGBM baseline, not tuned.

    max_bin=63 is retained on CPU too so that only the execution device
    changes if GPU support is unavailable.
    """
    common = dict(
        objective="binary",
        metric="auc",

        n_estimators=5000,
        learning_rate=0.03,

        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,

        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,

        reg_lambda=2.0,
        reg_alpha=0.0,

        max_bin=63,

        random_state=SEED,

        n_jobs=-1,
        verbosity=-1,

        importance_type="gain",

        device_type=device_type,
    )

    # deterministic is CPU-only.
    if device_type == "cpu":
        common[
            "deterministic"
        ] = True

        common[
            "force_col_wise"
        ] = True

    return lgb.LGBMClassifier(
        **common
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
        args.xgb_oof,
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
        "Loading competition data and frozen folds..."
    )

    print(
        f"LightGBM version: {lgb.__version__}"
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

    y, positive_label = (
        encode_binary_target(
            train[
                target
            ]
        )
    )

    fold_ids, id_col = (
        validate_folds(
            folds_df,
            train,
        )
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

    X, X_test = (
        prepare_lightgbm_frames(
            train=train,
            test=test,
            features=features,
            categorical_features=categorical_features,
        )
    )

    xgb_oof = load_oof(
        path=args.xgb_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=True,
    )

    catboost_oof = load_oof(
        path=args.catboost_oof,
        train=train,
        fold_ids=fold_ids,
        id_col=id_col,
        required=False,
    )

    xgb_auc = float(
        roc_auc_score(
            y,
            xgb_oof,
        )
    )

    device_type, device_message = (
        detect_device()
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"ID excluded: {id_col!r}"
    )

    print(
        f"Features: {len(features)} "
        f"({len(numeric_features)} numeric, "
        f"{len(categorical_features)} categorical)"
    )

    print()
    print(
        f"Current best depth-4 XGBoost OOF: "
        f"{xgb_auc:.8f}"
    )

    print()
    print(
        f"Selected LightGBM device: "
        f"{device_type!r}"
    )

    print(
        device_message
    )

    print()
    print(
        "LightGBM baseline parameters:"
    )

    print(
        "  n_estimators=5000"
    )

    print(
        "  learning_rate=0.03"
    )

    print(
        "  num_leaves=31"
    )

    print(
        "  min_child_samples=100"
    )

    print(
        "  subsample=0.90"
    )

    print(
        "  colsample_bytree=0.90"
    )

    print(
        "  reg_lambda=2.0"
    )

    print(
        "  max_bin=63"
    )

    print(
        "  early_stopping_rounds=200"
    )

    print()
    print(
        "Running frozen 5-fold LightGBM CV..."
    )
    print()

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
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        model = build_model(
            device_type=device_type
        )

        fit_start = (
            time.perf_counter()
        )

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ],

            eval_set=[
                (
                    X.iloc[
                        valid_idx
                    ],
                    y[
                        valid_idx
                    ],
                )
            ],

            eval_metric="auc",

            categorical_feature=categorical_features,

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=200,
                    verbose=False,
                ),
                lgb.log_evaluation(
                    period=0
                ),
            ],
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
                X.iloc[
                    valid_idx
                ],
                num_iteration=model.best_iteration_,
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test,
                num_iteration=model.best_iteration_,
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

        fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                valid_pred,
            )
        )

        xgb_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                xgb_oof[
                    valid_idx
                ],
            )
        )

        fold_delta = (
            fold_auc
            - xgb_fold_auc
        )

        metric_rows.append(
            {
                "fold": fold,
                "train_rows": len(
                    train_idx
                ),
                "valid_rows": len(
                    valid_idx
                ),
                "lightgbm_auc": fold_auc,
                "xgboost_depth4_auc": xgb_fold_auc,
                "delta_vs_xgboost": fold_delta,
                "best_iteration": int(
                    model.best_iteration_
                ),
                "fit_seconds": float(
                    fit_seconds
                ),
                "inference_seconds_valid_plus_test": float(
                    inference_seconds
                ),
                "device_type": device_type,
            }
        )

        gain_importance = (
            model.booster_
            .feature_importance(
                importance_type="gain"
            )
        )

        gain_sum = float(
            gain_importance.sum()
        )

        if gain_sum > 0:
            gain_importance = (
                gain_importance
                / gain_sum
            )

        for feature, importance in zip(
            features,
            gain_importance,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {fold}: "
            f"LGBM={fold_auc:.6f} | "
            f"XGB={xgb_fold_auc:.6f} | "
            f"delta={fold_delta:+.6f} | "
            f"best_iter={model.best_iteration_} | "
            f"fit={fit_seconds:.1f}s"
        )

        del model

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "LightGBM OOF predictions contain NaNs."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    lgb_oof_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_xgb = (
        lgb_oof_auc
        - xgb_auc
    )

    mean_fold_auc = float(
        fold_metrics[
            "lightgbm_auc"
        ].mean()
    )

    fold_auc_std = float(
        fold_metrics[
            "lightgbm_auc"
        ].std(
            ddof=1
        )
    )

    folds_beating_xgb = int(
        (
            fold_metrics[
                "delta_vs_xgboost"
            ] > 0
        ).sum()
    )

    xgb_probability_corr = float(
        np.corrcoef(
            oof,
            xgb_oof,
        )[0, 1]
    )

    xgb_rank_corr = (
        rank_correlation(
            oof,
            xgb_oof,
        )
    )

    cat_auc = None
    cat_probability_corr = None
    cat_rank_corr = None

    if catboost_oof is not None:
        cat_auc = float(
            roc_auc_score(
                y,
                catboost_oof,
            )
        )

        cat_probability_corr = float(
            np.corrcoef(
                oof,
                catboost_oof,
            )[0, 1]
        )

        cat_rank_corr = (
            rank_correlation(
                oof,
                catboost_oof,
            )
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
            mean_gain_importance=(
                "gain_importance",
                "mean",
            ),
            std_gain_importance=(
                "gain_importance",
                "std",
            ),
        )
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ----------------------------
    # Save artifacts
    # ----------------------------

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
            "prediction": test_prediction.astype(
                np.float32
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

    # ----------------------------
    # Summary
    # ----------------------------

    summary_lines = [
        "EXPERIMENT: LIGHTGBM_RAW_BASELINE",
        "=" * 76,
        "",
        "QUESTION",
        "Can LightGBM capture useful structure that differs from our",
        "current best depth-4 XGBoost model?",
        "",
        "DEVICE",
        f"LightGBM version: {lgb.__version__}",
        f"Selected device_type: {device_type!r}",
        f"Device probe message: {device_message}",
        "",
        "VALIDATION",
        "Same frozen 5-fold competition split.",
        "",
        "FEATURE POLICY",
        f"Excluded ID: {id_col!r}",
        f"Raw features: {len(features)}",
        f"Numeric features: {numeric_features}",
        f"Categorical features: {categorical_features}",
        "",
        "MODEL",
        "LGBMClassifier",
        "n_estimators=5000",
        "learning_rate=0.03",
        "num_leaves=31",
        "min_child_samples=100",
        "subsample=0.90",
        "colsample_bytree=0.90",
        "reg_lambda=2.0",
        "max_bin=63",
        "early_stopping_rounds=200",
        "",
        "RESULTS",
        f"LightGBM OOF AUC: {lgb_oof_auc:.8f}",
        f"Mean fold AUC: {mean_fold_auc:.8f}",
        f"Fold AUC std: {fold_auc_std:.8f}",
        f"Depth-4 XGBoost OOF AUC: {xgb_auc:.8f}",
        f"Delta vs XGBoost: {delta_vs_xgb:+.8f}",
        f"Folds beating XGBoost: {folds_beating_xgb}/5",
        f"OOF probability correlation vs XGBoost: "
        f"{xgb_probability_corr:.6f}",
        f"OOF rank correlation vs XGBoost: "
        f"{xgb_rank_corr:.6f}",
        f"Total runtime: {total_seconds:.2f} seconds",
    ]

    if cat_auc is not None:
        summary_lines.extend(
            [
                "",
                "DIVERSITY VS CATBOOST",
                f"CatBoost OOF AUC: {cat_auc:.8f}",
                f"OOF probability correlation vs CatBoost: "
                f"{cat_probability_corr:.6f}",
                f"OOF rank correlation vs CatBoost: "
                f"{cat_rank_corr:.6f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "TOP MEAN GAIN IMPORTANCES",
        ]
    )

    for row in (
        importance_summary
        .head(10)
        .itertuples()
    ):
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "A LightGBM model can still be useful even if it is slightly below",
            "XGBoost, provided its OOF rankings are sufficiently different.",
            "",
            "This is a raw baseline, not tuned LightGBM.",
        ]
    )

    (
        args.output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "LIGHTGBM RAW BASELINE COMPLETE"
    )
    print("=" * 78)

    print(
        f"Device          : {device_type}"
    )

    print(
        f"LightGBM OOF    : {lgb_oof_auc:.8f}"
    )

    print(
        f"XGBoost OOF     : {xgb_auc:.8f}"
    )

    print(
        f"Delta           : {delta_vs_xgb:+.8f}"
    )

    print(
        f"Folds won       : {folds_beating_xgb}/5"
    )

    print(
        f"Rank corr vs XGB: {xgb_rank_corr:.6f}"
    )

    print(
        f"Runtime         : {total_seconds:.2f}s"
    )

    print(
        f"Artifacts       : {args.output_dir.resolve()}"
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
