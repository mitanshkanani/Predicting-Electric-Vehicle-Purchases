"""
catboost_value_ids_interactions_gpu.py

Phase 5E — Exact-value identities + explicit subsidy interactions.

CURRENT CHAMPION
----------------
CatBoost GPU + exact value identities:
    Local OOF AUC = 0.94536428
    Public LB     = 0.94552

HYPOTHESIS
----------
The model already benefits strongly from exact-value identity features.
Now test whether three explicit, target-free subsidy interactions improve
split allocation further:

    Subsidy_x_EnvConcern
    Subsidy_x_Income
    Subsidy_x_HomeCharging

CONTROLLED EXPERIMENT
---------------------
SAME:
- frozen 5 folds
- all 13 raw features
- exact-value categorical copies:
      Annual_Income_USD_id
      Daily_Commute_km_id
- CatBoost GPU
- iterations=2000
- learning_rate=0.05
- depth=6
- early_stopping_rounds=150
- random_seed=42

ONLY CHANGE:
- add 3 explicit subsidy interaction features

Run:
    python catboost_value_ids_interactions_gpu.py
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

VALUE_ID_FEATURES = [
    "Annual_Income_USD",
    "Daily_Commute_km",
]

INTERACTION_FEATURES = [
    "Subsidy_x_EnvConcern",
    "Subsidy_x_Income",
    "Subsidy_x_HomeCharging",
]

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_CHAMPION_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_exact_value_ids_gpu"
    / "oof_predictions.csv"
)

DEFAULT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_depth_sweep_gpu"
    / "best_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_value_ids_interactions_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test 3 subsidy interactions on top of exact-value CatBoost."
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
        "--champion-oof",
        type=Path,
        default=DEFAULT_CHAMPION_OOF,
    )

    parser.add_argument(
        "--xgb-oof",
        type=Path,
        default=DEFAULT_XGB_OOF,
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
            f"Expected exactly one train-only target column, found {train_only}"
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

    return (
        (y == positive).astype(np.int8).to_numpy(),
        positive,
    )


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

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
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


def binary_yes_indicator(
    series: pd.Series,
) -> np.ndarray:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        values == "yes"
    ).astype(
        np.int8
    ).to_numpy()


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_features: list[str],
    raw_categoricals: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
    x_train = train[
        raw_features
    ].copy()

    x_test = test[
        raw_features
    ].copy()

    # Normalize raw categorical values for CatBoost.
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

        value_id_columns.append(
            id_col
        )

    # ---------------------------------------------------------
    # Explicit target-free interactions
    # ---------------------------------------------------------

    train_sub = binary_yes_indicator(
        train["Subsidy_Available"]
    )

    test_sub = binary_yes_indicator(
        test["Subsidy_Available"]
    )

    train_home = binary_yes_indicator(
        train["Home_Charging_Possible"]
    )

    test_home = binary_yes_indicator(
        test["Home_Charging_Possible"]
    )

    train_env = pd.to_numeric(
        train["Environmental_Concern_Level"],
        errors="raise",
    ).to_numpy(
        dtype=np.float32
    )

    test_env = pd.to_numeric(
        test["Environmental_Concern_Level"],
        errors="raise",
    ).to_numpy(
        dtype=np.float32
    )

    train_income = pd.to_numeric(
        train["Annual_Income_USD"],
        errors="raise",
    ).to_numpy(
        dtype=np.float32
    )

    test_income = pd.to_numeric(
        test["Annual_Income_USD"],
        errors="raise",
    ).to_numpy(
        dtype=np.float32
    )

    x_train["Subsidy_x_EnvConcern"] = (
        train_sub * train_env
    ).astype(np.float32)

    x_test["Subsidy_x_EnvConcern"] = (
        test_sub * test_env
    ).astype(np.float32)

    x_train["Subsidy_x_Income"] = (
        train_sub * train_income
    ).astype(np.float32)

    x_test["Subsidy_x_Income"] = (
        test_sub * test_income
    ).astype(np.float32)

    x_train["Subsidy_x_HomeCharging"] = (
        train_sub * train_home
    ).astype(np.int8)

    x_test["Subsidy_x_HomeCharging"] = (
        test_sub * test_home
    ).astype(np.int8)

    categorical_features = (
        raw_categoricals
        + value_id_columns
    )

    return (
        x_train,
        x_test,
        categorical_features,
        value_id_columns,
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"OOF file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "fold",
        "oof_prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected OOF columns in {path}"
        )

    if len(df) != len(train):
        raise ValueError(
            f"OOF row count mismatch in {path}"
        )

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(
            f"OOF row order mismatch in {path}"
        )

    if not np.array_equal(
        df["fold"].to_numpy(),
        fold_ids,
    ):
        raise ValueError(
            f"OOF fold mismatch in {path}"
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
            f"OOF ID mismatch in {path}"
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
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    br = (
        pd.Series(b)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    return float(
        np.corrcoef(
            ar,
            br,
        )[0, 1]
    )


def build_model() -> CatBoostClassifier:
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
        args.champion_oof,
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
        "Loading data, frozen folds, and champion predictions..."
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
        value_id_columns,
    ) = prepare_frames(
        train,
        test,
        raw_features,
        raw_categoricals,
    )

    champion_oof = load_oof(
        args.champion_oof,
        train,
        fold_ids,
        id_col,
    )

    xgb_oof = load_oof(
        args.xgb_oof,
        train,
        fold_ids,
        id_col,
    )

    champion_auc = float(
        roc_auc_score(
            y,
            champion_oof,
        )
    )

    xgb_auc = float(
        roc_auc_score(
            y,
            xgb_oof,
        )
    )

    print()
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )

    print(
        f"Current exact-value CatBoost OOF: "
        f"{champion_auc:.8f}"
    )

    print(
        f"Depth-4 XGBoost OOF: "
        f"{xgb_auc:.8f}"
    )

    print()
    print(
        "Added interaction features:"
    )

    for feature in INTERACTION_FEATURES:
        print(
            f"  - {feature}"
        )

    print()
    print(
        f"Total model features: {len(X.columns)}"
    )

    print(
        f"CatBoost categorical features: "
        f"{len(categorical_features)}"
    )

    print()
    print(
        "Running frozen 5-fold interaction experiment..."
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

    test_fold_predictions: list[np.ndarray] = []

    fold_rows = []
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

        train_pool = Pool(
            X.iloc[
                train_idx
            ],
            label=y[
                train_idx
            ],
            cat_features=categorical_features,
            feature_names=list(
                X.columns
            ),
        )

        valid_pool = Pool(
            X.iloc[
                valid_idx
            ],
            label=y[
                valid_idx
            ],
            cat_features=categorical_features,
            feature_names=list(
                X.columns
            ),
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

        champion_fold_auc = float(
            roc_auc_score(
                y[
                    valid_idx
                ],
                champion_oof[
                    valid_idx
                ],
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

        fold_rows.append(
            {
                "fold": fold,

                "value_id_champion_auc": (
                    champion_fold_auc
                ),

                "value_id_interactions_auc": (
                    fold_auc
                ),

                "delta_vs_value_id_champion": (
                    fold_auc
                    - champion_fold_auc
                ),

                "xgboost_depth4_auc": (
                    xgb_fold_auc
                ),

                "delta_vs_xgboost": (
                    fold_auc
                    - xgb_fold_auc
                ),

                "best_iteration_zero_based": int(
                    model.get_best_iteration()
                ),

                "tree_count": int(
                    model.tree_count_
                ),

                "fit_seconds": float(
                    fit_seconds
                ),
            }
        )

        importances = (
            model.get_feature_importance(
                type="PredictionValuesChange"
            )
        )

        for feature, importance in zip(
            X.columns,
            importances,
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature": feature,

                    "is_value_identity": (
                        feature
                        in value_id_columns
                    ),

                    "is_explicit_interaction": (
                        feature
                        in INTERACTION_FEATURES
                    ),

                    "importance": float(
                        importance
                    ),
                }
            )

        print()
        print(
            f"Fold {fold}: "
            f"champion={champion_fold_auc:.6f} | "
            f"interactions={fold_auc:.6f} | "
            f"delta={fold_auc - champion_fold_auc:+.6f} | "
            f"best_iter={model.get_best_iteration()} | "
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

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "OOF predictions contain NaN values."
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    experiment_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_champion = (
        experiment_auc
        - champion_auc
    )

    improved_folds = int(
        (
            fold_metrics[
                "delta_vs_value_id_champion"
            ] > 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            champion_oof,
        )[0, 1]
    )

    rank_corr = rank_correlation(
        oof,
        champion_oof,
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
            [
                "feature",
                "is_value_identity",
                "is_explicit_interaction",
            ],
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

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

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

    decision = (
        "KEEP"
        if (
            delta_vs_champion > 0
            and
            improved_folds >= 3
        )
        else
        "REJECT_FOR_NOW"
    )

    summary_lines = [
        "EXPERIMENT: CATBOOST_VALUE_IDS_PLUS_INTERACTIONS_GPU",
        "=" * 76,
        "",
        "HYPOTHESIS",
        "Do three explicit subsidy interactions improve the exact-value",
        "identity CatBoost champion?",
        "",
        "CONTROL",
        "Same exact-value CatBoost pipeline.",
        "Only three target-free subsidy interactions were added.",
        f"Interactions: {INTERACTION_FEATURES}",
        "",
        "RESULTS",
        f"Value-ID champion OOF AUC: {champion_auc:.8f}",
        f"Value-ID + interactions OOF AUC: {experiment_auc:.8f}",
        f"OOF AUC delta: {delta_vs_champion:+.8f}",
        f"Folds improved: {improved_folds}/5",
        f"OOF probability correlation vs champion: {probability_corr:.6f}",
        f"OOF rank correlation vs champion: {rank_corr:.6f}",
        f"Total runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD DELTAS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.value_id_champion_auc:.8f} -> "
            f"{row.value_id_interactions_auc:.8f} "
            f"({row.delta_vs_value_id_champion:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "EXPLICIT INTERACTION IMPORTANCES",
        ]
    )

    interaction_importance = importance_summary[
        importance_summary[
            "is_explicit_interaction"
        ]
    ]

    for row in interaction_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_importance:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "If this improves consistently, these interactions become part",
            "of the champion representation before seed averaging/tuning.",
            "",
            "If it does not, we keep the simpler value-ID champion.",
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
        "VALUE-ID + INTERACTIONS EXPERIMENT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Champion OOF     : {champion_auc:.8f}"
    )

    print(
        f"Interactions OOF : {experiment_auc:.8f}"
    )

    print(
        f"Delta            : {delta_vs_champion:+.8f}"
    )

    print(
        f"Folds improved   : {improved_folds}/5"
    )

    print(
        f"Decision         : {decision}"
    )

    print(
        f"Runtime          : {total_seconds:.2f}s"
    )

    print(
        f"Artifacts        : {args.output_dir.resolve()}"
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
