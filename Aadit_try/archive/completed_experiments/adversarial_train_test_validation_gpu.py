"""
adversarial_train_test_validation_gpu.py

Kaggle Playground Series S6E9
Train-vs-test adversarial validation.

QUESTION
--------
Can a model distinguish competition TRAIN rows from competition TEST rows
using only target-free features?

If yes, which representations expose the shift?

IMPORTANT
---------
This is NOT a competition-target experiment and is NOT directly comparable
to the frozen target OOF scores.

The competition target is NEVER used.

ADVERSARIAL LABEL
-----------------
0 = competition train row
1 = competition test row

CRITICAL
--------
'id' is excluded because it is sequential and trivially separates train/test.

FEATURE VIEW
------------
- raw 13 competition features
- exact Annual_Income_USD categorical identity
- exact Daily_Commute_km categorical identity
- income buckets: $50 / $250 / $1k / $10k / $100k
- commute buckets: 1 / 5 / 10 km
- income digit decomposition

No target encoding.
No competition target.
No model predictions.
No source data.
No train/test-specific frequency features.

VALIDATION
----------
5-fold StratifiedKFold on the combined train+test adversarial population:
    n_splits=5
    shuffle=True
    random_state=42

This separate adversarial split is appropriate because the frozen competition
target folds do not cover test rows.

Run:
    python adversarial_train_test_validation_gpu.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U xgboost\n"
    ) from exc


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "adversarial_train_test_validation_gpu"
)

TARGET = "Will_Buy_EV"
ID_COL = "id"
SEED = 42


INCOME_BUCKETS = [
    ("ADV__income_50_cat", 50.0),
    ("ADV__income_250_cat", 250.0),
    ("ADV__income_1k_cat", 1_000.0),
    ("ADV__income_10k_cat", 10_000.0),
    ("ADV__income_100k_cat", 100_000.0),
]

COMMUTE_BUCKETS = [
    ("ADV__commute_1km_cat", 1.0),
    ("ADV__commute_5km_cat", 5.0),
    ("ADV__commute_10km_cat", 10.0),
]


def canonical_numeric_string(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{series.name} contains non-finite values."
        )

    rounded = np.rint(values)

    if np.allclose(
        values,
        rounded,
        rtol=0.0,
        atol=1e-10,
    ):
        strings = (
            rounded
            .astype(np.int64)
            .astype(str)
        )
    else:
        strings = np.array(
            [
                format(float(v), ".12g")
                for v in values
            ],
            dtype=object,
        )

    return pd.Series(
        strings,
        index=series.index,
        dtype="object",
    )


def bucket_string(
    series: pd.Series,
    width: float,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{series.name} contains non-finite values."
        )

    buckets = np.floor(
        values / width
    ).astype(np.int64)

    return pd.Series(
        buckets.astype(str),
        index=series.index,
        dtype="object",
    )


def add_income_digits(
    frame: pd.DataFrame,
    income: pd.Series,
) -> list[str]:
    x = (
        pd.to_numeric(
            income,
            errors="raise",
        )
        .round()
        .astype(np.int64)
        .to_numpy()
    )

    features = [
        "ADV__income_ones",
        "ADV__income_tens",
        "ADV__income_hundreds",
        "ADV__income_thousands",
        "ADV__income_ten_thousands",
        "ADV__income_hundred_thousands",
        "ADV__income_last2",
        "ADV__income_last3",
    ]

    frame[features[0]] = x % 10
    frame[features[1]] = (x // 10) % 10
    frame[features[2]] = (x // 100) % 10
    frame[features[3]] = (x // 1_000) % 10
    frame[features[4]] = (x // 10_000) % 10
    frame[features[5]] = (x // 100_000) % 10
    frame[features[6]] = x % 100
    frame[features[7]] = x % 1_000

    return features


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[str],
    pd.DataFrame,
]:
    if TARGET not in train.columns:
        raise ValueError(
            f"Expected target column {TARGET!r} in train."
        )

    if TARGET in test.columns:
        raise ValueError(
            f"Unexpected target column {TARGET!r} in test."
        )

    common_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != ID_COL
    ]

    if len(common_features) != 13:
        print(
            f"[WARN] Expected 13 raw features but found "
            f"{len(common_features)}."
        )

    combined = pd.concat(
        [
            train[common_features],
            test[common_features],
        ],
        axis=0,
        ignore_index=True,
    )

    source_label = np.concatenate(
        [
            np.zeros(
                len(train),
                dtype=np.int8,
            ),
            np.ones(
                len(test),
                dtype=np.int8,
            ),
        ]
    )

    raw_categoricals = [
        c
        for c in common_features
        if (
            pd.api.types.is_object_dtype(
                combined[c]
            )
            or pd.api.types.is_bool_dtype(
                combined[c]
            )
        )
    ]

    for c in raw_categoricals:
        combined[c] = (
            combined[c]
            .fillna("__MISSING__")
            .astype(str)
        )

    engineered_cats = []

    combined[
        "ADV__income_exact_cat"
    ] = canonical_numeric_string(
        combined[
            "Annual_Income_USD"
        ]
    )
    engineered_cats.append(
        "ADV__income_exact_cat"
    )

    combined[
        "ADV__commute_exact_cat"
    ] = canonical_numeric_string(
        combined[
            "Daily_Commute_km"
        ]
    )
    engineered_cats.append(
        "ADV__commute_exact_cat"
    )

    for name, width in INCOME_BUCKETS:
        combined[name] = bucket_string(
            combined[
                "Annual_Income_USD"
            ],
            width,
        )
        engineered_cats.append(name)

    for name, width in COMMUTE_BUCKETS:
        combined[name] = bucket_string(
            combined[
                "Daily_Commute_km"
            ],
            width,
        )
        engineered_cats.append(name)

    digit_features = add_income_digits(
        combined,
        combined[
            "Annual_Income_USD"
        ],
    )

    categorical_features = (
        raw_categoricals
        + engineered_cats
    )

    diagnostics = []

    n_train = len(train)

    for c in categorical_features:
        train_values = set(
            combined.loc[
                : n_train - 1,
                c,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        test_series = (
            combined.loc[
                n_train:,
                c,
            ]
            .astype(str)
        )

        diagnostics.append(
            {
                "feature": c,
                "train_unique": int(
                    combined.loc[
                        : n_train - 1,
                        c,
                    ].nunique()
                ),
                "test_unique": int(
                    test_series.nunique()
                ),
                "test_seen_in_train_rate": float(
                    test_series.isin(
                        train_values
                    ).mean()
                ),
            }
        )

    # XGBoost native categorical support requires pandas category dtype.
    # Categories are constructed jointly from train+test feature values.
    # This uses no adversarial labels and no competition target.
    for c in categorical_features:
        combined[c] = combined[c].astype(
            "category"
        )

    return (
        combined,
        source_label,
        categorical_features,
        pd.DataFrame(
            diagnostics
        ),
    )


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=SEED,
        n_jobs=10,
        early_stopping_rounds=150,
    )


def gain_importance_frame(
    model: xgb.XGBClassifier,
    fold: int,
) -> pd.DataFrame:
    booster = model.get_booster()

    gain = booster.get_score(
        importance_type="gain"
    )

    rows = []

    for feature in model.feature_names_in_:
        rows.append(
            {
                "fold": fold,
                "feature": feature,
                "gain": float(
                    gain.get(
                        feature,
                        0.0,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    for path in [
        TRAIN_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    train = pd.read_csv(
        TRAIN_PATH
    )
    test = pd.read_csv(
        TEST_PATH
    )

    if (
        ID_COL in train.columns
        and ID_COL in test.columns
    ):
        print(
            "CRITICAL CHECK: id is excluded from adversarial features."
        )

    (
        X,
        y_adv,
        categorical_features,
        category_diagnostics,
    ) = prepare_features(
        train,
        test,
    )

    print("=" * 100)
    print(
        "ADVERSARIAL TRAIN-vs-TEST VALIDATION"
    )
    print("=" * 100)
    print(
        f"Competition train rows : {len(train):,}"
    )
    print(
        f"Competition test rows  : {len(test):,}"
    )
    print(
        f"Combined rows          : {len(X):,}"
    )
    print(
        f"Features               : {X.shape[1]}"
    )
    print(
        f"Categorical features   : {len(categorical_features)}"
    )
    print(
        f"Adversarial positive rate (test rows): "
        f"{y_adv.mean():.6f}"
    )
    print()
    print(
        "Competition target is NOT used."
    )
    print(
        "Sequential id is NOT used."
    )
    print(
        "No target encoding / source data / model predictions are used."
    )
    print()
    print(
        "Validation: StratifiedKFold(5, shuffle=True, random_state=42)"
    )
    print()

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.full(
        len(X),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []
    importance_frames = []

    total_start = (
        time.perf_counter()
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        splitter.split(
            X,
            y_adv,
        )
    ):
        fold_start = (
            time.perf_counter()
        )

        model = build_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            y_adv[
                train_idx
            ],
            eval_set=[
                (
                    X.iloc[
                        valid_idx
                    ],
                    y_adv[
                        valid_idx
                    ],
                )
            ],
            verbose=False,
        )

        if (
            model.best_iteration
            is None
        ):
            iteration_range = None
            best_iteration = -1
        else:
            best_iteration = int(
                model.best_iteration
            )
            iteration_range = (
                0,
                best_iteration + 1,
            )

        pred = model.predict_proba(
            X.iloc[
                valid_idx
            ],
            iteration_range=iteration_range,
        )[:, 1]

        oof[
            valid_idx
        ] = pred

        fold_auc = float(
            roc_auc_score(
                y_adv[
                    valid_idx
                ],
                pred,
            )
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": (
                    best_iteration
                ),
                "seconds": (
                    fold_seconds
                ),
            }
        )

        importance_frames.append(
            gain_importance_frame(
                model,
                fold,
            )
        )

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.8f} | "
            f"best_iter={best_iteration} | "
            f"{fold_seconds:.2f}s"
        )

        del model

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "Adversarial OOF contains NaNs."
        )

    overall_auc = float(
        roc_auc_score(
            y_adv,
            oof,
        )
    )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    importance_by_fold = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    importance_summary = (
        importance_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_gain=(
                "gain",
                "mean",
            ),
            std_gain=(
                "gain",
                "std",
            ),
        )
        .sort_values(
            "mean_gain",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # Normalize mean gain for easier reading.
    gain_total = float(
        importance_summary[
            "mean_gain"
        ].sum()
    )

    if gain_total > 0:
        importance_summary[
            "gain_share"
        ] = (
            importance_summary[
                "mean_gain"
            ]
            / gain_total
        )
    else:
        importance_summary[
            "gain_share"
        ] = 0.0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    importance_by_fold.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    importance_summary.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    category_diagnostics.to_csv(
        OUTPUT_DIR
        / "categorical_shift_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index_combined": np.arange(
                len(X),
                dtype=np.int64,
            ),
            "dataset_source": y_adv,
            "oof_prediction_test_probability": (
                oof.astype(
                    np.float32
                )
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    top_n = min(
        20,
        len(
            importance_summary
        ),
    )

    summary_lines = [
        "EXPERIMENT: ADVERSARIAL TRAIN-vs-TEST VALIDATION",
        "=" * 86,
        "",
        "QUESTION",
        "Can target-free features distinguish competition train from test?",
        "",
        "LABEL",
        "0 = train row",
        "1 = test row",
        "",
        "CRITICAL EXCLUSIONS",
        "- competition target NOT used",
        "- id NOT used",
        "- no target encoding",
        "- no source data",
        "- no model predictions",
        "",
        "VALIDATION",
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "",
        "RESULT",
        f"Adversarial OOF AUC: {overall_auc:.8f}",
        f"Mean fold AUC: {fold_metrics['auc'].mean():.8f}",
        f"Fold AUC std: {fold_metrics['auc'].std(ddof=1):.8f}",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"{row.auc:.8f}"
        )

    summary_lines.extend(
        [
            "",
            f"TOP {top_n} FEATURES BY MEAN GAIN",
        ]
    )

    for row in (
        importance_summary
        .head(top_n)
        .itertuples()
    ):
        summary_lines.append(
            f"{row.feature}: "
            f"mean_gain={row.mean_gain:.8f}, "
            f"gain_share={row.gain_share:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "AUC near 0.50 means little detectable covariate shift.",
            "Higher AUC means train/test membership is increasingly predictable.",
            "Feature importance identifies candidate shift drivers; it does not prove",
            "that a feature should be removed from the competition model.",
        ]
    )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "ADVERSARIAL VALIDATION COMPLETE"
    )
    print("=" * 100)
    print(
        f"OOF AUC        : "
        f"{overall_auc:.8f}"
    )
    print(
        f"Mean fold AUC  : "
        f"{fold_metrics['auc'].mean():.8f}"
    )
    print(
        f"Fold AUC std   : "
        f"{fold_metrics['auc'].std(ddof=1):.8f}"
    )
    print(
        f"Runtime        : "
        f"{total_seconds:.2f}s"
    )
    print()
    print(
        f"Top {top_n} shift features:"
    )
    print(
        importance_summary[
            [
                "feature",
                "mean_gain",
                "gain_share",
            ]
        ]
        .head(top_n)
        .to_string(
            index=False
        )
    )
    print()
    print(
        f"Artifacts      : "
        f"{OUTPUT_DIR}"
    )
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. feature_importance.csv")
    print("  4. categorical_shift_diagnostics.csv")


if __name__ == "__main__":
    main()
