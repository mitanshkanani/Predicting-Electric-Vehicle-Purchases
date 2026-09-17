"""
xgboost_source_dual_target_encoding_gpu.py

Kaggle Playground Series S6E9
Controlled structural experiment: source-data dual target encoding on top of
the CURRENT hierarchical-income XGBoost champion.

HYPOTHESIS
----------
Our competition-derived target encodings learn conditional target structure
inside the synthetic competition population.

The original 10,000-row EV source dataset may contain a slightly different but
related target surface from the generator/original population. Public S6E9 work
has explored "dual-target encoding", so we test the mechanism rigorously on OUR
frozen folds instead of trusting external CV/LB numbers.

This experiment adds a small parallel set of SOURCE-ONLY Bayesian target
statistics. No competition targets are used to build these new features.

SOURCE-ONLY FEATURES
--------------------
1. SOURCE_TE__income_10k_bucket
2. SOURCE_TE__environmental_concern
3. SOURCE_TE__subsidy
4. SOURCE_TE__range_anxiety
5. SOURCE_TE__behavior_state

behavior_state =
    Environmental_Concern_Level
    x Subsidy_Available
    x Range_Anxiety_Level

All source encodings use the SAME fixed Bayesian smoothing m=2.

ONLY CHANGE
-----------
Add the five source-only target-statistic features above.

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw features
- income digit decomposition
- exact income/commute frequency features
- exact income/commute nested Bayesian TE
- hierarchical income TE at $1k/$10k/$100k
- smoothing m=2
- nested learned logistic base margin
- depth-4 XGBoost configuration
- seed 42
- CUDA GPU
- early stopping
- no public-LB optimization

LEAKAGE SAFETY
--------------
The new source features are fitted ONLY from the external/original source
dataset and its labels.

Therefore:
- no competition validation target enters these new features
- no competition training target enters these new features
- all competition train/test rows receive mappings learned only from source

The existing competition target encodings remain nested/OOF exactly as before.

SOURCE FILE HANDLING
--------------------
The handoff documents the exact source filename:

    EV_Adoption_and_Range_Anxiety_Dataset.csv

Its local directory was not frozen in the handoff. This script DOES NOT invent
a path. It recursively searches the current repo for that exact filename.

- exactly 1 match -> use it and print the resolved path
- 0 matches       -> stop with a clear error
- >1 match        -> stop and print all matches

Run:
    python xgboost_source_dual_target_encoding_gpu.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U xgboost\n"
    ) from exc

try:
    import lightgbm_engineered_learned_margin_cpu as base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import lightgbm_engineered_learned_margin_cpu.py.\n"
        "Place this experiment file in the same repo root as that validated script.\n"
    ) from exc

try:
    import xgboost_hierarchical_income_te_gpu as champion
except ImportError as exc:
    raise SystemExit(
        "\nCould not import xgboost_hierarchical_income_te_gpu.py.\n"
        "Place this experiment file in the same repo root as that validated script.\n"
    ) from exc


SEED = 42
SMOOTHING = 2.0

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_BASELINE_AUC = 0.94587366
AUC_CHECK_TOLERANCE = 2e-5

SOURCE_FILENAME = "EV_Adoption_and_Range_Anxiety_Dataset.csv"

DEFAULT_FOLDS_PATH = (
    Path("artifacts")
    / "validation"
    / "candidate_folds.csv"
)

DEFAULT_BASELINE_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_income_te_gpu"
    / "oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "xgboost_source_dual_target_encoding_gpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Current hierarchical-income XGBoost champion + "
            "source-only dual target encoding."
        )
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
        "--baseline-oof",
        type=Path,
        default=DEFAULT_BASELINE_OOF,
    )

    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the documented source CSV. "
            "If omitted, search the current repo recursively for the exact filename."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def find_source_file(
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Explicit --source-path does not exist:\n"
                f"{path}"
            )

        if path.name != SOURCE_FILENAME:
            raise ValueError(
                "Explicit --source-path has the wrong filename.\n"
                f"Expected exact filename: {SOURCE_FILENAME}\n"
                f"Received              : {path.name}"
            )

        return path

    cwd = Path(".").resolve()

    # First search the directory from which the experiment is being run.
    # Then, if Aadit_try is a subdirectory of a larger Git repository,
    # locate the nearest ancestor containing .git and search that entire repo.
    search_roots: list[Path] = [cwd]

    git_root: Path | None = None
    for candidate in [cwd, *cwd.parents]:
        if (candidate / ".git").exists():
            git_root = candidate
            break

    if git_root is not None and git_root not in search_roots:
        search_roots.append(git_root)

    matches: list[Path] = []

    for root in search_roots:
        root_matches = [
            p.resolve()
            for p in root.rglob(SOURCE_FILENAME)
            if p.is_file()
        ]

        for match in root_matches:
            if match not in matches:
                matches.append(match)

    matches = sorted(matches)

    if len(matches) == 0:
        searched = "\n".join(
            f"  - {root}"
            for root in search_roots
        )

        git_note = (
            str(git_root)
            if git_root is not None
            else "No ancestor containing .git was found."
        )

        raise FileNotFoundError(
            "\nCould not find the documented source dataset.\n"
            f"Exact filename searched: {SOURCE_FILENAME}\n"
            "Search roots checked:\n"
            f"{searched}\n"
            f"Detected Git root: {git_note}\n\n"
            "No path was guessed. If the source CSV is stored somewhere else, "
            "rerun this SAME experiment with its real path, for example:\n"
            "    python xgboost_source_dual_target_encoding_gpu.py "
            '--source-path "C:\\full\\path\\EV_Adoption_and_Range_Anxiety_Dataset.csv"\n'
        )

    if len(matches) > 1:
        formatted = "\n".join(
            f"  - {p}"
            for p in matches
        )

        raise RuntimeError(
            "\nMultiple copies of the source dataset were found.\n"
            "The experiment will not guess which one is correct.\n"
            f"{formatted}\n\n"
            "Rerun this SAME file with an explicit --source-path."
        )

    return matches[0]


def normalize_string_key(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("__MISSING__")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def source_income_10k_key(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    out = pd.Series(
        "__MISSING__",
        index=series.index,
        dtype="object",
    )

    mask = numeric.notna()

    if mask.any():
        values = np.rint(
            numeric.loc[mask].to_numpy(
                dtype=np.float64
            )
        ).astype(np.int64)

        out.loc[mask] = (
            values // 10_000
        ).astype(str)

    return out


def environmental_key(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    out = pd.Series(
        "__MISSING__",
        index=series.index,
        dtype="object",
    )

    mask = numeric.notna()

    if mask.any():
        values = numeric.loc[
            mask
        ].to_numpy(dtype=np.float64)

        rounded = np.rint(values)

        if np.allclose(
            values,
            rounded,
            atol=1e-8,
        ):
            out.loc[mask] = (
                rounded.astype(np.int64)
                .astype(str)
            )
        else:
            out.loc[mask] = [
                format(float(v), ".12g")
                for v in values
            ]

    return out


def build_source_keys(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    required = [
        "Annual_Income_USD",
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required source-key columns: {missing}"
        )

    income_10k = source_income_10k_key(
        df["Annual_Income_USD"]
    )

    env = environmental_key(
        df["Environmental_Concern_Level"]
    )

    subsidy = normalize_string_key(
        df["Subsidy_Available"]
    )

    anxiety = normalize_string_key(
        df["Range_Anxiety_Level"]
    )

    behavior = (
        env.astype(str)
        + "|"
        + subsidy.astype(str)
        + "|"
        + anxiety.astype(str)
    )

    return {
        "SOURCE_TE__income_10k_bucket": income_10k,
        "SOURCE_TE__environmental_concern": env,
        "SOURCE_TE__subsidy": subsidy,
        "SOURCE_TE__range_anxiety": anxiety,
        "SOURCE_TE__behavior_state": behavior,
    }


def fit_source_mapping(
    keys: pd.Series,
    y: np.ndarray,
    prior: float,
    smoothing: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "target": y,
        }
    )

    stats = (
        frame
        .groupby(
            "key",
            observed=True,
        )["target"]
        .agg(
            ["sum", "count"]
        )
    )

    stats["encoded"] = (
        stats["sum"]
        + smoothing * prior
    ) / (
        stats["count"]
        + smoothing
    )

    return stats


def apply_source_mapping(
    keys: pd.Series,
    mapping: pd.DataFrame,
    prior: float,
) -> tuple[np.ndarray, float]:
    encoded = (
        keys
        .map(
            mapping["encoded"]
        )
    )

    unseen_rate = float(
        encoded.isna().mean()
    )

    return (
        encoded
        .fillna(prior)
        .to_numpy(dtype=np.float32),
        unseen_rate,
    )


def build_source_only_te_features(
    source: pd.DataFrame,
    competition_train: pd.DataFrame,
    competition_test: pd.DataFrame,
    target: str,
    smoothing: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if target not in source.columns:
        raise ValueError(
            f"Source dataset missing target column {target!r}."
        )

    source_y, source_positive_label = (
        base.encode_binary_target(
            source[target]
        )
    )

    competition_values = set(
        competition_train[target]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    source_values = set(
        source[target]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if competition_values != source_values:
        raise ValueError(
            "Source target labels do not match competition target labels.\n"
            f"Competition labels: {sorted(competition_values)}\n"
            f"Source labels     : {sorted(source_values)}"
        )

    source_prior = float(
        source_y.mean()
    )

    source_keys = build_source_keys(
        source
    )
    train_keys = build_source_keys(
        competition_train
    )
    test_keys = build_source_keys(
        competition_test
    )

    train_features = pd.DataFrame(
        index=np.arange(
            len(competition_train)
        )
    )

    test_features = pd.DataFrame(
        index=np.arange(
            len(competition_test)
        )
    )

    diagnostic_rows = []

    for feature_name in source_keys:
        mapping = fit_source_mapping(
            keys=source_keys[
                feature_name
            ],
            y=source_y,
            prior=source_prior,
            smoothing=smoothing,
        )

        train_encoded, train_unseen = (
            apply_source_mapping(
                keys=train_keys[
                    feature_name
                ],
                mapping=mapping,
                prior=source_prior,
            )
        )

        test_encoded, test_unseen = (
            apply_source_mapping(
                keys=test_keys[
                    feature_name
                ],
                mapping=mapping,
                prior=source_prior,
            )
        )

        train_features[
            feature_name
        ] = train_encoded

        test_features[
            feature_name
        ] = test_encoded

        diagnostic_rows.append(
            {
                "feature": feature_name,
                "smoothing": smoothing,
                "source_prior": source_prior,
                "source_positive_label": str(
                    source_positive_label
                ),
                "source_unique_keys": int(
                    len(mapping)
                ),
                "source_min_group_count": int(
                    mapping["count"].min()
                ),
                "source_median_group_count": float(
                    mapping["count"].median()
                ),
                "source_max_group_count": int(
                    mapping["count"].max()
                ),
                "competition_train_unseen_rate": train_unseen,
                "competition_test_unseen_rate": test_unseen,
                "encoded_min": float(
                    mapping["encoded"].min()
                ),
                "encoded_max": float(
                    mapping["encoded"].max()
                ),
            }
        )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    return (
        train_features,
        test_features,
        diagnostics,
    )


def load_and_verify_baseline_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    pred = base.load_oof(
        path,
        train,
        fold_ids,
        id_col,
    )

    auc = float(
        roc_auc_score(
            y,
            pred,
        )
    )

    if abs(
        auc - EXPECTED_BASELINE_AUC
    ) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            "Loaded baseline OOF does not match the documented current "
            "hierarchical-income XGB champion.\n"
            f"Expected approximately: {EXPECTED_BASELINE_AUC:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}\n"
            f"File                  : {path.resolve()}"
        )

    return pred, auc


def rank_corr(
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

    required_paths = [
        train_path,
        test_path,
        args.folds_path,
        args.baseline_oof,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    source_path = find_source_file(
        args.source_path
    )

    fold_hash = base.sha256_file(
        args.folds_path
    )

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}\n"
            f"File    : {args.folds_path.resolve()}"
        )

    train = pd.read_csv(
        train_path
    )
    test = pd.read_csv(
        test_path
    )
    source = pd.read_csv(
        source_path
    )
    folds_df = pd.read_csv(
        args.folds_path
    )

    target = base.detect_target(
        train,
        test,
    )

    y, positive_label = (
        base.encode_binary_target(
            train[target]
        )
    )

    fold_ids, id_col = (
        base.validate_folds(
            folds_df,
            train,
        )
    )

    baseline_oof, baseline_auc = (
        load_and_verify_baseline_oof(
            path=args.baseline_oof,
            train=train,
            fold_ids=fold_ids,
            id_col=id_col,
            y=y,
        )
    )

    required_source_features = [
        c
        for c in test.columns
        if c != id_col
    ]

    missing_source_features = [
        c
        for c in required_source_features
        if c not in source.columns
    ]

    if missing_source_features:
        raise ValueError(
            "Source dataset does not contain all conceptual competition "
            "features required for alignment.\n"
            f"Missing: {missing_source_features}"
        )

    (
        source_te_train,
        source_te_test,
        source_te_diagnostics,
    ) = build_source_only_te_features(
        source=source,
        competition_train=train,
        competition_test=test,
        target=target,
        smoothing=SMOOTHING,
    )

    raw_features = [
        c
        for c in test.columns
        if c in train.columns
        and c != id_col
    ]

    raw_categoricals = (
        base.detect_raw_categoricals(
            train,
            raw_features,
        )
    )

    X_base, X_test_base = (
        base.prepare_base_frames(
            train=train,
            test=test,
            raw_features=raw_features,
            categorical_features=raw_categoricals,
        )
    )

    (
        X_income_base,
        X_income_test_base,
        digit_features,
    ) = base.add_income_digit_features(
        X_train=X_base,
        X_test=X_test_base,
        train_source=train,
        test_source=test,
    )

    (
        X_candidate_base,
        X_candidate_test_base,
        exact_frequency_features,
    ) = base.add_exact_frequency_features(
        X_train=X_income_base,
        X_test=X_income_test_base,
        train_source=train,
        test_source=test,
    )

    logistic_train_matrix = (
        base.build_logistic_recipe_matrix(
            train
        )
    )

    logistic_test_matrix = (
        base.build_logistic_recipe_matrix(
            test
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print("XGBOOST + SOURCE-DATA DUAL TARGET ENCODING")
    print("=" * 96)
    print(f"XGBoost version: {xgb.__version__}")
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"Current hierarchical-XGB baseline: {baseline_auc:.8f}")
    print(f"Source file resolved: {source_path}")
    print(f"Source shape: {source.shape[0]:,} rows x {source.shape[1]} columns")
    print()
    print("HYPOTHESIS:")
    print(
        "  Parallel source-only target statistics may expose the original "
        "generator/behavior target surface that competition-only encodings "
        "do not fully capture."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Add five source-only Bayesian target-statistic features. "
        "No competition target is used to construct them."
    )
    print()
    print("SOURCE FEATURES:")
    for c in source_te_train.columns:
        print(f"  - {c}")
    print()
    print("HELD FIXED:")
    print("  - frozen 5 folds + hash")
    print("  - current raw features")
    print("  - income digit decomposition")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - hierarchical income TE at $1k/$10k/$100k")
    print("  - smoothing m=2")
    print("  - nested learned logistic base margin")
    print("  - depth-4 XGBoost configuration")
    print("  - seed 42")
    print("  - CUDA GPU")
    print()

    print("SOURCE-TE DIAGNOSTICS")
    print(
        source_te_diagnostics.to_string(
            index=False
        )
    )
    print()

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    exact_diag_frames: list[pd.DataFrame] = []
    hierarchical_diag_frames: list[pd.DataFrame] = []
    logistic_diag_frames: list[pd.DataFrame] = []

    total_start = time.perf_counter()

    for outer_fold in range(5):
        fold_start = time.perf_counter()

        train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        (
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            exact_diag,
        ) = base.build_exact_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        exact_diag_frames.append(
            exact_diag
        )

        (
            train_hte,
            valid_hte,
            test_hte,
            hte_diag,
        ) = champion.build_hierarchical_income_te_for_outer_fold(
            train=train,
            test=test,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
            smoothing=SMOOTHING,
        )

        hierarchical_diag_frames.append(
            hte_diag
        )

        (
            learned_train_margin,
            learned_valid_margin,
            learned_test_margin,
            logistic_diag,
        ) = base.build_learned_logistic_margins_for_outer_fold(
            train_matrix=logistic_train_matrix,
            test_matrix=logistic_test_matrix,
            y=y,
            fold_ids=fold_ids,
            outer_fold=outer_fold,
        )

        logistic_diag_frames.append(
            logistic_diag
        )

        X_train = (
            X_candidate_base.iloc[
                train_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_valid = (
            X_candidate_base.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
            .copy()
        )

        X_test = (
            X_candidate_test_base
            .reset_index(drop=True)
            .copy()
        )

        exact_te_columns = list(
            train_exact_te.columns
        )

        for c in exact_te_columns:
            X_train[c] = train_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_exact_te[
                c
            ].to_numpy(
                dtype=np.float32
            )

        hte_columns = list(
            train_hte.columns
        )

        for c in hte_columns:
            X_train[c] = train_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_valid[c] = valid_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

            X_test[c] = test_hte[
                c
            ].to_numpy(
                dtype=np.float32
            )

        source_columns = list(
            source_te_train.columns
        )

        for c in source_columns:
            X_train[c] = (
                source_te_train.iloc[
                    train_idx
                ][c]
                .reset_index(drop=True)
                .to_numpy(dtype=np.float32)
            )

            X_valid[c] = (
                source_te_train.iloc[
                    valid_idx
                ][c]
                .reset_index(drop=True)
                .to_numpy(dtype=np.float32)
            )

            X_test[c] = (
                source_te_test[c]
                .reset_index(drop=True)
                .to_numpy(dtype=np.float32)
            )

        model = champion.build_model()

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y[train_idx],
            base_margin=learned_train_margin,
            eval_set=[
                (
                    X_valid,
                    y[valid_idx],
                )
            ],
            base_margin_eval_set=[
                learned_valid_margin
            ],
            verbose=False,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        if model.best_iteration is None:
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

        valid_pred = model.predict_proba(
            X_valid,
            base_margin=learned_valid_margin,
            iteration_range=iteration_range,
        )[:, 1]

        test_pred = model.predict_proba(
            X_test,
            base_margin=learned_test_margin,
            iteration_range=iteration_range,
        )[:, 1]

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            test_pred.astype(np.float32)
        )

        baseline_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                baseline_oof[valid_idx],
            )
        )

        candidate_fold_auc = float(
            roc_auc_score(
                y[valid_idx],
                valid_pred,
            )
        )

        delta = (
            candidate_fold_auc
            - baseline_fold_auc
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_rows.append(
            {
                "fold": outer_fold,
                "baseline_auc": baseline_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "delta_vs_baseline": delta,
                "best_iteration": best_iteration,
                "fit_seconds": fit_seconds,
                "total_fold_seconds": fold_seconds,
            }
        )

        for feature, importance in zip(
            X_train.columns,
            model.feature_importances_,
        ):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature,
                    "is_source_te": (
                        feature
                        in source_columns
                    ),
                    "is_hierarchical_income_te": (
                        feature
                        in hte_columns
                    ),
                    "is_exact_te": (
                        feature
                        in exact_te_columns
                    ),
                    "is_income_digit": (
                        feature
                        in digit_features
                    ),
                    "is_exact_frequency": (
                        feature
                        in exact_frequency_features
                    ),
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        print(
            f"Fold {outer_fold}: "
            f"baseline={baseline_fold_auc:.8f} -> "
            f"candidate={candidate_fold_auc:.8f} "
            f"({delta:+.8f}) | "
            f"best_iter={best_iteration}"
        )

        del (
            model,
            X_train,
            X_valid,
            X_test,
            train_exact_te,
            valid_exact_te,
            test_exact_te,
            train_hte,
            valid_hte,
            test_hte,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Candidate OOF contains NaNs."
        )

    candidate_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    delta_vs_baseline = (
        candidate_auc
        - baseline_auc
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    folds_improved = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] > 0
        ).sum()
    )

    folds_worse = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ] < 0
        ).sum()
    )

    probability_corr = float(
        np.corrcoef(
            oof,
            baseline_oof,
        )[0, 1]
    )

    rank_correlation = rank_corr(
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
            [
                "feature",
                "is_source_te",
                "is_hierarchical_income_te",
                "is_exact_te",
                "is_income_digit",
                "is_exact_frequency",
            ],
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
        .reset_index(drop=True)
    )

    source_importance = (
        importance_summary[
            importance_summary[
                "is_source_te"
            ]
        ]
        .copy()
        .sort_values(
            "mean_gain_importance",
            ascending=False,
        )
    )

    if (
        delta_vs_baseline > 0
        and folds_improved >= 3
    ):
        decision = (
            "KEEP_SOURCE_DUAL_TARGET_ENCODING"
        )
    else:
        decision = (
            "REJECT_SOURCE_DUAL_TARGET_ENCODING"
        )

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

    source_importance.to_csv(
        args.output_dir
        / "source_te_feature_importance.csv",
        index=False,
    )

    source_te_diagnostics.to_csv(
        args.output_dir
        / "source_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        exact_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "exact_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        hierarchical_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "hierarchical_income_te_diagnostics.csv",
        index=False,
    )

    pd.concat(
        logistic_diag_frames,
        ignore_index=True,
    ).to_csv(
        args.output_dir
        / "learned_logistic_margin_coefficients.csv",
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

    summary_lines = [
        "EXPERIMENT: XGBOOST + SOURCE-DATA DUAL TARGET ENCODING",
        "=" * 84,
        "",
        "HYPOTHESIS",
        "Can source-only target statistics expose generator/original-population",
        "target structure that complements competition-only target encodings?",
        "",
        "ONLY CHANGE",
        "Add five Bayesian target-statistic features learned ONLY from the",
        "documented external/original source dataset.",
        "",
        "SOURCE FILE",
        str(source_path),
        f"Source shape: {source.shape[0]} x {source.shape[1]}",
        "",
        "HELD FIXED",
        f"- frozen fold SHA256: {fold_hash}",
        "- hierarchical-income XGB champion representation",
        "- exact income + commute nested TE",
        "- hierarchical income TE at 1k/10k/100k",
        "- income digits",
        "- exact income + commute frequency features",
        "- learned nested logistic base margin",
        "- smoothing m=2",
        "- depth-4 XGBoost configuration",
        "- seed 42",
        "- CUDA GPU",
        "",
        "RESULTS",
        f"Current hierarchical-XGB baseline OOF: {baseline_auc:.8f}",
        f"Source-dual-TE candidate OOF: {candidate_auc:.8f}",
        f"Delta vs baseline: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs baseline: {probability_corr:.6f}",
        f"Rank corr vs baseline: {rank_correlation:.6f}",
        f"Runtime: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"baseline={row.baseline_auc:.8f} -> "
            f"candidate={row.candidate_auc:.8f} "
            f"({row.delta_vs_baseline:+.8f})"
        )

    summary_lines.extend(
        [
            "",
            "SOURCE-TE FEATURE IMPORTANCE",
        ]
    )

    for row in source_importance.itertuples():
        summary_lines.append(
            f"{row.feature}: "
            f"{row.mean_gain_importance:.8f}"
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
    print("=" * 96)
    print("SOURCE-DATA DUAL TARGET ENCODING EXPERIMENT COMPLETE")
    print("=" * 96)
    print(f"Current XGB champion : {baseline_auc:.8f}")
    print(f"Candidate            : {candidate_auc:.8f}")
    print(f"Delta                : {delta_vs_baseline:+.8f}")
    print(f"Folds improved       : {folds_improved}/5")
    print(f"Folds worse          : {folds_worse}/5")
    print(f"Probability corr     : {probability_corr:.6f}")
    print(f"Rank corr            : {rank_correlation:.6f}")
    print(f"Decision             : {decision}")
    print(f"Runtime              : {total_seconds:.2f}s")
    print(f"Artifacts            : {args.output_dir.resolve()}")
    print()
    print("Source-TE feature importances:")
    if len(source_importance):
        print(
            source_importance[
                [
                    "feature",
                    "mean_gain_importance",
                ]
            ].to_string(
                index=False
            )
        )
    print("=" * 96)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. source_te_feature_importance.csv")
    print("  5. source_te_diagnostics.csv")


if __name__ == "__main__":
    main()
