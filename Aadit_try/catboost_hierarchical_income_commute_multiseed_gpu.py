"""
catboost_hierarchical_income_commute_multiseed_gpu.py

Kaggle Playground Series S6E9

Controlled experiment:
Test whether the validated hierarchical-income+commute CatBoost representation
benefits from multi-seed averaging.

VALIDATED SEED-42 RESULT
------------------------
Hierarchical-income CatBoost seed42:
    0.94546676

Hierarchical-income+commute CatBoost seed42:
    0.94552424

Delta:
    +0.00005748

Folds improved:
    5/5

ONLY CHANGE
-----------
Random-seed diversity:
- reuse existing seed 42 predictions
- train seed 7
- train seed 2026
- evaluate 3-seed probability average
- evaluate 3-seed rank average

HELD FIXED
----------
- frozen 5-fold assignment + SHA256
- raw 13 features
- raw income + commute numeric columns
- exact income + commute categorical IDs
- hierarchical income categorical buckets
- hierarchical commute categorical buckets
- iterations=2000
- learning_rate=0.05
- depth=6
- loss_function="Logloss"
- eval_metric="AUC"
- task_type="GPU"
- devices="0"
- early_stopping_rounds=150
- all unspecified CatBoost parameters remain defaults
- no public leaderboard optimization

Seed 42 is NOT retrained.

Run:
    python catboost_hierarchical_income_commute_multiseed_gpu.py
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
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise SystemExit(
        "\nCatBoost is not installed.\n"
        "Install/update it with:\n"
        "    python -m pip install -U catboost\n"
    ) from exc

try:
    import catboost_hierarchical_income_buckets_gpu as income_base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import catboost_hierarchical_income_buckets_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc

try:
    import catboost_hierarchical_income_commute_buckets_gpu as commute_base
except ImportError as exc:
    raise SystemExit(
        "\nCould not import catboost_hierarchical_income_commute_buckets_gpu.py.\n"
        "Place this file in the same repo root as that validated script.\n"
    ) from exc


EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED_SEED42_AUC = 0.94552424
EXPECTED_PREVIOUS_CAT3_AUC = 0.94553699
AUC_CHECK_TOLERANCE = 2e-5

TRAIN_SEEDS = [7, 2026]
ALL_SEEDS = [42, 7, 2026]

DEFAULT_FOLDS_PATH = Path("artifacts") / "validation" / "candidate_folds.csv"
DEFAULT_SEED42_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_buckets_gpu"
    / "oof_predictions.csv"
)
DEFAULT_SEED42_TEST = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_buckets_gpu"
    / "test_predictions.csv"
)
DEFAULT_PREVIOUS_CAT3_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed audit for hierarchical-income+commute CatBoost."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)
    p.add_argument("--seed42-oof", type=Path, default=DEFAULT_SEED42_OOF)
    p.add_argument("--seed42-test", type=Path, default=DEFAULT_SEED42_TEST)
    p.add_argument("--previous-cat3-oof", type=Path, default=DEFAULT_PREVIOUS_CAT3_OOF)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_prediction_column(df: pd.DataFrame, path: Path, kind: str) -> str:
    if kind == "oof":
        preferred = [
            "oof_prediction",
            "prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {"row_index", "fold", "target", "target_encoded", "id"}
    elif kind == "test":
        preferred = [
            "prediction",
            "test_prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
        excluded = {"row_index", "fold", "id"}
    else:
        raise ValueError(f"Unknown kind: {kind}")

    for c in preferred:
        if c in df.columns:
            return c

    numeric = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) == 1:
        return numeric[0]

    raise ValueError(
        f"Could not safely identify {kind} prediction column in:\n"
        f"{path.resolve()}\nColumns: {list(df.columns)}"
    )


def load_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"OOF file not found:\n{path.resolve()}")

    df = pd.read_csv(path)

    if len(df) != len(train):
        raise ValueError(f"OOF row count mismatch in {path}")

    if "row_index" not in df.columns:
        raise ValueError(f"OOF row_index missing in {path}")

    if not np.array_equal(
        df["row_index"].to_numpy(),
        np.arange(len(train), dtype=np.int64),
    ):
        raise ValueError(f"OOF row order mismatch in {path}")

    if "fold" not in df.columns:
        raise ValueError(f"OOF fold missing in {path}")

    if not np.array_equal(df["fold"].to_numpy(), fold_ids):
        raise ValueError(f"OOF fold mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if not np.array_equal(df[id_col].to_numpy(), train[id_col].to_numpy()):
            raise ValueError(f"OOF ID mismatch in {path}")

    pred_col = pick_prediction_column(df, path, "oof")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite OOF predictions in {path}")

    return pred


def load_test(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Test prediction file not found:\n{path.resolve()}")

    df = pd.read_csv(path)

    if len(df) != len(test):
        raise ValueError(f"Test row count mismatch in {path}")

    if id_col is not None and id_col in df.columns:
        if not np.array_equal(df[id_col].to_numpy(), test[id_col].to_numpy()):
            raise ValueError(f"Test ID mismatch in {path}")

    pred_col = pick_prediction_column(df, path, "test")
    pred = df[pred_col].to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite test predictions in {path}")

    return pred


def verify_auc(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    expected: float,
) -> float:
    auc = float(roc_auc_score(y, pred))
    if abs(auc - expected) > AUC_CHECK_TOLERANCE:
        raise ValueError(
            f"{name} OOF AUC mismatch.\n"
            f"Expected approximately: {expected:.8f}\n"
            f"Loaded artifact AUC   : {auc:.8f}"
        )
    return auc


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.corrcoef(
            percentile_rank(a),
            percentile_rank(b),
        )[0, 1]
    )


def build_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
    )


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_features = [
        c for c in test.columns
        if c in train.columns and c != id_col
    ]

    raw_categoricals = income_base.detect_raw_categoricals(train, raw_features)

    X_train = train[raw_features].copy()
    X_test = test[raw_features].copy()

    for c in raw_categoricals:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
        X_test[c] = X_test[c].fillna("__MISSING__").astype(str)

    X_train, X_test, exact_id_columns = income_base.add_exact_value_ids(
        X_train=X_train,
        X_test=X_test,
        train_source=train,
        test_source=test,
    )

    X_train, X_test, income_bucket_columns = (
        income_base.add_hierarchical_income_categories(
            X_train=X_train,
            X_test=X_test,
            train_source=train,
            test_source=test,
        )
    )

    X_train, X_test, commute_bucket_columns = (
        commute_base.add_hierarchical_commute_categories(
            X_train=X_train,
            X_test=X_test,
            train_source=train,
            test_source=test,
        )
    )

    categorical_columns = (
        raw_categoricals
        + exact_id_columns
        + income_bucket_columns
        + commute_bucket_columns
    )

    if len(set(categorical_columns)) != len(categorical_columns):
        raise ValueError("Duplicate categorical feature names detected.")

    return X_train, X_test, categorical_columns


def save_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    id_col: str | None,
    y: np.ndarray,
    pred: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold_ids,
            "target_encoded": y,
            "oof_prediction": pred.astype(np.float32),
        }
    )

    if id_col is not None:
        df.insert(1, id_col, train[id_col].to_numpy())

    df.to_csv(path, index=False)


def save_test(
    path: Path,
    test: pd.DataFrame,
    id_col: str | None,
    pred: np.ndarray,
) -> None:
    df = pd.DataFrame({"prediction": pred.astype(np.float32)})

    if id_col is not None:
        df.insert(0, id_col, test[id_col].to_numpy())

    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    for path in [
        train_path,
        test_path,
        args.folds_path,
        args.seed42_oof,
        args.seed42_test,
        args.previous_cat3_oof,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path.resolve()}")

    fold_hash = sha256_file(args.folds_path)

    if fold_hash != EXPECTED_FOLDS_SHA256:
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Found   : {fold_hash}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds_df = pd.read_csv(args.folds_path)

    target = income_base.detect_target(train, test)
    y, positive_label = income_base.encode_binary_target(train[target])
    fold_ids, id_col = income_base.validate_folds(folds_df, train)

    seed42_oof = load_oof(
        args.seed42_oof,
        train,
        fold_ids,
        id_col,
    )
    seed42_test = load_test(
        args.seed42_test,
        test,
        id_col,
    )

    seed42_auc = verify_auc(
        "Hierarchical income+commute CatBoost seed42",
        y,
        seed42_oof,
        EXPECTED_SEED42_AUC,
    )

    previous_cat3_oof = load_oof(
        args.previous_cat3_oof,
        train,
        fold_ids,
        id_col,
    )
    previous_cat3_auc = verify_auc(
        "Previous hierarchical-income CatBoost 3-seed",
        y,
        previous_cat3_oof,
        EXPECTED_PREVIOUS_CAT3_AUC,
    )

    X_train, X_test, categorical_columns = prepare_frames(
        train,
        test,
        id_col,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CATBOOST HIERARCHICAL-INCOME+COMMUTE MULTI-SEED AUDIT")
    print("=" * 100)
    print(f"Target: {target!r} | positive label: {positive_label!r}")
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print(f"New representation seed42: {seed42_auc:.8f}")
    print(f"Previous hierarchical-income Cat3: {previous_cat3_auc:.8f}")
    print()
    print("ONLY CHANGE:")
    print("  Reuse seed 42, train seeds 7 and 2026, then compare averages.")
    print()
    print("Training only seeds: [7, 2026]")
    print()

    seed_oof = {42: seed42_oof}
    seed_test = {42: seed42_test}
    seed_auc = {42: seed42_auc}
    fold_rows = []

    total_start = time.perf_counter()

    for seed in TRAIN_SEEDS:
        print("=" * 100)
        print(f"TRAINING SEED {seed}")
        print("=" * 100)

        oof = np.full(len(train), np.nan, dtype=np.float64)
        test_fold_predictions = []

        for fold in range(5):
            fold_start = time.perf_counter()

            train_idx = np.flatnonzero(fold_ids != fold)
            valid_idx = np.flatnonzero(fold_ids == fold)

            model = build_model(seed)

            model.fit(
                X_train.iloc[train_idx],
                y[train_idx],
                cat_features=categorical_columns,
                eval_set=(
                    X_train.iloc[valid_idx],
                    y[valid_idx],
                ),
                use_best_model=True,
                early_stopping_rounds=150,
                verbose=100,
            )

            valid_pred = model.predict_proba(
                X_train.iloc[valid_idx]
            )[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]

            oof[valid_idx] = valid_pred
            test_fold_predictions.append(
                test_pred.astype(np.float32)
            )

            fold_auc = float(
                roc_auc_score(
                    y[valid_idx],
                    valid_pred,
                )
            )
            seed42_fold_auc = float(
                roc_auc_score(
                    y[valid_idx],
                    seed42_oof[valid_idx],
                )
            )

            best_iteration = int(model.get_best_iteration())
            fold_seconds = time.perf_counter() - fold_start

            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "seed42_auc": seed42_fold_auc,
                    "seed_auc": fold_auc,
                    "delta_vs_seed42": fold_auc - seed42_fold_auc,
                    "best_iteration": best_iteration,
                    "total_fold_seconds": fold_seconds,
                }
            )

            print(
                f"Seed {seed} fold {fold}: "
                f"seed42={seed42_fold_auc:.8f} -> "
                f"seed{seed}={fold_auc:.8f} "
                f"({fold_auc - seed42_fold_auc:+.8f}) | "
                f"best_iter={best_iteration}"
            )

            del model

        if np.isnan(oof).any():
            raise RuntimeError(f"Seed {seed} OOF contains NaNs.")

        test_prediction = np.mean(
            np.vstack(test_fold_predictions),
            axis=0,
        )

        auc = float(roc_auc_score(y, oof))

        seed_oof[seed] = oof
        seed_test[seed] = test_prediction
        seed_auc[seed] = auc

        save_oof(
            args.output_dir / f"oof_predictions_seed{seed}.csv",
            train,
            fold_ids,
            id_col,
            y,
            oof,
        )
        save_test(
            args.output_dir / f"test_predictions_seed{seed}.csv",
            test,
            id_col,
            test_prediction,
        )

        print()
        print(f"Seed {seed} overall OOF AUC: {auc:.8f}")
        print()

    total_seconds = time.perf_counter() - total_start

    oof_stack = np.vstack([seed_oof[s] for s in ALL_SEEDS])
    test_stack = np.vstack([seed_test[s] for s in ALL_SEEDS])

    probability_average_oof = np.mean(oof_stack, axis=0)
    probability_average_test = np.mean(test_stack, axis=0)
    probability_average_auc = float(
        roc_auc_score(y, probability_average_oof)
    )

    rank_oof_stack = np.vstack(
        [percentile_rank(seed_oof[s]) for s in ALL_SEEDS]
    )
    rank_test_stack = np.vstack(
        [percentile_rank(seed_test[s]) for s in ALL_SEEDS]
    )

    rank_average_oof = np.mean(rank_oof_stack, axis=0)
    rank_average_test = np.mean(rank_test_stack, axis=0)
    rank_average_auc = float(
        roc_auc_score(y, rank_average_oof)
    )

    if rank_average_auc >= probability_average_auc:
        best_method = "rank_average"
        best_oof = rank_average_oof
        best_test = rank_average_test
        best_auc = rank_average_auc
    else:
        best_method = "probability_average"
        best_oof = probability_average_oof
        best_test = probability_average_test
        best_auc = probability_average_auc

    delta_vs_seed42 = best_auc - seed42_auc
    delta_vs_previous_cat3 = best_auc - previous_cat3_auc

    ensemble_fold_rows = []
    folds_improved_vs_seed42 = 0

    for fold in range(5):
        mask = fold_ids == fold

        seed42_fold_auc = float(
            roc_auc_score(
                y[mask],
                seed42_oof[mask],
            )
        )
        ensemble_fold_auc = float(
            roc_auc_score(
                y[mask],
                best_oof[mask],
            )
        )

        delta = ensemble_fold_auc - seed42_fold_auc

        if delta > 0:
            folds_improved_vs_seed42 += 1

        ensemble_fold_rows.append(
            {
                "fold": fold,
                "seed42_auc": seed42_fold_auc,
                "best_ensemble_auc": ensemble_fold_auc,
                "delta_vs_seed42": delta,
            }
        )

    ensemble_fold_metrics = pd.DataFrame(
        ensemble_fold_rows
    )

    pair_rows = []

    for i, seed_a in enumerate(ALL_SEEDS):
        for seed_b in ALL_SEEDS[i + 1:]:
            pair_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "probability_corr": float(
                        np.corrcoef(
                            seed_oof[seed_a],
                            seed_oof[seed_b],
                        )[0, 1]
                    ),
                    "rank_corr": rank_corr(
                        seed_oof[seed_a],
                        seed_oof[seed_b],
                    ),
                }
            )

    seed_pair_correlations = pd.DataFrame(pair_rows)

    if delta_vs_seed42 > 0 and folds_improved_vs_seed42 >= 3:
        decision = "KEEP_HIERARCHICAL_INCOME_COMMUTE_CATBOOST_3SEED"
    else:
        decision = "REJECT_HIERARCHICAL_INCOME_COMMUTE_CATBOOST_3SEED"

    pd.DataFrame(fold_rows).to_csv(
        args.output_dir / "seed_fold_metrics.csv",
        index=False,
    )
    ensemble_fold_metrics.to_csv(
        args.output_dir / "ensemble_fold_metrics.csv",
        index=False,
    )
    seed_pair_correlations.to_csv(
        args.output_dir / "seed_pair_correlations.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"model": "seed_42", "oof_auc": seed_auc[42]},
            {"model": "seed_7", "oof_auc": seed_auc[7]},
            {"model": "seed_2026", "oof_auc": seed_auc[2026]},
            {"model": "probability_average", "oof_auc": probability_average_auc},
            {"model": "rank_average", "oof_auc": rank_average_auc},
        ]
    ).to_csv(
        args.output_dir / "seed_scores.csv",
        index=False,
    )

    save_oof(
        args.output_dir / "probability_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        probability_average_oof,
    )
    save_test(
        args.output_dir / "probability_average_test_predictions.csv",
        test,
        id_col,
        probability_average_test,
    )

    save_oof(
        args.output_dir / "rank_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        rank_average_oof,
    )
    save_test(
        args.output_dir / "rank_average_test_predictions.csv",
        test,
        id_col,
        rank_average_test,
    )

    save_oof(
        args.output_dir / "best_average_oof_predictions.csv",
        train,
        fold_ids,
        id_col,
        y,
        best_oof,
    )
    save_test(
        args.output_dir / "best_average_test_predictions.csv",
        test,
        id_col,
        best_test,
    )

    summary_lines = [
        "EXPERIMENT: HIERARCHICAL-INCOME+COMMUTE CATBOOST MULTI-SEED AUDIT",
        "=" * 90,
        "",
        "ONLY CHANGE",
        "Random-seed diversity. Reuse seed 42, train seeds 7 and 2026.",
        "",
        "SINGLE-SEED RESULTS",
        f"Seed 42: {seed_auc[42]:.8f}",
        f"Seed 7: {seed_auc[7]:.8f}",
        f"Seed 2026: {seed_auc[2026]:.8f}",
        "",
        "ENSEMBLE RESULTS",
        f"3-seed probability average: {probability_average_auc:.8f}",
        f"3-seed rank average: {rank_average_auc:.8f}",
        f"Best averaging method: {best_method}",
        f"Best 3-seed AUC: {best_auc:.8f}",
        f"Delta vs new representation seed42: {delta_vs_seed42:+.8f}",
        f"Delta vs previous hierarchical-income Cat3: {delta_vs_previous_cat3:+.8f}",
        f"Best ensemble folds improved vs seed42: {folds_improved_vs_seed42}/5",
        f"Runtime for newly trained seeds: {total_seconds:.2f} seconds",
        "",
        f"DECISION: {decision}",
        "",
        "BEST-ENSEMBLE FOLD RESULTS",
    ]

    for row in ensemble_fold_metrics.itertuples():
        summary_lines.append(
            f"Fold {row.fold}: "
            f"seed42={row.seed42_auc:.8f} -> "
            f"best_ensemble={row.best_ensemble_auc:.8f} "
            f"({row.delta_vs_seed42:+.8f})"
        )

    summary_lines.extend(["", "SEED PAIR CORRELATIONS"])

    for row in seed_pair_correlations.itertuples():
        summary_lines.append(
            f"{row.seed_a} vs {row.seed_b}: "
            f"prob_corr={row.probability_corr:.6f}, "
            f"rank_corr={row.rank_corr:.6f}"
        )

    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("HIERARCHICAL-INCOME+COMMUTE CATBOOST MULTI-SEED AUDIT COMPLETE")
    print("=" * 100)
    print(f"Seed 42             : {seed_auc[42]:.8f}")
    print(f"Seed 7              : {seed_auc[7]:.8f}")
    print(f"Seed 2026           : {seed_auc[2026]:.8f}")
    print(f"Probability average : {probability_average_auc:.8f}")
    print(f"Rank average        : {rank_average_auc:.8f}")
    print(f"Best method         : {best_method}")
    print(f"Best 3-seed AUC     : {best_auc:.8f}")
    print(f"Delta vs seed42     : {delta_vs_seed42:+.8f}")
    print(f"Delta vs prev Cat3  : {delta_vs_previous_cat3:+.8f}")
    print(f"Folds improved vs seed42: {folds_improved_vs_seed42}/5")
    print(f"Decision            : {decision}")
    print(f"Artifacts           : {args.output_dir.resolve()}")
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. seed_scores.csv")
    print("  4. ensemble_fold_metrics.csv")
    print("  5. seed_pair_correlations.csv")


if __name__ == "__main__":
    main()
