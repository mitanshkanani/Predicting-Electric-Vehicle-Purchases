"""
realmlp_frozen5_vectorized_gpu.py

Kaggle Playground Series S6E9
Neural-diversity experiment: RealMLP with the exact frozen five folds.

GOAL
----
Not merely to beat XGBoost standalone.

We want a strong neural OOF prediction stream whose ranking errors differ
meaningfully from the current tree ensemble, so it may add ensemble value.

WHY THIS VERSION
----------------
The previous RealMLP attempt was computationally impractical when run fold by
fold. pytabkit can train multiple CV splits through one vectorized RealMLP
object. We therefore pass the EXACT frozen five validation folds at once.

ONE FIXED MODEL CONFIGURATION. NO HPO.

FEATURE VIEW
------------
Target-free only:
- original raw features
- exact income categorical identity
- exact commute categorical identity
- income categorical buckets: $50/$250/$1k/$10k/$100k
- commute categorical buckets: 1/5/10 km
- income digit decomposition

Deliberately excluded:
- target encodings
- learned logistic margin
- model predictions
- source target statistics

This makes the model family + representation genuinely different from the
current fine-income XGBoost pipeline.

VALIDATION
----------
Frozen fold SHA256:
55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee

Run:
    python realmlp_frozen5_vectorized_gpu.py
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent

EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "realmlp_frozen5_vectorized_gpu"
)

TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

# Fixed, speed-conscious RealMLP recipe.
# No sweep. No tuning loop.
SEED = 42
N_CV = 5
N_EPOCHS = 64
BATCH_SIZE = 4096
PREDICT_BATCH_SIZE = 16384

# Optional OOF references used ONLY for diversity diagnostics after RealMLP
# has been trained. They are never used as training features.
REFERENCE_OOF = {
    "fine_xgb": (
        ROOT
        / "artifacts"
        / "experiments"
        / "xgboost_fine_income_te_gpu"
        / "oof_predictions.csv"
    ),
    "fine_cat": (
        ROOT
        / "artifacts"
        / "experiments"
        / "catboost_fine_income_multiseed_gpu"
        / "best_average_oof_predictions.csv"
    ),
    "lgbm": (
        ROOT
        / "artifacts"
        / "experiments"
        / "lightgbm_engineered_learned_margin_cpu"
        / "oof_predictions.csv"
    ),
    "champion": (
        ROOT
        / "artifacts"
        / "experiments"
        / "blend_fine_income_xgb_validated_submission"
        / "oof_predictions.csv"
    ),
}

REFERENCE_EXPECTED_AUC = {
    "fine_xgb": 0.94606664,
    "fine_cat": 0.94578633,
    "lgbm": 0.94578042,
    "champion": 0.94611253,
}


def ensure_dependencies():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "\nPyTorch is not installed.\n"
            "Install a CUDA-enabled PyTorch build first, then rerun this SAME file.\n"
        ) from exc

    try:
        import pytabkit
    except ImportError:
        print("=" * 96)
        print("pytabkit is not installed.")
        print("Installing pytabkit into the current Python environment...")
        print("=" * 96)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "pytabkit"]
        )
        importlib.invalidate_caches()
        import pytabkit

    from pytabkit import RealMLP_TD_Classifier

    if not torch.cuda.is_available():
        raise SystemExit(
            "\nCUDA is not available to PyTorch.\n"
            "Do NOT run this experiment on CPU; it would be unnecessarily slow.\n"
            f"PyTorch version: {torch.__version__}\n"
        )

    return torch, pytabkit, RealMLP_TD_Classifier


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile_rank(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(x)
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


def canonical_string(series: pd.Series) -> pd.Series:
    """
    Stable train/test categorical string view for a numeric column.
    """
    numeric = pd.to_numeric(series, errors="raise")

    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite values in {series.name}")

    if np.allclose(
        numeric.to_numpy(dtype=np.float64),
        np.round(numeric.to_numpy(dtype=np.float64)),
        rtol=0.0,
        atol=1e-10,
    ):
        return (
            np.round(numeric.to_numpy(dtype=np.float64))
            .astype(np.int64)
            .astype(str)
        )

    return numeric.map(lambda v: format(float(v), ".12g")).astype(str)


def bucket_string(series: pd.Series, width: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)

    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite values in {series.name}")

    bucket = np.floor(numeric / width).astype(np.int64)

    return pd.Series(
        bucket.astype(str),
        index=series.index,
        dtype="object",
    )


def add_income_digits(
    frame: pd.DataFrame,
    income: pd.Series,
) -> None:
    x = (
        pd.to_numeric(income, errors="raise")
        .round()
        .astype(np.int64)
        .to_numpy()
    )

    frame["NN__income_ones"] = x % 10
    frame["NN__income_tens"] = (x // 10) % 10
    frame["NN__income_hundreds"] = (x // 100) % 10
    frame["NN__income_thousands"] = (x // 1_000) % 10
    frame["NN__income_ten_thousands"] = (x // 10_000) % 10
    frame["NN__income_hundred_thousands"] = (x // 100_000) % 10
    frame["NN__income_last2"] = x % 100
    frame["NN__income_last3"] = x % 1_000


def prepare_neural_view(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    target_only = [c for c in train.columns if c not in test.columns]
    if target_only != [TARGET]:
        raise ValueError(
            f"Unexpected target-only columns: {target_only}. "
            f"Expected exactly [{TARGET!r}]."
        )

    raw_features = [c for c in test.columns if c != "id"]

    X_train = train[raw_features].copy()
    X_test = test[raw_features].copy()

    raw_cats = [
        c
        for c in raw_features
        if (
            pd.api.types.is_object_dtype(X_train[c])
            or isinstance(X_train[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(X_train[c])
        )
    ]

    for c in raw_cats:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
        X_test[c] = X_test[c].fillna("__MISSING__").astype(str)

    # Exact numeric identities as categorical copies.
    X_train["NN__income_exact_cat"] = canonical_string(
        train["Annual_Income_USD"]
    )
    X_test["NN__income_exact_cat"] = canonical_string(
        test["Annual_Income_USD"]
    )

    X_train["NN__commute_exact_cat"] = canonical_string(
        train["Daily_Commute_km"]
    )
    X_test["NN__commute_exact_cat"] = canonical_string(
        test["Daily_Commute_km"]
    )

    engineered_cats = [
        "NN__income_exact_cat",
        "NN__commute_exact_cat",
    ]

    # Local + coarse income hierarchy.
    for width, label in [
        (50.0, "50"),
        (250.0, "250"),
        (1_000.0, "1k"),
        (10_000.0, "10k"),
        (100_000.0, "100k"),
    ]:
        col = f"NN__income_{label}_cat"
        X_train[col] = bucket_string(train["Annual_Income_USD"], width)
        X_test[col] = bucket_string(test["Annual_Income_USD"], width)
        engineered_cats.append(col)

    # Commute hierarchy.
    for width, label in [
        (1.0, "1km"),
        (5.0, "5km"),
        (10.0, "10km"),
    ]:
        col = f"NN__commute_{label}_cat"
        X_train[col] = bucket_string(train["Daily_Commute_km"], width)
        X_test[col] = bucket_string(test["Daily_Commute_km"], width)
        engineered_cats.append(col)

    # Target-free numerical digit decomposition.
    add_income_digits(X_train, train["Annual_Income_USD"])
    add_income_digits(X_test, test["Annual_Income_USD"])

    categorical_columns = raw_cats + engineered_cats

    # pytabkit is happiest when categoricals have explicit string/object values.
    for c in categorical_columns:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
        X_test[c] = X_test[c].fillna("__MISSING__").astype(str)

    if list(X_train.columns) != list(X_test.columns):
        raise RuntimeError("Train/test neural feature columns differ.")

    diagnostics = []

    for c in engineered_cats:
        train_unique_values = set(X_train[c].unique().tolist())
        diagnostics.append(
            {
                "feature": c,
                "train_unique": int(X_train[c].nunique()),
                "test_unique": int(X_test[c].nunique()),
                "test_seen_rate": float(
                    X_test[c].isin(train_unique_values).mean()
                ),
            }
        )

    return (
        X_train,
        X_test,
        categorical_columns,
        pd.DataFrame(diagnostics),
    )


def load_folds(
    train: pd.DataFrame,
) -> np.ndarray:
    if sha256_file(FOLDS_PATH) != EXPECTED_FOLDS_SHA256:
        raise RuntimeError("Frozen-fold SHA256 mismatch.")

    folds_df = pd.read_csv(FOLDS_PATH)

    if len(folds_df) != len(train):
        raise RuntimeError("Frozen fold row count mismatch.")

    if "fold" not in folds_df.columns:
        raise RuntimeError("Frozen fold file has no 'fold' column.")

    if "id" in train.columns and "id" in folds_df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            folds_df["id"].to_numpy(),
        ):
            raise RuntimeError("Frozen fold ID order mismatch.")

    folds = folds_df["fold"].to_numpy(dtype=np.int64)

    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected frozen folds: {sorted(np.unique(folds).tolist())}"
        )

    fold_sizes = [int(np.sum(folds == f)) for f in range(5)]

    if len(set(fold_sizes)) != 1:
        raise RuntimeError(
            "pytabkit vectorized custom CV requires equal-length validation "
            f"fold arrays for this script. Got sizes: {fold_sizes}"
        )

    return folds


def normalize_ensemble_probabilities(
    raw: np.ndarray,
    *,
    n_splits: int,
    n_samples: int,
) -> np.ndarray:
    """
    Convert pytabkit predict_proba_ensemble output to:
        [n_splits, n_samples, 2]

    Handles an optional singleton/internal-ensemble dimension by averaging
    all axes other than split/sample/class.
    """
    arr = np.asarray(raw)

    if arr.ndim < 3:
        raise RuntimeError(
            f"Unexpected predict_proba_ensemble shape: {arr.shape}"
        )

    axes = list(range(arr.ndim))

    sample_candidates = [a for a in axes if arr.shape[a] == n_samples]
    class_candidates = [a for a in axes if arr.shape[a] == 2]
    split_candidates = [a for a in axes if arr.shape[a] == n_splits]

    if not sample_candidates:
        raise RuntimeError(
            f"Could not locate sample axis in shape {arr.shape}"
        )
    if not class_candidates:
        raise RuntimeError(
            f"Could not locate binary-class axis in shape {arr.shape}"
        )

    sample_axis = sample_candidates[0]
    class_axis = next(
        (a for a in class_candidates if a != sample_axis),
        None,
    )
    if class_axis is None:
        raise RuntimeError(
            f"Could not uniquely locate class axis in shape {arr.shape}"
        )

    split_axis = next(
        (
            a
            for a in split_candidates
            if a not in {sample_axis, class_axis}
        ),
        None,
    )
    if split_axis is None:
        raise RuntimeError(
            f"Could not uniquely locate CV-split axis in shape {arr.shape}"
        )

    moved = np.moveaxis(
        arr,
        [split_axis, sample_axis, class_axis],
        [0, 1, 2],
    )

    if moved.ndim > 3:
        extra_axes = tuple(range(3, moved.ndim))
        moved = moved.mean(axis=extra_axes)

    if moved.shape != (n_splits, n_samples, 2):
        raise RuntimeError(
            f"Normalized ensemble-probability shape is {moved.shape}, "
            f"expected {(n_splits, n_samples, 2)}"
        )

    return moved.astype(np.float64, copy=False)


def detect_pred_col(df: pd.DataFrame) -> str:
    for c in [
        "oof_prediction",
        "candidate_oof_prediction",
        "fine_xgb_replacement_prediction",
        "prediction",
    ]:
        if c in df.columns:
            return c

    excluded = {"row_index", "fold", "target", "target_encoded", "id"}
    candidates = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot identify reference OOF prediction column: {list(df.columns)}"
        )

    return candidates[0]


def load_reference_oof(
    path: Path,
    n_rows: int,
    expected_auc: float,
    y: np.ndarray,
) -> np.ndarray | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if len(df) != n_rows:
        return None

    col = detect_pred_col(df)
    pred = df[col].to_numpy(dtype=np.float64)

    score = float(roc_auc_score(y, pred))

    if abs(score - expected_auc) > 5e-5:
        print(
            f"[WARN] Skipping reference {path.name}: "
            f"AUC {score:.8f} != expected {expected_auc:.8f}"
        )
        return None

    return pred


def main() -> None:
    torch, pytabkit, RealMLP_TD_Classifier = ensure_dependencies()

    print("=" * 100)
    print("REALMLP FROZEN-5 VECTORIZED GPU — NEURAL DIVERSITY EXPERIMENT")
    print("=" * 100)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"pytabkit        : {getattr(pytabkit, '__version__', 'unknown')}")
    print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
    print(f"Frozen fold SHA : {EXPECTED_FOLDS_SHA256}")
    print()
    print("HYPOTHESIS:")
    print(
        "  A structurally different RealMLP model can provide useful OOF diversity "
        "even if it does not beat the fine-income XGB standalone."
    )
    print()
    print("ONLY MODEL CONFIGURATION:")
    print(f"  n_cv={N_CV}, seed={SEED}, n_epochs={N_EPOCHS}")
    print(f"  batch_size={BATCH_SIZE}, predict_batch_size={PREDICT_BATCH_SIZE}")
    print("  RealMLP tuned-default architecture, n_ens=1")
    print("  validation metric = 1-auc_ovr")
    print("  label smoothing disabled for AUC-oriented validation")
    print()
    print("NO HPO. NO TARGET ENCODING. NO BLEND SEARCH.")
    print()

    for path in [TRAIN_PATH, TEST_PATH, FOLDS_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TARGET not in train.columns:
        raise RuntimeError(f"Missing target {TARGET!r}")

    y = (
        train[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(POSITIVE_LABEL.lower())
        .astype(np.int64)
        .to_numpy()
    )

    folds = load_folds(train)

    X_train, X_test, cat_cols, diagnostics = prepare_neural_view(
        train,
        test,
    )

    print(f"Train rows       : {len(train):,}")
    print(f"Test rows        : {len(test):,}")
    print(f"Neural features  : {X_train.shape[1]}")
    print(f"Categoricals     : {len(cat_cols)}")
    print(f"Positive rate    : {y.mean():.6f}")
    print()

    print("Engineered categorical diagnostics:")
    print(diagnostics.to_string(index=False))
    print()

    # Exact frozen validation arrays.
    val_idxs = np.stack(
        [
            np.flatnonzero(folds == fold)
            for fold in range(N_CV)
        ],
        axis=0,
    )

    print(f"val_idxs shape   : {val_idxs.shape}")
    print(
        "Fold sizes       : "
        + ", ".join(str(len(v)) for v in val_idxs)
    )
    print()

    model = RealMLP_TD_Classifier(
        device="cuda",
        random_state=SEED,
        n_cv=N_CV,
        n_refit=0,
        n_repeats=1,
        n_threads=10,
        verbosity=2,
        val_metric_name="1-auc_ovr",
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE,
        use_ls=False,
        n_ens=1,
    )

    print("=" * 100)
    print("TRAINING REALMLP")
    print("=" * 100)

    start = time.perf_counter()

    model.fit(
        X_train,
        y,
        val_idxs=val_idxs,
        cat_col_names=cat_cols,
    )

    fit_seconds = time.perf_counter() - start

    print()
    print(f"Training complete in {fit_seconds:.2f}s")
    print()
    print("Extracting per-fold OOF probabilities...")

    raw_train_ensemble = model.predict_proba_ensemble(X_train)

    train_ensemble = normalize_ensemble_probabilities(
        raw_train_ensemble,
        n_splits=N_CV,
        n_samples=len(train),
    )

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []

    for fold in range(N_CV):
        idx = val_idxs[fold]
        fold_pred = train_ensemble[fold, idx, 1]
        oof[idx] = fold_pred

        fold_auc = float(
            roc_auc_score(
                y[idx],
                fold_pred,
            )
        )

        fold_rows.append(
            {
                "fold": fold,
                "n": len(idx),
                "auc": fold_auc,
            }
        )

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.8f} | n={len(idx):,}"
        )

    if np.isnan(oof).any():
        raise RuntimeError("RealMLP OOF contains NaNs.")

    oof_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    print()
    print("Predicting test...")

    test_proba = np.asarray(
        model.predict_proba(X_test),
        dtype=np.float64,
    )

    if (
        test_proba.ndim != 2
        or test_proba.shape[0] != len(test)
        or test_proba.shape[1] != 2
    ):
        raise RuntimeError(
            f"Unexpected test predict_proba shape: {test_proba.shape}"
        )

    test_pred = test_proba[:, 1]

    if not np.isfinite(test_pred).all():
        raise RuntimeError("Non-finite RealMLP test predictions.")

    # Diversity diagnostics against our validated tree members.
    correlation_rows = []

    print()
    print("--- Diversity diagnostics ---")

    for name, path in REFERENCE_OOF.items():
        ref = load_reference_oof(
            path,
            len(train),
            REFERENCE_EXPECTED_AUC[name],
            y,
        )

        if ref is None:
            print(f"{name:10s}: reference artifact unavailable/skipped")
            continue

        prob_corr = float(
            np.corrcoef(
                oof,
                ref,
            )[0, 1]
        )
        r_corr = rank_corr(
            oof,
            ref,
        )

        correlation_rows.append(
            {
                "reference": name,
                "reference_auc": REFERENCE_EXPECTED_AUC[name],
                "probability_corr": prob_corr,
                "rank_corr": r_corr,
            }
        )

        print(
            f"{name:10s}: "
            f"prob_corr={prob_corr:.6f} | "
            f"rank_corr={r_corr:.6f}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    diagnostics.to_csv(
        OUTPUT_DIR / "feature_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        correlation_rows
    ).to_csv(
        OUTPUT_DIR / "diversity_correlations.csv",
        index=False,
    )

    oof_df = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": folds,
            "target_encoded": y,
            "oof_prediction": oof.astype(
                np.float32
            ),
        }
    )

    if "id" in train.columns:
        oof_df.insert(
            1,
            "id",
            train["id"].to_numpy(),
        )

    oof_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False,
    )

    test_df = pd.DataFrame(
        {
            "prediction": test_pred.astype(
                np.float32
            )
        }
    )

    if "id" in test.columns:
        test_df.insert(
            0,
            "id",
            test["id"].to_numpy(),
        )

    test_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    summary_lines = [
        "EXPERIMENT: REALMLP FROZEN-5 VECTORIZED GPU",
        "=" * 88,
        "",
        "GOAL",
        "Create a genuinely different neural OOF member for ensemble diversity.",
        "",
        "VALIDATION",
        f"Frozen fold SHA256: {EXPECTED_FOLDS_SHA256}",
        f"n_cv: {N_CV}",
        "",
        "FIXED MODEL",
        f"seed: {SEED}",
        f"n_epochs: {N_EPOCHS}",
        f"batch_size: {BATCH_SIZE}",
        f"predict_batch_size: {PREDICT_BATCH_SIZE}",
        "n_ens: 1",
        "val_metric_name: 1-auc_ovr",
        "use_ls: False",
        "",
        "FEATURE VIEW",
        f"total features: {X_train.shape[1]}",
        f"categorical features: {len(cat_cols)}",
        "target encodings: NONE",
        "model prediction features: NONE",
        "",
        "RESULT",
        f"RealMLP OOF AUC: {oof_auc:.8f}",
        f"Training seconds: {fit_seconds:.2f}",
        "",
        "FOLD AUC",
    ]

    for row in fold_rows:
        summary_lines.append(
            f"Fold {row['fold']}: {row['auc']:.8f}"
        )

    summary_lines.extend(
        [
            "",
            "DIVERSITY CORRELATIONS",
        ]
    )

    for row in correlation_rows:
        summary_lines.append(
            f"{row['reference']}: "
            f"prob_corr={row['probability_corr']:.6f}, "
            f"rank_corr={row['rank_corr']:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "IMPORTANT",
            "Do not KEEP/REJECT from standalone AUC alone.",
            "Manual decision must consider both AUC and correlation/ensemble value.",
        ]
    )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("REALMLP EXPERIMENT COMPLETE")
    print("=" * 100)
    print(f"OOF AUC             : {oof_auc:.8f}")
    print(f"Training time       : {fit_seconds:.2f}s")

    if correlation_rows:
        print()
        for row in correlation_rows:
            print(
                f"Rank corr vs {row['reference']:8s}: "
                f"{row['rank_corr']:.6f}"
            )

    print(f"Artifacts           : {OUTPUT_DIR}")
    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. diversity_correlations.csv")


if __name__ == "__main__":
    main()
