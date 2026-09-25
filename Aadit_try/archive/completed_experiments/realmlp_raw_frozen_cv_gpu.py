from __future__ import annotations

import hashlib
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

RAW_XGB_REFERENCE_AUC = 0.94206547
CURRENT_XGB_REFERENCE_AUC = 0.94591178
CURRENT_ENSEMBLE_REFERENCE_AUC = 0.94596441

FOLDS_PATH = Path("artifacts/validation/candidate_folds.csv")

CURRENT_CAT_OOF = (
    Path("artifacts")
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu"
    / "best_average_oof_predictions.csv"
)

CURRENT_XGB_OOF = (
    Path("artifacts")
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu"
    / "oof_predictions.csv"
)

CURRENT_LGBM_OOF = (
    Path("artifacts")
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu"
    / "oof_predictions.csv"
)

CURRENT_ENSEMBLE_OOF = (
    Path("artifacts")
    / "experiments"
    / "blend_income_commute_catboost_commute_xgb_lgbm_rank_audit"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    Path("artifacts")
    / "experiments"
    / "realmlp_raw_frozen_cv_gpu"
)



class TrainingHeartbeat:
    """Print a non-invasive heartbeat while a RealMLP fold is training."""

    def __init__(
        self,
        torch_module,
        fold: int,
        interval_seconds: int = 30,
    ) -> None:
        self.torch = torch_module
        self.fold = fold
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _gpu_memory_text(self) -> str:
        if not self.torch.cuda.is_available():
            return "GPU memory unavailable"

        allocated = (
            self.torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )
        reserved = (
            self.torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        return (
            f"GPU mem allocated={allocated:.2f} GiB | "
            f"reserved={reserved:.2f} GiB"
        )

    def _run(self) -> None:
        while not self._stop_event.wait(
            self.interval_seconds
        ):
            elapsed = (
                time.perf_counter()
                - self._start_time
            )

            print(
                f"[HEARTBEAT] Fold {self.fold} still training | "
                f"elapsed={self._format_elapsed(elapsed)} | "
                f"{self._gpu_memory_text()}",
                flush=True,
            )

    def start(self) -> None:
        self._start_time = time.perf_counter()

        print(
            f"[HEARTBEAT] Fold {self.fold} training started. "
            f"Status will print every {self.interval_seconds}s.",
            flush=True,
        )

        self._thread = threading.Thread(
            target=self._run,
            name=f"realmlp-heartbeat-fold-{self.fold}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=2.0
            )

        elapsed = (
            time.perf_counter()
            - self._start_time
        )

        print(
            f"[HEARTBEAT] Fold {self.fold} training call finished | "
            f"elapsed={self._format_elapsed(elapsed)} | "
            f"{self._gpu_memory_text()}",
            flush=True,
        )


def require_realmlp():
    try:
        import torch
        import pytabkit
        from pytabkit import RealMLP_TD_Classifier
    except ImportError as exc:
        raise SystemExit(
            "\nRealMLP dependencies are not available in this Python environment.\n\n"
            "Install the Python-3.13-compatible PyTabKit release with:\n\n"
            f'    & "{sys.executable}" -m pip install -U "pytabkit>=1.6.1"\n\n'
            "Then run THIS SAME FILE again.\n"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "\nPyTorch cannot see a CUDA GPU in this Python environment.\n"
            "This experiment is intentionally GPU-only because a 5-fold "
            "RealMLP run on CPU would be unnecessarily slow.\n"
        )

    return torch, pytabkit, RealMLP_TD_Classifier


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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
            f"Expected one train-only target column, found: {train_only}"
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
            f"Expected binary target, found: {values}"
        )

    for value in values:
        if str(value).strip().lower() == "yes":
            positive = value
            break
    else:
        positive = y.value_counts().idxmin()

    encoded = (
        (y == positive)
        .astype(np.int64)
        .to_numpy()
    )

    return encoded, positive


def validate_folds(
    folds: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, str | None]:
    required = {
        "row_index",
        "fold",
    }

    missing = required - set(
        folds.columns
    )

    if missing:
        raise ValueError(
            f"Frozen folds missing columns: {sorted(missing)}"
        )

    if len(folds) != len(train):
        raise ValueError(
            "Frozen fold row count does not match train.csv."
        )

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        np.arange(
            len(train),
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Frozen fold row order does not match train.csv."
        )

    unique_folds = sorted(
        folds["fold"]
        .unique()
        .tolist()
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
        if c not in {
            "row_index",
            "fold",
        }
    ]

    if len(extras) > 1:
        raise ValueError(
            f"Unexpected extra frozen-fold columns: {extras}"
        )

    id_col = (
        extras[0]
        if extras
        else None
    )

    if id_col is not None:
        if id_col not in train.columns:
            raise ValueError(
                f"Frozen fold ID column {id_col!r} missing from train."
            )

        if not np.array_equal(
            folds[id_col].to_numpy(),
            train[id_col].to_numpy(),
        ):
            raise ValueError(
                "Frozen fold IDs do not align with train.csv."
            )

    return (
        folds["fold"].to_numpy(
            dtype=np.int16
        ),
        id_col,
    )


def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:
    return (
        pd.Series(values)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


def rank_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            percentile_rank(a),
            percentile_rank(b),
        )[0, 1]
    )


def find_prediction_column(
    df: pd.DataFrame,
    preferred: list[str],
) -> str:
    for c in preferred:
        if c in df.columns:
            return c

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "id",
    }

    numeric = [
        c
        for c in df.columns
        if (
            c not in excluded
            and pd.api.types.is_numeric_dtype(
                df[c]
            )
        )
    ]

    if len(numeric) != 1:
        raise ValueError(
            "Could not safely identify prediction column.\n"
            f"Columns: {list(df.columns)}"
        )

    return numeric[0]


def load_reference_oof(
    path: Path,
    train: pd.DataFrame,
    fold_ids: np.ndarray,
    preferred_columns: list[str],
) -> np.ndarray | None:
    if not path.exists():
        return None

    df = pd.read_csv(
        path
    )

    if len(df) != len(train):
        raise ValueError(
            f"Reference OOF row mismatch: {path}"
        )

    if "row_index" in df.columns:
        if not np.array_equal(
            df["row_index"].to_numpy(),
            np.arange(
                len(train),
                dtype=np.int64,
            ),
        ):
            raise ValueError(
                f"Reference OOF row order mismatch: {path}"
            )

    if "fold" in df.columns:
        if not np.array_equal(
            df["fold"].to_numpy(),
            fold_ids,
        ):
            raise ValueError(
                f"Reference OOF fold mismatch: {path}"
            )

    col = find_prediction_column(
        df,
        preferred_columns,
    )

    pred = df[col].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"Reference OOF has non-finite values: {path}"
        )

    return pred


def prepare_raw_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_col: str | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    drop_cols = {
        target,
    }

    if id_col is not None:
        drop_cols.add(
            id_col
        )

    feature_cols = [
        c
        for c in train.columns
        if c not in drop_cols
    ]

    if set(
        feature_cols
    ) != set(
        c
        for c in test.columns
        if c != id_col
    ):
        raise ValueError(
            "Train/test raw feature columns do not align."
        )

    X_train = (
        train[
            feature_cols
        ]
        .copy()
    )

    X_test = (
        test[
            feature_cols
        ]
        .copy()
    )

    categorical_cols = []

    for c in feature_cols:
        if (
            pd.api.types.is_object_dtype(
                X_train[c]
            )
            or pd.api.types.is_string_dtype(
                X_train[c]
            )
            or isinstance(
                X_train[c].dtype,
                pd.CategoricalDtype,
            )
        ):
            categorical_cols.append(
                c
            )

            X_train[c] = (
                X_train[c]
                .astype(str)
            )

            X_test[c] = (
                X_test[c]
                .astype(str)
            )
        else:
            X_train[c] = pd.to_numeric(
                X_train[c],
                errors="raise",
            )

            X_test[c] = pd.to_numeric(
                X_test[c],
                errors="raise",
            )

    if (
        X_train.isna().any().any()
        or X_test.isna().any().any()
    ):
        raise ValueError(
            "RealMLP raw baseline expects no missing values."
        )

    return (
        X_train,
        X_test,
        categorical_cols,
    )


def make_model(
    RealMLP_TD_Classifier,
):
    return RealMLP_TD_Classifier(
        device="cuda",
        random_state=42,
        n_cv=1,
        n_refit=0,
        n_ens=1,
        val_metric_name="1-auc_ovr",
        use_ls=False,
        verbosity=1,
    )


def main() -> None:
    torch, pytabkit, RealMLP_TD_Classifier = (
        require_realmlp()
    )

    train_path = Path(
        "data/train.csv"
    )

    test_path = Path(
        "data/test.csv"
    )

    for path in [
        train_path,
        test_path,
        FOLDS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = sha256_file(
        FOLDS_PATH
    )

    if (
        fold_hash
        != EXPECTED_FOLD_SHA256
    ):
        raise ValueError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found   : {fold_hash}"
        )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    target = detect_target(
        train,
        test,
    )

    y, positive_label = (
        encode_binary_target(
            train[target]
        )
    )

    fold_ids, id_col = (
        validate_folds(
            folds_df,
            train,
        )
    )

    (
        X,
        X_test,
        categorical_cols,
    ) = prepare_raw_features(
        train,
        test,
        target,
        id_col,
    )

    print("=" * 100)
    print("REALMLP RAW-FEATURE FROZEN 5-FOLD CV")
    print("=" * 100)
    print(
        f"Python: {platform.python_version()}"
    )
    print(
        f"PyTorch: {torch.__version__}"
    )
    print(
        f"PyTabKit: "
        f"{getattr(pytabkit, '__version__', 'unknown')}"
    )
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Target: {target!r} | "
        f"positive label: {positive_label!r}"
    )
    print(
        f"Frozen fold SHA256 verified: "
        f"{fold_hash}"
    )
    print(
        f"Raw features: {X.shape[1]}"
    )
    print(
        f"Categorical features: "
        f"{len(categorical_cols)}"
    )
    print(
        "Categoricals: "
        + ", ".join(
            categorical_cols
        )
    )
    print()
    print("HYPOTHESIS:")
    print(
        "  A neural tabular model may provide a different ranking error "
        "structure from our tree ensemble."
    )
    print()
    print("ONLY CHANGE:")
    print(
        "  Model family -> RealMLP. Representation remains RAW features only."
    )
    print()
    print("REALMLP CONFIG:")
    print("  device=cuda")
    print("  random_state=42")
    print("  n_cv=1")
    print("  n_refit=0")
    print("  n_ens=1")
    print("  val_metric_name=1-auc_ovr")
    print("  use_ls=False")
    print("  terminal heartbeat=every 30 seconds (logging only)")
    print()
    print(
        f"Raw XGB reference       : "
        f"{RAW_XGB_REFERENCE_AUC:.8f}"
    )
    print(
        f"Current engineered XGB  : "
        f"{CURRENT_XGB_REFERENCE_AUC:.8f}"
    )
    print(
        f"Current ensemble meta-CV: "
        f"{CURRENT_ENSEMBLE_REFERENCE_AUC:.8f}"
    )
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    test_fold_predictions = []
    fold_rows = []

    total_start = (
        time.perf_counter()
    )

    for fold in range(5):
        fold_start = (
            time.perf_counter()
        )

        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        valid_idx = np.flatnonzero(
            fold_ids == fold
        )

        X_train_fold = (
            X.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        y_train_fold = (
            y[
                train_idx
            ]
        )

        X_valid_fold = (
            X.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        y_valid_fold = (
            y[
                valid_idx
            ]
        )

        print()
        print(
            f"FOLD {fold} | "
            f"train={len(train_idx):,} | "
            f"valid={len(valid_idx):,}"
        )

        model = make_model(
            RealMLP_TD_Classifier
        )

        heartbeat = TrainingHeartbeat(
            torch_module=torch,
            fold=fold,
            interval_seconds=30,
        )

        heartbeat.start()

        try:
            model.fit(
                X_train_fold,
                y_train_fold,
                X_val=X_valid_fold,
                y_val=y_valid_fold,
                cat_col_names=categorical_cols,
            )
        finally:
            heartbeat.stop()

        valid_pred = (
            model.predict_proba(
                X_valid_fold
            )[:, 1]
        )

        test_pred = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        oof[
            valid_idx
        ] = valid_pred

        test_fold_predictions.append(
            np.asarray(
                test_pred,
                dtype=np.float32,
            )
        )

        fold_auc = float(
            roc_auc_score(
                y_valid_fold,
                valid_pred,
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
                "train_rows": len(
                    train_idx
                ),
                "valid_rows": len(
                    valid_idx
                ),
                "runtime_seconds": fold_seconds,
            }
        )

        print(
            f"Fold {fold} AUC: "
            f"{fold_auc:.8f} | "
            f"runtime={fold_seconds:.1f}s"
        )

        del model

        torch.cuda.empty_cache()

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            "RealMLP OOF contains NaNs."
        )

    overall_auc = float(
        roc_auc_score(
            y,
            oof,
        )
    )

    test_prediction = np.mean(
        np.vstack(
            test_fold_predictions
        ),
        axis=0,
    )

    delta_vs_raw_xgb = (
        overall_auc
        - RAW_XGB_REFERENCE_AUC
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    correlation_rows = []

    references = [
        (
            "current_catboost",
            CURRENT_CAT_OOF,
            [
                "oof_prediction",
                "prediction",
            ],
        ),
        (
            "current_xgboost",
            CURRENT_XGB_OOF,
            [
                "oof_prediction",
                "prediction",
            ],
        ),
        (
            "old_engineered_lgbm",
            CURRENT_LGBM_OOF,
            [
                "oof_prediction",
                "prediction",
            ],
        ),
        (
            "current_ensemble_meta",
            CURRENT_ENSEMBLE_OOF,
            [
                "candidate_meta_oof_prediction",
                "oof_prediction",
                "prediction",
            ],
        ),
    ]

    for (
        name,
        path,
        preferred,
    ) in references:
        ref = load_reference_oof(
            path,
            train,
            fold_ids,
            preferred,
        )

        if ref is None:
            continue

        correlation_rows.append(
            {
                "reference": name,
                "probability_corr": float(
                    np.corrcoef(
                        oof,
                        ref,
                    )[0, 1]
                ),
                "rank_corr": rank_corr(
                    oof,
                    ref,
                ),
                "reference_auc": float(
                    roc_auc_score(
                        y,
                        ref,
                    )
                ),
            }
        )

    correlations = pd.DataFrame(
        correlation_rows
    )

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    correlations.to_csv(
        OUTPUT_DIR
        / "correlations.csv",
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
            "oof_prediction": (
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
        OUTPUT_DIR
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
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # We intentionally do NOT auto-KEEP based only on standalone score.
    # For a new model family, diversity matters too.
    if (
        overall_auc
        >= 0.9450
    ):
        decision = (
            "PROMISING_REALMLP_TEST_IN_ENSEMBLE"
        )
    elif (
        overall_auc
        > RAW_XGB_REFERENCE_AUC
    ):
        decision = (
            "REALMLP_BEATS_RAW_XGB_BUT_REVIEW_DIVERSITY_BEFORE_ENSEMBLE"
        )
    else:
        decision = (
            "REJECT_RAW_REALMLP"
        )

    summary = [
        "EXPERIMENT: REALMLP RAW-FEATURE FROZEN 5-FOLD CV",
        "=" * 86,
        f"Frozen fold SHA256: {fold_hash}",
        f"Raw features: {X.shape[1]}",
        f"Categorical features: {len(categorical_cols)}",
        "",
        "MODEL",
        "RealMLP_TD_Classifier",
        "device=cuda",
        "random_state=42",
        "n_cv=1",
        "n_refit=0",
        "n_ens=1",
        "val_metric_name=1-auc_ovr",
        "use_ls=False",
        "",
        "RESULTS",
        f"RealMLP raw OOF AUC: {overall_auc:.8f}",
        f"Raw XGB reference AUC: {RAW_XGB_REFERENCE_AUC:.8f}",
        f"Delta vs raw XGB: {delta_vs_raw_xgb:+.8f}",
        f"Current engineered XGB reference: {CURRENT_XGB_REFERENCE_AUC:.8f}",
        f"Current ensemble reference: {CURRENT_ENSEMBLE_REFERENCE_AUC:.8f}",
        f"Runtime: {total_seconds:.2f}s",
        f"Decision: {decision}",
        "",
        "FOLD RESULTS",
    ]

    for row in (
        fold_metrics
        .itertuples()
    ):
        summary.append(
            f"Fold {row.fold}: "
            f"AUC={row.auc:.8f}, "
            f"runtime={row.runtime_seconds:.2f}s"
        )

    if len(
        correlations
    ):
        summary.extend(
            [
                "",
                "OOF CORRELATIONS",
            ]
        )

        for row in (
            correlations
            .itertuples()
        ):
            summary.append(
                f"{row.reference}: "
                f"prob_corr={row.probability_corr:.6f}, "
                f"rank_corr={row.rank_corr:.6f}, "
                f"reference_auc={row.reference_auc:.8f}"
            )

    (
        OUTPUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("REALMLP RAW-FEATURE EXPERIMENT COMPLETE")
    print("=" * 100)
    print(
        f"RealMLP OOF AUC       : "
        f"{overall_auc:.8f}"
    )
    print(
        f"Raw XGB reference     : "
        f"{RAW_XGB_REFERENCE_AUC:.8f}"
    )
    print(
        f"Delta vs raw XGB      : "
        f"{delta_vs_raw_xgb:+.8f}"
    )
    print(
        f"Current engineered XGB: "
        f"{CURRENT_XGB_REFERENCE_AUC:.8f}"
    )
    print(
        f"Current ensemble      : "
        f"{CURRENT_ENSEMBLE_REFERENCE_AUC:.8f}"
    )
    print(
        f"Decision              : "
        f"{decision}"
    )
    print(
        f"Runtime               : "
        f"{total_seconds:.2f}s"
    )
    print(
        f"Artifacts             : "
        f"{OUTPUT_DIR.resolve()}"
    )

    if len(
        correlations
    ):
        print()
        print("OOF diversity:")
        print(
            correlations[
                [
                    "reference",
                    "probability_corr",
                    "rank_corr",
                    "reference_auc",
                ]
            ].to_string(
                index=False
            )
        )

    print("=" * 100)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. fold_metrics.csv")
    print("  4. correlations.csv")


if __name__ == "__main__":
    main()
