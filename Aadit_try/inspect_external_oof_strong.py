"""
inspect_external_oof_strong.py

S6E9 strong external OOF diagnostic.

Purpose:
- inspect every *_oof.csv in external_oof_strong/
- verify row counts and ID alignment against competition train.csv
- identify prediction / target / fold-like columns
- recompute standalone OOF AUC when safely possible
- check whether any external fold column exactly matches our frozen fold vector
- inspect matching test/submission files
- produce one auditable summary before any ensemble experiment

NO model training.
NO blending.
NO leaderboard optimization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "external_oof_strong"

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "external_oof_strong_inspection"
)

EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) != 1:
        raise RuntimeError(
            f"Expected one train-only target column, found {train_only}"
        )
    return train_only[0]


def is_binary_label_series(y: pd.Series) -> bool:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        return False

    normalized = {str(v).strip().lower() for v in values}
    known_binary = [
        {"yes", "no"},
        {"true", "false"},
        {"0", "1"},
        {"0.0", "1.0"},
    ]
    return normalized in known_binary


def encode_target(y: pd.Series) -> np.ndarray:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        preview = values[:10]
        raise RuntimeError(
            "Expected binary target with exactly 2 unique values; "
            f"found {len(values)}. Preview={preview}"
        )

    yes = [v for v in values if str(v).strip().lower() == "yes"]
    positive = yes[0] if yes else y.value_counts().idxmin()
    return (y == positive).astype(np.int8).to_numpy()


def numeric_prediction_candidates(
    df: pd.DataFrame,
    target: str,
    id_col: str | None,
) -> list[str]:
    excluded = {
        "fold",
        "fold_id",
        "cv_fold",
        "row_index",
    }

    # Some public OOF files name the prediction column exactly like the
    # competition target (e.g. Will_Buy_EV). Exclude label-looking names
    # only when the column actually contains binary labels.
    for c in [target, "target", "target_encoded", "label", "y"]:
        if c in df.columns and is_binary_label_series(df[c]):
            excluded.add(c)
    if id_col:
        excluded.add(id_col)

    preferred_tokens = (
        "pred",
        "prob",
        "oof",
        "score",
        "will_buy_ev",
    )

    numeric = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    preferred = [
        c
        for c in numeric
        if any(token in c.lower() for token in preferred_tokens)
    ]

    probability_like = []
    for c in preferred + [c for c in numeric if c not in preferred]:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().all() and len(vals):
            mn = float(vals.min())
            mx = float(vals.max())
            if mn >= -1e-12 and mx <= 1.0 + 1e-12:
                probability_like.append(c)

    return list(dict.fromkeys(probability_like))


def fold_candidates(df: pd.DataFrame) -> list[str]:
    names = []
    for c in df.columns:
        lc = c.lower()
        if "fold" in lc:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().all():
                unique = sorted(pd.unique(vals.astype(int)).tolist())
                if 2 <= len(unique) <= 20:
                    names.append(c)
    return names


def find_matching_test_file(oof_path: Path) -> Path | None:
    name = oof_path.name

    explicit = {
        "01_blend_oof.csv": "01_submission.csv",
        "Sergey_LGBM_oof.csv": "Sergey_LGBM_submission.csv",
    }
    if name in explicit:
        p = DATA_DIR / explicit[name]
        return p if p.exists() else None

    if name.endswith("_oof.csv"):
        p = DATA_DIR / name.replace("_oof.csv", "_test.csv")
        return p if p.exists() else None

    return None


def main() -> None:
    for path in [TRAIN_PATH, TEST_PATH, FOLDS_PATH, DATA_DIR]:
        if not path.exists():
            raise FileNotFoundError(path)

    fold_hash = sha256_file(FOLDS_PATH)
    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Frozen fold SHA mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)

    target = detect_target(train, test)
    y = encode_target(train[target])

    if len(folds_df) != len(train):
        raise RuntimeError("Frozen folds row count mismatch.")

    frozen_folds = folds_df["fold"].to_numpy(dtype=np.int64)

    id_col = "id" if "id" in train.columns and "id" in test.columns else None

    oof_files = sorted(DATA_DIR.glob("*_oof.csv"))
    if not oof_files:
        raise RuntimeError(f"No *_oof.csv files found under {DATA_DIR}")

    print("=" * 110)
    print("S6E9 STRONG EXTERNAL OOF — STRUCTURE / ALIGNMENT INSPECTION")
    print("=" * 110)
    print(f"Train rows         : {len(train):,}")
    print(f"Test rows          : {len(test):,}")
    print(f"Target             : {target!r}")
    print(f"ID column          : {id_col!r}")
    print(f"Frozen fold SHA256 : {fold_hash}")
    print(f"OOF files found    : {len(oof_files)}")
    print()

    rows: list[dict] = []
    detail: dict[str, dict] = {}

    for oof_path in oof_files:
        df = pd.read_csv(oof_path)
        test_path = find_matching_test_file(oof_path)

        row_count_ok = len(df) == len(train)

        id_alignment = "NO_ID_COLUMN"
        detected_id_col = None

        if id_col and id_col in df.columns:
            detected_id_col = id_col
            if row_count_ok and np.array_equal(
                df[id_col].to_numpy(),
                train[id_col].to_numpy(),
            ):
                id_alignment = "EXACT"
            else:
                id_alignment = "MISMATCH"
        elif id_col:
            for c in df.columns:
                if len(df) != len(train):
                    break
                try:
                    if np.array_equal(
                        pd.to_numeric(df[c], errors="raise").to_numpy(),
                        train[id_col].to_numpy(),
                    ):
                        detected_id_col = c
                        id_alignment = "EXACT_ALIAS"
                        break
                except Exception:
                    pass

        target_alignment = "NO_TARGET_COLUMN"
        target_col = None
        if target in df.columns:
            target_col = target
            if is_binary_label_series(df[target]):
                ext_y = encode_target(df[target])
                target_alignment = (
                    "EXACT"
                    if row_count_ok and np.array_equal(ext_y, y)
                    else "MISMATCH"
                )
            else:
                # Public Kaggle submission/OOF files often use the target
                # name for predicted probabilities. Do not mistake that for
                # the ground-truth label vector.
                target_alignment = "COLUMN_NAME_IS_PREDICTION_NOT_LABEL"

        folds = fold_candidates(df)
        fold_match_cols = []
        for c in folds:
            vals = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=np.int64)
            if row_count_ok and np.array_equal(vals, frozen_folds):
                fold_match_cols.append(c)

        pred_candidates = numeric_prediction_candidates(
            df=df,
            target=target,
            id_col=detected_id_col,
        )

        auc_results = {}
        for c in pred_candidates:
            if not row_count_ok:
                continue
            p = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(p).all():
                continue
            auc_results[c] = float(roc_auc_score(y, p))

        best_pred_col = None
        best_auc = np.nan
        if auc_results:
            best_pred_col = max(auc_results, key=auc_results.get)
            best_auc = auc_results[best_pred_col]

        test_rows = None
        test_columns = None
        test_id_alignment = None
        if test_path is not None:
            tdf = pd.read_csv(test_path)
            test_rows = len(tdf)
            test_columns = list(tdf.columns)

            if id_col and id_col in tdf.columns:
                test_id_alignment = (
                    "EXACT"
                    if len(tdf) == len(test)
                    and np.array_equal(
                        tdf[id_col].to_numpy(),
                        test[id_col].to_numpy(),
                    )
                    else "MISMATCH"
                )
            else:
                test_id_alignment = "NO_ID_COLUMN"

        rows.append(
            {
                "file": oof_path.name,
                "oof_rows": len(df),
                "row_count_ok": row_count_ok,
                "columns": " | ".join(map(str, df.columns)),
                "detected_id_col": detected_id_col,
                "id_alignment": id_alignment,
                "target_col": target_col,
                "target_alignment": target_alignment,
                "fold_candidates": " | ".join(folds),
                "frozen_fold_match_columns": " | ".join(fold_match_cols),
                "prediction_candidates": " | ".join(pred_candidates),
                "best_prediction_column": best_pred_col,
                "best_recomputed_auc": best_auc,
                "matching_test_file": (
                    test_path.name if test_path is not None else None
                ),
                "test_rows": test_rows,
                "test_row_count_ok": (
                    test_rows == len(test) if test_rows is not None else None
                ),
                "test_id_alignment": test_id_alignment,
            }
        )

        detail[oof_path.name] = {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "first_three_rows": df.head(3).to_dict(orient="records"),
            "prediction_candidate_aucs": auc_results,
            "fold_candidates": folds,
            "frozen_fold_match_columns": fold_match_cols,
            "matching_test_file": (
                test_path.name if test_path is not None else None
            ),
            "test_columns": test_columns,
        }

        auc_text = f"{best_auc:.8f}" if np.isfinite(best_auc) else "N/A"
        print(
            f"{oof_path.name:36s} | "
            f"rows={'OK' if row_count_ok else 'BAD':3s} | "
            f"id={id_alignment:12s} | "
            f"fold_match={','.join(fold_match_cols) or 'NONE':12s} | "
            f"pred={str(best_pred_col):18s} | "
            f"AUC={auc_text}"
        )

    result = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_DIR / "inspection.csv",
        index=False,
    )

    (OUTPUT_DIR / "schema_details.json").write_text(
        json.dumps(detail, indent=2, default=str),
        encoding="utf-8",
    )

    exact_row_files = int(result["row_count_ok"].fillna(False).sum())
    exact_id_files = int(
        result["id_alignment"].isin(["EXACT", "EXACT_ALIAS"]).sum()
    )
    fold_match_files = int(
        result["frozen_fold_match_columns"]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .sum()
    )

    score_table = (
        result[
            [
                "file",
                "best_prediction_column",
                "best_recomputed_auc",
                "id_alignment",
                "frozen_fold_match_columns",
                "matching_test_file",
            ]
        ]
        .sort_values(
            "best_recomputed_auc",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    score_table.to_csv(
        OUTPUT_DIR / "score_table.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: STRONG EXTERNAL OOF STRUCTURE / ALIGNMENT INSPECTION",
        "=" * 92,
        "",
        "TYPE",
        "Diagnostic only. No blending, training or leaderboard optimization.",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"OOF files inspected: {len(result)}",
        f"Correct OOF row count: {exact_row_files}/{len(result)}",
        f"Exact/alias ID alignment: {exact_id_files}/{len(result)}",
        f"Files exposing an exact frozen-fold match column: {fold_match_files}/{len(result)}",
        "",
        "SCORE TABLE",
    ]

    for r in score_table.itertuples():
        auc_text = (
            f"{r.best_recomputed_auc:.8f}"
            if pd.notna(r.best_recomputed_auc)
            else "N/A"
        )
        summary.append(
            f"{r.file}: "
            f"pred={r.best_prediction_column}, "
            f"AUC={auc_text}, "
            f"id={r.id_alignment}, "
            f"fold_match={r.frozen_fold_match_columns or 'NONE'}, "
            f"test={r.matching_test_file}"
        )

    summary.extend(
        [
            "",
            "INTERPRETATION RULE",
            (
                "Do not perform nested meta-blend selection on an external OOF "
                "whose generation folds cannot be shown to match our frozen "
                "outer folds. Different base folds can leak held-fold labels "
                "through base-model training into meta-fold selection."
            ),
            "",
            "NEXT STEP",
            (
                "Manual review of inspection.csv/schema_details.json before "
                "choosing a scientifically valid blend or standalone comparison."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("INSPECTION COMPLETE")
    print("=" * 110)
    print(f"Correct row counts      : {exact_row_files}/{len(result)}")
    print(f"Exact ID alignment      : {exact_id_files}/{len(result)}")
    print(f"Exact frozen-fold column: {fold_match_files}/{len(result)}")
    print()
    print(score_table.to_string(index=False))
    print()
    print(f"Artifacts: {OUTPUT_DIR.resolve()}")
    print("=" * 110)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. inspection.csv")


if __name__ == "__main__":
    main()
