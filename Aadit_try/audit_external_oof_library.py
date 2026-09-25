"""
audit_external_oof_library.py

Kaggle Playground Series S6E9
EXTERNAL OOF DIVERSITY AUDIT

QUESTION
--------
Can any member from the downloaded 19-model public OOF library add validated
ranking diversity to our current champion, despite being weaker standalone?

TYPE
----
Ensemble audit. NO model training. NO public-leaderboard optimization.

PRIMARY TEST
------------
For each external member independently:

1. Verify:
   - our frozen fold SHA256
   - external folds_seed42.npy exactly equals our frozen fold assignment
   - regenerated StratifiedKFold(seed=42) exactly equals both
   - OOF/test row counts
   - finite predictions
   - shipped OOF AUC and per-fold AUC reproduce manifest.csv

2. Compute standalone AUC and probability/rank correlation vs our champion.

3. Run a nested coarse rank-blend audit:
   - held fold = evaluation fold
   - blend weight chosen ONLY on the other four folds
   - alpha grid = [0, .05, .10, .15, .20, .25, .30]
   - blend = (1-alpha)*champion_rank + alpha*external_rank
   - no negative weights in this first audit
   - no fine weight sweep

4. Pool the five held-fold predictions into one meta-OOF score.

This is intentionally conservative. If convex blending finds no useful signal,
we do NOT silently switch to a more flexible stack in the same experiment.

NOTES ON EXTERNAL LIBRARY
-------------------------
The library is positional (no IDs). Exact fold-vector equality plus exact
reproduction of the shipped manifest AUCs is used as the alignment audit.

Members whose manifest discloses held-out-fold early stopping or transductive
features are flagged. They are reported but never silently treated as strictly
clean evidence.

Run:
    python audit_external_oof_library.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPECTED_FOLD_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)
EXPECTED_CHAMPION_AUC = 0.94611253
AUC_TOL = 5e-6
MANIFEST_AUC_TOL = 2e-6

ALPHA_GRID = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    dtype=np.float64,
)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"

CHAMPION_OOF_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "blend_fine_income_xgb_validated_submission"
    / "oof_predictions.csv"
)

EXTERNAL_DIR = ROOT / "external_oof_library"
MANIFEST_PATH = EXTERNAL_DIR / "manifest.csv"
EXTERNAL_FOLDS_PATH = EXTERNAL_DIR / "folds_seed42.npy"

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "external_oof_diversity_audit"
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
            f"Expected exactly one train-only target column, found {train_only}"
        )
    return train_only[0]


def encode_target(y: pd.Series) -> np.ndarray:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise RuntimeError(f"Expected binary target, found {values}")

    yes = [v for v in values if str(v).strip().lower() == "yes"]
    positive = yes[0] if yes else y.value_counts().idxmin()

    return (y == positive).astype(np.int8).to_numpy()


def prediction_column(df: pd.DataFrame) -> str:
    for c in [
        "oof_prediction",
        "candidate_oof_prediction",
        "prediction",
        "meta_oof_prediction",
    ]:
        if c in df.columns:
            return c

    excluded = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "id",
    }
    candidates = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify champion prediction column. "
            f"Columns={list(df.columns)}"
        )
    return candidates[0]


def load_champion(
    path: Path,
    n_rows: int,
    expected_folds: np.ndarray,
    train: pd.DataFrame,
) -> np.ndarray:
    df = pd.read_csv(path)

    if len(df) != n_rows:
        raise RuntimeError(
            f"Champion row count mismatch: {len(df)} vs {n_rows}"
        )

    if "row_index" in df.columns:
        expected = np.arange(n_rows, dtype=np.int64)
        if not np.array_equal(
            df["row_index"].to_numpy(dtype=np.int64),
            expected,
        ):
            raise RuntimeError("Champion row_index does not match train.csv order.")

    if "fold" in df.columns:
        found = df["fold"].to_numpy(dtype=np.int64)
        if not np.array_equal(found, expected_folds):
            raise RuntimeError("Champion fold column differs from frozen folds.")

    if "id" in train.columns and "id" in df.columns:
        if not np.array_equal(
            train["id"].to_numpy(),
            df["id"].to_numpy(),
        ):
            raise RuntimeError("Champion ID order does not match train.csv.")

    pred = pd.to_numeric(
        df[prediction_column(df)],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError("Champion predictions contain NaN/Inf.")

    return pred


def rank_pct(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(x, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank_pct(a), rank_pct(b))[0, 1])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def load_vector(path: Path, expected_len: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr)

    if arr.ndim != 1:
        arr = np.squeeze(arr)

    if arr.ndim != 1:
        raise RuntimeError(
            f"{path.name} must be a 1D prediction vector, got {arr.shape}"
        )

    if len(arr) != expected_len:
        raise RuntimeError(
            f"{path.name} row count mismatch: {len(arr)} vs {expected_len}"
        )

    arr = arr.astype(np.float64, copy=False)

    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path.name} contains NaN/Inf.")

    return arr


def classify_evidence(row: pd.Series) -> str:
    early = str(row.get("early_stopping", "")).lower()
    features = str(row.get("feature_set", "")).lower()

    heldout_es = (
        "held-out validation fold itself" in early
        or "validation fold itself" in early
    )
    transductive = "transductive" in features

    # Batch-2 rows say only "see batch-2 note", so preserve that uncertainty.
    undocumented_batch2 = "see batch-2 note" in early

    if heldout_es and transductive:
        return "CAUTION_HELDOUT_ES_AND_TRANSDUCTIVE"
    if heldout_es:
        return "CAUTION_HELDOUT_EARLY_STOPPING"
    if transductive:
        return "CAUTION_TRANSDUCTIVE"
    if undocumented_batch2:
        return "REVIEW_BATCH2_TRAINING_DETAIL"
    return "CLEAN_BY_DISCLOSED_PROTOCOL"


def verify_fold_alignment(
    y: np.ndarray,
    frozen_folds: np.ndarray,
    external_folds: np.ndarray,
) -> None:
    if not np.array_equal(frozen_folds, external_folds):
        mismatch = np.flatnonzero(frozen_folds != external_folds)
        raise RuntimeError(
            "External folds_seed42.npy does NOT match our frozen folds. "
            f"Mismatched rows: {len(mismatch)}. First mismatches: "
            f"{mismatch[:10].tolist()}"
        )

    regenerated = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    dummy = np.zeros((len(y), 1), dtype=np.int8)

    for fold, (_, valid_idx) in enumerate(skf.split(dummy, y)):
        regenerated[valid_idx] = fold

    if not np.array_equal(regenerated, frozen_folds):
        mismatch = np.flatnonzero(regenerated != frozen_folds)
        raise RuntimeError(
            "Regenerated StratifiedKFold(seed=42) does NOT match our frozen "
            f"fold file. Mismatched rows: {len(mismatch)}."
        )


def manifest_fold_auc_columns() -> list[str]:
    return [f"fold{i}_auc" for i in range(5)]


def nested_rank_blend(
    y: np.ndarray,
    folds: np.ndarray,
    champion: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    meta_pred = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict] = []

    for held_fold in range(5):
        fit_mask = folds != held_fold
        held_mask = folds == held_fold

        y_fit = y[fit_mask]
        y_held = y[held_mask]

        champ_fit_rank = rank_pct(champion[fit_mask])
        cand_fit_rank = rank_pct(candidate[fit_mask])

        # Coarse grid chosen on the other four folds only.
        train_scores = []
        for alpha in ALPHA_GRID:
            blend_fit = (
                (1.0 - alpha) * champ_fit_rank
                + alpha * cand_fit_rank
            )
            train_scores.append(auc(y_fit, blend_fit))

        best_score = max(train_scores)

        # Conservative tie-break: if scores tie numerically, choose less
        # external weight.
        best_indices = [
            i
            for i, score in enumerate(train_scores)
            if np.isclose(
                score,
                best_score,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        chosen_idx = min(best_indices)
        alpha = float(ALPHA_GRID[chosen_idx])

        champ_held_rank = rank_pct(champion[held_mask])
        cand_held_rank = rank_pct(candidate[held_mask])

        blend_held = (
            (1.0 - alpha) * champ_held_rank
            + alpha * cand_held_rank
        )

        meta_pred[held_mask] = blend_held

        baseline_auc = auc(y_held, champ_held_rank)
        candidate_auc = auc(y_held, blend_held)

        rows.append(
            {
                "held_fold": held_fold,
                "selected_alpha": alpha,
                "selection_auc_on_other_4_folds": float(best_score),
                "held_baseline_auc": baseline_auc,
                "held_blend_auc": candidate_auc,
                "held_delta": candidate_auc - baseline_auc,
            }
        )

    if np.isnan(meta_pred).any():
        raise RuntimeError("Nested blend meta predictions contain NaNs.")

    return meta_pred, rows


def screen_signal(
    delta: float,
    folds_improved: int,
    mean_alpha: float,
) -> str:
    if delta >= 3e-5 and folds_improved >= 4 and mean_alpha > 0:
        return "STRONG_KEEP_CANDIDATE"
    if delta >= 1e-5 and folds_improved >= 4 and mean_alpha > 0:
        return "KEEP_CANDIDATE"
    if delta > 0 and mean_alpha > 0:
        return "WEAK_OR_INCONSISTENT"
    if mean_alpha == 0:
        return "ZERO_WEIGHT"
    return "REJECT"


def main() -> None:
    start = time.perf_counter()

    required = [
        TRAIN_PATH,
        TEST_PATH,
        FOLDS_PATH,
        CHAMPION_OOF_PATH,
        MANIFEST_PATH,
        EXTERNAL_FOLDS_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = sha256_file(FOLDS_PATH)
    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Frozen fold SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FOLD_SHA256}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target = detect_target(train, test)
    y = encode_target(train[target])

    folds_df = pd.read_csv(FOLDS_PATH)
    if "fold" not in folds_df.columns:
        raise RuntimeError("candidate_folds.csv has no 'fold' column.")

    if len(folds_df) != len(train):
        raise RuntimeError(
            "Frozen fold row count does not match train.csv."
        )

    if "row_index" in folds_df.columns:
        expected_rows = np.arange(len(train), dtype=np.int64)
        if not np.array_equal(
            folds_df["row_index"].to_numpy(dtype=np.int64),
            expected_rows,
        ):
            raise RuntimeError(
                "candidate_folds.csv is not in original train.csv row order."
            )

    frozen_folds = folds_df["fold"].to_numpy(dtype=np.int8)
    external_folds = load_vector(
        EXTERNAL_FOLDS_PATH,
        len(train),
    ).astype(np.int8)

    verify_fold_alignment(
        y=y,
        frozen_folds=frozen_folds,
        external_folds=external_folds,
    )

    champion = load_champion(
        CHAMPION_OOF_PATH,
        len(train),
        frozen_folds,
        train,
    )

    champion_auc = auc(y, champion)
    if abs(champion_auc - EXPECTED_CHAMPION_AUC) > AUC_TOL:
        raise RuntimeError(
            "Champion OOF mismatch.\n"
            f"Expected approximately: {EXPECTED_CHAMPION_AUC:.8f}\n"
            f"Loaded:                 {champion_auc:.8f}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_manifest = {
        "member",
        "model_family",
        "implementation",
        "feature_set",
        "early_stopping",
        "oof_auc",
        "n_oof_rows",
        "n_test_rows",
    }
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise RuntimeError(
            "manifest.csv missing columns: "
            f"{sorted(missing_manifest)}"
        )

    if manifest["member"].duplicated().any():
        dupes = manifest.loc[
            manifest["member"].duplicated(),
            "member",
        ].tolist()
        raise RuntimeError(f"Duplicate manifest member names: {dupes}")

    print("=" * 104)
    print("S6E9 EXTERNAL OOF LIBRARY -> CURRENT CHAMPION DIVERSITY AUDIT")
    print("=" * 104)
    print(f"Train rows             : {len(train):,}")
    print(f"Test rows              : {len(test):,}")
    print(f"Target                 : {target!r}")
    print(f"Frozen fold SHA256     : {fold_hash}")
    print("External fold vector   : EXACT MATCH")
    print("Regenerated seed42 CV  : EXACT MATCH")
    print(f"Champion OOF           : {champion_auc:.8f}")
    print(f"External members       : {len(manifest)}")
    print(f"Alpha grid             : {ALPHA_GRID.tolist()}")
    print()
    print(
        "PRIMARY QUESTION: does a public OOF member add held-fold rank-blend "
        "value to our fixed champion?"
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    member_rows: list[dict] = []
    fold_rows: list[dict] = []
    meta_predictions: dict[str, np.ndarray] = {}

    fold_auc_cols = manifest_fold_auc_columns()

    for row in manifest.itertuples(index=False):
        member = str(row.member)

        oof_path = EXTERNAL_DIR / f"oof_{member}.npy"
        test_path = EXTERNAL_DIR / f"test_{member}.npy"

        external_oof = load_vector(oof_path, len(train))
        external_test = load_vector(test_path, len(test))

        # Keep the test load as a validation step even though this audit
        # intentionally does not build a submission.
        del external_test

        external_auc = auc(y, external_oof)
        manifest_auc = float(row.oof_auc)

        if abs(external_auc - manifest_auc) > MANIFEST_AUC_TOL:
            raise RuntimeError(
                f"Manifest OOF AUC mismatch for member {member}.\n"
                f"Manifest: {manifest_auc:.8f}\n"
                f"Recomputed: {external_auc:.8f}"
            )

        per_fold_recomputed = []
        for fold in range(5):
            mask = frozen_folds == fold
            fold_auc = auc(y[mask], external_oof[mask])
            per_fold_recomputed.append(fold_auc)

            col = f"fold{fold}_auc"
            if col in manifest.columns and pd.notna(getattr(row, col)):
                reported = float(getattr(row, col))
                if abs(fold_auc - reported) > MANIFEST_AUC_TOL:
                    raise RuntimeError(
                        f"Manifest fold AUC mismatch for {member}, fold {fold}.\n"
                        f"Manifest: {reported:.8f}\n"
                        f"Recomputed: {fold_auc:.8f}"
                    )

        prob_corr = float(
            np.corrcoef(champion, external_oof)[0, 1]
        )
        rcorr = rank_corr(champion, external_oof)

        meta_pred, candidate_fold_rows = nested_rank_blend(
            y=y,
            folds=frozen_folds,
            champion=champion,
            candidate=external_oof,
        )

        champion_meta = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            mask = frozen_folds == fold
            champion_meta[mask] = rank_pct(champion[mask])

        baseline_meta_auc = auc(y, champion_meta)
        blend_meta_auc = auc(y, meta_pred)
        delta = blend_meta_auc - baseline_meta_auc

        folds_improved = sum(
            r["held_delta"] > 0
            for r in candidate_fold_rows
        )
        folds_worse = sum(
            r["held_delta"] < 0
            for r in candidate_fold_rows
        )
        mean_alpha = float(
            np.mean(
                [
                    r["selected_alpha"]
                    for r in candidate_fold_rows
                ]
            )
        )

        manifest_row = manifest.loc[
            manifest["member"].astype(str) == member
        ].iloc[0]

        evidence_class = classify_evidence(manifest_row)
        signal = screen_signal(
            delta=delta,
            folds_improved=folds_improved,
            mean_alpha=mean_alpha,
        )

        member_rows.append(
            {
                "member": member,
                "model_family": str(row.model_family),
                "implementation": str(row.implementation),
                "evidence_class": evidence_class,
                "standalone_auc": external_auc,
                "delta_standalone_vs_champion": external_auc - champion_auc,
                "probability_corr_vs_champion": prob_corr,
                "rank_corr_vs_champion": rcorr,
                "champion_meta_auc": baseline_meta_auc,
                "nested_blend_meta_auc": blend_meta_auc,
                "nested_delta_vs_champion_meta": delta,
                "folds_improved": folds_improved,
                "folds_worse": folds_worse,
                "mean_selected_alpha": mean_alpha,
                "max_selected_alpha": float(
                    max(r["selected_alpha"] for r in candidate_fold_rows)
                ),
                "screening_signal": signal,
            }
        )

        for fold_row in candidate_fold_rows:
            fold_rows.append(
                {
                    "member": member,
                    **fold_row,
                }
            )

        meta_predictions[member] = meta_pred

        print(
            f"{member:18s} | "
            f"AUC={external_auc:.6f} | "
            f"rank_corr={rcorr:.6f} | "
            f"nested_delta={delta:+.8f} | "
            f"folds={folds_improved}/5 | "
            f"alpha={mean_alpha:.3f} | "
            f"{signal}"
        )

    results = (
        pd.DataFrame(member_rows)
        .sort_values(
            [
                "nested_delta_vs_champion_meta",
                "folds_improved",
                "standalone_auc",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    folds_out = pd.DataFrame(fold_rows)

    results.to_csv(
        OUTPUT_DIR / "member_audit.csv",
        index=False,
    )
    folds_out.to_csv(
        OUTPUT_DIR / "fold_blend_audit.csv",
        index=False,
    )

    # Save only meta predictions, not external source arrays.
    meta_df = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": frozen_folds,
            "target_encoded": y,
        }
    )
    for member in results["member"]:
        meta_df[f"meta_{member}"] = meta_predictions[str(member)]

    meta_df.to_csv(
        OUTPUT_DIR / "meta_oof_predictions.csv",
        index=False,
    )

    clean = results[
        results["evidence_class"]
        == "CLEAN_BY_DISCLOSED_PROTOCOL"
    ].copy()

    strict_positive = clean[
        (
            clean["nested_delta_vs_champion_meta"] > 0
        )
        & (
            clean["mean_selected_alpha"] > 0
        )
    ].copy()

    clean.to_csv(
        OUTPUT_DIR / "clean_members_only.csv",
        index=False,
    )

    elapsed = time.perf_counter() - start

    best = results.iloc[0]
    best_clean = (
        clean.iloc[0]
        if len(clean)
        else None
    )

    summary_lines = [
        "EXPERIMENT: EXTERNAL OOF LIBRARY DIVERSITY AUDIT",
        "=" * 92,
        "",
        "TYPE",
        "Ensemble audit; no training; no public leaderboard optimization.",
        "",
        "ALIGNMENT",
        f"Frozen fold SHA256: {fold_hash}",
        "External folds_seed42.npy: EXACT MATCH",
        "Regenerated StratifiedKFold(seed=42): EXACT MATCH",
        f"Train rows: {len(train)}",
        f"Test rows: {len(test)}",
        "",
        "BASELINE",
        f"Current champion raw OOF: {champion_auc:.8f}",
        "",
        "METHOD",
        (
            "For each external member independently, select alpha from "
            f"{ALPHA_GRID.tolist()} on the other four folds, then score on "
            "the held fold using rank blending."
        ),
        "No negative weights. No fine alpha sweep.",
        "",
        "BEST OVERALL SCREENING RESULT",
        f"Member: {best['member']}",
        f"Evidence class: {best['evidence_class']}",
        f"Standalone AUC: {best['standalone_auc']:.8f}",
        (
            "Nested blend meta AUC: "
            f"{best['nested_blend_meta_auc']:.8f}"
        ),
        (
            "Delta vs champion meta: "
            f"{best['nested_delta_vs_champion_meta']:+.8f}"
        ),
        f"Folds improved: {int(best['folds_improved'])}/5",
        f"Mean selected alpha: {best['mean_selected_alpha']:.3f}",
        f"Screening signal: {best['screening_signal']}",
        "",
    ]

    if best_clean is not None:
        summary_lines.extend(
            [
                "BEST STRICTLY CLEAN-BY-DISCLOSED-PROTOCOL MEMBER",
                f"Member: {best_clean['member']}",
                f"Standalone AUC: {best_clean['standalone_auc']:.8f}",
                (
                    "Nested blend meta AUC: "
                    f"{best_clean['nested_blend_meta_auc']:.8f}"
                ),
                (
                    "Delta vs champion meta: "
                    f"{best_clean['nested_delta_vs_champion_meta']:+.8f}"
                ),
                (
                    f"Folds improved: "
                    f"{int(best_clean['folds_improved'])}/5"
                ),
                (
                    f"Mean selected alpha: "
                    f"{best_clean['mean_selected_alpha']:.3f}"
                ),
                (
                    f"Screening signal: "
                    f"{best_clean['screening_signal']}"
                ),
                "",
            ]
        )

    summary_lines.extend(
        [
            "TOP 10 BY NESTED HELD-FOLD DELTA",
        ]
    )

    for r in results.head(10).itertuples():
        summary_lines.append(
            f"{r.member}: "
            f"delta={r.nested_delta_vs_champion_meta:+.8f}, "
            f"folds={r.folds_improved}/5, "
            f"alpha={r.mean_selected_alpha:.3f}, "
            f"standalone={r.standalone_auc:.6f}, "
            f"rank_corr={r.rank_corr_vs_champion:.6f}, "
            f"evidence={r.evidence_class}, "
            f"signal={r.screening_signal}"
        )

    summary_lines.extend(
        [
            "",
            "STRICT-CLEAN POSITIVE COUNT",
            str(len(strict_positive)),
            "",
            f"Runtime: {elapsed:.2f}s",
            "",
            "MANUAL REVIEW REQUIRED",
            (
                "Do not automatically change the champion from this screening "
                "result alone. A positive member should receive a dedicated "
                "follow-up confirmation/audit."
            ),
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "audit_config.json").write_text(
        json.dumps(
            {
                "expected_fold_sha256": EXPECTED_FOLD_SHA256,
                "expected_champion_auc": EXPECTED_CHAMPION_AUC,
                "alpha_grid": ALPHA_GRID.tolist(),
                "method": (
                    "nested 4-fold selection / 1-fold held-out rank blend"
                ),
                "negative_weights": False,
                "fine_weight_search": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("AUDIT COMPLETE")
    print("=" * 104)
    print(f"Champion raw OOF       : {champion_auc:.8f}")
    print(
        f"Best overall           : {best['member']} | "
        f"{best['nested_delta_vs_champion_meta']:+.8f} | "
        f"{int(best['folds_improved'])}/5 folds | "
        f"alpha={best['mean_selected_alpha']:.3f}"
    )
    if best_clean is not None:
        print(
            f"Best disclosed-clean   : {best_clean['member']} | "
            f"{best_clean['nested_delta_vs_champion_meta']:+.8f} | "
            f"{int(best_clean['folds_improved'])}/5 folds | "
            f"alpha={best_clean['mean_selected_alpha']:.3f}"
        )
    print(f"Artifacts              : {OUTPUT_DIR.resolve()}")
    print(f"Runtime                : {elapsed:.2f}s")
    print("=" * 104)
    print()
    print("Send me:")
    print("  1. terminal output")
    print("  2. summary.txt")
    print("  3. member_audit.csv")
    print("  4. fold_blend_audit.csv")


if __name__ == "__main__":
    main()
