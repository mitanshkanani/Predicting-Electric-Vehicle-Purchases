from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# FINAL VALIDATED FINE-INCOME XGB ENSEMBLE SUBMISSION
#
# NO TRAINING.
# NO WEIGHT SEARCH.
#
# This reproduces the already-validated replacement ensemble:
#   CatBoost = existing income+commute 3-seed member
#   XGBoost  = new fine-income ($50/$250) member
#   LightGBM = existing older diverse engineered member
#
# OOF uses the already-validated fold-specific weights:
#   folds 0-1: CAT=.25 / XGB=.50 / LGBM=.25
#   folds 2-4: CAT=.20 / XGB=.55 / LGBM=.25
#
# Test follows the exact historical submission procedure:
#   1. globally rank each test member
#   2. blend with mean validated weights:
#      CAT=.22 / XGB=.53 / LGBM=.25
# ============================================================

EXPECTED_HASH = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

EXPECTED = {
    "cat": 0.94558572,
    "fine_xgb": 0.94606664,
    "lgbm": 0.94578042,
    "old_champion": 0.94596441,
    "new_champion": 0.94611253,
}

TOL = 2e-5
NEW_CHAMPION_TOL = 2e-6

FOLDS = Path("artifacts/validation/candidate_folds.csv")

CAT = Path(
    "artifacts/experiments/"
    "catboost_hierarchical_income_commute_multiseed_gpu/"
    "best_average_oof_predictions.csv"
)
FINE_XGB = Path(
    "artifacts/experiments/"
    "xgboost_fine_income_te_gpu/"
    "oof_predictions.csv"
)
LGBM = Path(
    "artifacts/experiments/"
    "lightgbm_engineered_learned_margin_cpu/"
    "oof_predictions.csv"
)
OLD_CHAMPION = Path(
    "artifacts/experiments/"
    "blend_income_commute_catboost_commute_xgb_lgbm_rank_audit/"
    "oof_predictions.csv"
)

CAT_TEST = Path(
    "artifacts/experiments/"
    "catboost_hierarchical_income_commute_multiseed_gpu/"
    "best_average_test_predictions.csv"
)
FINE_XGB_TEST = Path(
    "artifacts/experiments/"
    "xgboost_fine_income_te_gpu/"
    "test_predictions.csv"
)
LGBM_TEST = Path(
    "artifacts/experiments/"
    "lightgbm_engineered_learned_margin_cpu/"
    "test_predictions.csv"
)

OUT = Path(
    "artifacts/experiments/"
    "blend_fine_income_xgb_validated_submission"
)

# Exact weights recovered from the saved champion and used in the
# validated no-training member-replacement audit.
FOLD_WEIGHTS = {
    0: (0.25, 0.50, 0.25),
    1: (0.25, 0.50, 0.25),
    2: (0.20, 0.55, 0.25),
    3: (0.20, 0.55, 0.25),
    4: (0.20, 0.55, 0.25),
}

TEST_WEIGHTS = (
    np.mean([w[0] for w in FOLD_WEIGHTS.values()]),
    np.mean([w[1] for w in FOLD_WEIGHTS.values()]),
    np.mean([w[2] for w in FOLD_WEIGHTS.values()]),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pred_col(df: pd.DataFrame, kind: str) -> str:
    if kind == "oof":
        pref = [
            "oof_prediction",
            "candidate_meta_oof_prediction",
            "prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]
    else:
        pref = [
            "prediction",
            "test_prediction",
            "rank_average_prediction",
            "best_average_prediction",
        ]

    for c in pref:
        if c in df.columns:
            return c

    exclude = {
        "row_index",
        "fold",
        "target",
        "target_encoded",
        "id",
    }

    nums = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(nums) != 1:
        raise ValueError(
            f"Cannot identify prediction column: {list(df.columns)}"
        )

    return nums[0]


def load_oof(
    path: Path,
    n: int,
    folds: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != n:
        raise ValueError(f"row mismatch: {path}")

    if (
        "row_index" not in df
        or not np.array_equal(
            df["row_index"].to_numpy(),
            np.arange(n),
        )
    ):
        raise ValueError(f"row order mismatch: {path}")

    if (
        "fold" not in df
        or not np.array_equal(
            df["fold"].to_numpy(),
            folds,
        )
    ):
        raise ValueError(f"fold mismatch: {path}")

    p = df[pred_col(df, "oof")].to_numpy(float)

    if not np.isfinite(p).all():
        raise ValueError(f"non-finite predictions: {path}")

    return p


def load_test(
    path: Path,
    n: int,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != n:
        raise ValueError(f"test row mismatch: {path}")

    p = df[pred_col(df, "test")].to_numpy(float)

    if not np.isfinite(p).all():
        raise ValueError(f"non-finite test predictions: {path}")

    return p


def rank(v: np.ndarray) -> np.ndarray:
    return (
        pd.Series(v)
        .rank(method="average", pct=True)
        .to_numpy(float)
    )


def fold_rank(
    v: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(v), dtype=float)

    for f in range(5):
        mask = folds == f
        out[mask] = rank(v[mask])

    return out


def main() -> None:
    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")

    required = [
        train_path,
        test_path,
        FOLDS,
        CAT,
        FINE_XGB,
        LGBM,
        OLD_CHAMPION,
        CAT_TEST,
        FINE_XGB_TEST,
        LGBM_TEST,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path.resolve()}"
            )

    fold_hash = sha256(FOLDS)

    if fold_hash != EXPECTED_HASH:
        raise ValueError(
            "Frozen fold SHA mismatch\n"
            f"Expected: {EXPECTED_HASH}\n"
            f"Found:    {fold_hash}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    fold_df = pd.read_csv(FOLDS)

    if len(fold_df) != len(train):
        raise ValueError("fold count mismatch")

    folds = fold_df["fold"].to_numpy(int)

    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"unexpected frozen folds: {sorted(np.unique(folds).tolist())}"
        )

    target_candidates = [
        c for c in train.columns
        if c not in test.columns
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            f"target detection failed: {target_candidates}"
        )

    target = target_candidates[0]

    y = (
        train[target]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(np.int8)
        .to_numpy()
    )

    cat = load_oof(CAT, len(train), folds)
    fine_xgb = load_oof(FINE_XGB, len(train), folds)
    lgbm = load_oof(LGBM, len(train), folds)
    old_champion = load_oof(
        OLD_CHAMPION,
        len(train),
        folds,
    )

    members = {
        "cat": cat,
        "fine_xgb": fine_xgb,
        "lgbm": lgbm,
        "old_champion": old_champion,
    }

    print("=" * 100)
    print("VALIDATED FINE-INCOME XGB ENSEMBLE SUBMISSION")
    print("NO TRAINING | NO WEIGHT SEARCH")
    print("=" * 100)
    print(f"Frozen fold SHA256 verified: {fold_hash}")
    print()

    for name, p in members.items():
        score = float(roc_auc_score(y, p))

        if abs(score - EXPECTED[name]) > TOL:
            raise ValueError(
                f"{name} AUC mismatch: "
                f"{score:.8f} vs {EXPECTED[name]:.8f}"
            )

        print(
            f"{name:12s}: "
            f"{score:.8f}"
        )

    # Same rank representation as the original blend script.
    cat_r = fold_rank(cat, folds)
    xgb_r = fold_rank(fine_xgb, folds)
    lgbm_r = fold_rank(lgbm, folds)

    candidate_oof = np.full(
        len(train),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    print()
    print("Fold-specific fixed replacement weights:")
    for held in range(5):
        val = folds == held
        wc, wx, wl = FOLD_WEIGHTS[held]

        pred = (
            wc * cat_r[val]
            + wx * xgb_r[val]
            + wl * lgbm_r[val]
        )

        candidate_oof[val] = pred

        old_auc = float(
            roc_auc_score(
                y[val],
                old_champion[val],
            )
        )

        new_auc = float(
            roc_auc_score(
                y[val],
                pred,
            )
        )

        delta = new_auc - old_auc

        fold_rows.append(
            {
                "held_fold": held,
                "cat_weight": wc,
                "xgb_weight": wx,
                "lgbm_weight": wl,
                "old_champion_auc": old_auc,
                "new_champion_auc": new_auc,
                "delta": delta,
            }
        )

        print(
            f"Fold {held}: "
            f"{old_auc:.8f} -> {new_auc:.8f} "
            f"({delta:+.8f}) | "
            f"weights={wc:.2f}/{wx:.2f}/{wl:.2f}"
        )

    if np.isnan(candidate_oof).any():
        raise RuntimeError("candidate OOF contains NaNs")

    old_auc = float(
        roc_auc_score(
            y,
            old_champion,
        )
    )

    new_auc = float(
        roc_auc_score(
            y,
            candidate_oof,
        )
    )

    delta = new_auc - old_auc

    if (
        abs(
            new_auc
            - EXPECTED["new_champion"]
        )
        > NEW_CHAMPION_TOL
    ):
        raise ValueError(
            "Validated new champion reproduction mismatch.\n"
            f"Expected: {EXPECTED['new_champion']:.8f}\n"
            f"Found:    {new_auc:.8f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)

    improved = int(
        (fold_metrics["delta"] > 0).sum()
    )

    worse = int(
        (fold_metrics["delta"] < 0).sum()
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # Preserve the exact historical submission procedure:
    # global rank each member -> mean validated weights -> blend.
    # --------------------------------------------------------
    cat_test_r = rank(
        load_test(
            CAT_TEST,
            len(test),
        )
    )

    xgb_test_r = rank(
        load_test(
            FINE_XGB_TEST,
            len(test),
        )
    )

    lgbm_test_r = rank(
        load_test(
            LGBM_TEST,
            len(test),
        )
    )

    wc, wx, wl = TEST_WEIGHTS

    test_prediction = (
        wc * cat_test_r
        + wx * xgb_test_r
        + wl * lgbm_test_r
    )

    if not np.isfinite(test_prediction).all():
        raise RuntimeError(
            "non-finite blended test predictions"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        OUT / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_index": np.arange(
                len(train),
                dtype=np.int64,
            ),
            "fold": folds,
            "target_encoded": y,
            "oof_prediction": candidate_oof,
        }
    ).to_csv(
        OUT / "oof_predictions.csv",
        index=False,
    )

    test_predictions = pd.DataFrame(
        {
            "prediction": (
                test_prediction
                .astype(np.float32)
            )
        }
    )

    if "id" in test.columns:
        test_predictions.insert(
            0,
            "id",
            test["id"].to_numpy(),
        )

    test_predictions.to_csv(
        OUT / "test_predictions.csv",
        index=False,
    )

    # Kaggle-ready submission:
    # same predictions, target column name restored.
    submission = pd.DataFrame()

    if "id" in test.columns:
        submission["id"] = test["id"].to_numpy()

    submission[target] = (
        test_prediction.astype(np.float64)
    )

    submission.to_csv(
        OUT / "submission.csv",
        index=False,
    )

    summary = [
        "EXPERIMENT: VALIDATED FINE-INCOME XGB ENSEMBLE SUBMISSION",
        "=" * 84,
        "",
        "NO TRAINING",
        "NO WEIGHT SEARCH",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        "",
        "MEMBERS",
        f"CatBoost OOF: {roc_auc_score(y, cat):.8f}",
        f"Fine-income XGB OOF: {roc_auc_score(y, fine_xgb):.8f}",
        f"LightGBM OOF: {roc_auc_score(y, lgbm):.8f}",
        "",
        "VALIDATED META-CV",
        f"Old champion: {old_auc:.8f}",
        f"New champion: {new_auc:.8f}",
        f"Delta: {delta:+.8f}",
        f"Folds improved: {improved}/5",
        f"Folds worse: {worse}/5",
        "",
        "TEST BLEND",
        f"CAT={wc:.4f}",
        f"XGB={wx:.4f}",
        f"LGBM={wl:.4f}",
        "",
        f"Submission: {(OUT / 'submission.csv').resolve()}",
    ]

    (OUT / "summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("VALIDATION REPRODUCTION")
    print("=" * 100)
    print(f"Old champion : {old_auc:.8f}")
    print(f"New champion : {new_auc:.8f}")
    print(f"Delta        : {delta:+.8f}")
    print(
        f"Folds        : "
        f"{improved}/5 improved, "
        f"{worse}/5 worse"
    )
    print()
    print(
        "Test weights  : "
        f"CAT={wc:.4f}, "
        f"XGB={wx:.4f}, "
        f"LGBM={wl:.4f}"
    )
    print(
        f"Submission    : "
        f"{(OUT / 'submission.csv').resolve()}"
    )
    print("=" * 100)
    print()
    print(
        "Upload submission.csv to Kaggle, then send me "
        "the public leaderboard score."
    )


if __name__ == "__main__":
    main()
