from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# Champion OOF residual / disagreement audit
# NO MODEL TRAINING.
# ============================================================

ROOT = Path(__file__).resolve().parent

FOLDS_PATH = ROOT / "artifacts" / "validation" / "candidate_folds.csv"
EXPECTED_FOLDS_SHA256 = (
    "55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
)

ARTIFACT_DIRS = {
    "CAT": ROOT
    / "artifacts"
    / "experiments"
    / "catboost_hierarchical_income_commute_multiseed_gpu",
    "XGB": ROOT
    / "artifacts"
    / "experiments"
    / "xgboost_hierarchical_commute_te_gpu",
    "LGBM": ROOT
    / "artifacts"
    / "experiments"
    / "lightgbm_engineered_learned_margin_cpu",
    "CHAMPION": ROOT
    / "artifacts"
    / "experiments"
    / "blend_income_commute_catboost_commute_xgb_lgbm_rank_audit",
}

EXPECTED_AUC = {
    "CAT": 0.94558572,
    "XGB": 0.94591178,
    "LGBM": 0.94578042,
    "CHAMPION": 0.94596441,
}

EXPECTED_N = 668_665
TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"

RAW_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "champion_oof_residual_disagreement_audit"
)

MIN_SEGMENT_ROWS = 500
MIN_SEGMENT_CLASS = 50
AUC_VERIFY_TOL = 5e-6


@dataclass
class PredictionCandidate:
    source: Path
    column: str
    values: np.ndarray
    ids: Optional[np.ndarray]
    score: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def discover_train_csv() -> Path:
    required = set(RAW_FEATURES + [TARGET])
    candidates: list[Path] = []

    for p in ROOT.rglob("train.csv"):
        try:
            cols = set(pd.read_csv(p, nrows=2).columns)
        except Exception:
            continue
        if required.issubset(cols):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not discover a train.csv under the repo containing all expected "
            "raw features plus Will_Buy_EV."
        )

    valid: list[Path] = []
    for p in candidates:
        try:
            n = sum(1 for _ in p.open("rb")) - 1
        except Exception:
            n = -1
        if n == EXPECTED_N:
            valid.append(p)

    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise RuntimeError(
            "Multiple train.csv files match the expected schema and row count:\n"
            + "\n".join(f"  - {p}" for p in valid)
        )

    raise RuntimeError(
        "Found train.csv candidate(s), but none had the expected 668,665 rows:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def detect_fold_column(df: pd.DataFrame) -> str:
    exact = [c for c in df.columns if c.lower() == "fold"]
    if len(exact) == 1:
        return exact[0]

    fuzzy = [c for c in df.columns if "fold" in c.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise RuntimeError(
        f"Could not uniquely identify fold column in {FOLDS_PATH}. "
        f"Columns: {list(df.columns)}"
    )


def align_folds(train: pd.DataFrame, folds_df: pd.DataFrame) -> np.ndarray:
    fold_col = detect_fold_column(folds_df)

    if "id" in train.columns and "id" in folds_df.columns:
        if train["id"].duplicated().any() or folds_df["id"].duplicated().any():
            raise RuntimeError("Duplicate ids found while aligning frozen folds.")
        mapping = folds_df.set_index("id")[fold_col]
        aligned = train["id"].map(mapping)
        if aligned.isna().any():
            raise RuntimeError("Some train ids were missing from candidate_folds.csv.")
        return aligned.to_numpy(dtype=int)

    if len(folds_df) != len(train):
        raise RuntimeError(
            "candidate_folds.csv has no usable id alignment and row count differs "
            "from train.csv."
        )

    print(
        "[INFO] candidate_folds.csv has no shared usable id column; "
        "using row-order alignment."
    )
    return folds_df[fold_col].to_numpy(dtype=int)


def _candidate_name_score(path: Path, column: str, model_name: str) -> int:
    s = f"{path.name} {column}".lower()
    score = 0

    if "oof" in s:
        score += 20
    if any(k in s for k in ("pred", "proba", "probability", "score")):
        score += 8

    if model_name == "CHAMPION":
        if any(k in s for k in ("blend", "champion", "ensemble", "meta")):
            score += 10
        if "rank" in s:
            score += 4

    if model_name == "CAT":
        if any(k in s for k in ("rank_avg", "rank_average", "rankmean", "rank_mean")):
            score += 12
        elif "rank" in s:
            score += 5
        if any(k in s for k in ("avg", "average", "ensemble", "blend")):
            score += 4

    if any(k in s for k in ("test", "submission", "sample_submission")):
        score -= 40
    if any(k in s for k in ("target", "label", "truth", "y_true")):
        score -= 20
    if "fold" in s and "oof" not in s:
        score -= 4

    return score


def _valid_prediction_vector(v: np.ndarray, n: int) -> bool:
    if v.ndim != 1 or len(v) != n:
        return False
    if not np.issubdtype(v.dtype, np.number):
        return False
    vv = v.astype(float, copy=False)
    if not np.all(np.isfinite(vv)):
        return False
    # OOF probabilities and rank-normalized scores should both satisfy this.
    if vv.min() < -1e-9 or vv.max() > 1 + 1e-9:
        return False
    if np.unique(vv).size < 100:
        return False
    return True


def collect_prediction_candidates(
    artifact_dir: Path, model_name: str, n: int
) -> list[PredictionCandidate]:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Missing confirmed artifact directory: {artifact_dir}")

    candidates: list[PredictionCandidate] = []
    files = sorted(
        p
        for p in artifact_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".npy", ".npz"}
    )

    for p in files:
        suffix = p.suffix.lower()

        try:
            if suffix == ".npy":
                arr = np.load(p, allow_pickle=False)
                if arr.ndim == 2 and 1 in arr.shape:
                    arr = arr.reshape(-1)
                if _valid_prediction_vector(arr, n):
                    candidates.append(
                        PredictionCandidate(
                            p,
                            "<npy>",
                            arr.astype(float),
                            None,
                            _candidate_name_score(p, "<npy>", model_name),
                        )
                    )

            elif suffix == ".npz":
                z = np.load(p, allow_pickle=False)
                for key in z.files:
                    arr = np.asarray(z[key])
                    if arr.ndim == 2 and 1 in arr.shape:
                        arr = arr.reshape(-1)
                    if _valid_prediction_vector(arr, n):
                        candidates.append(
                            PredictionCandidate(
                                p,
                                key,
                                arr.astype(float),
                                None,
                                _candidate_name_score(p, key, model_name),
                            )
                        )

            else:
                if suffix == ".csv":
                    df = pd.read_csv(p)
                else:
                    df = pd.read_parquet(p)

                if len(df) != n:
                    continue

                ids = df["id"].to_numpy() if "id" in df.columns else None

                for c in df.columns:
                    if c == "id":
                        continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.isna().any():
                        continue
                    arr = s.to_numpy()
                    if _valid_prediction_vector(arr, n):
                        candidates.append(
                            PredictionCandidate(
                                p,
                                c,
                                arr.astype(float),
                                ids,
                                _candidate_name_score(p, c, model_name),
                            )
                        )
        except Exception:
            # Skip unreadable/non-tabular artifacts, then report the actual
            # candidates we did find.
            continue

    return candidates


def align_prediction(
    candidate: PredictionCandidate, train: pd.DataFrame
) -> np.ndarray:
    if candidate.ids is None:
        return candidate.values

    if "id" not in train.columns:
        raise RuntimeError(
            f"{candidate.source} has ids but train.csv does not; cannot align safely."
        )

    pred_df = pd.DataFrame(
        {"id": candidate.ids, "_pred": candidate.values}
    )
    if pred_df["id"].duplicated().any():
        raise RuntimeError(f"Duplicate ids in prediction file {candidate.source}")

    aligned = train[["id"]].merge(
        pred_df, on="id", how="left", validate="one_to_one", sort=False
    )
    if aligned["_pred"].isna().any():
        raise RuntimeError(
            f"Prediction ids in {candidate.source} do not fully cover train ids."
        )
    return aligned["_pred"].to_numpy(dtype=float)


def select_oof_candidate(
    artifact_dir: Path,
    model_name: str,
    train: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, PredictionCandidate, float]:
    candidates = collect_prediction_candidates(artifact_dir, model_name, len(train))
    if not candidates:
        raise RuntimeError(
            f"No full-length [0,1] prediction vector found in {artifact_dir}"
        )

    evaluated = []
    for c in candidates:
        try:
            p = align_prediction(c, train)
            auc = safe_auc(y, p)
        except Exception:
            continue
        evaluated.append((c, p, auc))

    if not evaluated:
        raise RuntimeError(
            f"Prediction candidates existed in {artifact_dir}, but none could be "
            "safely aligned to train.csv."
        )

    expected = EXPECTED_AUC[model_name]

    # First use known validated AUC as an integrity check/selector.
    evaluated.sort(
        key=lambda t: (
            abs(t[2] - expected),
            -t[0].score,
            str(t[0].source),
            t[0].column,
        )
    )

    best_c, best_p, best_auc = evaluated[0]

    if abs(best_auc - expected) > AUC_VERIFY_TOL:
        lines = [
            f"Could not verify {model_name} OOF against expected AUC {expected:.8f}.",
            "Closest discovered candidates:",
        ]
        for c, _, auc in evaluated[:12]:
            lines.append(
                f"  AUC={auc:.8f} heuristic={c.score:>3} "
                f"path={c.source.relative_to(ROOT)} column={c.column}"
            )
        raise RuntimeError("\n".join(lines))

    # If two candidates have effectively identical distance to expected, prefer
    # the stronger filename/column heuristic. This avoids silently taking a
    # random seed file when a rank-average OOF is present.
    near = [
        t
        for t in evaluated
        if abs(abs(t[2] - expected) - abs(best_auc - expected)) <= 1e-10
    ]
    if len(near) > 1:
        near.sort(
            key=lambda t: (-t[0].score, str(t[0].source), t[0].column)
        )
        best_c, best_p, best_auc = near[0]

    return best_p, best_c, best_auc


def percentile_rank(p: np.ndarray) -> np.ndarray:
    return pd.Series(p).rank(method="average", pct=True).to_numpy(dtype=float)


def auc_error_contribution(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Per-row AUC-ranking error contribution.

    Positive row:
      fraction of negatives ranked above it (+ 0.5 ties)
    Negative row:
      fraction of positives ranked below it (+ 0.5 ties)

    Mean contribution within either class equals 1 - AUC globally.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)

    pos_vals = np.sort(p[y == 1])
    neg_vals = np.sort(p[y == 0])
    n_pos = len(pos_vals)
    n_neg = len(neg_vals)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required for AUC error contributions.")

    out = np.empty(len(y), dtype=float)

    pos_mask = y == 1
    pv = p[pos_mask]
    neg_lt = np.searchsorted(neg_vals, pv, side="left")
    neg_le = np.searchsorted(neg_vals, pv, side="right")
    neg_gt = n_neg - neg_le
    neg_eq = neg_le - neg_lt
    out[pos_mask] = (neg_gt + 0.5 * neg_eq) / n_neg

    nv = p[~pos_mask]
    pos_lt = np.searchsorted(pos_vals, nv, side="left")
    pos_le = np.searchsorted(pos_vals, nv, side="right")
    pos_eq = pos_le - pos_lt
    out[~pos_mask] = (pos_lt + 0.5 * pos_eq) / n_pos

    return out


def format_segment_value(v) -> str:
    if pd.isna(v):
        return "<NA>"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def make_audit_frame(
    train: pd.DataFrame,
    folds: np.ndarray,
    y: np.ndarray,
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    df = train.copy()
    df["_y"] = y
    df["_fold"] = folds

    for name, p in preds.items():
        df[f"_p_{name}"] = p
        df[f"_r_{name}"] = percentile_rank(p)

    df["_champion_auc_error"] = auc_error_contribution(y, preds["CHAMPION"])

    df["_disagree_cat_xgb"] = np.abs(df["_r_CAT"] - df["_r_XGB"])
    df["_disagree_xgb_lgbm"] = np.abs(df["_r_XGB"] - df["_r_LGBM"])
    df["_disagree_cat_lgbm"] = np.abs(df["_r_CAT"] - df["_r_LGBM"])
    df["_disagree_max"] = np.maximum.reduce(
        [
            df["_disagree_cat_xgb"].to_numpy(),
            df["_disagree_xgb_lgbm"].to_numpy(),
            df["_disagree_cat_lgbm"].to_numpy(),
        ]
    )

    # Fixed, predeclared structural regimes from the handoff.
    income = pd.to_numeric(df["Annual_Income_USD"], errors="raise")
    commute = pd.to_numeric(df["Daily_Commute_km"], errors="raise")

    df["audit_income_1k"] = (np.floor(income / 1_000) * 1_000).astype("int64")
    df["audit_income_10k"] = (np.floor(income / 10_000) * 10_000).astype("int64")
    df["audit_income_100k"] = (np.floor(income / 100_000) * 100_000).astype("int64")

    df["audit_commute_1km"] = np.floor(commute / 1).astype("int64")
    df["audit_commute_5km"] = (np.floor(commute / 5) * 5).astype("int64")
    df["audit_commute_10km"] = (np.floor(commute / 10) * 10).astype("int64")

    # Disagreement decile is diagnostic only; it is not proposed as a deployable
    # feature/regime because it depends on model predictions.
    df["audit_disagreement_decile"] = pd.qcut(
        df["_disagree_max"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    ).astype(str)

    return df


def segment_report_for_feature(
    df: pd.DataFrame, feature: str
) -> pd.DataFrame:
    rows = []

    for value, g in df.groupby(feature, dropna=False, observed=False):
        n = len(g)
        pos = int(g["_y"].sum())
        neg = n - pos

        if n < MIN_SEGMENT_ROWS or pos < MIN_SEGMENT_CLASS or neg < MIN_SEGMENT_CLASS:
            continue

        aucs = {
            name: safe_auc(
                g["_y"].to_numpy(),
                g[f"_p_{name}"].to_numpy(),
            )
            for name in ("CAT", "XGB", "LGBM", "CHAMPION")
        }

        member_aucs = {k: aucs[k] for k in ("CAT", "XGB", "LGBM")}
        best_member = max(member_aucs, key=member_aucs.get)
        best_member_auc = member_aucs[best_member]

        fold_wins = {"CAT": 0, "XGB": 0, "LGBM": 0, "CHAMPION": 0}
        evaluable_folds = 0

        for _, fg in g.groupby("_fold"):
            fy = fg["_y"].to_numpy()
            if len(fg) < 100 or np.unique(fy).size < 2:
                continue
            fold_aucs = {
                name: safe_auc(fy, fg[f"_p_{name}"].to_numpy())
                for name in fold_wins
            }
            if any(np.isnan(v) for v in fold_aucs.values()):
                continue
            winner = max(fold_aucs, key=fold_aucs.get)
            fold_wins[winner] += 1
            evaluable_folds += 1

        pos_err = g.loc[g["_y"] == 1, "_champion_auc_error"].mean()
        neg_err = g.loc[g["_y"] == 0, "_champion_auc_error"].mean()
        balanced_rank_error = 0.5 * (pos_err + neg_err)

        rows.append(
            {
                "feature": feature,
                "segment": format_segment_value(value),
                "n": n,
                "positives": pos,
                "positive_rate": pos / n,
                "cat_auc": aucs["CAT"],
                "xgb_auc": aucs["XGB"],
                "lgbm_auc": aucs["LGBM"],
                "champion_auc": aucs["CHAMPION"],
                "best_member": best_member,
                "best_member_auc": best_member_auc,
                "best_member_minus_champion": best_member_auc - aucs["CHAMPION"],
                "champion_rank_error_pos": pos_err,
                "champion_rank_error_neg": neg_err,
                "champion_rank_error_balanced": balanced_rank_error,
                "cat_xgb_rank_disagreement": g["_disagree_cat_xgb"].mean(),
                "xgb_lgbm_rank_disagreement": g["_disagree_xgb_lgbm"].mean(),
                "cat_lgbm_rank_disagreement": g["_disagree_cat_lgbm"].mean(),
                "max_rank_disagreement": g["_disagree_max"].mean(),
                "evaluable_folds": evaluable_folds,
                "cat_fold_wins": fold_wins["CAT"],
                "xgb_fold_wins": fold_wins["XGB"],
                "lgbm_fold_wins": fold_wins["LGBM"],
                "champion_fold_wins": fold_wins["CHAMPION"],
            }
        )

    return pd.DataFrame(rows)


def crossfit_regime_router(
    df: pd.DataFrame,
    feature: str,
) -> dict:
    """
    Leave-one-frozen-fold-out diagnostic router.

    For every outer fold, choose the best prediction source
    (CHAMPION/CAT/XGB/LGBM) for each segment using only the other 4 OOF folds,
    then apply that choice to the held-out fold.

    This does NOT train a base model. It is an audit for stability of
    regime-specific model preference.

    Implementation note:
    routing application is fully vectorized. The earlier implementation used
    per-row DataFrame.iloc lookups and was unnecessarily slow on 668k rows.
    """
    sources = ("CHAMPION", "CAT", "XGB", "LGBM")
    source_to_code = {name: i for i, name in enumerate(sources)}
    code_to_source = np.array(sources, dtype=object)

    n_rows = len(df)
    routed = np.empty(n_rows, dtype=float)
    route_code = np.zeros(n_rows, dtype=np.int8)  # default CHAMPION

    y_all = df["_y"].to_numpy(dtype=np.int8, copy=False)
    fold_all = df["_fold"].to_numpy(copy=False)
    feature_all = df[feature]
    pred_matrix = np.column_stack(
        [df[f"_p_{source}"].to_numpy(dtype=float, copy=False) for source in sources]
    )

    outer_rows = []
    selection_rows = []

    for fold in sorted(np.unique(fold_all)):
        tr_mask = fold_all != fold
        va_mask = fold_all == fold

        tr_df = df.loc[tr_mask, [feature, "_y"] + [f"_p_{s}" for s in sources]]

        selected = {}

        for value, g in tr_df.groupby(feature, dropna=False, observed=False, sort=False):
            n = len(g)
            pos = int(g["_y"].sum())
            neg = n - pos

            if (
                n < MIN_SEGMENT_ROWS
                or pos < MIN_SEGMENT_CLASS
                or neg < MIN_SEGMENT_CLASS
            ):
                selected[value] = "CHAMPION"
                continue

            gy = g["_y"].to_numpy(dtype=np.int8, copy=False)
            aucs = {
                source: safe_auc(
                    gy,
                    g[f"_p_{source}"].to_numpy(dtype=float, copy=False),
                )
                for source in sources
            }
            winner = max(aucs, key=aucs.get)
            selected[value] = winner

            selection_rows.append(
                {
                    "outer_fold": int(fold),
                    "feature": feature,
                    "segment": format_segment_value(value),
                    "train_n": n,
                    "train_positives": pos,
                    "selected_source": winner,
                    "champion_auc_train4": aucs["CHAMPION"],
                    "cat_auc_train4": aucs["CAT"],
                    "xgb_auc_train4": aucs["XGB"],
                    "lgbm_auc_train4": aucs["LGBM"],
                }
            )

        # Vectorized held-out routing.
        # All current audit features are non-missing; fillna is retained as a
        # safe fallback to CHAMPION should an unmapped/unsupported value occur.
        va_feature = feature_all.loc[va_mask]
        chosen_names = va_feature.map(selected).fillna("CHAMPION")
        chosen_codes = chosen_names.map(source_to_code).to_numpy(dtype=np.int8)

        va_idx = np.flatnonzero(va_mask)
        route_code[va_idx] = chosen_codes
        routed[va_idx] = pred_matrix[va_idx, chosen_codes]

        y_fold = y_all[va_mask]
        champion_fold = pred_matrix[va_mask, source_to_code["CHAMPION"]]
        routed_fold = routed[va_mask]

        champion_auc = safe_auc(y_fold, champion_fold)
        routed_auc = safe_auc(y_fold, routed_fold)

        fold_codes = route_code[va_mask]
        outer_rows.append(
            {
                "feature": feature,
                "fold": int(fold),
                "champion_auc": champion_auc,
                "routed_auc": routed_auc,
                "delta": routed_auc - champion_auc,
                "pct_champion": float(
                    np.mean(fold_codes == source_to_code["CHAMPION"])
                ),
                "pct_cat": float(np.mean(fold_codes == source_to_code["CAT"])),
                "pct_xgb": float(np.mean(fold_codes == source_to_code["XGB"])),
                "pct_lgbm": float(np.mean(fold_codes == source_to_code["LGBM"])),
            }
        )

    overall_champion = safe_auc(
        y_all, pred_matrix[:, source_to_code["CHAMPION"]]
    )
    overall_routed = safe_auc(y_all, routed)

    outer = pd.DataFrame(outer_rows)
    selections = pd.DataFrame(selection_rows)
    route_source = code_to_source[route_code]

    return {
        "feature": feature,
        "routed_pred": routed,
        "route_source": route_source,
        "overall_champion_auc": overall_champion,
        "overall_routed_auc": overall_routed,
        "overall_delta": overall_routed - overall_champion,
        "folds_improved": int((outer["delta"] > 0).sum()),
        "folds_worse": int((outer["delta"] < 0).sum()),
        "outer": outer,
        "selections": selections,
    }

def main() -> None:
    print("=" * 78)
    print("CHAMPION OOF RESIDUAL / DISAGREEMENT AUDIT")
    print("NO MODEL TRAINING")
    print("=" * 78)

    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"Frozen folds file not found: {FOLDS_PATH}")

    actual_hash = sha256_file(FOLDS_PATH)
    print(f"\nFrozen fold file: {FOLDS_PATH.relative_to(ROOT)}")
    print(f"SHA256: {actual_hash}")
    if actual_hash != EXPECTED_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen fold hash mismatch.\n"
            f"Expected: {EXPECTED_FOLDS_SHA256}\n"
            f"Actual:   {actual_hash}"
        )
    print("[PASS] Frozen validation hash verified.")

    train_path = discover_train_csv()
    print(f"\nDiscovered train: {train_path.relative_to(ROOT)}")
    train = pd.read_csv(train_path)

    if len(train) != EXPECTED_N:
        raise RuntimeError(f"Unexpected train row count: {len(train):,}")
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in train.columns]
    if missing:
        raise RuntimeError(f"Missing expected train columns: {missing}")

    y = (train[TARGET].astype(str) == POSITIVE_LABEL).astype(np.int8).to_numpy()
    print(
        f"Rows: {len(train):,} | positives: {int(y.sum()):,} "
        f"| positive rate: {y.mean():.6f}"
    )

    folds_df = pd.read_csv(FOLDS_PATH)
    folds = align_folds(train, folds_df)
    unique_folds = sorted(np.unique(folds).tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected folds [0,1,2,3,4], got {unique_folds}")
    print(f"Frozen folds aligned: {unique_folds}")

    preds: dict[str, np.ndarray] = {}
    selected_rows = []

    print("\n--- OOF artifact integrity check ---")
    for model_name in ("CAT", "XGB", "LGBM", "CHAMPION"):
        p, c, auc = select_oof_candidate(
            ARTIFACT_DIRS[model_name], model_name, train, y
        )
        preds[model_name] = p
        rel = c.source.relative_to(ROOT)
        print(
            f"{model_name:8s} AUC={auc:.8f} "
            f"(expected {EXPECTED_AUC[model_name]:.8f})\n"
            f"          source={rel} | column={c.column}"
        )
        selected_rows.append(
            {
                "model": model_name,
                "auc": auc,
                "expected_auc": EXPECTED_AUC[model_name],
                "auc_delta_vs_expected": auc - EXPECTED_AUC[model_name],
                "source": str(rel),
                "column": c.column,
            }
        )

    print("\n[PASS] All four OOF vectors match the validated handoff scores.")

    audit = make_audit_frame(train, folds, y, preds)

    print("\n--- Global rank correlations ---")
    rank_cols = ["_r_CAT", "_r_XGB", "_r_LGBM", "_r_CHAMPION"]
    corr = audit[rank_cols].corr(method="pearson")
    print(corr.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n--- Global disagreement ---")
    for col in (
        "_disagree_cat_xgb",
        "_disagree_xgb_lgbm",
        "_disagree_cat_lgbm",
        "_disagree_max",
    ):
        x = audit[col]
        print(
            f"{col:27s} mean={x.mean():.6f} "
            f"p90={x.quantile(.90):.6f} p99={x.quantile(.99):.6f}"
        )

    segment_features = [
        "Environmental_Concern_Level",
        "audit_income_1k",
        "audit_income_10k",
        "audit_income_100k",
        "audit_commute_1km",
        "audit_commute_5km",
        "audit_commute_10km",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Current_Car_Type",
        "City_Type",
        "Home_Charging_Possible",
        "Gender",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "audit_disagreement_decile",
    ]

    reports = []
    for feature in segment_features:
        rep = segment_report_for_feature(audit, feature)
        if not rep.empty:
            reports.append(rep)

    segment_report = (
        pd.concat(reports, ignore_index=True)
        if reports
        else pd.DataFrame()
    )

    if segment_report.empty:
        raise RuntimeError("No segment met the minimum support thresholds.")

    # Structural shortlist: large enough slices where a member beats champion
    # and the same source wins repeatedly across folds.
    seg = segment_report.copy()
    member_win_cols = {
        "CAT": "cat_fold_wins",
        "XGB": "xgb_fold_wins",
        "LGBM": "lgbm_fold_wins",
    }
    seg["best_member_fold_wins"] = [
        row[member_win_cols[row["best_member"]]]
        for _, row in seg.iterrows()
    ]
    seg["stable_member_signal"] = (
        (seg["best_member_minus_champion"] > 0)
        & (seg["best_member_fold_wins"] >= 4)
        & (seg["evaluable_folds"] >= 4)
    )

    print("\n--- Highest champion AUC-ranking-error segments ---")
    top_error = seg.sort_values(
        ["champion_rank_error_balanced", "n"],
        ascending=[False, False],
    ).head(20)
    print(
        top_error[
            [
                "feature",
                "segment",
                "n",
                "positive_rate",
                "champion_auc",
                "champion_rank_error_balanced",
                "max_rank_disagreement",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n--- Highest model-disagreement segments ---")
    top_disagree = seg.sort_values(
        ["max_rank_disagreement", "n"],
        ascending=[False, False],
    ).head(20)
    print(
        top_disagree[
            [
                "feature",
                "segment",
                "n",
                "champion_auc",
                "best_member",
                "best_member_minus_champion",
                "max_rank_disagreement",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    stable = seg.loc[seg["stable_member_signal"]].sort_values(
        ["best_member_minus_champion", "n"],
        ascending=[False, False],
    )
    print("\n--- Stable segment-level member advantage (>=4 fold wins) ---")
    if stable.empty:
        print("None.")
    else:
        print(
            stable[
                [
                    "feature",
                    "segment",
                    "n",
                    "best_member",
                    "best_member_auc",
                    "champion_auc",
                    "best_member_minus_champion",
                    "best_member_fold_wins",
                    "evaluable_folds",
                ]
            ]
            .head(30)
            .to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )

    # Cross-fitted diagnostic routing is run only on deployable/raw-data regimes.
    route_features = [
        "Environmental_Concern_Level",
        "audit_income_10k",
        "audit_income_100k",
        "audit_commute_1km",
        "audit_commute_5km",
        "audit_commute_10km",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Current_Car_Type",
        "City_Type",
        "Home_Charging_Possible",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ]

    router_summaries = []
    outer_frames = []
    selection_frames = []

    print("\n--- Leave-one-fold-out regime routing audit ---")
    print(
        "Each held-out fold uses only the other 4 folds to decide whether a "
        "segment should use CHAMPION/CAT/XGB/LGBM."
    )

    for feature in route_features:
        result = crossfit_regime_router(audit, feature)
        router_summaries.append(
            {
                "feature": feature,
                "champion_auc": result["overall_champion_auc"],
                "routed_auc": result["overall_routed_auc"],
                "delta": result["overall_delta"],
                "folds_improved": result["folds_improved"],
                "folds_worse": result["folds_worse"],
            }
        )
        outer_frames.append(result["outer"])
        if not result["selections"].empty:
            selection_frames.append(result["selections"])

    router_summary = pd.DataFrame(router_summaries).sort_values(
        ["delta", "folds_improved"], ascending=[False, False]
    )
    print(
        router_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    best_router = router_summary.iloc[0]
    print("\n--- Audit conclusion primitive ---")
    print(
        f"Best cross-fitted single-regime router: {best_router['feature']}\n"
        f"Champion AUC: {best_router['champion_auc']:.8f}\n"
        f"Routed AUC:   {best_router['routed_auc']:.8f}\n"
        f"Delta:        {best_router['delta']:+.8f}\n"
        f"Folds improved: {int(best_router['folds_improved'])}/5"
    )

    if (
        best_router["delta"] > 0
        and int(best_router["folds_improved"]) >= 4
    ):
        print(
            "SIGNAL: There is cross-fold evidence that a regime-aware next "
            "experiment may be justified."
        )
    else:
        print(
            "NO CLEAR ROUTING SIGNAL: Do not assume regime-specific weighting "
            "will help from this audit alone."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "selected_oof_artifacts.csv", index=False
    )
    corr.to_csv(OUTPUT_DIR / "global_rank_correlations.csv")
    seg.to_csv(OUTPUT_DIR / "segment_audit.csv", index=False)
    router_summary.to_csv(OUTPUT_DIR / "router_summary.csv", index=False)
    pd.concat(outer_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "router_fold_results.csv", index=False
    )
    if selection_frames:
        pd.concat(selection_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "router_segment_selections.csv", index=False
        )

    print(f"\nSaved audit outputs to: {OUTPUT_DIR.relative_to(ROOT)}")
    print("Done. No models were trained.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("AUDIT FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)
