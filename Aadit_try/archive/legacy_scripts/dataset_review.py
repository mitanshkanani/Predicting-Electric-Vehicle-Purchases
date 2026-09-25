from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import ks_2samp
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scipy. Install dependencies with:\n"
        "pip install pandas numpy scipy scikit-learn matplotlib"
    ) from exc

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn. Install dependencies with:\n"
        "pip install pandas numpy scipy scikit-learn matplotlib"
    ) from exc


SEED = 42
RNG = np.random.default_rng(SEED)
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
KS_SAMPLE_SIZE = 50_000
REL_SAMPLE_SIZE = 120_000
CAT_SAMPLE_SIZE = 150_000
MAX_PLOT_FEATURES = 6
MAX_CATEGORY_PLOT_LEVELS = 15
NEAR_CONSTANT_THRESHOLD = 0.995
HIGH_CARDINALITY_MIN = 50


@dataclass
class DetectionResult:
    target: str
    likely_id: str | None
    id_scores: dict[str, float]
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive train/test audit for a Kaggle tabular binary-classification competition."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_review"))
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def sample_series(s: pd.Series, max_n: int, seed: int = SEED) -> pd.Series:
    if len(s) <= max_n:
        return s
    return s.sample(max_n, random_state=seed)


def finite_numeric(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    arr = values.to_numpy(dtype=float, na_value=np.nan)
    mask = np.isfinite(arr)
    return pd.Series(arr[mask], index=values.index[mask], name=s.name)


def canonical_category(s: pd.Series) -> pd.Series:
    # String conversion makes train/test category comparison stable across mixed object dtypes.
    return s.astype("string").fillna(MISSING_TOKEN)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned[:120] or "feature"


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    train_only = [c for c in train.columns if c not in test.columns]

    if len(train_only) == 1:
        target = train_only[0]
        reasons.append(f"Only column present in train but absent from test: {target!r}.")
        return target, reasons

    sample_candidates = [c for c in sample.columns if c in train.columns and c not in test.columns]
    if len(sample_candidates) == 1:
        target = sample_candidates[0]
        reasons.append(
            f"Matched the sole train-only column also present in sample_submission: {target!r}."
        )
        return target, reasons

    common_sample_test = [c for c in sample.columns if c in test.columns]
    non_id_sample_cols = [c for c in sample.columns if c not in common_sample_test]
    fallback = [c for c in non_id_sample_cols if c in train.columns]
    if len(fallback) == 1:
        target = fallback[0]
        reasons.append(
            f"Inferred from sample_submission output column not shared with test: {target!r}."
        )
        return target, reasons

    name_candidates = [
        c for c in train_only
        if str(c).lower() in {"target", "label", "y", "class", "outcome", "response", "purchase", "purchased"}
    ]
    if len(name_candidates) == 1:
        target = name_candidates[0]
        reasons.append(f"Inferred from a conventional target-like name: {target!r}.")
        return target, reasons

    raise ValueError(
        "Could not safely infer the target column. "
        f"Train-only columns: {train_only}; sample columns: {list(sample.columns)}. "
        "This script intentionally refuses to guess when inference is ambiguous."
    )


def infer_id(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
) -> tuple[str | None, dict[str, float], list[str]]:
    reasons: list[str] = []
    candidates = [c for c in test.columns if c in train.columns and c != target]
    scores: dict[str, float] = {}

    for c in candidates:
        tr = train[c]
        te = test[c]
        tr_unique = tr.nunique(dropna=False)
        te_unique = te.nunique(dropna=False)
        score = 0.0
        lower = str(c).lower()

        if lower == "id":
            score += 6.0
        elif lower.endswith("_id") or lower.startswith("id_"):
            score += 5.0
        elif "identifier" in lower:
            score += 4.0
        elif "id" in lower:
            score += 2.0

        tr_ratio = safe_ratio(tr_unique, len(tr))
        te_ratio = safe_ratio(te_unique, len(test))
        if tr_ratio >= 0.999:
            score += 2.5
        elif tr_ratio >= 0.98:
            score += 1.5
        if te_ratio >= 0.999:
            score += 1.5
        elif te_ratio >= 0.98:
            score += 0.75

        if c in sample.columns:
            score += 3.0
            if len(sample) == len(test):
                try:
                    same = sample[c].reset_index(drop=True).equals(te.reset_index(drop=True))
                except Exception:
                    same = False
                if same:
                    score += 5.0

        scores[c] = score

    if not scores:
        return None, scores, ["No common train/test column was available as an ID candidate."]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_col, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -np.inf

    if best_score >= 5.0 and (best_score - second_score >= 1.0 or best_score >= 9.0):
        reasons.append(f"Likely ID: {best_col!r} (heuristic score {best_score:.2f}).")
        return best_col, scores, reasons

    reasons.append(
        "No ID column was inferred with enough confidence. "
        f"Highest heuristic candidate was {best_col!r} with score {best_score:.2f}."
    )
    return None, scores, reasons


def validate_binary_target(y: pd.Series) -> tuple[list[Any], dict[Any, int], float | None]:
    values = list(pd.unique(y.dropna()))
    counts = y.value_counts(dropna=False).to_dict()
    positive_rate: float | None = None

    if len(values) == 2:
        # Prefer {0,1}; otherwise use the less frequent class only for a descriptive rate.
        if set(values) == {0, 1} or set(values) == {0.0, 1.0}:
            positive_rate = float((y == 1).mean())
        else:
            vc = y.value_counts(dropna=True)
            if len(vc) == 2:
                positive_label = vc.index[-1]
                positive_rate = float((y == positive_label).mean())
    return values, counts, positive_rate


def encode_binary_target(y: pd.Series) -> tuple[pd.Series, Any]:
    non_null = y.dropna()
    values = list(pd.unique(non_null))
    if len(values) != 2:
        raise ValueError(f"Expected binary target, found {len(values)} non-null values: {values[:10]}")

    if set(values) == {0, 1} or set(values) == {0.0, 1.0}:
        return y.astype(float), 1

    counts = non_null.value_counts()
    positive_label = counts.index[-1]
    encoded = (y == positive_label).astype(float)
    encoded[y.isna()] = np.nan
    return encoded, positive_label


def decimal_precision_95(s: pd.Series, max_decimals: int = 8) -> int | None:
    x = finite_numeric(sample_series(s.dropna(), 20_000))
    if x.empty:
        return None
    arr = x.to_numpy(dtype=float)
    scale = np.maximum(1.0, np.abs(arr))
    for d in range(max_decimals + 1):
        rounded = np.round(arr, d)
        # Tight relative tolerance: identifies values effectively stored at d decimal places.
        close = np.abs(arr - rounded) <= (1e-10 * scale + 10 ** (-(d + 9)))
        if float(np.mean(close)) >= 0.95:
            return d
    return None


def classify_column(s: pd.Series, name: str, n_rows: int) -> dict[str, Any]:
    nunique = int(s.nunique(dropna=True))
    ratio = safe_ratio(nunique, n_rows)
    is_numeric = pd.api.types.is_numeric_dtype(s)
    is_bool = pd.api.types.is_bool_dtype(s)
    is_integer_dtype = pd.api.types.is_integer_dtype(s)

    name_lower = str(name).lower()
    ordinal_name_hint = bool(
        re.search(r"(^|_)(level|grade|tier|stage|rank|rating|band|class|category|cat|group|segment)($|_)", name_lower)
    )

    if is_bool or nunique <= 2:
        kind = "binary"
    elif not is_numeric:
        kind = "categorical"
    else:
        integer_like = is_integer_dtype
        if not integer_like:
            sample = finite_numeric(sample_series(s.dropna(), 20_000))
            if not sample.empty:
                integer_like = bool(np.mean(np.isclose(sample, np.round(sample), atol=1e-10)) >= 0.999)

        discrete_cutoff = max(20, min(100, int(math.sqrt(max(n_rows, 1)))))
        if nunique <= discrete_cutoff:
            kind = "discrete_numeric"
        else:
            kind = "continuous_numeric"

        if integer_like and nunique <= max(50, int(math.sqrt(max(n_rows, 1)))):
            integer_category_candidate = True
        else:
            integer_category_candidate = False

        return {
            "feature_kind": kind,
            "is_numeric": True,
            "integer_like": integer_like,
            "integer_category_candidate": integer_category_candidate,
            "ordinal_name_hint": ordinal_name_hint,
            "high_cardinality": False,
        }

    high_cardinality = kind == "categorical" and (
        nunique >= HIGH_CARDINALITY_MIN and (ratio >= 0.02 or nunique >= 500)
    )
    return {
        "feature_kind": kind,
        "is_numeric": is_numeric,
        "integer_like": bool(is_integer_dtype),
        "integer_category_candidate": False,
        "ordinal_name_hint": ordinal_name_hint,
        "high_cardinality": high_cardinality,
    }


def top_frequency_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return np.nan
    vc = s.value_counts(dropna=False, sort=True)
    return float(vc.iloc[0] / len(s)) if len(vc) else np.nan


def build_column_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    likely_id: str | None,
    id_scores: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in train.columns:
        s = train[c]
        n = len(train)
        nunique = int(s.nunique(dropna=True))
        missing = int(s.isna().sum())
        info = classify_column(s, c, n)
        top_ratio = top_frequency_ratio(s)
        constant = nunique <= 1
        near_constant = bool(not constant and top_ratio >= NEAR_CONSTANT_THRESHOLD)
        unique_ratio = safe_ratio(nunique, n)
        # High uniqueness alone does NOT make a continuous float an ID. Many genuine
        # measurements are unique row-by-row. Treat high uniqueness as ID-like only
        # when the values are integer-like (or the stronger ID heuristics already fired).
        id_like = bool(
            c == likely_id
            or id_scores.get(c, 0) >= 5.0
            or (
                unique_ratio >= 0.995
                and nunique > 100
                and bool(info["integer_like"])
                and c != target
            )
        )

        rows.append({
            "column": c,
            "in_test": c in test.columns,
            "is_target": c == target,
            "is_likely_id": c == likely_id,
            "id_heuristic_score": id_scores.get(c, np.nan),
            "pandas_dtype": str(s.dtype),
            "feature_kind": "target" if c == target else info["feature_kind"],
            "n_unique": nunique,
            "unique_ratio": unique_ratio,
            "missing_count": missing,
            "missing_pct": 100 * safe_ratio(missing, n),
            "top_frequency_ratio": top_ratio,
            "constant": constant,
            "near_constant": near_constant,
            "high_cardinality": info["high_cardinality"],
            "integer_like": info["integer_like"],
            "integer_category_candidate": info["integer_category_candidate"],
            "ordinal_name_hint": info["ordinal_name_hint"],
            "id_like": id_like,
        })
    return pd.DataFrame(rows)


def build_numerical_summary(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        raw = pd.to_numeric(train[c], errors="coerce")
        arr = raw.to_numpy(dtype=float, na_value=np.nan)
        inf_count = int(np.isinf(arr).sum())
        finite = raw[np.isfinite(arr)]
        q = finite.quantile(quantiles) if len(finite) else pd.Series(index=quantiles, dtype=float)

        row: dict[str, Any] = {
            "column": c,
            "count_finite": int(len(finite)),
            "missing_count": int(raw.isna().sum()),
            "infinite_count": inf_count,
            "mean": finite.mean() if len(finite) else np.nan,
            "std": finite.std() if len(finite) else np.nan,
            "min": finite.min() if len(finite) else np.nan,
            "max": finite.max() if len(finite) else np.nan,
            "skew": finite.skew() if len(finite) > 2 else np.nan,
            "n_unique": int(raw.nunique(dropna=True)),
            "decimal_precision_95": decimal_precision_95(raw),
        }
        for qq in quantiles:
            row[f"q{int(qq * 100):02d}"] = q.get(qq, np.nan)
        if len(finite):
            abs_nonzero = np.abs(finite.to_numpy(dtype=float))
            abs_nonzero = abs_nonzero[abs_nonzero > 0]
            row["max_abs"] = float(np.max(np.abs(finite)))
            row["dynamic_range_abs"] = (
                float(np.max(abs_nonzero) / np.min(abs_nonzero)) if len(abs_nonzero) else np.nan
            )
        else:
            row["max_abs"] = np.nan
            row["dynamic_range_abs"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_categorical_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    column_summary: pd.DataFrame,
) -> pd.DataFrame:
    kind_map = column_summary.set_index("column")["feature_kind"].to_dict()
    rows: list[dict[str, Any]] = []

    for c in feature_cols:
        kind = kind_map.get(c, "")
        if kind not in {"categorical", "binary", "discrete_numeric"}:
            continue

        tr = canonical_category(train[c])
        te = canonical_category(test[c]) if c in test.columns else pd.Series(dtype="string")
        tr_vc = tr.value_counts(dropna=False)
        te_vc = te.value_counts(dropna=False) if len(te) else pd.Series(dtype=int)
        rows.append({
            "column": c,
            "feature_kind": kind,
            "train_n_unique": int(train[c].nunique(dropna=True)),
            "test_n_unique": int(test[c].nunique(dropna=True)) if c in test.columns else np.nan,
            "train_top_value": str(tr_vc.index[0]) if len(tr_vc) else None,
            "train_top_count": int(tr_vc.iloc[0]) if len(tr_vc) else 0,
            "train_top_pct": 100 * float(tr_vc.iloc[0] / len(tr)) if len(tr_vc) else np.nan,
            "test_top_value": str(te_vc.index[0]) if len(te_vc) else None,
            "test_top_count": int(te_vc.iloc[0]) if len(te_vc) else 0,
            "test_top_pct": 100 * float(te_vc.iloc[0] / len(te)) if len(te_vc) else np.nan,
        })

    return pd.DataFrame(rows)


def psi_from_numeric(train_s: pd.Series, test_s: pd.Series, bins: int = 10) -> float:
    tr = finite_numeric(train_s)
    te = finite_numeric(test_s)
    if len(tr) < 20 or len(te) < 20 or tr.nunique() < 2:
        return np.nan

    edges = np.unique(np.quantile(tr.to_numpy(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf

    tr_hist, _ = np.histogram(tr, bins=edges)
    te_hist, _ = np.histogram(te, bins=edges)
    tr_p = np.clip(tr_hist / max(tr_hist.sum(), 1), 1e-6, None)
    te_p = np.clip(te_hist / max(te_hist.sum(), 1), 1e-6, None)
    return float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))


def categorical_shift_metrics(train_s: pd.Series, test_s: pd.Series) -> dict[str, Any]:
    tr_full_nunique = int(train_s.nunique(dropna=False))
    te_full_nunique = int(test_s.nunique(dropna=False))

    tr = canonical_category(sample_series(train_s, CAT_SAMPLE_SIZE, SEED))
    te = canonical_category(sample_series(test_s, CAT_SAMPLE_SIZE, SEED + 1))

    tr_vc = tr.value_counts(normalize=True)
    te_vc = te.value_counts(normalize=True)

    # Keep dominant levels and combine the long tail to avoid huge unions for ID-like strings.
    dominant = list((tr_vc.add(te_vc, fill_value=0)).nlargest(500).index)
    tr_top = tr_vc.reindex(dominant, fill_value=0.0)
    te_top = te_vc.reindex(dominant, fill_value=0.0)
    tr_other = max(0.0, 1.0 - float(tr_top.sum()))
    te_other = max(0.0, 1.0 - float(te_top.sum()))
    p = np.append(tr_top.to_numpy(dtype=float), tr_other)
    q = np.append(te_top.to_numpy(dtype=float), te_other)
    js = float(jensenshannon(p, q, base=2.0) ** 2)

    tr_levels = set(tr.unique().tolist())
    te_levels = set(te.unique().tolist())
    test_unseen_mask = ~te.isin(tr_levels)
    train_only_sample = tr_levels - te_levels
    test_only_sample = te_levels - tr_levels

    return {
        "train_n_unique": tr_full_nunique,
        "test_n_unique": te_full_nunique,
        "cardinality_difference": te_full_nunique - tr_full_nunique,
        "js_divergence": js,
        "sample_train_only_levels": len(train_only_sample),
        "sample_test_only_levels": len(test_only_sample),
        "sample_test_unseen_row_pct": 100 * float(test_unseen_mask.mean()),
    }


def build_shift_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    column_summary: pd.DataFrame,
) -> pd.DataFrame:
    meta = column_summary.set_index("column").to_dict("index")
    rows: list[dict[str, Any]] = []

    for c in feature_cols:
        if c not in test.columns:
            continue
        m = meta[c]
        row: dict[str, Any] = {
            "column": c,
            "feature_kind": m["feature_kind"],
            "is_likely_id": m["is_likely_id"],
            "id_like": m["id_like"],
        }

        if pd.api.types.is_numeric_dtype(train[c]) and m["feature_kind"] == "continuous_numeric":
            tr_full = finite_numeric(train[c])
            te_full = finite_numeric(test[c])
            tr = sample_series(tr_full, KS_SAMPLE_SIZE, SEED)
            te = sample_series(te_full, KS_SAMPLE_SIZE, SEED + 1)

            if len(tr) and len(te):
                ks = ks_2samp(tr.to_numpy(), te.to_numpy(), method="auto")
                pooled_std = float(np.sqrt((np.nanvar(tr, ddof=1) + np.nanvar(te, ddof=1)) / 2))
                smd = (
                    float((np.nanmean(te) - np.nanmean(tr)) / pooled_std)
                    if pooled_std > 0 and np.isfinite(pooled_std)
                    else np.nan
                )
                row.update({
                    "shift_method": "numeric",
                    "train_mean": float(np.nanmean(tr_full)) if len(tr_full) else np.nan,
                    "test_mean": float(np.nanmean(te_full)) if len(te_full) else np.nan,
                    "train_std": float(np.nanstd(tr_full, ddof=1)) if len(tr_full) > 1 else np.nan,
                    "test_std": float(np.nanstd(te_full, ddof=1)) if len(te_full) > 1 else np.nan,
                    "train_median": float(np.nanmedian(tr_full)) if len(tr_full) else np.nan,
                    "test_median": float(np.nanmedian(te_full)) if len(te_full) else np.nan,
                    "train_min": float(np.nanmin(tr_full)) if len(tr_full) else np.nan,
                    "test_min": float(np.nanmin(te_full)) if len(te_full) else np.nan,
                    "train_max": float(np.nanmax(tr_full)) if len(tr_full) else np.nan,
                    "test_max": float(np.nanmax(te_full)) if len(te_full) else np.nan,
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "standardized_mean_difference": smd,
                    "psi": psi_from_numeric(train[c], test[c]),
                    "js_divergence": np.nan,
                    "sample_test_unseen_row_pct": np.nan,
                })
            else:
                row.update({"shift_method": "numeric", "ks_statistic": np.nan, "psi": np.nan})
        else:
            metrics = categorical_shift_metrics(train[c], test[c])
            row.update({
                "shift_method": "categorical_or_discrete",
                **metrics,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "standardized_mean_difference": np.nan,
                "psi": np.nan,
            })

        if row.get("shift_method") == "numeric":
            ks_stat = row.get("ks_statistic", np.nan)
            psi = row.get("psi", np.nan)
            row["shift_score"] = float(np.nanmax([
                0 if pd.isna(ks_stat) else ks_stat,
                0 if pd.isna(psi) else min(1.0, psi),
            ]))
            row["shift_flag"] = bool(
                (pd.notna(ks_stat) and ks_stat >= 0.10)
                or (pd.notna(psi) and psi >= 0.20)
            )
        else:
            js = row.get("js_divergence", np.nan)
            unseen = row.get("sample_test_unseen_row_pct", np.nan)
            row["shift_score"] = float(np.nanmax([
                0 if pd.isna(js) else js,
                0 if pd.isna(unseen) else min(1.0, unseen / 100.0),
            ]))
            row["shift_flag"] = bool(
                (pd.notna(js) and js >= 0.10)
                or (pd.notna(unseen) and unseen >= 5.0)
            )

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["shift_flag", "shift_score"], ascending=[False, False])
    return result


def binned_numeric_target_signal(x: pd.Series, y: pd.Series, bins: int = 10) -> dict[str, float]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y}).dropna()
    if len(data) < 50 or data["x"].nunique() < 3:
        return {"binned_target_rate_range": np.nan, "binned_target_rate_std": np.nan}
    try:
        data["bin"] = pd.qcut(data["x"], q=min(bins, data["x"].nunique()), duplicates="drop")
        rates = data.groupby("bin", observed=True)["y"].agg(["mean", "count"])
    except (ValueError, TypeError):
        return {"binned_target_rate_range": np.nan, "binned_target_rate_std": np.nan}

    if len(rates) < 2:
        return {"binned_target_rate_range": np.nan, "binned_target_rate_std": np.nan}
    return {
        "binned_target_rate_range": float(rates["mean"].max() - rates["mean"].min()),
        "binned_target_rate_std": float(np.average((rates["mean"] - np.average(rates["mean"], weights=rates["count"])) ** 2, weights=rates["count"]) ** 0.5),
    }


def exact_value_target_signal(x: pd.Series, y: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna(subset=["y"])
    if len(data) == 0:
        return {
            "repeated_value_coverage": np.nan,
            "repeated_value_purity": np.nan,
            "smoothed_target_rate_range": np.nan,
        }

    # Cap work for huge high-cardinality columns while preserving a diagnostic sample.
    if len(data) > REL_SAMPLE_SIZE:
        data = data.sample(REL_SAMPLE_SIZE, random_state=SEED)

    global_rate = float(data["y"].mean())
    grouped = data.groupby("x", dropna=False)["y"].agg(["mean", "count"])
    repeated = grouped[grouped["count"] >= 5]
    if repeated.empty:
        return {
            "repeated_value_coverage": 0.0,
            "repeated_value_purity": np.nan,
            "smoothed_target_rate_range": np.nan,
        }

    coverage = float(repeated["count"].sum() / len(data))
    purity = np.maximum(repeated["mean"], 1 - repeated["mean"])
    weighted_purity = float(np.average(purity, weights=repeated["count"]))
    smoothing = 20.0
    smoothed = (
        repeated["mean"] * repeated["count"] + global_rate * smoothing
    ) / (repeated["count"] + smoothing)
    return {
        "repeated_value_coverage": coverage,
        "repeated_value_purity": weighted_purity,
        "smoothed_target_rate_range": float(smoothed.max() - smoothed.min()),
    }


def build_target_relationships(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    column_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    y_encoded, positive_label = encode_binary_target(train[target])
    meta = column_summary.set_index("column").to_dict("index")
    rows: list[dict[str, Any]] = []

    for c in feature_cols:
        m = meta[c]
        row: dict[str, Any] = {
            "column": c,
            "feature_kind": m["feature_kind"],
            "is_likely_id": m["is_likely_id"],
            "id_like": m["id_like"],
            "n_unique": m["n_unique"],
        }

        if pd.api.types.is_numeric_dtype(train[c]) and m["feature_kind"] in {"continuous_numeric", "discrete_numeric", "binary"}:
            data = pd.DataFrame({"x": pd.to_numeric(train[c], errors="coerce"), "y": y_encoded})
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > REL_SAMPLE_SIZE:
                data = data.sample(REL_SAMPLE_SIZE, random_state=SEED)

            if len(data) >= 20 and data["x"].nunique() >= 2 and data["y"].nunique() == 2:
                raw_auc = float(roc_auc_score(data["y"], data["x"]))
                row["univariate_auc_raw"] = raw_auc
                row["univariate_auc_strength"] = max(raw_auc, 1.0 - raw_auc)
                row["pearson_corr"] = float(data["x"].corr(data["y"], method="pearson"))
                row["spearman_corr"] = float(data["x"].corr(data["y"], method="spearman"))
                pos = data.loc[data["y"] == 1, "x"]
                neg = data.loc[data["y"] == 0, "x"]
                row["positive_mean"] = float(pos.mean()) if len(pos) else np.nan
                row["negative_mean"] = float(neg.mean()) if len(neg) else np.nan
                row["positive_median"] = float(pos.median()) if len(pos) else np.nan
                row["negative_median"] = float(neg.median()) if len(neg) else np.nan
                row.update(binned_numeric_target_signal(data["x"], data["y"]))
                row.update(exact_value_target_signal(data["x"], data["y"]))
            else:
                row["univariate_auc_raw"] = np.nan
                row["univariate_auc_strength"] = np.nan
        else:
            data = pd.DataFrame({"x": canonical_category(train[c]), "y": y_encoded}).dropna(subset=["y"])
            if len(data) > REL_SAMPLE_SIZE:
                data = data.sample(REL_SAMPLE_SIZE, random_state=SEED)

            global_rate = float(data["y"].mean()) if len(data) else np.nan
            grouped = data.groupby("x", dropna=False)["y"].agg(["mean", "count"])
            min_count = max(5, int(0.0005 * len(data)))
            stable = grouped[grouped["count"] >= min_count]
            smoothing = 20.0
            if len(stable):
                smoothed = (
                    stable["mean"] * stable["count"] + global_rate * smoothing
                ) / (stable["count"] + smoothing)
                weighted_mean = float(np.average(smoothed, weights=stable["count"]))
                weighted_std = float(
                    np.average((smoothed - weighted_mean) ** 2, weights=stable["count"]) ** 0.5
                )
                row.update({
                    "category_min_count_used": min_count,
                    "stable_category_count": int(len(stable)),
                    "smoothed_target_rate_min": float(smoothed.min()),
                    "smoothed_target_rate_max": float(smoothed.max()),
                    "smoothed_target_rate_range": float(smoothed.max() - smoothed.min()),
                    "smoothed_target_rate_weighted_std": weighted_std,
                    "largest_category_count": int(grouped["count"].max()) if len(grouped) else 0,
                })
            row.update(exact_value_target_signal(data["x"], data["y"]))
            # Deliberately no in-sample category target-encoded AUC: it would be misleading/leaky.
            row["univariate_auc_raw"] = np.nan
            row["univariate_auc_strength"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        numeric_strength = result["univariate_auc_strength"].fillna(0.5)
        cat_strength = result.get("smoothed_target_rate_range", pd.Series(0, index=result.index)).fillna(0)
        result["relationship_score"] = np.maximum((numeric_strength - 0.5) * 2, cat_strength)
        result = result.sort_values("relationship_score", ascending=False)
    return result, positive_label


def row_hashes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.array([], dtype=np.uint64)
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)


def duplicate_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    likely_id: str | None,
) -> dict[str, Any]:
    common_features = [c for c in test.columns if c in train.columns and c != target]
    model_features = [c for c in common_features if c != likely_id]

    train_full_dups = int(train.duplicated().sum())
    test_full_dups = int(test.duplicated().sum())

    if model_features:
        tr_hash = row_hashes(train, model_features)
        te_hash = row_hashes(test, model_features)
        train_feature_dup_count = int(pd.Series(tr_hash).duplicated().sum())
        test_feature_dup_count = int(pd.Series(te_hash).duplicated().sum())
        overlap_unique_hashes = np.intersect1d(np.unique(tr_hash), np.unique(te_hash), assume_unique=True)
        train_rows_matching_test = int(np.isin(tr_hash, overlap_unique_hashes).sum())
        test_rows_matching_train = int(np.isin(te_hash, overlap_unique_hashes).sum())

        # Conflicting labels for exactly repeated feature vectors inside train.
        temp = pd.DataFrame({"row_hash": tr_hash, "target": train[target].to_numpy()})
        grouped = temp.groupby("row_hash", sort=False)["target"].nunique(dropna=False)
        conflicting_hashes = grouped[grouped > 1].index.to_numpy(dtype=np.uint64)
        conflicting_groups = int(len(conflicting_hashes))
        conflicting_rows = int(np.isin(tr_hash, conflicting_hashes).sum()) if conflicting_groups else 0
    else:
        train_feature_dup_count = test_feature_dup_count = 0
        train_rows_matching_test = test_rows_matching_train = 0
        conflicting_groups = conflicting_rows = 0

    return {
        "train_duplicate_full_rows": train_full_dups,
        "test_duplicate_full_rows": test_full_dups,
        "train_duplicate_feature_rows_excluding_id": train_feature_dup_count,
        "test_duplicate_feature_rows_excluding_id": test_feature_dup_count,
        "train_rows_with_feature_match_in_test": train_rows_matching_test,
        "test_rows_with_feature_match_in_train": test_rows_matching_train,
        "conflicting_duplicate_feature_groups": conflicting_groups,
        "rows_in_conflicting_duplicate_feature_groups": conflicting_rows,
        "feature_hash_excluded_id": likely_id,
        "feature_hash_column_count": len(model_features),
    }


def build_suspicious_features(
    train: pd.DataFrame,
    target: str,
    column_summary: pd.DataFrame,
    numerical_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    col = column_summary.set_index("column")
    num = numerical_summary.set_index("column") if len(numerical_summary) else pd.DataFrame()
    sh = shift_summary.set_index("column") if len(shift_summary) else pd.DataFrame()
    rel = relationships.set_index("column") if len(relationships) else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for c in column_summary.loc[~column_summary["is_target"], "column"]:
        flags: list[str] = []
        severity = 0
        m = col.loc[c]

        if bool(m["constant"]):
            flags.append("constant")
            severity = max(severity, 3)
        if bool(m["near_constant"]):
            flags.append("near_constant")
            severity = max(severity, 2)
        if bool(m["is_likely_id"]):
            flags.append("likely_id")
            severity = max(severity, 2)
        elif bool(m["id_like"]):
            flags.append("id_like_high_uniqueness")
            severity = max(severity, 1)
        if bool(m["integer_category_candidate"]):
            flags.append("numeric_but_category_candidate")
            severity = max(severity, 1)
        if bool(m["ordinal_name_hint"]):
            flags.append("ordinal_name_hint")
            severity = max(severity, 1)
        if bool(m["high_cardinality"]):
            flags.append("high_cardinality_categorical")
            severity = max(severity, 1)

        if len(num) and c in num.index:
            inf_count = num.loc[c].get("infinite_count", 0)
            max_abs = num.loc[c].get("max_abs", np.nan)
            dyn = num.loc[c].get("dynamic_range_abs", np.nan)
            precision = num.loc[c].get("decimal_precision_95", np.nan)
            if pd.notna(inf_count) and inf_count > 0:
                flags.append("contains_infinite_values")
                severity = max(severity, 3)
            if pd.notna(max_abs) and max_abs > 1e9:
                flags.append("extreme_absolute_scale")
                severity = max(severity, 1)
            if pd.notna(dyn) and dyn > 1e12:
                flags.append("extreme_dynamic_range")
                severity = max(severity, 1)
            if pd.notna(precision) and precision <= 2 and m["n_unique"] > 20:
                flags.append(f"strong_rounding_pattern_{int(precision)}dp")
                severity = max(severity, 1)

        if len(sh) and c in sh.index and bool(sh.loc[c].get("shift_flag", False)):
            flags.append("train_test_shift")
            severity = max(severity, 2)

        if len(rel) and c in rel.index:
            auc_strength = rel.loc[c].get("univariate_auc_strength", np.nan)
            rate_range = rel.loc[c].get("smoothed_target_rate_range", np.nan)
            purity = rel.loc[c].get("repeated_value_purity", np.nan)
            coverage = rel.loc[c].get("repeated_value_coverage", np.nan)
            corr = rel.loc[c].get("pearson_corr", np.nan)
            if pd.notna(auc_strength) and auc_strength >= 0.995:
                flags.append("near_perfect_univariate_auc_possible_leakage")
                severity = max(severity, 4)
            elif pd.notna(auc_strength) and auc_strength >= 0.90:
                flags.append("very_strong_univariate_numeric_signal")
                severity = max(severity, 2)
            if pd.notna(corr) and abs(corr) >= 0.98:
                flags.append("near_perfect_target_correlation")
                severity = max(severity, 4)
            if pd.notna(rate_range) and rate_range >= 0.90 and m["n_unique"] <= 100:
                flags.append("extreme_category_target_rate_spread")
                severity = max(severity, 3)
            if (
                pd.notna(purity) and pd.notna(coverage)
                and purity >= 0.98 and coverage >= 0.50
            ):
                flags.append("exact_values_highly_target_predictive")
                severity = max(severity, 3)

        # Direct target equality / inverse equality check for binary-like columns.
        if m["n_unique"] <= 2:
            try:
                aligned = pd.DataFrame({"x": train[c], "y": train[target]}).dropna()
                if len(aligned) and aligned["x"].equals(aligned["y"]):
                    flags.append("exact_target_copy")
                    severity = max(severity, 5)
            except Exception:
                pass

        if flags:
            rows.append({
                "column": c,
                "severity": severity,
                "flags": "; ".join(flags),
                "feature_kind": m["feature_kind"],
                "n_unique": m["n_unique"],
                "unique_ratio": m["unique_ratio"],
                "missing_pct": m["missing_pct"],
            })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["severity", "column"], ascending=[False, True])
    return result


def plot_target(train: pd.DataFrame, target: str, out_dir: Path) -> None:
    counts = train[target].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(x) for x in counts.index], counts.values)
    ax.set_title(f"Target distribution: {target}")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(train):.1%})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_missingness(column_summary: pd.DataFrame, out_dir: Path) -> None:
    miss = column_summary.loc[
        (column_summary["missing_pct"] > 0) & (~column_summary["is_target"]),
        ["column", "missing_pct"],
    ].nlargest(25, "missing_pct")
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(miss))))
    y = np.arange(len(miss))
    ax.barh(y, miss["missing_pct"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(miss["column"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Missing (%)")
    ax.set_title("Top missing-value features")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness_top_features.png", dpi=150)
    plt.close(fig)


def plot_numeric_shift(train: pd.DataFrame, test: pd.DataFrame, shift: pd.DataFrame, out_dir: Path) -> None:
    if shift.empty:
        return
    candidates = shift.loc[
        (shift["shift_method"] == "numeric")
        & (~shift["is_likely_id"].fillna(False))
        & (shift["shift_score"].fillna(0) > 0)
    ].nlargest(MAX_PLOT_FEATURES, "shift_score")
    for _, r in candidates.iterrows():
        c = r["column"]
        tr = finite_numeric(sample_series(train[c].dropna(), 50_000, SEED))
        te = finite_numeric(sample_series(test[c].dropna(), 50_000, SEED + 1))
        if tr.empty or te.empty:
            continue
        combined = np.concatenate([tr.to_numpy(), te.to_numpy()])
        lo, hi = np.nanquantile(combined, [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(combined), np.nanmax(combined)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(tr.clip(lo, hi), bins=50, density=True, alpha=0.55, label="train")
        ax.hist(te.clip(lo, hi), bins=50, density=True, alpha=0.55, label="test")
        ax.set_title(f"Train vs test: {c} | KS={r.get('ks_statistic', np.nan):.3f}")
        ax.set_xlabel(c)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"shift_numeric__{sanitize_filename(c)}.png", dpi=150)
        plt.close(fig)


def plot_categorical_shift(train: pd.DataFrame, test: pd.DataFrame, shift: pd.DataFrame, out_dir: Path) -> None:
    if shift.empty:
        return
    candidates = shift.loc[
        (shift["shift_method"] == "categorical_or_discrete")
        & (~shift["is_likely_id"].fillna(False))
        & (shift["shift_score"].fillna(0) > 0.01)
    ].nlargest(3, "shift_score")
    for _, r in candidates.iterrows():
        c = r["column"]
        tr = canonical_category(sample_series(train[c], CAT_SAMPLE_SIZE, SEED))
        te = canonical_category(sample_series(test[c], CAT_SAMPLE_SIZE, SEED + 1))
        tr_p = tr.value_counts(normalize=True)
        te_p = te.value_counts(normalize=True)
        levels = list((tr_p.add(te_p, fill_value=0)).nlargest(MAX_CATEGORY_PLOT_LEVELS).index)
        df = pd.DataFrame({
            "train": tr_p.reindex(levels, fill_value=0),
            "test": te_p.reindex(levels, fill_value=0),
        })
        x = np.arange(len(df))
        width = 0.42
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.65), 4.8))
        ax.bar(x - width / 2, df["train"].values, width=width, label="train")
        ax.bar(x + width / 2, df["test"].values, width=width, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v)[:30] for v in df.index], rotation=45, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Train vs test categories: {c}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"shift_categorical__{sanitize_filename(c)}.png", dpi=150)
        plt.close(fig)


def plot_numeric_target_relationships(
    train: pd.DataFrame,
    target: str,
    relationships: pd.DataFrame,
    out_dir: Path,
) -> None:
    if relationships.empty:
        return
    numeric = relationships.dropna(subset=["univariate_auc_strength"]).nlargest(4, "univariate_auc_strength")
    y, positive_label = encode_binary_target(train[target])
    for _, r in numeric.iterrows():
        c = r["column"]
        data = pd.DataFrame({"x": pd.to_numeric(train[c], errors="coerce"), "y": y}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(data) > 100_000:
            data = data.sample(100_000, random_state=SEED)
        if len(data) < 50 or data["x"].nunique() < 3:
            continue
        try:
            data["bin"] = pd.qcut(data["x"], q=min(10, data["x"].nunique()), duplicates="drop")
            grouped = data.groupby("bin", observed=True).agg(
                x_median=("x", "median"), target_rate=("y", "mean"), count=("y", "size")
            )
        except (ValueError, TypeError):
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(grouped["x_median"], grouped["target_rate"], marker="o")
        ax.axhline(float(y.mean()), linestyle="--", linewidth=1, label="global target rate")
        ax.set_title(
            f"Binned target relationship: {c}\n"
            f"univariate AUC strength={r['univariate_auc_strength']:.4f}, positive={positive_label!r}"
        )
        ax.set_xlabel(f"{c} (bin median)")
        ax.set_ylabel("Target rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"target_numeric__{sanitize_filename(c)}.png", dpi=150)
        plt.close(fig)


def write_summary_text(
    path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
    likely_id: str | None,
    detection_reasons: list[str],
    positive_label: Any,
    column_summary: pd.DataFrame,
    shift: pd.DataFrame,
    relationships: pd.DataFrame,
    suspicious: pd.DataFrame,
    duplicate_info: dict[str, Any],
) -> None:
    y_values, y_counts, positive_rate = validate_binary_target(train[target])
    type_counts = (
        column_summary.loc[~column_summary["is_target"], "feature_kind"].value_counts().to_dict()
    )

    lines: list[str] = []
    lines.append("DATASET REVIEW — EXECUTIVE REPORT")
    lines.append("=" * 72)
    lines.append(f"Train shape: {train.shape}")
    lines.append(f"Test shape: {test.shape}")
    lines.append(f"Sample submission shape: {sample.shape}")
    lines.append(f"Train memory: {memory_mb(train):.2f} MB")
    lines.append(f"Test memory: {memory_mb(test):.2f} MB")
    lines.append("")
    lines.append(f"Detected target: {target!r}")
    lines.append(f"Likely ID: {likely_id!r}")
    for reason in detection_reasons:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append(f"Target values: {y_values}")
    lines.append(f"Target counts: {y_counts}")
    lines.append(f"Encoded positive label for diagnostics: {positive_label!r}")
    if positive_rate is not None:
        lines.append(f"Positive rate (descriptive): {positive_rate:.6f}")
    lines.append("")
    lines.append(f"Feature type counts: {type_counts}")
    lines.append(
        f"Columns with missing values: {int((column_summary['missing_count'] > 0).sum())} / {len(column_summary)}"
    )
    lines.append(
        f"Constant features: {int(column_summary['constant'].sum())}; "
        f"near-constant features: {int(column_summary['near_constant'].sum())}"
    )
    lines.append(
        f"Integer/category candidates: {int(column_summary['integer_category_candidate'].sum())}; "
        f"high-cardinality categoricals: {int(column_summary['high_cardinality'].sum())}"
    )
    lines.append("")
    lines.append("DUPLICATE / LEAKAGE-STRUCTURE CHECKS")
    for k, v in duplicate_info.items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("TOP TRAIN/TEST SHIFT CANDIDATES")
    if len(shift):
        for _, r in shift.head(12).iterrows():
            lines.append(
                f"  {r['column']}: kind={r['feature_kind']}, method={r['shift_method']}, "
                f"score={r.get('shift_score', np.nan):.4f}, flag={bool(r.get('shift_flag', False))}"
            )
    else:
        lines.append("  None computed.")

    lines.append("")
    lines.append("TOP FEATURE ↔ TARGET DIAGNOSTICS")
    if len(relationships):
        for _, r in relationships.head(12).iterrows():
            auc = r.get("univariate_auc_strength", np.nan)
            rate_range = r.get("smoothed_target_rate_range", np.nan)
            lines.append(
                f"  {r['column']}: kind={r['feature_kind']}, "
                f"auc_strength={auc if pd.notna(auc) else 'NA'}, "
                f"smoothed_rate_range={rate_range if pd.notna(rate_range) else 'NA'}"
            )
    else:
        lines.append("  None computed.")

    lines.append("")
    lines.append("SUSPICIOUS FEATURES")
    if len(suspicious):
        for _, r in suspicious.head(25).iterrows():
            lines.append(f"  severity={r['severity']} | {r['column']}: {r['flags']}")
    else:
        lines.append("  No heuristic flags triggered.")

    lines.append("")
    lines.append("IMPORTANT INTERPRETATION NOTE")
    lines.append(
        "Feature-target diagnostics in this report are exploratory only. They are NOT validation scores. "
        "In particular, category target-rate summaries are computed only to inspect structure and must not "
        "be used as model encodings until we implement leakage-safe out-of-fold logic."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_executive_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    likely_id: str | None,
    positive_label: Any,
    column_summary: pd.DataFrame,
    shift: pd.DataFrame,
    relationships: pd.DataFrame,
    suspicious: pd.DataFrame,
    duplicate_info: dict[str, Any],
    out_dir: Path,
) -> None:
    y = train[target]
    print("\n" + "=" * 78)
    print("DATASET REVIEW COMPLETE — EXECUTIVE SUMMARY")
    print("=" * 78)
    print(f"Train / test: {train.shape} / {test.shape}")
    print(f"Target: {target!r} | likely ID: {likely_id!r}")
    print(f"Target counts: {y.value_counts(dropna=False).to_dict()}")
    print(f"Positive label used for binary diagnostics: {positive_label!r}")
    print(
        "Feature kinds: "
        + str(column_summary.loc[~column_summary["is_target"], "feature_kind"].value_counts().to_dict())
    )
    print(
        f"Missing columns: {(column_summary['missing_count'] > 0).sum()} | "
        f"constant: {column_summary['constant'].sum()} | "
        f"near-constant: {column_summary['near_constant'].sum()}"
    )
    print(
        f"Duplicate feature rows (train, excluding likely ID): "
        f"{duplicate_info['train_duplicate_feature_rows_excluding_id']:,}"
    )
    print(
        f"Train rows with an exact feature match in test: "
        f"{duplicate_info['train_rows_with_feature_match_in_test']:,}"
    )
    print(
        f"Conflicting duplicate feature groups in train: "
        f"{duplicate_info['conflicting_duplicate_feature_groups']:,}"
    )

    if len(shift):
        top = shift.loc[~shift["is_likely_id"].fillna(False)].head(5)
        print("\nTop train/test shift candidates (excluding likely ID):")
        for _, r in top.iterrows():
            print(
                f"  - {r['column']}: {r['shift_method']}, score={r['shift_score']:.4f}, "
                f"flag={bool(r['shift_flag'])}"
            )

    if len(relationships):
        print("\nTop exploratory feature-target relationships:")
        for _, r in relationships.head(5).iterrows():
            auc = r.get("univariate_auc_strength", np.nan)
            rate_range = r.get("smoothed_target_rate_range", np.nan)
            auc_text = f"{auc:.4f}" if pd.notna(auc) else "NA"
            range_text = f"{rate_range:.4f}" if pd.notna(rate_range) else "NA"
            print(
                f"  - {r['column']}: AUC-strength={auc_text}, smoothed target-rate range={range_text}"
            )

    print(f"\nSuspicious features flagged: {len(suspicious)}")
    if len(suspicious):
        for _, r in suspicious.head(8).iterrows():
            print(f"  - severity {r['severity']} | {r['column']}: {r['flags']}")

    print(f"\nArtifacts written to: {out_dir.resolve()}")
    print("Do NOT choose a model from this terminal summary alone; inspect the CSVs next.")
    print("=" * 78)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.output_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / args.train
    test_path = data_dir / args.test
    sample_path = data_dir / args.sample_submission

    print("Loading data...")
    train = read_csv(train_path)
    test = read_csv(test_path)
    sample = read_csv(sample_path)

    target, target_reasons = infer_target(train, test, sample)
    likely_id, id_scores, id_reasons = infer_id(train, test, sample, target)
    detection_reasons = target_reasons + id_reasons

    # Validate target early: this audit is specifically for binary classification.
    y_non_null = train[target].dropna()
    if y_non_null.nunique() != 2:
        raise ValueError(
            f"Detected target {target!r}, but it has {y_non_null.nunique()} non-null unique values. "
            "This does not look like binary classification, so the script is stopping instead of producing misleading diagnostics."
        )

    feature_cols = [c for c in test.columns if c in train.columns and c != target]

    print("Building column/type summaries...")
    column_summary = build_column_summary(train, test, target, likely_id, id_scores)
    numerical_summary = build_numerical_summary(train, feature_cols)
    categorical_summary = build_categorical_summary(train, test, feature_cols, column_summary)

    print("Comparing train vs test distributions...")
    shift_summary = build_shift_summary(train, test, feature_cols, column_summary)

    print("Inspecting feature-target relationships...")
    relationships, positive_label = build_target_relationships(
        train, feature_cols, target, column_summary
    )

    print("Checking duplicates and possible leakage structure...")
    duplicate_info = duplicate_diagnostics(train, test, target, likely_id)

    print("Collecting suspicious-feature flags...")
    suspicious = build_suspicious_features(
        train,
        target,
        column_summary,
        numerical_summary,
        shift_summary,
        relationships,
    )

    # Save tables.
    column_summary.to_csv(out_dir / "column_summary.csv", index=False)
    numerical_summary.to_csv(out_dir / "numerical_summary.csv", index=False)
    categorical_summary.to_csv(out_dir / "categorical_summary.csv", index=False)
    shift_summary.to_csv(out_dir / "train_test_shift.csv", index=False)
    relationships.to_csv(out_dir / "target_relationships.csv", index=False)
    suspicious.to_csv(out_dir / "suspicious_features.csv", index=False)

    write_summary_text(
        out_dir / "dataset_summary.txt",
        train,
        test,
        sample,
        target,
        likely_id,
        detection_reasons,
        positive_label,
        column_summary,
        shift_summary,
        relationships,
        suspicious,
        duplicate_info,
    )

    print("Creating a small set of decision-useful plots...")
    plot_target(train, target, plots_dir)
    plot_missingness(column_summary, plots_dir)
    plot_numeric_shift(train, test, shift_summary, plots_dir)
    plot_categorical_shift(train, test, shift_summary, plots_dir)
    plot_numeric_target_relationships(train, target, relationships, plots_dir)

    print_executive_summary(
        train,
        test,
        target,
        likely_id,
        positive_label,
        column_summary,
        shift_summary,
        relationships,
        suspicious,
        duplicate_info,
        out_dir,
    )


if __name__ == "__main__":
    main()