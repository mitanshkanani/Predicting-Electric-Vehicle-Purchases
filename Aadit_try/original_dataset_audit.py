
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

DATASET_HANDLE = "itzzomkar/ev-adoption-behavior-and-range-anxiety"
SOURCE_FILENAME = "EV_Adoption_and_Range_Anxiety_Dataset.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--original-path", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "original_data_audit",
    )
    return p.parse_args()


def find_target(train, test):
    cols = [c for c in train.columns if c not in test.columns]
    if len(cols) != 1:
        raise ValueError(f"Expected one train-only target column, found {cols}")
    return cols[0]


def find_comp_id(train, test, target):
    common = [c for c in test.columns if c in train.columns and c != target]
    for c in common:
        if c.lower() in {"id", "row_id", "rowid", "index"}:
            return c
    return None


def find_source_id(df, target):
    for c in df.columns:
        if c != target and c.lower() in {
            "buyer_id", "customer_id", "id", "row_id", "rowid"
        }:
            return c
    return None


def encode_target(y):
    vals = list(pd.unique(y.dropna()))
    if len(vals) != 2:
        raise ValueError(f"Target must be binary; found {vals}")
    preferred = {"yes", "true", "1", "positive", "buy", "will_buy"}
    pos = next(
        (v for v in vals if str(v).strip().lower() in preferred),
        y.value_counts().idxmin(),
    )
    return (y == pos).astype(np.int8), pos


def locate_source(explicit, data_dir):
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates += [
        data_dir / "original" / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
        Path(SOURCE_FILENAME),
    ]
    for p in candidates:
        if p.exists():
            print(f"Using local source dataset: {p.resolve()}")
            return p

    print("Source dataset not found locally; trying kagglehub...")
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "\nInstall kagglehub first:\n"
            "  python -m pip install kagglehub\n"
            "Then rerun this script.\n\n"
            f"Or place {SOURCE_FILENAME} in data/original/\n"
        ) from e

    try:
        root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as e:
        raise SystemExit(
            "\nAutomatic download failed. Manually download the public Kaggle "
            f"dataset and place {SOURCE_FILENAME} in data/original/.\n"
            f"Original error: {e}"
        ) from e

    matches = list(root.rglob(SOURCE_FILENAME))
    if not matches:
        csvs = list(root.rglob("*.csv"))
        if len(csvs) == 1:
            matches = csvs
    if not matches:
        raise FileNotFoundError(f"Could not find source CSV under {root}")
    print(f"Downloaded source dataset: {matches[0].resolve()}")
    return matches[0]


def stringlike(s):
    return (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or pd.api.types.is_bool_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def kind(comp, src):
    if stringlike(comp) or stringlike(src):
        return "categorical"
    if max(comp.nunique(dropna=True), src.nunique(dropna=True)) <= 20:
        return "discrete"
    return "continuous_numeric"


def canonical(s):
    x = s.dropna()
    if pd.api.types.is_numeric_dtype(x.dtype):
        return x.map(
            lambda v: str(int(float(v)))
            if float(v).is_integer()
            else f"{float(v):.12g}"
        )
    return x.astype(str)


def js_div(a, b):
    pa = canonical(a).value_counts(normalize=True)
    pb = canonical(b).value_counts(normalize=True)
    levels = sorted(set(pa.index) | set(pb.index))
    p = np.array([pa.get(x, 0.0) for x in levels])
    q = np.array([pb.get(x, 0.0) for x in levels])
    return float(jensenshannon(p, q, base=2.0) ** 2)


def exact_overlap(comp, src):
    a, b = comp.dropna(), src.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    return float(a.isin(set(b)).mean()), float(b.isin(set(a)).mean())


def auc_strength(x, y):
    z = pd.to_numeric(x, errors="coerce")
    m = z.notna() & y.notna()
    if m.sum() == 0 or z[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan, np.nan
    raw = float(roc_auc_score(y[m], z[m]))
    return raw, max(raw, 1.0 - raw)


def categorical_rates(feature, comp_x, comp_y, src_x, src_y):
    def canon_with_missing(s):
        out = pd.Series(index=s.index, dtype="object")
        missing = s.isna()
        if pd.api.types.is_numeric_dtype(s.dtype):
            out.loc[~missing] = s.loc[~missing].map(
                lambda v: str(int(float(v)))
                if float(v).is_integer()
                else f"{float(v):.12g}"
            )
        else:
            out.loc[~missing] = s.loc[~missing].astype(str)
        out.loc[missing] = "__MISSING__"
        return out

    a = pd.DataFrame({"level": canon_with_missing(comp_x), "y": comp_y})
    b = pd.DataFrame({"level": canon_with_missing(src_x), "y": src_y})
    a = a.groupby("level").agg(
        competition_count=("y", "size"),
        competition_target_rate=("y", "mean"),
    ).reset_index()
    b = b.groupby("level").agg(
        original_count=("y", "size"),
        original_target_rate=("y", "mean"),
    ).reset_index()
    out = a.merge(b, on="level", how="outer")
    out.insert(0, "feature", feature)
    return out


def numeric_bin_rates(feature, comp_x, comp_y, src_x, src_y, q=10):
    a = pd.to_numeric(comp_x, errors="coerce")
    b = pd.to_numeric(src_x, errors="coerce")
    if a.nunique(dropna=True) < 3:
        return pd.DataFrame()

    edges = np.unique(
        np.nanquantile(
            a.dropna().to_numpy(float),
            np.linspace(0, 1, min(q, a.nunique(dropna=True)) + 1),
        )
    )
    if len(edges) < 4:
        return pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf

    def stats(x, y, prefix):
        bins = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
        d = pd.DataFrame({"bin": bins, "y": y}).dropna(subset=["bin"])
        return d.groupby("bin", observed=False).agg(
            **{
                f"{prefix}_count": ("y", "size"),
                f"{prefix}_target_rate": ("y", "mean"),
            }
        ).reset_index()

    out = stats(a, comp_y, "competition").merge(
        stats(b, src_y, "original"),
        on="bin",
        how="outer",
    )
    out.insert(0, "feature", feature)
    out["bin"] = out["bin"].astype(str)
    return out


def agreement(df, count_a, rate_a, count_b, rate_b, min_a, min_b):
    u = df[
        df[rate_a].notna()
        & df[rate_b].notna()
        & (df[count_a] >= min_a)
        & (df[count_b] >= min_b)
    ]
    if len(u) < 2:
        return np.nan, np.nan, len(u)
    a, b = u[rate_a].to_numpy(float), u[rate_b].to_numpy(float)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if len(u) >= 3 and np.std(a) > 0 and np.std(b) > 0
        else np.nan
    )
    return corr, rmse, len(u)


def main():
    args = parse_args()
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected data/train.csv and data/test.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading competition data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = find_target(train, test)
    comp_id = find_comp_id(train, test, target)

    source_path = locate_source(args.original_path, args.data_dir)
    print("Loading original/source data...")
    src = pd.read_csv(source_path)
    if target not in src.columns:
        raise ValueError(f"Source data does not contain target {target!r}")

    src_id = find_source_id(src, target)
    comp_y, comp_pos = encode_target(train[target])
    src_y, src_pos = encode_target(src[target])

    comp_features = [
        c for c in train.columns if c not in {target, comp_id}
    ]
    src_features = [
        c for c in src.columns if c not in {target, src_id}
    ]
    shared = [c for c in comp_features if c in src_features]

    print(f"Competition ID: {comp_id!r}")
    print(f"Source ID: {src_id!r}")
    print(f"Shared modeling features: {len(shared)}")

    missing_rows, dist_rows, rel_rows = [], [], []
    cat_tables, num_tables = [], []

    for f in shared:
        cx, tx, sx = train[f], test[f], src[f]
        k = kind(cx, sx)

        missing_rows.append(
            {
                "feature": f,
                "competition_train_missing_pct": cx.isna().mean(),
                "competition_test_missing_pct": tx.isna().mean(),
                "original_missing_pct": sx.isna().mean(),
            }
        )

        comp_seen, src_seen = exact_overlap(cx, sx)
        d = {
            "feature": f,
            "comparison_kind": k,
            "competition_n_unique": cx.nunique(dropna=True),
            "original_n_unique": sx.nunique(dropna=True),
            "competition_values_seen_in_original_rate": comp_seen,
            "original_values_seen_in_competition_rate": src_seen,
        }

        if k == "continuous_numeric":
            cnum = pd.to_numeric(cx, errors="coerce").dropna()
            tnum = pd.to_numeric(tx, errors="coerce").dropna()
            snum = pd.to_numeric(sx, errors="coerce").dropna()
            d.update(
                {
                    "source_vs_train_shift": ks_2samp(snum, cnum).statistic,
                    "source_vs_test_shift": ks_2samp(snum, tnum).statistic,
                    "competition_mean": cnum.mean(),
                    "original_mean": snum.mean(),
                    "competition_std": cnum.std(),
                    "original_std": snum.std(),
                }
            )

            craw, cstrength = auc_strength(cx, comp_y)
            sraw, sstrength = auc_strength(sx, src_y)
            bt = numeric_bin_rates(f, cx, comp_y, sx, src_y)
            corr, rmse, groups = (
                agreement(
                    bt,
                    "competition_count",
                    "competition_target_rate",
                    "original_count",
                    "original_target_rate",
                    500,
                    30,
                )
                if not bt.empty
                else (np.nan, np.nan, 0)
            )
            if not bt.empty:
                num_tables.append(bt)

            rel_rows.append(
                {
                    "feature": f,
                    "comparison_kind": k,
                    "competition_auc_strength": cstrength,
                    "original_auc_strength": sstrength,
                    "auc_strength_delta_original_minus_competition": (
                        sstrength - cstrength
                    ),
                    "target_pattern_correlation": corr,
                    "target_rate_rmse": rmse,
                    "usable_groups": groups,
                }
            )
        else:
            d.update(
                {
                    "source_vs_train_shift": js_div(sx, cx),
                    "source_vs_test_shift": js_div(sx, tx),
                    "competition_mean": np.nan,
                    "original_mean": np.nan,
                    "competition_std": np.nan,
                    "original_std": np.nan,
                }
            )
            rt = categorical_rates(f, cx, comp_y, sx, src_y)
            corr, rmse, groups = agreement(
                rt,
                "competition_count",
                "competition_target_rate",
                "original_count",
                "original_target_rate",
                500,
                30,
            )
            cat_tables.append(rt)
            rel_rows.append(
                {
                    "feature": f,
                    "comparison_kind": k,
                    "competition_auc_strength": np.nan,
                    "original_auc_strength": np.nan,
                    "auc_strength_delta_original_minus_competition": np.nan,
                    "target_pattern_correlation": corr,
                    "target_rate_rmse": rmse,
                    "usable_groups": groups,
                }
            )

        dist_rows.append(d)

    missing = pd.DataFrame(missing_rows)
    dist = pd.DataFrame(dist_rows).sort_values(
        "source_vs_train_shift", ascending=False
    )
    rel = pd.DataFrame(rel_rows).sort_values(
        "target_pattern_correlation", ascending=False, na_position="last"
    )

    missing.to_csv(args.output_dir / "missingness_comparison.csv", index=False)
    dist.to_csv(
        args.output_dir / "feature_distribution_comparison.csv", index=False
    )
    rel.to_csv(
        args.output_dir / "target_relationship_comparison.csv", index=False
    )
    if cat_tables:
        pd.concat(cat_tables, ignore_index=True).to_csv(
            args.output_dir / "categorical_target_rates.csv", index=False
        )
    if num_tables:
        pd.concat(num_tables, ignore_index=True).to_csv(
            args.output_dir / "numeric_bin_target_rates.csv", index=False
        )

    comp_rate = float(comp_y.mean())
    src_rate = float(src_y.mean())

    lines = [
        "ORIGINAL / SOURCE DATASET AUDIT",
        "=" * 72,
        f"Competition train shape: {train.shape}",
        f"Competition test shape: {test.shape}",
        f"Original shape: {src.shape}",
        f"Competition target: {target!r}",
        f"Competition positive label: {comp_pos!r}",
        f"Original positive label: {src_pos!r}",
        f"Competition ID: {comp_id!r}",
        f"Original ID: {src_id!r}",
        f"Shared modeling features: {len(shared)}",
        "",
        f"Competition positive rate: {comp_rate:.6f}",
        f"Original positive rate: {src_rate:.6f}",
        f"Original-minus-competition rate gap: {src_rate - comp_rate:+.6f}",
        "",
        "Largest source-vs-competition distribution shifts:",
    ]

    for r in dist.head(8).itertuples():
        lines.append(
            f"- {r.feature}: {r.source_vs_train_shift:.6f} "
            f"({r.comparison_kind})"
        )

    lines += ["", "Feature→target relationship agreement:"]
    for r in rel.itertuples():
        c = (
            f"{r.target_pattern_correlation:.4f}"
            if np.isfinite(r.target_pattern_correlation)
            else "NA"
        )
        rmse = (
            f"{r.target_rate_rmse:.4f}"
            if np.isfinite(r.target_rate_rmse)
            else "NA"
        )
        auc_c = (
            f"{r.competition_auc_strength:.6f}"
            if np.isfinite(r.competition_auc_strength)
            else "NA"
        )
        auc_s = (
            f"{r.original_auc_strength:.6f}"
            if np.isfinite(r.original_auc_strength)
            else "NA"
        )
        lines.append(
            f"- {r.feature}: pattern_corr={c}, rate_RMSE={rmse}, "
            f"competition_AUC_strength={auc_c}, original_AUC_strength={auc_s}"
        )

    lines += [
        "",
        "DECISION RULE:",
        "Do not use the source labels just because marginal distributions match.",
        "We want strong agreement in feature→target relationships first.",
        "If that holds, the next experiment will add the source rows only to each",
        "training fold and measure the OOF delta on untouched competition rows.",
    ]

    summary = "\n".join(lines)
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print()
    print("=" * 78)
    print("SOURCE DATASET AUDIT COMPLETE")
    print("=" * 78)
    print(f"Competition positive rate: {comp_rate:.6f}")
    print(f"Original positive rate   : {src_rate:.6f}")
    print(f"Rate gap                 : {src_rate - comp_rate:+.6f}")
    print()
    print("Top target-pattern agreements:")
    shown = 0
    for r in rel.itertuples():
        if np.isfinite(r.target_pattern_correlation):
            print(
                f"  - {r.feature}: corr={r.target_pattern_correlation:.4f}, "
                f"RMSE={r.target_rate_rmse:.4f}"
            )
            shown += 1
            if shown == 5:
                break
    if shown == 0:
        print("  - No finite correlations available.")
    print()
    print(f"Artifacts: {args.output_dir.resolve()}")
    print("=" * 78)
    print("Send me summary.txt plus the four comparison CSVs.")


if __name__ == "__main__":
    main()
