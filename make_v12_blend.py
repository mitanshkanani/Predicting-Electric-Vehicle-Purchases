"""Build a blended submission from N member CSVs, in the discipline this repo has
already validated: rank-space averaging, then the v11 band prior re-applied.

Two modes:

  --validate   Members are OOF prediction CSVs from Aadit_try/artifacts/experiments/*,
               which carry the frozen fold id. Reports each member's real AUC against
               train labels, the blend's AUC, and a leave-one-fold-out weight search so
               the number is honest rather than selection-overfitted. Use this FIRST:
               it is the only mode that can tell you a blend is worth anything.

  (default)    Members are submission CSVs (id, Will_Buy_EV). Rank-averages them with
               the weights in MEMBERS, re-applies the v11 bands, writes OUT.

Why rank space and not probability space: members here are already epsilon-ordered to a
uniform grid by their own generators, so probabilities are not comparable across files but
percentiles are. Averaging raw values would let whichever member has the widest spread
dominate the top and bottom of the ranking.

Why the diversity report is printed every run: this repo measured that blending members
which correlate at 0.9995 is worth +3e-6, i.e. nothing. A blend only earns its slot if at
least one member is genuinely decorrelated. If the matrix below shows 0.9995 everywhere,
stop and go model something different instead.

Usage:
  python make_v12_blend.py --validate
  python make_v12_blend.py
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev_s6e9_v11 as V

TARGET, ID_COL = V.TARGET, V.ID_COL

# ---------------------------------------------------------------- MEMBERS
# Drop Aadit's files anywhere in this folder (or give absolute paths) and edit here.
MEMBERS = [
    # v11 is deliberately EXCLUDED: its Spearman against v12 is 0.99997, i.e. the same
    # model twice, so including it only inflates our own weight without adding information.
    ("v14",          "v14/submission_v14.csv", 0.40),
    ("v12",          "v12/submission_v12.csv", 0.40),
    ("aad_cand",     "aadit_champ/candidate_submission.csv", 0.32),
    ("aad_sub",      "aadit_champ/submission.csv", 0.28),
    ("ext_xgb10",    "external/XGBoost_Triple_TE_10folds_test.csv", 0.00),
]
# Ask Aadit for XGBoost_Triple_TE_10folds_test.csv -> drop it in external/. His own audit
# (external_xgb10_fixed_rank_blend_audit) measured that kernel at OOF 0.946243 and the
# 50/50 rank blend of it with his champion at 0.946323, +0.000211, 5/5 folds improved.
# That blend IS his 0.94643 submission (rho 1.00000 against aadit_champ/candidate_submission.csv).
OUT = "submission_v12.csv"
MIN_DIVERSITY = 0.995
# If v14 (which contains a CatBoost member and so carries genuinely extra information) is
# present, v12 is redundant: they are both built on the same LightGBM design. v11 vs v12
# measured rho 0.99997, i.e. the same model twice, so stacking near-duplicates only
# reweights our own side without adding anything.
SUPERSEDES = {"v14": ["v12", "v11"]}        # if every pair is above this, the blend is worth ~nothing

# OOF vectors used by --validate. Paths are relative to Aadit_try/artifacts/experiments.
VALIDATE_SET = [
    ("fine_xgb", "xgboost_fine_income_te_gpu/oof_predictions.csv", "oof_prediction"),
    ("fine_cat", "catboost_fine_income_multiseed_gpu/best_average_oof_predictions.csv",
     "oof_prediction"),
    ("fine_lgbm", "lightgbm_fine_income_te_cpu/oof_predictions.csv", "oof_prediction"),
    ("xgb_lossguide", "xgboost_fine_income_lossguide_gpu/oof_predictions.csv", "oof_prediction"),
    ("source_prior_xgb", "xgboost_source_prior_feature_gpu/oof_predictions.csv", "oof_prediction"),
    ("champion", "champion_hard_edges_audit/oof_rank_predictions.csv", "champion_fold_rank"),
]
BASE_DIR = os.path.join(HERE, "Aadit_try", "artifacts", "experiments")
# ---------------------------------------------------------------------------


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return rankdata(a, method="average") / a.size


def load_sub(path):
    d = pd.read_csv(path)
    if ID_COL not in d.columns:
        raise SystemExit(f"{path}: no '{ID_COL}' column")
    col = TARGET if TARGET in d.columns else [c for c in d.columns if c != ID_COL][0]
    return d[[ID_COL, col]].rename(columns={col: "p"})


def blend(members, ids_order):
    """Weighted average in percentile space; missing members are reweighted away."""
    live = [(nm, df, w) for nm, df, w in members if df is not None]
    tot = sum(w for _, _, w in live)
    acc = np.zeros(len(ids_order), dtype=np.float64)
    for _, df, w in live:
        acc += (w / tot) * pct(df["p"].to_numpy())
    return acc, [(nm, w / tot) for nm, _, w in live]


def do_validate():
    tr = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
    y = V.encode_target(tr[TARGET])
    vecs, folds = {}, None
    for nm, rel, col in VALIDATE_SET:
        p = os.path.join(BASE_DIR, rel)
        if not os.path.isfile(p):
            print(f"  skip {nm:<16} (missing {rel})")
            continue
        d = pd.read_csv(p)
        if len(d) != len(y):
            print(f"  skip {nm:<16} ({len(d)} rows != {len(y)})")
            continue
        if col not in d.columns:
            cand = [c for c in d.columns if "oof" in c.lower() or "rank" in c.lower()]
            if not cand:
                print(f"  skip {nm:<16} (no column {col})")
                continue
            col = cand[0]
        v = d[col].to_numpy(float)
        if not np.isfinite(v).all():
            print(f"  skip {nm:<16} (non-finite)")
            continue
        vecs[nm] = pct(v)
        if folds is None and "fold" in d.columns:
            folds = d["fold"].to_numpy()
        print(f"  loaded {nm:<16} {col:<26} AUC={roc_auc_score(y, vecs[nm]):.6f}")
    if len(vecs) < 2:
        raise SystemExit("need at least 2 OOF members to validate")

    names = list(vecs)
    print("\npairwise rank correlation (a blend needs at least one pair BELOW "
          f"{MIN_DIVERSITY} to be worth a slot):")
    M = np.array([[spearmanr(vecs[a], vecs[b]).statistic for b in names] for a in names])
    print("            " + "".join(f"{n[:11]:>13}" for n in names))
    for i, a in enumerate(names):
        print(f"{a[:11]:<12}" + "".join(f"{M[i, j]:>13.5f}" for j in range(len(names))))
    off = M[~np.eye(len(names), dtype=bool)]
    print(f"  min pairwise {off.min():.5f}   median {np.median(off):.5f}")

    Vm = np.array([vecs[n] for n in names])
    eq = Vm.mean(0)
    single_best = max(roc_auc_score(y, vecs[n]) for n in names)
    print()
    print(f"equal-weight blend AUC          : {roc_auc_score(y, eq):.6f}")
    print(f"best single member AUC          : {single_best:.6f}")

    if folds is None:
        print("no fold column -> cannot cross-fit weights")
        return
    uf = np.unique(folds)
    honest = np.full(len(y), np.nan)
    for f in uf:
        fit, sc = folds != f, folds == f
        w = np.full(len(names), 1.0 / len(names))
        cur = eq.copy()
        for _ in range(30):
            base = roc_auc_score(y[fit], cur[fit])
            gain, bstep = 1e-9, None
            for i in range(len(names)):
                for step in (0.10, -0.10):
                    t = w.copy()
                    t[i] += step
                    if t.min() < 0:
                        continue
                    t /= t.sum()
                    a = roc_auc_score(y[fit], np.tensordot(t, Vm, axes=1)[fit])
                    if a - base > gain:
                        gain, bstep = a - base, (i, step)
            if bstep is None:
                break
            i, step = bstep
            w[i] += step
            w /= w.sum()
            cur = np.tensordot(w, Vm, axes=1)
        honest[sc] = cur[sc]
        print(f"  fold {f}: held-out AUC {roc_auc_score(y[sc], cur[sc]):.6f}  "
              f"weights {dict(zip(names, np.round(w, 2)))}")
    ha = roc_auc_score(y, honest)
    print()
    print(f"honest LOFO-selected blend AUC  : {ha:.6f}   <- the only number to trust")
    print(f"verdict: {'BLEND EARNS IT' if ha > single_best else 'BLEND DOES NOT BEAT BEST SINGLE'}")

def do_blend(args):
    members, ref_ids = [], None
    for nm, path, w in MEMBERS:
        p = path if os.path.isabs(path) else os.path.join(HERE, path)
        if not os.path.isfile(p):
            print(f"  MISSING  {nm:<12} {p}")
            members.append((nm, None, w))
            continue
        d = load_sub(p)
        if ref_ids is None:
            ref_ids = d[ID_COL].to_numpy()
            members.append((nm, d.set_index(ID_COL).reindex(ref_ids), w))
        else:
            members.append((nm, d.set_index(ID_COL).reindex(ref_ids), w))
        print(f"  loaded   {nm:<12} {len(d)} rows  w={w}")
    present = {nm for nm, df, _ in members if df is not None}
    drop = {d for winner, losers in SUPERSEDES.items() if winner in present for d in losers}
    live = [(nm, df, w) for nm, df, w in members if df is not None and nm not in drop]
    for d in sorted(drop):
        print(f"  dropped    {d:<12} (superseded by a richer member)")
    if len(live) < 1:
        raise SystemExit("no members found")
    if len(live) == 1:
        print("\nonly one member present -> nothing to blend. Add Aadit's CSVs.")
        return

    print("\npairwise Spearman between members:")
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            r = spearmanr(live[i][1]["p"], live[j][1]["p"]).statistic
            flag = "  <- near-duplicate, contributes ~nothing" if r > MIN_DIVERSITY else ""
            print(f"  {live[i][0]:<10} x {live[j][0]:<10} {r:.6f}{flag}")

    score, used = blend(live, ref_ids)
    print(f"\nweights used: {dict(used)}")

    ss_path = os.path.join(HERE, "data", "sample_submission.csv")
    out = pd.DataFrame({ID_COL: ref_ids, TARGET: score})
    if args.bands:
        te = pd.read_csv(os.path.join(HERE, "data", "test.csv"))
        inc = np.rint(pd.to_numeric(te["Annual_Income_USD"], errors="coerce").fillna(0)).to_numpy(float)
        com = pd.to_numeric(te["Daily_Commute_km"], errors="coerce").fillna(0).to_numpy(float)
        sub = pd.DataFrame({ID_COL: te[ID_COL].to_numpy(), TARGET: score}).set_index(ID_COL)
        sub = sub.reindex(ref_ids)
        aligned = sub[TARGET].to_numpy()
        if np.isnan(aligned).any():
            raise SystemExit("test.csv ids do not align with the member submissions")
        banded = V.apply_bands(aligned, inc, com)
        pos = {k: i for i, k in enumerate(te[ID_COL].to_numpy())}
        out[TARGET] = [banded[pos[i]] for i in ref_ids]
    if os.path.isfile(ss_path):
        ss = pd.read_csv(ss_path)
        if ID_COL in ss.columns:
            out = ss[[ID_COL]].merge(out, on=ID_COL, how="left")
    assert out[TARGET].notna().all() and len(out) == len(ref_ids)
    path = os.path.join(HERE, args.out)
    out.to_csv(path, index=False, float_format="%.12g")
    print(f"\nwrote {args.out} rows={len(out)} unique={out[TARGET].nunique()} "
          f"range {out[TARGET].min():.3g}..{out[TARGET].max():.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="score blends against real train labels")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-bands", dest="bands", action="store_false")
    a = ap.parse_args()
    if a.validate:
        do_validate()
    else:
        do_blend(a)
