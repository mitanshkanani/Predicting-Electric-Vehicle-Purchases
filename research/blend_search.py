"""Cross-fitted ensemble selection over every OOF vector Aadit's workstream produced.

Why this is the right next move: Aadit_try/artifacts/experiments holds 110 distinct
out-of-fold prediction vectors over the SAME 668,665 frozen folds, and 93 of those
directories also hold the matching TEST predictions. So a better blend can be found
and BUILT offline, with real labels, without spending a Kaggle run.

The trap to avoid: picking the blend that maximises OOF over 110 candidates is
selection overfitting, and at this competition's scale (public-LB SE ~0.0013, real
deltas ~0.0002) a greedy search will happily find +0.001 that does not exist. So the
selection is done leave-one-fold-out: members and weights are chosen on 4 of the 5
frozen folds and scored on the held-out fold, and only that rotated number is reported.
"""
import hashlib
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = "Aadit_try/artifacts/experiments"
tr = pd.read_csv("data/train.csv")
y = (tr.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").to_numpy(np.int8)
N = len(y)


def to_vec(v):
    v = np.asarray(v, dtype=np.float64)
    return (rankdata(v) / v.size).astype(np.float32)


models = {}          # key -> dict(oof_rank, fold, test_file, dir, col, auc)
seen_hash = {}
for d in sorted(os.listdir(BASE)):
    dd = os.path.join(BASE, d)
    if not os.path.isdir(dd):
        continue
    tsts = [f for f in glob.glob(dd + "/*.csv")
            if "test" in os.path.basename(f).lower() and "audit" not in os.path.basename(f).lower()]
    for of in glob.glob(dd + "/*.csv"):
        bn = os.path.basename(of).lower()
        if "oof" not in bn:
            continue
        try:
            df = pd.read_csv(of)
        except Exception:
            continue
        if len(df) != N:
            continue
        fold = df["fold"].to_numpy() if "fold" in df.columns else None
        for c in df.columns:
            if c in ("row_index", "id", "fold", "target_encoded", "realmlp_alpha",
                     "champion_alpha", "full_oof_auc_diagnostic"):
                continue
            if df[c].dtype.kind != "f":
                continue
            v = df[c].to_numpy(float)
            if not np.isfinite(v).all() or np.nanstd(v) == 0:
                continue
            r = to_vec(v)
            h = hashlib.md5(r.tobytes()).hexdigest()[:12]
            a = roc_auc_score(y, r)
            a = a if a > 0.5 else 1 - a
            if h in seen_hash:                       # same vector under another name
                if tsts and not seen_hash[h]["test_files"]:
                    seen_hash[h]["test_files"] = tsts
                continue
            rec = dict(oof=r, fold=fold, auc=a, dir=d, col=c, test_files=tsts)
            seen_hash[h] = rec
            models[h] = rec

M = sorted(models.values(), key=lambda m: -m["auc"])
print(f"{len(M)} DISTINCT oof vectors after de-duplication")
for m in M[:12]:
    print(f"  {m['auc']:.6f}  {m['dir'][:56]:<57} {m['col'][:26]:<27} test={'Y' if m['test_files'] else '-'}")

usable = [m for m in M if m["fold"] is not None]
# Cap the search pool: the greedy selection's honest cost is one AUC per (member x
# round x fold), and members below ~0.945 never earn a slot.
POOL = usable[:40]
print(f"\n{len(usable)} carry the frozen fold id; searching over the top {len(POOL)}")
folds = POOL[0]["fold"]
assert all(np.array_equal(m["fold"], folds) for m in POOL), "fold vectors disagree"
UF = np.unique(folds)


def auc_on(mask, blend):
    return roc_auc_score(y[mask], blend[mask])


def greedy_idx(pool, fit_mask, rounds=30):
    counts = np.zeros(len(pool))
    cur = np.zeros(N, dtype=np.float64)
    best = -1
    for step in range(rounds):
        bs, bi = -1, -1
        for i, m in enumerate(pool):
            trial = (cur * counts.sum() + m["oof"]) / (counts.sum() + 1)
            a = auc_on(fit_mask, trial.astype(np.float32))
            if a > bs:
                bs, bi = a, i
        if bs <= best + 1e-9:
            break
        best = bs
        cur = (cur * counts.sum() + pool[bi]["oof"]) / (counts.sum() + 1)
        counts[bi] += 1
    return cur, counts


print("\n=== honest evaluation: leave-one-fold-out ensemble selection ===")
oof_honest = np.full(N, np.nan, dtype=np.float64)
for f in UF:
    fit = folds != f
    sc = folds == f
    cur, counts = greedy_idx(POOL, fit, rounds=14)
    # apply the weights learned on the other folds to the held-out fold rows
    w = counts / counts.sum()
    blend = np.zeros(N)
    for i, m in enumerate(POOL):
        if w[i]:
            blend += w[i] * m["oof"]
    oof_honest[sc] = blend[sc]
    print(f"  fold {f}: selected {int((w>0).sum())} members | held-out AUC "
          f"{auc_on(sc, blend.astype(np.float32)):.6f} | best single on that fold "
          f"{max(auc_on(sc, m['oof']) for m in POOL):.6f}", flush=True)

blend_full, wfull = greedy_idx(POOL, np.ones(N, bool), rounds=20)
print(f"\n  HONEST cross-fitted blend OOF  : {roc_auc_score(y, oof_honest.astype(np.float32)):.6f}")
print(f"  best single model OOF          : {POOL[0]['auc']:.6f}  ({POOL[0]['dir']})")
print(f"  v8 reference                   : 0.946055")
print(f"  in-sample (optimistic) blend   : {roc_auc_score(y, blend_full.astype(np.float32)):.6f}")
nz = [(POOL[i]['dir'], POOL[i]['col'], int(wfull[i])) for i in np.argsort(-wfull) if wfull[i] > 0]
print(f"\n  members chosen in-sample ({len(nz)}):")
for d, c, k in nz[:15]:
    print(f"     x{k}  {d[:58]:<59} {c[:24]}")
np.save("research/honest_blend_oof.npy", oof_honest)
pd.DataFrame(nz, columns=["dir", "col", "weight"]).to_csv("research/blend_members.csv", index=False)
