"""SPIKE — is there hardcodable determinism left above 0.9462?

Output of this file is an ANSWER, not code we keep. Nothing here ships.

The failure this is designed to avoid: last session a scan for all-zero income *intervals*
found 95 candidates over 24,666 train rows, looked worth +0.052 AUC, and on held-out data
cost -0.0177. The bug was the representation. A contiguous run of zero-label rows in
income-sorted order can span a wide, sparse income range; as an *interval* it swallows
unseen values that were never constrained.

So this probe uses exact keys, not intervals, and every candidate is discovered on one
half of train and scored on the other half using a real model's ranking (the cached
CatBoost out-of-fold predictions in research/pb_holdout.npy). Purity on unseen data and
the actual AUC delta are reported separately, because a perfectly pure cell the model
already ranks correctly is worth nothing.

Nothing is kept unless it is BOTH pure on the validation half AND gains AUC there.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TR = "data/train.csv"
PB = "research/pb_holdout.npy"
BASE_RATE = 0.174645

tr = pd.read_csv(TR)
tr["y"] = (tr.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").to_numpy(np.int8)
A, B = train_test_split(tr, test_size=0.5, random_state=42, stratify=tr.y)
A, B = A.reset_index(drop=True), B.reset_index(drop=True)
assert os.path.exists(PB), "run research/measure_bagging_anchors.py first (needs pb_holdout.npy)"
pB = np.load(PB)
yB = B.y.to_numpy()
base_auc = roc_auc_score(yB, pB)
print(f"discovery half A={len(A)}  validation half B={len(B)}")
print(f"model ranking on B (CatBoost, no bands): AUC {base_auc:.6f}\n")

NUMK = ["Annual_Income_USD", "Daily_Commute_km", "Age", "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
CATK = ["Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible",
        "City_Type", "Current_Car_Type", "Gender"]


def keyframe(df, spec):
    """A cell key = one or more exact feature values. spec lists column names,
    optionally 'col:qNN' meaning quantised to NN units."""
    parts = []
    for s in spec:
        if ":" in s:
            col, q = s.split(":")
            parts.append(np.floor(pd.to_numeric(df[col], errors="coerce").fillna(0) / float(q)))
        else:
            parts.append(df[s].astype(str).to_numpy())
    return pd.DataFrame({f"k{i}": p for i, p in enumerate(parts)})


def pin(probs, prio):
    """Stable epsilon-ordered re-rank that respects a 3-level priority."""
    p = np.asarray(probs, float)
    order = np.lexsort((np.arange(len(p)), p, prio))
    out = np.empty(len(p))
    out[order] = np.linspace(1e-6, 1 - 1e-6, len(p))
    return out


def probe(name, spec, minsup, maxminority_frac=0.0):
    """Find cells 100% pure on A, then report purity and AUC delta on unseen B."""
    ka, kb = keyframe(A, spec), keyframe(B, spec)
    g = pd.DataFrame({"k": ka.astype(str).agg("|".join, axis=1), "y": A.y.to_numpy()}) \
          .groupby("k").y.agg(["sum", "count"])
    g = g[g["count"] >= minsup]
    neg = g.index[(g["sum"] == 0)].to_numpy()
    pos = g.index[(g["sum"] == g["count"])].to_numpy()
    neg = [k for k in neg if binomtest(0, int(g.loc[k, "count"]), BASE_RATE).pvalue < 1e-8]
    pos = [k for k in pos
           if binomtest(int(g.loc[k, "sum"]), int(g.loc[k, "count"]), BASE_RATE,
                        alternative="greater").pvalue < 1e-8]
    if not len(neg) and not len(pos):
        print(f"{name:<34} no pure cell with support>={minsup}")
        return
    kbv = kb.astype(str).agg("|".join, axis=1).to_numpy()
    mn = np.isin(kbv, neg); mp = np.isin(kbv, pos)
    prio = np.ones(len(yB), np.int8)
    prio[mn] = 0; prio[mp & ~mn] = 2
    new = pin(pB, prio)
    auc = roc_auc_score(yB, new)
    imp_n = int(yB[mn].sum()); imp_p = int((yB[mp] == 0).sum())
    purity_ok = (imp_n / max(mn.sum(), 1) <= maxminority_frac
                 and imp_p / max(mp.sum(), 1) <= maxminority_frac)
    print(f"{name:<34} cells {len(neg):>6}neg/{len(pos):>4}pos | "
          f"B rows {int(mn.sum())+int(mp.sum()):>6} | "
          f"impurities {imp_n}+{imp_p:>5} | "
          f"delta {auc-base_auc:+.6f} | "
          f"{'KEEP' if purity_ok and auc-base_auc>1e-5 else 'DROP'}")
    return dict(name=name, neg=len(neg), pos=len(pos), rows=int(mn.sum() + mp.sum()),
                imp=imp_n + imp_p, delta=auc - base_auc, keep=purity_ok and auc - base_auc > 1e-5)


print("=== exact-value cells (the representation that survives cross-fitting) ===")
results = []
for nm, spec, ms in [
    ("income exact",            ["Annual_Income_USD"], 30),
    ("income exact",            ["Annual_Income_USD"], 60),
    ("commute exact",           ["Daily_Commute_km"], 30),
    ("income x subsidy",        ["Annual_Income_USD", "Subsidy_Available"], 30),
    ("income x anxiety",        ["Annual_Income_USD", "Range_Anxiety_Level"], 30),
    ("income x homecharge",     ["Annual_Income_USD", "Home_Charging_Possible"], 30),
    ("income x env",            ["Annual_Income_USD", "Environmental_Concern_Level"], 30),
    ("income x commute:1",      ["Annual_Income_USD", "Daily_Commute_km:1"], 30),
    ("income x subsidy x anx",  ["Annual_Income_USD", "Subsidy_Available", "Range_Anxiety_Level"], 30),
    ("full 13-feature tuple",   NUMK + CATK, 5),
]:
    r = probe(nm, spec, ms)
    if r:
        results.append(r)

print("\n=== quantised income bands (interval-style, for contrast) ===")
for q in (1, 10, 50, 100, 250):
    r = probe(f"income floored to {q}", [f"Annual_Income_USD:1" if q == 1 else f"Annual_Income_USD:{q}"], 120)
    if r:
        results.append(r)

print("\n=== the two band sets we ALREADY ship in v11, re-checked on unseen B ===")
cur_no = [(38174, 41384), (48002, 48589), (48657, 48779), (48981, 49491), (49646, 49809),
          (50345, 50525), (56945, 57208), (59081, 59192), (59227, 59296), (59602, 59711),
          (59754, 59802), (59989, 60417), (60425, 60499), (62223, 62260), (62998, 63361),
          (63363, 63384), (63991, 64404), (64640, 64706), (65108, 65190), (65228, 65409),
          (83985, 84164), (84223, 84353), (90151, 90189), (103103, 103314)]
cur_buy = [(170537, 188549), (92106, 92207)]
incB = np.floor(pd.to_numeric(B.Annual_Income_USD)).to_numpy()
comB = pd.to_numeric(B.Daily_Commute_km).to_numpy()
mn = np.zeros(len(B), bool); mp = np.zeros(len(B), bool)
for a, b in cur_no: mn |= (incB >= a) & (incB <= b)
mn |= comB >= 83.0
for a, b in cur_buy: mp |= (incB >= a) & (incB <= b)
mp &= ~mn
prio = np.ones(len(B), np.int8); prio[mn] = 0; prio[mp] = 2
auc = roc_auc_score(yB, pin(pB, prio))
print(f"v11's current bands on unseen B: rows {int(mn.sum())+int(mp.sum())}, "
      f"impurities {int(yB[mn].sum())}+{int((yB[mp]==0).sum())}, delta {auc-base_auc:+.6f}")

R = pd.DataFrame(results)
if len(R):
    R.to_csv("research/spike_hardcode_results.csv", index=False)
    print(f"\nbest delta seen: {R.delta.max():+.6f}   cells worth KEEPING: {int(R.keep.sum())}")
