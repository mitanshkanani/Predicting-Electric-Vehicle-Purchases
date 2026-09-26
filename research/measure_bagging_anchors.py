"""Measure the two remaining candidate gains, on a held-out split, before spending
a submission on either:

  (1) REPEATED-CV BAGGING. Averaging K independently-seeded models of the SAME recipe
      improves the test prediction without changing CV. This is the only lever that
      works when you are at a plateau, and it is the one thing CV structurally cannot
      show you. Measured here as single-model AUC vs averaged AUC on unseen data.

  (2) ANCHOR BREADTH. najiama's actual kernel source (LB 0.94638) maps the real 10k
      dataset's per-column target means onto EVERY categorical, numeric AND digit
      column - roughly 60 `_org_mean` features. Our v8/v10 do it for 13. That is a
      concrete un-replicated difference in the one recipe we are chasing.
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoost, Pool

tr = pd.read_csv("data/train.csv")
tr["y"] = (tr.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").to_numpy(np.int8)
A, B = train_test_split(tr, test_size=0.5, random_state=42, stratify=tr.y)
A, B = A.reset_index(drop=True), B.reset_index(drop=True)

NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home",
       "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
orig["y"] = (orig.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").astype(float)


def build(df, anchors):
    d = df[NUM].copy()
    for c in CATS:
        d[c] = df[c].astype(str)
    for k in range(-4, 4):
        for c in NUM:
            d[f"{c}_d{k}"] = (d[c] // (10.0 ** k) % 10).astype("int8")
    if anchors:
        gm = float(orig.y.mean())
        keys = {c: orig[c].astype(str) for c in CATS}
        for c in NUM:
            keys[c] = np.floor(pd.to_numeric(orig[c], errors="coerce").fillna(0)).astype(int).astype(str)
        stats = {c: orig.groupby(keys[c], observed=False).y.mean().to_dict() for c in list(CATS) + list(NUM)}
        for c in CATS:
            d[f"{c}_org_mean"] = d[c].map(stats[c]).fillna(gm).astype(float)
        for c in NUM:
            kk = np.floor(pd.to_numeric(d[c], errors="coerce").fillna(0)).astype(int).astype(str)
            d[f"{c}_org_mean"] = kk.map(stats[c]).fillna(gm).astype(float)
        for k in range(-4, 4):
            for c in NUM:
                src = f"{c}_d{k}"
                ok = np.floor(pd.to_numeric(orig[c], errors="coerce").fillna(0)) // (10 ** k) % 10
                st = orig.groupby(ok.astype(int).astype(str), observed=False).y.mean().to_dict()
                d[f"{src}_org_mean"] = d[src].astype(str).map(st).fillna(gm).astype(float)
    ci = list(range(len(NUM), len(NUM) + len(CATS)))
    return d, ci


def fit(seed, anchors, XA, ya, XB):
    m = CatBoost({"iterations": 1200, "learning_rate": 0.07, "depth": 7,
                  "loss_function": "Logloss", "eval_metric": "AUC",
                  "random_seed": seed, "verbose": 0, "thread_count": -1})
    m.fit(Pool(XA, ya, cat_features=anchors[1]))
    return np.asarray(m.predict(XB, prediction_type="Probability"))[:, 1]


yA, yB = A.y.to_numpy(), B.y.to_numpy()
Xb, ci = build(A, False)
Yb, _ = build(B, False)
preds = []
for seed in (42, 7, 13):
    t = time.time()
    p = fit(seed, (None, ci), Xb, yA, Yb)
    preds.append(p)
    print(f"base seed={seed:>3}  held-out AUC = {roc_auc_score(yB, p):.6f}   ({time.time()-t:.0f}s)",
          flush=True)
print(f"\n(1) BAGGING  mean of single-model AUCs : {np.mean([roc_auc_score(yB,p) for p in preds]):.6f}")
print(f"    BAGGING  averaged prediction AUC    : {roc_auc_score(yB, np.mean(preds,axis=0)):.6f}")
print(f"    >>> GAIN FROM 3-SEED BAGGING        : "
      f"{roc_auc_score(yB,np.mean(preds,axis=0)) - np.mean([roc_auc_score(yB,p) for p in preds]):+.6f}",
      flush=True)

Xa2, ci2 = build(A, True)
Yb2, _ = build(B, True)
p2 = fit(42, (None, ci2), Xa2, yA, Yb2)
base42 = roc_auc_score(yB, preds[0])
print(f"\n(2) ANCHORS  13 anchors (base)         : {base42:.6f}")
print(f"    ANCHORS  ~66 anchors (najiama-style): {roc_auc_score(yB, p2):.6f}")
print(f"    >>> GAIN FROM FULL ANCHOR SET       : {roc_auc_score(yB,p2)-base42:+.6f}")
np.save("research/bag_base.npy", np.array(preds))
np.save("research/bag_anchor.npy", p2)
