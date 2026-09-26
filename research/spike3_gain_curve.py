"""SPIKE 3 — calibrate "how much is a blend worth?" against measured correlation.

Every blend decision in this repo so far has been made by feeling. Aadit's three GBDT
families correlate at 0.9948-0.9991 and their blend is worth +0.000046. Our v8/v10/v11
correlate at 0.9995, so blending them is a no-op. Aadit's 0.94643 candidate will arrive
with some correlation rho against v11, and the question will be: is it worth a slot?

This answers it in advance. Across the 117 de-duplicated OOF vectors already in this repo,
for every pair it measures rho and the actual gain of a 50/50 rank blend over the better
member. The resulting curve turns "should we blend?" into a lookup.

Throwaway probe; the artefact is the curve, not the code.
"""
import hashlib
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "Aadit_try", "artifacts", "experiments")
tr = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
y = (tr.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").to_numpy(np.int8)
N = len(y)
MIN_AUC = 0.9440          # only pairs that could plausibly go in a serious blend

vecs, seen = {}, set()
for d in sorted(os.listdir(BASE)):
    dd = os.path.join(BASE, d)
    if not os.path.isdir(dd):
        continue
    for of in glob.glob(dd + "/*.csv"):
        if "oof" not in os.path.basename(of).lower():
            continue
        try:
            df = pd.read_csv(of)
        except Exception:
            continue
        if len(df) != N:
            continue
        for c in df.columns:
            if c in ("row_index", "id", "fold", "target_encoded") or df[c].dtype.kind != "f":
                continue
            v = df[c].to_numpy(float)
            if not np.isfinite(v).all() or np.nanstd(v) == 0:
                continue
            r = (rankdata(v) / v.size).astype(np.float32)
            h = hashlib.md5(r.tobytes()).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)
            a = roc_auc_score(y, r)
            if a >= MIN_AUC:
                vecs[f"{d}::{c}"] = (r, a)

keys = list(vecs)
print(f"{len(keys)} OOF vectors with AUC >= {MIN_AUC}")
A = np.array([vecs[k][1] for k in keys])
M = np.stack([vecs[k][0] for k in keys])

rows = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        rho = float(spearmanr(M[i], M[j]).statistic)
        if rho > 0.99995:
            continue                       # literally the same predictions
        b = roc_auc_score(y, (M[i].astype(np.float64) + M[j]) / 2)
        rows.append((rho, A[i], A[j], max(A[i], A[j]), b - max(A[i], A[j])))
R = pd.DataFrame(rows, columns=["rho", "auc_i", "auc_j", "best", "gain"])
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_gain_curve.csv"), index=False)
print(f"{len(R)} distinct pairs measured\n")

bins = [0.90, 0.97, 0.99, 0.994, 0.996, 0.998, 0.999, 0.9995, 1.0]
print("rho bucket        n     mean gain    median gain    max gain")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = R[(R.rho >= lo) & (R.rho < hi)]
    if len(s) == 0:
        continue
    print(f"[{lo:.4f},{hi:.4f}) {len(s):>6}   {s.gain.mean():>+12.6f}  "
          f"{s.gain.median():>+13.6f}  {s.gain.max():>+11.6f}")

print("\nbest blend seen for any pair, by rho:")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = R[(R.rho >= lo) & (R.rho < hi)]
    if len(s):
        t = s.loc[s.gain.idxmax()]
        print(f"  rho {t.rho:.5f}  members at {t.auc_i:.6f}/{t.auc_j:.6f} -> gain {t.gain:+.6f}")

eq = R[np.isclose(R.auc_i, R.auc_j, atol=2e-4)]
if len(eq) > 20:
    c = np.corrcoef(eq.rho, eq.gain)[0, 1]
    print(f"\nequal-strength pairs (n={len(eq)}): corr(rho, gain) = {c:+.3f}")
    print("  -> lower rho means larger gain, as theory says; the magnitude above is what to read")
print("\nWhen Aadit's file lands: measure rho against v11, look it up in the table, and")
print("only spend a submission if the bucket's median gain clears ~0.0002.")
