"""SPIKE 2 — do v11's two changes survive a FAITHFUL pipeline?

Why this exists: v11 landed at LB 0.94620, only +0.00003 over v8, when the held-out
measurement had predicted +0.0005. The v11 Kaggle log would have said which change failed to
transfer, and there is no log. But the log is not the only way to find out.

The earlier measurement (research/measure_bagging_anchors.py) used a TOY feature set: 13 raw
columns plus digits, CatBoost depth 7. v11's real design is ~290 columns, of which 231 are
already target-encoded. An anchor feature that adds information to a 13-column model can add
nothing to a model that already target-encodes the same keys three times over -- the TE columns
may already carry everything the anchors do. That is the most likely reason the gain did not
transfer, and it is testable directly.

So this runs the REAL ev_s6e9_v11.build_features / fold_matrix, and ablates each of v11's two
changes against it on a 50/50 held-out split:
  full anchors (69) vs v8's anchors (13)      -> is C1 real on the faithful pipeline?
  with vs without the $50/$250 income bins    -> is v10's keeper real?
  1 seed vs 3 seeds of the same design        -> is C2 real here, not just on the toy set?
Absolute AUC is a CatBoost proxy (~0.94x), so only the deltas matter.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoost, Pool
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ev_s6e9_v11 as V

OUT = os.path.dirname(os.path.abspath(__file__))
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
orig = V.load_original(tr, [])
y = V.encode_target(tr[V.TARGET])

A, B = train_test_split(np.arange(len(tr)), test_size=0.5, random_state=42, stratify=y)
print(f"fit rows {len(A)}  held-out rows {len(B)}", flush=True)


def design(full_anchors, smooth_keys):
    V.FULL_ANCHOR_SET = full_anchors
    V.SMOOTH_KEYS = list(smooth_keys)
    F = V.build_features(tr, te, orig)
    (bt, bv, _), (wt, wv, _) = V.fold_matrix(F, y, A, B, 42)
    Xa, Xb = np.hstack(bt), np.hstack(bv)      # tree design: no LR-only window blocks
    ncol = Xa.shape[1]
    del F, bt, bv, wt, wv
    return Xa, Xb, ncol


BASE_KEYS = ["income_exact_int", "income50_floor", "income250_floor",
             "income100_floor", "income1000_floor", "commute_integer"]
NO_FINE = ["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"]


def run(tag, full_anchors, keys, seed):
    t = time.time()
    Xa, Xb, ncol = design(full_anchors, keys)
    m = CatBoost({"iterations": 1200, "learning_rate": 0.07, "depth": 7,
                  "loss_function": "Logloss", "eval_metric": "AUC",
                  "random_seed": seed, "verbose": 0, "thread_count": -1})
    m.fit(Pool(Xa, y[A]))
    p = np.asarray(m.predict(Xb, prediction_type="Probability"))[:, 1]
    np.save(os.path.join(OUT, f"spike2_{tag}.npy"), p)
    print(f"{tag:<26} cols={ncol:<5} held-out AUC={roc_auc_score(y[B], p):.6f} "
          f"({time.time()-t:.0f}s)", flush=True)
    return p


yB = y[B]
p_small = run("anchors13", False, BASE_KEYS, 42)
p_nofine = run("no_fine_income_bins", True, NO_FINE, 42)
p_f42 = run("full_anchors69_s42", True, BASE_KEYS, 42)
p_f7 = run("full_anchors69_s7", True, BASE_KEYS, 7)
p_f13 = run("full_anchors69_s13", True, BASE_KEYS, 13)

a13, anf, a42 = roc_auc_score(yB, p_small), roc_auc_score(yB, p_nofine), roc_auc_score(yB, p_f42)
bag3 = roc_auc_score(yB, (p_f42 + p_f7 + p_f13) / 3)
print("\n" + "=" * 66)
print("FAITHFUL-PIPELINE ABLATION  (reference = v11's own design, seed 42)")
print(f"  C1  full 69 anchors vs v8's 13      : {a42-a13:+.6f}")
print(f"  v10 keeper: $50/$250 bins on/off    : {a42-anf:+.6f}")
print(f"  C2  3-seed bagging vs single seed   : {bag3-a42:+.6f}")
print(f"      (single {a42:.6f} -> bagged {bag3:.6f})")
print("=" * 66)
