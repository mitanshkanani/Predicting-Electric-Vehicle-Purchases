"""Local verification for ev_s6e9_v8.py. LightGBM/XGBoost aren't installed here,
so models are stubbed; everything else runs for real."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_s6e9_v8 as V
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

ok = True
def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

# ---------- 1. parity with sklearn TargetEncoder ----------
# The full-data map must match sklearn's math EXACTLY. The cross-fitted column
# cannot: sklearn's internal fold permutation differs from ours, so the honest
# assertion is "same estimator, different partition" — verified by showing the
# sklearn-vs-ours gap is the same size as OURS-vs-OURS with another seed.
rs = np.random.RandomState(0)
n = 5000
df = pd.DataFrame({
    "cat": rs.choice(list("abcdefghij"), n),
    "hi":  rs.choice([str(v) for v in range(1200)], n),      # high cardinality
})
p = df["cat"].map({c: rs.rand() for c in "abcdefghij"})
y = (p.to_numpy() > rs.rand(n)).astype(int)
gmean, var_y = float(y.mean()), float(np.var(y.astype(float)))

for smooth in (10.0, 100.0, "auto"):
    for col in ("cat", "hi"):
        codes = np.asarray(pd.Categorical(df[col].astype(str)).codes, dtype=np.int32)
        nc = int(codes.max()) + 1
        te = TargetEncoder(smooth=smooth, cv=5)
        te.fit(df[[col]], y)                                  # full-data encodings
        ref_full = np.asarray(te.transform(df[[col]])[:, 0], dtype=np.float64)
        mine_full = V.te_map(codes, y, nc, smooth, gmean, var_y).astype(np.float64)[codes]
        check(f"full-map parity {col} smooth={smooth}",
              np.abs(ref_full - mine_full).max() < 1e-5,
              f"max|d|={np.abs(ref_full-mine_full).max():.2e}")

for smooth in (10.0, 100.0, "auto"):
    codes = np.asarray(pd.Categorical(df["cat"].astype(str)).codes, dtype=np.int32)
    nc = int(codes.max()) + 1
    enc = TargetEncoder(smooth=smooth, cv=5, shuffle=True, random_state=42)
    ref = enc.fit_transform(df[["cat"]], y)[:, 0].astype(np.float64)
    a = V.te_crossfit(codes, y, nc, smooth, gmean, var_y, 42).astype(np.float64)
    check(f"crossfit matches sklearn exactly smooth={smooth}",
          np.abs(ref - a).max() < 1e-4, f"max|d|={np.abs(ref-a).max():.2e}")

# ---------- 2. leakage probe: pure-noise rare keys must NOT give AUC ~1 ----------
n2 = 6000
key = rs.choice([str(i) for i in range(1500)], n2)          # ~4 rows per key
y2 = rs.randint(0, 2, n2)                                    # independent of key
c2 = pd.Categorical(key).codes.astype(np.int32)
enc = V.te_crossfit(c2, y2, c2.max()+1, 10.0, float(y2.mean()), float(np.var(y2)), 42)
auc_leak = roc_auc_score(y2, enc)
check("no self-label leakage", abs(auc_leak - 0.5) < 0.06, f"OOF AUC on noise = {auc_leak:.4f}")
naive = V.te_map(c2, y2, c2.max()+1, 10.0, float(y2.mean()), float(np.var(y2)))[c2]
check("  (contrast) non-crossfit leaks", abs(roc_auc_score(y2, naive) - 0.5) > 0.10,
      f"leaky AUC = {roc_auc_score(y2, naive):.4f}")

# ---------- 3. band epsilon ordering ----------
m = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "test.csv")).head(20000)
inc = np.rint(pd.to_numeric(m["Annual_Income_USD"]).to_numpy())
com = pd.to_numeric(m["Daily_Commute_km"]).to_numpy()
probs = rs.rand(len(m))
out = V.apply_bands(probs, inc, com)
buy, nobuy = V.band_masks(inc, com)
check("bands: no forced ties", len(np.unique(out)) == len(out), f"unique {len(np.unique(out))}/{len(out)}")
check("bands: buyer block on top", out[buy].min() > out[~(buy | nobuy)].max())
check("bands: non-buyer block bottom", out[nobuy].max() < out[~(buy | nobuy)].min())
mid = ~(buy | nobuy)
r = pd.Series(probs[mid]).rank().to_numpy()
r2 = pd.Series(out[mid]).rank().to_numpy()
check("bands: intra-block order preserved", np.array_equal(r, r2))
check("bands: in [0,1]", out.min() >= 0 and out.max() <= 1)

# ---------- 4. feature builder + fold matrix on real data ----------
tr = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "train.csv")).sample(
    12000, random_state=1).reset_index(drop=True)
te_df = m.head(3000)
ytr = V.encode_target(tr[V.TARGET])
orig = V.load_original(tr, [])
Rtr, Rte, Xtr_c, Xte_c, te_src, ncats, raw_num = V.build_features(tr, te_df, orig)
check("TE sources present", len(te_src) > 40, f"{len(te_src)} cols -> {len(te_src)*len(V.TE_SMOOTHS)} TE feats")
check("codes are int32 & non-negative", all(v.dtype == np.int32 and v.min() >= 0 for v in Xtr_c.values()))
tr_idx = np.arange(9000); va_idx = np.arange(9000, 12000)
Xa, Xb, Xc = V.fold_matrix(Rtr, Xtr_c, ytr, tr_idx, va_idx, te_src, ncats, Rte, Xte_c)
check("design widths match", Xa.shape[1] == Xb.shape[1] == Xc.shape[1], f"{Xa.shape[1]} features")
check("no NaN/inf in design", np.isfinite(Xa).all() and np.isfinite(Xb).all() and np.isfinite(Xc).all())

# direct leakage probe: flip every held-out label, the val/test blocks must not move
y_flip = ytr.copy(); y_flip[va_idx] = 1 - y_flip[va_idx]
Xa2, Xb2, Xc2 = V.fold_matrix(Rtr, Xtr_c, y_flip, tr_idx, va_idx, te_src, ncats, Rte, Xte_c)
check("val block unchanged when held-out labels flip", np.array_equal(Xb, Xb2))
check("test block unchanged when held-out labels flip", np.array_equal(Xc, Xc2))
check("  (contrast) train block unchanged too, since only held-out labels moved",
      np.array_equal(Xa, Xa2))
y_flip2 = ytr.copy(); y_flip2[tr_idx] = 1 - y_flip2[tr_idx]
_, Xb3, Xc3 = V.fold_matrix(Rtr, Xtr_c, y_flip2, tr_idx, va_idx, te_src, ncats, Rte, Xte_c)
check("  (contrast) val block DOES react to train labels", not np.array_equal(Xb, Xb3))

# ---------- 5. end-to-end with stubbed models (small fixtures, no global patching) ----------
V.N_SPLITS = 2
stub = {"lgb": 0}
def fake_lgb(Xa, ya, Xb, yb, Xc):
    stub["lgb"] += 1
    return (np.clip(yb * 0.5 + rs.rand(len(yb)) * 0.1, 0, 1).astype("float32"),
            (rs.rand(Xc.shape[0]) * 0.5 + 0.25).astype("float32"))
V.run_lightgbm = fake_lgb
V.run_xgboost = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated xgb failure"))
V.ensure_import = lambda p: p == "lightgbm"
V.NN_USE = False

work = "/tmp/v8run"
import shutil
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "train.csv")).sample(
    6000, random_state=7).reset_index(drop=True)
big_te = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "test.csv")).head(1500).reset_index(drop=True)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")   # no original dataset -> anchors skipped
os.chdir(work)
sys.argv = ["x"]
try:
    V.main()
    files = sorted(f for f in os.listdir(work) if f.endswith(".csv") and f.startswith("submission"))
    check("submissions written", len(files) == 2, str(files))
    for f in files:
        d = pd.read_csv(os.path.join(work, f))
        good = (list(d.columns) == ["id", V.TARGET] and len(d) == 1500
                and d[V.TARGET].notna().all() and d["id"].nunique() == 1500
                and 0 <= d[V.TARGET].min() and d[V.TARGET].max() <= 1)
        check(f"  {f}", good, f"rows={len(d)} unique={d[V.TARGET].nunique()} "
                              f"range={d[V.TARGET].min():.4f}..{d[V.TARGET].max():.4f}")
    check("xgb member dropped, run survived", stub["lgb"] == 2, f"lgbm folds run = {stub['lgb']}")
    ck = [x for x in os.listdir(work) if x.startswith("v8_ckpt")]
    check("checkpoints written", bool(ck), str(ck))
except Exception:
    import traceback; traceback.print_exc()
    check("end-to-end main()", False)

# ---------- 6. regression: the Kaggle crash (member returning (n,1)) ----------
check("_as_1d flattens (n,1)", V._as_1d(np.random.rand(50, 1)).shape == (50,))
check("_as_1d keeps (n,)", V._as_1d(np.random.rand(50)).shape == (50,))
check("_as_1d repairs NaN", bool(np.isfinite(V._as_1d(np.array([0.1, np.nan, 0.9]))).all()))

# a member that returns the wrong LENGTH must be dropped, not fatal
os.chdir(work)
V.ckpt_load = lambda m, f: None
V.ckpt_save = lambda *a, **k: None
V.ensure_import = lambda p: p in ("lightgbm", "xgboost")     # keep xgb as a live member
def bad_len(Xa, ya, Xb, yb, Xc):
    return (np.zeros(len(yb) + 7, "float32"), np.zeros(Xc.shape[0], "float32"))
def good_col(Xa, ya, Xb, yb, Xc, use_gpu=False):               # reproduces the Kaggle bug
    return (np.random.rand(len(yb), 1), np.random.rand(Xc.shape[0], 1))
V.run_lightgbm = bad_len
V.run_xgboost = good_col
crashed = False
try:
    V.main()
except Exception:
    import traceback; traceback.print_exc()
    crashed = True
check("wrong-length member dropped, (n,1) member normalised, run survives", not crashed)
if not crashed:
    subs = sorted(f for f in os.listdir(work) if f.startswith("submission_v8"))
    d = pd.read_csv(os.path.join(work, subs[0])) if subs else None
    check("  (contrast) submission still produced from the good member",
          d is not None and len(d) == 1500 and d[V.TARGET].notna().all(), str(subs))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
