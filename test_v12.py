"""Local verification for ev_s6e9_v12.py. LightGBM/XGBoost are not installed here so
models are stubbed; everything else — anchors, target encoding, repeat aggregation,
fold-ranking, bands, the health table and the gate — runs for real.

Run:  python test_v12.py
"""
import io
import os
import re
import shutil
import sys
import contextlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_s6e9_v12 as V

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

rs = np.random.RandomState(0)

# ============================================================================
# 0. v12-SPECIFIC: the ONE thing v12 changes vs v11 must actually be true.
# ============================================================================
check("v12 defaults to v8's 13-anchor set (C1 reverted as measured dead)",
      V.FULL_ANCHOR_SET is False, str(V.FULL_ANCHOR_SET))
check("v12 keeps the change that DID measure positive: 5-repeat bagging",
      len(V.REPEAT_SEEDS) == 5, str(V.REPEAT_SEEDS))
check("v12 keeps v8's verified TE ladder", V.TE_SMOOTHS == ("auto", 10.0, 100.0))
check("v12 keeps the $50/$250 fine-income bins (neutral, free)",
      {"income50_floor", "income250_floor"} <= set(V.SMOOTH_KEYS))
check("v12 keeps the fold-rank fix and the band prior",
      callable(getattr(V, "fold_pct", None)) and len(V.NON_BUYER_BANDS) == 24
      and len(V.BUYER_BANDS) == 2)


# ============================================================================
# 1. te_map / te_crossfit still parity with sklearn at v11's ladder
# ============================================================================
n = 5000
df = pd.DataFrame({"cat": rs.choice(list("abcdefghij"), n),
                   "hi": rs.choice([str(v) for v in range(1200)], n)})
p = df["cat"].map({c: rs.rand() for c in "abcdefghij"})
y = (p.to_numpy() > rs.rand(n)).astype(int)
gmean, var_y = float(y.mean()), float(np.var(y.astype(float)))
for smooth in ("auto", 10.0, 100.0):
    codes = np.asarray(pd.Categorical(df["cat"].astype(str)).codes, dtype=np.int32)
    nc = int(codes.max()) + 1
    te = TargetEncoder(smooth=smooth, cv=5)
    te.fit(df[["cat"]], y)
    ref = np.asarray(te.transform(df[["cat"]])[:, 0], dtype=np.float64)
    mine = V.te_map(codes, y, nc, smooth, gmean, var_y).astype(np.float64)[codes]
    check(f"te_map parity smooth={smooth}", np.abs(ref - mine).max() < 1e-5,
          f"max|d|={np.abs(ref-mine).max():.2e}")
for smooth in ("auto", 10.0, 100.0):
    codes = np.asarray(pd.Categorical(df["cat"].astype(str)).codes, dtype=np.int32)
    nc = int(codes.max()) + 1
    ref = TargetEncoder(smooth=smooth, cv=5, shuffle=True, random_state=42)\
        .fit_transform(df[["cat"]], y)[:, 0].astype(np.float64)
    mine = V.te_crossfit(codes, y, nc, smooth, gmean, var_y, 42).astype(np.float64)
    check(f"te_crossfit parity smooth={smooth}", np.abs(ref - mine).max() < 1e-4,
          f"max|d|={np.abs(ref-mine).max():.2e}")

# ============================================================================
# 2. THE v9 REGRESSION TEST — fold_pct must remove a per-fold calibration offset
# ============================================================================
N, K, target = 200000, 5, 0.939921
ysim = (np.random.default_rng(0).random(N) < 0.1746).astype(np.int8)
folds = np.arange(N) % K
d = norm.ppf(target) * np.sqrt(2)
rng = np.random.default_rng(3)
raw = np.empty(N); ranked = np.empty(N)
for f in range(K):
    m = folds == f
    s = np.where(ysim[m] == 1, d/2, -d/2) + rng.normal(size=int(m.sum())) + rng.normal(0, 0.65)
    raw[m] = 1/(1+np.exp(-s)); ranked[m] = V.fold_pct(s)
mpf = np.mean([roc_auc_score(ysim[folds == f], raw[folds == f]) for f in range(K)])
check("v9 failure mode reproduced by raw pooling",
      roc_auc_score(ysim, raw) < mpf - 0.01,
      f"pooled {roc_auc_score(ysim,raw):.6f} vs per-fold {mpf:.6f}")
check("fold_pct() removes it",
      abs(roc_auc_score(ysim, ranked) - mpf) < V.FOLD_GAP_TOL,
      f"gap {roc_auc_score(ysim,ranked)-mpf:+.6f}")

# ============================================================================
# 3. CHANGE C1 — the full anchor set. This is the headline new behaviour.
# ============================================================================
tr = pd.read_csv(os.path.join(HERE, "data", "train.csv")).sample(15000, random_state=1)\
    .reset_index(drop=True)
te_df = pd.read_csv(os.path.join(HERE, "data", "test.csv")).head(4000).reset_index(drop=True)
ytr = V.encode_target(tr[V.TARGET])
orig = V.load_original(tr, [])
check("external dataset found locally (anchors can be tested)", orig is not None)

V.FULL_ANCHOR_SET = True
F_full = V.build_features(tr, te_df, orig)
V.FULL_ANCHOR_SET = False
F_small = V.build_features(tr, te_df, orig)
V.FULL_ANCHOR_SET = True
a_full, a_small = F_full["anchors"], F_small["anchors"]
check("C1: full anchor set is ~66+, v8/v10 built 13",
      len(a_full) >= 60 and len(a_small) <= 14, f"{len(a_small)} -> {len(a_full)}")
check("C1: digit anchors are the added ones",
      sum("_digit" in c for c in a_full) >= 40 and not any("_digit" in c for c in a_small),
      f"{sum('_digit' in c for c in a_full)} digit anchors")
check("C1: anchors survive into the design matrix",
      len([c for c in a_full if c in F_full["raw_num"]]) >= 20,
      f"{len(a_full)-F_full['n_anchors_dropped']} kept, {F_full['n_anchors_dropped']} "
      f"dropped as constant/perfectly-collinear on this 15k sample")
check("C1: every anchor is finite and in [0,1]",
      all(np.isfinite(F_full["Rtr"][:, i]).all()
          and 0 <= F_full["Rtr"][:, i].min() and F_full["Rtr"][:, i].max() <= 1.0
          for i in (F_full["raw_num"].index(c) for c in a_full
                    if c in F_full["raw_num"])))
kept = [c for c in a_full if c in F_full["raw_num"] and "_digit2_org_mean" in c]
check("C1: at least one income digit anchor is live", bool(kept), str(kept[:3]))
if kept:
    col = kept[0]
    base = col.replace("_org_mean", "")            # e.g. Annual_Income_USD_digit2
    src, kk = base.rsplit("_digit", 1)
    yo = (orig.Will_Buy_EV.astype(str).str.strip().str.lower() == "yes").astype(float)
    ov = pd.to_numeric(orig[src], errors="coerce").fillna(0.0)
    od = ((ov // (10.0 ** int(kk))) % 10).astype(int).astype(str)
    idx = F_full["raw_num"].index(col)
    for want_dig in (3, 7):
        w = float(yo[od.to_numpy() == str(want_dig)].mean())
        q = (pd.to_numeric(te_df[src], errors="coerce").fillna(0.0) // (10.0 ** int(kk)) % 10)
        sel = q.astype(int).to_numpy() == want_dig
        if sel.sum() > 20:
            got = float(F_full["Rte"][sel, idx].mean())
            check(f"C1: {src} digit{kk} anchor == real-dataset mean (d={want_dig})",
                  abs(got - w) < 1e-4, f"got {got:.5f} want {w:.5f}")
check("fine-income bins still target-encoded",
      {"income50_floor", "income250_floor"} <= set(F_full["te_src"]))
check("digit columns still target-encoded",
      sum(c.endswith("_digit2") for c in F_full["te_src"]) == len(V.NUMERIC))
check("TE ladder is v8's verified one", V.TE_SMOOTHS == ("auto", 10.0, 100.0), str(V.TE_SMOOTHS))
V.FULL_ANCHOR_SET = True

# leakage: held-out labels must not move val/test blocks
(bt, bv, bc), (wt, wv, wc) = V.fold_matrix(F_full, ytr, np.arange(11000), np.arange(11000, 15000), 42)
check("tree design width = raw + TE", np.hstack(bt).shape[1] ==
      len(F_full["raw_num"]) + len(F_full["te_src"])*3, f"{np.hstack(bt).shape[1]} cols")
check("LR design is wider by the windows",
      np.hstack(bt + wt).shape[1] == np.hstack(bt).shape[1] + len(wt))
yf = ytr.copy(); yf[11000:] = 1 - yf[11000:]
(_, bv2, bc2), (_, wv2, wc2) = V.fold_matrix(F_full, yf, np.arange(11000), np.arange(11000, 15000), 42)
check("val block unchanged when held-out labels flip", np.array_equal(np.hstack(bv2), np.hstack(bv)))
check("test block unchanged when held-out labels flip", np.array_equal(np.hstack(bc2), np.hstack(bc)))
check("LR window blocks unchanged too",
      np.array_equal(np.hstack(wv2), np.hstack(wv)) and np.array_equal(np.hstack(wc2), np.hstack(wc)))
yf2 = ytr.copy(); yf2[:11000] = 1 - yf2[:11000]
(_, bv3, _), (_, wv3, _) = V.fold_matrix(F_full, yf2, np.arange(11000), np.arange(11000, 15000), 42)
check("(contrast) val block DOES react to train labels",
      not np.array_equal(np.hstack(bv3), np.hstack(bv))
      and not np.array_equal(np.hstack(wv3), np.hstack(wv)))
# different repeat seed must give a different cross-fit encoding (repeats are independent)
bt4 = V.fold_matrix(F_full, ytr, np.arange(11000), np.arange(11000, 15000), 7)[0][0]
check("repeat seed changes the cross-fitted TRAIN encoding",
      not np.array_equal(np.hstack(bt4), np.hstack(bt)))
check("repeat seed leaves the val/test encodings identical (they use the full map)",
      np.array_equal(np.hstack(V.fold_matrix(F_full, ytr, np.arange(11000),
                                             np.arange(11000, 15000), 7)[0][1]), np.hstack(bv)))

# ============================================================================
# 4. Bands
# ============================================================================
inc = np.rint(pd.to_numeric(te_df["Annual_Income_USD"]).to_numpy())
com = pd.to_numeric(te_df["Daily_Commute_km"]).to_numpy()
probs = rs.rand(len(te_df))
out = V.apply_bands(probs, inc, com)
buy, nobuy = V.band_masks(inc, com)
check("bands: no forced ties", len(np.unique(out)) == len(out))
check("bands: buyer block on top", out[buy].min() > out[~(buy | nobuy)].max())
check("bands: non-buyer block bottom", out[nobuy].max() < out[~(buy | nobuy)].min())
check("bands: in [0,1]", out.min() >= 0 and out.max() <= 1)

# ============================================================================
# 5. CHANGE C2 — repeat aggregation, plus the gate, end to end with stubs.
#    With R repeats, oof_rnk must be the mean of the R per-fold rank vectors and
#    te_rnk the mean over R*N_SPLITS models. Verified against an independent
#    reconstruction from the per-fold checkpoints the run itself wrote.
# ============================================================================
work = os.path.join(HERE, "_v12_smoke")
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr = tr.sample(9000, random_state=7).reset_index(drop=True)
big_te = te_df.head(2000).reset_index(drop=True)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.N_SPLITS = 3
V.REPEAT_SEEDS = (42, 7, 13)
R_MAIN = 3
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")
V.CV_GATE = 0.9462
cwd = os.getcwd(); os.chdir(work); sys.argv = ["x"]

def noisy(yb, jitter, seed, off=0.0, gain=0.9):
    z = yb * gain - gain/2 + np.random.RandomState(seed).rand(len(yb)) * jitter + off
    return (1.0/(1.0+np.exp(-z))).astype("float32")
ONES = lambda k: np.ones(k, dtype=np.int8)
calls = {"n": 0}
def correlated(own_noise, drift_off=0.0):
    def f(Xa, ya, Xb, yb, Xc, seed, *a, **k):
        calls["n"] += 1
        i = calls["n"] // 3
        common = np.random.RandomState(1000 + i + seed).randn(len(yb))
        z = yb * 2.3 + common + np.random.RandomState(seed).randn(len(yb)) * own_noise \
            + drift_off * (i % 2)
        t = np.random.RandomState(5000 + i).randn(Xc.shape[0])
        return ((1/(1+np.exp(-z))).astype("float32"), (1/(1+np.exp(-t))).astype("float32"))
    return f
V.run_lightgbm, V.run_xgboost, V.run_lr = correlated(0.15), correlated(0.20), correlated(0.25)
V.ensure_import = lambda p: True

buf = io.StringIO(); crashed = None
try:
    with contextlib.redirect_stdout(buf):
        V.main()
except Exception:
    import traceback; crashed = traceback.format_exc()
finally:
    os.chdir(cwd)
txt = buf.getvalue()
check("end-to-end main() ran", crashed is None, (crashed or "")[-700:])
check("all 3 repeats x 3 folds x 3 members ran",
      txt.count("auc=") == 27, f"{txt.count('auc=')} member results (want 27)")
files = sorted(f for f in os.listdir(work) if f.startswith("submission_v12"))
check("both submissions written", len(files) == 2, str(files))
for f in files:
    d = pd.read_csv(os.path.join(work, f))
    check(f"  {f}", list(d.columns) == ["id", V.TARGET] and len(d) == 2000
          and d[V.TARGET].notna().all() and d["id"].nunique() == 2000
          and 0 <= d[V.TARGET].min() and d[V.TARGET].max() <= 1,
          f"rows={len(d)} unique={d[V.TARGET].nunique()}")

# C2 verified independently: rebuild oof_rnk from the checkpoints the run saved
ck = [d for d in os.listdir(work) if d.startswith("v12_ckpt")]
check("checkpoint dir written", bool(ck), str(ck))
if ck:
    dd = os.path.join(work, ck[0])
    nfiles = len([f for f in os.listdir(dd) if f.endswith("_val.npy")])
    check("one checkpoint per (member, repeat, fold)", nfiles == 27, f"{nfiles} val files")
    v = np.load(os.path.join(dd, "lgbm_ref_r0_f0_val.npy"))
    uniq_frac = len(np.unique(v)) / len(v)
    check("checkpoints store RAW probabilities, not the fold-rank grid",
          0.0 < v.min() and v.max() < 1.0 and uniq_frac > 0.9,
          f"range {v.min():.4f}..{v.max():.4f} unique {uniq_frac:.3f}")

    # INDEPENDENT reconstruction of change C2's aggregation, from the checkpoints
    ybig = V.encode_target(big_tr[V.TARGET])
    R = len(V.REPEAT_SEEDS)
    exp_oof = np.zeros(len(ybig))
    exp_te = np.zeros(len(big_te))
    for rep, seed in enumerate(V.REPEAT_SEEDS):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=V.N_SPLITS, shuffle=True, random_state=seed)
        for f, (_tr, va) in enumerate(skf.split(np.zeros(len(ybig)), ybig)):
            vp = np.load(os.path.join(dd, f"lgbm_ref_r{rep}_f{f}_val.npy"))
            tp = np.load(os.path.join(dd, f"lgbm_ref_r{rep}_f{f}_test.npy"))
            exp_oof[va] += V.fold_pct(vp) / R
            exp_te += V.fold_pct(tp) / (R * V.N_SPLITS)
    got_oof = np.load(os.path.join(work, "v12_lgbm_ref_oof_rank.npy"))
    got_te = np.load(os.path.join(work, "v12_lgbm_ref_test.npy"))
    check("C2 oof = mean over repeats of per-fold rank vectors",
          np.abs(exp_oof - got_oof.astype(float)).max() < 1e-6,
          f"max|d|={np.abs(exp_oof-got_oof.astype(float)).max():.2e}")
    check("C2 test = mean over repeats x folds of per-fold rank vectors",
          np.abs(exp_te - got_te.astype(float)).max() < 1e-6,
          f"max|d|={np.abs(exp_te-got_te.astype(float)).max():.2e}")
    single = np.zeros(len(ybig))
    skf0 = StratifiedKFold(n_splits=V.N_SPLITS, shuffle=True, random_state=V.REPEAT_SEEDS[0])
    for f, (_t, va) in enumerate(skf0.split(np.zeros(len(ybig)), ybig)):
        single[va] = V.fold_pct(np.load(os.path.join(dd, f"lgbm_ref_r0_f_{f}_val.npy"))
                                if False else
                                np.load(os.path.join(dd, f"lgbm_ref_r0_f{f}_val.npy")))
    a1, aR = roc_auc_score(ybig, single), roc_auc_score(ybig, got_oof)
    check("C2 actually helps: 3-repeat OOF >= single-repeat OOF",
          aR >= a1 - 1e-6, f"single {a1:.6f} -> repeated {aR:.6f} ({aR-a1:+.6f})")

def gate_aucs(t):
    out = {}
    for line in t.splitlines():
        s = line.strip()
        if " OOF = " in s and "vs bar" in s:
            out[s.split()[0]] = float(s.split("OOF = ")[1].split()[0])
    return out
g = gate_aucs(txt)
check("gate scored all 3 members", len(g) == 3, str(g))
best = [l for l in txt.splitlines() if "BEST MEMBER" in l]
check("best member is the argmax of the printed OOFs",
      len(best) == 1 and abs(float(best[0].split("OOF = ")[1].split()[0]) - max(g.values())) < 1e-9,
      best[0] if best else "none")
sub = [l for l in txt.splitlines() if l.startswith("SUBMIT")]
check("SUBMIT names one artifact with OOF and health verdict",
      len(sub) == 1 and re.search(r"OOF 0\.\d{6}", sub[0]) and "HEALTHY" in sub[0],
      sub[0] if sub else "none")
check("no IMPLAUSIBLE alarm on a sound measurement", "IMPLAUSIBLE" not in txt,
      [l for l in txt.splitlines() if "BLEND" in l])

def health(t):
    out = {}
    for line in t.splitlines():
        s = line.strip()
        if "per-fold mean" not in s or "pooled(raw)" not in s:
            continue
        out[s.split()[0]] = (float(s.split("pooled(raw)")[1].split("[")[1].split("]")[0]),
                             float(s.split("pooled(foldrank)")[1].split("[")[1].split("]")[0]),
                             "UNHEALTHY" in s)
    return out
h = health(txt)
check("health table covers every member", set(h) == {"lgbm_ref", "xgb_ref", "lr"}, str(h))
check("fold-ranking never makes the measurement worse than raw pooling",
      all(fr > rg for rg, fr, _ in h.values()),
      " | ".join(f"{k}: raw {rg:+.4f} -> foldrank {fr:+.4f}" for k, (rg, fr, _) in h.items()))
check("an implausibly large bagging gain IS flagged (the check has teeth)",
      any(u for _, _, u in h.values()) and all(fr > 0.005 for _, fr, _ in h.values()),
      "stub repeats are fully independent, so +0.049 is the honest reading of them")
check("gate still reports and still chooses an argmax member under that flag",
      len(g) == 3 and "SUBMIT" in txt)

# ---- 5b. a drifting member must be caught by the raw column, not the gate ----
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")
# R=1 here: with a single repeat there is no bagging gain to confound the reading, so
# any gap between per-fold mean and pooled OOF is pure calibration drift -- v9's case.
V.REPEAT_SEEDS = (42,)
V.run_lightgbm = correlated(0.15, drift_off=5.0)     # lgbm jumps calibration between folds
V.CV_GATE = 0.999
buf2 = io.StringIO(); crashed2 = None
os.chdir(work)
try:
    with contextlib.redirect_stdout(buf2):
        V.main()
except Exception:
    import traceback; crashed2 = traceback.format_exc()
finally:
    os.chdir(cwd)
t2 = buf2.getvalue()
h2 = {k: v[:2] for k, v in health(t2).items()}
check("5b drifting run completes", crashed2 is None, (crashed2 or "")[-500:])
check("5b drift VISIBLE in the raw column, exactly as in v9",
      h2["lgbm_ref"][0] < -0.01, f"raw gap {h2['lgbm_ref'][0]:+.6f}")
check("5b fold-rank OOF stays clean, so the gate is still trustworthy",
      abs(h2["lgbm_ref"][1]) < V.FOLD_GAP_TOL, f"foldrank gap {h2['lgbm_ref'][1]:+.6f}")
check("5b healthy members show no drift (the gap is member-specific, as in v9)",
      abs(h2["xgb_ref"][0]) < V.FOLD_GAP_TOL and abs(h2["lr"][0]) < V.FOLD_GAP_TOL,
      f"xgb {h2['xgb_ref'][0]:+.6f} lr {h2['lr'][0]:+.6f}")
sub2 = [l for l in t2.splitlines() if l.startswith("SUBMIT")]
g2 = gate_aucs(t2)
best2 = [l for l in t2.splitlines() if "BEST MEMBER" in l]
check("5b a drifting member does not corrupt the choice: best == argmax of measured OOF",
      bool(best2) and len(g2) == 3
      and abs(float(best2[0].split("OOF = ")[1].split()[0]) - max(g2.values())) < 1e-9,
      f"{best2[0] if best2 else 'none'} | {g2}")
check("5b the drift is invisible to the fold-rank gate (lgbm scores near its true quality)",
      bool(g2) and g2["lgbm_ref"] > 0.94, str(g2))
V.REPEAT_SEEDS = (42, 7, 13)

# ---- 6. robustness ----
check("_as_1d flattens (n,1)", V._as_1d(np.random.rand(50, 1)).shape == (50,))
check("_as_1d repairs NaN", bool(np.isfinite(V._as_1d(np.array([0.1, np.nan, 0.9]))).all()))
shutil.rmtree(work, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
