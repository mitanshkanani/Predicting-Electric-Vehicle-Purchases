"""Local verification for ev_s6e9_v10.py. LightGBM/XGBoost aren't installed here, so
models are stubbed; everything else — target encoding, windows, fold-ranking, bands,
the health check and the gate — runs for real.

Run:  python test_v10.py
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
import ev_s6e9_v10 as V

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

rs = np.random.RandomState(0)

# ============================================================================
# 1. te_map / te_crossfit parity with sklearn TargetEncoder, at v10's smoothings
# ============================================================================
n = 5000
df = pd.DataFrame({"cat": rs.choice(list("abcdefghij"), n),
                   "hi": rs.choice([str(v) for v in range(1200)], n)})
p = df["cat"].map({c: rs.rand() for c in "abcdefghij"})
y = (p.to_numpy() > rs.rand(n)).astype(int)
gmean, var_y = float(y.mean()), float(np.var(y.astype(float)))

for smooth in (2.0, 5.0, 25.0, "auto"):
    for col in ("cat", "hi"):
        codes = np.asarray(pd.Categorical(df[col].astype(str)).codes, dtype=np.int32)
        nc = int(codes.max()) + 1
        te = TargetEncoder(smooth=smooth, cv=5)
        te.fit(df[[col]], y)
        ref = np.asarray(te.transform(df[[col]])[:, 0], dtype=np.float64)
        mine = V.te_map(codes, y, nc, smooth, gmean, var_y).astype(np.float64)[codes]
        check(f"full-map parity {col} smooth={smooth}", np.abs(ref - mine).max() < 1e-5,
              f"max|d|={np.abs(ref-mine).max():.2e}")

for smooth in (2.0, 5.0, 25.0):
    codes = np.asarray(pd.Categorical(df["cat"].astype(str)).codes, dtype=np.int32)
    nc = int(codes.max()) + 1
    ref = TargetEncoder(smooth=smooth, cv=5, shuffle=True, random_state=42)\
        .fit_transform(df[["cat"]], y)[:, 0].astype(np.float64)
    mine = V.te_crossfit(codes, y, nc, smooth, gmean, var_y, 42).astype(np.float64)
    check(f"crossfit matches sklearn smooth={smooth}", np.abs(ref - mine).max() < 1e-4,
          f"max|d|={np.abs(ref-mine).max():.2e}")

# ============================================================================
# 2. THE v9 REGRESSION TEST. Raw probabilities concatenated across folds must
#    collapse; fold_pct() must restore them. This is the bug that cost 0.0025 LB.
# ============================================================================
N, K, target = 200000, 5, 0.939921
ysim = (np.random.default_rng(0).random(N) < 0.1746).astype(np.int8)
folds = np.arange(N) % K
d = norm.ppf(target) * np.sqrt(2)
OFF = 0.65            # sd that reproduced v9's -0.0327 gap in the design sim
rng = np.random.default_rng(3)
raw = np.empty(N)
ranked = np.empty(N)
for f in range(K):
    m = folds == f
    s = np.where(ysim[m] == 1, d / 2, -d / 2) + rng.normal(size=int(m.sum())) + rng.normal(0, OFF)
    raw[m] = 1.0 / (1.0 + np.exp(-s))
    ranked[m] = V.fold_pct(s)
mean_per_fold = np.mean([roc_auc_score(ysim[folds == f], raw[folds == f]) for f in range(K)])
pooled_raw = roc_auc_score(ysim, raw)
pooled_rank = roc_auc_score(ysim, ranked)
check("v9 failure mode reproduced by raw-probability pooling",
      pooled_raw < mean_per_fold - 0.01, f"pooled {pooled_raw:.6f} vs per-fold {mean_per_fold:.6f}")
check("fold_pct() removes it", abs(pooled_rank - mean_per_fold) < V.FOLD_GAP_TOL,
      f"pooled(foldrank) {pooled_rank:.6f} gap {pooled_rank-mean_per_fold:+.6f} "
      f"(tol {V.FOLD_GAP_TOL})")
check("fold_pct output is a percentile in (0,1]",
      ranked.min() > 0 and ranked.max() <= 1.0 + 1e-6)

# ============================================================================
# 3. WindowTE: self-exclusion, and no leakage from a row's own label
# ============================================================================
vals = np.sort(rs.uniform(30000, 60000, 4000))
yv = (rs.rand(4000) < 0.2).astype(float)
w = V.WindowTE(200.0).fit(vals, yv)
solo = V.WindowTE(200.0).fit(vals, yv)
base = solo.transform(vals, self_y=yv)
check("window self-exclusion changes the value", not np.allclose(base, solo.transform(vals)))
# a row whose own label is 1 must be pulled DOWN by self-exclusion, never up
ones = yv == 1
check("self-exclusion removes, not adds, the own label",
      np.mean(base[ones] - solo.transform(vals)[ones]) < 0)
gm = float(yv.mean())
check("window mean is centred on the base rate", abs(base.mean() - gm) < 0.02,
      f"{base.mean():.4f} vs {gm:.4f}")
# leakage probe: flip ALL labels in a tight neighbourhood, its encoding must move
y_flip = yv.copy()
y_flip[:50] = 1 - y_flip[:50]
w2 = V.WindowTE(200.0).fit(vals, y_flip)
b2 = w2.transform(vals, self_y=y_flip)
check("window reacts to label changes elsewhere", not np.allclose(base[600:800], b2[600:800]))
check("window is finite", np.isfinite(base).all())

# ============================================================================
# 4. build_features: the two new fine-income keys are TE'd AND frequency-encoded
# ============================================================================
tr = pd.read_csv(os.path.join(HERE, "data", "train.csv")).sample(15000, random_state=1)\
    .reset_index(drop=True)
te_df = pd.read_csv(os.path.join(HERE, "data", "test.csv")).head(4000).reset_index(drop=True)
ytr = V.encode_target(tr[V.TARGET])
orig = V.load_original(tr, [])
F = V.build_features(tr, te_df, orig)
check("fine-income bins are target-encoded",
      {"income50_floor", "income250_floor"} <= set(F["te_src"]))
check("v8's income ladder survived",
      {"income_exact_int", "income100_floor", "income1000_floor", "commute_integer"}
      <= set(F["te_src"]))
check("digit columns are STILL target-encoded",
      sum(c.endswith("_digit-1") for c in F["te_src"]) == len(V.NUMERIC),
      f"{sum('_digit' in c for c in F['te_src'])} digit keys")
check("TE_SMOOTHS is the low ladder", V.TE_SMOOTHS == (2.0, 5.0, 25.0), str(V.TE_SMOOTHS))
check("anchors loaded from the real dataset", orig is not None, f"{0 if orig is None else len(orig)} rows")

(bt, bv, bc), (wt, wv, wc) = V.fold_matrix(F, ytr, np.arange(11000), np.arange(11000, 15000))
Xa = np.hstack(bt)
check("tree design width = raw + TE", Xa.shape[1] == len(F["raw_num"]) + len(F["te_src"])*3,
      f"{Xa.shape[1]} cols")
check("LR design is strictly wider by the windows",
      np.hstack(bt + wt).shape[1] == Xa.shape[1] + len(wt))
check("no NaN/inf in either design",
      np.isfinite(Xa).all() and np.isfinite(np.hstack(bv + wv)).all()
      and np.isfinite(np.hstack(bc + wc)).all())

# held-out labels must not move the val or test blocks (leakage)
yf = ytr.copy(); yf[11000:] = 1 - yf[11000:]
(bt2, bv2, bc2), (wt2, wv2, wc2) = V.fold_matrix(F, yf, np.arange(11000), np.arange(11000, 15000))
check("val block unchanged when held-out labels flip", np.array_equal(np.hstack(bv2), np.hstack(bv)))
check("test block unchanged when held-out labels flip", np.array_equal(np.hstack(bc2), np.hstack(bc)))
check("LR val window block unchanged too", np.array_equal(np.hstack(wv2), np.hstack(wv)))
check("LR test window block unchanged too", np.array_equal(np.hstack(wc2), np.hstack(wc)))
yf2 = ytr.copy(); yf2[:11000] = 1 - yf2[:11000]
(_, bv3, _), (_, wv3, _) = V.fold_matrix(F, yf2, np.arange(11000), np.arange(11000, 15000))
check("(contrast) val block DOES react to train labels",
      not np.array_equal(np.hstack(bv3), np.hstack(bv))
      and not np.array_equal(np.hstack(wv3), np.hstack(wv)))

# ============================================================================
# 5. Bands
# ============================================================================
m = te_df.head(20000) if len(te_df) >= 20000 else te_df
inc = np.rint(pd.to_numeric(m["Annual_Income_USD"]).to_numpy())
com = pd.to_numeric(m["Daily_Commute_km"]).to_numpy()
probs = rs.rand(len(m))
out = V.apply_bands(probs, inc, com)
buy, nobuy = V.band_masks(inc, com)
check("bands: no forced ties", len(np.unique(out)) == len(out))
check("bands: buyer block on top", out[buy].min() > out[~(buy | nobuy)].max())
check("bands: non-buyer block bottom", out[nobuy].max() < out[~(buy | nobuy)].min())
mid = ~(buy | nobuy)
check("bands: intra-block order preserved",
      np.array_equal(pd.Series(probs[mid]).rank().to_numpy(), pd.Series(out[mid]).rank().to_numpy()))
check("bands: in [0,1]", out.min() >= 0 and out.max() <= 1)

# ============================================================================
# 6. End-to-end with stubbed models.
#    Two things must hold, and they are precisely v9's two failure modes:
#      (a) the health check must SEE a per-fold drifting member (raw gap large)
#          while the fold-rank OOF it actually gates on stays clean;
#      (b) when LightGBM is the weak member the gate must still write the strong
#          one as submission_v10_single.csv — v9 hardcoded lgbm_ref and shipped
#          one of the two worst candidates.
# ============================================================================
work = os.path.join(HERE, "_v10_smoke")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
big_tr = tr.sample(9000, random_state=7).reset_index(drop=True)
big_te = te_df.head(2000).reset_index(drop=True)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.N_SPLITS = 3
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")
V.CV_GATE = 0.999          # force the "below gate" branch so we can read it in the log
cwd = os.getcwd(); os.chdir(work); sys.argv = ["x"]

def noisy(yb, jitter, seed, off=0.0, gain=0.9):
    z = yb * gain - gain / 2 + np.random.RandomState(seed).rand(len(yb)) * jitter + off
    return (1.0 / (1.0 + np.exp(-z))).astype("float32")

ONES = lambda n: np.ones(n, dtype=np.int8)
drift = {"n": 0}
def weak_lgb(Xa, ya, Xb, yb, Xc):
    """Weakest member, and it jumps calibration level between folds: v9's LightGBM."""
    drift["n"] += 1
    return (noisy(yb, 1.45, drift["n"], off=4.0 * (drift["n"] % 2)),
            noisy(ONES(Xc.shape[0]), 1.45, 100 + drift["n"]))
def mid_xgb(Xa, ya, Xb, yb, Xc, use_gpu=False):
    return (noisy(yb, 1.38, 200), noisy(ONES(Xc.shape[0]), 1.38, 300))
def best_lr(Xa, ya, Xb, yb, Xc):
    return (noisy(yb, 1.32, 400), noisy(ONES(Xc.shape[0]), 1.32, 500))

V.run_lightgbm, V.run_xgboost, V.run_lr = weak_lgb, mid_xgb, best_lr
V.ensure_import = lambda p: True
buf = io.StringIO()
crashed = None
try:
    with contextlib.redirect_stdout(buf):
        V.main()
except Exception:
    import traceback; crashed = traceback.format_exc()
finally:
    os.chdir(cwd)
txt = buf.getvalue()
check("end-to-end main() ran", crashed is None, (crashed or "")[-600:])

files = sorted(f for f in os.listdir(work) if f.startswith("submission_v10"))
check("both submissions written", len(files) == 2, str(files))
for f in files:
    d = pd.read_csv(os.path.join(work, f))
    check(f"  {f}", list(d.columns) == ["id", V.TARGET] and len(d) == 2000
          and d[V.TARGET].notna().all() and d["id"].nunique() == 2000
          and 0 <= d[V.TARGET].min() and d[V.TARGET].max() <= 1,
          f"rows={len(d)} unique={d[V.TARGET].nunique()}")

def health_lines(t):
    out = {}
    for line in t.splitlines():
        s = line.strip()
        if "per-fold mean" in s:
            name = s.split()[0]
            raw = float(s.split("pooled(raw)")[1].split("[")[1].split("]")[0])
            frn = float(s.split("pooled(foldrank)")[1].split("[")[1].split("]")[0])
            out[name] = (raw, frn, "UNHEALTHY" in s)
    return out
h = health_lines(txt)
check("health check ran for all 3 members", set(h) == {"lgbm_ref", "xgb_ref", "lr"}, str(sorted(h)))
check("(a) drift DETECTED in the raw-probability OOF, as in v9",
      h.get("lgbm_ref", (0, 0, 0))[0] < -0.01, f"raw gap {h.get('lgbm_ref', ('?',))[0]:+.6f}")
check("(a) fold-rank OOF stays clean despite it -> gate is trustworthy",
      all(abs(frn) <= V.FOLD_GAP_TOL for _, frn, _ in h.values())
      and not any(u for _, _, u in h.values()), str(h))
def gate_aucs(t):
    out = {}
    for line in t.splitlines():
        s = line.strip()
        if " OOF = " in s and "vs bar" in s:
            out[s.split()[0]] = float(s.split("OOF = ")[1].split()[0])
    return out
g = gate_aucs(txt)
check("members ranked by real quality (lgbm_ref weakest)",
      len(g) == 3 and g["lgbm_ref"] < g["xgb_ref"] < g["lr"], str(g))

best_line = [l for l in txt.splitlines() if "BEST MEMBER" in l]
check("(b) gate chose a member, and it is NOT the hardcoded lgbm_ref",
      len(best_line) == 1 and best_line[0].split()[0] in ("lr", "xgb_ref"),
      best_line[0] if best_line else "none")
sub = [l for l in txt.splitlines() if l.startswith("SUBMIT")]
check("(b) SUBMIT ships that chosen member, not lgbm_ref",
      len(sub) == 1 and "-> single: " in sub[0]
      and sub[0].split("single: ")[1].split(" ")[0] in ("lr", "xgb_ref"),
      sub[0] if sub else "none")
check("below-gate branch reported", "BELOW gate 0.999" in (sub[0] if sub else ""), sub[0] if sub else "")
check("HEALTHY verdict surfaced on the SUBMIT line", "HEALTHY" in (sub[0] if sub else ""))
check("no alarm on member lines (only the blend line can carry it)",
      not any("IMPLAUSIBLE" in l for l in txt.splitlines() if "vs best:" in l))
check("implausible blend gain is named, not shipped",
      any("IMPLAUSIBLE" in l for l in txt.splitlines())
      and "-> single: " in sub[0],
      [l for l in txt.splitlines() if "BLEND" in l])
check("blend-vs-best gate printed with a paired interval",
      any("vs best member: delta=" in l and "se=" in l and "folds+=" in l for l in txt.splitlines()))

# ---- 6b. realistic members: ~0.99 correlated, all within ~0.004, small blend gain.
#         Nothing here should trip the artifact alarm. ----
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")
V.CV_GATE = 0.9462
calls = {"c": 0}
def correlated(own_noise, seed):
    """Shares one latent per fold, so members correlate ~0.99 like real GBDTs here."""
    def f(Xa, ya, Xb, yb, Xc, *a, **k):
        i = calls["c"] // 3
        calls["c"] += 1
        common = np.random.RandomState(1000 + i).randn(len(yb))
        z = yb * 2.3 + common + np.random.RandomState(seed).randn(len(yb)) * own_noise
        t = np.random.RandomState(500 + i).randn(Xc.shape[0]) \
            + np.random.RandomState(seed).randn(Xc.shape[0]) * own_noise
        return ((1/(1+np.exp(-z))).astype("float32"), (1/(1+np.exp(-t))).astype("float32"))
    return f
V.run_lightgbm, V.run_xgboost, V.run_lr = correlated(0.15, 11), correlated(0.20, 22), correlated(0.25, 33)
buf3 = io.StringIO(); crashed3 = None
os.chdir(work)
try:
    with contextlib.redirect_stdout(buf3):
        V.main()
except Exception:
    import traceback; crashed3 = traceback.format_exc()
finally:
    os.chdir(cwd)
txt3 = buf3.getvalue()
check("6b realistic run completes", crashed3 is None, (crashed3 or "")[-500:])
g3 = gate_aucs(txt3)
check("6b members land within 0.01 of each other",
      len(g3) == 3 and max(g3.values()) - min(g3.values()) < 0.01, str(g3))
check("6b no artifact alarm when the measurement is sound",
      "IMPLAUSIBLE" not in txt3, [l for l in txt3.splitlines() if "BLEND" in l])
sub3 = [l for l in txt3.splitlines() if l.startswith("SUBMIT")]
check("6b SUBMIT names exactly one artifact with its OOF",
      len(sub3) == 1 and ("-> single: " in sub3[0] or "-> blend" in sub3[0])
      and re.search(r"OOF 0\.\d{6}", sub3[0]) and "HEALTHY" in sub3[0], sub3[0] if sub3 else "none")
best3 = [l for l in txt3.splitlines() if "BEST MEMBER" in l]
check("6b best member is exactly the highest measured OOF, not a hardcoded name",
      len(best3) == 1 and abs(float(best3[0].split("OOF = ")[1].split()[0]) - max(g3.values())) < 1e-9,
      best3[0] if best3 else "none")
check("6b gate threshold reported, not forced",
      any("[PASS gate]" in l or "[below gate]" in l for l in txt3.splitlines()))

# ============================================================================
# 7. A member that dies must be dropped, not fatal; wrong shapes too
# ============================================================================
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nonexistent")
V.init_ckpt(work, "fresh-resume-off")
V.RESUME = False
os.chdir(work); sys.argv = ["x"]
V.run_lightgbm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated lgbm death"))
V.run_xgboost = lambda Xa, ya, Xb, yb, Xc, use_gpu=False: (np.random.rand(len(yb), 1),
                                                           np.random.rand(Xc.shape[0], 1))
V.run_lr = lambda *a, **k: (np.zeros(len(a[3]) + 11, "float32"), np.zeros(1, "float32"))
buf2 = io.StringIO()
crashed2 = None
try:
    with contextlib.redirect_stdout(buf2):
        V.main()
except Exception:
    import traceback; crashed2 = traceback.format_exc()
finally:
    os.chdir(cwd)
txt2 = buf2.getvalue()
check("dead lgbm dropped, run survives", crashed2 is None, (crashed2 or "")[-600:])
check("(n,1) xgb output normalised, not dropped",
      any("xgb_ref f0" in l for l in txt2.splitlines())
      and not any("xgb_ref" in l and "dropped" in l for l in txt2.splitlines()))
check("wrong-length lr dropped", any("lr f0" in l and "dropped" in l for l in txt2.splitlines()),
      [l for l in txt2.splitlines() if "lr f0" in l])
check("submission produced from the surviving member",
      any(f.startswith("submission_v10") for f in os.listdir(work)),
      sorted(f for f in os.listdir(work) if f.endswith(".csv")))
check("_as_1d flattens (n,1)", V._as_1d(np.random.rand(50, 1)).shape == (50,))
check("_as_1d repairs NaN", bool(np.isfinite(V._as_1d(np.array([0.1, np.nan, 0.9]))).all()))
V.RESUME = True
shutil.rmtree(work, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
