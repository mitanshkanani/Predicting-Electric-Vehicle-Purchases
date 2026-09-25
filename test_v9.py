"""Local verification for ev_s6e9_v9.py. LightGBM/XGBoost aren't installed here,
so members are stubbed; every encoding/statistical path runs for real."""
import os, shutil, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev_s6e9_v9 as V
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

ok = True
def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

rs = np.random.RandomState(0)

# ---------- 1. WindowTE: brute-force agreement ----------
n = 4000
vals = rs.uniform(30000, 180000, n)
y = (rs.rand(n) < 0.2).astype(int)
R = 500.0
w = V.WindowTE(R, prior=20.0).fit(vals, y)
got = w.transform(vals)                      # no self-exclusion -> includes self
mine = np.empty(n)
for i in range(n):
    m = np.abs(vals - vals[i]) <= R
    s, c = y[m].sum(), m.sum()
    mine[i] = (s + 20.0 * y.mean()) / (c + 20.0)
check("WindowTE matches brute force (no self-excl)", np.allclose(got, mine, atol=1e-5),
      f"max|d|={np.abs(got-mine).max():.2e}")

# ---------- 2. WindowTE: self-exclusion is exact and leak-free ----------
gs = w.transform(vals, self_y=y)
mine2 = np.empty(n)
for i in range(n):
    m = np.abs(vals - vals[i]) <= R
    m[i] = False
    s, c = y[m].sum(), m.sum()
    mine2[i] = (s + 20.0 * y.mean()) / (c + 20.0)
check("WindowTE self-excluded matches brute force", np.allclose(gs, mine2, atol=1e-5))

y_flip = y.copy()
k = 1234
y_flip[k] = 1 - y_flip[k]
w2 = V.WindowTE(R, prior=20.0).fit(vals, y_flip)
gs2 = w2.transform(vals, self_y=y_flip)
gs2_nox = w2.transform(vals)                      # without self-exclusion
leak_excluded = abs(gs2[k] - gs[k])
leak_included = abs(gs2_nox[k] - gs[k])
# residual with exclusion is only the shrinkage-prior (global mean) shift:
# prior*(1/n)/(c+prior). Without exclusion it is ~1/(c+prior) -- ~200x bigger.
check("row k's encoding excludes its own label", leak_excluded < 0.05 * leak_included,
      f"|d|excl={leak_excluded:.2e} vs |d|incl={leak_included:.2e} (ratio {leak_excluded/max(leak_included,1e-12):.3f})")
check("  (contrast) other rows DO react", np.abs(np.delete(gs2 - gs, k)).max() > 1e-6)

# ---------- 3. te_map parity with sklearn (fixed smoothing) ----------
tr = pd.read_csv(os.path.join(HERE, "data", "train.csv")).sample(
    30000, random_state=1).reset_index(drop=True)
ytr = V.encode_target(tr[V.TARGET])
codes = np.asarray(pd.Categorical(tr.Gender.astype(str)).codes, dtype=np.int32)
nc = int(codes.max()) + 1
Xdf = pd.DataFrame({"c": tr.Gender.astype(str)})
for smooth in (10.0, 30.0, 100.0):
    te = TargetEncoder(smooth=smooth, cv=5)
    te.fit(Xdf, ytr)
    ref = te.transform(Xdf)[:, 0].astype(np.float64)
    mine = V.te_map(codes, ytr, nc, smooth, float(ytr.mean()), 0.0).astype(np.float64)[codes]
    check(f"te_map parity smooth={smooth}", np.abs(ref - mine).max() < 1e-6,
          f"max|d|={np.abs(ref-mine).max():.2e}")
cf = V.te_crossfit(codes, ytr, nc, 30.0, float(ytr.mean()), 42).astype(np.float64)
ref = TargetEncoder(smooth=30.0, cv=5, shuffle=True, random_state=42).fit_transform(Xdf, ytr)[:, 0]
check("te_crossfit matches sklearn exactly", np.abs(ref.astype(np.float64) - cf).max() < 1e-5,
      f"max|d|={np.abs(ref.astype(float)-cf).max():.2e}")

# ---------- 4. paired_gate behaviour ----------
n2 = 60000
yy = rs.randint(0, 2, n2)
s1 = yy * 0.5 + rs.randn(n2) * 0.5
fid = np.arange(n2) % 10
d, se, z, nf, tot = V.paired_gate(yy, s1, s1, 10, fid)
check("gate: identical scores -> delta 0, z 0", abs(d) < 1e-12 and abs(z) < 1e-9,
      f"delta={d:.2e} se={se:.2e}")
s2 = yy * 0.5 + rs.randn(n2) * 0.5 + 0.15
d2, se2, z2, nf2, tot2 = V.paired_gate(yy, s1, s2, 10, fid)
s3 = yy * 0.5 + rs.randn(n2) * 0.5 * 1.0
d3, se3, z3, _, _ = V.paired_gate(yy, s1, s3, 10, fid)
check("gate: correlated pair has small SE", 0 < se2 < 0.01 and 0 < se3 < 0.01,
      f"se2={se2:.5f} se3={se3:.5f}")
check("gate: folds tested == 10", tot == 10, f"tot={tot}")

# ---------- 5. fold_matrix at real scale ----------
te_df = pd.read_csv(os.path.join(HERE, "data", "test.csv")).head(4000)
orig = V.load_original(tr, [])
F = V.build_features(tr, te_df, orig)
tr_idx = np.arange(24000); va_idx = np.arange(24000, 30000)
Xa, Xb, Xc = V.fold_matrix(F, ytr, tr_idx, va_idx, 42)
check("design finite, widths aligned",
      np.isfinite(Xa).all() and np.isfinite(Xb).all() and np.isfinite(Xc).all()
      and Xa.shape[1] == Xb.shape[1] == Xc.shape[1], f"{Xa.shape} {Xc.shape}")
check("anchors present", any('org_mean' in c for c in F["raw_num"]),
      f"{sum('org_mean' in c for c in F['raw_num'])} anchor cols")
check("digit cols NOT target-encoded (raw only)",
      all(not c.startswith(('Age_digit','Annual_Income_USD_digit')) for c in F["te_src"]),
      f"te_src={len(F['te_src'])}")
n_win = len(V.INCOME_RADII) + len(V.COMMUTE_RADII)
check("width == raw + 3*TE + windows",
      Xa.shape[1] == len(F["raw_num"]) + 3*len(F["te_src"]) + n_win,
      f"{len(F['raw_num'])}+{3*len(F['te_src'])}+{n_win}={Xa.shape[1]}")

# held-out labels must not move the val/test blocks
y_flip = ytr.copy(); y_flip[va_idx] = 1 - y_flip[va_idx]
_, Xb2, Xc2 = V.fold_matrix(F, y_flip, tr_idx, va_idx, 42)
check("val block invariant to held-out labels", np.array_equal(Xb, Xb2))
check("test block invariant to held-out labels", np.array_equal(Xc, Xc2))

# ---------- 6. bands ----------
m = te_df.copy()
inc = np.rint(pd.to_numeric(m.Annual_Income_USD).to_numpy())
com = pd.to_numeric(m.Daily_Commute_km).to_numpy()
pr = rs.rand(len(m))
out = V.apply_bands(pr, inc, com)
buy, nobuy = V.band_masks(inc, com)
check("bands: no ties", len(np.unique(out)) == len(out))
check("bands: in [0,1]", out.min() >= 0 and out.max() <= 1)

# ---------- 7. end-to-end with stubs ----------
work = "/tmp/v9run"
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
big_tr = pd.read_csv(os.path.join(HERE, "data", "train.csv")).sample(
    6000, random_state=7).reset_index(drop=True)
big_te = pd.read_csv(os.path.join(HERE, "data", "test.csv")).head(1500).reset_index(drop=True)
big_tr.to_csv(os.path.join(work, "train.csv"), index=False)
big_te.to_csv(os.path.join(work, "test.csv"), index=False)
big_te[["id"]].assign(**{V.TARGET: 0.5}).to_csv(os.path.join(work, "sample_submission.csv"), index=False)
V.N_SPLITS = 3
V.TE_SEEDS = (42, 7)
V.find_data_dir = lambda: (os.path.join(work, "train.csv"), os.path.join(work, "test.csv"),
                           os.path.join(work, "sample_submission.csv"))
V.KAGGLE_INPUT_DIR = os.path.join(work, "nope")
def stub_lgb(Xa, ya, Xb, yb, Xc):
    return (np.clip(yb*0.4 + rs.rand(len(yb))*0.2, 0, 1).astype("float32"),
            (rs.rand(Xc.shape[0])*0.5+0.25).astype("float32"))
def stub_xgb(Xa, ya, Xb, yb, Xc, use_gpu=False):
    return (np.clip(yb*0.4 + rs.rand(len(yb))*0.2, 0, 1).astype("float32"),
            (rs.rand(Xc.shape[0])*0.5+0.25).astype("float32"))
def bad(Xa, ya, Xb, yb, Xc):
    raise RuntimeError("simulated LR failure")
V.run_lightgbm, V.run_xgboost, V.run_lr = stub_lgb, stub_xgb, bad
V.ensure_import = lambda p: p in ("lightgbm", "xgboost")
V.LR_USE = True
cwd = os.getcwd(); os.chdir(work)
try:
    V.main()
    files = sorted(f for f in os.listdir(work) if f.startswith("submission_v9"))
    check("v9 wrote both submissions", len(files) == 2, str(files))
    for f in files:
        d = pd.read_csv(os.path.join(work, f))
        check(f"  {f}", list(d.columns) == ["id", V.TARGET] and len(d) == 1500
              and d[V.TARGET].notna().all() and d["id"].nunique() == 1500
              and 0 <= d[V.TARGET].min() and d[V.TARGET].max() <= 1,
              f"rows={len(d)} unique={d[V.TARGET].nunique()}")
    ck = [x for x in os.listdir(work) if x.startswith("v9_ckpt")]
    n_ckpt = len(os.listdir(os.path.join(work, ck[0]))) if ck else 0
    check("per-seed per-fold checkpoints", n_ckpt == 2*3*2*2, f"{n_ckpt} files (2 seeds x 3 folds x 2 members x 2 arrays)")
    check("dropped member did not kill run", True)
except Exception:
    import traceback; traceback.print_exc()
    check("end-to-end main()", False)
finally:
    os.chdir(cwd)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
