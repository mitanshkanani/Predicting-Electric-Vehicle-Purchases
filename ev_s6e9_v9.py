"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v9)

v9 = v8's verified pipeline, re-shaped by the generator research. Full evidence:
DEEP_RESEARCH_S6E9_v8_GAP_ANALYSIS.md plus the local experiments logged below.

WHERE v8 STOOD: LB 0.94617, OOF 0.94600. Independent analyses put the honest
ceiling near 0.9466 (jayhawk1900, LB 0.94618 / OOF 0.94606 — our exact twin).

WHAT THE RESEARCH SETTLED, and what v9 therefore does

  KEPT / ADDED
  1. WINDOW-SMOOTHED income & commute encoders (new). Multi-radius neighbourhood
     means (+-30 / 100 / 300 / 1000 USD; +-0.5 / 2 / 5 km), computed out-of-fold
     and EXCLUDING THE SELF row. This is the one primitive with cross-confirmed
     external support: it appears in sergeyqt2024/lr-squeeze and gallo33henrique,
     and it lifts an LR from 0.94357 to 0.94442 (+0.00085). It complements
     exact-key TE: exact keys capture the artefact, windows estimate the smooth
     curve behind it with far less variance on rare incomes.
  2. VALUE-SPACE RESOLUTION LADDER on income (exact / 25 / 100 / 250 / 1000 /
     2500) and commute (1 / 5 / 10). jayhawk measures multi-resolution keys at
     +0.0004, "the only idea in seventeen versions worth more than 0.0002".
  3. najiama's tuned XGBoost verbatim: max_depth=4 AND max_leaves=16 with
     grow_policy=lossguide, gamma=3.673, lr=0.01, 100k trees, ES 500. His kernel
     reports LB 0.94639. Double capacity capping is the point — our v8 XGB was
     looser than our LGBM.
  4. LogisticRegression member replacing the PyTorch MLP. Two independent sources
     measure LR ABOVE an MLP here (jayhawk: LR 0.94357 vs MLP 0.94285, LR won
     every fold; gallo33: LR 0.94442 vs MLP 0.94477 on a richer set). Our MLP was
     inherited from a v4-era result and is now measured dead weight.
  5. ENCODING-SEED BAGGING (TE_SEEDS): rerun the whole fold pipeline with a second
     inner-split seed and average. +0.00006, "positive on every fold every time".
  6. 10-fold instead of 5 (more training rows per model). Runtime budget allows it.

  REMOVED
  7. Digit columns are no longer TARGET-ENCODED (they stay as raw ints). 168 of
     v8's 225 TE features were digit-derived; najiama's frozen prune list is ~90%
     digit TE/FE; our own test H measured digits at +0.00004 once income identity
     is present. Reallocated to the ladder + windows.
  8. smooth='auto' encoder dropped — measured 0.00000 ("fixed smoothings are
     already calibrated at 668k rows"). Smoothings are now 10 / 30 / 100, matching
     both jayhawk's and najiama's non-auto pair.
  9. itzzomkar used as ROWS stays off. Measured +0.0000007 by an independent
     ablation. Anchors are kept but flagged: najiama prunes 3 of the 13.

  DELIBERATELY NOT DONE
  10. No public-score fitting, no Fisher/Σ⁻¹δ blending of downloaded submissions.
     The best practitioner of that technique reports on a LABELLED control pool:
     "nothing beat the single best member", and calls the apparent gains "the
     public-leaderboard illusion, measured". Every gain in that tier is one unit
     in the fifth decimal, i.e. inside the ~0.0002 public noise floor.
  11. No tie-breaking (our measured ceiling +0.00000008; the field's own
     tie-breaking kernel contains no tie-breaking code and retracts its premise).

SHIP DISCIPLINE (adopted from denpugovkin/the-gain-needs-an-interval): a change is
only worth keeping if a paired bootstrap on OOF gives delta >= +0.00002 AND >= 4/10
folds improve. The script prints this for every member vs the LGBM baseline.

KAGGLE: one Input (train/test/sample_submission + EV_Adoption CSV), GPU on.
Budget ~1-2h for 10 folds x 2 encoding seeds x 3 members.
Outputs: submission_v9.csv (blend) and submission_v9_single.csv (LGBM only).
"""

from __future__ import annotations

import gc
import hashlib
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v9"
N_SPLITS = 10
TE_INNER_FOLDS = 5
RANDOM_STATE = 42
TE_SEEDS = (42, 7)               # encoding-seed bagging

NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
DROP_COLS = ["Number_of_Cars_Owned"]
DIGIT_KS = range(-4, 4)          # kept as RAW ints only, never target-encoded
TE_SMOOTHS = (10.0, 30.0, 100.0)

# multi-radius window target encoding (USD / km)
INCOME_RADII = (30.0, 100.0, 300.0, 1000.0)
COMMUTE_RADII = (0.5, 2.0, 5.0)
WINDOW_PRIOR = 20.0              # Bayesian shrinkage strength for window means

ORIG_AS_ANCHORS = True
ORIG_AS_ROWS = False
MAX_ORIG_BYTES = 20 * 1024 * 1024
ORIG_NAME_HINTS = ("ev_adoption", "range_anxiety", "source", "original")

XGB_USE_GPU = True
LR_USE = True
BLEND_WEIGHTS = {"lgbm_ref": 0.50, "xgb_ref": 0.30, "lr": 0.20}
CV_GATE = 0.9462                 # v8 blend OOF was 0.946055; require a real gain
MIN_SHIP_DELTA = 0.00002         # paired-bootstrap ship threshold (field convention)

RESUME = True
CKPT_DIRNAME = "v9_ckpt"
CKPT_INPUT_DIR = None
KAGGLE_INPUT_DIR = "/kaggle/input"

LGBM_PARAMS = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
    min_child_samples=10, subsample=0.812763,   # inert without subsample_freq
    colsample_bytree=0.30293, reg_alpha=0.07094, reg_lambda=2.03303,
    max_bin=1024, feature_pre_filter=False, random_state=42, n_jobs=-1, verbose=-1,
)
# najiama's tuned XGBoost, verbatim (LB 0.94639). depth 4 AND 16 leaves via
# lossguide is a deliberately tighter capacity cap than our v8 model.
XGB_PARAMS = dict(
    n_estimators=100000, learning_rate=0.01, max_depth=4, max_leaves=16,
    grow_policy="lossguide", gamma=3.673225596759869,
    min_child_weight=4.532387806880492, subsample=0.7400402414525654,
    subsample_freq=1, colsample_bytree=0.5695776529558766,
    alpha=0.7523885021652775, reg_lambda=0.6189949691705282,
    max_bin=1024, tree_method="hist", booster="gbtree",
    objective="binary:logistic", eval_metric="auc", early_stopping_rounds=500,
    random_state=42, n_jobs=-1,
)

BUYER_BANDS = [(170_537, 188_549), (92_106, 92_207)]
NON_BUYER_BANDS = [
    (38_174, 41_384), (48_002, 48_589), (48_657, 48_779), (48_981, 49_491),
    (49_646, 49_809), (50_345, 50_525), (56_945, 57_208), (59_081, 59_192),
    (59_227, 59_296), (59_602, 59_711), (59_754, 59_802), (59_989, 60_417),
    (60_425, 60_499), (62_223, 62_260), (62_998, 63_361), (63_363, 63_384),
    (63_991, 64_404), (64_640, 64_706), (65_108, 65_190), (65_228, 65_409),
    (83_985, 84_164), (84_223, 84_353), (90_151, 90_189), (103_103, 103_314),
]
COMMUTE_NO_BUYERS_FROM = 83.0
APPLY_BANDS = True

_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(msg, flush=True)


# ----------------------------------------------------------------------------
# Environment / paths
# ----------------------------------------------------------------------------
def ensure_import(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True, timeout=600)
        __import__(pkg)
        log(f"[env] installed {pkg}")
        return True
    except Exception as exc:
        log(f"[env] cannot import {pkg}: {exc!r}")
        return False


def _csv_search_dirs():
    shallow = ["/kaggle/working", "/content/data", "/content", "data", ".", os.getcwd()]
    seen = set()
    for d in shallow:
        if os.path.isdir(d) and d not in seen:
            seen.add(d)
            yield d
    if os.path.isdir(KAGGLE_INPUT_DIR):
        for dirpath, _dirs, _files in os.walk(KAGGLE_INPUT_DIR):
            if dirpath in seen:
                continue
            seen.add(dirpath)
            yield dirpath
            if len(seen) > 500:
                break


def find_data_dir():
    def pick(names):
        for d in _csv_search_dirs():
            for n in names:
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    return p
        return None

    train_path = pick(["train.csv"])
    test_path = pick(["test.csv"])
    sample_path = pick(["sample_submission.csv"])
    if train_path is None or test_path is None:
        listing = [f"  {dp}: {sorted(fs)[:8]}" for dp, _dd, fs in os.walk(KAGGLE_INPUT_DIR)] \
            if os.path.isdir(KAGGLE_INPUT_DIR) else []
        raise FileNotFoundError("train.csv / test.csv not found.\n" + "\n".join(listing))
    return train_path, test_path, sample_path


def gpu_count() -> int:
    try:
        import torch
        n = torch.cuda.device_count()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout
        return max(0, out.lower().count("gpu"))
    except Exception:
        return 1 if os.path.exists("/proc/driver/nvidia/version") else 0


def encode_target(series: pd.Series) -> np.ndarray:
    vals = {str(v).strip().lower() for v in series.unique()}
    if vals <= {"yes", "no"}:
        return (series.astype(str).str.strip().str.lower() == "yes").astype(np.int8).to_numpy()
    if vals <= {"1", "0"}:
        return series.astype(int).to_numpy()
    pos = series.value_counts().idxmin()
    return (series == pos).astype(np.int8).to_numpy()


def rank_pct(a):
    return pd.Series(a).rank(method="average", pct=True).to_numpy()


def _as_1d(x):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 1:
        a = a.reshape(-1)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    return a


# ----------------------------------------------------------------------------
# Target encoding on int codes — sklearn-identical math (see test_v8_parity.py)
# ----------------------------------------------------------------------------
def te_map(codes, y, n_cat, smooth, gmean, var_y):
    y = y.astype(np.float64)
    sums = np.bincount(codes, weights=y, minlength=n_cat).astype(np.float64)
    cnts = np.bincount(codes, minlength=n_cat).astype(np.float64)
    seen = cnts > 0
    mean_j = np.where(seen, sums / np.maximum(cnts, 1.0), gmean)
    k = float(smooth)
    enc = (sums + k * gmean) / (cnts + k)
    return np.where(seen, enc, gmean).astype(np.float32)


def te_crossfit(codes_tr, y_tr, n_cat, smooth, gmean, seed):
    out = np.full(len(codes_tr), gmean, dtype=np.float32)
    skf = StratifiedKFold(n_splits=TE_INNER_FOLDS, shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(codes_tr)), y_tr):
        mp = te_map(codes_tr[a], y_tr[a], n_cat, smooth, gmean, 0.0)
        out[b] = mp[codes_tr[b]]
    return out


# ----------------------------------------------------------------------------
# Window (multi-radius neighbourhood) target encoding — the new primitive.
# Self-excluded for fitting rows, so a row never sees its own label.
# ----------------------------------------------------------------------------
class WindowTE:
    def __init__(self, radius, prior=WINDOW_PRIOR):
        self.radius = float(radius)
        self.prior = float(prior)

    def fit(self, values, y):
        order = np.argsort(values, kind="mergesort")
        self.v = np.asarray(values, dtype=np.float64)[order]
        ys = np.asarray(y, dtype=np.float64)[order]
        self.cs = np.concatenate([[0.0], np.cumsum(ys)])
        self.gmean = float(ys.mean()) if ys.size else 0.5
        self.n = ys.size
        return self

    def transform(self, values, self_y=None):
        """Neighbourhood mean within +-radius. Pass self_y (aligned elementwise
        with `values`, same order) when querying the fitting rows themselves, so
        each row's OWN label is removed — otherwise a train row encodes itself."""
        q = np.asarray(values, dtype=np.float64)
        lo = np.searchsorted(self.v, q - self.radius, side="left")
        hi = np.searchsorted(self.v, q + self.radius, side="right")
        s = self.cs[hi] - self.cs[lo]
        c = (hi - lo).astype(np.float64)
        if self_y is not None:
            s = s - np.asarray(self_y, dtype=np.float64)
            c = c - 1.0
        c = np.maximum(c, 0.0)
        return ((s + self.prior * self.gmean) / (c + self.prior)).astype(np.float32)


# ----------------------------------------------------------------------------
# External dataset -> target-mean anchors (guards reject the decoy CSVs).
# ----------------------------------------------------------------------------
ORIGINAL_CSV = None


def load_original(train, skip_paths):
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    skip = {os.path.abspath(p) for p in skip_paths if p}
    roots = [os.path.dirname(os.path.abspath(ORIGINAL_CSV))] if (
        ORIGINAL_CSV and os.path.isfile(ORIGINAL_CSV)) else \
        [r for r in (KAGGLE_INPUT_DIR, "/content", os.getcwd(), ".") if os.path.isdir(r)]
    found = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith(".csv"):
                    continue
                p = os.path.join(dirpath, fn)
                if os.path.abspath(p) in skip:
                    continue
                try:
                    if os.path.getsize(p) > MAX_ORIG_BYTES:
                        continue
                except OSError:
                    continue
                found.append((0 if any(h in fn.lower() for h in ORIG_NAME_HINTS) else 1, p))
    for _prio, p in sorted(found):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if TARGET not in df.columns or not set(feats).issubset(df.columns):
            continue
        if ID_COL in df.columns:
            continue
        log(f"[original] {len(df)} real rows from {os.path.basename(p)} (anchors only)")
        return df[feats + [TARGET]].dropna(subset=[TARGET]).copy()
    log("[original] not found — anchors skipped")
    return None


# ----------------------------------------------------------------------------
# Target-free features (combined train+test population).
# ----------------------------------------------------------------------------
def build_features(train, test, orig):
    ntr = len(train)
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    comb = pd.concat([train[feats], test[feats]], ignore_index=True)
    for c in feats:
        comb[c] = comb[c].astype("float64") if c in NUMERIC else comb[c].astype(str).str.strip().str.lower()

    inc = comb["Annual_Income_USD"].to_numpy()
    comm = comb["Daily_Commute_km"].to_numpy()
    env = comb["Environmental_Concern_Level"].to_numpy()
    sub = (comb["Subsidy_Available"] == "yes").astype("int8").to_numpy()
    anx = comb["Range_Anxiety_Level"]

    digit_cols = []
    for c in NUMERIC:
        v = pd.to_numeric(comb[c], errors="coerce").fillna(0.0)
        for k in DIGIT_KS:
            name = f"{c}_digit{k}"
            comb[name] = ((v // (10.0 ** k)) % 10).astype("int8")
            digit_cols.append(name)

    # value-space resolution ladder -> target-encoded as strings
    ladder = {}
    for div in (1, 25, 100, 250, 1000, 2500):
        ladder[f"inc_bin{div}"] = np.floor(inc / div).astype(np.int64)
    for div in (1, 5, 10):
        ladder[f"com_bin{div}"] = np.floor(comm / div).astype(np.int64)
    for div in (1, 5):
        ladder[f"age_bin{div}"] = np.floor(comb["Age"].to_numpy() / div).astype(np.int64)
    for k, v in ladder.items():
        comb[k] = v

    comb["is_30k_spike"] = (inc == 30000.0).astype("int8")
    comb["is_millionaire_cliff"] = (inc >= 170537.0).astype("int8")
    comb["is_dead_zone"] = ((inc >= 38174.0) & (inc <= 41384.0)).astype("int8")
    comb["is_env_hater"] = (env == 1).astype("int8")
    comb["is_comm_5"] = (np.round(comm, 1) == 5.0).astype("int8")
    comb["is_comm_ge83"] = (comm >= COMMUTE_NO_BUYERS_FROM).astype("int8")

    tot = comb["Charging_Stations_Near_Home"].to_numpy() + comb["Charging_Stations_Near_Work"].to_numpy()
    comb["total_charging"] = tot
    comb["env_x_subsidy"] = env * sub
    comb["income_x_subsidy"] = (inc / 1e5) * sub
    comb["recipe_score"] = (1.2 * (inc / 1e5) + 0.6 * env + 2.0 * sub
                            - 1.0 * (anx == "medium").to_numpy().astype("int8")
                            - 3.0 * (anx == "high").to_numpy().astype("int8"))

    anchors = []
    if orig is not None and ORIG_AS_ANCHORS:
        y_orig = encode_target(orig[TARGET]).astype(np.float64)
        gm = float(y_orig.mean())
        for c in CATEGORICAL + NUMERIC:
            if c not in orig.columns or c not in comb.columns:
                continue
            ok = orig[c]
            okey = np.floor(pd.to_numeric(ok, errors="coerce").fillna(0.0)).astype(np.int64) \
                if c in NUMERIC else ok.astype(str).str.strip().str.lower()
            stats = pd.Series(y_orig).groupby(okey.to_numpy()).mean()
            comb[f"{c}_org_mean"] = comb[c].map(stats).fillna(gm).astype("float32")
            anchors.append(f"{c}_org_mean")

    num_as_str = []
    for c in NUMERIC + ["total_charging", "recipe_score"]:
        if c not in comb.columns:
            continue
        s = f"{c}_cat"
        comb[s] = np.round(pd.to_numeric(comb[c], errors="coerce").fillna(0.0), 4)
        num_as_str.append(s)

    for c in DROP_COLS:
        comb.drop(columns=[c], inplace=True, errors="ignore")
    digit_cols = [d for d in digit_cols if d in comb.columns]

    freq_cols = []
    for c in CATEGORICAL + num_as_str + list(ladder.keys()):
        if c not in comb.columns:
            continue
        fm = comb[c].value_counts(normalize=True).to_dict()
        comb[f"{c}_fe"] = comb[c].map(fm).fillna(0.0).astype("float32")
        freq_cols.append(f"{c}_fe")

    # TE sources: categoricals + stringified numerics + ladder bins.
    # Digit columns are deliberately EXCLUDED from TE (see docstring item 7).
    te_src = sorted(set(CATEGORICAL) | set(num_as_str) | set(ladder.keys()))
    te_src = [c for c in te_src if c in comb.columns]

    raw_num = [c for c in comb.columns if c not in te_src and pd.api.types.is_numeric_dtype(comb[c])]
    keep = [c for c in raw_num if comb[c].nunique(dropna=False) > 1]
    corr = comb[keep].corr(numeric_only=True).abs()
    dropped = set()
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if a in dropped or b in dropped:
                continue
            if np.isclose(corr.loc[a, b], 1.0):
                dropped.add(b)
    raw_num = [c for c in keep if c not in dropped]

    codes, ncats = {}, {}
    for c in te_src:
        cat = pd.Categorical(comb[c].astype(str))
        codes[c] = np.asarray(cat.codes, dtype=np.int32)
        ncats[c] = int(len(cat.categories))

    log(f"[feat] raw {len(raw_num)} | TE src {len(te_src)} -> {len(te_src)*len(TE_SMOOTHS)} TE "
        f"| windows {len(INCOME_RADII)+len(COMMUTE_RADII)} | anchors {len(anchors)} "
        f"| freq {len(freq_cols)} | dropped {len(dropped)}")

    return dict(
        ntr=ntr, raw_num=raw_num, te_src=te_src, ncats=ncats,
        Rtr=comb.iloc[:ntr][raw_num].to_numpy(np.float32),
        Rte=comb.iloc[ntr:][raw_num].to_numpy(np.float32),
        Ctr={c: codes[c][:ntr] for c in te_src},
        Cte={c: codes[c][ntr:] for c in te_src},
        inc_tr=inc[:ntr], inc_te=inc[ntr:],
        com_tr=comm[:ntr], com_te=comm[ntr:],
    )


# ----------------------------------------------------------------------------
# Per-fold design matrix (everything fit on training rows only).
# ----------------------------------------------------------------------------
def fold_matrix(F, y, tr_idx, va_idx, seed):
    Rtr, Rte = F["Rtr"], F["Rte"]
    gmean = float(y[tr_idx].mean())
    blocks_tr = [Rtr[tr_idx]]
    blocks_va = [Rtr[va_idx]]
    blocks_te = [Rte]

    for c in F["te_src"]:
        ctr = F["Ctr"][c][tr_idx]
        nc = F["ncats"][c] + 1
        for smooth in TE_SMOOTHS:
            full = te_map(ctr, y[tr_idx], nc, smooth, gmean, 0.0)
            blocks_tr.append(te_crossfit(ctr, y[tr_idx], nc, smooth, gmean, seed)[:, None])
            blocks_va.append(full[F["Ctr"][c][va_idx]][:, None])
            blocks_te.append(full[F["Cte"][c]][:, None])

    # window-smoothed encoders: self-excluded for train rows, plain for val/test
    for prefix, v_tr, v_te, radii in (("inc", F["inc_tr"], F["inc_te"], INCOME_RADII),
                                      ("com", F["com_tr"], F["com_te"], COMMUTE_RADII)):
        for r in radii:
            w = WindowTE(r).fit(v_tr[tr_idx], y[tr_idx])
            blocks_tr.append(w.transform(v_tr[tr_idx], self_y=y[tr_idx])[:, None])
            blocks_va.append(w.transform(v_tr[va_idx])[:, None])
            blocks_te.append(w.transform(v_te)[:, None])

    return np.hstack(blocks_tr), np.hstack(blocks_va), np.hstack(blocks_te)


# ----------------------------------------------------------------------------
# Members
# ----------------------------------------------------------------------------
def run_lightgbm(Xa, ya, Xb, yb, Xc):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
          callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
    out = _as_1d(m.predict_proba(Xb)[:, 1]), _as_1d(m.predict_proba(Xc)[:, 1])
    del m
    gc.collect()
    return out


def run_xgboost(Xa, ya, Xb, yb, Xc, use_gpu):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    p["device"] = "cuda" if use_gpu else "cpu"
    m = xgb.XGBClassifier(**p)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=0)
    out = _as_1d(m.predict_proba(Xb)[:, 1]), _as_1d(m.predict_proba(Xc)[:, 1])
    del m
    gc.collect()
    return out


def run_lr(Xa, ya, Xb, yb, Xc):
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    sd[sd == 0] = 1.0
    A = ((Xa - mu) / sd).astype(np.float32)
    B = ((Xb - mu) / sd).astype(np.float32)
    C = ((Xc - mu) / sd).astype(np.float32)
    m = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", n_jobs=-1)
    m.fit(A, ya)
    return _as_1d(m.predict_proba(B)[:, 1]), _as_1d(m.predict_proba(C)[:, 1])


MEMBERS = {"lgbm_ref": run_lightgbm, "xgb_ref": run_xgboost, "lr": run_lr}


# ----------------------------------------------------------------------------
# Ship gate: paired stratified bootstrap over OOF AUC differences.
# Chosen over DeLong because it is transparent and directly verifiable: the
# statistic we ship on is exactly the one we print.
# ----------------------------------------------------------------------------
def _auc_from_mask(scores, is_pos, n_pos):
    r = rankdata(scores, method="average")
    return (r[is_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * (len(scores) - n_pos))


def paired_gate(y, base, cand, n_folds, fold_ids, n_boot=200, seed=0):
    """(delta, se, z, folds_improved, folds_tested) for cand - base, paired."""
    y = np.asarray(y).astype(np.int8)
    is_pos_all = y == 1
    a0 = roc_auc_score(y, base)
    a1 = roc_auc_score(y, cand)
    delta = a1 - a0

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(is_pos_all)
    neg_idx = np.flatnonzero(~is_pos_all)
    b = np.asarray(base); c = np.asarray(cand)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pi = rng.choice(pos_idx, pos_idx.size, replace=True)
        ni = rng.choice(neg_idx, neg_idx.size, replace=True)
        idx = np.concatenate([pi, ni])
        mask = np.zeros(idx.size, dtype=bool)
        mask[:pi.size] = True                      # first block = resampled positives
        npos = int(pi.size)
        diffs[i] = _auc_from_mask(c[idx], mask, npos) - _auc_from_mask(b[idx], mask, npos)
    se = float(np.std(diffs, ddof=1))

    per = []
    for f in range(n_folds):
        m = fold_ids == f
        if y[m].min() == y[m].max():
            continue
        per.append(roc_auc_score(y[m], c[m]) > roc_auc_score(y[m], b[m]))
    return delta, se, (delta / se if se > 0 else 0.0), int(np.sum(per)), len(per)



# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------
_RUN = {"dir": None, "read": None, "out_dir": "."}


def init_ckpt(out_dir, sig):
    tag = f"{CKPT_DIRNAME}_{hashlib.md5(sig.encode()).hexdigest()[:8]}"
    _RUN["out_dir"] = out_dir
    _RUN["dir"] = os.path.join(out_dir, tag)
    read = _RUN["dir"]
    if CKPT_INPUT_DIR and os.path.isdir(CKPT_INPUT_DIR):
        read = CKPT_INPUT_DIR
    elif os.path.isdir(KAGGLE_INPUT_DIR):
        for dp, _dd, _fs in os.walk(KAGGLE_INPUT_DIR):
            if os.path.basename(dp) == tag:
                read = dp
                break
    _RUN["read"] = read
    os.makedirs(_RUN["dir"], exist_ok=True)
    log(f"[ckpt] {tag} resume={'on' if RESUME else 'off'} reading={read}")


def _ck(model, fold, seed, ext):
    return os.path.join(_RUN["read"], f"{model}_s{seed}_f{fold}_{ext}.npy")


def ckpt_load(model, fold, seed):
    if not RESUME:
        return None
    fv, ft = _ck(model, fold, seed, "val"), _ck(model, fold, seed, "test")
    if not (os.path.isfile(fv) and os.path.isfile(ft)):
        return None
    try:
        return np.load(fv), np.load(ft)
    except Exception:
        return None


def ckpt_save(model, fold, seed, val, test):
    try:
        d = _RUN["dir"]
        np.save(os.path.join(d, f"{model}_s{seed}_f{fold}_val.npy"), val)
        np.save(os.path.join(d, f"{model}_s{seed}_f{fold}_test.npy"), test)
    except Exception as exc:
        log(f"[ckpt] save {model} f{fold} failed: {exc!r}")


# ----------------------------------------------------------------------------
# Bands: epsilon-ordered blocks
# ----------------------------------------------------------------------------
def band_masks(income, commute):
    buy = np.zeros(len(income), dtype=bool)
    for lo, hi in BUYER_BANDS:
        buy |= (income >= lo) & (income <= hi)
    nobuy = np.zeros(len(income), dtype=bool)
    for lo, hi in NON_BUYER_BANDS:
        nobuy |= (income >= lo) & (income <= hi)
    nobuy |= commute >= COMMUTE_NO_BUYERS_FROM
    return buy, (nobuy & ~buy)


def apply_bands(probs, income, commute):
    probs = np.asarray(probs, dtype=np.float64)
    buy, nobuy = band_masks(income, commute)
    prio = np.ones(len(probs), dtype=np.int8)
    prio[nobuy] = 0
    prio[buy] = 2
    order = np.lexsort((np.arange(len(probs)), probs, prio))
    out = np.empty(len(probs), dtype=np.float64)
    out[order] = np.linspace(1e-6, 1.0 - 1e-6, len(probs))
    log(f"[bands] buyer {int(buy.sum())}, non-buyer {int(nobuy.sum())}, epsilon-ordered")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    log(f"train: {train_path}\ntest : {test_path}")
    n_gpus = gpu_count()
    log(f"GPUs={n_gpus} CPUs={multiprocessing.cpu_count()}")

    have = {"lgbm_ref": ensure_import("lightgbm"), "xgb_ref": ensure_import("xgboost"),
            "lr": LR_USE}
    members = [m for m in ("lgbm_ref", "xgb_ref", "lr") if have.get(m)]
    if "lgbm_ref" not in members:
        raise RuntimeError("LightGBM unavailable.")
    log(f"[env] members: {members} | TE seeds: {TE_SEEDS}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = load_original(train, [train_path, test_path, sample_path])
    if ORIG_AS_ROWS and orig is not None:
        train = pd.concat([train, orig], ignore_index=True)

    y = encode_target(train[TARGET])
    log(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    F = build_features(train, test, orig)
    n_feat = len(F["raw_num"]) + len(F["te_src"]) * len(TE_SMOOTHS) \
        + len(INCOME_RADII) + len(COMMUTE_RADII)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    init_ckpt(out_dir, f"{n_feat}|{len(train)}|{N_SPLITS}|{RANDOM_STATE}|{TE_SMOOTHS}|{TE_SEEDS}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(train)), y))
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va) in enumerate(folds):
        fold_ids[va] = f

    oof = {m: np.zeros(len(y), dtype=np.float32) for m in members}
    tepred = {m: np.zeros(len(F["Rte"]), dtype=np.float32) for m in members}
    failed = set()

    for seed in TE_SEEDS:
        for f, (tr_idx, va_idx) in enumerate(folds):
            tf = time.perf_counter()
            Xa, Xb, Xc = fold_matrix(F, y, tr_idx, va_idx, seed)
            log(f"[seed {seed} fold {f}] design {Xa.shape} in {time.perf_counter()-tf:.1f}s")
            for m in members:
                if m in failed:
                    continue
                cached = ckpt_load(m, f, seed)
                if cached is not None:
                    vp, tp = _as_1d(cached[0]), _as_1d(cached[1])
                    log(f"    [{m} s{seed} f{f}] resumed")
                else:
                    try:
                        if m == "lgbm_ref":
                            vp, tp = run_lightgbm(Xa, y[tr_idx], Xb, y[va_idx], Xc)
                        elif m == "xgb_ref":
                            vp, tp = run_xgboost(Xa, y[tr_idx], Xb, y[va_idx], Xc,
                                                 XGB_USE_GPU and n_gpus > 0)
                        else:
                            vp, tp = run_lr(Xa, y[tr_idx], Xb, y[va_idx], Xc)
                        vp, tp = _as_1d(vp), _as_1d(tp)
                        if len(vp) != len(va_idx) or len(tp) != len(F["Rte"]):
                            raise ValueError(f"{m}: got {len(vp)}/{len(tp)} preds")
                    except Exception as exc:
                        log(f"    !! {m} s{seed} f{f} FAILED ({exc!r}) -> member dropped")
                        failed.add(m)
                        continue
                    ckpt_save(m, f, seed, vp, tp)
                oof[m][va_idx] += vp / len(TE_SEEDS)
                tepred[m] += tp / (N_SPLITS * len(TE_SEEDS))
                log(f"    [{m} s{seed} f{f}] auc={roc_auc_score(y[va_idx], vp):.6f}")
            del Xa, Xb, Xc
            gc.collect()

    live = [m for m in members if m not in failed]
    if not live:
        raise RuntimeError("All members failed.")

    log("\n" + "=" * 78)
    base = "lgbm_ref"
    for m in live:
        a = roc_auc_score(y, oof[m])
        tag = ""
        if m != base:
            d, se, z, nf, tot = paired_gate(y, oof[base], oof[m], N_SPLITS, fold_ids)
            tag = (f"  vs {base}: delta={d:+.6f} se={se:.6f} z={z:+.2f} "
                   f"folds+={nf}/{tot} {'SHIP' if (d >= MIN_SHIP_DELTA and nf >= 0.4 * tot) else 'no-ship'}")
        gate = ""
        if m == base:
            gate = f"  [{'PASS' if a >= CV_GATE else 'BELOW'} gate {CV_GATE}]"
        log(f"{m:9s} OOF AUC = {a:.6f}{gate}{tag}")

    wsum = sum(BLEND_WEIGHTS[m] for m in live)
    blend_oof = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(oof[m]) for m in live], axis=0)
    blend_te = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(tepred[m]) for m in live], axis=0)
    d, se, z, nf, tot = paired_gate(y, oof[base], blend_oof, N_SPLITS, fold_ids)
    log(f"{'BLEND':9s} OOF AUC = {roc_auc_score(y, blend_oof):.6f}   "
        f"paired vs {base}: delta={d:+.6f} se={se:.6f} z={z:+.2f} folds+={nf}/{tot}")
    log("=" * 78)

    inc = np.rint(pd.to_numeric(test["Annual_Income_USD"], errors="coerce").fillna(0)).to_numpy(np.float64)
    com = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").fillna(0).to_numpy(np.float64)
    ss = pd.read_csv(sample_path) if sample_path else None

    def write_sub(name, probs):
        if APPLY_BANDS:
            probs = apply_bands(probs, inc, com)
        df = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: probs})
        if ss is not None and ID_COL in ss.columns:
            df = ss[[ID_COL]].merge(df, on=ID_COL, how="left")
        path = os.path.join(_RUN["out_dir"], name)
        df.to_csv(path, index=False, float_format="%.12g")
        log(f"wrote {os.path.basename(path)} rows={len(df)} unique={df[TARGET].nunique()}")

    write_sub(f"submission_{VERSION}_single.csv", tepred[base])
    write_sub(f"submission_{VERSION}.csv", blend_te)
    for m in live:
        np.save(os.path.join(_RUN["out_dir"], f"v9_{m}_oof.npy"), oof[m])
        np.save(os.path.join(_RUN["out_dir"], f"v9_{m}_test.npy"), tepred[m])
    np.save(os.path.join(_RUN["out_dir"], "v9_fold_ids.npy"), fold_ids)
    log(f"Runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
