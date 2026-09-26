"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v10)

THE BAR IS v8: LB 0.94617, blend OOF 0.946055. v9 fell to 0.94371. v10 is NOT a
third rewrite — it is v8's verified pipeline with four changes, each of which has a
first-party measurement behind it, plus the measurement fix that made v9's own
decision table worthless.

WHY v9 LOST (read this before trusting any number below)
  v9 wrote each fold's RAW LightGBM probabilities into one concatenated OOF vector.
  AUC then compared ranks ACROSS folds, and every fold sits at its own calibration
  level. The log proves it: lgbm's ten per-fold AUCs average 0.939921 but its pooled
  OOF reads 0.907175 — a gap of -0.032746. Its LR member, which standardizes by each
  fold's own mean/std, shows a gap of +0.000001. A simulation with a per-fold logit
  offset of sd=0.65 reproduces -0.032823, and converting each fold's predictions to
  within-fold percentile ranks restores 0.9396. So v9's CV_GATE verdict, its paired
  bootstrap z-scores and its "SHIP" flags were all computed on a corrupted baseline,
  and the run shipped a blend whose OOF (0.9334) was 0.0103 BELOW its own leaderboard
  score — while v8's OOF sat 0.0001 from its LB. Aadit's LightGBM on this same data
  shows a gap of -0.00002, because its pipeline already stores foldwise ranks.

WHAT v10 CHANGES vs v8

  1. FOLD-RANK OOF (the fix, and a hard invariant). Every fold's validation
     predictions are percentile-ranked WITHIN that fold before entering the OOF
     vector, and each fold's test predictions are percentile-ranked before being
     averaged. OOF and test are then built the same way, which is what makes the
     CV->LB offset small. The script prints mean-per-fold vs pooled for both the raw
     and the fold-rank OOF of every member and flags the run MEASUREMENT UNHEALTHY
     (surfaced on the SUBMIT line) if the fold-rank gap exceeds 0.0010, which is
     ~30x smaller than the gap that killed v9. Raw-probability OOF is still printed,
     for v8 comparability only.

  2. TE SMOOTHING LADDER (2, 5, 25) replaces v8's ('auto', 10, 100). Three sweeps in
     Aadit_try/artifacts/experiments/ run monotonically downward, 5/5 folds at every
     step: m=100->25 +0.000563, 25->5 +0.000494, 5->2 +0.000109, and m=1 ~ m=2. v7's
     own leaderboard history agrees (2/10 -> LB 0.94540, 10/50 -> 0.94503, TE removed
     -> 0.94452). v9 moved the ladder UP to (10,30,100) and lost ~0.006 of per-fold
     tree AUC. Optimum is at the low end.

  3. FINE-INCOME TE BINS at $50 and $250 added to the target-encoded and
     frequency-encoded key sets. This is the best single measured additive gain in the
     whole repo: +0.00015486 (5/5 folds) on XGBoost and +0.00017165 (5/5 folds) on
     LightGBM, and it is orthogonal to v8's existing 1/100/1000 income bins.

  4. LOGISTIC REGRESSION MEMBER replaces v8's PyTorch MLP, and it is the ONE thing v9
     got right: LR scored 0.945075 OOF on a strictly weaker feature set, measured on a
     clean (self-standardizing) scale, and it is the most decorrelated member we have
     — while blending near-duplicate GBDTs measures at +3e-6. LR alone also gets v9's
     multi-radius WINDOW target encodings (+-30/100/300/1000 USD, +-0.5/2/5 km,
     self-excluded, Bayesian-shrunk with prior 20). Windows are deliberately NOT given
     to the tree members: there is no first-party measurement that helps a GBDT here,
     and keeping the tree path at v8's exact feature set is what lets us attribute any
     regression to change (2) or (3) instead of to everything at once.

  5. THE GATE IS WIRED TO THE OUTPUT. v9 hardcoded base="lgbm_ref" and wrote only
     (a) that member and (b) a fixed blend, so the winning member was never written to
     a file — the submitted artifact was one of the two worst candidates by
     construction. v10 compares every candidate against the absolute bar 0.946055,
     never against a sibling the same run flagged as broken, refuses to ship a blend
     that claims to beat its own best member by more than 0.005 (nothing in this
     competition has ever done that; v9's artifact claimed +0.038 at z=+107), and
     writes the gate's chosen best member as submission_v10_single.csv.

  UNCHANGED FROM v8 (deliberately — v8 is the thing that works)
     5-fold, seed 42, the same LightGBM config, the same XGBoost config (v9's
     "najiama tuned" lossguide/max_leaves=16 XGB scored 0.9277 per-fold and did not
     transfer), digit-decomposition columns STILL target-encoded (income digits
     measured +0.000127, 5/5 folds, as a keeper), the 13 itzzomkar target-mean
     anchors, combined-population frequency encoding, the 24 non-buyer + 2 buyer
     bands with epsilon ordering, and no tie-breaking (measured ceiling +8e-8).

  NOT DONE
     - v9's 10-fold + 2x TE-seed bagging. The bagging was worth +0.00006, which is 5%
       of one public-LB standard error, for double runtime; and going 5->10 folds
       DOUBLES the number of calibration levels concatenated into the OOF, i.e. it
       magnifies exactly the failure that killed v9.
     - OOF-fitted blend weights. Structurally biased (OOF rows carry one model's
       noise, test rows carry the fold-averaged noise); the repo already concluded
       this and weights here are fixed a priori.

KAGGLE: one Input (train/test/sample_submission + EV_Adoption CSV), GPU on.
Expect ~35-50 min for 5 folds x 3 members.
Outputs: submission_v10.csv (fold-rank blend + bands), submission_v10_single.csv
(gate's best member + bands), v10_ckpt_*/ per-fold raw predictions for offline audits.
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

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v10"
N_SPLITS = 5
TE_INNER_FOLDS = 5
RANDOM_STATE = 42

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
DIGIT_KS = range(-4, 4)           # STILL target-encoded (measured +0.000127, 5/5)
TE_SMOOTHS = (2.0, 5.0, 25.0)     # was ('auto', 10, 100) in v8; see docstring item 2

# v8's income/commute key ladder, plus the two fine bins from change (3)
SMOOTH_KEYS = [
    "income_exact_int", "income50_floor", "income250_floor",
    "income100_floor", "income1000_floor", "commute_integer",
]

# window encodings — LR member ONLY
INCOME_RADII = (30.0, 100.0, 300.0, 1000.0)
COMMUTE_RADII = (0.5, 2.0, 5.0)
WINDOW_PRIOR = 20.0

ORIG_AS_ANCHORS = True
ORIG_AS_ROWS = False
MAX_ORIG_BYTES = 20 * 1024 * 1024
ORIG_NAME_HINTS = ("ev_adoption", "range_anxiety", "source", "original")

XGB_USE_GPU = True
LR_USE = True
BLEND_WEIGHTS = {"lgbm_ref": 0.50, "xgb_ref": 0.25, "lr": 0.25}

V8_BLEND_OOF = 0.946055          # the bar. Every gate compares to THIS, not a sibling.
CV_GATE = 0.9462                 # must clear v8 by more than the noise floor
FOLD_GAP_TOL = 0.0010            # pooled-foldrank vs mean-per-fold: above this, distrust
MIN_SHIP_DELTA = 0.00002
# A blend beating its own best member by more than this is not a win, it is a
# measurement artifact. Measured reality on this competition: blend-over-best-member
# lands between +0.0000 and +0.0005, and the public-LB standard error is ~0.0013.
# v9 printed delta=+0.037900 z=+107.15 for exactly this artifact.
BLEND_MAX_GAIN = 0.005

LGBM_PARAMS = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
    min_child_samples=10, subsample=0.812763,   # inert without subsample_freq
    colsample_bytree=0.30293, reg_alpha=0.07094, reg_lambda=2.03303,
    max_bin=1024, feature_pre_filter=False, random_state=42, n_jobs=-1, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=6, min_child_weight=10,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.35,
    reg_alpha=0.07, reg_lambda=2.0, max_bin=1024, tree_method="hist",
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

RESUME = True
CKPT_DIRNAME = "v10_ckpt"
CKPT_INPUT_DIR = None
KAGGLE_INPUT_DIR = "/kaggle/input"

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


def fold_pct(a):
    """Percentile rank WITHIN the vector. This is what kills the per-fold
    calibration offset that destroyed v9's OOF."""
    a = np.asarray(a, dtype=np.float64)
    return (rankdata(a, method="average") / a.size).astype(np.float32)


def _as_1d(x):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 1:
        a = a.reshape(-1)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    return a


# ----------------------------------------------------------------------------
# Target encoding on int codes — sklearn-identical math (asserted by
# test_v8_parity.py, which imports these two functions unchanged).
# ----------------------------------------------------------------------------
def te_map(codes, y, n_cat, smooth, gmean, var_y):
    y = y.astype(np.float64)
    sums = np.bincount(codes, weights=y, minlength=n_cat).astype(np.float64)
    cnts = np.bincount(codes, minlength=n_cat).astype(np.float64)
    seen = cnts > 0
    mean_j = np.where(seen, sums / np.maximum(cnts, 1.0), gmean)
    if smooth == "auto":
        sq = np.bincount(codes, weights=y * y, minlength=n_cat).astype(np.float64)
        s2 = np.where(seen, np.maximum(sq / np.maximum(cnts, 1.0) - mean_j ** 2, 0.0), 0.0)
        denom = var_y * cnts + s2
        lam = np.where(seen & (denom > 0), (var_y * cnts) / np.maximum(denom, 1e-12), 0.0)
        enc = lam * mean_j + (1.0 - lam) * gmean
    else:
        k = float(smooth)
        enc = (sums + k * gmean) / (cnts + k)
    return np.where(seen, enc, gmean).astype(np.float32)


def te_crossfit(codes_tr, y_tr, n_cat, smooth, gmean, var_y, seed):
    out = np.full(len(codes_tr), gmean, dtype=np.float32)
    skf = StratifiedKFold(n_splits=TE_INNER_FOLDS, shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(codes_tr)), y_tr):
        mp = te_map(codes_tr[a], y_tr[a], n_cat, smooth, gmean, var_y)
        out[b] = mp[codes_tr[b]]
    return out


# ----------------------------------------------------------------------------
# Multi-radius neighbourhood target encoding. Self-excluded for fitting rows so
# a row never encodes its own label. Feeds the LR member only.
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
        return self

    def transform(self, values, self_y=None):
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
    roots = [os.path.dirname(os.path.abspath(ORIGINAL_CSV))] \
        if (ORIGINAL_CSV and os.path.isfile(ORIGINAL_CSV)) else \
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
    log("[original] not found - anchors skipped")
    return None


# ----------------------------------------------------------------------------
# Target-free features on the combined train+test population.
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

    comb["income_exact_int"] = np.floor(inc).astype(np.int64)
    comb["income50_floor"] = np.floor(inc / 50.0).astype(np.int64)      # new
    comb["income250_floor"] = np.floor(inc / 250.0).astype(np.int64)    # new
    comb["income100_floor"] = np.floor(inc / 100.0).astype(np.int64)
    comb["income1000_floor"] = np.floor(inc / 1000.0).astype(np.int64)
    comb["commute_integer"] = np.floor(comm).astype(np.int64)

    comb["is_30k_spike"] = (inc == 30000.0).astype("int8")
    comb["is_millionaire_cliff"] = (inc >= 170537.0).astype("int8")
    comb["is_dead_zone"] = ((inc >= 38174.0) & (inc <= 41384.0)).astype("int8")
    comb["is_env_hater"] = (env == 1).astype("int8")
    comb["is_comm_5"] = (np.round(comm, 1) == 5.0).astype("int8")
    comb["is_comm_ge83"] = (comm >= COMMUTE_NO_BUYERS_FROM).astype("int8")

    comb["total_charging"] = comb["Charging_Stations_Near_Home"].to_numpy() \
        + comb["Charging_Stations_Near_Work"].to_numpy()
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
    for c in CATEGORICAL + num_as_str + SMOOTH_KEYS:
        if c not in comb.columns:
            continue
        fm = comb[c].value_counts(normalize=True).to_dict()
        comb[f"{c}_fe"] = comb[c].map(fm).fillna(0.0).astype("float32")
        freq_cols.append(f"{c}_fe")

    te_src = sorted(set(CATEGORICAL) | set(num_as_str) | set(digit_cols) | set(SMOOTH_KEYS))
    te_src = [c for c in te_src if c in comb.columns]

    raw_num = [c for c in comb.columns
               if c not in te_src and pd.api.types.is_numeric_dtype(comb[c])]
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

    log(f"[feat] tree {len(raw_num) + len(te_src)*len(TE_SMOOTHS)} cols "
        f"(raw {len(raw_num)} + TE {len(te_src)}x{len(TE_SMOOTHS)}={len(te_src)*len(TE_SMOOTHS)}) "
        f"| LR +{len(INCOME_RADII)+len(COMMUTE_RADII)} windows | anchors {len(anchors)} "
        f"| freq {len(freq_cols)} | dropped {len(dropped)}")

    return dict(
        ntr=ntr, raw_num=raw_num, te_src=te_src, ncats=ncats,
        Rtr=comb.iloc[:ntr][raw_num].to_numpy(np.float32),
        Rte=comb.iloc[ntr:][raw_num].to_numpy(np.float32),
        Ctr={c: codes[c][:ntr] for c in te_src},
        Cte={c: codes[c][ntr:] for c in te_src},
        inc_tr=inc[:ntr], inc_te=inc[ntr:], com_tr=comm[:ntr], com_te=comm[ntr:],
    )


# ----------------------------------------------------------------------------
# Per-fold design. Encoders fit on training rows ONLY. Returns the tree blocks
# and, separately, the LR-only window blocks.
# ----------------------------------------------------------------------------
def fold_matrix(F, y, tr_idx, va_idx):
    ytr = y[tr_idx]
    gmean = float(ytr.mean())
    var_y = float(np.var(ytr.astype(np.float64)))
    b_tr, b_va, b_te = [F["Rtr"][tr_idx]], [F["Rtr"][va_idx]], [F["Rte"]]

    for c in F["te_src"]:
        ctr = F["Ctr"][c][tr_idx]
        nc = F["ncats"][c] + 1
        for smooth in TE_SMOOTHS:
            full = te_map(ctr, ytr, nc, smooth, gmean, var_y)
            b_tr.append(te_crossfit(ctr, ytr, nc, smooth, gmean, var_y, RANDOM_STATE)[:, None])
            b_va.append(full[F["Ctr"][c][va_idx]][:, None])
            b_te.append(full[F["Cte"][c]][:, None])

    w_tr, w_va, w_te = [], [], []
    for v_tr, v_te, radii in ((F["inc_tr"], F["inc_te"], INCOME_RADII),
                              (F["com_tr"], F["com_te"], COMMUTE_RADII)):
        for r in radii:
            w = WindowTE(r).fit(v_tr[tr_idx], ytr)
            w_tr.append(w.transform(v_tr[tr_idx], self_y=ytr)[:, None])
            w_va.append(w.transform(v_tr[va_idx])[:, None])
            w_te.append(w.transform(v_te)[:, None])

    return (b_tr, b_va, b_te), (w_tr, w_va, w_te)


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
    """Standardizes IN PLACE — the caller must hand in freshly stacked matrices."""
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    sd[sd == 0] = 1.0
    for M in (Xa, Xb, Xc):
        M -= mu
        M /= sd
    m = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", n_jobs=-1)
    m.fit(Xa, ya)
    return _as_1d(m.predict_proba(Xb)[:, 1]), _as_1d(m.predict_proba(Xc)[:, 1])


# ----------------------------------------------------------------------------
# Paired stratified bootstrap over OOF AUC differences. The statistic we ship on
# is exactly the one we print.
# ----------------------------------------------------------------------------
def _auc_from_mask(scores, is_pos, n_pos):
    r = rankdata(scores, method="average")
    return (r[is_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * (len(scores) - n_pos))


def paired_gate(y, base, cand, n_boot=200, seed=0):
    y = np.asarray(y).astype(np.int8)
    a0, a1 = roc_auc_score(y, base), roc_auc_score(y, cand)
    delta = a1 - a0
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    b, c = np.asarray(base), np.asarray(cand)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pi = rng.choice(pos_idx, pos_idx.size, replace=True)
        ni = rng.choice(neg_idx, neg_idx.size, replace=True)
        idx = np.concatenate([pi, ni])
        mask = np.zeros(idx.size, dtype=bool)
        mask[:pi.size] = True
        diffs[i] = _auc_from_mask(c[idx], mask, pi.size) - _auc_from_mask(b[idx], mask, pi.size)
    se = float(np.std(diffs, ddof=1))
    return delta, se, (delta / se if se > 0 else 0.0)


def per_fold_verdicts(y, fold_ids, n_folds, base, cand):
    won = tested = 0
    for f in range(n_folds):
        m = fold_ids == f
        if y[m].min() == y[m].max():
            continue
        tested += 1
        won += roc_auc_score(y[m], cand[m]) > roc_auc_score(y[m], base[m])
    return won, tested


# ----------------------------------------------------------------------------
# Checkpoints (raw per-fold predictions; fold-ranking is applied at aggregation)
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


def _ck(model, fold, ext):
    return os.path.join(_RUN["read"], f"{model}_f{fold}_{ext}.npy")


def ckpt_load(model, fold):
    if not RESUME:
        return None
    fv, ft = _ck(model, fold, "val"), _ck(model, fold, "test")
    if not (os.path.isfile(fv) and os.path.isfile(ft)):
        return None
    try:
        return np.load(fv), np.load(ft)
    except Exception:
        return None


def ckpt_save(model, fold, val, test):
    try:
        d = _RUN["dir"]
        np.save(os.path.join(d, f"{model}_f{fold}_val.npy"), val)
        np.save(os.path.join(d, f"{model}_f{fold}_test.npy"), test)
    except Exception as exc:
        log(f"[ckpt] save {model} f{fold} failed: {exc!r}")


# ----------------------------------------------------------------------------
# Bands: epsilon-ordered blocks, no forced ties
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
        raise RuntimeError("LightGBM unavailable — v10's primary model cannot run.")
    log(f"[env] members: {members} | TE smooths: {TE_SMOOTHS} | fold-rank OOF: ON")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = load_original(train, [train_path, test_path, sample_path])
    if ORIG_AS_ROWS and orig is not None:
        train = pd.concat([train, orig], ignore_index=True)

    y = encode_target(train[TARGET])
    log(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    F = build_features(train, test, orig)
    n_tree = len(F["raw_num"]) + len(F["te_src"]) * len(TE_SMOOTHS)
    n_lr = n_tree + len(INCOME_RADII) + len(COMMUTE_RADII)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    init_ckpt(out_dir, f"{n_lr}|{len(train)}|{N_SPLITS}|{RANDOM_STATE}|{TE_SMOOTHS}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(train)), y))
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va) in enumerate(folds):
        fold_ids[va] = f

    oof_raw = {m: np.full(len(y), np.nan, dtype=np.float32) for m in members}
    oof_rnk = {m: np.full(len(y), np.nan, dtype=np.float32) for m in members}
    te_rnk = {m: np.zeros(len(F["Rte"]), dtype=np.float32) for m in members}
    fold_auc = {m: [] for m in members}
    failed = set()
    n_te = len(F["Rte"])

    def run_member(m, va_idx, f, fn):
        """One member, one fold. Checkpointed, shape-guarded, and non-fatal: a member
        that throws is dropped for the rest of the run."""
        if m in failed:
            return
        cached = ckpt_load(m, f)
        if cached is not None:
            vp, tp = _as_1d(cached[0]), _as_1d(cached[1])
            tag = "resumed"
        else:
            try:
                vp, tp = fn()
            except Exception as exc:
                log(f"    !! {m} f{f} FAILED ({exc!r}) -> member dropped")
                failed.add(m)
                return
            vp, tp = _as_1d(vp), _as_1d(tp)
            if len(vp) != len(va_idx) or len(tp) != n_te:
                log(f"    !! {m} f{f} BAD SHAPE {len(vp)}/{len(tp)} -> member dropped")
                failed.add(m)
                return
            ckpt_save(m, f, vp, tp)
            tag = f"auc={roc_auc_score(y[va_idx], vp):.6f}"
        # THE FIX: percentile-rank inside the fold before it enters the OOF vector,
        # so cross-fold rank comparisons are the only thing left standing.
        oof_raw[m][va_idx] = vp
        oof_rnk[m][va_idx] = fold_pct(vp)
        te_rnk[m] += fold_pct(tp) / N_SPLITS
        fold_auc[m].append(roc_auc_score(y[va_idx], vp))
        log(f"    [{m} f{f}] {tag}")

    for f, (tr_idx, va_idx) in enumerate(folds):
        tf = time.perf_counter()
        (b_tr, b_va, b_te), (w_tr, w_va, w_te) = fold_matrix(F, y, tr_idx, va_idx)
        Xa, Xb, Xc = np.hstack(b_tr), np.hstack(b_va), np.hstack(b_te)
        log(f"[fold {f}] tree design {Xa.shape} in {time.perf_counter()-tf:.1f}s")

        if "lgbm_ref" in members:
            run_member("lgbm_ref", va_idx, f,
                       lambda: run_lightgbm(Xa, y[tr_idx], Xb, y[va_idx], Xc))
        if "xgb_ref" in members:
            run_member("xgb_ref", va_idx, f,
                       lambda: run_xgboost(Xa, y[tr_idx], Xb, y[va_idx], Xc,
                                           XGB_USE_GPU and n_gpus > 0))
        # Free the tree matrix before stacking the wider LR one; they coexist otherwise.
        del Xa, Xb, Xc
        gc.collect()
        if "lr" in members:
            run_member("lr", va_idx, f, lambda: run_lr(np.hstack(b_tr + w_tr), y[tr_idx],
                                                       np.hstack(b_va + w_va), y[va_idx],
                                                       np.hstack(b_te + w_te)))
        del b_tr, b_va, b_te, w_tr, w_va, w_te
        gc.collect()

    live = [m for m in members if m not in failed and np.isfinite(oof_rnk[m]).all()]
    if not live:
        raise RuntimeError("All members failed.")

    # ---- measurement health check (this is what v9 never ran) ----------------
    log("\n" + "=" * 78)
    log("MEASUREMENT HEALTH  (pooled must sit within %g of the mean per-fold AUC)" % FOLD_GAP_TOL)
    healthy = True
    for m in live:
        mean_pf = float(np.mean(fold_auc[m]))
        raw, rnk = roc_auc_score(y, oof_raw[m]), roc_auc_score(y, oof_rnk[m])
        gap = rnk - mean_pf
        ok = abs(gap) <= FOLD_GAP_TOL
        healthy &= ok
        log(f"  {m:9s} per-fold mean {mean_pf:.6f} | pooled(raw) {raw:.6f} "
            f"[{raw-mean_pf:+.6f}] | pooled(foldrank) {rnk:.6f} [{gap:+.6f}] "
            f"{'OK' if ok else '!! UNHEALTHY'}")
    if not healthy:
        log("  -> fold-rank OOF disagrees with per-fold mean. Do NOT trust the gate.")
    log("=" * 78)

    # ---- gate: every candidate vs the absolute v8 bar, paired vs the best member ----
    log(f"\nBAR = v8 blend OOF {V8_BLEND_OOF} (LB 0.94617). Gate {CV_GATE}.\n")
    scored = {m: roc_auc_score(y, oof_rnk[m]) for m in live}
    best_member = max(scored, key=scored.get)
    for m in live:
        note = "PASS" if scored[m] >= CV_GATE else "below"
        pair = ""
        if m != best_member:
            d, se, z = paired_gate(y, oof_rnk[best_member], oof_rnk[m])
            won, tested = per_fold_verdicts(y, fold_ids, N_SPLITS,
                                            oof_rnk[best_member], oof_rnk[m])
            pair = (f"  vs best: delta={d:+.6f} se={se:.6f} z={z:+.2f} "
                    f"folds+={won}/{tested}")
        log(f"{m:9s} OOF = {scored[m]:.6f}  [{note} gate]  "
            f"vs bar {scored[m]-V8_BLEND_OOF:+.6f}  "
            f"{'<- BEST MEMBER' if m == best_member else ''}{pair}")

    wsum = sum(BLEND_WEIGHTS[m] for m in live)
    blend_oof = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(oof_rnk[m]) for m in live], axis=0)
    blend_te = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(te_rnk[m]) for m in live], axis=0)
    blend_auc = roc_auc_score(y, blend_oof)
    d, se, z = paired_gate(y, oof_rnk[best_member], blend_oof)
    won, tested = per_fold_verdicts(y, fold_ids, N_SPLITS, oof_rnk[best_member], blend_oof)
    ship_blend = (MIN_SHIP_DELTA <= d <= BLEND_MAX_GAIN and won >= 0.8 * tested)
    alarm = ""
    if d > BLEND_MAX_GAIN:
        alarm = ("  << IMPLAUSIBLE: no blend here has ever beaten its best member by\n"
                 f"     more than {BLEND_MAX_GAIN}. This is the v9 artifact (it printed\n"
                 f"     delta=+0.037900 z=+107.15). Recheck the HEALTHY table above.")
    log(f"{'BLEND':9s} OOF = {blend_auc:.6f}  vs best member: delta={d:+.6f} "
        f"se={se:.6f} z={z:+.2f} folds+={won}/{tested}{alarm}")

    ship = blend_te if ship_blend else te_rnk[best_member]
    ship_auc = blend_auc if ship_blend else scored[best_member]
    log("-" * 78)
    log(f"SUBMIT submission_{VERSION}.csv -> {'blend' if ship_blend else 'single: ' + best_member}"
        f" | OOF {ship_auc:.6f} | {'CLEARS' if ship_auc >= CV_GATE else 'BELOW'} gate {CV_GATE}"
        f" | {'HEALTHY' if healthy else 'MEASUREMENT UNHEALTHY - DO NOT SHIP'}")
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

    write_sub(f"submission_{VERSION}_single.csv", te_rnk[best_member])
    write_sub(f"submission_{VERSION}.csv", ship)
    for m in live:
        np.save(os.path.join(_RUN["out_dir"], f"v10_{m}_oof_rank.npy"), oof_rnk[m])
        np.save(os.path.join(_RUN["out_dir"], f"v10_{m}_oof_raw.npy"), oof_raw[m])
        np.save(os.path.join(_RUN["out_dir"], f"v10_{m}_test.npy"), te_rnk[m])
    np.save(os.path.join(_RUN["out_dir"], "v10_fold_ids.npy"), fold_ids)
    log(f"Runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
