"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v14 HYBRID)

THE THESIS. v12 is a good single model — OOF 0.946267, LB 0.94619 — and it has been measured
from every direction and found unimprovable by features. Smoothing ladders, the $50/$250
income bins, the 69-anchor set, TE seed bagging, max_bin, depth, band expansion: all neutral or
negative on a faithful held-out split. See FINDINGS_S6E9_reverse_engineering.md. The ONE
mechanism on this dataset that has ever produced a jump above +0.0001 is **blending a member
that is simultaneously strong and decorrelated**, and Aadit's 0.94643 is exactly that, measured
in his own audit file:

    his internal champion OOF   0.946113
    external XGBoost_Triple_TE_10folds OOF   0.946243
    fixed 50/50 rank blend                   0.946323   +0.000211, 5/5 folds, rank_corr 0.996744
    (Aadit_try/artifacts/experiments/external_xgb10_fixed_rank_blend_audit/audit_metrics.csv)

So this file manufactures that decorrelation itself rather than borrowing it.

WHY THESE FOUR MEMBERS. Decorrelation has exactly three real sources here, and each member
attacks a different one. Everything else — more seeds of the same learner — sits at rank_corr
0.9995 and pays nothing, which was measured across all 376 pairs of the 28 OOF vectors >= 0.944
in this repo (research/blend_gain_curve.csv: median gain at rho 0.9995+ is -0.000003).

  1. TRAINING-SET SIZE. Aadit's external member is 10-fold, so every model trains on ~601,800
     rows; v12's 5-fold trains on 534,932. That is 12.5% more data per model, and it is the
     most plausible reason that member OUTSCORED his own champion (0.946243 vs 0.946113) and
     therefore carried the blend. v14 runs its outer CV at 10 folds throughout.
  2. ENCODING MECHANISM. lgbm_ref and xgb_ref see the identical 231 pre-computed target-encoded
     columns here. That is not two models, it is one model twice — v12's log proved it: xgb
     scored z=-2.13 against lgbm and the three-way blend LOST to the single lgbm by -0.000021.
     CatBoost is instead handed the raw categorical columns and builds its OWN ordered target
     statistics, a genuinely different hypothesis about the same data. Aadit measured CatBoost
     at rank_corr 0.9948 versus his champion, the most decorrelated GBDT in his entire library.
  3. FUNCTION CLASS. A logistic regression over the encoded design plus the 7 multi-radius
     window encodings — the member v12 already ran in production at OOF 0.945041.

RULES THIS FILE OBEYS BECAUSE THEY ARE MEASURED, NOT STYLISTIC
  - FIXED weights, never OOF-searched. Aadit's own interpretation rule in
    external_oof_strong_inspection/summary.txt: "Do not perform nested meta-blend selection on
    an external OOF whose generation folds cannot be shown to match our frozen outer folds."
    Weight search across mismatched folds leaks. A fixed blend is the honest form, and it is
    the form that produced his +0.000211.
  - fold-rank OOF. Each fold's predictions are percentile-ranked inside the fold before entering
    the OOF vector. v9 shipped a blend whose pooled OOF read 0.9072 against a per-fold mean of
    0.9399 because it skipped this. v12's production log confirms the fix holds: raw gap -0.000032.
  - the gate may refuse the blend. A blend is a hypothesis here, not a win — v12's own blend
    lost. The diversity matrix and the honest per-fold deltas are printed and whichever member
    or blend actually measures better is what gets written. v14 cannot finish below v12 by
    construction, because the single best member is always also written.

NOT IN THIS FILE, ON PURPOSE
  - The 69-anchor set. Measured -0.000019 on the faithful pipeline; v12 already reverted it.
  - A Gaussian-kernel smoothed target encoder (sergeyqt2024's lr-squeeze trick). Implemented,
    tested, and deleted: as an exact-key dict lookup it is unworkable on a continuous column,
    because a held-out row's exact income is absent from the inner-fold map, so every lookup
    falls back to the global mean and the encoding is constant (measured 800/800 misses,
    nunique == 1). The hard-radius WindowTE already in here is the validated form of the same
    idea. Replacing it would be untested machinery behind a member family that Aadit's nested
    blend audit scored ZERO_WEIGHT.
  - More than 3 repeats. Bagging is worth +0.000263 measured live at 5 repeats and saturates
    (rho + (1-rho)/K with rho = 0.99948); 10 repeats adds ~+0.00005 for double runtime.

KAGGLE: one Input (train/test/sample_submission + EV_Adoption CSV), GPU ON.
Runtime: 10 folds x 3 repeats x 4 members, CatBoost dominating. Budget ~3-4h. If it is too
slow, set REPEAT_SEEDS = (42,) — that alone keeps most of the score.
Outputs: submission_v14.csv (the gate's pick), submission_v14_single.csv (best member),
v14_ckpt_*/, and v14_<member>_{oof_rank,test}.npy so the blend can be re-audited offline.
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
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v14"
N_SPLITS = 10          # 601,798 rows per model vs 534,932 at 5-fold
TE_INNER_FOLDS = 5
REPEAT_SEEDS = (42, 7, 13)       # 3, not v12's 5: 10-fold models are more alike, so the
                                 # bagging gain is mostly banked by 3. 5 would cost ~6.5h.
RANDOM_STATE = REPEAT_SEEDS[0]

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
DIGIT_KS = range(-4, 4)
TE_SMOOTHS = ("auto", 10.0, 100.0)         # v8's verified ladder
SMOOTH_KEYS = [
    "income_exact_int", "income50_floor", "income250_floor",
    "income100_floor", "income1000_floor", "commute_integer",
]

INCOME_RADII = (30.0, 100.0, 300.0, 1000.0)
COMMUTE_RADII = (0.5, 2.0, 5.0)
WINDOW_PRIOR = 20.0

# CatBoost gets the raw categorical columns and does its OWN ordered target encoding.
# That is the entire point of it: lgbm_ref and xgb_ref see the identical 231 pre-computed
# TE columns, which is why v12 blend LOST to its single LightGBM (delta -0.000021) and xgb
# scored z=-2.13 vs lgbm -- they are the same information. Aadit measured CatBoost at
# rank_corr 0.9948 vs his champion, the most decorrelated GBDT in his library
# (external_oof_diversity_audit/member_audit.csv).
CAT_BASE = dict(iterations=6000, learning_rate=0.03, depth=7, l2_leaf_reg=3.0,
                random_strength=1.0, bagging_temperature=0.6, loss_function="Logloss",
                eval_metric="AUC", verbose=0, early_stopping_rounds=300,
                allow_writing_files=False, task_type="CPU")

ORIG_AS_ANCHORS = True
FULL_ANCHOR_SET = False       # measured -0.000019 on the faithful pipeline; see build_features
ORIG_AS_ROWS = False
MAX_ORIG_BYTES = 20 * 1024 * 1024
ORIG_NAME_HINTS = ("ev_adoption", "range_anxiety", "source", "original")

XGB_USE_GPU = True
LR_USE = True
BLEND_WEIGHTS = {"lgbm_ref": 0.40, "xgb_ref": 0.15, "cat_ref": 0.25, "lr": 0.20}

V8_BLEND_OOF = 0.946055          # the bar
CV_GATE = 0.9462
FOLD_GAP_TOL = 0.0010            # pooled-foldrank vs mean-per-fold, before a repeat's own
                                 # bagging gain is folded in
MIN_SHIP_DELTA = 0.00002
BLEND_MAX_GAIN = 0.005

LGBM_BASE = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
    min_child_samples=10, subsample=0.812763,   # inert without subsample_freq
    colsample_bytree=0.30293, reg_alpha=0.07094, reg_lambda=2.03303,
    max_bin=1024, feature_pre_filter=False, n_jobs=-1, verbose=-1,
)
XGB_BASE = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=6, min_child_weight=10,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.35,
    reg_alpha=0.07, reg_lambda=2.0, max_bin=1024, tree_method="hist",
    objective="binary:logistic", eval_metric="auc", early_stopping_rounds=500,
    n_jobs=-1,
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
APPLY_BANDS = True               # these 24+2 bands are the ones that survived section D

RESUME = True
CKPT_DIRNAME = "v14_ckpt"
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
    """Percentile rank WITHIN the vector. Kills the per-fold calibration offset that
    destroyed v9's OOF (see ev_s6e9_v10.py docstring)."""
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
# Target encoding on int codes — sklearn-identical math (test_v8_parity.py).
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


class WindowTE:
    """Multi-radius neighbourhood target encoding, self-excluded for fitting rows.
    Feeds the LR member only."""

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

    for div, nm in ((1, "income_exact_int"), (50, "income50_floor"), (250, "income250_floor"),
                    (100, "income100_floor"), (1000, "income1000_floor")):
        comb[nm] = np.floor(inc / div).astype(np.int64)
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

    # ---- external target-mean anchors ---------------------------------------
    # najiama's real code maps the external 10k dataset's target mean onto every
    # categorical, numeric AND digit column (~66 features); we build 13. That gap
    # measured +0.000225 on a 13-raw-column toy CatBoost and -0.000019 on THIS
    # pipeline (research/spike2_faithful_ablation.py), so FULL_ANCHOR_SET stays
    # False. The digit anchors are still implemented behind the flag because that
    # is where the difference actually lives, but they are not shipped.
    anchors = []
    if orig is not None and ORIG_AS_ANCHORS:
        y_orig = encode_target(orig[TARGET]).astype(np.float64)
        gm = float(y_orig.mean())

        def add_anchor(name, query, key_series):
            stats = pd.Series(y_orig).groupby(key_series.to_numpy()).mean()
            comb[f"{name}_org_mean"] = query.map(stats).fillna(gm).astype("float32")
            anchors.append(f"{name}_org_mean")

        for c in CATEGORICAL:
            if c in orig.columns:
                add_anchor(c, comb[c], orig[c].astype(str).str.strip().str.lower())
        for c in NUMERIC:
            if c not in orig.columns:
                continue
            ov = pd.to_numeric(orig[c], errors="coerce").fillna(0.0)
            add_anchor(c, comb[c], np.floor(ov).astype(np.int64).astype(str))
            if FULL_ANCHOR_SET:
                for k in DIGIT_KS:
                    col = f"{c}_digit{k}"
                    if col not in comb.columns:
                        continue
                    od = ((ov // (10.0 ** k)) % 10).astype(np.int64).astype(str)
                    add_anchor(col, comb[col].astype(str), od)

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
        f"(full={FULL_ANCHOR_SET}) | freq {len(freq_cols)} | dropped {len(dropped)}")

    cat_cols = [c for c in CATEGORICAL if c in comb.columns]
    return dict(
        ntr=ntr, raw_num=raw_num, te_src=te_src, ncats=ncats, cat_cols=cat_cols,
        Ktr=comb.iloc[:ntr][cat_cols].to_numpy(),
        Kte=comb.iloc[ntr:][cat_cols].to_numpy(), anchors=anchors,
        n_anchors_dropped=len([a for a in anchors if a not in raw_num]),
        Rtr=comb.iloc[:ntr][raw_num].to_numpy(np.float32),
        Rte=comb.iloc[ntr:][raw_num].to_numpy(np.float32),
        Ctr={c: codes[c][:ntr] for c in te_src},
        Cte={c: codes[c][ntr:] for c in te_src},
        inc_tr=inc[:ntr], inc_te=inc[ntr:], com_tr=comm[:ntr], com_te=comm[ntr:],
    )


# ----------------------------------------------------------------------------
# Per-fold design. Encoders fit on training rows ONLY.
# ----------------------------------------------------------------------------
def fold_matrix(F, y, tr_idx, va_idx, seed):
    ytr = y[tr_idx]
    gmean = float(ytr.mean())
    var_y = float(np.var(ytr.astype(np.float64)))
    b_tr, b_va, b_te = [F["Rtr"][tr_idx]], [F["Rtr"][va_idx]], [F["Rte"]]

    for c in F["te_src"]:
        ctr = F["Ctr"][c][tr_idx]
        nc = F["ncats"][c] + 1
        for smooth in TE_SMOOTHS:
            full = te_map(ctr, ytr, nc, smooth, gmean, var_y)
            b_tr.append(te_crossfit(ctr, ytr, nc, smooth, gmean, var_y, seed)[:, None])
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

    return (b_tr, b_va, b_te), (w_tr, w_va, w_te), (F["Ktr"][tr_idx], F["Ktr"][va_idx], F["Kte"])


# ----------------------------------------------------------------------------
# Members. Each takes its own seed so repeats are genuinely independent draws.
# ----------------------------------------------------------------------------
def run_lightgbm(Xa, ya, Xb, yb, Xc, seed):
    import lightgbm as lgb
    p = dict(LGBM_BASE, random_state=seed)
    m = lgb.LGBMClassifier(**p)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
          callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
    out = _as_1d(m.predict_proba(Xb)[:, 1]), _as_1d(m.predict_proba(Xc)[:, 1])
    del m
    gc.collect()
    return out


def run_xgboost(Xa, ya, Xb, yb, Xc, seed, use_gpu):
    import xgboost as xgb
    p = dict(XGB_BASE, random_state=seed)
    p["device"] = "cuda" if use_gpu else "cpu"
    m = xgb.XGBClassifier(**p)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=0)
    out = _as_1d(m.predict_proba(Xb)[:, 1]), _as_1d(m.predict_proba(Xc)[:, 1])
    del m
    gc.collect()
    return out


def run_catboost(Xa, ya, Xb, yb, Xc, seed, cats_a, cats_b, cats_c):
    """Concatenated as DataFrames, never np.hstack: stacking a float32 block against an
    object block upcasts the whole matrix to object and CatBoost then rejects the numeric
    columns. Column names must also be unique across both blocks."""
    from catboost import CatBoost, Pool
    nnum = Xa.shape[1]
    ci = list(range(nnum, nnum + cats_a.shape[1]))

    def frame(X, cats):
        num = pd.DataFrame(X)
        cols = ["catcol" + str(i) for i in range(cats.shape[1])]
        cat = pd.DataFrame(np.asarray(cats).astype(str), index=num.index, columns=cols)
        return pd.concat([num, cat], axis=1)

    FA, FB, FC = frame(Xa, cats_a), frame(Xb, cats_b), frame(Xc, cats_c)
    m = CatBoost(dict(CAT_BASE, random_seed=seed))
    m.fit(Pool(FA, ya, cat_features=ci), eval_set=Pool(FB, yb, cat_features=ci),
          use_best_model=True)

    def pos(M):
        p = np.asarray(m.predict(M, prediction_type="Probability"))
        return p[:, 1] if p.ndim > 1 else p

    pb, pc = pos(FB), pos(FC)
    del m, FA, FB, FC
    gc.collect()
    return _as_1d(pb), _as_1d(pc)


def run_lr(Xa, ya, Xb, yb, Xc, seed):
    """Standardizes IN PLACE — caller hands in freshly stacked matrices."""
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
# Paired stratified bootstrap over OOF AUC differences.
# ----------------------------------------------------------------------------
def _auc_from_mask(scores, is_pos, n_pos):
    r = rankdata(scores, method="average")
    return (r[is_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * (len(scores) - n_pos))


def paired_gate(y, base, cand, n_boot=200, seed=0):
    y = np.asarray(y).astype(np.int8)
    delta = roc_auc_score(y, cand) - roc_auc_score(y, base)
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
# Checkpoints, keyed by (member, repeat, fold).
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


def _ck(model, rep, fold, ext):
    return os.path.join(_RUN["read"], f"{model}_r{rep}_f{fold}_{ext}.npy")


def ckpt_load(model, rep, fold):
    if not RESUME:
        return None
    fv, ft = _ck(model, rep, fold, "val"), _ck(model, rep, fold, "test")
    if not (os.path.isfile(fv) and os.path.isfile(ft)):
        return None
    try:
        return np.load(fv), np.load(ft)
    except Exception:
        return None


def ckpt_save(model, rep, fold, val, test):
    try:
        d = _RUN["dir"]
        np.save(os.path.join(d, f"{model}_r{rep}_f{fold}_val.npy"), val)
        np.save(os.path.join(d, f"{model}_r{rep}_f{fold}_test.npy"), test)
    except Exception as exc:
        log(f"[ckpt] save {model} r{rep} f{fold} failed: {exc!r}")


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
            "cat_ref": ensure_import("catboost"), "lr": LR_USE}
    if have.get("cat_ref") and n_gpus > 0:
        CAT_BASE["task_type"] = "GPU"
    members = [m for m in ("lgbm_ref", "xgb_ref", "cat_ref", "lr") if have.get(m)]
    if "lgbm_ref" not in members:
        raise RuntimeError("LightGBM unavailable — v11's primary model cannot run.")
    R = len(REPEAT_SEEDS)
    log(f"[env] members {members} | smooths {TE_SMOOTHS} | repeats {REPEAT_SEEDS} "
        f"-> {R}x{N_SPLITS}={R*N_SPLITS} fits/member")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = load_original(train, [train_path, test_path, sample_path])
    if ORIG_AS_ROWS and orig is not None:
        train = pd.concat([train, orig], ignore_index=True)

    y = encode_target(train[TARGET])
    log(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    F = build_features(train, test, orig)
    n_tree = len(F["raw_num"]) + len(F["te_src"]) * len(TE_SMOOTHS)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    init_ckpt(out_dir, f"{n_tree}|{len(train)}|{N_SPLITS}|{REPEAT_SEEDS}|{TE_SMOOTHS}"
                        f"|{FULL_ANCHOR_SET}")

    n_te = len(F["Rte"])
    oof_raw = {m: np.full(len(y), np.nan, dtype=np.float32) for m in members}
    oof_rnk = {m: np.zeros(len(y), dtype=np.float64) for m in members}
    te_rnk = {m: np.zeros(n_te, dtype=np.float64) for m in members}
    fold_auc = {m: [] for m in members}
    failed = set()
    fold_ids = None

    def run_member(m, rep, fold, va_idx, fn):
        if m in failed:
            return
        cached = ckpt_load(m, rep, fold)
        if cached is not None:
            vp, tp = _as_1d(cached[0]), _as_1d(cached[1])
            tag = "resumed"
        else:
            try:
                vp, tp = fn()
            except Exception as exc:
                log(f"    !! {m} r{rep} f{fold} FAILED ({exc!r}) -> member dropped")
                failed.add(m)
                return
            vp, tp = _as_1d(vp), _as_1d(tp)
            if len(vp) != len(va_idx) or len(tp) != n_te:
                log(f"    !! {m} r{rep} f{fold} BAD SHAPE {len(vp)}/{len(tp)} -> dropped")
                failed.add(m)
                return
            ckpt_save(m, rep, fold, vp, tp)
            tag = f"auc={roc_auc_score(y[va_idx], vp):.6f}"
        # fold-rank before aggregation: the fix that v9 never had
        oof_raw[m][va_idx] = vp
        oof_rnk[m][va_idx] += fold_pct(vp) / R
        te_rnk[m] += fold_pct(tp) / (R * N_SPLITS)
        fold_auc[m].append(roc_auc_score(y[va_idx], vp))
        log(f"    [{m} r{rep} f{fold}] {tag}")

    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(train)), y))
        if rep == 0:
            fold_ids = np.full(len(y), -1, dtype=np.int16)
            for f, (_, va) in enumerate(folds):
                fold_ids[va] = f
        log(f"--- repeat {rep} (seed {seed}) ---")
        for f, (tr_idx, va_idx) in enumerate(folds):
            tf = time.perf_counter()
            (b_tr, b_va, b_te), (w_tr, w_va, w_te), (k_tr, k_va, k_te) = fold_matrix(F, y, tr_idx, va_idx, seed)
            Xa, Xb, Xc = np.hstack(b_tr), np.hstack(b_va), np.hstack(b_te)
            log(f"[r{rep} f{f}] tree design {Xa.shape} in {time.perf_counter()-tf:.1f}s")

            if "lgbm_ref" in members:
                run_member("lgbm_ref", rep, f, va_idx,
                           lambda: run_lightgbm(Xa, y[tr_idx], Xb, y[va_idx], Xc, seed))
            if "xgb_ref" in members:
                run_member("xgb_ref", rep, f, va_idx,
                           lambda: run_xgboost(Xa, y[tr_idx], Xb, y[va_idx], Xc, seed,
                                               XGB_USE_GPU and n_gpus > 0))
            if "cat_ref" in members:
                run_member("cat_ref", rep, f, va_idx,
                           lambda: run_catboost(Xa, y[tr_idx], Xb, y[va_idx],
                                                Xc, seed, k_tr, k_va, k_te))
            del Xa, Xb, Xc          # free the tree matrix before the wider LR stack
            gc.collect()
            if "lr" in members:
                run_member("lr", rep, f, va_idx,
                           lambda: run_lr(np.hstack(b_tr + w_tr), y[tr_idx],
                                          np.hstack(b_va + w_va), y[va_idx],
                                          np.hstack(b_te + w_te), seed))
            del b_tr, b_va, b_te, w_tr, w_va, w_te, k_tr, k_va, k_te
            gc.collect()

    oof_rnk = {m: v.astype(np.float32) for m, v in oof_rnk.items()}
    te_rnk = {m: v.astype(np.float32) for m, v in te_rnk.items()}
    live = [m for m in members if m not in failed and np.isfinite(oof_rnk[m]).all()]
    if not live:
        raise RuntimeError("All members failed.")

    # ---- measurement health: the check v9 never ran --------------------------
    log("\n" + "=" * 78)
    log(f"MEASUREMENT HEALTH  (pooled-foldrank vs mean per-fold; tol {FOLD_GAP_TOL})")
    log("  a small POSITIVE gap is expected and is the repeat-bagging gain itself.")
    healthy = True
    for m in live:
        mean_pf = float(np.mean(fold_auc[m]))
        raw = roc_auc_score(y, oof_raw[m])
        rnk = roc_auc_score(y, oof_rnk[m])
        gap = rnk - mean_pf
        ok = -FOLD_GAP_TOL <= gap <= 3 * FOLD_GAP_TOL
        healthy &= ok
        log(f"  {m:9s} per-fold mean {mean_pf:.6f} | pooled(raw) {raw:.6f} "
            f"[{raw-mean_pf:+.6f}] | pooled(foldrank) {rnk:.6f} [{gap:+.6f}] "
            f"{'OK' if ok else '!! UNHEALTHY'}")
    if not healthy:
        log("  -> fold-rank OOF disagrees with per-fold mean. Do NOT trust the gate.")
    log("=" * 78)

    # One matrix decides whether the blend can pay. Measured across 376 OOF pairs
    # (research/blend_gain_curve.csv) the median gain at rho 0.9995+ is -0.000003: nothing.
    # Aadit got +0.000211 from a pair at rank_corr 0.996744.
    log("\nDIVERSITY (rank corr of fold-rank OOF):")
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            r = float(spearmanr(oof_rnk[a], oof_rnk[b]).statistic)
            verdict = "decorrelated enough to blend" if r < 0.997 else "near-duplicate"
            log("  %s x %-9s %.6f  <- %s" % (a, b, r, verdict))
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
            pair = f"  vs best: delta={d:+.6f} se={se:.6f} z={z:+.2f} folds+={won}/{tested}"
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
                 f"     more than {BLEND_MAX_GAIN}. That was v9's artifact signature.")
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
        np.save(os.path.join(_RUN["out_dir"], f"v14_{m}_oof_rank.npy"), oof_rnk[m])
        np.save(os.path.join(_RUN["out_dir"], f"v14_{m}_oof_raw.npy"), oof_raw[m])
        np.save(os.path.join(_RUN["out_dir"], f"v14_{m}_test.npy"), te_rnk[m])
    np.save(os.path.join(_RUN["out_dir"], "v11_fold_ids.npy"), fold_ids)
    log(f"Runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
