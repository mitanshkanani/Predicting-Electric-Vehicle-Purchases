"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v8)

v8 re-implements the ONLY top public solution whose source we could retrieve and
verify (najiama, CV 0.94607 / LB 0.94638), plus two fixes our own first-party
audits proved we needed. Rationale and citations: DEEP_RESEARCH_S6E9_v8_GAP_ANALYSIS.md.

WHY v8 EXISTS. v7 = LB 0.94563 from a 2.5h five-model ensemble and still loses to
a single LightGBM that runs in 20 minutes. The gap is not the blend, not bin
resolution (that solution uses max_bin=1024, which we already use), not
post-processing. It is FEATURE BREADTH: its CV (0.94607) sits 0.00056 above our
best single member's CV (0.94557), so the deficit is visible in CV and testable
before we spend a submission.

WHAT v8 CHANGES vs v7
  1. Target-encode EVERYTHING string-like at three smoothings ('auto'/10/100):
     the 6 categoricals, every numeric stringified, all 56 digit columns, and 4
     multi-scale income/commute bins -> ~75 keys x 3 = ~225 TE features.
     v7 encoded 15 hand-picked keys at 2 smoothings.
  2. Frequency encoding over the COMBINED train+test population (transductive).
  3. The external itzzomkar dataset becomes per-column target-mean ANCHOR
     features instead of 10k concatenated rows — real-world priors, and it
     dodges the distribution mismatch we measured (real 60-65k income peak is
     10.7% vs 5.0% synthetic).
  4. Verified LightGBM config verbatim (num_leaves=32, max_depth=5, lr=0.02,
     colsample_bytree=0.303, min_child_samples=10, n_estimators=20000,
     ES=500, max_bin=1024, feature_pre_filter=False), 5-fold.
  5. Bands epsilon-ORDER inside each block instead of pinning rows to identical
     0.0/1.0. v7 created 5,189 forced ties; a band hiding 1% positives costs
     ~0.0005, and intra-band ordering recovers that for free.
  6. The lexsort secondary-model tie-break is deleted — our audit measured its
     perfect-execution ceiling at +0.00000008.

IMPLEMENTATION NOTE — why we do not call sklearn's TargetEncoder directly
  Feeding it ~75 object/string columns over 955k rows would materialise ~50M
  Python strings (several GB) and risk the same OOM death we've already had
  twice. Instead we encode on int32 factorise codes and reproduce sklearn's
  exact math: fixed smoothing (sum_y + K*ybar)/(n + K) and 'auto' empirical-Bayes
  lambda = var_y*n / (var_y*n + s2_within). Parity with sklearn is asserted by
  test_v8_parity.py, not assumed. Cross-fitting over 5 inner folds is replicated
  for train rows, so nothing sees its own label.

OUTPUTS
  submission_v8_single.csv  = pure LightGBM replication + bands  <- submit FIRST
  submission_v8.csv         = blend (lgbm .55 / xgb .25 / nn .20) + bands
  v8_ckpt/                  = per-fold checkpoints (resume)
  v8_<member>_oof/test.npy  = per-member predictions for offline blend audits

KAGGLE: one Input with train/test/sample_submission + EV_Adoption CSV, GPU on.
Measured locally at full scale: features 22s, target encoding 44s per fold
(~3.7 min for 5 folds), peak design matrices 1.03 GB, 270 features. LightGBM
dominates; the reference ran this same config in 1,167s on 4 CPUs. Expect
~30-50 min end to end.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v8"
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
DIGIT_KS = range(-4, 4)
TE_SMOOTHS = ("auto", 10.0, 100.0)

ORIG_AS_ANCHORS = True
ORIG_AS_ROWS = False
MAX_ORIG_BYTES = 20 * 1024 * 1024
ORIG_NAME_HINTS = ("ev_adoption", "range_anxiety", "source", "original")

XGB_USE_GPU = True
NN_USE = True
BATCH = 8192
MAX_EPOCHS = 30
PATIENCE = 5

BLEND_WEIGHTS = {"lgbm_ref": 0.55, "xgb_a": 0.25, "nn": 0.20}
CV_GATE = 0.9460

RESUME = True
CKPT_DIRNAME = "v8_ckpt"
CKPT_INPUT_DIR = None
KAGGLE_INPUT_DIR = "/kaggle/input"

LGBM_PARAMS = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
    min_child_samples=10, subsample=0.812763,
    # subsample_freq deliberately NOT set: LightGBM ignores `subsample` unless
    # bagging_freq > 0, so in the reference recipe row-sampling is inert. We
    # replicate the verified behaviour rather than "fixing" it.
    colsample_bytree=0.30293, reg_alpha=0.07094, reg_lambda=2.03303,
    max_bin=1024, feature_pre_filter=False, random_state=42,
    n_jobs=-1, verbose=-1,
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

_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(msg, flush=True)


# ----------------------------------------------------------------------------
# Environment / path helpers.
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
    shallow = [
        "/kaggle/input/datasets/mitanshkanani/dataset-for-experimentation-final",
        "/kaggle/working", "/content/data", "/content", "data", ".", os.getcwd(),
    ]
    seen = set()
    for d in shallow:
        if os.path.isdir(d) and d not in seen:
            seen.add(d)
            yield d
    if os.path.isdir(KAGGLE_INPUT_DIR):
        for dirpath, _dirs, _files in os.walk(KAGGLE_INPUT_DIR):   # no depth pruning
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
        raise FileNotFoundError("train.csv / test.csv not found.\nContents of "
                                + KAGGLE_INPUT_DIR + ":\n"
                                + ("\n".join(listing) or "  (nothing mounted)"))
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
    """Every member must hand back a flat float32 vector. A model whose head
    emits (n,1) instead of (n,) otherwise blows up the OOF assignment and, since
    that assignment sits outside the per-member guard, kills the whole run."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 1:
        a = a.reshape(-1)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    return a


# ----------------------------------------------------------------------------
# Target encoding on integer codes, mathematically identical to sklearn's
# TargetEncoder (verified by test_v8_parity.py).
# ----------------------------------------------------------------------------
def te_map(codes, y, n_cat, smooth, gmean, var_y):
    """Return per-category encodings. codes: int array >=0; y: 0/1."""
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
    enc = np.where(seen, enc, gmean)
    return enc.astype(np.float32)


def te_crossfit(codes_tr, y_tr, n_cat, smooth, gmean, var_y, seed):
    """Encode training rows using only the OTHER inner folds (leak-safe),
    mirroring TargetEncoder(cv=5)."""
    out = np.full(len(codes_tr), gmean, dtype=np.float32)
    skf = StratifiedKFold(n_splits=TE_INNER_FOLDS, shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(codes_tr)), y_tr):
        mp = te_map(codes_tr[a], y_tr[a], n_cat, smooth, gmean, var_y)
        out[b] = mp[codes_tr[b]]
    return out


# ----------------------------------------------------------------------------
# External source dataset (target-mean anchors). Guards reject the decoy CSVs
# (fullypreprocessed/onehotenc carry the same columns but have an `id` column).
# ----------------------------------------------------------------------------
ORIGINAL_CSV = None


def load_original(train, skip_paths):
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    skip = {os.path.abspath(p) for p in skip_paths if p}
    roots = []
    if ORIGINAL_CSV and os.path.isfile(ORIGINAL_CSV):
        roots = [os.path.dirname(os.path.abspath(ORIGINAL_CSV))]
    else:
        roots = [r for r in (KAGGLE_INPUT_DIR, "/content", os.getcwd(), ".") if os.path.isdir(r)]
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
        log(f"[original] loaded {len(df)} real rows from {os.path.basename(p)}")
        return df[feats + [TARGET]].dropna(subset=[TARGET]).copy()
    log("[original] source dataset not found — anchors skipped")
    return None


# ----------------------------------------------------------------------------
# Target-free features on the combined train+test frame, then int32 codes for
# every column we will target-encode.
# ----------------------------------------------------------------------------
def build_features(train, test, orig):
    ntr = len(train)
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    comb = pd.concat([train[feats], test[feats]], ignore_index=True)
    for c in feats:
        comb[c] = comb[c].astype("float64") if c in NUMERIC else comb[c].astype(str).str.strip().str.lower()

    digit_cols = []
    for c in NUMERIC:
        v = pd.to_numeric(comb[c], errors="coerce").fillna(0.0)
        for k in DIGIT_KS:
            name = f"{c}_digit{k}"
            comb[name] = ((v // (10.0 ** k)) % 10).astype("int8")
            digit_cols.append(name)

    inc = pd.to_numeric(comb["Annual_Income_USD"], errors="coerce").fillna(0.0)
    comm = pd.to_numeric(comb["Daily_Commute_km"], errors="coerce").fillna(0.0)
    env = pd.to_numeric(comb["Environmental_Concern_Level"], errors="coerce").fillna(0.0)
    sub_flag = (comb["Subsidy_Available"] == "yes").astype("int8")
    anx = comb["Range_Anxiety_Level"]

    comb["income_exact_int"] = np.floor(inc).astype(np.int64)
    comb["income100_floor"] = np.floor(inc / 100.0).astype(np.int64)
    comb["income1000_floor"] = np.floor(inc / 1000.0).astype(np.int64)
    comb["commute_integer"] = np.floor(comm).astype(np.int64)
    smooth_keys = ["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"]

    comb["is_30k_spike"] = (inc == 30000.0).astype("int8")
    comb["is_millionaire_cliff"] = (inc >= 170537.0).astype("int8")
    comb["is_dead_zone"] = ((inc >= 38174.0) & (inc <= 41384.0)).astype("int8")
    comb["is_env_hater"] = (env == 1).astype("int8")
    comb["is_comm_5"] = (comm.round(1) == 5.0).astype("int8")
    comb["is_comm_ge83"] = (comm >= COMMUTE_NO_BUYERS_FROM).astype("int8")
    flags = ["is_30k_spike", "is_millionaire_cliff", "is_dead_zone", "is_env_hater",
             "is_comm_5", "is_comm_ge83"]

    tot_ch = (pd.to_numeric(comb["Charging_Stations_Near_Home"], errors="coerce").fillna(0.0)
              + pd.to_numeric(comb["Charging_Stations_Near_Work"], errors="coerce").fillna(0.0))
    comb["total_charging"] = tot_ch
    comb["env_x_subsidy"] = env * sub_flag
    comb["income_x_subsidy"] = (inc / 100000.0) * sub_flag
    comb["recipe_score"] = (1.2 * (inc / 100000.0) + 0.6 * env + 2.0 * sub_flag
                            - 1.0 * (anx == "medium").astype("int8")
                            - 3.0 * (anx == "high").astype("int8"))

    anchors = []
    if orig is not None and ORIG_AS_ANCHORS:
        y_orig = encode_target(orig[TARGET]).astype(np.float64)
        gmean = float(y_orig.mean())
        for c in CATEGORICAL + NUMERIC:
            if c not in orig.columns or c not in comb.columns:
                continue
            ok = orig[c]
            okey = np.floor(pd.to_numeric(ok, errors="coerce").fillna(0.0)).astype(np.int64) \
                if c in NUMERIC else ok.astype(str).str.strip().str.lower()
            stats = pd.Series(y_orig).groupby(okey.to_numpy()).mean()
            q = comb[c] if comb[c].dtype != object else comb[c]
            comb[f"{c}_org_mean"] = q.map(stats).fillna(gmean).astype("float32")
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

    # frequency encoding over the combined population (no labels involved)
    freq_cols = []
    for c in CATEGORICAL + num_as_str + smooth_keys:
        if c not in comb.columns:
            continue
        fm = comb[c].value_counts(normalize=True).to_dict()
        comb[f"{c}_fe"] = comb[c].map(fm).fillna(0.0).astype("float32")
        freq_cols.append(f"{c}_fe")

    # columns to target-encode: strings + stringified numerics + digits + bins
    te_src = sorted(set(CATEGORICAL) | set(num_as_str) | set(digit_cols) | set(smooth_keys))
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

    # int32 codes for every TE source column (shared vocabulary over train+test)
    codes = {}
    ncats = {}
    for c in te_src:
        cat = pd.Categorical(comb[c].astype(str))
        codes[c] = np.asarray(cat.codes, dtype=np.int32)
        ncats[c] = int(len(cat.categories))

    log(f"[feat] raw {len(raw_num)} | TE sources {len(te_src)} "
        f"(-> {len(te_src)*len(TE_SMOOTHS)} TE features) | anchors {len(anchors)} "
        f"| freq {len(freq_cols)} | dropped {len(dropped)}")

    Xtr = {c: codes[c][:ntr] for c in te_src}
    Xte = {c: codes[c][ntr:] for c in te_src}
    Rtr = comb.iloc[:ntr][raw_num].to_numpy(np.float32)
    Rte = comb.iloc[ntr:][raw_num].to_numpy(np.float32)
    return (Rtr, Rte, Xtr, Xte, te_src, ncats, raw_num)


# ----------------------------------------------------------------------------
# Per-fold design matrix. Encoders are fit on training rows ONLY.
# ----------------------------------------------------------------------------
def fold_matrix(Rtr, Xtr_codes, y, tr_idx, va_idx, te_src, ncats, Rte, Xte_codes):
    ytr = y[tr_idx]
    gmean = float(ytr.mean())
    var_y = float(np.var(ytr.astype(np.float64)))
    tr_blocks = [Rtr[tr_idx]]
    va_blocks = [Rtr[va_idx]]
    te_blocks = [Rte]
    for c in te_src:
        ctr = Xtr_codes[c][tr_idx]
        cva = Xtr_codes[c][va_idx]
        cte = Xte_codes[c]
        nc = ncats[c] + 1
        for smooth in TE_SMOOTHS:
            full = te_map(ctr, ytr, nc, smooth, gmean, var_y)
            tr_blocks.append(te_crossfit(ctr, ytr, nc, smooth, gmean, var_y,
                                         RANDOM_STATE)[:, None])
            va_blocks.append(full[cva][:, None])
            te_blocks.append(full[cte][:, None])
    return (np.hstack(tr_blocks), np.hstack(va_blocks), np.hstack(te_blocks))


# ----------------------------------------------------------------------------
# Models.
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


def run_nn(Xa, ya, Xb, yb, Xc, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(42); np.random.seed(42)
    mu = Xa.mean(axis=0); sd = Xa.std(axis=0); sd[sd == 0] = 1.0
    Xa = ((Xa - mu) / sd).astype(np.float32)
    Xb = ((Xb - mu) / sd).astype(np.float32)
    Xc = ((Xc - mu) / sd).astype(np.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xa.shape[1], 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xa), torch.tensor(ya.astype(np.float32))),
                        batch_size=BATCH, shuffle=True, drop_last=True)
    vb = torch.tensor(Xb, dtype=torch.float32).to(device)
    tb = torch.tensor(Xc, dtype=torch.float32).to(device)
    best, bva, bte, bad = -1.0, None, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        for xb, yb_ in loader:
            xb, yb_ = xb.to(device), yb_.to(device)
            opt.zero_grad()
            crit(net(xb).squeeze(-1), yb_).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            # the head is Linear(64,1) -> (n,1); squeeze to (n,) or the OOF write fails
            vp = torch.sigmoid(net(vb)).squeeze(-1).cpu().numpy()
            a = roc_auc_score(yb, vp)
        if a > best:
            best = a
            bva = vp
            with torch.no_grad():
                bte = torch.sigmoid(net(tb)).squeeze(-1).cpu().numpy()
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    del net, loader, vb, tb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _as_1d(bva), _as_1d(bte)


# ----------------------------------------------------------------------------
# Bands: epsilon-ordered blocks, no forced ties.
# ----------------------------------------------------------------------------
def band_masks(income, commute):
    buy = np.zeros(len(income), dtype=bool)
    for lo, hi in BUYER_BANDS:
        buy |= (income >= lo) & (income <= hi)
    nobuy = np.zeros(len(income), dtype=bool)
    for lo, hi in NON_BUYER_BANDS:
        nobuy |= (income >= lo) & (income <= hi)
    nobuy |= commute >= COMMUTE_NO_BUYERS_FROM
    nobuy &= ~buy
    return buy, nobuy


def apply_bands(probs, income, commute):
    probs = np.asarray(probs, dtype=np.float64)
    buy, nobuy = band_masks(income, commute)
    priority = np.ones(len(probs), dtype=np.int8)
    priority[nobuy] = 0
    priority[buy] = 2
    order = np.lexsort((np.arange(len(probs)), probs, priority))
    out = np.empty(len(probs), dtype=np.float64)
    out[order] = np.linspace(1e-6, 1.0 - 1e-6, len(probs))
    log(f"[bands] buyer block {int(buy.sum())}, non-buyer block {int(nobuy.sum())}, "
        f"intra-block order kept (no forced ties)")
    return out


# ----------------------------------------------------------------------------
# Checkpoints (per model, per fold).
# ----------------------------------------------------------------------------
_RUN = {"dir": None, "out_dir": "."}


def init_ckpt(out_dir, sig):
    tag = f"{CKPT_DIRNAME}_{hashlib.md5(sig.encode()).hexdigest()[:8]}"
    _RUN["out_dir"] = out_dir
    _RUN["dir"] = os.path.join(out_dir, tag)
    if CKPT_INPUT_DIR and os.path.isdir(CKPT_INPUT_DIR):
        read = CKPT_INPUT_DIR
    else:
        read = _RUN["dir"]
        if os.path.isdir(KAGGLE_INPUT_DIR):
            for dp, _dd, _fs in os.walk(KAGGLE_INPUT_DIR):
                if os.path.basename(dp) == tag:
                    read = dp
                    break
    _RUN["read"] = read
    os.makedirs(_RUN["dir"], exist_ok=True)
    log(f"[ckpt] {tag} resume={'on' if RESUME else 'off'} reading={read}")


def ckpt_load(model, fold):
    if not RESUME:
        return None
    d = _RUN.get("read")
    fv, ft = os.path.join(d, f"{model}_f{fold}_val.npy"), os.path.join(d, f"{model}_f{fold}_test.npy")
    if not (os.path.isfile(fv) and os.path.isfile(ft)):
        return None
    try:
        return np.load(fv), np.load(ft)
    except Exception:
        return None


def ckpt_save(model, fold, val, test):
    try:
        np.save(os.path.join(_RUN["dir"], f"{model}_f{fold}_val.npy"), val)
        np.save(os.path.join(_RUN["dir"], f"{model}_f{fold}_test.npy"), test)
    except Exception as exc:
        log(f"[ckpt] save {model} f{fold} failed: {exc!r}")


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    log(f"train: {train_path}\ntest : {test_path}")
    n_gpus = gpu_count()
    log(f"GPUs={n_gpus} CPUs={multiprocessing.cpu_count()}")

    have = {"lgbm_ref": ensure_import("lightgbm"), "xgb_a": ensure_import("xgboost")}
    if not have["lgbm_ref"]:
        raise RuntimeError("LightGBM unavailable — v8's primary model cannot run.")
    device = None
    if NN_USE and ensure_import("torch"):
        import torch
        have["nn"] = True
        device = torch.device("cuda" if (torch.cuda.is_available() and n_gpus > 0) else "cpu")
    else:
        have["nn"] = False
    models = [m for m in ("lgbm_ref", "xgb_a", "nn") if have.get(m)]
    log(f"[env] members: {models}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = load_original(train, [train_path, test_path, sample_path])
    if ORIG_AS_ROWS and orig is not None:
        train = pd.concat([train, orig], ignore_index=True)

    y = encode_target(train[TARGET])
    log(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    Rtr, Rte, Xtr_codes, Xte_codes, te_src, ncats, raw_num = build_features(train, test, orig)
    n_feat = len(raw_num) + len(te_src) * len(TE_SMOOTHS)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    init_ckpt(out_dir, f"{n_feat}|{len(train)}|{N_SPLITS}|{RANDOM_STATE}|{TE_SMOOTHS}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(train)), y))

    oof = {m: np.full(len(y), np.nan, dtype=np.float32) for m in models}
    tepred = {m: np.zeros(len(Rte), dtype=np.float32) for m in models}
    failed = set()

    for f, (tr_idx, va_idx) in enumerate(folds):
        tf = time.perf_counter()
        Xa, Xb, Xc = fold_matrix(Rtr, Xtr_codes, y, tr_idx, va_idx, te_src, ncats, Rte, Xte_codes)
        log(f"[fold {f}] design {Xa.shape} built in {time.perf_counter()-tf:.1f}s")
        for m in models:
            if m in failed:
                continue
            cached = ckpt_load(m, f)
            if cached is not None:
                vp, tp = _as_1d(cached[0]), _as_1d(cached[1])
                log(f"    [{m} f{f}] resumed")
            else:
                try:
                    if m == "lgbm_ref":
                        vp, tp = run_lightgbm(Xa, y[tr_idx], Xb, y[va_idx], Xc)
                    elif m == "xgb_a":
                        vp, tp = run_xgboost(Xa, y[tr_idx], Xb, y[va_idx], Xc,
                                             XGB_USE_GPU and n_gpus > 0)
                    else:
                        vp, tp = run_nn(Xa, y[tr_idx], Xb, y[va_idx], Xc, device)
                    vp, tp = _as_1d(vp), _as_1d(tp)
                    if len(vp) != len(va_idx):
                        raise ValueError(f"{m} returned {len(vp)} val preds, expected {len(va_idx)}")
                    if len(tp) != len(Rte):
                        raise ValueError(f"{m} returned {len(tp)} test preds, expected {len(Rte)}")
                except Exception as exc:
                    log(f"    !! {m} fold {f} FAILED ({exc!r}) -> member dropped")
                    failed.add(m)
                    continue
                ckpt_save(m, f, vp, tp)
            oof[m][va_idx] = vp
            tepred[m] += tp / N_SPLITS
            log(f"    [{m} f{f}] auc={roc_auc_score(y[va_idx], vp):.6f}")
        del Xa, Xb, Xc
        gc.collect()

    live = [m for m in models if m not in failed and np.isfinite(oof[m]).all()]
    if not live:
        raise RuntimeError("All members failed — nothing to submit.")

    log("\n" + "=" * 74)
    for m in live:
        a = roc_auc_score(y, oof[m])
        extra = ""
        if m == "lgbm_ref":
            extra = f"  [{'PASS' if a >= CV_GATE else 'BELOW'} CV gate {CV_GATE}]"
        log(f"{m:9s} OOF AUC = {a:.6f}{extra}")
    wsum = sum(BLEND_WEIGHTS[m] for m in live)
    blend_oof = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(oof[m]) for m in live], axis=0)
    blend_te = np.sum([BLEND_WEIGHTS[m] / wsum * rank_pct(tepred[m]) for m in live], axis=0)
    log(f"{'BLEND':9s} OOF AUC = {roc_auc_score(y, blend_oof):.6f}   weights "
        f"{ {m: round(BLEND_WEIGHTS[m]/wsum, 3) for m in live} }")
    log("=" * 74)

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
        log(f"wrote {os.path.basename(path)} rows={len(df)} unique={df[TARGET].nunique()} "
            f"range {df[TARGET].min():.6f}..{df[TARGET].max():.6f}")

    if "lgbm_ref" in live:
        write_sub(f"submission_{VERSION}_single.csv", tepred["lgbm_ref"])
    write_sub(f"submission_{VERSION}.csv", blend_te)
    for m in live:
        np.save(os.path.join(_RUN["out_dir"], f"v8_{m}_oof.npy"), oof[m])
        np.save(os.path.join(_RUN["out_dir"], f"v8_{m}_test.npy"), tepred[m])
    log(f"Runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
