"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v5)
Pseudo-labeling (self-training) build. v1/v2/v3 all plateaued at LB ~0.9454,
and v3 showed the low-smoothing exact-value TE carries real signal (so v5
reverts to smoothing 2/10). To break the plateau, v5 adds a self-training
round: train the ensemble, then fold the *confident* test predictions back in
as extra training rows and retrain — leveraging the 286k unlabeled test rows.

  * Phase A: normal 10-fold CV of the 5-model GBDT ensemble (TE 2/10).
  * Phase B: retrain on train + confident pseudo-labeled test rows.
  * Final test = rank-blend of Phase A + Phase B.
  * Both T4s via CatBoost multi-GPU. Saves submission_v5.csv + v5_oof/test.npy.

Judge by the LB, not the (optimistic) OOF. Run on Kaggle (Save & Run All).
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings
import tempfile
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import ndtr

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v5"
N_SPLITS = 10
RANDOM_STATE = 42
# v3 (smoothing 10/50) scored LOWER on LB than v1/v2 (2/10) -> the low-smoothing
# exact-value TE carries REAL signal, so v5 reverts to 2/10.
TE_SMOOTHINGS = (2.0, 10.0)

# Pseudo-labeling (self-training): after a first pass, add the test rows the
# ensemble is very confident about back into training, then retrain. Uses the
# 286k unlabeled test rows — the classic way to break a tabular plateau.
PL_LO = 0.02          # pseudo-label "No" if blended prob < this
PL_HI = 0.98          # pseudo-label "Yes" if blended prob > this
PL_PHASES = 2         # 1 = no pseudo-labeling (sanity), 2 = one self-train round

USE_GPU = None            # None = auto-detect
CAT_MULTI_GPU = True      # train CatBoost across both GPUs ('0:1') — works in notebooks
# Fold-level multiprocessing does NOT work when Kaggle runs the .py inside a
# notebook kernel (spawn workers can't re-import functions from __main__).
# Keep this False; both T4s are still used via CatBoost multi-GPU above.
USE_PARALLEL_FOLDS = False

# Each spec: (family_key, model_kind, seeds). Seeds trimmed vs v3 because v5
# runs TWO full phases (pseudo-labeling), so we keep total runtime ~6h.
MODEL_SPECS = [
    ("cat",     "catboost", (0,)),
    ("xgb_a",   "xgboost",  (42,)),
    ("xgb_b",   "xgboost",  (13,)),
    ("lgbm_a",  "lightgbm", (42, 7)),
    ("lgbm_b",  "lightgbm", (13,)),
]

GPU_KINDS = {"catboost", "xgboost"}

# ----------------------------------------------------------------------------
# Path detection.
# ----------------------------------------------------------------------------
def find_data_dir():
    candidates = ["/kaggle/input/datasets/mitanshkanani/dataset-for-experimentation-final"]
    if os.path.isdir("/kaggle/input"):
        for root in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", root)
            if os.path.isdir(base):
                candidates.append(base)
                for sub in os.listdir(base):
                    subp = os.path.join(base, sub)
                    if os.path.isdir(subp):
                        candidates.append(subp)
    candidates += ["data", ".", os.getcwd()]

    def pick(names):
        for d in candidates:
            for n in names:
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    return p
        return None

    train_path = pick(["train.csv"])
    test_path = pick(["test.csv"])
    sample_path = pick(["sample_submission.csv"])
    if train_path is None or test_path is None:
        raise FileNotFoundError("Could not locate train.csv / test.csv.")
    return train_path, test_path, sample_path


def gpu_count() -> int:
    if USE_GPU is False:
        return 0
    try:
        import torch
        n = torch.cuda.device_count()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    try:
        import subprocess
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


# ----------------------------------------------------------------------------
# Feature engineering (target-free).
# ----------------------------------------------------------------------------
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    income = pd.to_numeric(df["Annual_Income_USD"], errors="coerce").fillna(0.0)
    commute = pd.to_numeric(df["Daily_Commute_km"], errors="coerce").fillna(0.0)
    age = pd.to_numeric(df["Age"], errors="coerce").fillna(0.0)
    cars = pd.to_numeric(df["Number_of_Cars_Owned"], errors="coerce").fillna(0.0)
    csh = pd.to_numeric(df["Charging_Stations_Near_Home"], errors="coerce").fillna(0.0)
    csw = pd.to_numeric(df["Charging_Stations_Near_Work"], errors="coerce").fillna(0.0)
    env = pd.to_numeric(df["Environmental_Concern_Level"], errors="coerce").fillna(0.0)

    x["Age"] = age
    x["Annual_Income_USD"] = income
    x["Daily_Commute_km"] = commute
    x["Number_of_Cars_Owned"] = cars
    x["Charging_Stations_Near_Home"] = csh
    x["Charging_Stations_Near_Work"] = csw
    x["Environmental_Concern_Level"] = env

    x["Gender"] = df["Gender"].astype(str)
    x["City_Type"] = df["City_Type"].astype(str)
    x["Current_Car_Type"] = df["Current_Car_Type"].astype(str)
    x["Home_Charging_Possible"] = df["Home_Charging_Possible"].astype(str)
    x["Subsidy_Available"] = df["Subsidy_Available"].astype(str)
    x["Range_Anxiety_Level"] = df["Range_Anxiety_Level"].astype(str)

    anxiety_map = {"low": 0, "medium": 1, "high": 2}
    anx = df["Range_Anxiety_Level"].astype(str).str.strip().str.lower()
    x["anxiety_ord"] = anx.map(anxiety_map).fillna(-1).astype(np.int8)
    x["subsidy_flag"] = (df["Subsidy_Available"].astype(str).str.strip().str.lower() == "yes").astype(np.int8)
    x["home_charge_flag"] = (df["Home_Charging_Possible"].astype(str).str.strip().str.lower() == "yes").astype(np.int8)

    inc_i = np.rint(income.to_numpy(dtype=np.float64)).astype(np.int64)
    for div in (1, 10, 100, 1_000, 10_000, 100_000):
        x[f"inc_digit_{div}"] = ((inc_i // div) % 10).astype(np.int8)
    x["inc_last2"] = (inc_i % 100).astype(np.int16)
    x["inc_last3"] = (inc_i % 1_000).astype(np.int16)
    x["inc_mod_5000"] = (inc_i % 5_000).astype(np.int16)
    x["inc_b_1k"] = (inc_i // 1_000).astype(np.int32)
    x["inc_b_5k"] = (inc_i // 5_000).astype(np.int32)
    x["inc_b_10k"] = (inc_i // 10_000).astype(np.int32)

    comm = np.rint(commute.to_numpy(dtype=np.float64) * 10).astype(np.int64)
    x["comm_b1"] = (comm // 10).astype(np.int16)
    x["comm_b5"] = (comm // 50).astype(np.int16)

    x["income_per_car"] = income / (cars + 1.0)
    x["income_per_age"] = income / (age + 1.0)
    x["total_charging"] = csh + csw
    x["charging_balance"] = csh - csw
    x["commute_per_charging"] = commute / (csh + csw + 1.0)
    x["env_x_subsidy"] = env * x["subsidy_flag"]
    x["env_x_homecharge"] = env * x["home_charge_flag"]
    x["income_x_env"] = (income / 100_000.0) * env
    x["anxiety_penalty"] = -x["anxiety_ord"].astype(np.float64)
    x["age_sq"] = age * age
    x["log_income"] = np.log1p(income)

    score = (
        1.2 * (income / 100_000.0)
        + 0.6 * env
        + 2.0 * x["subsidy_flag"]
        - 1.0 * (anx == "medium").astype(np.int8)
        - 3.0 * (anx == "high").astype(np.int8)
    )
    x["recipe_score"] = score
    x["recipe_prob"] = ndtr((score - 5.5).to_numpy(dtype=np.float64))
    return x


# ----------------------------------------------------------------------------
# Leakage-safe exact-value target encoding (shares the model CV folds).
# ----------------------------------------------------------------------------
def _te_key(df, cols):
    if len(cols) == 1:
        c = cols[0]
        if c == "Annual_Income_USD":
            arr = np.rint(pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)).astype(np.int64)
            return pd.Series(arr, index=df.index).astype(str)
        if c == "Daily_Commute_km":
            arr = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)
            return pd.Series([f"{v:.1f}" for v in arr], index=df.index)
        return df[c].astype(str)
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def _fit_te(keys, y, prior, smoothing):
    frame = pd.DataFrame({"k": keys.to_numpy(), "t": y})
    stats = frame.groupby("k", observed=True)["t"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


TE_KEYS = {
    "te_income": ["Annual_Income_USD"],
    "te_commute": ["Daily_Commute_km"],
    "te_inc_subsidy": ["Annual_Income_USD", "Subsidy_Available"],
    "te_inc_env": ["Annual_Income_USD", "Environmental_Concern_Level"],
    "te_inc_city": ["Annual_Income_USD", "City_Type"],
    "te_inc_car": ["Annual_Income_USD", "Current_Car_Type"],
    "te_inc_gender": ["Annual_Income_USD", "Gender"],
    "te_inc_anx": ["Annual_Income_USD", "Range_Anxiety_Level"],
    "te_comm_homecharge": ["Daily_Commute_km", "Home_Charging_Possible"],
    "te_comm_city": ["Daily_Commute_km", "City_Type"],
    "te_incb": ["inc_b_5k"],
}


def add_target_encodings(train, test, y, fold_ids):
    for name, cols in TE_KEYS.items():
        tr_keys = _te_key(train, cols)
        te_keys = _te_key(test, cols)
        for m in TE_SMOOTHINGS:
            col = f"{name}_m{int(m)}"
            enc_tr = np.full(len(train), np.nan, dtype=np.float32)
            for f in range(N_SPLITS):
                tr_mask = fold_ids != f
                va_mask = fold_ids == f
                prior_f = float(y[tr_mask].mean())
                mapping = _fit_te(tr_keys[tr_mask], y[tr_mask], prior_f, m)
                enc_tr[va_mask] = tr_keys[va_mask].map(mapping).fillna(prior_f).to_numpy(np.float32)
            train[col] = enc_tr
            full_map = _fit_te(tr_keys, y, float(y.mean()), m)
            test[col] = te_keys.map(full_map).fillna(float(y.mean())).to_numpy(np.float32)
        counts = tr_keys.value_counts()
        train[f"{name}_cnt"] = tr_keys.map(counts).fillna(0).to_numpy(np.float32)
        test[f"{name}_cnt"] = te_keys.map(counts).fillna(0).to_numpy(np.float32)
    return train, test


# ----------------------------------------------------------------------------
# Model factories.
# ----------------------------------------------------------------------------
def _cat_cols(X):
    return [c for c in X.columns
            if X[c].dtype == object or pd.api.types.is_string_dtype(X[c].dtype)
            or isinstance(X[c].dtype, pd.CategoricalDtype)]


def _cat_devices(use_gpu):
    if not use_gpu:
        return "0"
    if CAT_MULTI_GPU:
        return ":".join(str(i) for i in range(max(1, gpu_count())))
    return "0"


def _cast_cat(Xtr, Xva, Xte, cat_features):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in cat_features:
        cats = pd.Index(pd.concat([Xtr[c], Xva[c], Xte[c]]).astype(str).unique())
        dt = pd.CategoricalDtype(categories=cats, ordered=False)
        Xtr[c] = Xtr[c].astype(str).astype(dt)
        Xva[c] = Xva[c].astype(str).astype(dt)
        Xte[c] = Xte[c].astype(str).astype(dt)
    return Xtr, Xva, Xte


def train_catboost(Xtr, ytr, Xva, yva, Xte, seed, use_gpu, cfg):
    from catboost import CatBoostClassifier, Pool
    cat_features = _cat_cols(Xtr)
    model = CatBoostClassifier(
        iterations=cfg.get("iterations", 2000),
        learning_rate=cfg.get("lr", 0.05),
        depth=cfg.get("depth", 8),
        l2_leaf_reg=cfg.get("l2", 5.0),
        bagging_temperature=0.6,
        random_strength=0.7,
        border_count=254,
        eval_metric="AUC",
        random_seed=seed,
        logging_level="Silent",
        allow_writing_files=False,
        task_type="GPU" if use_gpu else "CPU",
        devices=_cat_devices(use_gpu),
    )
    model.fit(
        Pool(Xtr, ytr, cat_features=cat_features),
        eval_set=Pool(Xva, yva, cat_features=cat_features),
        early_stopping_rounds=250,
    )
    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    del model
    gc.collect()
    return va_pred, te_pred


def train_xgboost(Xtr, ytr, Xva, yva, Xte, seed, use_gpu, cfg):
    import xgboost as xgb
    cat_features = _cat_cols(Xtr)
    Xtr, Xva, Xte = _cast_cat(Xtr, Xva, Xte, cat_features)
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=6000, learning_rate=cfg.get("lr", 0.03),
        max_depth=cfg.get("depth", 6), min_child_weight=cfg.get("mcw", 12),
        subsample=0.85, colsample_bytree=cfg.get("csb", 0.85),
        reg_lambda=cfg.get("lambda", 3.0), reg_alpha=0.0, gamma=0.0,
        max_cat_to_onehot=8, tree_method="hist",
        device="cuda" if use_gpu else "cpu", enable_categorical=True,
        early_stopping_rounds=250, random_state=seed, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=0)
    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    del model
    gc.collect()
    return va_pred, te_pred


def train_lightgbm(Xtr, ytr, Xva, yva, Xte, seed, use_gpu, cfg):
    import lightgbm as lgb
    cat_features = _cat_cols(Xtr)
    Xtr, Xva, Xte = _cast_cat(Xtr, Xva, Xte, cat_features)
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        n_estimators=6000, learning_rate=cfg.get("lr", 0.03),
        num_leaves=cfg.get("num_leaves", 63), max_depth=-1,
        min_child_samples=cfg.get("mcs", 40), subsample=0.85, subsample_freq=1,
        colsample_bytree=cfg.get("csb", 0.85), reg_lambda=cfg.get("lambda", 3.0),
        reg_alpha=0.0, n_jobs=-1, random_state=seed, verbose=-1,
    )
    model.fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
        categorical_feature=cat_features if cat_features else "auto",
        callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)],
    )
    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    del model
    gc.collect()
    return va_pred, te_pred


FACTORIES = {"catboost": train_catboost, "xgboost": train_xgboost, "lightgbm": train_lightgbm}

# Per-spec hyperparameters.
CFG = {
    "cat":    {"iterations": 2000, "lr": 0.05, "depth": 8, "l2": 5.0},
    "xgb_a":  {"lr": 0.03, "depth": 6, "mcw": 12, "csb": 0.85, "lambda": 3.0},
    "xgb_b":  {"lr": 0.025, "depth": 8, "mcw": 25, "csb": 0.7, "lambda": 6.0},
    "lgbm_a": {"lr": 0.03, "num_leaves": 63, "mcs": 40, "csb": 0.85, "lambda": 3.0},
    "lgbm_b": {"lr": 0.025, "num_leaves": 127, "mcs": 60, "csb": 0.75, "lambda": 5.0},
}


# ----------------------------------------------------------------------------
# Sequential CV runner (one spec).
# ----------------------------------------------------------------------------
def run_spec_sequential(key, kind, seeds, X, Xte, y, fold_ids, use_gpu, n_real=None):
    factory = FACTORIES[kind]
    cfg = CFG[key]
    nr = n_real if n_real is not None else len(y)
    oof_sum = np.zeros(nr)
    te_sum = np.zeros(len(Xte))
    for seed in seeds:
        oof = np.full(nr, np.nan)
        tes = []
        for f in range(N_SPLITS):
            tr = fold_ids != f
            va = fold_ids == f
            vp, tp = factory(
                X[tr].reset_index(drop=True), y[tr],
                X[va].reset_index(drop=True), y[va],
                Xte, seed + f, use_gpu, cfg,
            )
            va_idx = np.flatnonzero(va)  # all < nr: pseudo rows have fold -1
            oof[va_idx] = vp
            tes.append(tp)
            print(f"    [{key} seed={seed} fold={f}] auc={roc_auc_score(y[va], vp):.6f}", flush=True)
        print(f"  [{key}] seed={seed} OOF AUC = {roc_auc_score(y[:nr], oof):.8f}", flush=True)
        oof_sum += oof
        te_sum += np.mean(np.vstack(tes), axis=0)
        del oof, tes
        gc.collect()
    oof_avg = oof_sum / len(seeds)
    te_avg = te_sum / len(seeds)
    return oof_avg, te_avg, roc_auc_score(y[:nr], oof_avg)


# ----------------------------------------------------------------------------
# Parallel GPU fold runner (used for catboost + xgboost specs).
# Each worker pins to one GPU and only returns numpy arrays (safe to pickle).
# ----------------------------------------------------------------------------
_W = {}


def _pin_init(gpu_id, paths):
    import os as _os
    if gpu_id is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _W["X"] = pd.read_pickle(paths["x"])
    _W["Xte"] = pd.read_pickle(paths["xte"])
    _W["y"] = np.load(paths["y"])
    _W["fold"] = np.load(paths["fold"])


def _pin_task(job):
    key, seed, fold = job
    X, Xte, y, fid = _W["X"], _W["Xte"], _W["y"], _W["fold"]
    kind = dict((k, kd) for k, kd, _ in MODEL_SPECS)[key]
    tr = fid != fold
    va = fid == fold
    vp, tp = FACTORIES[kind](
        X[tr].reset_index(drop=True), y[tr],
        X[va].reset_index(drop=True), y[va],
        Xte, seed + fold, True, CFG[key],
    )
    return key, seed, int(fold), np.flatnonzero(va).astype(np.int32), \
        np.asarray(vp, dtype=np.float32), np.asarray(tp, dtype=np.float32)


def run_specs_parallel(specs, X, Xte, y, fold_ids, n_gpus):
    tmpdir = tempfile.mkdtemp(prefix="evgpu_")
    paths = {
        "x": os.path.join(tmpdir, "xtr.pkl"),
        "xte": os.path.join(tmpdir, "xte.pkl"),
        "y": os.path.join(tmpdir, "y.npy"),
        "fold": os.path.join(tmpdir, "fold.npy"),
    }
    X.to_pickle(paths["x"])
    Xte.to_pickle(paths["xte"])
    np.save(paths["y"], y)
    np.save(paths["fold"], fold_ids)

    jobs = [(k, s, f) for (k, _kd, seeds) in specs for s in seeds for f in range(N_SPLITS)]
    print(f"[parallel] {len(jobs)} GPU fold-jobs across {n_gpus} GPUs", flush=True)

    ctx = mp.get_context("spawn")
    execs = [
        ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_pin_init, initargs=(g, paths))
        for g in range(n_gpus)
    ]
    futures = [execs[i % n_gpus].submit(_pin_task, job) for i, job in enumerate(jobs)]

    collected = defaultdict(list)
    done = 0
    try:
        for fut in as_completed(futures):
            key, seed, fold, vidx, vp, tp = fut.result()
            collected[(key, seed)].append((fold, vidx, vp, tp))
            done += 1
            print(f"  [parallel] {done}/{len(jobs)} {key} seed={seed} fold={fold}", flush=True)
    finally:
        for ex in execs:
            ex.shutdown()

    out = {}
    for (key, _kd, seeds) in specs:
        oof_sum = np.zeros(len(y))
        te_sum = np.zeros(len(Xte))
        for s in seeds:
            oof = np.full(len(y), np.nan)
            tes = []
            for fold, vidx, vp, tp in collected[(key, s)]:
                oof[vidx] = vp
                tes.append(tp)
            oof_sum += oof
            te_sum += np.mean(np.vstack(tes), axis=0)
            print(f"  [{key}] seed={s} OOF AUC = {roc_auc_score(y, oof):.8f}", flush=True)
        oof_avg = oof_sum / len(seeds)
        te_avg = te_sum / len(seeds)
        out[key] = (oof_avg, te_avg, roc_auc_score(y, oof_avg))
        print(f"== {key} multi-seed OOF AUC = {out[key][2]:.8f} ==", flush=True)

    for p in paths.values():
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    return out


# ----------------------------------------------------------------------------
# AUC-optimized non-negative blend weights.
# ----------------------------------------------------------------------------
def rank_pct(a):
    return pd.Series(a).rank(method="average", pct=True).to_numpy()


def optimize_weights(oof_dict, y, n_iter=1500, seed=0):
    names = list(oof_dict)
    R = np.vstack([rank_pct(oof_dict[n]) for n in names])  # (k, n)
    k = len(names)
    rng = np.random.default_rng(seed)
    # Subsample rows for fast search; evaluate final weights on full data.
    sub = rng.choice(len(y), size=min(250_000, len(y)), replace=False)
    ys = y[sub]
    Rs = R[:, sub]

    def auc_w(w, mat, yy):
        return roc_auc_score(yy, w @ mat)

    best_w = np.ones(k) / k
    best_auc = auc_w(best_w, Rs, ys)
    for _ in range(n_iter):
        w = rng.random(k)
        w /= w.sum()
        a = auc_w(w, Rs, ys)
        if a > best_auc:
            best_auc, best_w = a, w
    improved = True
    while improved:
        improved = False
        for i in range(k):
            for d in (0.08, 0.04, -0.04, -0.08):
                w = best_w.copy()
                w[i] += d
                if (w >= 0).all():
                    w /= w.sum()
                    a = auc_w(w, Rs, ys)
                    if a > best_auc + 1e-9:
                        best_auc, best_w, improved = a, w, True
    final_auc = auc_w(best_w, R, y)
    return dict(zip(names, best_w)), final_auc


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    print(f"train: {train_path}\ntest : {test_path}")

    n_gpus = gpu_count()
    use_gpu = n_gpus > 0
    parallel = USE_PARALLEL_FOLDS and n_gpus >= 2
    print(f"GPUs={n_gpus} use_gpu={use_gpu} parallel={parallel} cat_multi_gpu={CAT_MULTI_GPU}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = encode_target(train[TARGET])
    print(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    Xtr = build_base_features(train)
    Xte = build_base_features(test)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va_idx) in enumerate(skf.split(Xtr, y)):
        fold_ids[va_idx] = f

    Xtr, Xte = add_target_encodings(Xtr, Xte, y, fold_ids)
    feature_cols = list(Xtr.columns)
    print(f"features: {len(feature_cols)}", flush=True)

    for c in feature_cols:
        if Xtr[c].dtype == object:
            Xtr[c] = Xtr[c].astype(str)
            Xte[c] = Xte[c].astype(str)

    Xtr_f = Xtr[feature_cols]
    Xte_f = Xte[feature_cols]

    def run_all(Xa, ya, fida, nreal):
        res = {}
        for key, kind, seeds in MODEL_SPECS:
            spec_gpu = kind in GPU_KINDS and use_gpu
            res[key] = run_spec_sequential(
                key, kind, seeds, Xa, Xte_f, ya, fida, spec_gpu, n_real=nreal
            )
        return res

    # ---- Phase A: normal CV on the real training rows ----
    results = run_all(Xtr_f, y, fold_ids, len(y))
    te_prob = np.mean([results[k][1] for k in results], axis=0)  # blended test prob

    # ---- Phase B: pseudo-labeling (self-training on confident test rows) ----
    results_B = None
    if PL_PHASES >= 2:
        confident = (te_prob < PL_LO) | (te_prob > PL_HI)
        n_conf = int(confident.sum())
        pseudo_X = Xte_f[confident].reset_index(drop=True)
        pseudo_y = (te_prob[confident] > 0.5).astype(np.int8)
        print(f"[pseudo] adding {n_conf} confident test rows ({n_conf/len(te_prob):.1%})", flush=True)
        if n_conf > 0:
            X_aug = pd.concat([Xtr_f, pseudo_X], ignore_index=True)
            y_aug = np.concatenate([y, pseudo_y])
            fold_aug = np.concatenate([fold_ids, np.full(n_conf, -1, dtype=np.int16)])
            results_B = run_all(X_aug, y_aug, fold_aug, len(y))

    # ---- Report honest OOF (real train only) ----
    def blend_te(res):
        return np.mean([rank_pct(res[k][1]) for k in res], axis=0)

    eq_oof = np.mean([rank_pct(results[k][0]) for k in results], axis=0)
    eq_auc = roc_auc_score(y, eq_oof)
    print("\n" + "=" * 70)
    for k, (_o, _t, auc) in results.items():
        print(f"{k:8s} PhaseA OOF AUC = {auc:.8f}")
    if results_B is not None:
        for k, (_o, _t, auc) in results_B.items():
            print(f"{k:8s} PhaseB OOF AUC = {auc:.8f}")
    print(f"{'BLEND-A':8s} OOF AUC = {eq_auc:.8f}  (real-train CV; judge by LB, not this)")
    print("=" * 70, flush=True)

    # ---- Final test blend: Phase A + Phase B (rank space) ----
    te_rank = blend_te(results)
    if results_B is not None:
        te_rank = 0.5 * te_rank + 0.5 * blend_te(results_B)

    # Map blend ranks onto the best model's test probability distribution.
    best_key = max(results, key=lambda k: results[k][2])
    ref_sorted = np.sort(results[best_key][1])
    ranks = np.argsort(np.argsort(te_rank))
    idx = (ranks / (len(te_rank) - 1) * (len(ref_sorted) - 1)).astype(int)
    blended_probs = ref_sorted[idx]

    np.save("v5_test.npy", blended_probs)
    np.save("v5_oof.npy", eq_oof)

    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: blended_probs})
    if sample_path:
        submission = pd.read_csv(sample_path)
        if ID_COL in submission.columns:
            sub = submission[[ID_COL]].merge(sub, on=ID_COL, how="left")

    fname = f"submission_{VERSION}.csv"
    out = os.path.join("/kaggle/working", fname) if os.path.isdir("/kaggle/working") else fname
    sub.to_csv(out, index=False)
    print(f"Wrote {out} ({len(sub)} rows) | prob {blended_probs.min():.4f}..{blended_probs.max():.4f} mean {blended_probs.mean():.4f}")
    print(f"Runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
