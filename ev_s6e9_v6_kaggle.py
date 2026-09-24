"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v6 · KAGGLE build)
Same model/feature recipe as ev_s6e9_v6.py (the Colab single-T4 build), but the
compute layout is rewritten to use BOTH Kaggle T4s AND the CPU pool at the same
time. Identical features/TE/blend/post-processing, so scores are comparable.

HOW BOTH GPUs GET USED (no multiprocessing — that breaks on Kaggle, see below):
  1. CatBoost  -> task_type="GPU", devices="0:1"  (one model spread over both T4s)
  2. XGBoost   -> xgb_a pinned to cuda:0, xgb_b pinned to cuda:1
  3. LightGBM  -> CPU, run in a BACKGROUND THREAD while the GPU lane trains.
     LightGBM releases the GIL during fitting, so the CPU lane and the GPU lane
     genuinely overlap. This is where the ~2h saving comes from: on the Colab
     build the GPU sits idle for hours while LightGBM grinds on 2 cores.
  4. PyTorch MLP -> cuda:0 (tiny model; DataParallel would only add overhead).

WHY NOT ProcessPoolExecutor fold-parallelism: Kaggle's "Save & Run All" executes
the .py inside the notebook kernel, so spawned workers cannot re-import functions
from __main__ -> BrokenProcessPool. Threads avoid that entirely (no pickling of
callables, shared memory).

=========================  KAGGLE SETUP  ======================================
New Kaggle Notebook -> Add Input (twice):
  1. Competition data  "Playground Series S6E9" (train.csv / test.csv /
     sample_submission.csv)  — OR your own dataset
     "mitanshkanani/dataset-for-experimentation-final"
  2. Dataset "itzzomkar/ev-adoption-behavior-and-range-anxiety"  -> gives the
     10k-row EV_Adoption_and_Range_Anxiety_Dataset.csv, auto-detected + merged.
Then either attach this file as code and run it, or paste it into a cell.
Output: /kaggle/working/submission_v6_kaggle.csv (+ v6k_*.npy artifacts).
GPU quota note: Kaggle gives 30 GPU-hours/week; this run uses roughly 3-5h of
wall-clock. Enable GPU accelerator, then "Save & Run All".
===============================================================================
"""

from __future__ import annotations

import gc
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
from scipy.special import ndtr

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v6_kaggle"
N_SPLITS = 10
RANDOM_STATE = 42
TE_SMOOTHINGS = (10.0, 30.0)     # bucketed/triple TE (GBDT)
NN_TE_SMOOTHING = 50.0           # light exact TE (NN only, v4-proven)

USE_ORIGINAL_DATA = True         # merge itzzomkar source dataset if found
EDGE_CLAMP = True                # clamp deterministic edge rows at the end
USE_NN = True                    # PyTorch MLP diversity model (auto-skips on failure)
NN_SEEDS = (42, 7)
NN_WEIGHT = 0.25                 # verified best ratio: 0.75 GBDT / 0.25 NN
BATCH = 8192
MAX_EPOCHS = 40
PATIENCE = 5

CAT_MULTI_GPU = True             # CatBoost across both T4s (devices="0:1")
XGB_GPU_OF_SPEC = {"xgb_a": 0, "xgb_b": 1}   # one spec per GPU
OVERLAP_CPU_LANE = True          # run LightGBM in a thread while GPU lane trains

KAGGLE_INPUT_DIR = "/kaggle/input"   # Kaggle mounts Inputs here (may nest deeper)

MODEL_SPECS = [
    ("cat",    "catboost", (0, 1)),
    ("xgb_a",  "xgboost",  (42, 7)),
    ("xgb_b",  "xgboost",  (13,)),
    ("lgbm_a", "lightgbm", (42, 7)),
    ("lgbm_b", "lightgbm", (13,)),
]
GPU_KINDS = {"catboost", "xgboost"}   # LightGBM stays on CPU (GPU build unreliable)

CFG = {
    "cat":    {"iterations": 2000, "lr": 0.05, "depth": 8, "l2": 5.0},
    "xgb_a":  {"lr": 0.03, "depth": 6, "mcw": 12, "csb": 0.85, "lambda": 3.0},
    "xgb_b":  {"lr": 0.025, "depth": 8, "mcw": 25, "csb": 0.7, "lambda": 6.0},
    "lgbm_a": {"lr": 0.03, "num_leaves": 63, "mcs": 40, "csb": 0.85, "lambda": 3.0, "max_bin": 1023},
    "lgbm_b": {"lr": 0.025, "num_leaves": 127, "mcs": 60, "csb": 0.75, "lambda": 5.0},
}

# Deterministic generator edges (verified on train.csv).
CLIFF_INCOME = 170_537
DEAD_LO, DEAD_HI = 31_004, 41_970
COMMUTE_ZERO_EDGE = 83.0

_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(msg, flush=True)


# ----------------------------------------------------------------------------
# Environment helpers.
# ----------------------------------------------------------------------------
def ensure_import(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
        return True
    except Exception:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       check=True, timeout=600)
        __import__(name)
        log(f"[env] installed {pkg}")
        return True
    except Exception as exc:
        log(f"[env] could not install/import {pkg}: {exc!r} -> related specs skipped")
        return False


def _csv_search_dirs():
    """Every directory worth looking in. Kaggle mounts inputs at
    /kaggle/input/<slug> OR /kaggle/input/datasets/<owner>/<slug>, so the walk
    must descend all the way — no depth pruning (a previous version stopped one
    level too early and missed the files entirely). Only a visited-dir cap."""
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
        for dirpath, _dirnames, _files in os.walk(KAGGLE_INPUT_DIR):
            if dirpath in seen:
                continue
            seen.add(dirpath)
            yield dirpath
            if len(seen) > 500:                 # pathological tree guard
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
        listing = []
        if os.path.isdir(KAGGLE_INPUT_DIR):
            for dirpath, _dirnames, files in os.walk(KAGGLE_INPUT_DIR):
                listing.append(f"  {dirpath}: {sorted(files)[:8]}")
        raise FileNotFoundError(
            "train.csv / test.csv not found.\nContents of " + KAGGLE_INPUT_DIR + ":\n"
            + ("\n".join(listing) or "  (nothing mounted — no Input attached)")
            + "\nUse the Input panel -> Datasets -> Add your dataset, then RESTART "
              "the session so the mount appears."
        )
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


# ----------------------------------------------------------------------------
# GBDT feature engineering (target-free) — IDENTICAL to the Colab v6 build.
# ----------------------------------------------------------------------------
def income_digits(inc: pd.Series) -> pd.DataFrame:
    """Integer digit decomposition. Income is integer-valued here; fractional
    digits are harmful (float-repr floors them), so we never create any."""
    inc_i = np.rint(pd.to_numeric(inc, errors="coerce").fillna(0).to_numpy(np.float64)).astype(np.int64)
    out = {}
    for div in (1, 10, 100, 1_000, 10_000, 100_000):
        out[f"inc_digit_{div}"] = ((inc_i // div) % 10).astype(np.int8)
    out["inc_last2"] = (inc_i % 100).astype(np.int16)
    out["inc_last3"] = (inc_i % 1_000).astype(np.int16)
    out["inc_mod_5000"] = (inc_i % 5_000).astype(np.int16)
    out["inc_b_1k"] = (inc_i // 1_000).astype(np.int32)
    out["inc_b_5k"] = (inc_i // 5_000).astype(np.int32)
    out["inc_b_10k"] = (inc_i // 10_000).astype(np.int32)
    return pd.DataFrame(out, index=inc.index)


def edge_flags(income, commute, env) -> pd.DataFrame:
    """Flags for the deterministic/semideterministic generator structure."""
    x = pd.DataFrame(index=income.index)
    x["is_cliff"] = (income >= CLIFF_INCOME).astype(np.int8)
    x["is_dead_zone"] = ((income >= DEAD_LO) & (income <= DEAD_HI)).astype(np.int8)
    x["is_30k_spike"] = (income == 30_000).astype(np.int8)
    x["is_comm_5"] = (commute.round(1) == 5.0).astype(np.int8)
    x["is_comm_ge83"] = (commute >= COMMUTE_ZERO_EDGE).astype(np.int8)
    x["is_env1"] = (env == 1).astype(np.int8)
    return x


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

    x = pd.concat([x, income_digits(income), edge_flags(income, commute, env)], axis=1)

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
    x["age_sq"] = age * age
    x["log_income"] = np.log1p(income)

    # Generator recipe (public EDA): buys if this score > ~5.5.
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
# Leakage-safe target encoding (shares the model CV folds).
# ----------------------------------------------------------------------------
def _te_key(df, cols):
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def _fit_te(keys, y, prior, smoothing):
    frame = pd.DataFrame({"k": keys.to_numpy(), "t": y})
    stats = frame.groupby("k", observed=True)["t"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


# Bucketed doubles + TRIPLES on low-cardinality interactions (public 0.9463
# recipe). No exact-value TE for GBDTs: it inflates CV without transferring.
TE_KEYS = {
    "te_incb_5k": ["inc_b_5k"],
    "te_incb_1k": ["inc_b_1k"],
    "te_cmb": ["comm_b5"],
    "te_incb_subsidy": ["inc_b_5k", "Subsidy_Available"],
    "te_incb_env": ["inc_b_5k", "Environmental_Concern_Level"],
    "te_incb_anx": ["inc_b_5k", "Range_Anxiety_Level"],
    "te_inc_sub_anx": ["inc_b_5k", "Subsidy_Available", "Range_Anxiety_Level"],
    "te_inc_sub_env": ["inc_b_5k", "Subsidy_Available", "Environmental_Concern_Level"],
    "te_spike_sub_anx": ["is_30k_spike", "Subsidy_Available", "Range_Anxiety_Level"],
    "te_cmb_sub": ["comm_b5", "Subsidy_Available"],
    "te_city_sub_anx": ["City_Type", "Subsidy_Available", "Range_Anxiety_Level"],
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
# GBDT model factories.
# ----------------------------------------------------------------------------
def _cat_cols(X):
    return [c for c in X.columns
            if X[c].dtype == object or pd.api.types.is_string_dtype(X[c].dtype)
            or isinstance(X[c].dtype, pd.CategoricalDtype)]


def _cast_cat(Xtr, Xva, Xte, cat_features):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in cat_features:
        cats = pd.Index(pd.concat([Xtr[c], Xva[c], Xte[c]]).astype(str).unique())
        dt = pd.CategoricalDtype(categories=cats, ordered=False)
        Xtr[c] = Xtr[c].astype(str).astype(dt)
        Xva[c] = Xva[c].astype(str).astype(dt)
        Xte[c] = Xte[c].astype(str).astype(dt)
    return Xtr, Xva, Xte


def _cat_devices(n_gpus: int) -> str:
    if CAT_MULTI_GPU and n_gpus >= 2:
        return ":".join(str(i) for i in range(n_gpus))   # "0:1"
    return "0"


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
        devices=_cat_devices(int(cfg.get("n_gpus", 1))) if use_gpu else "0",
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
    device = cfg.get("device", "cuda" if use_gpu else "cpu")

    def _make(dev):
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            n_estimators=6000, learning_rate=cfg.get("lr", 0.03),
            max_depth=cfg.get("depth", 6), min_child_weight=cfg.get("mcw", 12),
            subsample=0.85, colsample_bytree=cfg.get("csb", 0.85),
            reg_lambda=cfg.get("lambda", 3.0), reg_alpha=0.0, gamma=0.0,
            max_cat_to_onehot=8, tree_method="hist",
            device=dev, enable_categorical=True,
            early_stopping_rounds=250, random_state=seed, n_jobs=-1,
        )

    try:
        model = _make(device)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=0)
    except Exception as exc:
        # e.g. this XGBoost build rejects "cuda:1" -> fall back to generic cuda.
        log(f"    [xgb] device={device} failed ({exc!r}) -> retry on 'cuda'")
        model = _make("cuda" if use_gpu else "cpu")
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
        reg_alpha=0.0, max_bin=cfg.get("max_bin", 255),
        n_jobs=cfg.get("n_jobs", -1), random_state=seed, verbose=-1,
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


def run_spec_sequential(key, kind, seeds, X, Xte, y, fold_ids, use_gpu, cfg):
    factory = FACTORIES[kind]
    oof_sum = np.zeros(len(y))
    te_sum = np.zeros(len(Xte))
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        tes = []
        for f in range(N_SPLITS):
            tr = fold_ids != f
            va = fold_ids == f
            vp, tp = factory(
                X[tr].reset_index(drop=True), y[tr],
                X[va].reset_index(drop=True), y[va],
                Xte, seed + f, use_gpu, cfg,
            )
            oof[va] = vp
            tes.append(tp)
            log(f"    [{key} seed={seed} fold={f}] auc={roc_auc_score(y[va], vp):.6f}")
        log(f"  [{key}] seed={seed} OOF AUC = {roc_auc_score(y, oof):.8f}")
        oof_sum += oof
        te_sum += np.mean(np.vstack(tes), axis=0)
        del oof, tes
        gc.collect()
    oof_avg = oof_sum / len(seeds)
    te_avg = te_sum / len(seeds)
    return oof_avg, te_avg, roc_auc_score(y, oof_avg)


def run_lane(specs, X, Xte, y, fold_ids, n_gpus, results, cfg_extra=None):
    """Run a list of specs (one compute lane) and store results thread-safely."""
    for key, kind, seeds in specs:
        cfg = dict(CFG[key])
        if cfg_extra:
            cfg.update(cfg_extra)
        if kind == "catboost":
            cfg["n_gpus"] = n_gpus
        elif kind == "xgboost" and n_gpus >= 1:
            idx = XGB_GPU_OF_SPEC.get(key, 0) % n_gpus
            cfg["device"] = f"cuda:{idx}" if n_gpus >= 2 else "cuda"
        use_gpu = kind in GPU_KINDS and n_gpus > 0
        res = run_spec_sequential(key, kind, seeds, X, Xte, y, fold_ids, use_gpu, cfg)
        with _LOG_LOCK:
            results[key] = res
        log(f"== {key} OOF AUC = {res[2]:.8f} ==")


# ----------------------------------------------------------------------------
# PyTorch MLP (diversity model). Deliberately different feature space:
# one-hot cats + light EXACT TE (m50) + edge flags + integer income digits.
# ----------------------------------------------------------------------------
NN_NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
NN_CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]


def build_nn_base(df):
    x = df[NN_NUMERIC].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    inc = x["Annual_Income_USD"]
    age = x["Age"]
    cars = x["Number_of_Cars_Owned"]
    commute = x["Daily_Commute_km"]
    env = x["Environmental_Concern_Level"]
    tot = x["Charging_Stations_Near_Home"] + x["Charging_Stations_Near_Work"]
    x["log_income"] = np.log1p(inc)
    x["income_per_car"] = inc / (cars + 1.0)
    x["income_per_age"] = inc / (age + 1.0)
    x["total_charging"] = tot
    x["commute_per_charging"] = commute / (tot + 1.0)
    anx = df["Range_Anxiety_Level"].astype(str).str.lower().map({"low": 0, "medium": 1, "high": 2}).fillna(-1)
    x["anxiety_ord"] = anx.astype(np.float32)
    x["subsidy_flag"] = (df["Subsidy_Available"].astype(str).str.lower() == "yes").astype(np.float32)
    x["home_charge_flag"] = (df["Home_Charging_Possible"].astype(str).str.lower() == "yes").astype(np.float32)
    flags = edge_flags(inc, commute, env).astype(np.float32)
    digs = income_digits(inc).astype(np.float32)
    oh = pd.get_dummies(df[NN_CATEGORICAL].astype(str), columns=NN_CATEGORICAL, prefix=NN_CATEGORICAL)
    return pd.concat([x, flags, digs, oh.astype(np.float32)], axis=1)


def _te_key_exact(df, col):
    if col == "Annual_Income_USD":
        arr = np.rint(pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float64)).astype(np.int64)
        return pd.Series(arr, index=df.index).astype(str)
    arr = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float64)
    return pd.Series([f"{v:.1f}" for v in arr], index=df.index)


def add_light_te(train, test, y, fold_ids):
    for col in ["Annual_Income_USD", "Daily_Commute_km"]:
        tr_keys = _te_key_exact(train, col)
        te_keys = _te_key_exact(test, col)
        name = f"te_{col}_m{int(NN_TE_SMOOTHING)}"
        enc = np.full(len(train), np.nan, np.float32)
        for f in range(N_SPLITS):
            trm = fold_ids != f
            vam = fold_ids == f
            prior = float(y[trm].mean())
            mp = _fit_te(tr_keys[trm], y[trm], prior, NN_TE_SMOOTHING)
            enc[vam] = tr_keys[vam].map(mp).fillna(prior).to_numpy(np.float32)
        train[name] = enc
        full = _fit_te(tr_keys, y, float(y.mean()), NN_TE_SMOOTHING)
        test[name] = te_keys.map(full).fillna(float(y.mean())).to_numpy(np.float32)
    return train, test


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except Exception:
        return None


def train_fold_nn(Xtr_np, ytr, Xva_np, yva, Xte_np, seed, device):
    imported = _try_import_torch()
    if imported is None:
        return _train_fold_sklearn(Xtr_np, ytr, Xva_np, yva, Xte_np, seed)
    torch, nn, DataLoader, TensorDataset = imported

    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = Xtr_np.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, z):
            return self.net(z).squeeze(-1)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.BCEWithLogitsLoss()

    tr_t = torch.tensor(Xtr_np, dtype=torch.float32)
    tr_y = torch.tensor(ytr, dtype=torch.float32)
    va_t = torch.tensor(Xva_np, dtype=torch.float32).to(device)
    te_t = torch.tensor(Xte_np, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(tr_t, tr_y), batch_size=BATCH, shuffle=True, drop_last=True)

    best_auc, best_va, best_te, bad = -1.0, None, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(model(va_t)).cpu().numpy()
            va_auc = roc_auc_score(yva, va_probs)
        if va_auc > best_auc:
            best_auc = va_auc
            best_va = va_probs
            with torch.no_grad():
                best_te = torch.sigmoid(model(te_t)).cpu().numpy()
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    del model, tr_t, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_va, best_te, best_auc


def _train_fold_sklearn(Xtr_np, ytr, Xva_np, yva, Xte_np, seed):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", n_jobs=-1)
    m.fit(Xtr_np, ytr)
    va = m.predict_proba(Xva_np)[:, 1]
    te = m.predict_proba(Xte_np)[:, 1]
    return va, te, roc_auc_score(yva, va)


def run_nn(train, test, y, fold_ids, n_gpus):
    from sklearn.preprocessing import StandardScaler
    imported = _try_import_torch()
    if imported is not None:
        torch = imported[0]
        device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpus > 0) else "cpu")
        log(f"[nn] PyTorch MLP on {device}")
    else:
        device = "cpu"
        log("[nn] torch not found -> sklearn LogisticRegression fallback")

    Xtr = build_nn_base(train)
    Xte = build_nn_base(test)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1)
    Xtr = Xtr.fillna(0.0)
    Xte = Xte.fillna(0.0)
    Xtr, Xte = add_light_te(Xtr, Xte, y, fold_ids)
    cols = list(Xtr.columns)
    log(f"[nn] features: {len(cols)}")

    oof_sum = np.zeros(len(y))
    te_sum = np.zeros(len(Xte))
    for seed in NN_SEEDS:
        oof = np.full(len(y), np.nan)
        tes = []
        for f in range(N_SPLITS):
            tr = fold_ids != f
            va = fold_ids == f
            scaler = StandardScaler()
            Xtr_np = scaler.fit_transform(Xtr.loc[tr, cols].to_numpy(np.float32))
            Xva_np = scaler.transform(Xtr.loc[va, cols].to_numpy(np.float32))
            Xte_np = scaler.transform(Xte[cols].to_numpy(np.float32))
            vp, tp, fauc = train_fold_nn(Xtr_np, y[tr], Xva_np, y[va], Xte_np, seed + f, device)
            oof[va] = vp
            tes.append(tp)
            log(f"    [nn seed={seed} fold={f}] auc={fauc:.6f}")
        log(f"  [nn] seed={seed} OOF AUC = {roc_auc_score(y, oof):.8f}")
        oof_sum += oof
        te_sum += np.mean(np.vstack(tes), axis=0)
        del oof, tes
        gc.collect()
    oof_avg = oof_sum / len(NN_SEEDS)
    te_avg = te_sum / len(NN_SEEDS)
    return oof_avg, te_avg, roc_auc_score(y, oof_avg)


# ----------------------------------------------------------------------------
# OOF weight optimizer — INFO ONLY (OOF-tuned weights overfit; submit fixed).
# ----------------------------------------------------------------------------
def optimize_weights(oof_dict, y, n_iter=1500, seed=0):
    names = list(oof_dict)
    R = np.vstack([rank_pct(oof_dict[n]) for n in names])
    k = len(names)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), size=min(250_000, len(y)), replace=False)
    ys, Rs = y[sub], R[:, sub]

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
    return dict(zip(names, best_w)), auc_w(best_w, R, y)


# ----------------------------------------------------------------------------
# Original source dataset (itzzomkar EV Adoption, 10k rows) — auto-detected.
# Guarded: your Kaggle dataset also ships fullypreprocessed_train.csv (298MB)
# and onehotenc_train.csv, which carry the same 13 feature names + target and
# would otherwise be mistaken for the source and merged as duplicate train rows.
# The real source has no competition `id` column (it uses Buyer_ID) and is tiny.
# ----------------------------------------------------------------------------
ORIGINAL_CSV = None  # set the exact path here to bypass auto-detection entirely
ORIG_NAME_HINTS = ("ev_adoption", "range_anxiety", "source", "original")
MAX_ORIG_BYTES = 20 * 1024 * 1024


def load_original(train, skip_paths):
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    skip = {os.path.abspath(p) for p in skip_paths if p}

    found = []  # (priority, path) — 0 = filename says it's the source
    roots = []
    if ORIGINAL_CSV and os.path.isfile(ORIGINAL_CSV):
        roots = [os.path.dirname(os.path.abspath(ORIGINAL_CSV))]
    else:
        for r in (KAGGLE_INPUT_DIR, "/content", os.getcwd(), "."):
            if os.path.isdir(r):
                roots.append(r)
    for root in roots:
        for dirpath, _dirnames, files in os.walk(root):
            # No depth pruning: Kaggle nests the mount several levels deep.
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
                prio = 0 if any(h in fn.lower() for h in ORIG_NAME_HINTS) else 1
                found.append((prio, p))

    for _prio, p in sorted(found):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if TARGET not in df.columns or not set(feats).issubset(df.columns):
            continue
        if ID_COL in df.columns:      # competition/preprocessed data, not the source
            continue
        sub = df[feats + [TARGET]].dropna(subset=[TARGET]).copy()
        log(f"[original] merged {len(sub)} rows from {os.path.basename(p)}")
        return sub
    log("[original] source dataset not found — training on competition data only")
    return None


# ----------------------------------------------------------------------------
# Post-processing: tie-break + edge clamp (public 0.94656/0.94657 recipe).
# ----------------------------------------------------------------------------
def finalize_test_probs(blend_score, tiebreak_score, ref_probs, test):
    order = np.lexsort((rank_pct(tiebreak_score), blend_score))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))

    ref_sorted = np.sort(ref_probs)
    idx = (ranks / (len(ranks) - 1) * (len(ref_sorted) - 1)).astype(np.int64)
    probs = ref_sorted[idx]

    if EDGE_CLAMP:
        inc = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").fillna(0).to_numpy()
        comm = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").fillna(0).to_numpy()
        cliff = inc >= CLIFF_INCOME
        zero_edge = ((inc >= DEAD_LO) & (inc <= DEAD_HI)) | (comm >= COMMUTE_ZERO_EDGE)
        probs[cliff] = 1.0
        probs[zero_edge] = 0.0
        log(f"[edges] clamped {int(cliff.sum())} rows -> 1.0, {int(zero_edge.sum())} rows -> 0.0")
    return probs


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    log(f"train: {train_path}\ntest : {test_path}")

    n_gpus = gpu_count()
    n_cpus = multiprocessing.cpu_count()
    log(f"GPUs={n_gpus} CPUs={n_cpus} cat_multi_gpu={CAT_MULTI_GPU}")

    available = {
        "catboost": ensure_import("catboost"),
        "xgboost": ensure_import("xgboost"),
        "lightgbm": ensure_import("lightgbm"),
    }
    specs = [s for s in MODEL_SPECS if available.get(s[1], False)]
    if not specs:
        raise RuntimeError("No GBDT library available.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if USE_ORIGINAL_DATA:
        orig = load_original(train, [train_path, test_path, sample_path])
        if orig is not None:
            train = pd.concat([train, orig], ignore_index=True)

    y = encode_target(train[TARGET])
    log(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va_idx) in enumerate(skf.split(np.zeros(len(train)), y)):
        fold_ids[va_idx] = f

    Xtr = build_base_features(train)
    Xte = build_base_features(test)
    Xtr, Xte = add_target_encodings(Xtr, Xte, y, fold_ids)
    feature_cols = list(Xtr.columns)
    log(f"[gbdt] features: {len(feature_cols)}")
    for c in feature_cols:
        if Xtr[c].dtype == object:
            Xtr[c] = Xtr[c].astype(str)
            Xte[c] = Xte[c].astype(str)
    Xtr = Xtr[feature_cols]
    Xte = Xte[feature_cols]

    # ---- Two overlapping lanes: GPU (CatBoost+XGBoost) and CPU (LightGBM) ----
    results = {}
    gpu_specs = [s for s in specs if s[1] in GPU_KINDS]
    cpu_specs = [s for s in specs if s[1] == "lightgbm"]

    if OVERLAP_CPU_LANE and gpu_specs and cpu_specs:
        per_spec_jobs = max(1, n_cpus // len(cpu_specs))
        log(f"[schedule] GPU lane: {[k for k, _, _ in gpu_specs]} (both T4s) | "
            f"CPU lane: {[k for k, _, _ in cpu_specs]} (n_jobs={per_spec_jobs} each, overlapping)")
        cpu_thread = threading.Thread(
            target=run_lane, args=(cpu_specs, Xtr, Xte, y, fold_ids, n_gpus, results),
            kwargs={"cfg_extra": {"n_jobs": per_spec_jobs}}, daemon=True,
        )
        cpu_thread.start()
        run_lane(gpu_specs, Xtr, Xte, y, fold_ids, n_gpus, results)
        cpu_thread.join()
    else:
        log("[schedule] single sequential lane")
        run_lane(specs, Xtr, Xte, y, fold_ids, n_gpus, results)

    # ---- NN diversity model (never kills the run) ----
    nn_ok = False
    if USE_NN:
        try:
            oof_nn, te_nn, auc_nn = run_nn(train, test, y, fold_ids, n_gpus)
            results["nn"] = (oof_nn, te_nn, auc_nn)
            nn_ok = True
        except Exception as exc:
            log(f"[nn] FAILED ({exc!r}) -> continuing with GBDTs only")

    # ---- Fixed-weight rank blend (0.75 GBDT / 0.25 NN when NN ran) ----
    keys = list(results)
    if nn_ok:
        gbdt_keys = [k for k in keys if k != "nn"]
        weights = {k: (1.0 - NN_WEIGHT) / len(gbdt_keys) for k in gbdt_keys}
        weights["nn"] = NN_WEIGHT
    else:
        weights = {k: 1.0 / len(keys) for k in keys}

    blend_oof = np.sum([weights[k] * rank_pct(results[k][0]) for k in keys], axis=0)
    blend_te = np.sum([weights[k] * rank_pct(results[k][1]) for k in keys], axis=0)
    blend_auc = roc_auc_score(y, blend_oof)

    opt_w, opt_auc = optimize_weights({k: results[k][0] for k in keys}, y)
    log("\n" + "=" * 70)
    for k in keys:
        log(f"{k:8s} OOF AUC = {results[k][2]:.8f}   (blend w {weights[k]:.3f} | opt w {opt_w[k]:.3f})")
    log(f"{'BLEND':8s} OOF AUC = {blend_auc:.8f}   <-- SUBMITTED (fixed weights)")
    log(f"{'OPT':8s} OOF AUC = {opt_auc:.8f}   (info only, overfits OOF)")
    log("=" * 70)

    # ---- Tie-break + edge clamp ----
    tie_key = "nn" if nn_ok else ("cat" if "cat" in results else keys[0])
    best_gbdt = max((k for k in keys if k != "nn"), key=lambda k: results[k][2])
    final_probs = finalize_test_probs(blend_te, results[tie_key][1], results[best_gbdt][1], test)

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    np.save(os.path.join(out_dir, "v6k_oof.npy"), blend_oof)
    np.save(os.path.join(out_dir, "v6k_test.npy"), final_probs)
    if nn_ok:
        np.save(os.path.join(out_dir, "v6k_nn_oof.npy"), results["nn"][0])
        np.save(os.path.join(out_dir, "v6k_nn_test.npy"), results["nn"][1])

    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: final_probs})
    if sample_path:
        submission = pd.read_csv(sample_path)
        if ID_COL in submission.columns:
            sub = submission[[ID_COL]].merge(sub, on=ID_COL, how="left")

    fname = f"submission_{VERSION}.csv"
    out = os.path.join(out_dir, fname)
    sub.to_csv(out, index=False)
    log(f"Wrote {out} ({len(sub)} rows) | prob {final_probs.min():.4f}..{final_probs.max():.4f} mean {final_probs.mean():.4f}")
    log(f"Runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
