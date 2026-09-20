"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09
Self-contained multi-model rank-average blend (CatBoost + XGBoost + LightGBM).

Run this single .py on Kaggle. It auto-detects the dataset folder, builds the
feature set that our team validated (income-digit decomposition, leakage-safe
exact-value target encoding, reverse-engineered recipe score, interactions),
trains three GBDT families with repeated stratified 10-fold CV, blends them by
rank-averaging, prints the OOF AUC-ROC, and writes submission.csv.

Design notes / why this should beat ~0.94618:
  * The data is synthetic with a hidden latent probability that is a smooth
    function of the features. Near-continuous Annual_Income_USD (~32k unique
    values) means an exact-value Bayesian target encoding is close to a
    per-value Bayes estimate — the single strongest tabular lever here.
  * Income digit decomposition + income buckets exploit the generator's
    structure that plain trees under-use.
  * Three diverse GBDT families + several seeds, combined by rank averaging,
    reliably add ~0.001-0.003 AUC over the best single model on playgrounds.
  * All target encodings are computed with the SAME frozen folds used for model
    CV, so no validation row ever sees its own label. Leakage-safe.

Nothing here depends on local artifacts: it only needs train.csv + test.csv.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import ndtr

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG — tweak these if you want to trade runtime for score.
# ----------------------------------------------------------------------------
TARGET = "Will_Buy_EV"
ID_COL = "id"
N_SPLITS = 10
CATBOLD_SEEDS = (0, 7, 42)     # CatBoost seeds (GPU-friendly, cheap)
XGB_SEEDS = (42, 7)            # XGBoost seeds
LGBM_SEEDS = (42, 7)           # LightGBM seeds
TE_SMOOTHINGS = (2.0, 10.0)    # Bayesian smoothing for target encodings
RANDOM_STATE = 42

# Blend weights over the rank-averaged per-family test predictions.
# Equal weights are robust; nudge toward CatBoost if its OOF clearly leads.
FAMILY_WEIGHTS = {"catboost": 1.0, "xgboost": 1.0, "lightgbm": 1.0}

USE_GPU = None  # None = auto-detect; True/False to force.

# Kaggle notebooks expose a single GPU, so the default uses device '0'.
# Set True only on a verified multi-GPU machine (e.g. your own rig with 2xT4);
# CatBoost will then train one model across all visible GPUs via '0:1'.
CAT_MULTI_GPU = False


# ----------------------------------------------------------------------------
# Path detection (Kaggle + local).
# ----------------------------------------------------------------------------
def find_data_dir() -> tuple[str, str, str]:
    """Return (train_path, test_path, sample_path)."""
    candidates = []

    # Explicit Kaggle dataset path the user provided.
    candidates.append(
        "/kaggle/input/datasets/mitanshkanani/dataset-for-experimentation-final"
    )

    # Any folder mounted under /kaggle/input.
    if os.path.isdir("/kaggle/input"):
        for root in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", root)
            if os.path.isdir(base):
                candidates.append(base)
                for sub in os.listdir(base):
                    subp = os.path.join(base, sub)
                    if os.path.isdir(subp):
                        candidates.append(subp)

    # Local repo layout.
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
        raise FileNotFoundError(
            "Could not locate train.csv / test.csv. Mount the dataset or set "
            "the paths near the top of find_data_dir()."
        )
    return train_path, test_path, sample_path


def gpu_count() -> int:
    """Number of CUDA GPUs visible (0 if CPU-only)."""
    if USE_GPU is False:
        return 0
    try:
        import torch

        n = torch.cuda.device_count()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    # Fallback: count GPUs via nvidia-smi without importing torch.
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15
        ).stdout
        return max(0, out.lower().count("gpu"))
    except Exception:
        return 1 if os.path.exists("/proc/driver/nvidia/version") else 0


def detect_gpu() -> bool:
    return gpu_count() > 0


# ----------------------------------------------------------------------------
# Target / label handling.
# ----------------------------------------------------------------------------
def encode_target(series: pd.Series) -> np.ndarray:
    vals = {str(v).strip().lower() for v in series.unique()}
    if vals <= {"yes", "no"}:
        return (series.astype(str).str.strip().str.lower() == "yes").astype(np.int8).to_numpy()
    if vals <= {"1", "0"}:
        return series.astype(int).to_numpy()
    # Fallback: minority class == positive.
    pos = series.value_counts().idxmin()
    return (series == pos).astype(np.int8).to_numpy()


# ----------------------------------------------------------------------------
# Feature engineering (target-free part).
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

    # Binary / categorical (kept as strings; each family encodes them).
    x["Gender"] = df["Gender"].astype(str)
    x["City_Type"] = df["City_Type"].astype(str)
    x["Current_Car_Type"] = df["Current_Car_Type"].astype(str)
    x["Home_Charging_Possible"] = df["Home_Charging_Possible"].astype(str)
    x["Subsidy_Available"] = df["Subsidy_Available"].astype(str)
    x["Range_Anxiety_Level"] = df["Range_Anxiety_Level"].astype(str)

    # Ordinal encodings of the ordered categoricals.
    anxiety_map = {"low": 0, "medium": 1, "high": 2}
    anx = df["Range_Anxiety_Level"].astype(str).str.strip().str.lower()
    x["anxiety_ord"] = anx.map(anxiety_map).fillna(-1).astype(np.int8)
    x["subsidy_flag"] = (df["Subsidy_Available"].astype(str).str.strip().str.lower() == "yes").astype(np.int8)
    x["home_charge_flag"] = (df["Home_Charging_Possible"].astype(str).str.strip().str.lower() == "yes").astype(np.int8)

    # Income digit decomposition (validated winner feature group).
    inc_i = np.rint(income.to_numpy(dtype=np.float64)).astype(np.int64)
    for div in (1, 10, 100, 1_000, 10_000, 100_000):
        x[f"inc_digit_{div}"] = ((inc_i // div) % 10).astype(np.int8)
    x["inc_last2"] = (inc_i % 100).astype(np.int16)
    x["inc_last3"] = (inc_i % 1_000).astype(np.int16)
    x["inc_mod_5000"] = (inc_i % 5_000).astype(np.int16)

    # Income buckets at several resolutions (exact value is high-cardinality).
    x["inc_b_1k"] = (inc_i // 1_000).astype(np.int32)
    x["inc_b_5k"] = (inc_i // 5_000).astype(np.int32)
    x["inc_b_10k"] = (inc_i // 10_000).astype(np.int32)

    # Commute discretisation.
    comm = np.rint(commute.to_numpy(dtype=np.float64) * 10).astype(np.int64)  # 0.1 km
    x["comm_b1"] = (comm // 10).astype(np.int16)          # whole km
    x["comm_b5"] = (comm // 50).astype(np.int16)          # 5 km buckets

    # Ratios / interactions.
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

    # Reverse-engineered generation recipe (target-free, weak but free signal).
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
# Leakage-safe exact-value target encoding.
# ----------------------------------------------------------------------------
def _te_key(series: pd.Series, kind: str) -> pd.Series:
    if kind == "income":
        arr = np.rint(pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(np.float64)).astype(np.int64)
        return pd.Series(arr, index=series.index).astype(str)
    if kind == "commute":
        arr = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(np.float64)
        return pd.Series([f"{v:.1f}" for v in arr], index=series.index)
    # generic string key (already composed)
    return series.astype(str)


def _fit_te(keys: pd.Series, y: np.ndarray, prior: float, smoothing: float) -> pd.Series:
    frame = pd.DataFrame({"k": keys.to_numpy(), "t": y})
    stats = frame.groupby("k", observed=True)["t"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


def add_target_encodings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build OOF target encodings for train (using frozen folds) and full-fit
    encodings for test. Adds count features too.
    """
    # Define keys as (name, train_key_series, test_key_series)
    def key_from_cols(df, cols, sep="|"):
        if len(cols) == 1:
            return _te_key(df[cols[0]], cols[0]) if cols[0] in ("Annual_Income_USD", "Daily_Commute_km") else df[cols[0]].astype(str)
        return df[cols[0]].astype(str) + sep + df[cols[1]].astype(str)

    key_specs = {
        "te_income": (["Annual_Income_USD"], ),
        "te_commute": (["Daily_Commute_km"], ),
        "te_inc_subsidy": (["Annual_Income_USD", "Subsidy_Available"], ),
        "te_inc_env": (["Annual_Income_USD", "Environmental_Concern_Level"], ),
        "te_inc_city": (["Annual_Income_USD", "City_Type"], ),
        "te_comm_homecharge": (["Daily_Commute_km", "Home_Charging_Possible"], ),
        "te_incbucket": (["inc_b_5k"], ),
    }

    global_prior = float(y.mean())

    for name, (cols,) in key_specs.items():
        tr_keys = key_from_cols(train, cols)
        te_keys = key_from_cols(test, cols)

        for m in TE_SMOOTHINGS:
            col = f"{name}_m{int(m)}"
            enc_tr = np.full(len(train), np.nan, dtype=np.float32)
            for f in range(N_SPLITS):
                tr_mask = fold_ids != f
                va_mask = fold_ids == f
                prior_f = float(y[tr_mask].mean())
                mapping = _fit_te(tr_keys[tr_mask], y[tr_mask], prior_f, m)
                enc_tr[va_mask] = (
                    tr_keys[va_mask].map(mapping).fillna(prior_f).to_numpy(np.float32)
                )
            train[col] = enc_tr
            full_prior = float(y.mean())
            full_map = _fit_te(tr_keys, y, full_prior, m)
            test[col] = te_keys.map(full_map).fillna(full_prior).to_numpy(np.float32)

        # Frequency / count of the key in train (helps trees trust rare values).
        counts = tr_keys.value_counts()
        train[f"{name}_cnt"] = tr_keys.map(counts).fillna(0).to_numpy(np.float32)
        test[f"{name}_cnt"] = te_keys.map(counts).fillna(0).to_numpy(np.float32)

    return train, test


# ----------------------------------------------------------------------------
# Model factories.
# ----------------------------------------------------------------------------
def _cat_cols(X):
    """Columns to treat as categorical (robust to object/string dtypes)."""
    out = []
    for c in X.columns:
        d = X[c].dtype
        if d == object or pd.api.types.is_string_dtype(d) or isinstance(d, pd.CategoricalDtype):
            out.append(c)
    return out


def _cat_devices(use_gpu):
    """
    CatBoost `devices` string. Default is a single GPU ('0'), which is what a
    Kaggle notebook actually exposes. Only set CAT_MULTI_GPU=True on a machine
    where you have verified N>1 real GPUs (CatBoost uses colon syntax '0:1').
    """
    if not use_gpu:
        return "0"
    if CAT_MULTI_GPU:
        return ":".join(str(i) for i in range(max(1, gpu_count())))
    return "0"


def train_catboost(Xtr, ytr, Xva, yva, Xte, seed, use_gpu):
    from catboost import CatBoostClassifier, Pool

    cat_features = _cat_cols(Xtr)
    params = dict(
        iterations=2500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=4.0,
        bagging_temperature=0.6,
        random_strength=0.7,
        border_count=254,
        grow_policy="SymmetricTree",
        eval_metric="AUC",
        random_seed=seed,
        logging_level="Silent",
        allow_writing_files=False,
        task_type="GPU" if use_gpu else "CPU",
        devices=_cat_devices(use_gpu),
    )
    model = CatBoostClassifier(**params)
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


def train_xgboost(Xtr, ytr, Xva, yva, Xte, seed, use_gpu):
    import xgboost as xgb

    # XGBoost needs pandas `category` dtype (not object) when
    # enable_categorical=True. Cast the string columns consistently across
    # train/valid/test so the category vocabulary matches.
    cat_features = _cat_cols(Xtr)
    Xtr = Xtr.copy()
    Xva = Xva.copy()
    Xte = Xte.copy()
    for c in cat_features:
        categories = pd.Index(pd.concat([Xtr[c], Xva[c], Xte[c]]).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        Xtr[c] = Xtr[c].astype(str).astype(dtype)
        Xva[c] = Xva[c].astype(str).astype(dtype)
        Xte[c] = Xte[c].astype(str).astype(dtype)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=6000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.0,
        gamma=0.0,
        max_cat_to_onehot=8,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        enable_categorical=True,
        early_stopping_rounds=250,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        verbose=0,
    )
    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    del model
    gc.collect()
    return va_pred, te_pred


def train_lightgbm(Xtr, ytr, Xva, yva, Xte, seed, use_gpu):
    import lightgbm as lgb

    cat_features = _cat_cols(Xtr)
    Xtr = Xtr.copy()
    Xva = Xva.copy()
    Xte = Xte.copy()
    for c in cat_features:
        categories = pd.Index(pd.concat([Xtr[c], Xva[c], Xte[c]]).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        Xtr[c] = Xtr[c].astype(str).astype(dtype)
        Xva[c] = Xva[c].astype(str).astype(dtype)
        Xte[c] = Xte[c].astype(str).astype(dtype)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        categorical_feature=cat_features if cat_features else "auto",
        callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)],
    )
    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    del model
    gc.collect()
    return va_pred, te_pred


# ----------------------------------------------------------------------------
# CV runner for one family across seeds.
# ----------------------------------------------------------------------------
def run_family(name, factory, X, Xte, y, fold_ids, seeds, use_gpu):
    oof_sum = np.zeros(len(y), dtype=np.float64)
    te_sum = np.zeros(len(Xte), dtype=np.float64)

    for seed in seeds:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        te_fold = []
        for f in range(N_SPLITS):
            tr = fold_ids != f
            va = fold_ids == f
            va_pred, te_pred = factory(
                X[tr].reset_index(drop=True),
                y[tr],
                X[va].reset_index(drop=True),
                y[va],
                Xte,
                seed + f,
                use_gpu,
            )
            oof[va] = va_pred
            te_fold.append(te_pred)
            print(f"    [{name} seed={seed} fold={f}] auc={roc_auc_score(y[va], va_pred):.6f}", flush=True)
        s_auc = roc_auc_score(y, oof)
        print(f"  [{name}] seed={seed} OOF AUC = {s_auc:.8f}", flush=True)
        oof_sum += oof
        te_sum += np.mean(np.vstack(te_fold), axis=0)
        del oof, te_fold
        gc.collect()

    oof_avg = oof_sum / len(seeds)
    te_avg = te_sum / len(seeds)
    fam_auc = roc_auc_score(y, oof_avg)
    print(f"== {name} multi-seed OOF AUC = {fam_auc:.8f} ==", flush=True)
    return oof_avg, te_avg, fam_auc


def rank_pct(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy()


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    print(f"train.csv : {train_path}")
    print(f"test.csv  : {test_path}")
    print(f"sample    : {sample_path}")

    n_gpus = gpu_count()
    use_gpu = n_gpus > 0
    print(f"GPUs visible: {n_gpus} | use_gpu={use_gpu} | cat_multi_gpu={CAT_MULTI_GPU}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"train shape {train.shape} | test shape {test.shape}")

    y = encode_target(train[TARGET])
    print(f"positive rate: {y.mean():.4f}")

    # Build features.
    Xtr = build_base_features(train)
    Xte = build_base_features(test)

    # Frozen stratified folds (reused for TE + model CV).
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va_idx) in enumerate(skf.split(Xtr, y)):
        fold_ids[va_idx] = f

    Xtr, Xte = add_target_encodings(Xtr, Xte, y, fold_ids, train, test)

    feature_cols = list(Xtr.columns)
    print(f"total features: {len(feature_cols)}")

    # Make categoricals 'object' so each family applies its own encoding.
    for c in feature_cols:
        if Xtr[c].dtype == object:
            Xtr[c] = Xtr[c].astype(str)
            Xte[c] = Xte[c].astype(str)

    results = {}
    results["catboost"] = run_family(
        "catboost", train_catboost, Xtr[feature_cols], Xte[feature_cols], y, fold_ids,
        CATBOLD_SEEDS, use_gpu,
    )
    results["xgboost"] = run_family(
        "xgboost", train_xgboost, Xtr[feature_cols], Xte[feature_cols], y, fold_ids,
        XGB_SEEDS, use_gpu,
    )
    # LightGBM runs on CPU (Kaggle's GPU LightGBM build is unreliable).
    results["lightgbm"] = run_family(
        "lightgbm", train_lightgbm, Xtr[feature_cols], Xte[feature_cols], y, fold_ids,
        LGBM_SEEDS, False,
    )

    # ---- Rank-average blend ----
    oof_rank = np.zeros(len(y), dtype=np.float64)
    te_rank = np.zeros(len(Xte), dtype=np.float64)
    wsum = 0.0
    for fam, (oof, te, _auc) in results.items():
        w = FAMILY_WEIGHTS[fam]
        oof_rank += w * rank_pct(oof)
        te_rank += w * rank_pct(te)
        wsum += w
    oof_rank /= wsum
    te_rank /= wsum

    blend_auc = roc_auc_score(y, oof_rank)
    print("\n" + "=" * 70)
    for fam, (_o, _t, auc) in results.items():
        print(f"{fam:12s} OOF AUC = {auc:.8f}")
    print(f"{'BLEND':12s} OOF AUC = {blend_auc:.8f}")
    print("=" * 70)

    # Map blended ranks onto the CatBoost test-probability distribution so the
    # submission carries calibrated probabilities. AUC is rank-invariant, so the
    # leaderboard score depends only on this ordering, which is preserved here.
    ref_te = results["catboost"][1]
    ref_sorted = np.sort(ref_te)
    ranks = np.argsort(np.argsort(te_rank))
    idx = (ranks / (len(te_rank) - 1) * (len(ref_sorted) - 1)).astype(int)
    blended_probs = ref_sorted[idx]

    submission = pd.read_csv(sample_path) if sample_path else pd.DataFrame({ID_COL: test[ID_COL]})
    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: blended_probs})
    # Align to sample submission order if present.
    if sample_path and ID_COL in submission.columns:
        sub = submission[[ID_COL]].merge(sub, on=ID_COL, how="left")

    out = "submission.csv"
    if os.path.isdir("/kaggle/working"):
        out = "/kaggle/working/submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(sub)} rows)")
    print(f"pred prob range: {blended_probs.min():.4f} .. {blended_probs.max():.4f}")
    print(f"mean: {blended_probs.mean():.4f}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")

    # Save OOF for later stacking if desired.
    np.save("oof_blend.npy", oof_rank)
    np.save("oof_y.npy", y)


if __name__ == "__main__":
    sys.exit(main())
