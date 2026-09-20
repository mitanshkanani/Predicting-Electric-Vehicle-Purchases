"""
Predicting Electric Vehicle Purchases — Kaggle Playground S06E09  (v4)
A DIVERSE model to run on a SECOND machine (Google Colab, 1x T4) while v3 runs
on Kaggle. v3 is GBDT-heavy (CatBoost/XGBoost/LightGBM + target encoding). v4 is
a PyTorch MLP — a different inductive bias — so blending v3 + v4 by rank gives a
REAL gain (not the OOF mirage we saw in v1/v2).

Design goals:
  * Runs on ONE GPU (Colab T4) or CPU; auto-detects.
  * Self-contained: reads train.csv/test.csv from the current directory (where
    Colab's Files panel puts them) or /kaggle/input.
  * Saves submission_v4.csv AND v4_oof.npy / v4_test.npy so we can stack/blend
    with v3 afterwards.
  * Falls back to a sklearn LogisticRegression if torch is unavailable, so the
    run never dies partway.

After BOTH v3 and v4 finish, blend them (see merge_note at the bottom).
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
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr

warnings.filterwarnings("ignore")

TARGET = "Will_Buy_EV"
ID_COL = "id"
VERSION = "v4"
N_SPLITS = 10
SEEDS = (42, 7)          # two NN seeds, averaged
BATCH = 8192
MAX_EPOCHS = 40
PATIENCE = 5
TE_SMOOTHINGS = (50.0,)  # high smoothing = robust, low overfit (for the NN)
RANDOM_STATE = 42


def find_data_dir():
    candidates = [os.getcwd(), "."]
    if os.path.isdir("/kaggle/input"):
        for root in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", root)
            if os.path.isdir(base):
                candidates.append(base)
    candidates.append("/kaggle/input/datasets/mitanshkanani/dataset-for-experimentation-final")

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
        raise FileNotFoundError("Could not find train.csv / test.csv. Upload them to the Colab session.")
    return train_path, test_path, sample_path


def encode_target(series):
    vals = {str(v).strip().lower() for v in series.unique()}
    if vals <= {"yes", "no"}:
        return (series.astype(str).str.strip().str.lower() == "yes").astype(np.int8).to_numpy()
    if vals <= {"1", "0"}:
        return series.astype(int).to_numpy()
    pos = series.value_counts().idxmin()
    return (series == pos).astype(np.int8).to_numpy()


# ----------------------------------------------------------------------------
# Features for the NN: numeric + one-hot categoricals + light, robust TE.
# Deliberately simpler than v3's feature space -> more diversity in the blend.
# ----------------------------------------------------------------------------
NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]


def build_nn_base(df):
    x = df[NUMERIC].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    # A few engineered numerics the NN can exploit.
    inc = x["Annual_Income_USD"]
    age = x["Age"]
    cars = x["Number_of_Cars_Owned"]
    tot = x["Charging_Stations_Near_Home"] + x["Charging_Stations_Near_Work"]
    x["log_income"] = np.log1p(inc)
    x["income_per_car"] = inc / (cars + 1.0)
    x["income_per_age"] = inc / (age + 1.0)
    x["total_charging"] = tot
    x["commute_per_charging"] = x["Daily_Commute_km"] / (tot + 1.0)
    # Ordinal anxiety + binary flags.
    anx = df["Range_Anxiety_Level"].astype(str).str.lower().map({"low": 0, "medium": 1, "high": 2}).fillna(-1)
    x["anxiety_ord"] = anx.astype(np.float32)
    x["subsidy_flag"] = (df["Subsidy_Available"].astype(str).str.lower() == "yes").astype(np.float32)
    x["home_charge_flag"] = (df["Home_Charging_Possible"].astype(str).str.lower() == "yes").astype(np.float32)
    # One-hot categoricals.
    oh = pd.get_dummies(df[CATEGORICAL].astype(str), columns=CATEGORICAL, prefix=CATEGORICAL)
    x = pd.concat([x, oh.astype(np.float32)], axis=1)
    return x


def _te_key_exact(df, col):
    if col == "Annual_Income_USD":
        arr = np.rint(pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float64)).astype(np.int64)
        return pd.Series(arr, index=df.index).astype(str)
    arr = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float64)
    return pd.Series([f"{v:.1f}" for v in arr], index=df.index)


def _fit_te(keys, y, prior, smoothing):
    frame = pd.DataFrame({"k": keys.to_numpy(), "t": y})
    st = frame.groupby("k", observed=True)["t"].agg(["sum", "count"])
    return (st["sum"] + smoothing * prior) / (st["count"] + smoothing)


def add_light_te(train, test, y, fold_ids):
    for col in ["Annual_Income_USD", "Daily_Commute_km"]:
        tr_keys = _te_key_exact(train, col)
        te_keys = _te_key_exact(test, col)
        for m in TE_SMOOTHINGS:
            name = f"te_{col}_m{int(m)}"
            enc = np.full(len(train), np.nan, np.float32)
            for f in range(N_SPLITS):
                trm = fold_ids != f
                vam = fold_ids == f
                prior = float(y[trm].mean())
                mp = _fit_te(tr_keys[trm], y[trm], prior, m)
                enc[vam] = tr_keys[vam].map(mp).fillna(prior).to_numpy(np.float32)
            train[name] = enc
            full = _fit_te(tr_keys, y, float(y.mean()), m)
            test[name] = te_keys.map(full).fillna(float(y.mean())).to_numpy(np.float32)
    return train, test


# ----------------------------------------------------------------------------
# PyTorch MLP (with sklearn fallback).
# ----------------------------------------------------------------------------
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


def rank_pct(a):
    return pd.Series(a).rank(method="average", pct=True).to_numpy()


def main():
    t0 = time.perf_counter()
    train_path, test_path, sample_path = find_data_dir()
    print(f"train: {train_path}\ntest : {test_path}")

    torch_mod = _try_import_torch()
    if torch_mod is not None:
        torch = torch_mod[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model: PyTorch MLP on {device}")
    else:
        device = "cpu"
        print("Model: sklearn LogisticRegression fallback (torch not found)")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = encode_target(train[TARGET])
    print(f"train {train.shape} test {test.shape} pos_rate {y.mean():.4f}")

    Xtr = build_nn_base(train)
    Xte = build_nn_base(test)
    # Align one-hot columns (test may miss a category).
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1)
    Xtr = Xtr.fillna(0.0)
    Xte = Xte.fillna(0.0)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for f, (_, va_idx) in enumerate(skf.split(Xtr, y)):
        fold_ids[va_idx] = f

    Xtr, Xte = add_light_te(Xtr, Xte, y, fold_ids)
    cols = list(Xtr.columns)
    print(f"features: {len(cols)}", flush=True)

    oof_sum = np.zeros(len(y))
    te_sum = np.zeros(len(Xte))
    for seed in SEEDS:
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
            print(f"    [nn seed={seed} fold={f}] auc={fauc:.6f}", flush=True)
        print(f"  [nn] seed={seed} OOF AUC = {roc_auc_score(y, oof):.8f}", flush=True)
        oof_sum += oof
        te_sum += np.mean(np.vstack(tes), axis=0)
        del oof, tes
        gc.collect()

    oof_avg = oof_sum / len(SEEDS)
    te_avg = te_sum / len(SEEDS)
    nn_auc = roc_auc_score(y, oof_avg)
    print("=" * 70)
    print(f"v4 NN OOF AUC = {nn_auc:.8f}")
    print("=" * 70, flush=True)

    np.save("v4_oof.npy", oof_avg)
    np.save("v4_test.npy", te_avg)

    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: te_avg})
    if sample_path:
        submission = pd.read_csv(sample_path)
        if ID_COL in submission.columns:
            sub = submission[[ID_COL]].merge(sub, on=ID_COL, how="left")
    sub.to_csv(f"submission_{VERSION}.csv", index=False)
    print(f"Wrote submission_{VERSION}.csv ({len(sub)} rows) | prob {te_avg.min():.4f}..{te_avg.max():.4f} mean {te_avg.mean():.4f}")
    print(f"Runtime: {time.perf_counter() - t0:.1f}s")


# ----------------------------------------------------------------------------
# merge_note: after BOTH v3 (Kaggle) and v4 (Colab) finish, blend them on Kaggle
# (or anywhere) with this tiny snippet. AUC is rank-based, so rank-averaging the
# two submissions is a valid, robust blend:
#
#   import pandas as pd, numpy as np
#   a = pd.read_csv("submission_v3.csv"); b = pd.read_csv("submission_v4.csv")
#   ra = a["Will_Buy_EV"].rank(pct=True); rb = b["Will_Buy_EV"].rank(pct=True)
#   out = a.copy(); out["Will_Buy_EV"] = (ra + rb) / 2
#   out.to_csv("submission_v3v4.csv", index=False)
#
# Weight it toward whichever has the better LB score, e.g. 0.7*ra + 0.3*rb.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
