
from __future__ import annotations
import hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXPECTED_HASH="55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
EXPECTED={"old_cat":0.94553699,"new_cat":0.94558572,"xgb":0.94591178,"lgbm":0.94578042}
CONTROL_META=0.94595446
TOL=2e-5
STEP=0.05

FOLDS=Path("artifacts/validation/candidate_folds.csv")
OLD_CAT=Path("artifacts/experiments/catboost_hierarchical_income_multiseed_gpu/best_average_oof_predictions.csv")
NEW_CAT=Path("artifacts/experiments/catboost_hierarchical_income_commute_multiseed_gpu/best_average_oof_predictions.csv")
XGB=Path("artifacts/experiments/xgboost_hierarchical_commute_te_gpu/oof_predictions.csv")
LGBM=Path("artifacts/experiments/lightgbm_engineered_learned_margin_cpu/oof_predictions.csv")

OLD_CAT_TEST=Path("artifacts/experiments/catboost_hierarchical_income_multiseed_gpu/best_average_test_predictions.csv")
NEW_CAT_TEST=Path("artifacts/experiments/catboost_hierarchical_income_commute_multiseed_gpu/best_average_test_predictions.csv")
XGB_TEST=Path("artifacts/experiments/xgboost_hierarchical_commute_te_gpu/test_predictions.csv")
LGBM_TEST=Path("artifacts/experiments/lightgbm_engineered_learned_margin_cpu/test_predictions.csv")

OUT=Path("artifacts/experiments/blend_income_commute_catboost_commute_xgb_lgbm_rank_audit")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def pred_col(df, kind):
    pref=["oof_prediction","prediction","rank_average_prediction","best_average_prediction"] if kind=="oof" else ["prediction","test_prediction","rank_average_prediction","best_average_prediction"]
    for c in pref:
        if c in df.columns: return c
    exclude={"row_index","fold","target","target_encoded","id"}
    nums=[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(nums)!=1: raise ValueError(f"Cannot identify prediction column: {list(df.columns)}")
    return nums[0]

def load_oof(path,n,folds):
    df=pd.read_csv(path)
    if len(df)!=n: raise ValueError(f"row mismatch: {path}")
    if "row_index" not in df or not np.array_equal(df.row_index.to_numpy(),np.arange(n)): raise ValueError(f"row order mismatch: {path}")
    if "fold" not in df or not np.array_equal(df.fold.to_numpy(),folds): raise ValueError(f"fold mismatch: {path}")
    p=df[pred_col(df,"oof")].to_numpy(float)
    if not np.isfinite(p).all(): raise ValueError(f"non-finite predictions: {path}")
    return p

def load_test(path,n):
    df=pd.read_csv(path)
    if len(df)!=n: raise ValueError(f"test row mismatch: {path}")
    p=df[pred_col(df,"test")].to_numpy(float)
    if not np.isfinite(p).all(): raise ValueError(f"non-finite test predictions: {path}")
    return p

def rank(v):
    return pd.Series(v).rank(method="average",pct=True).to_numpy(float)

def fold_rank(v,folds):
    out=np.empty(len(v))
    for f in range(5):
        m=folds==f
        out[m]=rank(v[m])
    return out

def grid():
    u=int(round(1/STEP))
    return [(c/u,x/u,l/u) for l in range(u+1) for c in range(u-l+1) for x in [u-l-c]]

def choose(y,cat,xgb,lgb,mask,g):
    best=(-1,None)
    for wc,wx,wl in g:
        auc=roc_auc_score(y[mask],wc*cat[mask]+wx*xgb[mask]+wl*lgb[mask])
        if auc>best[0]+1e-12: best=(auc,(wc,wx,wl))
    return (*best[1],best[0])

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
fold_df=pd.read_csv(FOLDS)

if sha256(FOLDS)!=EXPECTED_HASH: raise ValueError("Frozen fold SHA mismatch")
if len(fold_df)!=len(train): raise ValueError("fold count mismatch")
folds=fold_df["fold"].to_numpy(int)

target=[c for c in train.columns if c not in test.columns]
if len(target)!=1: raise ValueError(f"target detection failed: {target}")
target=target[0]
y=(train[target].astype(str).str.strip().str.lower()=="yes").astype(np.int8).to_numpy()

old_cat=load_oof(OLD_CAT,len(train),folds)
new_cat=load_oof(NEW_CAT,len(train),folds)
xgb=load_oof(XGB,len(train),folds)
lgbm=load_oof(LGBM,len(train),folds)

for name,p in [("old_cat",old_cat),("new_cat",new_cat),("xgb",xgb),("lgbm",lgbm)]:
    auc=roc_auc_score(y,p)
    if abs(auc-EXPECTED[name])>TOL: raise ValueError(f"{name} AUC mismatch: {auc:.8f} vs {EXPECTED[name]:.8f}")

old_cat_r,new_cat_r,xgb_r,lgbm_r=[fold_rank(v,folds) for v in (old_cat,new_cat,xgb,lgbm)]
g=grid()

print("="*100)
print("ENSEMBLE UPDATE: HIERARCHICAL-INCOME CAT3 -> INCOME+COMMUTE CAT3")
print("="*100)
print(f"Frozen fold SHA256 verified: {EXPECTED_HASH}")
print(f"Old Cat3 : {roc_auc_score(y,old_cat):.8f}")
print(f"New Cat3 : {roc_auc_score(y,new_cat):.8f}")
print(f"XGB      : {roc_auc_score(y,xgb):.8f}")
print(f"LightGBM : {roc_auc_score(y,lgbm):.8f}")
print(f"old CAT vs new CAT rank corr: {np.corrcoef(rank(old_cat),rank(new_cat))[0,1]:.6f}")
print()

ctrl=np.full(len(train),np.nan)
cand=np.full(len(train),np.nan)
rows=[]
t0=time.time()

for held in range(5):
    fit=folds!=held
    val=folds==held
    oc,ox,ol,fitauc=choose(y,old_cat_r,xgb_r,lgbm_r,fit,g)
    nc,nx,nl,nfitauc=choose(y,new_cat_r,xgb_r,lgbm_r,fit,g)
    cp=oc*old_cat_r[val]+ox*xgb_r[val]+ol*lgbm_r[val]
    npred=nc*new_cat_r[val]+nx*xgb_r[val]+nl*lgbm_r[val]
    ca=roc_auc_score(y[val],cp)
    na=roc_auc_score(y[val],npred)
    ctrl[val]=cp
    cand[val]=npred
    d=na-ca
    rows.append([held,oc,ox,ol,ca,nc,nx,nl,na,d])
    print(f"Fold {held}: control={ca:.8f} ({oc:.2f}/{ox:.2f}/{ol:.2f}) -> candidate={na:.8f} ({nc:.2f}/{nx:.2f}/{nl:.2f}) {d:+.8f}")

fm=pd.DataFrame(rows,columns=[
    "held_fold","control_old_cat_weight","control_xgb_weight","control_lgbm_weight","control_held_auc",
    "candidate_new_cat_weight","candidate_xgb_weight","candidate_lgbm_weight","candidate_held_auc","candidate_delta_vs_control"
])

ctrl_auc=roc_auc_score(y,ctrl)
cand_auc=roc_auc_score(y,cand)
delta=cand_auc-ctrl_auc

if abs(ctrl_auc-CONTROL_META)>2e-5:
    raise ValueError(f"Control reproduction mismatch: {ctrl_auc:.8f} vs {CONTROL_META:.8f}")

improved=int((fm.candidate_delta_vs_control>0).sum())
worse=int((fm.candidate_delta_vs_control<0).sum())

weights=np.array([
    fm.candidate_new_cat_weight.mean(),
    fm.candidate_xgb_weight.mean(),
    fm.candidate_lgbm_weight.mean(),
])
weights/=weights.sum()

decision="KEEP_INCOME_COMMUTE_CATBOOST_IN_3MODEL_ENSEMBLE" if delta>0 and improved>=4 else "REJECT_INCOME_COMMUTE_CATBOOST_ENSEMBLE_UPDATE"

OUT.mkdir(parents=True,exist_ok=True)
fm.to_csv(OUT/"meta_fold_metrics.csv",index=False)

pd.DataFrame({
    "row_index":np.arange(len(train)),
    "fold":folds,
    "target_encoded":y,
    "control_meta_oof_prediction":ctrl,
    "candidate_meta_oof_prediction":cand,
}).to_csv(OUT/"oof_predictions.csv",index=False)

new_cat_test=rank(load_test(NEW_CAT_TEST,len(test)))
xgb_test=rank(load_test(XGB_TEST,len(test)))
lgbm_test=rank(load_test(LGBM_TEST,len(test)))
tp=weights[0]*new_cat_test+weights[1]*xgb_test+weights[2]*lgbm_test

test_out=pd.DataFrame({"prediction":tp.astype(np.float32)})
if "id" in test.columns:
    test_out.insert(0,"id",test["id"].to_numpy())
test_out.to_csv(OUT/"test_predictions.csv",index=False)

summary=[
"EXPERIMENT: INCOME+COMMUTE CATBOOST ENSEMBLE UPDATE",
"="*80,
f"Control meta-CV: {ctrl_auc:.8f}",
f"Candidate meta-CV: {cand_auc:.8f}",
f"Delta: {delta:+.8f}",
f"Folds improved: {improved}/5",
f"Folds worse: {worse}/5",
f"Mean weights: CAT={weights[0]:.4f}, XGB={weights[1]:.4f}, LGBM={weights[2]:.4f}",
f"Decision: {decision}",
"",
"FOLD DETAILS",
]
for r in fm.itertuples():
    summary.append(
        f"Fold {r.held_fold}: {r.control_held_auc:.8f} -> {r.candidate_held_auc:.8f} ({r.candidate_delta_vs_control:+.8f})"
    )
(OUT/"summary.txt").write_text("\n".join(summary),encoding="utf-8")

print()
print("="*100)
print(f"Control current 3-model : {ctrl_auc:.8f}")
print(f"Candidate updated model : {cand_auc:.8f}")
print(f"Delta vs control        : {delta:+.8f}")
print(f"Folds improved/worse    : {improved}/{worse}")
print(f"Mean weights            : CAT={weights[0]:.4f}, XGB={weights[1]:.4f}, LGBM={weights[2]:.4f}")
print(f"Decision                : {decision}")
print(f"Artifacts               : {OUT.resolve()}")
print("="*100)
print("Send me terminal output, summary.txt, and meta_fold_metrics.csv")
