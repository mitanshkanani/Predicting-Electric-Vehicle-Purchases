from __future__ import annotations
import hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm_engineered_learned_margin_cpu as feat
import xgboost_hierarchical_income_te_gpu as income_hte
import xgboost_hierarchical_commute_te_gpu as commute_hte

EXPECTED_HASH="55223bc6f44db9b6292f91c3c11cdd7d75175a5e04774f7e0dd0ac512ec0e6ee"
EXPECTED_SEED42_AUC=0.94591178
TOL=2e-5
SMOOTHING=2.0
TRAIN_SEEDS=[7,2026]
ALL_SEEDS=[42,7,2026]

FOLDS=Path("artifacts/validation/candidate_folds.csv")
SEED42_OOF=Path("artifacts/experiments/xgboost_hierarchical_commute_te_gpu/oof_predictions.csv")
SEED42_TEST=Path("artifacts/experiments/xgboost_hierarchical_commute_te_gpu/test_predictions.csv")
OUT=Path("artifacts/experiments/xgboost_hierarchical_income_commute_multiseed_gpu")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def rank(v):
    return pd.Series(v).rank(method="average",pct=True).to_numpy(float)

def corr_rank(a,b):
    return float(np.corrcoef(rank(a),rank(b))[0,1])

def pred_col(df,kind):
    pref=["oof_prediction","prediction","rank_average_prediction","best_average_prediction"] if kind=="oof" else ["prediction","test_prediction","rank_average_prediction","best_average_prediction"]
    for c in pref:
        if c in df.columns: return c
    exclude={"row_index","fold","target","target_encoded","id"}
    nums=[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(nums)!=1: raise ValueError(f"Cannot identify prediction column: {list(df.columns)}")
    return nums[0]

def load_oof(path,train,folds):
    df=pd.read_csv(path)
    if len(df)!=len(train): raise ValueError(f"row mismatch: {path}")
    if "row_index" not in df or not np.array_equal(df.row_index.to_numpy(),np.arange(len(train))): raise ValueError(f"row order mismatch: {path}")
    if "fold" not in df or not np.array_equal(df.fold.to_numpy(),folds): raise ValueError(f"fold mismatch: {path}")
    p=df[pred_col(df,"oof")].to_numpy(float)
    if not np.isfinite(p).all(): raise ValueError(f"non-finite oof: {path}")
    return p

def load_test(path,test):
    df=pd.read_csv(path)
    if len(df)!=len(test): raise ValueError(f"test row mismatch: {path}")
    if "id" in df.columns and "id" in test.columns and not np.array_equal(df.id.to_numpy(),test.id.to_numpy()): raise ValueError(f"id mismatch: {path}")
    p=df[pred_col(df,"test")].to_numpy(float)
    if not np.isfinite(p).all(): raise ValueError(f"non-finite test: {path}")
    return p

def save_oof(path,train,folds,y,p):
    df=pd.DataFrame({"row_index":np.arange(len(train),dtype=np.int64),"fold":folds,"target_encoded":y,"oof_prediction":p.astype(np.float32)})
    if "id" in train.columns: df.insert(1,"id",train.id.to_numpy())
    df.to_csv(path,index=False)

def save_test(path,test,p):
    df=pd.DataFrame({"prediction":p.astype(np.float32)})
    if "id" in test.columns: df.insert(0,"id",test.id.to_numpy())
    df.to_csv(path,index=False)

def build_model(seed):
    m=income_hte.build_model()
    m.set_params(random_state=seed,seed=seed)
    return m

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
fold_df=pd.read_csv(FOLDS)

if sha256(FOLDS)!=EXPECTED_HASH: raise ValueError("Frozen fold SHA mismatch")
target=feat.detect_target(train,test)
y,_=feat.encode_binary_target(train[target])
folds,id_col=feat.validate_folds(fold_df,train)

seed42_oof=load_oof(SEED42_OOF,train,folds)
seed42_test=load_test(SEED42_TEST,test)
seed42_auc=float(roc_auc_score(y,seed42_oof))
if abs(seed42_auc-EXPECTED_SEED42_AUC)>TOL: raise ValueError(f"seed42 AUC mismatch: {seed42_auc:.8f}")

raw_features=[c for c in test.columns if c in train.columns and c!=id_col]
raw_cats=feat.detect_raw_categoricals(train,raw_features)
X0,X0t=feat.prepare_base_frames(train=train,test=test,raw_features=raw_features,categorical_features=raw_cats)
X1,X1t,_=feat.add_income_digit_features(X_train=X0,X_test=X0t,train_source=train,test_source=test)
Xbase,Xbaset,_=feat.add_exact_frequency_features(X_train=X1,X_test=X1t,train_source=train,test_source=test)
log_train=feat.build_logistic_recipe_matrix(train)
log_test=feat.build_logistic_recipe_matrix(test)

OUT.mkdir(parents=True,exist_ok=True)

print("="*100)
print("XGBOOST HIERARCHICAL-INCOME+COMMUTE MULTI-SEED AUDIT")
print("="*100)
print(f"Frozen fold SHA256 verified: {EXPECTED_HASH}")
print(f"Seed-42 current champion: {seed42_auc:.8f}")
print("Only change: reuse seed 42, train seeds 7 and 2026, compare probability/rank averaging.")
print()

seed_oof={42:seed42_oof}
seed_test={42:seed42_test}
seed_auc={42:seed42_auc}
fold_rows=[]
t0=time.perf_counter()

for seed in TRAIN_SEEDS:
    print("="*100)
    print(f"TRAINING XGBOOST SEED {seed}")
    print("="*100)
    oof=np.full(len(train),np.nan,float)
    test_preds=[]

    for outer in range(5):
        tr=np.flatnonzero(folds!=outer)
        va=np.flatnonzero(folds==outer)

        tr_exact,va_exact,te_exact,_=feat.build_exact_te_for_outer_fold(
            train=train,test=test,y=y,fold_ids=folds,outer_fold=outer,smoothing=SMOOTHING
        )
        tr_inc,va_inc,te_inc,_=income_hte.build_hierarchical_income_te_for_outer_fold(
            train=train,test=test,y=y,fold_ids=folds,outer_fold=outer,smoothing=SMOOTHING
        )
        tr_com,va_com,te_com,_=commute_hte.build_hierarchical_commute_te_for_outer_fold(
            train=train,test=test,y=y,fold_ids=folds,outer_fold=outer,smoothing=SMOOTHING
        )
        bm_tr,bm_va,bm_te,_=feat.build_learned_logistic_margins_for_outer_fold(
            train_matrix=log_train,test_matrix=log_test,y=y,fold_ids=folds,outer_fold=outer
        )

        Xtr=Xbase.iloc[tr].reset_index(drop=True).copy()
        Xva=Xbase.iloc[va].reset_index(drop=True).copy()
        Xte=Xbaset.reset_index(drop=True).copy()

        for a,b,c in [(tr_exact,va_exact,te_exact),(tr_inc,va_inc,te_inc),(tr_com,va_com,te_com)]:
            for col in a.columns:
                Xtr[col]=a[col].to_numpy(np.float32)
                Xva[col]=b[col].to_numpy(np.float32)
                Xte[col]=c[col].to_numpy(np.float32)

        model=build_model(seed)
        model.fit(
            Xtr,y[tr],
            base_margin=bm_tr,
            eval_set=[(Xva,y[va])],
            base_margin_eval_set=[bm_va],
            verbose=False,
        )

        best=-1 if model.best_iteration is None else int(model.best_iteration)
        ir=None if best<0 else (0,best+1)
        pva=model.predict_proba(Xva,base_margin=bm_va,iteration_range=ir)[:,1]
        pte=model.predict_proba(Xte,base_margin=bm_te,iteration_range=ir)[:,1]

        oof[va]=pva
        test_preds.append(pte.astype(np.float32))

        f_auc=float(roc_auc_score(y[va],pva))
        base_auc=float(roc_auc_score(y[va],seed42_oof[va]))
        d=f_auc-base_auc
        fold_rows.append({"seed":seed,"fold":outer,"seed42_auc":base_auc,"seed_auc":f_auc,"delta_vs_seed42":d,"best_iteration":best})
        print(f"Seed {seed} fold {outer}: seed42={base_auc:.8f} -> seed{seed}={f_auc:.8f} ({d:+.8f}) | best_iter={best}")

    if np.isnan(oof).any(): raise RuntimeError(f"NaN OOF for seed {seed}")
    tp=np.mean(np.vstack(test_preds),axis=0)
    auc=float(roc_auc_score(y,oof))
    seed_oof[seed]=oof
    seed_test[seed]=tp
    seed_auc[seed]=auc
    save_oof(OUT/f"oof_predictions_seed{seed}.csv",train,folds,y,oof)
    save_test(OUT/f"test_predictions_seed{seed}.csv",test,tp)
    print(f"Seed {seed} overall OOF AUC: {auc:.8f}")
    print()

runtime=time.perf_counter()-t0

prob_oof=np.mean(np.vstack([seed_oof[s] for s in ALL_SEEDS]),axis=0)
prob_test=np.mean(np.vstack([seed_test[s] for s in ALL_SEEDS]),axis=0)
prob_auc=float(roc_auc_score(y,prob_oof))

rank_oof=np.mean(np.vstack([rank(seed_oof[s]) for s in ALL_SEEDS]),axis=0)
rank_test=np.mean(np.vstack([rank(seed_test[s]) for s in ALL_SEEDS]),axis=0)
rank_auc=float(roc_auc_score(y,rank_oof))

if rank_auc>=prob_auc:
    method="rank_average"; best_oof=rank_oof; best_test=rank_test; best_auc=rank_auc
else:
    method="probability_average"; best_oof=prob_oof; best_test=prob_test; best_auc=prob_auc

delta=best_auc-seed42_auc
ens_rows=[]
improved=0
for f in range(5):
    m=folds==f
    a0=float(roc_auc_score(y[m],seed42_oof[m]))
    a1=float(roc_auc_score(y[m],best_oof[m]))
    d=a1-a0
    improved+=int(d>0)
    ens_rows.append({"fold":f,"seed42_auc":a0,"best_ensemble_auc":a1,"delta_vs_seed42":d})

pairs=[]
for i,a in enumerate(ALL_SEEDS):
    for b in ALL_SEEDS[i+1:]:
        pairs.append({"seed_a":a,"seed_b":b,"probability_corr":float(np.corrcoef(seed_oof[a],seed_oof[b])[0,1]),"rank_corr":corr_rank(seed_oof[a],seed_oof[b])})

decision="KEEP_HIERARCHICAL_INCOME_COMMUTE_XGB_3SEED" if delta>0 and improved>=3 else "REJECT_HIERARCHICAL_INCOME_COMMUTE_XGB_3SEED"

pd.DataFrame(fold_rows).to_csv(OUT/"seed_fold_metrics.csv",index=False)
pd.DataFrame(ens_rows).to_csv(OUT/"ensemble_fold_metrics.csv",index=False)
pd.DataFrame(pairs).to_csv(OUT/"seed_pair_correlations.csv",index=False)
pd.DataFrame([
    {"model":"seed_42","oof_auc":seed_auc[42]},
    {"model":"seed_7","oof_auc":seed_auc[7]},
    {"model":"seed_2026","oof_auc":seed_auc[2026]},
    {"model":"probability_average","oof_auc":prob_auc},
    {"model":"rank_average","oof_auc":rank_auc},
]).to_csv(OUT/"seed_scores.csv",index=False)

save_oof(OUT/"probability_average_oof_predictions.csv",train,folds,y,prob_oof)
save_test(OUT/"probability_average_test_predictions.csv",test,prob_test)
save_oof(OUT/"rank_average_oof_predictions.csv",train,folds,y,rank_oof)
save_test(OUT/"rank_average_test_predictions.csv",test,rank_test)
save_oof(OUT/"best_average_oof_predictions.csv",train,folds,y,best_oof)
save_test(OUT/"best_average_test_predictions.csv",test,best_test)

summary=[
"EXPERIMENT: HIERARCHICAL-INCOME+COMMUTE XGBOOST MULTI-SEED",
"="*80,
f"Seed 42: {seed_auc[42]:.8f}",
f"Seed 7: {seed_auc[7]:.8f}",
f"Seed 2026: {seed_auc[2026]:.8f}",
f"Probability average: {prob_auc:.8f}",
f"Rank average: {rank_auc:.8f}",
f"Best method: {method}",
f"Best 3-seed AUC: {best_auc:.8f}",
f"Delta vs seed42: {delta:+.8f}",
f"Folds improved vs seed42: {improved}/5",
f"Runtime for new seeds: {runtime:.2f}s",
f"Decision: {decision}",
"",
"FOLD RESULTS",
]
for r in ens_rows:
    summary.append(f"Fold {r['fold']}: {r['seed42_auc']:.8f} -> {r['best_ensemble_auc']:.8f} ({r['delta_vs_seed42']:+.8f})")
(OUT/"summary.txt").write_text("\n".join(summary),encoding="utf-8")

print("="*100)
print("XGBOOST MULTI-SEED AUDIT COMPLETE")
print("="*100)
print(f"Seed 42             : {seed_auc[42]:.8f}")
print(f"Seed 7              : {seed_auc[7]:.8f}")
print(f"Seed 2026           : {seed_auc[2026]:.8f}")
print(f"Probability average : {prob_auc:.8f}")
print(f"Rank average        : {rank_auc:.8f}")
print(f"Best method         : {method}")
print(f"Best 3-seed AUC     : {best_auc:.8f}")
print(f"Delta vs seed42     : {delta:+.8f}")
print(f"Folds improved      : {improved}/5")
print(f"Decision            : {decision}")
print(f"Artifacts           : {OUT.resolve()}")
print("="*100)
print("Send me terminal output, summary.txt, seed_scores.csv, ensemble_fold_metrics.csv, seed_pair_correlations.csv")
