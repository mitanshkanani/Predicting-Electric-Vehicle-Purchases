

# %% ==================== cell 0 [markdown] ====================
## 📈 Experiments & Results

### 🔢 Digit Features — Win

OOF AUC improved from **0.94204** to **0.94347** (+0.00143).

| Metric | Before digit | After digit | Gain |
|--------|--------------|-------------|------|
| OOF AUC | 0.94204 | **0.94347** | **+0.00143** |

Extracting 56 digit features from 7 numerical columns helped the model capture synthetic generation artifacts. XGBoost now uses many new uncorrelated signals, improving overall ranking.

---

### 📊 Frequency Encoding — Win

OOF AUC improved from **0.94347** to **0.94466** (+0.00119).

| Metric | Before Frequency | After Frequency | Gain |
|--------|------------------|-----------------|------|
| OOF AUC | 0.94347 | **0.94466** | **+0.00119** |

Frequency encoding adds the normalized occurrence rate of each value across train+test. This helps the model identify rare and common patterns, and exposes subtle distributional signals in numerical and digit features that raw values do not reveal.

---

### 🎯 Magic Features — No Effect

4 synthetic-artifact flags were tested and removed. No improvement was observed.

| Metric | Before Magic | After Magic | Change |
|--------|--------------|-------------|--------|
| OOF AUC | **0.94466** | 0.94465 | -0.00001 |

The four flags (`is_30k_spike`, `is_millionaire_cliff`, `is_dead_zone`, `is_env_hater`) captured deterministic synthetic-data artifacts, but XGBoost did not benefit from them.

---

### 🎯 Original Dataset Target Means — Slight Decrease

Added original-dataset target statistics for all features. XGBoost did not improve.

---

### 📂 Numeric-to-Categorical + Frequency — No Effect

Converted all numeric features (including digit features) into string categories and added frequency encoding for them.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| OOF AUC | **0.94466** | 0.94459 | -0.00007 |

This approach was inspired by the top LightGBM solution (0.94587), but XGBoost did not benefit from this representation.

---

### 🚀 Hyperparameter Upgrade — Win

Replaced the old fast-training parameters with a GPU-accelerated, deeply-regularized setup inspired by top LightGBM solutions.

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| learning_rate | 0.03 | **0.005** | Slower but deeper, more stable learning |
| n_estimators | 2000 | **10000** | More room to learn with low LR |
| min_child_weight | 5 | **10** | Stronger regularization against noise |
| subsample | 0.7 | **0.9** | More rows per tree, less variance |
| colsample_bytree | 0.5 | **0.9** | More features per split, richer signal |
| device | cpu | **cuda** | 3-5x faster training on GPU |
| reg_alpha | — | **0.071** | L1 regularization for feature selection |
| reg_lambda | — | **2.0** | L2 regularization for stable leaf weights |
| max_bin | 256 | **1024** | Better split resolution |
| early_stopping_rounds | 100 | **700** | Higher patience for slow learning |

**Result:** OOF AUC improved from **0.94466** to **0.94488** (+0.00022).

---

### 🧹 Feature Selection — Cleaner Model, Same Score

Removed constant and perfectly-correlated features. The model became faster and simpler while keeping the same performance.

| Metric | Before (154 features) | After (76 features) | Change |
|--------|----------------------|---------------------|--------|
| OOF AUC | 0.94488 | **0.94488** | 0.00000 |
| LB | 0.94460 | **0.94460** | 0.00000 |
| Features | 154 | **76** | -78 |

Post-processing tests (Isotonic, Rank, Clip) were also evaluated — no improvement. The cleaner 76-feature model is the final version.

## 🔬 Experiment: 5-fold vs 10-fold CV

Tested whether switching from 10 folds to 5 folds (like top solutions) improves the leaderboard score.

| Metric | 10-fold | 5-fold | Change |
|--------|---------|--------|--------|
| OOF AUC | **0.94488** | 0.94470 | -0.00018 |
| LB | **0.94460** | 0.94459 | -0.00001 |

**Conclusion:** Fold count does not affect leaderboard performance. 10-fold CV remains our validation strategy.

## 🔬 Experiment: Fixed Seed vs Dynamic Seeds

Tested whether using a single fixed seed (42) instead of dynamic per-fold seeds (42+fold) affects the leaderboard score.

| Metric | Dynamic Seeds (42+fold) | Fixed Seed (42) | Change |
|--------|-------------------------|-----------------|--------|
| OOF AUC | **0.94488** | 0.94487 | -0.00001 |
| LB | **0.94460** | 0.94461 | +0.00001 |

**Conclusion:** Seed strategy does not affect performance. Dynamic seeds remain our validation approach. Final model selection will be based on the best CV score, not the public leaderboard.

## 🔬 Experiment: 5-fold vs 10-fold vs 15-fold CV

Tested whether the number of folds affects model performance.

| Metric | 5-fold | 10-fold | 15-fold |
|--------|--------|---------|---------|
| OOF AUC | 0.94470 | 0.94488 | **0.94494** |
| LB | 0.94459 | 0.94460 | **0.94461** |

**Conclusion:** Fold count does not meaningfully affect the leaderboard score. All configurations produce similar results. 10-fold remains our standard.

## 🎯 Triple Target Encoding — Key Improvement

Inspired by the **CTBoost solution** (LB 0.94615), which used three smoothing strengths (`auto`, `10`, `100`) for target encoding. We applied this idea to 13 core columns (7 numeric + 6 label-encoded categories) inside each cross-validation fold.

Combined with **Dual TE** (auto, 10) from Naji's LightGBM solution, this became our **Triple TE** approach.

| Metric | Before TE | After Triple TE | Gain |
|--------|-----------|-----------------|------|
| OOF AUC | 0.94488 | **0.94583** | **+0.00095** |
| LB | 0.94460 | **0.94590** | **+0.00130** |

The three smoothing levels give the model different views of each feature: automatic, light, and heavy smoothing. Applied inside cross-validation to prevent leakage.

All experiments are conducted based on notebooks:

https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94587-lb-0-94612

https://www.kaggle.com/code/lucifer19/ev-quantum-forge-s6e9-xgboost?scriptVersionId=346769553

https://www.kaggle.com/code/kirill0212/s6e9-lightgbm

https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights

https://www.kaggle.com/code/mikhailnaumov/electric-vehicle-purchases-xgb

https://www.kaggle.com/code/maiernator/s6e9-ctboost-not-catboost-astra-baseline

# %% ==================== cell 1 [markdown] ====================
## 📦 1. IMPORTS + SETUP

# %% ==================== cell 2 [code] ====================
import sys
sys.path.append('/kaggle/input/datasets/evgendvorkin/dvorkin-visual-v1/')
from Dvorkin_style_engine import *

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

T0 = time.perf_counter()
def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)

h1("🚗 EV Adoption Prediction — S6E9")
info_card("✅ Setup Complete", "Импорты выполнены, стиль загружен.", style="vi")

# %% ==================== cell 3 [markdown] ====================
## 📥 2. LOAD DATA

# %% ==================== cell 4 [code] ====================
train = pd.read_csv('/kaggle/input/competitions/playground-series-s6e9/train.csv')
test = pd.read_csv('/kaggle/input/competitions/playground-series-s6e9/test.csv')
orig = pd.read_csv('/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv')

h2("📥 Data Loading")
info_card("Dataset Shapes", 
          f"Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape}", 
          style="vi")
pretty_df(train.head(3))

# %% ==================== cell 5 [markdown] ====================
## 🔧 3. PREPROCESSING (BASE)

# %% ==================== cell 6 [code] ====================
cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']

for col in cat_cols:
    mapping = {val: i for i, val in enumerate(train[col].unique())}
    train[f'LE_{col}'] = train[col].map(mapping)
    test[f'LE_{col}'] = test[col].map(mapping)

# Удаляем исходные категориальные колонки
train = train.drop(columns=cat_cols)
test = test.drop(columns=cat_cols)

# Убираем id
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Разделяем X и y
X = train.drop(columns=['Will_Buy_EV'])
y = train['Will_Buy_EV']

h2("🔧 Preprocessing")
info_card("✅ Base Preprocessing Done", f"X shape: {X.shape} | Test shape: {test.shape}", style="vi")

# %% ==================== cell 7 [markdown] ====================
## 🧪 4. FEATURE ENGINEERING — Powerful Interactions

# %% ==================== cell 8 [code] ====================
# Interaction признаки на основе EDA
X['Env_Concern_x_Subsidy'] = X['Environmental_Concern_Level'] * X['LE_Subsidy_Available']
test['Env_Concern_x_Subsidy'] = test['Environmental_Concern_Level'] * test['LE_Subsidy_Available']

X['Env_Concern_x_Income'] = X['Environmental_Concern_Level'] * X['Annual_Income_USD']
test['Env_Concern_x_Income'] = test['Environmental_Concern_Level'] * test['Annual_Income_USD']

X['Subsidy_x_Income'] = X['LE_Subsidy_Available'] * X['Annual_Income_USD']
test['Subsidy_x_Income'] = test['LE_Subsidy_Available'] * test['Annual_Income_USD']

X['Env_Concern_x_Commute'] = X['Environmental_Concern_Level'] * X['Daily_Commute_km']
test['Env_Concern_x_Commute'] = test['Environmental_Concern_Level'] * test['Daily_Commute_km']

X['Home_Charging_x_Subsidy'] = X['LE_Home_Charging_Possible'] * X['LE_Subsidy_Available']
test['Home_Charging_x_Subsidy'] = test['LE_Home_Charging_Possible'] * test['LE_Subsidy_Available']

X['Range_Anxiety_x_Subsidy'] = X['LE_Range_Anxiety_Level'] * X['LE_Subsidy_Available']
test['Range_Anxiety_x_Subsidy'] = test['LE_Range_Anxiety_Level'] * test['LE_Subsidy_Available']

X['Income_per_Concern'] = X['Annual_Income_USD'] / (X['Environmental_Concern_Level'] + 1)
test['Income_per_Concern'] = test['Annual_Income_USD'] / (test['Environmental_Concern_Level'] + 1)

X['Commute_per_Concern'] = X['Daily_Commute_km'] / (X['Environmental_Concern_Level'] + 1)
test['Commute_per_Concern'] = test['Daily_Commute_km'] / (test['Environmental_Concern_Level'] + 1)

print(f"✅ Добавлено interaction признаков")
print(f"X shape: {X.shape}")

# %% ==================== cell 9 [markdown] ====================
## 🔢 5. DIGIT FEATURES — ловим артефакты синтетики

# %% ==================== cell 10 [markdown] ====================
What are digit features?
We extract each digit of numerical features at positions -4 to 3: digit_k(x) = (x // 10^k) % 10. For 7 numeric columns this gives 56 lightweight int8 features.

Why they work:
Synthetic data often contains artifacts in digits (repeated decimals, rounding patterns). Trees struggle to capture these micro-patterns from raw values; explicit digits make them usable.

# %% ==================== cell 11 [code] ====================
# Числовые признаки для извлечения цифр
digit_num_cols = [
    'Age', 'Annual_Income_USD', 'Daily_Commute_km',
    'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
    'Charging_Stations_Near_Work', 'Environmental_Concern_Level'
]

# Извлекаем цифры в позициях от -4 до 3
for col in digit_num_cols:
    for k in range(-4, 4):
        new_col = f'{col}_digit{k}'
        X[new_col] = (X[col].fillna(0) // (10**k) % 10).astype('int8')
        test[new_col] = (test[col].fillna(0) // (10**k) % 10).astype('int8')

print(f"✅ Добавлено digit features: {len(digit_num_cols) * 8}")
print(f"X shape: {X.shape} | Test shape: {test.shape}")

# %% ==================== cell 12 [markdown] ====================
## 📊 6. FREQUENCY ENCODING — частоты для ВСЕХ признаков

# %% ==================== cell 13 [markdown] ====================
What:
Each value in every column is replaced by its normalized frequency across the combined train and test sets. Rare values get small frequencies, common values get large frequencies.

Why it works:
It gives XGBoost a different view of the data: how unusual or typical a value is. In synthetic datasets, generators often leave unusual distributions or rare combinations. Frequency features expose these patterns directly, without many sequential tree splits.

Effect:
OOF AUC improved from 0.94347 to 0.94466.

# %% ==================== cell 14 [code] ====================
# Считаем частоты по train+test для каждой колонки (включая digit и LE)
for column in X.columns:
    freq_map = pd.concat([X[column], test[column]], axis=0).value_counts(normalize=True).to_dict()
    X[f'{column}_freq'] = X[column].map(freq_map).astype('float32').values
    test[f'{column}_freq'] = test[column].map(freq_map).astype('float32').values

print(f"✅ Добавлено frequency features: {len([c for c in X.columns if c.endswith('_freq')])}")
print(f"X shape: {X.shape} | Test shape: {test.shape}")

# %% ==================== cell 15 [markdown] ====================
## 7. 🧹 УДАЛЯЕМ OBJECT КОЛОНКИ

# %% ==================== cell 16 [code] ====================
object_cols = X.select_dtypes(include=['object']).columns.tolist()
X = X.drop(columns=object_cols)
test = test.drop(columns=object_cols)

print(f"Удалено object: {len(object_cols)}")
print(f"X: {X.shape} | test: {test.shape}")

# %% ==================== cell 17 [markdown] ====================
## 8. 🎯 ОТБОР ПРИЗНАКОВ — удаляем мусор


# %% ==================== cell 18 [code] ====================
# Удаляем константные
const_cols = [c for c in X.columns if X[c].nunique() == 1]
X = X.drop(columns=const_cols)
test = test.drop(columns=const_cols)

# Удаляем коррелирующие (corr=1)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] == 1.0)]
X = X.drop(columns=to_drop)
test = test.drop(columns=to_drop)

print(f"✅ Удалено: {len(const_cols)} константных + {len(to_drop)} коррелирующих")
print(f"X: {X.shape} | test: {test.shape}")

del corr_matrix, upper_tri
gc.collect()

# %% ==================== cell 19 [markdown] ====================
## 🚀 9. XGBOOST — BASELINE MODEL (10-fold, dynamic seeds)

# %% ==================== cell 20 [markdown] ====================
🎯 Validation Strategy — 10-Fold CV with Dynamic Seeds

We use **10-fold stratified cross-validation** to get a more stable and honest estimate of model performance.  
On each fold, the random seed is **different** (`42 + fold`), which creates additional diversity across models and reduces the CV–LB gap.

| Setting | Value |
|---------|-------|
| Folds | 10 |
| Split type | StratifiedKFold |
| Shuffle | True |
| Base seed | 42 |
| Per-fold seed | 42 + fold (43, 44, ..., 52) |

This strategy is deliberately stronger than a single-seed 5-fold setup:  

- more training data per fold (90% train / 10% valid),  
- more robust OOF estimate,  
- lower variance across runs,  
- better generalization on the private leaderboard.  

We trust **OOF AUC** as the main optimization target — public LB is treated as a noisy sample.

# %% ==================== cell 21 [markdown] ====================
🎯 Triple Target Encoding — Key Improvement

**What is Triple Target Encoding?**  
Target Encoding replaces each value with the smoothed mean of the target for that value. Triple TE applies this three times with different smoothing strengths: `auto`, `10`, and `100`. Each smoothing level gives the model a different "view" of the same feature — from light smoothing (captures fine-grained patterns) to heavy smoothing (captures stable, general trends).

**Origin:**  
The Dual TE idea (`auto`, `10`) comes from Naji's LightGBM solution (LB 0.94612). We extended it with a third smoothing level (`100`) inspired by the CTBoost solution (LB 0.94615), creating **Triple TE**.

**Why it works:**  
Tree models struggle with high-cardinality features and rare categories. Target Encoding turns categorical and numeric values into smoothed buyer rates, exposing signal that raw values hide. Multiple smoothing levels let XGBoost choose the best bias-variance trade-off for each feature.

**How we applied it:**  
Inside each fold, TE is fitted on training rows only (`fit_transform`) and applied to validation/test rows (`transform`). This prevents label leakage. We encode 13 core columns (7 numeric + 6 label-encoded categories).

**Effect:**  
OOF AUC improved from **0.94488** to **0.94583** (+0.00095). Public LB improved from **0.94460** to **0.94590** (+0.00130).

# %% ==================== cell 22 [code] ====================
import xgboost as xgb
from sklearn.preprocessing import TargetEncoder

# Конвертируем таргет в 0/1
y = y.map({'No': 0, 'Yes': 1})

# Параметры: старые + 10 фолдов + динамические сиды
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.005,
    'max_depth': 7,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'device': 'cuda',
    'reg_alpha': 0.071,
    'reg_lambda': 2.0,
    'max_bin': 1024,
    'n_estimators': 10000,
    'early_stopping_rounds': 500,
    'nthread': -1,
    'deterministic_histogram': True,
}

# Колонки для Triple TE
te_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
           'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
           'Environmental_Concern_Level', 'LE_Gender', 'LE_City_Type', 'LE_Current_Car_Type',
           'LE_Home_Charging_Possible', 'LE_Subsidy_Available', 'LE_Range_Anxiety_Level']

# Читаем исходные данные для TE (один раз)
train_te_raw = pd.read_csv('/kaggle/input/competitions/playground-series-s6e9/train.csv')
test_te_raw = pd.read_csv('/kaggle/input/competitions/playground-series-s6e9/test.csv')
for col in ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']:
    mapping = {val: i for i, val in enumerate(train_te_raw[col].unique())}
    train_te_raw[f'LE_{col}'] = train_te_raw[col].map(mapping)
    test_te_raw[f'LE_{col}'] = test_te_raw[col].map(mapping)

# StratifiedKFold — 10 фолдов
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Массивы для сохранения
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
feature_importance = None

log("Starting 10-fold CV with Triple TE...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test_fold = test.copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    # Triple Target Encoding (auto, 10, 100) по 13 колонкам
    for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
        te = TargetEncoder(shuffle=True, cv=5, smooth=smooth_val, random_state=42)
        X_train_enc = te.fit_transform(train_te_raw.iloc[train_idx][te_cols], y_train).astype('float32')
        X_val_enc = te.transform(train_te_raw.iloc[val_idx][te_cols]).astype('float32')
        X_test_enc = te.transform(test_te_raw[te_cols]).astype('float32')
        
        for i, col in enumerate(te_cols):
            X_train[f'TE_{col}_{smooth_name}'] = X_train_enc[:, i]
            X_val[f'TE_{col}_{smooth_name}'] = X_val_enc[:, i]
            X_test_fold[f'TE_{col}_{smooth_name}'] = X_test_enc[:, i]
    
    current_seed = 42
    params['random_state'] = current_seed
    params['seed'] = current_seed
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    
    if feature_importance is None:
        feature_importance = np.zeros(X_train.shape[1])
    feature_importance += model.feature_importances_ / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    h2(f"Fold {fold}/10")
    md_metric("Fold AUC", f"{fold_auc:.5f}")
    md_metric("Seed", str(current_seed))
    md_metric("Elapsed", f"{time.perf_counter() - T0:.1f}s")

# Overall OOF score
oof_auc = roc_auc_score(y, oof_preds)
h2("🏆 Final OOF Performance")
md_metric("OOF AUC", f"{oof_auc:.5f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

h2("📊 Top 15 Features")
pretty_df(importance_df.head(15))

# %% ==================== cell 23 [markdown] ====================
## 💾 10. SAVE PREDICTIONS

# %% ==================== cell 24 [code] ====================
np.save('oof_preds_base.npy', oof_preds)
np.save('test_preds_base.npy', test_preds)

h2("💾 Save Predictions")
info_card("✅ Predictions Saved", 
          f"OOF: {oof_preds.shape} | Test: {test_preds.shape} | OOF AUC: {oof_auc:.5f}", 
          style="vi")

# %% ==================== cell 25 [markdown] ====================
## 📤 11. CREATE SUBMISSION

# %% ==================== cell 26 [code] ====================
submission = pd.DataFrame({
    'id': pd.read_csv('/kaggle/input/competitions/playground-series-s6e9/test.csv')['id'],
    'Will_Buy_EV': test_preds
})

submission.to_csv('submission.csv', index=False)

h2("📤 Submission Created")
info_card("✅ Submission Saved", f"Shape: {submission.shape}", style="vi")
pretty_df(submission.head(10))

# %% ==================== cell 27 [markdown] ====================
## 📋 12. FUTURE PLAN

# %% ==================== cell 28 [markdown] ====================
## ✅ Completed
- Baseline XGBoost model (10-fold CV)
- Final AUC: 0.94488 (OOF) / 0.94460 (LB)
- Label Encoding for categorical features
- Digit Features (+0.00143)
- Frequency Encoding (+0.00119)
- Hyperparameter Upgrade (+0.00022)
- Feature Selection: 154 → 76 features

## 🔥 Next Steps
- Secret Recipe Features (buy_score, worry_score from Chris Deotte)
- Target Encoding with m-schedule (5, 15, 80)
- Pseudo Labeling
- Level 2 Stacking
- Blending with LightGBM/CatBoost

# %% ==================== cell 29 [code] ====================


# %% ==================== cell 30 [code] ====================
