

# %% ==================== cell 0 [markdown] ====================
# Import

# %% ==================== cell 1 [code] ====================
import numpy as np 
import polars as pl 
import os
from tqdm import tqdm
import copy
import pandas as pd

import kagglehub

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

from matplotlib import pyplot as plt

kagglehub.dataset_download("itzzomkar/ev-adoption-behavior-and-range-anxiety")

# %% ==================== cell 2 [markdown] ====================
# Gauss smoothing encoder

# %% ==================== cell 3 [code] ====================
# This section “with a little help of my friend”  by DeepSeek
def smoothed_target_encoding(X_train, y_train, col, kernel_param=35.0):
    """
    Smoothed target encoding for col.
    return dict {cat: smoothed mean}.
    """
    # Aggregation
    grouped = y_train.groupby(X_train[col]).agg(['mean', 'count']).reset_index()
    cats = grouped[col].values.astype(float)   # unique
    means = grouped['mean'].values.astype(float)
    counts = grouped['count'].values.astype(float)

    # matrix of differences  (kernel)
    diff = cats[:, None] - cats[None, :]
    K = np.exp(-kernel_param * diff * diff)

    # Weighted average
    numerator = np.sum(K * (means * counts)[None, :], axis=1)
    denominator = np.sum(K * counts[None, :], axis=1)
    denominator = np.maximum(denominator, 1e-12)   
    smoothed = numerator / denominator

    return dict(zip(cats, smoothed))

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with Gaussian kernel smoothing and out-of-fold fitting.
    Parameters:
        kernel_param (float): gauss kernel.
        cv (int): folds for cv.
        shuffle (bool): shuffle.
        random_state (int).
    """
    def __init__(self, kernel_param=35.0, cv=5, shuffle=True, random_state=42):
        self.kernel_param = kernel_param
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.global_mean_ = None
        self.encoding_ = None   

    def _smooth_encoding(self, X_col, y):
        """
       return smooted mean
        """
        grouped = y.groupby(X_col).agg(['mean', 'count']).reset_index()

        cats = grouped.iloc[:, 0].values.astype(float)
        means = grouped['mean'].values.astype(float)
        counts = grouped['count'].values.astype(float)


        if len(cats) == 1:
            return {cats[0]: means[0]}

        # Difference Matrix
        diff = cats[:, None] - cats[None, :]  #- np.eye(len(cats)) # 2 ways possible - exclude itself and make more sharp gauss or include itseld and more smooth gauss - more or less same result
        K = np.exp(-self.kernel_param * diff * diff)

        numerator = np.sum(K * (means * counts)[None, :], axis=1)
        denominator = np.sum(K * counts[None, :], axis=1)
        denominator = np.maximum(denominator, 1e-12)
        smoothed = numerator / denominator
        return dict(zip(cats, smoothed))

    def fit(self, X, y):
        """
        Global on all data (not use)
        """
        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, 0]
        else:
            X_col = X
        self.global_mean_ = y.mean()
        self.encoding_ = self._smooth_encoding(X_col, y)
        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, 0]
        else:
            X_col = X
        return X_col.map(self.encoding_).fillna(self.global_mean_).values

    def fit_transform(self, X, y):

        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, 0]
        else:
            X_col = X

        # for future transform
        self.global_mean_ = y.mean()
        self.encoding_ = self._smooth_encoding(X_col, y)

        n = len(X_col)
        oof = np.zeros(n)

        # CV
        kf =StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)  

        for train_idx, val_idx in tqdm(kf.split(X_col, y)):
            X_train_fold = X_col.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_col.iloc[val_idx]

            # learn on train
            enc_fold = self._smooth_encoding(X_train_fold, y_train_fold)
            # apply to fold
            oof[val_idx] = X_val_fold.map(enc_fold).fillna(self.global_mean_).values
        return oof


# %% ==================== cell 4 [markdown] ====================
# Make features

# %% ==================== cell 5 [code] ====================
TARGET = "Will_Buy_EV"
ID_COL = "id"
ORIG_PATH  = '/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv'

df = pl.read_csv('/kaggle/input/competitions/playground-series-s6e9/train.csv')
orig = pl.read_csv(ORIG_PATH).rename({'Buyer_ID':'id'})
df_test = pl.read_csv('/kaggle/input/competitions/playground-series-s6e9/test.csv')


for c in ['Home_Charging_Possible',	'Subsidy_Available', 'Will_Buy_EV' ]:
    df=df.with_columns(pl.when(pl.col(c)=='Yes').then(1).otherwise(0).alias(c))
    orig=orig.with_columns(pl.when(pl.col(c)=='Yes').then(1).otherwise(0).alias(c))
    try:
        df_test=df_test.with_columns(pl.when(pl.col(c)=='Yes').then(1).otherwise(0).alias(c))
    except:
        continue

# Map Original Dataset Target Means

orig_global_mean = orig[TARGET].mean()
real_world_stats = orig.group_by('Annual_Income_USD').agg(pl.mean(TARGET))
df=df.join(real_world_stats, how='left', on = 'Annual_Income_USD').rename({f'{TARGET}_right': f"Annual_Income_USD_org_mean"})
df_test=df_test.join(real_world_stats, how='left', on = 'Annual_Income_USD').rename({f'{TARGET}': f"Annual_Income_USD_org_mean"})
df=df.with_columns(AIUSD_notinorig=pl.col("Annual_Income_USD_org_mean").is_null())             #keep missing
df_test=df_test.with_columns(AIUSD_notinorig=pl.col("Annual_Income_USD_org_mean").is_null())

df=df.with_columns(pl.col(f"Annual_Income_USD_org_mean").fill_null(orig_global_mean))  
df_test=df_test.with_columns(pl.col(f"Annual_Income_USD_org_mean").fill_null(orig_global_mean))

        

df=          df.with_columns(homecharge=pl.col('Charging_Stations_Near_Home')+5* pl.col('Home_Charging_Possible'))
df_test=df_test.with_columns(homecharge=pl.col('Charging_Stations_Near_Home')+5* pl.col('Home_Charging_Possible'))


mapping = {'Low': 0, 'High': 3, 'Medium': 1}  

df = df.with_columns(pl.col("Range_Anxiety_Level").replace(mapping).cast(pl.UInt8))
df_test = df_test.with_columns(pl.col("Range_Anxiety_Level").replace(mapping).cast(pl.UInt8))

mapping = {v: i for i, v in enumerate(df["Gender"].unique().to_list())}
df = df.with_columns(pl.col("Gender").replace(mapping).cast(pl.UInt8))
df_test = df_test.with_columns(pl.col("Gender").replace(mapping).cast(pl.UInt8))


# The Mode Collapse Spike
df=df.with_columns( is_30k_spike = (pl.col('Annual_Income_USD') == 30000.0).cast(pl.Int8))
df=df.with_columns( is_millionaire_cliff = (pl.col('Annual_Income_USD') >= 170537.0).cast(pl.Int8))
df=df.with_columns( is_dead_zone = ((pl.col('Annual_Income_USD') >= 38000.0) & (pl.col('Annual_Income_USD') <= 42000.0) ).cast(pl.Int8))
#df=df.with_columns( is_env_hater = (pl.col('Environmental_Concern_Level') == 1).cast(pl.Int8))

df_test=df_test.with_columns( is_30k_spike = (pl.col('Annual_Income_USD') == 30000.0).cast(pl.Int8))
df_test=df_test.with_columns( is_millionaire_cliff = (pl.col('Annual_Income_USD') >= 170537.0).cast(pl.Int8))
df_test=df_test.with_columns( is_dead_zone = ((pl.col('Annual_Income_USD') >= 38000.0) & (pl.col('Annual_Income_USD') <= 42000.0) ).cast(pl.Int8))
#df_test=df_test.with_columns( is_env_hater = (pl.col('Environmental_Concern_Level') == 1).cast(pl.Int8))



df_test=df_test.with_columns(fable=1.2*pl.col('Annual_Income_USD' )/100000 + 0.6*pl.col('Environmental_Concern_Level')+2*pl.col('Subsidy_Available')-pl.col('Range_Anxiety_Level')-5.5)
df=df.with_columns(fable=1.2*pl.col('Annual_Income_USD' )/100000 + 0.6*pl.col('Environmental_Concern_Level')+2*pl.col('Subsidy_Available')-pl.col('Range_Anxiety_Level')-5.5)


df=df.with_columns(df.to_dummies('Environmental_Concern_Level'))
df_test=df_test.with_columns(df_test.to_dummies('Environmental_Concern_Level') )

df=df.with_columns(df.to_dummies('Range_Anxiety_Level'))
df_test=df_test.with_columns(df_test.to_dummies('Range_Anxiety_Level') )


# Digits
digit_num_cols = ['Annual_Income_USD', 'Daily_Commute_km']

for col in digit_num_cols:
    for k in range(-1, 4):
        new_col = f'{col}_digit{k}'
        df=df.with_columns((pl.col(col) // (10**k) % 10).cast(pl.Int8).alias(new_col))
        df_test=df_test.with_columns((pl.col(col) // (10**k) % 10).cast(pl.Int8).alias(new_col))


df=df.with_columns(fe_s_i=pl.col('Annual_Income_USD')*pl.col('Subsidy_Available'))
df_test=df_test.with_columns(fe_s_i=pl.col('Annual_Income_USD')*pl.col('Subsidy_Available'))

df=df.drop('Annual_Income_USD_digit-1', 'Daily_Commute_km_digit2', 'Daily_Commute_km_digit3')
df_test=df_test.drop('Annual_Income_USD_digit-1', 'Daily_Commute_km_digit2', 'Daily_Commute_km_digit3')

df=df.with_columns(subsidy_and_env4_or_5= pl.col('Subsidy_Available') * (pl.col('Environmental_Concern_Level_4.0') + pl.col('Environmental_Concern_Level_5.0'))  )
df=df.with_columns(subsidy_and_env_ordinal = pl.col('Subsidy_Available') * pl.col('Environmental_Concern_Level') )
df_test=df_test.with_columns(subsidy_and_env4_or_5= pl.col('Subsidy_Available') * (pl.col('Environmental_Concern_Level_4.0') + pl.col('Environmental_Concern_Level_5.0'))  )
df_test=df_test.with_columns(subsidy_and_env_ordinal = pl.col('Subsidy_Available') * pl.col('Environmental_Concern_Level') )

df=df.with_columns(worry_scorry = pl.col('Daily_Commute_km') - 5*pl.col('Charging_Stations_Near_Home') - 5 *pl.col('Charging_Stations_Near_Work') - 150 * pl.col('Home_Charging_Possible') )
df_test=df_test.with_columns(worry_scorry = pl.col('Daily_Commute_km') - 5*pl.col('Charging_Stations_Near_Home') - 5 *pl.col('Charging_Stations_Near_Work') 
                             - 150 * pl.col('Home_Charging_Possible'))

# %% ==================== cell 6 [code] ====================

tecols_1=[ 'Age', 'Gender', 'Number_of_Cars_Owned',  'City_Type', 'Current_Car_Type',  'Range_Anxiety_Level', 'Annual_Income_USD',
         'Annual_Income_USD_digit2', 'Annual_Income_USD_digit3',
       'Daily_Commute_km_digit-1',
       'Daily_Commute_km_digit0', 'Daily_Commute_km_digit1','Annual_Income_USD_digit0', 'Annual_Income_USD_digit1', 'fe_s_i', 'Daily_Commute_km'
         ] 

tecols_2=['Annual_Income_USD', 'Daily_Commute_km']

dropcols=['Age', 'Daily_Commute_km_digit0', 'is_dead_zone',  'Annual_Income_USD_digit3',  'Number_of_Cars_Owned',
 'Annual_Income_USD_digit1',  'Annual_Income_USD_digit2', 'Current_Car_Type',
 'Daily_Commute_km',  'City_Type',  'Charging_Stations_Near_Home',  'Range_Anxiety_Level',  'Annual_Income_USD_digit0',   'Daily_Commute_km_digit1',  'Charging_Stations_Near_Work',
 'Environmental_Concern_Level',  'Gender',   'Daily_Commute_km_digit-1',  'fe_s_i']

features = [col for col in df.columns if col not in ['id', TARGET]] 

X_train_te= df.to_pandas()[features]
y_train =    df.to_pandas()[TARGET]
X_test_te =  df_test.to_pandas()[features]

# --- Target Encoding  ---

# Dual Sklearn Target Encoders (Strict and Auto)
te_auto = TargetEncoder(shuffle=True, cv=5, smooth='auto', random_state=42) #
te_10   = TargetEncoder(shuffle=True, cv=5, smooth=2.0, random_state=42)
te_100   = TargetEncoder(shuffle=True, cv=5, smooth=1200.0, random_state=42)

X_train_enc_auto = te_auto.fit_transform(X_train_te[tecols_1], y_train)
X_test_enc_auto  = te_auto.transform(X_test_te[tecols_1])

X_train_enc_10 = te_10.fit_transform(X_train_te[tecols_1], y_train)
X_test_enc_10  = te_10.transform(X_test_te[tecols_1])

X_train_enc_100 = te_100.fit_transform(X_train_te[tecols_1], y_train)
X_test_enc_100  = te_100.transform(X_test_te[tecols_1])

for i, col in enumerate(tecols_1):
    # Auto smoothing TE
    te_col = f'te_{col}'
    X_train_te[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
    X_test_te[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
    
    # Strict (10.0) smoothing TE
    X_train_te[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
    X_test_te[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')

    X_train_te[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
    X_test_te[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')

# --- Target Encoding with gauss kernel
    
ker_l=[0.02, 1.0]
for k, col in enumerate(tecols_2):
    te_col = f'te_{col}'
    encoder = SmoothedTargetEncoder(kernel_param=ker_l[k], cv=10, shuffle=True, random_state=42)
    X_train_te[te_col] = encoder.fit_transform(X_train_te[col], y_train)
    X_test_te[te_col]  = encoder.transform(X_test_te[col])


top6 = [
    'Subsidy_Available',                    
    'Annual_Income_USD',                    
    'Environmental_Concern_Level_5.0', 'Environmental_Concern_Level_1.0',   
    'Range_Anxiety_Level',       # 3 — ordinal (Low=0, Med=1, High=2)
    'Home_Charging_Possible',               
    'te_Annual_Income_USD',               
            ]   

for i in range(len(top6)):
    for j in range(i+1, len(top6)):
        X_train_te[f'prod_{i}_{j}'] = X_train_te[top6[i]] * X_train_te[top6[j]]
        X_test_te[f'prod_{i}_{j}'] = X_test_te[top6[i]] * X_test_te[top6[j]]

X_train_te=X_train_te.drop(dropcols, axis=1)
X_test_te=X_test_te.drop(dropcols, axis=1)

ttec=X_train_te.columns
scaler = StandardScaler()
X_train_te[ttec] = scaler.fit_transform(X_train_te[ttec])
X_test_te[ttec]  = scaler.transform(X_test_te[ttec])

clf = LogisticRegression(
    penalty='l2',
    C=2,          # 
    solver='newton-cholesky',#'lbfgs',
    max_iter=1000
)
clf.fit(X_train_te[ttec], y_train)


test_probas = clf.predict_proba(X_test_te[ttec])[:, 1]

# %% ==================== cell 7 [code] ====================
pl.read_csv('/kaggle/input/competitions/playground-series-s6e9/sample_submission.csv').with_columns(Will_Buy_EV=test_probas).write_csv('sub_lr.csv')