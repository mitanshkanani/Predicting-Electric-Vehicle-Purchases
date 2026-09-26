

# %% ==================== cell 0 [markdown] ====================
# Electric vehicle buyers with CTBoost

Train one **CTBoost 0.1.60** model to predict who will buy an electric vehicle.
Start with the settings below, then change one or two values at a time.

| Submitted public AUC | Recorded five-fold CV AUC |
| :---: | :---: |
| **0.94615** | **0.945958** |

These scores belong to the original recipe (submission 56020502). Edits need new validation.
Choose **GPU** and turn **Internet on**, then use **Run All**. Every function is defined in this notebook.


# %% ==================== cell 1 [markdown] ====================
## 1. Choose the settings and install packages
`RUN_CV=True` checks the model on five held-out groups before the final fit.
Leave it off for a quick full-data training run. Change `ITERATIONS` alongside the learning rate.


# %% ==================== cell 2 [code] ====================
%pip install ctboost
from pathlib import Path
import pandas as pd
import gc
import joblib
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import ctboost
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from IPython.display import display


# %% ==================== cell 3 [code] ====================
RUN_CV = False
SEED = 20260904
ITERATIONS = 1426

PARAMS = {
    'learning_rate': 0.039,  # Smaller steps usually need more trees.
    'max_depth': 4,  # Tree depth.
    'max_leaves': 16,  # Maximum leaves per tree.
    'grow_policy': 'LeafWise',
    'alpha': 0.5,  # CTBoost feature-test threshold.
    'lambda_l2': 8.0,  # Leaf regularization.
    'min_data_in_leaf': 100,
    'min_child_weight': 0.1,
    'subsample': 0.85,
    'bootstrap_type': 'Bernoulli',
    'colsample_bytree': 0.3,  # Fraction of features offered to each tree.
    'feature_test': 'quadratic',
    'feature_test_adjustment': 'none',
    'feature_test_bins': 8,
    'max_bins': 1024,
    'leaf_estimation_iterations': 3,
}


# %% ==================== cell 4 [markdown] ====================
## 3. Meet the data
The competition has one row per person. `Will_Buy_EV` is the answer: **Yes = 1**, **No = 0**.
The original 10,000-person dataset supplies extra group statistics; its rows are not added to training.


# %% ==================== cell 5 [code] ====================
input_dir = Path('/kaggle/input')
competition_dirs = [input_dir / 'playground-series-s6e9',
                    input_dir / 'competitions/playground-series-s6e9']
competition_dirs += list(input_dir.glob('**/playground-series-s6e9'))
data_dir = next((p for p in competition_dirs if (p / 'train.csv').is_file()), None)

train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')
original = pd.read_csv(input_dir / 'datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv')
original['Will_Buy_EV'] = original['Will_Buy_EV'].map({'No': 0, 'Yes': 1})

X = train.drop(columns='Will_Buy_EV')
labels = train['Will_Buy_EV'].map({'No': 0, 'Yes': 1})
y = labels.to_numpy(np.int32)
print(f'Train: {len(train):,} | Test: {len(test):,} | Buyers: {y.mean():.1%}')
display(train.head(3))

# %% ==================== cell 6 [markdown] ====================
## 4. Give the trees useful clues
The recipe produces **130 numeric features**. IDs are excluded.

| Feature group | What it tells the model |
| --- | --- |
| Raw numbers, digits and flags | Values, rounding patterns and a few income cutoffs |
| Original-data means | Buyer rates for matching values in the original dataset |
| Frequencies | How common a value is in the training rows |
| Target encodings | Smoothed buyer rates, with internal cross-fitting for training |

**First: clean values and build the numeric clues.** The digit arithmetic is kept exactly as in the scored recipe.


# %% ==================== cell 7 [code] ====================
def numeric_values(values):
    
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def category_keys(values):
    # A prefix keeps a missing value distinct from an ordinary category name.
    
    return ("V:" + values.astype("string")).fillna("M:").astype(object)


def base_features(data, state):
    missing = set(state["input_columns"]).difference(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    columns = {}
    for name in state["input_columns"]:
        if name in state["category_maps"]:
            columns[name] = category_keys(data[name]).map(state["category_maps"][name]).fillna(-1)
        else:
            columns[name] = numeric_values(data[name])
    for name in state["input_columns"]:
        if name in state["category_maps"]:
            continue
        values = numeric_values(data[name]).fillna(0).to_numpy(dtype=np.float64)
        for digit in range(-4, 4):
            # Keep floor division: changing this arithmetic changes the recipe.
            columns[f"{name}_digit{digit}"] = values // (10.0 ** digit) % 10
    income = pd.to_numeric(data["Annual_Income_USD"], errors="coerce")
    concern = pd.to_numeric(data["Environmental_Concern_Level"], errors="coerce")
    columns["is_30k_spike"] = income == 30000.0
    columns["is_millionaire_cliff"] = income >= 170537.0
    columns["is_dead_zone"] = income.between(38000.0, 42000.0)
    columns["is_env_hater"] = concern == 1
    
    return pd.DataFrame(columns, index=data.index, dtype=np.float32)


# %% ==================== cell 8 [markdown] ====================
**Next: keep useful columns and make the four extra keys.** Income is viewed at exact, $100 and $1,000 levels; commute distance is rounded down. Rare groups receive stronger smoothing.


# %% ==================== cell 9 [code] ====================
def unique_columns(frame):
    """Remove constants and exact duplicates, using training rows only."""
    keep, seen = [], set()
    for name in frame:
        if frame[name].nunique(dropna=False) <= 1:
            continue
        values = frame[name].to_numpy(dtype=np.float32, copy=True)
        values[values == 0] = 0
        values[np.isnan(values)] = np.nan
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
        if digest not in seen:
            keep.append(name)
            seen.add(digest)
            
    return keep


def smooth_keys(data):
    income = numeric_values(data["Annual_Income_USD"])
    commute = numeric_values(data["Daily_Commute_km"])
    values = {
        "income_exact_integer": income,
        "income100_floor": np.floor(income / 100),
        "income1000_floor": np.floor(income / 1000),
        "commute_integer": np.floor(commute),
    }
    
    return pd.DataFrame({
        name: category_keys(numeric_values(value).round().astype("Int64"))
        for name, value in values.items()
    }, index=data.index)


def fit_maps(data, base, original, state):
    state["original_mean"] = float(original["Will_Buy_EV"].mean())
    state["original_maps"] = {
        name: original.groupby(name, observed=True, dropna=False)["Will_Buy_EV"].mean().to_dict()
        for name in state["input_columns"] if name in original
    }
    state["core_columns"] = unique_columns(base)
    state["encoding_columns"] = [name for name in state["core_columns"] if not name.startswith("is_")]
    state["frequency_maps"] = {
        name: base[name].value_counts(normalize=True, dropna=False).to_dict()
        for name in state["encoding_columns"]
    }


# %% ==================== cell 10 [markdown] ====================
**Now: turn the saved statistics into columns.** Training calls `fit_transform` on each encoder. Validation and test rows only call `transform`.


# %% ==================== cell 11 [code] ====================
def encoded_features(data, base, state, y=None):
    columns = {
        name: base[name].to_numpy(dtype=np.float32, copy=False)
        for name in state["core_columns"] if name not in state["category_maps"]
    }
    for name, mapping in state["original_maps"].items():
        columns[f"{name}_org_mean"] = data[name].map(mapping).fillna(state["original_mean"])
    for name in state["encoding_columns"]:
        columns[f"{name}_fe"] = base[name].map(state["frequency_maps"][name]).fillna(0)
    codes = base[state["encoding_columns"]]
    for label in ("auto", "10"):
        encoder = state["encoders"][label]
        values = encoder.fit_transform(codes, y) if y is not None else encoder.transform(codes)
        for index, name in enumerate(codes.columns):
            columns[f"{name}_TE_{label}"] = values[:, index]
    keys = smooth_keys(data)
    encoder = state["encoders"]["100"]
    values = encoder.fit_transform(keys, y) if y is not None else encoder.transform(keys)
    for index, name in enumerate(keys.columns):
        columns[f"enc_smooth_{name}_TE_100"] = values[:, index]
        
    return pd.DataFrame(columns, index=data.index, dtype=np.float32)


def transform_features(data, state):
    """Apply training mappings to validation or test rows without fitting."""
    
    features = encoded_features(data, base_features(data, state), state)
    
    return features[state["feature_names"]]


# %% ==================== cell 12 [markdown] ====================
**Finally: fit the whole feature recipe on one training group.** The returned `state` stores the maps and encoders. It is a plain dictionary, so there is no custom module to import.


# %% ==================== cell 13 [code] ====================
def fit_features(data, y, original_frame, seed=42):
    """Return cross-fitted training features and a portable dictionary of state."""
    folds = min(5, int(np.bincount(y, minlength=2).min()))
    excluded = {"id", "Will_Buy_EV", "is_train", "Number_of_Cars_Owned"}
    names = [name for name in data if name not in excluded]
    categories = [name for name in names if not pd.api.types.is_numeric_dtype(data[name].dtype)]
    state = {"input_columns": names, "category_maps": {}, "encoders": {}, "seed": seed}
    
    for name in categories:
        values = sorted(category_keys(data[name]).unique())
        state["category_maps"][name] = {value: index for index, value in enumerate(values)}
    base = base_features(data, state)
    fit_maps(data, base, original_frame, state)
    
    for label, smooth in (("auto", "auto"), ("10", 10.0), ("100", 100.0)):
        state["encoders"][label] = TargetEncoder(
            target_type="binary", smooth=smooth, cv=5 if label == "100" else folds,
            shuffle=True, random_state=seed,
        )
    features = encoded_features(data, base, state, y)
    state["feature_names"] = list(features.columns)
    
    return features, state

# %% ==================== cell 14 [markdown] ====================
## 5. Check a change with cross-validation
Each fold gets fresh feature mappings and a fresh model. We keep the same five groups (seed 42)
and use a fixed tree count. This is development CV: the recipe was selected using earlier tuning.
The final submission will come from one model fitted on all rows.


# %% ==================== cell 15 [code] ====================
def new_model():
    return ctboost.CTBoostClassifier(
        iterations=ITERATIONS, task_type='GPU', devices='0',
        random_seed=SEED, eval_metric='AUC', verbose=False, **PARAMS)

cv_auc = None
fold_scores = []
if RUN_CV:
    oof = np.zeros(len(y), dtype=np.float32)
    fold_ids = np.zeros(len(y), dtype=np.int8)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(X, y)):
        fit_data, state = fit_features(X.iloc[fit_idx], y[fit_idx], original, seed=SEED)
        valid_data = transform_features(X.iloc[valid_idx], state)
        model = new_model().fit(fit_data, y[fit_idx])
        oof[valid_idx] = model.predict_proba(valid_data)[:, 1]
        fold_ids[valid_idx] = fold
        score = roc_auc_score(y[valid_idx], oof[valid_idx])
        fold_scores.append(score)
        print(f'Fold {fold + 1}: {score:.8f}', flush=True)
        del fit_data, valid_data, state, model
        gc.collect()
    cv_auc = float(roc_auc_score(y, oof))
    np.savez_compressed('oof.npz', id=train['id'], target=y, prediction=oof, fold=fold_ids)
    display(pd.DataFrame({'Fold': range(1, 6), 'AUC': fold_scores}))
    print(f'Pooled OOF AUC: {cv_auc:.8f}')
else:
    print('CV skipped. Set RUN_CV=True when you want to compare settings.')

# %% ==================== cell 16 [markdown] ====================
## 6. Train the final model
Now use all training rows. The test set receives the mappings learned from those rows.
This is the only model used for the submission.


# %% ==================== cell 17 [code] ====================
started = time.monotonic()
fit_data, feature_state = fit_features(X, y, original, seed=SEED)
test_data = transform_features(test, feature_state)
print(f'FINAL_REFIT_START: {fit_data.shape[1]} features, {ITERATIONS} trees', flush=True)
model = new_model().fit(fit_data, y)
prediction = model.predict_proba(test_data)[:, 1]
print(f'Training and prediction took {(time.monotonic() - started) / 60:.1f} minutes.')


# %% ==================== cell 18 [markdown] ====================
## 7. What did the model use?
These are the 15 most important features in **this run**. Importance describes the fitted model;
it does not show that a feature causes someone to buy an EV.


# %% ==================== cell 19 [code] ====================
importance = pd.Series(model.feature_importances_, index=fit_data.columns)
top = importance.nlargest(15).sort_values()
ax = top.plot.barh(color='#8da0cb', figsize=(10, 5))
ax.set_title('CTBoost: the strongest features in this run', loc='left')
ax.set_xlabel('Feature importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=140, bbox_inches='tight')
plt.show()

# %% ==================== cell 20 [markdown] ====================
## 8. Save the submission
Check the row order and probabilities, save the files, and reload them once.
Download **submission.csv** from Output when you are ready to submit it.


# %% ==================== cell 21 [code] ====================
assert len(prediction) == len(test) == len(sample)
assert np.array_equal(test['id'], sample['id'])
assert sample['id'].is_unique
assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
submission = sample.copy()
submission['Will_Buy_EV'] = prediction
submission.to_csv('submission.csv', index=False)
model.save_model('ctboost_single.json')
joblib.dump(feature_state, 'feature_state.joblib', compress=3)

loaded_state = joblib.load('feature_state.joblib')
loaded_model = ctboost.CTBoostClassifier.load_model('ctboost_single.json')
replay = loaded_model.predict_proba(transform_features(test.head(1000), loaded_state))[:, 1]
np.testing.assert_allclose(replay, prediction[:1000], rtol=1e-7, atol=1e-8)
run_info = {'ctboost': ctboost.__version__, 'task_type': 'GPU', 'iterations': ITERATIONS,
            'params': PARAMS, 'cv_auc_this_run': cv_auc, 'features': fit_data.shape[1],
            'reference_public_auc': 0.94615, 'test_rows': len(test),
            'reload_checked_rows': len(replay), 'submitted': False,
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
            'submission_sha256': hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()}
Path('run_summary.json').write_text(json.dumps(run_info, indent=2))
print('ALL_COMPLETE: submission.csv is ready; reload check passed.', flush=True)
display(submission.head())

# %% ==================== cell 22 [markdown] ====================
**Try next:** lower the learning rate and add trees, or adjust depth, L2 and feature sampling.
Keep the same CV folds so comparisons are useful. A rerun needs its own public score.

Layout inspired by [Chris Deotte's XGB starter](https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter).
Feature ideas: [Naji's preprocessing](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94587-lb-0-94612)
and [Lam Huy's encodings](https://www.kaggle.com/code/lamhuy8904/s6e9-94-6-transformer-and-gbdt-ensemble).
Workflow assisted by GPT Astra.
