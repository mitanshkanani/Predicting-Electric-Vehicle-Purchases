# ============================================================
# EXP-011: Optimized PyTorch Tabular MLP Ensemble
# ============================================================

import os
import gc
import json
import time
import random
import hashlib
import warnings
from itertools import combinations
from contextlib import nullcontext

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = (
    "/kaggle/input/datasets/mitanshkanani/"
    "dataset-for-experimentation-final"
)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

SPLIT_DIR = "/kaggle/working/catboost_experiment_results"

TRAIN_SPLIT_PATH = os.path.join(
    SPLIT_DIR,
    "train_split_ids.npy"
)

VALIDATION_SPLIT_PATH = os.path.join(
    SPLIT_DIR,
    "validation_split_ids.npy"
)

OUTPUT_DIR = "/kaggle/working/exp011"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Reproducibility
SEED = 42

# Cross-validation
N_OUTER_FOLDS = 5
INNER_VALIDATION_SIZE = 0.10

# Training
BATCH_SIZE = 1024
MAX_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 12
LR_SCHEDULER_PATIENCE = 5
MIN_DELTA = 5e-5

# Runtime
NUM_WORKERS = 0
USE_AMP = True

# Bootstrap
N_BOOTSTRAP = 30
BOOTSTRAP_RANDOM_STATE = 42

# Pairwise blending
BLEND_WEIGHT_GRID = np.arange(0.0, 1.01, 0.05)


# ============================================================
# 2. SEEDING AND DEVICE
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for EXP-010. "
        "Please enable a Kaggle GPU accelerator."
    )

DEVICE = torch.device("cuda")

print("=" * 80)
print("EXP-011: Optimized PyTorch Tabular MLP Ensemble")
print("=" * 80)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Output directory: {OUTPUT_DIR}")
print()
print(
    "NUM_WORKERS=0 means DataLoader uses the main process. "
    "This reduces RAM overhead and avoids multiprocessing issues "
    "with large tabular datasets on Kaggle."
)
print("=" * 80)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_PATH)

print(f"Train shape: {train_df.shape}")

TARGET_COLUMN = "Will_Buy_EV"
ID_COLUMN = "id"

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found."
    )

# Convert target to binary
y = (
    train_df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    })
)

if y.isna().any():
    raise ValueError(
        "Target conversion produced missing values. "
        "Please inspect target labels."
    )

y = y.astype(np.int64).to_numpy()

# Remove target and ID
feature_columns = [
    column
    for column in train_df.columns
    if column not in [TARGET_COLUMN, ID_COLUMN]
]

X_df = train_df[feature_columns].copy()

del train_df
gc.collect()

print(f"Feature shape: {X_df.shape}")
print(f"Number of positive samples: {y.sum():,}")
print(f"Positive rate: {y.mean():.6f}")


# ============================================================
# 4. FEATURE DEFINITIONS
# ============================================================

NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level"
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

missing_numeric = [
    column
    for column in NUMERIC_COLUMNS
    if column not in X_df.columns
]

missing_categorical = [
    column
    for column in CATEGORICAL_COLUMNS
    if column not in X_df.columns
]

if missing_numeric or missing_categorical:
    raise ValueError(
        f"Missing numeric columns: {missing_numeric}\n"
        f"Missing categorical columns: {missing_categorical}"
    )

print("\nNumeric columns:")
print(NUMERIC_COLUMNS)

print("\nCategorical columns:")
print(CATEGORICAL_COLUMNS)


# ============================================================
# 5. LOAD AND VERIFY FIXED SPLITS
# ============================================================

print("\nLoading fixed train and validation split indices...")

development_indices = np.load(TRAIN_SPLIT_PATH)
fixed_validation_indices = np.load(VALIDATION_SPLIT_PATH)

development_indices = np.asarray(
    development_indices,
    dtype=np.int64
)

fixed_validation_indices = np.asarray(
    fixed_validation_indices,
    dtype=np.int64
)

development_indices = np.sort(development_indices)
fixed_validation_indices = np.sort(fixed_validation_indices)

all_indices = np.arange(len(X_df))

if len(np.intersect1d(
    development_indices,
    fixed_validation_indices
)) != 0:
    raise ValueError(
        "Development and fixed-validation indices overlap."
    )

if len(np.union1d(
    development_indices,
    fixed_validation_indices
)) != len(X_df):
    raise ValueError(
        "Development and fixed-validation indices do not "
        "cover the complete dataset."
    )

development_y = y[development_indices]
validation_y = y[fixed_validation_indices]

print(f"Development rows: {len(development_indices):,}")
print(f"Fixed validation rows: {len(fixed_validation_indices):,}")
print(f"Development positive rate: {development_y.mean():.6f}")
print(f"Validation positive rate: {validation_y.mean():.6f}")


def file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


print("\nSplit file hashes:")
print(
    "Train split:",
    file_hash(TRAIN_SPLIT_PATH)
)

print(
    "Validation split:",
    file_hash(VALIDATION_SPLIT_PATH)
)


# ============================================================
# 6. MODEL ARCHITECTURES
# ============================================================

ARCHITECTURES = {
    "MLP_BASE": {
        "hidden_layers": [256, 128, 64],
        "dropouts": [0.20, 0.15, 0.00],
        "use_pos_weight": True
    },
    "MLP_DEEP": {
        "hidden_layers": [512, 256, 128, 64],
        "dropouts": [0.25, 0.20, 0.15, 0.00],
        "use_pos_weight": True
    },
    "MLP_WIDE": {
        "hidden_layers": [512, 512, 256],
        "dropouts": [0.15, 0.15, 0.00],
        "use_pos_weight": False
    },
    "MLP_REGULARIZED": {
        "hidden_layers": [256, 128, 64],
        "dropouts": [0.35, 0.30, 0.20],
        "use_pos_weight": False
    }
}


class TabularMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_layers,
        dropouts
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim, dropout_rate in zip(
            hidden_layers,
            dropouts
        ):
            layers.append(
                nn.Linear(previous_dim, hidden_dim)
            )

            layers.append(
                nn.BatchNorm1d(hidden_dim)
            )

            layers.append(
                nn.ReLU()
            )

            if dropout_rate > 0:
                layers.append(
                    nn.Dropout(dropout_rate)
                )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, 1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# 7. PREPROCESSING HELPERS
# ============================================================

def create_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def fit_preprocessor(X_train):
    scaler = StandardScaler()

    encoder = create_one_hot_encoder()

    numeric_train = X_train[NUMERIC_COLUMNS].copy()
    categorical_train = X_train[CATEGORICAL_COLUMNS].copy()

    numeric_train = numeric_train.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0.0)

    categorical_train = categorical_train.fillna(
        "__MISSING__"
    ).astype(str)

    scaler.fit(numeric_train)
    encoder.fit(categorical_train)

    return scaler, encoder


def transform_features(
    X_data,
    scaler,
    encoder
):
    numeric_data = X_data[NUMERIC_COLUMNS].copy()
    categorical_data = X_data[CATEGORICAL_COLUMNS].copy()

    numeric_data = numeric_data.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0.0)

    categorical_data = categorical_data.fillna(
        "__MISSING__"
    ).astype(str)

    numeric_array = scaler.transform(
        numeric_data
    ).astype(np.float32)

    categorical_array = encoder.transform(
        categorical_data
    ).astype(np.float32)

    combined_array = np.hstack([
        numeric_array,
        categorical_array
    ])

    return combined_array.astype(np.float32)


# ============================================================
# 8. METRIC HELPERS
# ============================================================

def calculate_metrics(y_true, predictions):
    predictions = np.clip(
        np.asarray(predictions),
        1e-7,
        1 - 1e-7
    )

    return {
        "auc": float(
            roc_auc_score(y_true, predictions)
        ),
        "logloss": float(
            log_loss(y_true, predictions)
        ),
        "brier": float(
            brier_score_loss(y_true, predictions)
        )
    }


def safe_auc(y_true, predictions):
    try:
        return float(
            roc_auc_score(y_true, predictions)
        )
    except Exception:
        return np.nan


# ============================================================
# 9. AMP HELPER
# ============================================================

def autocast_context():
    if not USE_AMP:
        return nullcontext()

    torch_version_major = int(
        torch.__version__.split(".")[0]
    )

    if torch_version_major >= 2:
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True
        )

    return torch.cuda.amp.autocast(
        enabled=True
    )


# ============================================================
# 10. PREDICTION FUNCTION
# ============================================================

def predict_model(
    model,
    features,
    batch_size=BATCH_SIZE
):
    model.eval()

    tensor_features = torch.tensor(
        features,
        dtype=torch.float32
    )

    dataset = TensorDataset(tensor_features)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(
                DEVICE,
                non_blocking=True
            )

            with autocast_context():
                logits = model(batch_x)

            probabilities = torch.sigmoid(
                logits
            ).detach().float().cpu().numpy()

            predictions.append(probabilities)

    if not predictions:
        return np.array([], dtype=np.float32)

    return np.concatenate(predictions).astype(
        np.float32
    )


# ============================================================
# 11. TRAIN ONE OUTER FOLD
# ============================================================

def train_single_fold(
    architecture_name,
    architecture_config,
    fold_number,
    outer_train_positions,
    outer_holdout_positions,
    development_indices,
    fixed_validation_indices,
    X_df,
    y
):
    print("\n" + "-" * 80)
    print(
        f"{architecture_name} | "
        f"Outer fold {fold_number}"
    )
    print("-" * 80)

    outer_train_global_indices = (
        development_indices[outer_train_positions]
    )

    outer_holdout_global_indices = (
        development_indices[outer_holdout_positions]
    )

    outer_train_y = y[
        outer_train_global_indices
    ]

    inner_train_positions, inner_valid_positions = (
        train_test_split(
            np.arange(len(outer_train_global_indices)),
            test_size=INNER_VALIDATION_SIZE,
            random_state=SEED + fold_number,
            stratify=outer_train_y
        )
    )

    inner_train_global_indices = (
        outer_train_global_indices[inner_train_positions]
    )

    inner_valid_global_indices = (
        outer_train_global_indices[inner_valid_positions]
    )

    X_inner_train = X_df.iloc[
        inner_train_global_indices
    ]

    X_inner_valid = X_df.iloc[
        inner_valid_global_indices
    ]

    X_outer_holdout = X_df.iloc[
        outer_holdout_global_indices
    ]

    X_fixed_validation = X_df.iloc[
        fixed_validation_indices
    ]

    y_inner_train = y[
        inner_train_global_indices
    ]

    y_inner_valid = y[
        inner_valid_global_indices
    ]

    # Fit preprocessing only on inner training data
    fold_scaler, fold_encoder = fit_preprocessor(
        X_inner_train
    )

    X_inner_train_processed = transform_features(
        X_inner_train,
        fold_scaler,
        fold_encoder
    )

    X_inner_valid_processed = transform_features(
        X_inner_valid,
        fold_scaler,
        fold_encoder
    )

    X_outer_holdout_processed = transform_features(
        X_outer_holdout,
        fold_scaler,
        fold_encoder
    )

    X_fixed_validation_processed = transform_features(
        X_fixed_validation,
        fold_scaler,
        fold_encoder
    )

    input_dim = X_inner_train_processed.shape[1]

    model = TabularMLP(
        input_dim=input_dim,
        hidden_layers=architecture_config["hidden_layers"],
        dropouts=architecture_config["dropouts"]
    ).to(DEVICE)

    positive_count = np.sum(y_inner_train == 1)
    negative_count = np.sum(y_inner_train == 0)

    positive_weight = (
        negative_count / max(positive_count, 1)
    )

    if architecture_config.get("use_pos_weight", True):
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=DEVICE
            )
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=3
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(MAX_EPOCHS - 3, 1),
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[3]
    )

    train_features_tensor = torch.tensor(
        X_inner_train_processed,
        dtype=torch.float32
    )

    train_target_tensor = torch.tensor(
        y_inner_train,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        train_features_tensor,
        train_target_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    scaler_amp = None

    if USE_AMP:
        scaler_amp = torch.cuda.amp.GradScaler(
            enabled=True
        )

    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if USE_AMP:
                with autocast_context():
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}"
                    )

                scaler_amp.scale(loss).backward()

                scaler_amp.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM
                )

                scaler_amp.step(optimizer)
                scaler_amp.update()

            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}"
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM
                )

                optimizer.step()

            current_batch_size = len(batch_y)

            epoch_loss += (
                loss.item() * current_batch_size
            )

            sample_count += current_batch_size

        epoch_loss /= max(sample_count, 1)

        validation_predictions = predict_model(
            model,
            X_inner_valid_processed
        )

        validation_auc = safe_auc(
            y_inner_valid,
            validation_predictions
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Inner AUC: {validation_auc:.6f} | "
            f"LR: {current_lr:.7f}"
        )

        if validation_auc > best_auc + MIN_DELTA:

            best_auc = validation_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(best_state)
    model.eval()

    outer_holdout_predictions = predict_model(
        model,
        X_outer_holdout_processed
    )

    fixed_validation_predictions = predict_model(
        model,
        X_fixed_validation_processed
    )

    result = {
        "outer_holdout_predictions": outer_holdout_predictions,
        "fixed_validation_predictions": fixed_validation_predictions,
        "best_inner_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "outer_holdout_indices": outer_holdout_global_indices
    }

    del model
    del train_loader
    del train_dataset
    del train_features_tensor
    del train_target_tensor
    del X_inner_train_processed
    del X_inner_valid_processed
    del X_outer_holdout_processed
    del X_fixed_validation_processed
    del fold_scaler
    del fold_encoder

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 12. OUTER FOLD DEFINITIONS
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=SEED
)

outer_splits = list(
    outer_cv.split(
        np.zeros(len(development_indices)),
        development_y
    )
)


# ============================================================
# 13. TRAIN ALL MLP ARCHITECTURES
# ============================================================

all_oof_predictions = {}
all_validation_predictions = {}
training_metadata = {}
individual_results = []

for architecture_name, architecture_config in ARCHITECTURES.items():

    print("\n\n" + "=" * 80)
    print(f"TRAINING ARCHITECTURE: {architecture_name}")
    print("=" * 80)

    oof_predictions = np.full(
        len(development_indices),
        np.nan,
        dtype=np.float32
    )

    validation_predictions_sum = np.zeros(
        len(fixed_validation_indices),
        dtype=np.float64
    )

    successful_folds = 0
    fold_metadata = []

    architecture_start_time = time.time()

    for fold_number, (
        outer_train_positions,
        outer_holdout_positions
    ) in enumerate(outer_splits, start=1):

        try:
            fold_result = train_single_fold(
                architecture_name=architecture_name,
                architecture_config=architecture_config,
                fold_number=fold_number,
                outer_train_positions=outer_train_positions,
                outer_holdout_positions=outer_holdout_positions,
                development_indices=development_indices,
                fixed_validation_indices=fixed_validation_indices,
                X_df=X_df,
                y=y
            )

            oof_predictions[
                outer_holdout_positions
            ] = fold_result[
                "outer_holdout_predictions"
            ]

            validation_predictions_sum += (
                fold_result["fixed_validation_predictions"]
            )

            successful_folds += 1

            fold_metadata.append({
                "fold": fold_number,
                "best_inner_auc": fold_result["best_inner_auc"],
                "best_epoch": fold_result["best_epoch"]
            })

        except Exception as error:
            print(
                f"Fold {fold_number} failed for "
                f"{architecture_name}: {repr(error)}"
            )

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if successful_folds > 0:
        validation_predictions = (
            validation_predictions_sum / successful_folds
        )
    else:
        validation_predictions = np.full(
            len(fixed_validation_indices),
            np.nan,
            dtype=np.float32
        )

    oof_complete = (
        successful_folds == N_OUTER_FOLDS
        and np.isfinite(oof_predictions).all()
    )

    validation_complete = (
        successful_folds == N_OUTER_FOLDS
        and np.isfinite(validation_predictions).all()
    )

    eligible = (
        oof_complete
        and validation_complete
    )

    architecture_time = (
        time.time() - architecture_start_time
    )

    print("\nArchitecture summary:")
    print(f"Successful folds: {successful_folds}/{N_OUTER_FOLDS}")
    print(f"OOF complete: {oof_complete}")
    print(f"Validation complete: {validation_complete}")
    print(f"Eligible for ensemble: {eligible}")
    print(f"Training time: {architecture_time / 60:.2f} minutes")

    if eligible:

        oof_metrics = calculate_metrics(
            development_y,
            oof_predictions
        )

        validation_metrics = calculate_metrics(
            validation_y,
            validation_predictions
        )

        individual_results.append({
            "model": architecture_name,
            "type": "individual",
            "oof_auc": oof_metrics["auc"],
            "oof_logloss": oof_metrics["logloss"],
            "oof_brier": oof_metrics["brier"],
            "validation_auc": validation_metrics["auc"],
            "validation_logloss": validation_metrics["logloss"],
            "validation_brier": validation_metrics["brier"],
            "successful_folds": successful_folds,
            "eligible": True
        })

        all_oof_predictions[architecture_name] = (
            oof_predictions
        )

        all_validation_predictions[architecture_name] = (
            validation_predictions
        )

    training_metadata[architecture_name] = {
        "architecture": architecture_config,
        "successful_folds": successful_folds,
        "eligible": eligible,
        "training_time_minutes": architecture_time,
        "folds": fold_metadata
    }

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{architecture_name}_oof_predictions.npy"
        ),
        oof_predictions
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{architecture_name}_validation_predictions.npy"
        ),
        validation_predictions
    )


# ============================================================
# 14. INDIVIDUAL MODEL RESULTS
# ============================================================

if not individual_results:
    raise RuntimeError(
        "No MLP architecture completed all folds successfully."
    )

individual_results_df = pd.DataFrame(
    individual_results
).sort_values(
    "oof_auc",
    ascending=False
).reset_index(drop=True)

print("\n\n" + "=" * 80)
print("INDIVIDUAL MODEL RESULTS")
print("=" * 80)

print(
    individual_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

best_individual_model = (
    individual_results_df.iloc[0]["model"]
)

best_individual_oof_auc = (
    individual_results_df.iloc[0]["oof_auc"]
)

best_individual_validation_auc = (
    individual_results_df.iloc[0]["validation_auc"]
)

print(
    f"\nBest individual model by OOF AUC: "
    f"{best_individual_model}"
)


# ============================================================
# 15. COMBINE PREDICTIONS INTO MATRICES
# ============================================================

eligible_model_names = list(
    all_oof_predictions.keys()
)

if len(eligible_model_names) == 0:
    raise RuntimeError(
        "No eligible models are available for blending."
    )

oof_matrix = np.column_stack([
    all_oof_predictions[name]
    for name in eligible_model_names
])

validation_matrix = np.column_stack([
    all_validation_predictions[name]
    for name in eligible_model_names
])

np.save(
    os.path.join(
        OUTPUT_DIR,
        "all_mlp_oof_predictions.npy"
    ),
    oof_matrix
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "all_mlp_validation_predictions.npy"
    ),
    validation_matrix
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "mlp_model_names.json"
    ),
    "w"
) as file:
    json.dump(
        eligible_model_names,
        file,
        indent=2
    )

print("\nCombined prediction files saved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "all_mlp_oof_predictions.npy"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "all_mlp_validation_predictions.npy"
    )
)


# ============================================================
# 16. BLENDING HELPERS
# ============================================================

def equal_weight_average(prediction_matrix):
    return np.mean(
        prediction_matrix,
        axis=1
    )


def rank_average(prediction_matrix):
    rank_matrix = np.zeros_like(
        prediction_matrix,
        dtype=np.float64
    )

    for column_index in range(
        prediction_matrix.shape[1]
    ):
        order = np.argsort(
            prediction_matrix[:, column_index],
            kind="stable"
        )
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)

        rank_matrix[:, column_index] = (
            ranks / max(len(ranks) - 1, 1)
        )

    return np.mean(
        rank_matrix,
        axis=1
    )


def evaluate_prediction_set(
    name,
    prediction_type,
    y_true,
    predictions
):
    metrics = calculate_metrics(
        y_true,
        predictions
    )

    return {
        "model": name,
        "type": prediction_type,
        "oof_auc": metrics["auc"] if prediction_type == "OOF" else np.nan,
        "oof_logloss": metrics["logloss"] if prediction_type == "OOF" else np.nan,
        "oof_brier": metrics["brier"] if prediction_type == "OOF" else np.nan,
        "validation_auc": metrics["auc"] if prediction_type == "Validation" else np.nan,
        "validation_logloss": metrics["logloss"] if prediction_type == "Validation" else np.nan,
        "validation_brier": metrics["brier"] if prediction_type == "Validation" else np.nan
    }


ensemble_results = []

# Equal-weight ensemble
equal_oof_predictions = equal_weight_average(
    oof_matrix
)

equal_validation_predictions = equal_weight_average(
    validation_matrix
)

equal_oof_metrics = calculate_metrics(
    development_y,
    equal_oof_predictions
)

equal_validation_metrics = calculate_metrics(
    validation_y,
    equal_validation_predictions
)

ensemble_results.append({
    "model": "EQUAL_WEIGHT_ENSEMBLE",
    "type": "ensemble",
    "oof_auc": equal_oof_metrics["auc"],
    "oof_logloss": equal_oof_metrics["logloss"],
    "oof_brier": equal_oof_metrics["brier"],
    "validation_auc": equal_validation_metrics["auc"],
    "validation_logloss": equal_validation_metrics["logloss"],
    "validation_brier": equal_validation_metrics["brier"]
})

# Rank ensemble
rank_oof_predictions = rank_average(
    oof_matrix
)

rank_validation_predictions = rank_average(
    validation_matrix
)

rank_oof_metrics = calculate_metrics(
    development_y,
    rank_oof_predictions
)

rank_validation_metrics = calculate_metrics(
    validation_y,
    rank_validation_predictions
)

ensemble_results.append({
    "model": "RANK_AVERAGE_ENSEMBLE",
    "type": "ensemble",
    "oof_auc": rank_oof_metrics["auc"],
    "oof_logloss": np.nan,
    "oof_brier": np.nan,
    "validation_auc": rank_validation_metrics["auc"],
    "validation_logloss": np.nan,
    "validation_brier": np.nan
})


# ============================================================
# 17. BEST PAIRWISE BLEND
# ============================================================

best_pair = None
best_pair_weight = None
best_pair_oof_auc = -np.inf
best_pair_oof_predictions = None
best_pair_validation_predictions = None

if len(eligible_model_names) >= 2:

    for model_a, model_b in combinations(
        eligible_model_names,
        2
    ):

        predictions_a = all_oof_predictions[model_a]
        predictions_b = all_oof_predictions[model_b]

        for weight_a in BLEND_WEIGHT_GRID:

            weight_b = 1.0 - weight_a

            blended_oof = (
                weight_a * predictions_a
                + weight_b * predictions_b
            )

            blended_auc = safe_auc(
                development_y,
                blended_oof
            )

            if blended_auc > best_pair_oof_auc:

                best_pair_oof_auc = blended_auc

                best_pair = (
                    model_a,
                    model_b
                )

                best_pair_weight = (
                    float(weight_a),
                    float(weight_b)
                )

                best_pair_oof_predictions = (
                    blended_oof.copy()
                )

                best_pair_validation_predictions = (
                    weight_a * all_validation_predictions[model_a]
                    + weight_b * all_validation_predictions[model_b]
                )

if best_pair is not None:

    best_pair_validation_metrics = calculate_metrics(
        validation_y,
        best_pair_validation_predictions
    )

    best_pair_oof_metrics = calculate_metrics(
        development_y,
        best_pair_oof_predictions
    )

    ensemble_results.append({
        "model": (
            f"BEST_PAIR_{best_pair[0]}_"
            f"{best_pair[1]}"
        ),
        "type": "ensemble",
        "oof_auc": best_pair_oof_metrics["auc"],
        "oof_logloss": best_pair_oof_metrics["logloss"],
        "oof_brier": best_pair_oof_metrics["brier"],
        "validation_auc": best_pair_validation_metrics["auc"],
        "validation_logloss": best_pair_validation_metrics["logloss"],
        "validation_brier": best_pair_validation_metrics["brier"]
    })

    print("\nBest pairwise blend:")
    print(f"Model A: {best_pair[0]}")
    print(f"Model B: {best_pair[1]}")
    print(f"Weight A: {best_pair_weight[0]:.2f}")
    print(f"Weight B: {best_pair_weight[1]:.2f}")
    print(f"OOF AUC: {best_pair_oof_metrics['auc']:.6f}")
    print(
        "Validation AUC: "
        f"{best_pair_validation_metrics['auc']:.6f}"
    )

else:
    print(
        "\nPairwise blending skipped because fewer than "
        "two eligible models completed successfully."
    )


# ============================================================
# 18. ENSEMBLE RESULTS
# ============================================================

ensemble_results_df = pd.DataFrame(
    ensemble_results
)

all_results_df = pd.concat(
    [
        individual_results_df,
        ensemble_results_df
    ],
    ignore_index=True
)

all_results_df = all_results_df.sort_values(
    "oof_auc",
    ascending=False
).reset_index(drop=True)

print("\n\n" + "=" * 80)
print("COMPLETE MODEL AND ENSEMBLE RESULTS")
print("=" * 80)

print(
    all_results_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
            if pd.notna(value)
            else "NaN"
        )
    )
)


# ============================================================
# 19. BEST BLEND AND IMPROVEMENT CALCULATION
# ============================================================

ensemble_only_df = all_results_df[
    all_results_df["type"] == "ensemble"
].copy()

if len(ensemble_only_df) > 0:

    best_blend_row = ensemble_only_df.loc[
        ensemble_only_df["oof_auc"].idxmax()
    ]

    best_blend_name = best_blend_row["model"]
    best_blend_oof_auc = best_blend_row["oof_auc"]
    best_blend_validation_auc = (
        best_blend_row["validation_auc"]
    )

    improvement_oof = (
        best_blend_oof_auc
        - best_individual_oof_auc
    )

    improvement_validation = (
        best_blend_validation_auc
        - best_individual_validation_auc
    )

    relative_improvement_oof = (
        improvement_oof
        / max(abs(best_individual_oof_auc), 1e-8)
    ) * 100.0

    relative_improvement_validation = (
        improvement_validation
        / max(abs(best_individual_validation_auc), 1e-8)
    ) * 100.0

    print("\n\n" + "=" * 80)
    print("BEST INDIVIDUAL VS BEST BLEND")
    print("=" * 80)

    print(f"Best individual: {best_individual_model}")
    print(f"Best individual OOF AUC: {best_individual_oof_auc:.6f}")
    print(
        "Best individual validation AUC: "
        f"{best_individual_validation_auc:.6f}"
    )

    print(f"\nBest blend: {best_blend_name}")
    print(f"Best blend OOF AUC: {best_blend_oof_auc:.6f}")
    print(
        "Best blend validation AUC: "
        f"{best_blend_validation_auc:.6f}"
    )

    print("\nAbsolute improvement:")
    print(f"OOF AUC improvement: {improvement_oof:+.6f}")
    print(
        "Validation AUC improvement: "
        f"{improvement_validation:+.6f}"
    )

    print("\nRelative improvement:")
    print(
        f"OOF relative improvement: "
        f"{relative_improvement_oof:+.4f}%"
    )

    print(
        f"Validation relative improvement: "
        f"{relative_improvement_validation:+.4f}%"
    )

else:
    best_blend_name = None
    best_blend_oof_auc = np.nan
    best_blend_validation_auc = np.nan
    improvement_oof = np.nan
    improvement_validation = np.nan

    print("\nNo valid ensemble was available.")


# ============================================================
# 20. BOOTSTRAP ANALYSIS
# ============================================================

bootstrap_results = []

if best_pair is not None:

    model_a, model_b = best_pair

    model_a_oof = all_oof_predictions[model_a]
    model_b_oof = all_oof_predictions[model_b]

    model_a_validation = all_validation_predictions[model_a]
    model_b_validation = all_validation_predictions[model_b]

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    print("\n\n" + "=" * 80)
    print("BOOTSTRAP ANALYSIS")
    print("=" * 80)

    for bootstrap_iteration in range(
        N_BOOTSTRAP
    ):

        sampled_indices = rng.integers(
            low=0,
            high=len(development_y),
            size=len(development_y)
        )

        sampled_y = development_y[
            sampled_indices
        ]

        sampled_a = model_a_oof[
            sampled_indices
        ]

        sampled_b = model_b_oof[
            sampled_indices
        ]

        best_bootstrap_weight = None
        best_bootstrap_oof_auc = -np.inf

        for weight_a in BLEND_WEIGHT_GRID:

            weight_b = 1.0 - weight_a

            sampled_blend = (
                weight_a * sampled_a
                + weight_b * sampled_b
            )

            sampled_auc = safe_auc(
                sampled_y,
                sampled_blend
            )

            if (
                np.isfinite(sampled_auc)
                and sampled_auc > best_bootstrap_oof_auc
            ):
                best_bootstrap_oof_auc = sampled_auc
                best_bootstrap_weight = weight_a

        if best_bootstrap_weight is None:
            continue

        best_bootstrap_weight_b = (
            1.0 - best_bootstrap_weight
        )

        bootstrap_validation_predictions = (
            best_bootstrap_weight * model_a_validation
            + best_bootstrap_weight_b * model_b_validation
        )

        bootstrap_validation_auc = safe_auc(
            validation_y,
            bootstrap_validation_predictions
        )

        bootstrap_results.append({
            "bootstrap_iteration": bootstrap_iteration + 1,
            "selected_weight_a": best_bootstrap_weight,
            "selected_weight_b": best_bootstrap_weight_b,
            "sampled_oof_auc": best_bootstrap_oof_auc,
            "fixed_validation_auc": bootstrap_validation_auc
        })

        if (
            bootstrap_iteration + 1 == 1
            or (bootstrap_iteration + 1) % 5 == 0
        ):
            print(
                f"Bootstrap {bootstrap_iteration + 1:02d}/"
                f"{N_BOOTSTRAP} | "
                f"Sampled OOF AUC: "
                f"{best_bootstrap_oof_auc:.6f} | "
                f"Fixed validation AUC: "
                f"{bootstrap_validation_auc:.6f} | "
                f"Weight A: {best_bootstrap_weight:.2f}"
            )

    bootstrap_results_df = pd.DataFrame(
        bootstrap_results
    )

    if len(bootstrap_results_df) > 0:

        print("\nBootstrap summary:")

        print(
            "Sampled OOF AUC mean: "
            f"{bootstrap_results_df['sampled_oof_auc'].mean():.6f}"
        )

        print(
            "Sampled OOF AUC std: "
            f"{bootstrap_results_df['sampled_oof_auc'].std():.6f}"
        )

        print(
            "Fixed validation AUC mean: "
            f"{bootstrap_results_df['fixed_validation_auc'].mean():.6f}"
        )

        print(
            "Fixed validation AUC std: "
            f"{bootstrap_results_df['fixed_validation_auc'].std():.6f}"
        )

        print(
            "Selected weight A mean: "
            f"{bootstrap_results_df['selected_weight_a'].mean():.4f}"
        )

        print(
            "Selected weight A std: "
            f"{bootstrap_results_df['selected_weight_a'].std():.4f}"
        )

        bootstrap_results_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "bootstrap_results.csv"
            ),
            index=False
        )

    else:
        bootstrap_results_df = pd.DataFrame()

        print(
            "Bootstrap analysis did not produce valid results."
        )

else:
    bootstrap_results_df = pd.DataFrame()

    print(
        "Bootstrap analysis skipped because no best pair "
        "was available."
    )


# ============================================================
# 21. MODEL DIVERSITY ANALYSIS
# ============================================================

print("\n\n" + "=" * 80)
print("MODEL DIVERSITY ANALYSIS")
print("=" * 80)

if len(eligible_model_names) >= 2:

    diversity_correlation = np.corrcoef(
        oof_matrix.T
    )

    diversity_correlation_df = pd.DataFrame(
        diversity_correlation,
        index=eligible_model_names,
        columns=eligible_model_names
    )

    print("\nOOF prediction correlation matrix:")
    print(
        diversity_correlation_df.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    pairwise_prediction_differences = []

    for model_a, model_b in combinations(
        eligible_model_names,
        2
    ):
        mean_absolute_difference = np.mean(
            np.abs(
                all_oof_predictions[model_a]
                - all_oof_predictions[model_b]
            )
        )

        pairwise_prediction_differences.append(
            mean_absolute_difference
        )

    print(
        "\nMean absolute prediction difference: "
        f"{np.mean(pairwise_prediction_differences):.6f}"
    )

else:
    print(
        "Diversity analysis skipped because fewer than "
        "two models are available."
    )


# ============================================================
# 22. FINAL RECOMMENDATION AND CONFIDENCE
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

if best_blend_name is None:

    recommendation = best_individual_model
    confidence = "LOW"
    recommendation_reason = (
        "No valid ensemble was available. "
        "Use the best individual model."
    )

else:

    validation_gain = (
        best_blend_validation_auc
        - best_individual_validation_auc
    )

    oof_gain = (
        best_blend_oof_auc
        - best_individual_oof_auc
    )

    if (
        oof_gain > 0
        and validation_gain > 0
    ):
        recommendation = best_blend_name
        confidence = "HIGH"
        recommendation_reason = (
            "The blend improved both OOF AUC and fixed-validation AUC "
            "relative to the best individual model."
        )

    elif (
        oof_gain > 0
        and validation_gain >= -0.001
    ):
        recommendation = best_blend_name
        confidence = "MEDIUM"
        recommendation_reason = (
            "The blend improved OOF AUC and showed approximately "
            "stable fixed-validation performance."
        )

    elif (
        oof_gain > 0
        and validation_gain < 0
    ):
        recommendation = best_individual_model
        confidence = "MEDIUM"
        recommendation_reason = (
            "The blend improved OOF AUC but reduced fixed-validation "
            "AUC. The individual model is more consistent."
        )

    else:
        recommendation = best_individual_model
        confidence = "HIGH"
        recommendation_reason = (
            "The blend did not improve OOF AUC over the best "
            "individual model."
        )

print(f"Recommended model or ensemble: {recommendation}")
print(f"Confidence: {confidence}")
print(f"Reason: {recommendation_reason}")


# ============================================================
# 23. SAVE FINAL REPORTS
# ============================================================

individual_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "individual_model_results.csv"
    ),
    index=False
)

ensemble_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_results.csv"
    ),
    index=False
)

all_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_model_results.csv"
    ),
    index=False
)

final_summary = {
    "experiment": "EXP-011",
    "device": str(DEVICE),
    "seed": SEED,
    "n_outer_folds": N_OUTER_FOLDS,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "num_workers": NUM_WORKERS,
    "use_amp": USE_AMP,
    "eligible_models": eligible_model_names,
    "best_individual_model": best_individual_model,
    "best_individual_oof_auc": float(
        best_individual_oof_auc
    ),
    "best_individual_validation_auc": float(
        best_individual_validation_auc
    ),
    "best_blend_name": best_blend_name,
    "best_blend_oof_auc": (
        None
        if not np.isfinite(best_blend_oof_auc)
        else float(best_blend_oof_auc)
    ),
    "best_blend_validation_auc": (
        None
        if not np.isfinite(best_blend_validation_auc)
        else float(best_blend_validation_auc)
    ),
    "oof_improvement": (
        None
        if not np.isfinite(improvement_oof)
        else float(improvement_oof)
    ),
    "validation_improvement": (
        None
        if not np.isfinite(improvement_validation)
        else float(improvement_validation)
    ),
    "recommendation": recommendation,
    "confidence": confidence,
    "recommendation_reason": recommendation_reason,
    "best_pair": best_pair,
    "best_pair_weight": best_pair_weight
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "final_summary.json"
    ),
    "w"
) as file:
    json.dump(
        final_summary,
        file,
        indent=2,
        default=str
    )

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_metadata.json"
    ),
    "w"
) as file:
    json.dump(
        training_metadata,
        file,
        indent=2,
        default=str
    )


# ============================================================
# 24. FINAL OUTPUT FILE LIST
# ============================================================

print("\n\n" + "=" * 80)
print("EXP-011 COMPLETED")
print("=" * 80)

print(f"\nAll output files are saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR
)
):
    print(f" - {filename}")

print("\nFinal recommendation:")
print(f" - Model: {recommendation}")
print(f" - Confidence: {confidence}")
print(f" - Reason: {recommendation_reason}")

print("\nExperiment finished successfully.")