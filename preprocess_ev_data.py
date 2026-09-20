# ============================================================
# EV PURCHASE PREDICTION COMPETITION
# COMPLETE DATA PREPROCESSING PIPELINE
# ============================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

ONEHOT_TRAIN_FILE = "onehotenc_train.csv"
ONEHOT_TEST_FILE = "onehotenc_test.csv"

SCALED_TRAIN_FILE = "fullypreprocessed_train.csv"
SCALED_TEST_FILE = "fullypreprocessed_test.csv"

TARGET_COLUMN = "Will_Buy_EV"
ID_COLUMN = "id"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_one_hot_encoder():
    """
    Creates a OneHotEncoder compatible with different
    scikit-learn versions.
    """

    try:
        # Newer scikit-learn versions
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:
        # Older scikit-learn versions
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def validate_columns(train_df, test_df):
    """
    Validates the expected dataset structure.
    """

    required_columns = [
        "id",
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Will_Buy_EV"
    ]

    missing_train_columns = [
        column for column in required_columns
        if column not in train_df.columns
    ]

    test_required_columns = [
        column for column in required_columns
        if column != TARGET_COLUMN
    ]

    missing_test_columns = [
        column for column in test_required_columns
        if column not in test_df.columns
    ]

    if missing_train_columns:
        raise ValueError(
            f"Missing columns in train.csv: {missing_train_columns}"
        )

    if missing_test_columns:
        raise ValueError(
            f"Missing columns in test.csv: {missing_test_columns}"
        )

    print("Column validation successful.")


def encode_binary_columns(df):
    """
    Encodes binary Yes/No columns into 1/0.
    """

    binary_columns = [
        "Home_Charging_Possible",
        "Subsidy_Available"
    ]

    df = df.copy()

    for column in binary_columns:
        if column in df.columns:
            unique_values = set(df[column].dropna().unique())

            if unique_values.issubset({"Yes", "No"}):
                df[column] = df[column].map({
                    "Yes": 1,
                    "No": 0
                })

            elif unique_values.issubset({0, 1}):
                df[column] = df[column].astype(int)

            else:
                raise ValueError(
                    f"Unexpected values found in {column}: "
                    f"{unique_values}"
                )

    return df


def encode_target(target_series):
    """
    Converts the target column from Yes/No to 1/0.
    """

    unique_values = set(target_series.dropna().unique())

    if unique_values.issubset({"Yes", "No"}):
        encoded_target = target_series.map({
            "Yes": 1,
            "No": 0
        })

    elif unique_values.issubset({0, 1}):
        encoded_target = target_series.astype(int)

    else:
        raise ValueError(
            f"Unexpected target values: {unique_values}"
        )

    if encoded_target.isna().any():
        raise ValueError(
            "Target encoding produced missing values."
        )

    return encoded_target.astype(int)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print_section("STARTING EV DATA PREPROCESSING")

    # --------------------------------------------------------
    # 1. CHECK INPUT FILES
    # --------------------------------------------------------

    print_section("1. CHECKING INPUT FILES")

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Could not find {TRAIN_FILE} in the current directory."
        )

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Could not find {TEST_FILE} in the current directory."
        )

    print(f"Found {TRAIN_FILE}")
    print(f"Found {TEST_FILE}")

    # --------------------------------------------------------
    # 2. LOAD DATASETS
    # --------------------------------------------------------

    print_section("2. LOADING DATASETS")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    validate_columns(train_df, test_df)

    # Keep original IDs for later use
    train_ids = train_df[ID_COLUMN].copy()
    test_ids = test_df[ID_COLUMN].copy()

    # --------------------------------------------------------
    # 3. CHECK MISSING VALUES AND DUPLICATES
    # --------------------------------------------------------

    print_section("3. DATA QUALITY CHECKS")

    print("\nMissing values in train:")
    print(train_df.isnull().sum().sum())

    print("\nMissing values in test:")
    print(test_df.isnull().sum().sum())

    print("\nDuplicate rows in train:")
    print(train_df.duplicated().sum())

    print("\nDuplicate rows in test:")
    print(test_df.duplicated().sum())

    if train_df.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values found in train.csv. "
            "Handle them before continuing."
        )

    if test_df.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values found in test.csv. "
            "Handle them before continuing."
        )

    # --------------------------------------------------------
    # 4. ENCODE TARGET
    # --------------------------------------------------------

    print_section("4. ENCODING TARGET COLUMN")

    train_target = encode_target(train_df[TARGET_COLUMN])

    print("Target encoding:")
    print("Yes -> 1")
    print("No  -> 0")

    print("\nTarget distribution:")
    print(train_target.value_counts())

    # --------------------------------------------------------
    # 5. PREPARE FEATURES
    # --------------------------------------------------------

    print_section("5. PREPARING FEATURES")

    # Remove target and ID from the feature data
    X_train = train_df.drop(
        columns=[TARGET_COLUMN, ID_COLUMN]
    ).copy()

    X_test = test_df.drop(
        columns=[ID_COLUMN]
    ).copy()

    # Encode binary columns
    X_train = encode_binary_columns(X_train)
    X_test = encode_binary_columns(X_test)

    # Ensure both datasets have the same column order
    X_test = X_test[X_train.columns]

    print(f"Feature shape before one-hot encoding: {X_train.shape}")

    # --------------------------------------------------------
    # 6. IDENTIFY CATEGORICAL COLUMNS
    # --------------------------------------------------------

    print_section("6. IDENTIFYING CATEGORICAL COLUMNS")

    categorical_columns = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Range_Anxiety_Level"
    ]

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    print("Categorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    # --------------------------------------------------------
    # 7. ONE-HOT ENCODING
    # --------------------------------------------------------

    print_section("7. APPLYING ONE-HOT ENCODING")

    encoder = create_one_hot_encoder()

    # Fit only on training data
    train_categorical = X_train[categorical_columns]
    test_categorical = X_test[categorical_columns]

    encoded_train_array = encoder.fit_transform(train_categorical)
    encoded_test_array = encoder.transform(test_categorical)

    encoded_feature_names = encoder.get_feature_names_out(
        categorical_columns
    )

    encoded_train_df = pd.DataFrame(
        encoded_train_array,
        columns=encoded_feature_names,
        index=X_train.index
    )

    encoded_test_df = pd.DataFrame(
        encoded_test_array,
        columns=encoded_feature_names,
        index=X_test.index
    )

    # Keep numerical columns unchanged
    train_numerical_df = X_train[numerical_columns].copy()
    test_numerical_df = X_test[numerical_columns].copy()

    # Combine numerical and one-hot encoded columns
    onehot_train_features = pd.concat(
        [
            train_numerical_df.reset_index(drop=True),
            encoded_train_df.reset_index(drop=True)
        ],
        axis=1
    )

    onehot_test_features = pd.concat(
        [
            test_numerical_df.reset_index(drop=True),
            encoded_test_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Ensure identical column order
    onehot_test_features = onehot_test_features[
        onehot_train_features.columns
    ]

    print(
        "Train shape after one-hot encoding:",
        onehot_train_features.shape
    )

    print(
        "Test shape after one-hot encoding:",
        onehot_test_features.shape
    )

    # --------------------------------------------------------
    # 8. SAVE ONE-HOT ENCODED DATASETS
    # --------------------------------------------------------

    print_section("8. SAVING ONE-HOT ENCODED DATASETS")

    onehot_train_output = onehot_train_features.copy()
    onehot_test_output = onehot_test_features.copy()

    # Add ID back to the beginning
    onehot_train_output.insert(
        0,
        ID_COLUMN,
        train_ids.reset_index(drop=True)
    )

    onehot_test_output.insert(
        0,
        ID_COLUMN,
        test_ids.reset_index(drop=True)
    )

    # Add encoded target only to training data
    onehot_train_output[TARGET_COLUMN] = train_target.reset_index(
        drop=True
    )

    onehot_train_output.to_csv(
        ONEHOT_TRAIN_FILE,
        index=False
    )

    onehot_test_output.to_csv(
        ONEHOT_TEST_FILE,
        index=False
    )

    print(f"Saved: {ONEHOT_TRAIN_FILE}")
    print(f"Saved: {ONEHOT_TEST_FILE}")

    # --------------------------------------------------------
    # 9. SCALE FEATURES
    # --------------------------------------------------------

    print_section("9. SCALING FEATURES")

    scaler = StandardScaler()

    # Scale only actual model features.
    # ID and target are never scaled.
    scaled_train_array = scaler.fit_transform(
        onehot_train_features
    )

    scaled_test_array = scaler.transform(
        onehot_test_features
    )

    scaled_train_features = pd.DataFrame(
        scaled_train_array,
        columns=onehot_train_features.columns
    )

    scaled_test_features = pd.DataFrame(
        scaled_test_array,
        columns=onehot_test_features.columns
    )

    print(
        "Scaled train shape:",
        scaled_train_features.shape
    )

    print(
        "Scaled test shape:",
        scaled_test_features.shape
    )

    # --------------------------------------------------------
    # 10. SAVE FULLY PREPROCESSED DATASETS
    # --------------------------------------------------------

    print_section("10. SAVING FULLY PREPROCESSED DATASETS")

    scaled_train_output = scaled_train_features.copy()
    scaled_test_output = scaled_test_features.copy()

    # Add original IDs back to the beginning
    scaled_train_output.insert(
        0,
        ID_COLUMN,
        train_ids.reset_index(drop=True)
    )

    scaled_test_output.insert(
        0,
        ID_COLUMN,
        test_ids.reset_index(drop=True)
    )

    # Add target only to the training dataset
    scaled_train_output[TARGET_COLUMN] = train_target.reset_index(
        drop=True
    )

    scaled_train_output.to_csv(
        SCALED_TRAIN_FILE,
        index=False
    )

    scaled_test_output.to_csv(
        SCALED_TEST_FILE,
        index=False
    )

    print(f"Saved: {SCALED_TRAIN_FILE}")
    print(f"Saved: {SCALED_TEST_FILE}")

    # --------------------------------------------------------
    # 11. FINAL VALIDATION
    # --------------------------------------------------------

    print_section("11. FINAL VALIDATION")

    output_files = [
        ONEHOT_TRAIN_FILE,
        ONEHOT_TEST_FILE,
        SCALED_TRAIN_FILE,
        SCALED_TEST_FILE
    ]

    for file in output_files:
        if os.path.exists(file):
            file_size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"{file}: Created successfully ({file_size_mb:.2f} MB)")
        else:
            print(f"{file}: NOT FOUND")

    # Check row counts
    onehot_train_check = pd.read_csv(ONEHOT_TRAIN_FILE, nrows=5)
    onehot_test_check = pd.read_csv(ONEHOT_TEST_FILE, nrows=5)
    scaled_train_check = pd.read_csv(SCALED_TRAIN_FILE, nrows=5)
    scaled_test_check = pd.read_csv(SCALED_TEST_FILE, nrows=5)

    print("\nOutput column counts:")
    print(
        f"{ONEHOT_TRAIN_FILE}: {len(onehot_train_check.columns)}"
    )
    print(
        f"{ONEHOT_TEST_FILE}: {len(onehot_test_check.columns)}"
    )
    print(
        f"{SCALED_TRAIN_FILE}: {len(scaled_train_check.columns)}"
    )
    print(
        f"{SCALED_TEST_FILE}: {len(scaled_test_check.columns)}"
    )

    print_section("PREPROCESSING COMPLETED SUCCESSFULLY")

    print("Generated files:")
    print(f"1. {ONEHOT_TRAIN_FILE}")
    print(f"2. {ONEHOT_TEST_FILE}")
    print(f"3. {SCALED_TRAIN_FILE}")
    print(f"4. {SCALED_TEST_FILE}")

    print("\nImportant:")
    print("- Use one-hot encoded data for tree-based models.")
    print("- Use scaled data for KNN, SVM, and neural networks.")
    print("- Do not use the ID column as a model feature.")
    print("- Keep test IDs for creating the final submission.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nERROR OCCURRED:")
        print(error)
        sys.exit(1)