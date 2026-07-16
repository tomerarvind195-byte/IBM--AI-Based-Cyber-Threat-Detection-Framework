"""
Step 4: Data Preprocessing for NSL-KDD Dataset
AI-Based Cyber Threat Detection Framework
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "KDDTrain+.txt"
TEST_PATH = BASE_DIR / "data" / "KDDTest+.txt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found:\n{TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found:\n{TEST_PATH}")

# ------------------------------------------------------------
# NSL-KDD Columns
# ------------------------------------------------------------
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

print("Loading dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=columns,
    sep=",",
    engine="python",
    on_bad_lines="skip"
)

test_df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns,
    sep=",",
    engine="python",
    on_bad_lines="skip"
)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# ------------------------------------------------------------
# Remove difficulty column
# ------------------------------------------------------------
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"], inplace=True)

# ------------------------------------------------------------
# Remove duplicates & missing values
# ------------------------------------------------------------
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# ------------------------------------------------------------
# Binary labels
# ------------------------------------------------------------
train_df["binary_label"] = train_df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

test_df["binary_label"] = test_df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

print("\nBinary Class Distribution:")
print(train_df["binary_label"].value_counts())

# ------------------------------------------------------------
# Encode categorical columns
# ------------------------------------------------------------
categorical_cols = ["protocol_type", "service", "flag"]

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    train_df[col] = le.fit_transform(train_df[col])

    known = set(le.classes_)

    test_df[col] = test_df[col].apply(
        lambda x: x if x in known else "unseen"
    )

    le.classes_ = np.append(le.classes_, "unseen")

    test_df[col] = le.transform(test_df[col])

    encoders[col] = le

# ------------------------------------------------------------
# Features & Labels
# ------------------------------------------------------------
feature_cols = [
    c for c in train_df.columns
    if c not in ["label", "binary_label"]
]

X_train = train_df[feature_cols]
y_train = train_df["binary_label"]

X_test = test_df[feature_cols]
y_test = test_df["binary_label"]

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols
)

# ------------------------------------------------------------
# SMOTE
# ------------------------------------------------------------
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE :", y_train_balanced.value_counts().to_dict())

# ------------------------------------------------------------
# Save processed datasets
# ------------------------------------------------------------
X_train_balanced["binary_label"] = y_train_balanced
X_test_scaled["binary_label"] = y_test.values

train_file = OUTPUT_DIR / "train_processed.csv"
test_file = OUTPUT_DIR / "test_processed.csv"

X_train_balanced.to_csv(train_file, index=False)
X_test_scaled.to_csv(test_file, index=False)

print("\nSaved:")
print(train_file)
print(test_file)

# ------------------------------------------------------------
# Save preprocessing objects
# ------------------------------------------------------------
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
joblib.dump(encoders, OUTPUT_DIR / "encoders.pkl")
joblib.dump(feature_cols, OUTPUT_DIR / "feature_columns.pkl")

print("\nSaved preprocessing objects:")
print("scaler.pkl")
print("encoders.pkl")
print("feature_columns.pkl")

print("\nPreprocessing Completed Successfully!")