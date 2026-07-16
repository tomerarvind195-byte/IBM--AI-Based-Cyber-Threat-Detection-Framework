"""
Step 8: Real-Time Detection Pipeline
AI-Based Cyber Threat Detection Framework
"""

import time
from pathlib import Path

import joblib
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = OUTPUT_DIR / "best_model.pkl"
SCALER_PATH = OUTPUT_DIR / "scaler.pkl"
ENCODER_PATH = OUTPUT_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.pkl"

TEST_FILE = DATA_DIR / "KDDTest+.txt"

# ==========================================================
# Check Required Files
# ==========================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    TEST_FILE
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"\nMissing File:\n{file}")

print("=" * 60)
print("Loading trained model...")
print("=" * 60)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

# ==========================================================
# Feature Names
# ==========================================================

columns = [
    "duration","protocol_type","service","flag",
    "src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in",
    "num_compromised","root_shell",
    "su_attempted","num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

# ==========================================================
# Preprocess Single Record
# ==========================================================

def preprocess_record(raw_record):

    df = pd.DataFrame([raw_record])

    for col in categorical_cols:

        le = encoders[col]

        encoded = []

        for value in df[col]:

            if value in le.classes_:
                encoded.append(value)
            else:
                encoded.append(le.classes_[0])

        df[col] = le.transform(encoded)

    df = df[feature_columns]

    scaled = scaler.transform(df)

    return pd.DataFrame(
        scaled,
        columns=feature_columns
    )

# ==========================================================
# Severity Level
# ==========================================================

def get_severity(confidence):

    if confidence >= 0.95:
        return "CRITICAL"

    elif confidence >= 0.90:
        return "HIGH"

    elif confidence >= 0.70:
        return "MEDIUM"

    else:
        return "LOW"

# ==========================================================
# Detection Function
# ==========================================================

def detect_threat(raw_record):

    X = preprocess_record(raw_record)

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0]

    confidence = probability.max()

    return {

        "prediction":
            "ATTACK" if prediction == 1 else "NORMAL",

        "confidence":
            round(float(confidence),4),

        "severity":
            get_severity(confidence) if prediction==1 else "-"
    }

# ==========================================================
# Main Simulation
# ==========================================================

if __name__ == "__main__":

    print("\nLoading test dataset...\n")

    raw_test = pd.read_csv(
        TEST_FILE,
        names=columns
    )

    sample = raw_test.sample(
        n=20,
        random_state=42
    ).reset_index(drop=True)

    print("=" * 70)
    print(" REAL-TIME NETWORK THREAT DETECTION ")
    print("=" * 70)

    correct = 0
    logs = []

    for i, row in sample.iterrows():

        raw_record = row.drop(
            ["label","difficulty"]
        ).to_dict()

        true_label = (
            "NORMAL"
            if row["label"] == "normal"
            else "ATTACK"
        )

        result = detect_threat(raw_record)

        is_correct = (
            result["prediction"] == true_label
        )

        if is_correct:
            correct += 1

        print(
            f"[{i+1:02d}] "
            f"Prediction: {result['prediction']:7s} | "
            f"Confidence: {result['confidence']:.4f} | "
            f"Severity: {result['severity']:8s} | "
            f"Actual: {true_label}"
        )

        logs.append({

            "Record": i+1,

            "Prediction": result["prediction"],

            "Confidence": result["confidence"],

            "Severity": result["severity"],

            "Actual": true_label,

            "Correct": is_correct

        })

        time.sleep(0.25)

    accuracy = (correct / len(sample)) * 100

    print("\n" + "=" * 70)
    print(f"Simulation Accuracy : {accuracy:.2f}%")
    print(f"Correct Predictions : {correct}/{len(sample)}")
    print("=" * 70)

    log_file = OUTPUT_DIR / "realtime_simulation_log.csv"

    pd.DataFrame(logs).to_csv(
        log_file,
        index=False
    )

    print("\nSimulation log saved to:")
    print(log_file)

    print("\nReal-Time Detection Completed Successfully.")