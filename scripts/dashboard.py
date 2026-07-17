"""
Step 9: Live Dashboard
AI-Based Cyber Threat Detection Framework
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="AI Cyber Threat Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

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
# Check Files
# ==========================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    TEST_FILE
]

missing = []

for file in required_files:
    if not file.exists():
        missing.append(str(file))

if missing:
    st.error("Missing Files")
    st.code("\n".join(missing))
    st.stop()

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, encoders, feature_columns


model, scaler, encoders, feature_columns = load_artifacts()

categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

RAW_COLUMNS = [
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
# Load Test Dataset
# ==========================================================

@st.cache_data
def load_test_data():

    df = pd.read_csv(
        TEST_FILE,
        names=RAW_COLUMNS,
        nrows=5000
    )

    return df


raw_test = load_test_data()

# ==========================================================
# Preprocessing
# ==========================================================

def preprocess_record(raw_record):

    df = pd.DataFrame([raw_record])

    for col in categorical_cols:

        le = encoders[col]

        values = []

        for x in df[col]:

            if x in le.classes_:
                values.append(x)
            else:
                values.append(le.classes_[0])

        df[col] = le.transform(values)

    df = df[feature_columns]

    df = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )

    return df

# ==========================================================
# Prediction
# ==========================================================

def severity(confidence):

    if confidence >= 0.90:
        return "HIGH"

    elif confidence >= 0.70:
        return "MEDIUM"

    return "LOW"


def detect(raw_record):

    X = preprocess_record(raw_record)

    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0][pred]

    return {
        "prediction":
            "ATTACK" if pred == 1 else "NORMAL",

        "confidence":
            round(float(prob), 4),

        "severity":
            severity(prob) if pred == 1 else "-"
    }

# ==========================================================
# Sidebar Controls
# ==========================================================

st.sidebar.title("⚙️ Controls")

n_records = st.sidebar.slider(
    "Number of Records",
    min_value=5,
    max_value=50,
    value=20,
    step=1,
    key="n_records_slider"
)

speed = st.sidebar.slider(
    "Delay (seconds)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    key="speed_slider"
)

start_btn = st.sidebar.button(
    "▶ Start Live Detection",
    type="primary"
)

reset_btn = st.sidebar.button("🔄 Reset Dashboard")

# ==========================================================
# Title
# ==========================================================

st.title("🛡 AI-Based Cyber Threat Detection Framework")

st.markdown(
    """
Live Network Traffic Monitoring using **XGBoost**
"""
)

# ==========================================================
# Session State
# ==========================================================

if "logs" not in st.session_state:
    st.session_state.logs = []

# Counter used only to build unique widget keys for repeated
# plotly_chart / dataframe calls inside render_dashboard().
if "render_count" not in st.session_state:
    st.session_state.render_count = 0

if reset_btn:
    st.session_state.logs = []
    st.session_state.render_count = 0

# ==========================================================
# Dashboard Metrics
# ==========================================================

metric1, metric2, metric3, metric4 = st.columns(4)

m_total = metric1.empty()
m_normal = metric2.empty()
m_attack = metric3.empty()
m_high = metric4.empty()

alert_box = st.empty()

chart_col1, chart_col2 = st.columns(2)

pie_placeholder = chart_col1.empty()

bar_placeholder = chart_col2.empty()

table_placeholder = st.empty()

# ==========================================================
# Dashboard Render Function
# ==========================================================

def render_dashboard(logs):

    if len(logs) == 0:

        m_total.metric("Total Records", 0)
        m_normal.metric("Normal", 0)
        m_attack.metric("Attacks", 0)
        m_high.metric("High Severity", 0)

        return

    # Bump the counter every time we render so each plotly_chart /
    # dataframe call below gets a fresh, unique key even when the
    # chart's data/parameters happen to be identical to a previous call.
    st.session_state.render_count += 1
    render_id = st.session_state.render_count

    df = pd.DataFrame(logs)

    total = len(df)

    attack = len(df[df["Prediction"] == "ATTACK"])

    normal = total - attack

    high = len(df[df["Severity"] == "HIGH"])

    m_total.metric("Total Records", total)

    m_normal.metric("🟢 Normal", normal)

    m_attack.metric("🔴 Attacks", attack)

    m_high.metric("⚠ High Severity", high)

    latest = df.iloc[-1]

    if latest["Prediction"] == "ATTACK":

        alert_box.error(
            f"""
🚨 Attack Detected

Confidence : {latest['Confidence']:.2%}

Severity : {latest['Severity']}
"""
        )

    else:

        alert_box.success(
            f"""
✅ Normal Traffic

Confidence : {latest['Confidence']:.2%}
"""
        )

    # Pie Chart

    pie = px.pie(
        df,
        names="Prediction",
        title="Traffic Distribution",
        color="Prediction",
        color_discrete_map={
            "NORMAL": "green",
            "ATTACK": "red"
        }
    )

    pie_placeholder.plotly_chart(
        pie,
        use_container_width=True,
        key=f"pie_chart_{render_id}"
    )

    # Severity Chart

    attack_df = df[df["Prediction"] == "ATTACK"]

    if len(attack_df):

        bar = px.histogram(
            attack_df,
            x="Severity",
            title="Attack Severity",
            color="Severity",
            category_orders={
                "Severity": [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ]
            }
        )

        bar_placeholder.plotly_chart(
            bar,
            use_container_width=True,
            key=f"bar_chart_{render_id}"
        )

    table_placeholder.dataframe(
        df.iloc[::-1],
        use_container_width=True,
        height=350,
        key=f"log_table_{render_id}"
    )

# ==========================================================
# Render Existing Logs
# ==========================================================

render_dashboard(st.session_state.logs)

# ==========================================================
# Start Live Detection
# ==========================================================

if start_btn:

    sample = raw_test.sample(
        n=n_records,
        random_state=np.random.randint(10000)
    ).reset_index(drop=True)

    progress = st.progress(0)

    status = st.empty()

    for i, row in sample.iterrows():

        raw_record = row.drop(
            labels=["label", "difficulty"]
        ).to_dict()

        result = detect(raw_record)

        st.session_state.logs.append({

            "Record": len(st.session_state.logs) + 1,

            "Prediction": result["prediction"],

            "Confidence": result["confidence"],

            "Severity": result["severity"],

            "Protocol": raw_record["protocol_type"],

            "Service": raw_record["service"]

        })

        render_dashboard(st.session_state.logs)

        progress.progress((i + 1) / len(sample))

        status.info(
            f"Processing Record {i+1} of {len(sample)}"
        )

        time.sleep(speed)

    progress.empty()

    status.success("Live Detection Completed Successfully!")

# ==========================================================
# Download Simulation Log
# ==========================================================

if len(st.session_state.logs):

    csv = pd.DataFrame(
        st.session_state.logs
    ).to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Detection Log",

        data=csv,

        file_name="realtime_detection_log.csv",

        mime="text/csv"

    )

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(
    """
### 📊 Dashboard Summary

This dashboard simulates a real-time intrusion detection system using the **NSL-KDD dataset**.

### Features

- ✅ Live Network Traffic Monitoring
- ✅ XGBoost Based Threat Detection
- ✅ Attack / Normal Classification
- ✅ Confidence Score
- ✅ Severity Prediction
- ✅ Live Charts
- ✅ Download Detection Logs

---
**AI-Based Cyber Threat Detection Framework**

Developed using **Python, Scikit-Learn, XGBoost, Pandas, Plotly and Streamlit**
"""
)