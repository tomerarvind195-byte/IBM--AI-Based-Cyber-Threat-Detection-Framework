"""
Step 6-7: Model Building + Evaluation
AI-Based Cyber Threat Detection Framework
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_FILE = OUTPUT_DIR / "train_processed.csv"
TEST_FILE = OUTPUT_DIR / "test_processed.csv"
MODEL_PATH = OUTPUT_DIR / "best_model.pkl"

# =====================================================
# Check Files
# =====================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"{TRAIN_FILE} not found!")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"{TEST_FILE} not found!")

print("Loading processed dataset...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train Shape :", train_df.shape)
print("Test Shape  :", test_df.shape)

# =====================================================
# Split Features & Labels
# =====================================================

X_train = train_df.drop(columns=["binary_label"])
y_train = train_df["binary_label"]

X_test = test_df.drop(columns=["binary_label"])
y_test = test_df["binary_label"]

# =====================================================
# Models
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear",
        max_iter=500,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
}

results = []
trained_models = {}

# =====================================================
# Train Models
# =====================================================

for name, model in models.items():

    print("\n" + "="*60)
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"FPR       : {fpr:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "False Positive Rate": fpr
    })

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal","Attack"],
        yticklabels=["Normal","Attack"]
    )

    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{name.replace(' ','_')}.png",
        dpi=150
    )

    plt.close()

# =====================================================
# Model Comparison
# =====================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\n================ MODEL COMPARISON ================\n")
print(results_df)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# =====================================================
# Save Best Model
# =====================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(best_model, MODEL_PATH)

print(f"\nBest Model : {best_model_name}")
print("Saved :", MODEL_PATH)

# =====================================================
# Feature Importance
# =====================================================

rf = trained_models["Random Forest"]

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values(
    ascending=False
).head(15)

plt.figure(figsize=(8,6))
importance.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Important Features")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150
)

plt.close()

print("\nFeature importance saved.")
print("\nTraining Completed Successfully!")