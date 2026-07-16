"""
Step 6-7: Model Building + Evaluation
AI-Based Cyber Threat Detection Framework

Trains and compares 3 models on the preprocessed NSL-KDD data:
1. Logistic Regression  (baseline)
2. Random Forest        (stronger, gives feature importance)
3. XGBoost              (advanced, usually best performer)

Evaluates each on: Accuracy, Precision, Recall, F1-score, ROC-AUC,
Confusion Matrix, and False Positive Rate (important for security use-cases).
Saves a comparison table + confusion matrix plots + the best model.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# -------------------------------------------------------------------
# 1. Load preprocessed data (from Step 4)
# -------------------------------------------------------------------
train_df = pd.read_csv("/mnt/user-data/outputs/train_processed.csv")
test_df = pd.read_csv("/mnt/user-data/outputs/test_processed.csv")

X_train = train_df.drop(columns=["binary_label"])
y_train = train_df["binary_label"]

X_test = test_df.drop(columns=["binary_label"])
y_test = test_df["binary_label"]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------------------------------------------
# 2. Define models
# -------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="logloss", random_state=42
    ),
}

# -------------------------------------------------------------------
# 3. Train, predict, evaluate each model
# -------------------------------------------------------------------
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix -> False Positive Rate = FP / (FP + TN)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
        "False Positive Rate": fpr
    })

    print(f"{name} -> Acc: {acc:.4f}, Precision: {prec:.4f}, "
          f"Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, FPR: {fpr:.4f}")

    # Save confusion matrix plot
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    plt.savefig(f"/mnt/user-data/outputs/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

# -------------------------------------------------------------------
# 4. Comparison table
# -------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("/mnt/user-data/outputs/model_comparison.csv", index=False)

# -------------------------------------------------------------------
# 5. Save the best model (highest F1-score)
# -------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
joblib.dump(best_model, "/mnt/user-data/outputs/best_model.pkl")
print(f"\nBest model: {best_model_name} -> saved as best_model.pkl")

# -------------------------------------------------------------------
# 6. Feature importance (from Random Forest) - useful for the report
# -------------------------------------------------------------------
rf_model = trained_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(7, 5))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feature_importance.png", dpi=150)
plt.close()

print("\nSaved: model_comparison.csv, best_model.pkl, confusion matrix plots, feature_importance.png")
