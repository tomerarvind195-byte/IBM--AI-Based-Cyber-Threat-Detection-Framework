"""
SHAP Explainability for Threat Detection
AI-Based Cyber Threat Detection Framework
"""

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

model = joblib.load(OUTPUT_DIR / "best_model.pkl")
feature_columns = joblib.load(OUTPUT_DIR / "feature_columns.pkl")
test_df = pd.read_csv(OUTPUT_DIR / "test_processed.csv")

X_test = test_df.drop(columns=["binary_label"])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:200])

plt.figure()
shap.summary_plot(shap_values, X_test.iloc[:200], show=False)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=150)
plt.close()

print("SHAP summary plot saved to assets/shap_summary.png")

def explain_single_prediction(record_index=0):
    single = X_test.iloc[[record_index]]
    sv = explainer.shap_values(single)

    contrib = pd.Series(sv[0], index=feature_columns)
    top_features = contrib.abs().sort_values(ascending=False).head(3)

    print(f"\nTop 3 features driving this prediction:")
    for feat in top_features.index:
        direction = "increased" if contrib[feat] > 0 else "decreased"
        print(f"  - {feat}: {direction} attack likelihood")

if __name__ == "__main__":
    explain_single_prediction(0)