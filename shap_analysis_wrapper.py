import pandas as pd
import shap
import joblib
import argparse
import matplotlib.pyplot as plt
import os

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to XGBoost .joblib model")
parser.add_argument("--data", type=str, required=True, help="Path to dataset used for training")
parser.add_argument("--id_col", type=str, default="ICUSTAY_ID", help="ID column to ignore")
parser.add_argument("--label_col", type=str, default="label", help="Target label column")
parser.add_argument("--prefix", type=str, default="model", help="Prefix for output files")
args = parser.parse_args()

# --- Load data and model ---
print(f"[INFO] Loading model: {args.model}")
model = joblib.load(args.model)

print(f"[INFO] Loading data: {args.data}")
df = pd.read_csv(args.data)
X = df.drop(columns=[args.label_col, args.id_col], errors="ignore")

# --- SHAP ---
print("[INFO] Running SHAP analysis...")
explainer = shap.Explainer(model)
shap_values = explainer(X)

# --- Save SHAP summary plot ---
os.makedirs("output", exist_ok=True)
summary_path = f"output/shap_summary_{args.prefix}.png"
shap.summary_plot(shap_values, X, show=False)
plt.savefig(summary_path, bbox_inches="tight")
plt.close()
print(f"[done] SHAP summary saved to: {summary_path}")

# --- Save force plots for top 3 samples ---
shap.initjs()
for i in range(3):
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[i].values,
        X.iloc[i],
        feature_names=X.columns
    )
    with open(f"output/force_plot_{args.prefix}_{i}.html", "w") as f:
        f.write(force_plot.html())
print("[done] Top 3 SHAP force plots saved")


#Example Usage
#For vitals-only model:
#python shap_analysis_wrapper.py \
#  --model models/xgboost_model_vitals.joblib \
#  --data output/merged_dataset.csv \
#  --prefix vitals
#For labs+vitals model:
#python shap_analysis_wrapper.py \
#  --model models/xgboost_model_merged_dataset_with_labs.joblib \
#  --data output/merged_dataset_with_labs.csv \
#  --prefix with_labs