import pandas as pd
import joblib
import argparse

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to .joblib model file")
parser.add_argument("--data", type=str, required=True, help="Path to input CSV with features")
parser.add_argument("--output", type=str, default="output/predictions.csv", help="Path to save predictions")
parser.add_argument("--id_col", type=str, default="ICUSTAY_ID", help="Column to keep for reference")
parser.add_argument("--label_col", type=str, default="label", help="Optional: known label column")
args = parser.parse_args()

# --- Load model ---
model = joblib.load(args.model)
print(f"[INFO] Loaded model: {args.model}")

# --- Load data ---
df = pd.read_csv(args.data)
X = df.drop(columns=[args.id_col], errors="ignore")
if args.label_col in X.columns:
    X = X.drop(columns=[args.label_col])

# --- Predict ---
y_pred = model.predict_proba(X)[:, 1]

# --- Save predictions ---
output_df = df[[args.id_col]].copy() if args.id_col in df.columns else pd.DataFrame()
output_df["predicted_sepsis_risk"] = y_pred

if args.label_col in df.columns:
    output_df["true_label"] = df[args.label_col]

output_df.to_csv(args.output, index=False)
print(f"[DONE] Predictions saved to: {args.output}")

# --- example usage ---
# python predict_models.py --model models/xgboost_model_merged_dataset_with_labs.joblib --data output/merged_dataset_with_labs.csv --output output/predictions.csv
# This will load the model, make predictions on the provided dataset, and save the results.