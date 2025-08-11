import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier
import joblib
import os

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to merged dataset CSV")
parser.add_argument("--label_col", type=str, default="label", help="Column name for label")
parser.add_argument("--id_col", type=str, default="ICUSTAY_ID", help="Column to ignore")
args = parser.parse_args()

# --- Load and prep ---
data = pd.read_csv(args.data)
X = data.drop(columns=[args.id_col, args.label_col])
y = data[args.label_col]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Train ---
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# --- Predict ---
y_pred = model.predict_proba(X_test)[:, 1]

# --- Evaluate ---
print(f"AUROC:  {roc_auc_score(y_test, y_pred):.4f}")
print(f"AUPRC:  {average_precision_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred > 0.5))

# Create output folder if needed
os.makedirs("models", exist_ok=True)

# Save model
model_path = f"models/xgboost_model_{os.path.basename(args.data).replace('.csv','')}.joblib"
joblib.dump(model, model_path)

print(f"[done] Model saved to: {model_path}")
