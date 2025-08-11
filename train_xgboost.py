import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

# Load merged data
data = pd.read_csv("output/merged_dataset.csv")
X = data.drop(columns=["ICUSTAY_ID", "label"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Evaluate
print("AUROC:", roc_auc_score(y_test, y_pred))
print("AUPRC:", average_precision_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred > 0.5))
