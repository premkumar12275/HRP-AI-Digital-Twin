import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("output/merged_dataset.csv")
X = data.drop(columns=["ICUSTAY_ID", "label"])
y = data["label"]

# Load or re-train model
model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X, y)

# Create TreeExplainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Global summary plot
shap.summary_plot(shap_values, X, show=True)
plt.savefig("output/shap_summary.png", bbox_inches="tight")

# (Optional) Force plot for individual patient
shap.initjs()
for i in range(3):  # First 3 patients
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[i].values,
        X.iloc[i],
        feature_names=X.columns
    )
    with open(f"output/force_plot_patient_{i}.html", "w") as f:
        f.write(shap_html.html())

    
