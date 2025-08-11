import pandas as pd

# Load vitals
sepsis_vitals = pd.read_csv("output/features_robust.csv")
control_vitals = pd.read_csv("output/control_features_robust.csv")

# Load labs
sepsis_labs = pd.read_csv("output/lab_features_sepsis.csv")
control_labs = pd.read_csv("output/lab_features_controls.csv")

# Merge vitals + labs on ICUSTAY_ID
sepsis = pd.merge(sepsis_vitals, sepsis_labs, on="ICUSTAY_ID", how="inner")
control = pd.merge(control_vitals, control_labs, on="ICUSTAY_ID", how="inner")

# Label each group
sepsis["label"] = 1
control["label"] = 0

# Combine and shuffle
data = pd.concat([sepsis, control], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Impute missing values
data = data.fillna(data.mean(numeric_only=True))

# Save merged dataset
data.to_csv("output/merged_dataset_with_labs.csv", index=False)
print(f"[DONE] Merged dataset saved → output/merged_dataset_with_labs.csv")
