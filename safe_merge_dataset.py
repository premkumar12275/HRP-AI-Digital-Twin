import pandas as pd

# Load robust feature sets
sepsis = pd.read_csv("output/features_robust.csv")
controls = pd.read_csv("output/control_features_robust.csv")

# Add binary labels
sepsis["label"] = 1
controls["label"] = 0

# Merge
data = pd.concat([sepsis, controls], ignore_index=True)

print(f"[INFO] Merged shape: {data.shape}")
print("[INFO] Missing value summary:\n", data.isnull().sum())

# Impute missing values
data = data.fillna(data.mean(numeric_only=True))

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged file
data.to_csv("output/merged_dataset.csv", index=False)
print("[DONE] Merged dataset saved to output/merged_dataset.csv")
