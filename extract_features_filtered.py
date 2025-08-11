import pandas as pd
from tqdm import tqdm
import numpy as np

# Load filtered chart data
chart = pd.read_csv("output/filtered_chartevents.csv.gz", parse_dates=["CHARTTIME"])
labels = pd.read_csv("output/sepsis_labels.csv")  # or control_labels.csv
labels["SEPSIS_ONSET"] = pd.to_datetime(labels["SEPSIS_ONSET"])

ITEM_IDS = {
    "Heart Rate": [211],
    "Systolic BP": [51],
    "Diastolic BP": [8368],
    "Mean BP": [456],
    "Respiratory Rate": [618],
    "SpO2": [646],
    "Temperature": [223761]
}

features = []

print("[INFO] Extracting features from filtered data (robust)...")

for _, row in tqdm(labels.iterrows(), total=len(labels)):
    icu_id = row["ICUSTAY_ID"]
    onset = row["SEPSIS_ONSET"]
    start = onset - pd.Timedelta(hours=12)

    window = chart[
        (chart["ICUSTAY_ID"] == icu_id) &
        (chart["CHARTTIME"] >= start) &
        (chart["CHARTTIME"] <= onset)
    ]

    feat_row = {"ICUSTAY_ID": icu_id}

    for name, ids in ITEM_IDS.items():
        subset = window[window["ITEMID"].isin(ids)]
        values = subset["VALUENUM"].dropna()

        feat_row[f"{name}_mean"] = values.mean() if not values.empty else np.nan
        feat_row[f"{name}_std"] = values.std() if not values.empty else np.nan
        feat_row[f"{name}_delta"] = (
            values.iloc[-1] - values.iloc[0] if len(values) > 1 else np.nan
        )

    features.append(feat_row)

# Save even incomplete rows
features_df = pd.DataFrame(features)
features_df.to_csv("output/features_robust.csv", index=False)
print(f"[DONE] Saved {len(features_df)} feature rows (some may contain NaNs)")
