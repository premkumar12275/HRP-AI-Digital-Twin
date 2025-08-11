import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--label_file", type=str, required=True, help="Path to sepsis/control label CSV")
parser.add_argument("--output_file", type=str, required=True, help="Output path for vitals features")
args = parser.parse_args()

# Load input
labels = pd.read_csv(args.label_file)
labels["SEPSIS_ONSET"] = pd.to_datetime(labels["SEPSIS_ONSET"])
chart = pd.read_csv("output/filtered_chartevents.csv.gz", parse_dates=["CHARTTIME"])

# Define ITEMIDs to extract (MIMIC-III)
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

print(f"[INFO] Extracting vitals from: {args.label_file}")

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

# Save features
df = pd.DataFrame(features)
df.to_csv(args.output_file, index=False)
print(f"[DONE] Saved vitals to: {args.output_file}")

# Usage:
# python extract_vitals_wrapper.py --label_file output/sepsis_labels.csv --output_file output/vitals_features.csv
# or for controls:
# python extract_vitals_wrapper.py --label_file output/control_labels.csv --output_file output/vitals_features_controls.csv