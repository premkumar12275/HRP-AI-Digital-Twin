import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label_file", type=str, required=True, help="Path to sepsis or control label file")
parser.add_argument("--output_file", type=str, required=True, help="Path to save lab features")
args = parser.parse_args()

# Load data
labels = pd.read_csv(args.label_file)
labels["SEPSIS_ONSET"] = pd.to_datetime(labels["SEPSIS_ONSET"])
# Load LABEVENTS and ICU stay metadata
labs = pd.read_csv("LABEVENTS.csv.gz", usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"], parse_dates=["CHARTTIME"])
icustays = pd.read_csv("ICUSTAYS.csv.gz", usecols=["ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME"], parse_dates=["INTIME", "OUTTIME"])

# Merge labs with icustays on HADM_ID
labs = pd.merge(labs, icustays, on="HADM_ID", how="inner")

# Filter lab rows that fall within ICU stay window
labs = labs[(labs["CHARTTIME"] >= labs["INTIME"]) & (labs["CHARTTIME"] <= labs["OUTTIME"])]


# Define lab item IDs (MIMIC-III)
LAB_ITEMS = {
    "Lactate": [50813],
    "WBC": [51300],
    "Creatinine": [50912],
    "Bilirubin": [50885],
    "Platelets": [51265]
}

features = []

print(f"[INFO] Extracting lab features from: {args.label_file}")

for _, row in tqdm(labels.iterrows(), total=len(labels)):
    icu_id = row["ICUSTAY_ID"]
    onset = row["SEPSIS_ONSET"]
    start = onset - pd.Timedelta(hours=12)

    window = labs[
        (labs["ICUSTAY_ID"] == icu_id) &
        (labs["CHARTTIME"] >= start) &
        (labs["CHARTTIME"] <= onset)
    ]

    row_feat = {"ICUSTAY_ID": icu_id}

    for name, ids in LAB_ITEMS.items():
        subset = window[window["ITEMID"].isin(ids)]
        values = subset["VALUENUM"].dropna()

        row_feat[f"{name}_mean"] = values.mean() if not values.empty else np.nan
        row_feat[f"{name}_std"] = values.std() if not values.empty else np.nan
        row_feat[f"{name}_delta"] = (
            values.iloc[-1] - values.iloc[0] if len(values) > 1 else np.nan
        )

    features.append(row_feat)

df = pd.DataFrame(features)
df.to_csv(args.output_file, index=False)
print(f"[DONE] Saved lab features to: {args.output_file}")


# Usage:
# python extract_labs_wrapper.py --label_file output/sepsis_labels.csv --output_file output/lab_features.csv
# or for controls:
# python extract_labs_wrapper.py --label_file output/control_labels.csv --output_file output/lab_features_controls.csv