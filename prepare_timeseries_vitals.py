import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--chartevents", type=str, default="output/filtered_chartevents.csv.gz")
parser.add_argument("--sepsis_labels", type=str, default="output/sepsis_labels.csv")
parser.add_argument("--control_labels", type=str, default="output/control_labels.csv")
parser.add_argument("--output_dir", type=str, default="output/timeseries")
args = parser.parse_args()

# Define 7 vital ITEMIDs (MIMIC-III)
VITALS = {
    "HR": [211],
    "SBP": [51],
    "DBP": [8368],
    "MAP": [456],
    "RR": [618],
    "SpO2": [646],
    "Temp": [223761]
}
FEATURES = list(VITALS.keys())
NUM_FEATURES = len(FEATURES)
WINDOW_HOURS = 12

# Load data
chart = pd.read_csv(args.chartevents, parse_dates=["CHARTTIME"])
sepsis = pd.read_csv(args.sepsis_labels, parse_dates=["SEPSIS_ONSET"])
control = pd.read_csv(args.control_labels, parse_dates=["SEPSIS_ONSET"])

all_labels = pd.concat([
    sepsis.assign(label=1),
    control.assign(label=0)
], ignore_index=True)

# Setup output arrays
X = []
M = []
Y = []

print("[INFO] Generating time-series arrays...")

for _, row in tqdm(all_labels.iterrows(), total=len(all_labels)):
    icu_id = row["ICUSTAY_ID"]
    onset = row["SEPSIS_ONSET"]
    label = row["label"]
    
    start = onset - pd.Timedelta(hours=WINDOW_HOURS)
    df = chart[(chart["ICUSTAY_ID"] == icu_id) & (chart["CHARTTIME"] >= start) & (chart["CHARTTIME"] <= onset)].copy()
    df["hour"] = ((df["CHARTTIME"] - start).dt.total_seconds() // 3600).astype(int)

    # Initialize patient matrix
    patient_matrix = np.full((WINDOW_HOURS, NUM_FEATURES), np.nan)
    mask_matrix = np.zeros_like(patient_matrix)

    for j, feat in enumerate(FEATURES):
        for h in range(WINDOW_HOURS):
            vals = df[(df["ITEMID"].isin(VITALS[feat])) & (df["hour"] == h)]["VALUENUM"].dropna()
            if not vals.empty:
                patient_matrix[h, j] = vals.mean()
                mask_matrix[h, j] = 1

    X.append(patient_matrix)
    M.append(mask_matrix)
    Y.append(label)

# Save outputs
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "X.npy"), np.array(X))
np.save(os.path.join(args.output_dir, "M.npy"), np.array(M))
np.save(os.path.join(args.output_dir, "labels.npy"), np.array(Y))

print(f"[done] Saved time-series data to {args.output_dir}")
