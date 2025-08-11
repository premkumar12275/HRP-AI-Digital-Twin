import pandas as pd
import numpy as np
from tqdm import tqdm

# Load ICU stays and sepsis cases
icustays = pd.read_csv("input/ICUSTAYS.csv.gz", usecols=["ICUSTAY_ID", "INTIME", "OUTTIME"])
labels = pd.read_csv("output/sepsis_labels.csv")

# Get non-sepsis ICU stays
sepsis_ids = set(labels["ICUSTAY_ID"])
non_sepsis = icustays[~icustays["ICUSTAY_ID"].isin(sepsis_ids)].dropna()
print(f"[INFO] Found {len(non_sepsis)} non-sepsis ICU stays")

# Sample same number of controls (or as many as available)
n_controls = min(len(non_sepsis), len(labels))
controls = non_sepsis.sample(n=n_controls, replace=False, random_state=42).copy()

controls["INTIME"] = pd.to_datetime(controls["INTIME"])
controls["OUTTIME"] = pd.to_datetime(controls["OUTTIME"])

# Pick a random timestamp 12+ hours after INTIME and 12h before OUTTIME
def random_control_time(row):
    start = row["INTIME"] + pd.Timedelta(hours=12)
    end = row["OUTTIME"] - pd.Timedelta(hours=12)
    if start >= end:
        return np.nan
    return pd.to_datetime(start + (end - start) * np.random.rand())

controls["SEPSIS_ONSET"] = controls.apply(random_control_time, axis=1)
controls = controls.dropna(subset=["SEPSIS_ONSET"])

# Save as pseudo-sepsis labels (for reuse)
controls[["ICUSTAY_ID", "SEPSIS_ONSET"]].to_csv("output/control_labels.csv", index=False)
print(f"[DONE] Sampled {len(controls)} control patients → output/control_labels.csv")
