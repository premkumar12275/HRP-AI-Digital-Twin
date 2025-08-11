import pandas as pd
from tqdm import tqdm

# Input files
CHART_PATH = "input/CHARTEVENTS.csv.gz"
LABELS_PATH = "output/sepsis_labels.csv"

# Vital sign ITEMIDs — you can expand or adjust
ITEM_IDS = {
    "Heart Rate": [211],
    "Systolic BP": [51],
    "Diastolic BP": [8368],
    "Mean BP": [456],
    "Respiratory Rate": [618],
    "SpO2": [646],
    "Temperature": [223761]  # Celsius
}

# Load sepsis cases
labels = pd.read_csv(LABELS_PATH)
labels["SEPSIS_ONSET"] = pd.to_datetime(labels["SEPSIS_ONSET"])
features = []

for chunk in tqdm(pd.read_csv(CHART_PATH, usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"], chunksize=500000)):
    chunk = chunk[chunk["ICUSTAY_ID"].isin(labels["ICUSTAY_ID"])]
    chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"])

    for _, row in labels.iterrows():
        icustay = row["ICUSTAY_ID"]
        onset = row["SEPSIS_ONSET"]
        window_start = onset - pd.Timedelta(hours=12)

        window = chunk[(chunk["ICUSTAY_ID"] == icustay) & (chunk["CHARTTIME"] >= window_start) & (chunk["CHARTTIME"] <= onset)]
        if window.empty:
            continue

        row_feat = {"ICUSTAY_ID": icustay}
        for vital_name, ids in ITEM_IDS.items():
            vs = window[window["ITEMID"].isin(ids)]
            if vs.empty:
                row_feat[f"{vital_name}_mean"] = None
                row_feat[f"{vital_name}_delta"] = None
            else:
                row_feat[f"{vital_name}_mean"] = vs["VALUENUM"].mean()
                row_feat[f"{vital_name}_delta"] = vs["VALUENUM"].iloc[-1] - vs["VALUENUM"].iloc[0]
        features.append(row_feat)

# Save feature matrix
pd.DataFrame(features).dropna().to_csv("output/features.csv", index=False)
print(f"[INFO] Saved features for {len(features)} ICU stays")
