import pandas as pd
from tqdm import tqdm

# Paths to CSVs
ICUSTAYS_PATH = "input/ICUSTAYS.csv.gz"
PRESCRIPTIONS_PATH = "input/PRESCRIPTIONS.csv.gz"
MICROBIO_PATH = "input/MICROBIOLOGYEVENTS.csv.gz"

# Load ICU stay data
icustays = pd.read_csv(ICUSTAYS_PATH, usecols=["ICUSTAY_ID", "HADM_ID", "INTIME"])
print(f"[INFO] Loaded {len(icustays)} ICU stays")

# Antibiotics (proxy: generic names with 'cillin', 'cef', 'micin', etc.)
prescriptions = pd.read_csv(PRESCRIPTIONS_PATH, usecols=["HADM_ID", "STARTDATE", "DRUG"])
prescriptions = prescriptions.dropna(subset=["STARTDATE", "DRUG"])
prescriptions = prescriptions[prescriptions["DRUG"].str.lower().str.contains("cillin|cef|micin|penem|floxacin|mycin", na=False)]

# Microbiology events
microbio = pd.read_csv(MICROBIO_PATH, usecols=["HADM_ID", "CHARTDATE"])
microbio = microbio.dropna(subset=["CHARTDATE"])

# Merge to find suspected infection window
merged = pd.merge(prescriptions, microbio, on="HADM_ID", how="inner")
merged["SUSPECTED_TIME"] = pd.to_datetime(merged[["STARTDATE", "CHARTDATE"]].max(axis=1))

# Map to ICU stay
merged = pd.merge(merged, icustays, on="HADM_ID", how="inner")
merged = merged[merged["SUSPECTED_TIME"] > merged["INTIME"]]
suspected_df = merged.groupby("ICUSTAY_ID").SUSPECTED_TIME.min().reset_index()

# Simulated SOFA threshold logic (you can refine later)
# Placeholder: label everyone with suspicion as "sepsis"
suspected_df["SEPSIS_ONSET"] = suspected_df["SUSPECTED_TIME"]

# Save label file
suspected_df[["ICUSTAY_ID", "SEPSIS_ONSET"]].to_csv("output/sepsis_labels.csv", index=False)
print(f"[INFO] Labeled {len(suspected_df)} sepsis cases")
