#  AI-Powered Predictive Interventions in ICU (Digital Twin of the Patient)

This project is part of a broader initiative to develop an AI-driven **Digital Twin for ICU patients** — a real-time, patient-specific simulation framework that can predict clinical deterioration and guide proactive interventions.

As an initial milestone, this implementation focuses on **predicting sepsis onset** using retrospective ICU data from the MIMIC-III dataset. We extract features from vital signs and lab results in the 12 hours preceding sepsis (or matched control) onset and train a machine learning model (XGBoost) to estimate risk in advance.

This pipeline demonstrates core components of the Digital Twin architecture, including:
- Temporal event alignment (relative to sepsis onset)
- Multimodal feature fusion (vitals + labs)
- Missing data handling
- Predictive modeling and explainability

Future phases will extend to additional deterioration types (e.g., shock, respiratory failure) and incorporate real-time streaming, reinforcement learning, and simulation for treatment policy testing.

```bash
+------------------+      +----------------+     +------------------+     +---------------------+
|  Raw MIMIC-III   | -->  |  Label Engine  | --> | Feature Extractor | --> |  ML Model (XGBoost) |
| (EHR, Labs, Vitals)     | (Sepsis/Control)|     | (Vitals + Labs)   |     |  + SHAP Explainability|
+------------------+      +----------------+     +------------------+     +---------------------+
        |                           |                      |                         |
        |                           |                      |                         |
        |                  12h window targeting     Time-aligned features      Risk score output
        |                  before sepsis/control         with missingness       + global/local insights
        V
     Future Modules:
  +---------------------+
  | Digital Twin Core   |
  | (Real-time data, RL |
  | treatment planning) |
  +---------------------+

```

Roadmap (Digital Twin Buildout)

```
Phase 1: Proof of Concept (Current Project)
 - Label ICU stays with Sepsis-3 definition.
 - Extract vitals and lab features in a 12-hour window.
 - Build baseline XGBoost model.
 - Apply SHAP for explainability.

Phase 2: Time-Series Modeling
 - Convert static features to hourly time-series format.
 - Train GRU-D or LSTM models to capture patient dynamics.
 - Handle missing data natively within sequence models.

Phase 3: Real-Time Digital Twin Engine
 - Stream data into digital twin in near real-time.
 - Predict deterioration events continuously (sepsis, shock, etc.).
 - Suggest proactive interventions (e.g., fluid bolus, antibiotics).

Phase 4: Clinical Integration & Simulation
 - Simulate outcomes for different treatment policies.
 - Integrate with EHR dashboards for clinician use.
 - Evaluate impact in retrospective and prospective studies.

```
---

## There is a single ipynb file that has all the necessary code blocks in a sequence. For individual project files used as process of learning and implementation are as below. download the MIMIC-III data files before proceeding


## Pipeline Overview

### Step 1: Generate Labels
Create sepsis and control labels with estimated onset times.
```bash
python sample_controls.py                         # Generate control_labels.csv
python extract_labels_csv.py                      # Generate sepsis_labels.csv (based on infection + SOFA)
```

### Step 2: Filter CHARTEVENTS
```bash
Extract only relevant vitals from the full CHARTEVENTS dataset for faster processing.
python preprocess_chartevents_filter.py           # Outputs filtered_chartevents.csv.gz
```

### Step 3: Extract Vitals Features
```bash
Run separately for sepsis and control patients.
python extract_vitals_wrapper.py --label_file output/sepsis_labels.csv --output_file output/features_robust.csv
python extract_vitals_wrapper.py --label_file output/control_labels.csv --output_file output/control_features_robust.csv
```

### Step 4: Extract Lab Features
```bash
Run separately for sepsis and control patients.

python extract_labs_wrapper.py --label_file output/sepsis_labels.csv --output_file output/lab_features_sepsis.csv
python extract_labs_wrapper.py --label_file output/control_labels.csv --output_file output/lab_features_controls.csv
```
### Step 5: Merge Vitals + Labs + Labels
```bash
Combine all features and impute missing values.

python merge_vitals_labs.py                       # Outputs merged_dataset_with_labs.csv
```
### Model Training
```bash
Train an XGBoost model on the merged dataset:

python train_xgboost.py                           # AUROC, AUPRC, classification report
```

### Explainability (SHAP)
```bash
Generate global feature importance and optional per-patient explanations:


python shap_analysis.py                           # Outputs SHAP plots in /output
```
## Output Files
```
File	Purpose
output/sepsis_labels.csv	Sepsis patient list with onset
output/control_labels.csv	Non-sepsis patient controls
output/features_robust.csv	Vitals features (sepsis)
output/control_features_robust.csv	Vitals features (controls)
output/lab_features_*.csv	Lab features
output/merged_dataset_with_labs.csv	Final training data
output/shap_summary.png	SHAP global importance plot
```


## Requirements
```
Install required packages:
pip install -r requirements.txt
```
## Notes

Dataset used: MIMIC-III (local CSVs)

ICU stays were labeled using Sepsis-3 definitions

12-hour windows used for prediction

Baseline model: XGBoost

Optional: Extend to GRU-D or LSTM models using time-series format
