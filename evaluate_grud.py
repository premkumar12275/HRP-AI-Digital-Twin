import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from grud_model import GRUD
import os

# Load data
X = torch.tensor(np.load("output/timeseries/X.npy"), dtype=torch.float32)
X = torch.nan_to_num(X, nan=0.0)
M = torch.tensor(np.load("output/timeseries/M.npy"), dtype=torch.float32)
y = torch.tensor(np.load("output/timeseries/labels.npy"), dtype=torch.float32)

# Train/val split (same as training)
from sklearn.model_selection import train_test_split
X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
    X, M, y, test_size=0.2, stratify=y, random_state=42
)

# Load model
input_size = X.shape[2]
model = GRUD(input_size=input_size, hidden_size=64)
model.load_state_dict(torch.load("models/grud_model.pt"))
model.eval()

# Set input means from training set
mean_vals = torch.nanmean(X_train, dim=(0,1))
model.input_means = mean_vals

# Predict
with torch.no_grad():
    y_prob = model(X_val, M_val).numpy()
    y_true = y_val.numpy()

# AUROC + AUPRC
auroc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
print(f"[done] AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

# Plot ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GRU-D ROC Curve")
plt.legend()
plt.savefig("output/grud_roc.png")
plt.close()

# Plot PR curve
prec, rec, _ = precision_recall_curve(y_true, y_prob)
plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("GRU-D PR Curve")
plt.legend()
plt.savefig("output/grud_pr.png")
plt.close()

# Save predictions
np.savetxt("output/grud_val_predictions.csv", np.vstack([y_true, y_prob]).T,
           delimiter=",", header="true_label,pred_prob", comments='')
