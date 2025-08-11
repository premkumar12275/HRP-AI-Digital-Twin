import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from grud_model import GRUD

# --- Load data ---
# --- Load data ---
X = np.load("output/timeseries/X.npy")          # (N, 12, 7)
M = np.load("output/timeseries/M.npy")
y = np.load("output/timeseries/labels.npy")

# Convert to torch and fix NaNs
X = torch.tensor(X, dtype=torch.float32)
X = torch.nan_to_num(X, nan=0.0)

M = torch.tensor(M, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# --- Train/val split ---
X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
    X, M, y, test_size=0.2, stratify=y, random_state=42
)

# --- Model, loss, optimizer ---
input_size = X.shape[2]
model = GRUD(input_size=input_size, hidden_size=64)
# Set input_means from training data only
train_np = X_train.numpy()
train_mean = np.nanmean(train_np, axis=(0, 1))
model.input_means = torch.tensor(train_mean, dtype=torch.float32)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
n_epochs = 20
batch_size = 64
device = torch.device("cpu")
model.to(device)

train_losses = []
val_aurocs = []

print(f"[INFO] Training on {device}...")

print("[DEBUG] Checking for NaNs in training data...")
print("X_train NaNs:", torch.isnan(X_train).sum().item())
print("M_train NaNs:", torch.isnan(M_train).sum().item())
print("y_train NaNs:", torch.isnan(y_train).sum().item())
print("input_means NaNs:", torch.isnan(model.input_means).sum().item())


for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb, mb, yb = X_train[idx].to(device), M_train[idx].to(device), y_train[idx].to(device)

        optimizer.zero_grad()
        out = model(xb, mb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val.to(device), M_val.to(device)).cpu().numpy()
        auroc = roc_auc_score(y_val.numpy(), y_val_pred)
        auprc = average_precision_score(y_val.numpy(), y_val_pred)
        val_aurocs.append(auroc)

    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

# --- Save model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/grud_model.pt")
print("[done] GRU-D model saved to models/grud_model.pt")

# --- Plot training loss ---
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("output/grud_loss_curve.png")
plt.close()
