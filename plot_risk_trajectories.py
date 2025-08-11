import torch
import numpy as np
import matplotlib.pyplot as plt
from grud_model import GRUD
import random

# Load data
X = torch.tensor(np.load("output/timeseries/X.npy"), dtype=torch.float32)
X = torch.nan_to_num(X, nan=0.0)
M = torch.tensor(np.load("output/timeseries/M.npy"), dtype=torch.float32)
y = torch.tensor(np.load("output/timeseries/labels.npy"), dtype=torch.float32)

# Load model
input_size = X.shape[2]
model = GRUD(input_size=input_size, hidden_size=64)
model.load_state_dict(torch.load("models/grud_model.pt"))
model.eval()

# Set input means
model.input_means = torch.nanmean(X, dim=(0, 1))

# Pick a few random sepsis + control patients
indices = {
    "sepsis": random.sample([i for i, l in enumerate(y) if l == 1], 3),
    "control": random.sample([i for i, l in enumerate(y) if l == 0], 3)
}

for label, idx_list in indices.items():
    for i in idx_list:
        xb = X[i:i+1]
        mb = M[i:i+1]

        h_t = []
        model.eval()
        with torch.no_grad():
            B, T, D = xb.shape
            h = torch.zeros(B, model.hidden_size)
            h_seq = []

            for t in range(T):
                x_t = xb[:, t]
                m_t = mb[:, t]

                x_hat = m_t * x_t + (1 - m_t) * model.input_means
                x_gamma = torch.exp(-model.gamma_x * (1 - m_t))
                x_tilde = x_gamma * x_hat + (1 - x_gamma) * model.input_means

                input_means = model.input_means.unsqueeze(0)  # shape: (1, D)
                inputs = torch.cat([x_tilde, input_means, m_t], dim=1)

                z_t = torch.sigmoid(model.z(inputs))
                r_t = torch.sigmoid(model.r(inputs))
                h_tilde = torch.tanh(model.h_tilde(inputs))

                h = (1 - z_t) * h + z_t * h_tilde
                h_seq.append(h @ model.h_out.weight.T + model.h_out.bias)

            risk = torch.sigmoid(torch.stack(h_seq).squeeze(dim=2).squeeze(dim=1)).numpy()

        plt.plot(range(1, len(risk)+1), risk, marker='o', label=f"{label.upper()} #{i}")
        plt.xlabel("Hours Before Onset")
        plt.ylabel("Sepsis Risk")
        plt.title("Risk Trajectory")
        plt.ylim(0, 1)

plt.legend()
plt.grid(True)
plt.savefig("output/grud_risk_trajectories.png")
plt.close()
