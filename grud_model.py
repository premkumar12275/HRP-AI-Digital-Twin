import torch
import torch.nn as nn

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable decay parameters
        self.gamma_x = nn.Parameter(torch.ones(input_size))     # for missing data
        self.gamma_h = nn.Parameter(torch.ones(hidden_size))    # for hidden state decay

        # GRU-D specific input transformation
        self.z = nn.Linear(input_size * 3, hidden_size)  # [X_t, X_mean, M_t]
        self.r = nn.Linear(input_size * 3, hidden_size)
        self.h_tilde = nn.Linear(input_size * 3, hidden_size)
        self.h_out = nn.Linear(hidden_size, output_size)

        # Non-learnable mean of inputs (use dataset mean)
        self.input_means = None  # To be filled externally

    def forward(self, X, M):
        B, T, D = X.shape
        H = self.hidden_size
        device = X.device

        if self.input_means is None:
            raise ValueError("input_means not set. Use model.input_means = torch.tensor(mean_vals)")

        h = torch.zeros(B, H).to(device)
        X_mean = self.input_means.unsqueeze(0).unsqueeze(0).expand(B, T, D).to(device)

        outputs = []

        for t in range(T):
            x_t = X[:, t]
            m_t = M[:, t]

            # Impute missing with mean and apply decay
            x_hat = m_t * x_t + (1 - m_t) * X_mean[:, t]
            x_gamma = torch.exp(-self.gamma_x * (1 - m_t))
            x_tilde = x_gamma * x_hat + (1 - x_gamma) * X_mean[:, t]

            inputs = torch.cat([x_tilde, X_mean[:, t], m_t], dim=1)

            z_t = torch.sigmoid(self.z(inputs))
            r_t = torch.sigmoid(self.r(inputs))
            h_tilde = torch.tanh(self.h_tilde(inputs))

            h = (1 - z_t) * h + z_t * h_tilde
            outputs.append(h.unsqueeze(1))

        h_seq = torch.cat(outputs, dim=1)
        out = self.h_out(h_seq[:, -1])        # shape: (B, 1)
        out = torch.sigmoid(out)              # shape: (B, 1)
        return out.squeeze(dim=1)             # shape: (B,)

