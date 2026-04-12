import torch
import torch.nn as nn
import numpy as np


# =========================
# MODEL (copy từ train)
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# =========================
# LOAD MODEL
# =========================
device = torch.device("cpu")

model = LSTMClassifier().to(device)

checkpoint = torch.load("model.pth", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
threshold = checkpoint["threshold"]

model.eval()

print("✅ Model loaded!")
print("Threshold:", threshold)


# =========================
# PREDICT FUNCTION
# =========================
def predict(sample):
    x = torch.tensor(sample, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    pred = 1 if prob >= threshold else 0

    return prob, pred


# =========================
# TEST SAMPLE
# =========================
sample = np.random.rand(1, 20, 300)  # thay bằng input thật

prob, pred = predict(sample)

print("Prob:", prob)
print("Label:", "SPAM" if pred == 1 else "HAM")