import numpy as np
import torch

# =========================
# MODEL
# =========================
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.models.test import evaluate
from src.models.train import train


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# =========================
# DATASET
# =========================
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# LOAD DATA
# =========================
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# =========================
# SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# =========================
# LOADERS
# =========================
train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=128, shuffle=True)
test_loader = DataLoader(EmailDataset(X_test, y_test), batch_size=128)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# MODEL
# =========================
model = LSTMClassifier().to(device)

# =========================
# TRAIN
# =========================
EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train(model, train_loader, device)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {loss:.4f}")

# =========================
# TEST
# =========================
evaluate(model, test_loader, device, name="TEST")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "model.pth")
print("\nModel saved.")
