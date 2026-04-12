from importlib.resources import path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.train import train_one_epoch
from src.models.test import evaluate


# =========================
# MODEL
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
DATA_PATH = "/content/drive/MyDrive/datasets"

X = np.load(f"{DATA_PATH}/X.npy")
y = np.load(f"{DATA_PATH}/y.npy")


# =========================
# SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)


# =========================
# LOADERS
# =========================
train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=128, shuffle=False)
test_loader = DataLoader(EmailDataset(X_test, y_test), batch_size=128, shuffle=False)


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# INIT
# =========================
model = LSTMClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# =========================
# TRAIN LOOP
# =========================
EPOCHS = 5

best_val_loss = float("inf")
best_threshold = 0.5

for epoch in range(EPOCHS):

    train_loss, val_loss, threshold, samples = train_one_epoch(
        model,
        train_loader,
        val_loader,
        device,
        optimizer
    )

    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
    print(f"Train Samples : {samples}")
    print(f"Train Loss    : {train_loss:.4f}")
    print(f"Val Loss      : {val_loss:.4f}")
    print(f"Threshold     : {threshold:.4f}")

    # save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_threshold = threshold
        torch.save(model.state_dict(), "best_model.pth")


# =========================
# TEST
# =========================
print("\nLoading best model...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

evaluate(
    model,
    test_loader,
    device,
    name="TEST",
    threshold=best_threshold
)


# =========================
# SAVE FINAL MODEL
# =========================
path = "/content/drive/MyDrive/datasets/model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "threshold": threshold,
    "val_loss": val_loss
}, path)

print(f"\n✅ Model saved successfully!")
print(f"📁 Path: {os.path.abspath(path)}")