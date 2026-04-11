import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# =========================
# PATH
# =========================
PROJECT_PATH = "/content/drive/MyDrive"
DATA_PATH = Path("/content/drive/MyDrive/datasets")

sys.path.append(PROJECT_PATH)

# =========================
# LOAD DATA
# =========================
X = np.load(DATA_PATH / "X.npy")
y = np.load(DATA_PATH / "y.npy")

print("Full shape:", X.shape, y.shape)

# reduce dataset (CPU-friendly)
n = len(X) // 7
X = X[:n]
y = y[:n]

print("Reduced shape:", X.shape, y.shape)

# =========================
# TRAIN / VAL SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# DATASET
# =========================
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    EmailDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    EmailDataset(X_val, y_val),
    batch_size=128,
    shuffle=False
)

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
        logits = self.fc(h_n[-1]).squeeze()
        return logits  # no sigmoid here

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = LSTMClassifier().to(device)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN FUNCTION
# =========================
def train_epoch():
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate():
    model.eval()

    total_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all)

    return val_loss, acc, f1

# =========================
# TRAIN LOOP
# =========================
EPOCHS = 5

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss, acc, f1 = evaluate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    print("-" * 50)

# =========================
# SAVE MODEL
# =========================
SAVE_PATH = DATA_PATH / "lstm_model.pth"
torch.save(model.state_dict(), SAVE_PATH)

print("Saved model to:", SAVE_PATH)

# =========================
# PLOT LOSS CURVE
# =========================
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()