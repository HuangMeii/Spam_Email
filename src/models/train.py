import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

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

# reduce dataset for CPU
n = len(X) // 5
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
    batch_size=128,
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
        return logits

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = LSTMClassifier().to(device)

# =========================
# LOSS + OPT
# =========================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN FUNCTION
# =========================
def train_epoch():
    model.train()
    total_loss = 0

    num_samples = 0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        batch_size = X_batch.size(0)
        num_samples += batch_size
        num_batches += 1

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"   ➜ Train samples: {num_samples} | Batches: {num_batches}")

    return total_loss / len(train_loader)

# =========================
# FIND BEST THRESHOLD
# =========================
def find_best_threshold(probs, labels):
    best_thresh = 0.5
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1

# =========================
# EVALUATION
# =========================
def evaluate():
    model.eval()

    total_loss = 0
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())

    val_loss = total_loss / len(val_loader)

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    best_thresh, best_f1 = find_best_threshold(probs_all, labels_all)

    preds = (probs_all >= best_thresh).astype(int)
    acc = accuracy_score(labels_all, preds)

    return val_loss, acc, best_f1, best_thresh

# =========================
# TRAIN LOOP
# =========================
EPOCHS = 5

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss, acc, f1, thresh = evaluate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Acc: {acc:.4f} | F1: {f1:.4f} | Threshold: {thresh:.2f}")
    print("-" * 60)

# =========================
# SAVE MODEL
# =========================
SAVE_PATH = DATA_PATH / "lstm_model.pth"
torch.save(model.state_dict(), SAVE_PATH)

print("Saved model to:", SAVE_PATH)

# =========================
# LOSS CURVE
# =========================
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()