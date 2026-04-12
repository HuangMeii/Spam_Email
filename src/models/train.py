import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve

from train import train_one_epoch
from evaluate import evaluate_loss
from test import evaluate_test

import torch.nn as nn

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
# LOAD DATA (Drive / local)
# =========================
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# =========================
# SPLIT DATA
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# =========================
# DATALOADER
# =========================
train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=128)
test_loader = DataLoader(EmailDataset(X_test, y_test), batch_size=128)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# MODEL INIT
# =========================
model = LSTMClassifier().to(device)

# =========================
# TRAIN LOOP
# =========================
EPOCHS = 5

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    train_loss, train_samples = train_one_epoch(model, train_loader, device)

    # -------------------------
    # VALIDATION
    # -------------------------
    val_loss, val_probs, val_labels, val_samples = evaluate_loss(
        model, val_loader, device
    )

    # -------------------------
    # ROC + Youden threshold
    # -------------------------
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    preds = (val_probs >= best_threshold).astype(int)

    acc = accuracy_score(val_labels, preds)
    f1 = f1_score(val_labels, preds)

    # -------------------------
    # PRINT FULL INFO
    # -------------------------
    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")

    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val F1: {f1:.4f}")
    print(f"Best Threshold (ROC): {best_threshold:.4f}")

    print("-" * 60)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# =========================
# TEST FINAL
# =========================
evaluate_test(model, test_loader, device)