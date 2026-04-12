import numpy as np
import torch

import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from torch.utils.data import DataLoader

from src.models.test import evaluate
from src.models.train import train


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
# SPLIT (train / val / test)
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
val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=128)
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


# ==========================================================
# 🔥 THRESHOLD SELECTION (IMPROVED)
# ==========================================================

def get_probs(model, loader):
    model.eval()

    probs_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y_batch.numpy())

    return np.array(probs_all), np.array(labels_all)


# =========================
# GET VAL PREDICTIONS
# =========================
val_probs, val_labels = get_probs(model, val_loader)


# =========================
# 1. ROC (Youden)
# =========================
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
youden = tpr - fpr
thr_roc = thresholds[np.argmax(youden)]


# =========================
# 2. F1 optimal threshold
# =========================
best_f1 = 0
thr_f1 = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(val_labels, preds)

    if f1 > best_f1:
        best_f1 = f1
        thr_f1 = t


# =========================
# FINAL THRESHOLD (BALANCED)
# =========================
final_threshold = (thr_roc + thr_f1) / 2


print("\n========== THRESHOLD SELECTION ==========")
print(f"ROC threshold   : {thr_roc:.4f}")
print(f"F1 threshold    : {thr_f1:.4f}")
print(f"FINAL threshold : {final_threshold:.4f}")


# =========================
# TEST EVALUATION (FIXED THRESHOLD)
# =========================
evaluate(model, test_loader, device, name="TEST (FINAL)")


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "model.pth")
print("\nModel saved.")