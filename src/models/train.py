import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =========================
# ADD PROJECT ROOT (nếu cần import src)
# =========================
PROJECT_PATH = "/content/drive/MyDrive"
sys.path.append(PROJECT_PATH)


# =========================
# PATH DATA
# =========================
DATA_PATH = Path("/content/drive/MyDrive/datasets")

# =========================
# LOAD DATA
# =========================
X_train = np.load(DATA_PATH / "X.npy")
y_train = np.load(DATA_PATH / "y.npy")

print("Full shape:", X_train.shape, y_train.shape)

n = len(X_train) // 7
X_train = X_train[:n]
y_train = y_train[:n]

print("Reduced shape:", X_train.shape, y_train.shape)



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


train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=128, shuffle=True)


# =========================
# LSTM MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# INIT MODEL
# =========================
model = LSTMClassifier().to(device)

criterion = nn.BCELoss()
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

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# TRAIN LOOP
# =========================
EPOCHS = 10

for epoch in range(EPOCHS):
    loss = train_epoch()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}")
    print("-" * 30)


# =========================
# SAVE MODEL
# =========================
SAVE_PATH = DATA_PATH / "lstm_model.pth"
torch.save(model.state_dict(), SAVE_PATH)

print("Saved model to:", SAVE_PATH)
