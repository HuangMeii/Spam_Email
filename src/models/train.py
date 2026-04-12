import torch
import torch.nn as nn


# =========================
# LOSS (GLOBAL)
# =========================
criterion = nn.BCEWithLogitsLoss()


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, train_loader, device, optimizer):
    model.train()

    total_loss = 0.0
    num_samples = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_samples += X_batch.size(0)

    avg_loss = total_loss / len(train_loader)

    return avg_loss, num_samples