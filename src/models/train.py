import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

criterion = nn.BCEWithLogitsLoss()


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, train_loader, val_loader, device, optimizer):
    model.train()

    train_loss = 0.0
    num_samples = 0

    # =====================
    # TRAIN
    # =====================
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits.squeeze(), y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_samples += X_batch.size(0)

    avg_train_loss = train_loss / len(train_loader)

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    val_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)

            probs = torch.sigmoid(logits)

            val_loss += loss.item()
            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

    avg_val_loss = val_loss / len(val_loader)

    # =====================
    # FIND BEST THRESHOLD (Youden's J)
    # =====================
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_threshold = thresholds[j_scores.argmax()]

    # =====================
    # PRINT
    # =====================
    # print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Threshold: {best_threshold:.4f}")

    return avg_train_loss, avg_val_loss, best_threshold, num_samples