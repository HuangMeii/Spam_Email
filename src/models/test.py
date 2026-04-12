import numpy as np
import torch
import torch.nn as nn

from metric import best_threshold_roc, compute_metrics

# =========================
# EVALUATE FUNCTION
# =========================
def evaluate(model, loader, device, name="TEST"):
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # =========================
    # ROC threshold
    # =========================
    threshold, j = best_threshold_roc(all_labels, all_probs)

    metrics = compute_metrics(all_labels, all_probs, threshold)

    print(f"\n========== {name} ==========")
    print(f"Loss: {total_loss / len(loader):.4f}")
    print(f"Threshold (ROC/Youden): {threshold:.4f}")
    print(f"Youden J: {j:.4f}")

    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["cm"])

    print("\nClassification Report:")
    print(metrics["report"])

    return metrics, threshold