import numpy as np
import torch
import torch.nn as nn

from src.models.metric import best_threshold_roc, compute_metrics


# =========================
# TEST / EVALUATION
# =========================
def evaluate(model, loader, device, name="TEST"):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            num_samples += X_batch.size(0)

            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # =========================
    # CONVERT
    # =========================
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # =========================
    # ROC + Youden threshold
    # =========================
    threshold, j_score = best_threshold_roc(all_labels, all_probs)

    # =========================
    # METRICS
    # =========================
    metrics = compute_metrics(all_labels, all_probs, threshold)

    # =========================
    # PRINT RESULTS
    # =========================
    print(f"\n========== {name} RESULTS ==========")

    print(f"Samples: {num_samples}")
    print(f"Loss: {total_loss / len(loader):.4f}")

    print(f"Best Threshold (ROC/Youden): {threshold:.4f}")
    print(f"Youden J Score: {j_score:.4f}")

    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["cm"])

    print("\nClassification Report:")
    print(metrics["report"])

    return metrics, threshold