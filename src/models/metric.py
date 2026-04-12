import numpy as np
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# =========================
# ROC - Youden J
# =========================
def best_threshold_roc(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    idx = np.argmax(j)

    return thresholds[idx], j[idx]


# =========================
# FULL EVAL
# =========================
def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "cm": confusion_matrix(labels, preds),
        "report": classification_report(labels, preds)
    }