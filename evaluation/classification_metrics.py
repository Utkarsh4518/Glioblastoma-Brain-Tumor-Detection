"""Binary classification evaluation metrics for tumor-presence prediction."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.

    Args:
        logits: (N, 1) or (N,) raw model outputs (pre-sigmoid).
        labels: (N,) integer 0/1 ground truth.
    """
    logits = logits.squeeze(-1) if logits.ndim > 1 else logits
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = labels.cpu().numpy().astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]),
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return metrics
