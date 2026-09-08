"""
Full classification evaluation: metrics plus confusion-matrix and ROC-curve
figures for the thesis results chapter.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from evaluation.classification_metrics import compute_classification_metrics


def _plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """Save a 2x2 confusion-matrix heatmap with count annotations."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["No tumor", "Tumor"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float, path: Path) -> None:
    """Save an ROC curve with the diagonal chance line."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="C0", label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_classification(
    model,
    loader,
    device,
    out_dir: str | Path = "outputs/classification_eval",
) -> dict:
    """
    Run the val/test set, compute metrics, and save metrics.json plus
    confusion_matrix.png and roc_curve.png. Returns the metrics dict.
    """
    model.eval().to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_logits, all_labels = [], []
    for batch in loader:
        image = batch["image"].to(device)
        labels = batch["label"]
        logits = model(image)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(logits, labels)

    cm = metrics["confusion_matrix"]
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    probs = torch.sigmoid(logits.squeeze(-1) if logits.ndim > 1 else logits).numpy()
    y_true = labels.numpy().astype(int)
    if len(np.unique(y_true)) > 1:
        _plot_roc_curve(y_true, probs, metrics["roc_auc"], out_dir / "roc_curve.png")

    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
        for k, v in metrics.items()
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    return metrics
