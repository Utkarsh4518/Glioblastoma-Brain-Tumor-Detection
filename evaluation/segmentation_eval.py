"""
Full segmentation evaluation: per-class Dice and HD95, plus MRI/mask overlays.

Works for both 3D (N, C, D, H, W) and 2D (N, C, H, W) batches. For 3D
volumes, overlays are drawn on the single axial slice with the largest
ground-truth tumor area.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot

from evaluation.metrics import BRATS_CLASS_NAMES, get_postprocessing_transform

# Distinct RGBA overlay colors per foreground class (index 1..); background
# (index 0) is never drawn.
_CLASS_COLORS = [
    (0.0, 0.0, 0.0, 0.0),  # background - transparent
    (1.0, 0.0, 0.0, 0.5),  # NCR/NET - red
    (0.0, 1.0, 0.0, 0.5),  # ED - green
    (0.0, 0.4, 1.0, 0.5),  # ET - blue
]


def _label_to_rgba(label2d: np.ndarray, num_classes: int) -> np.ndarray:
    """Map an integer label slice (H, W) to an RGBA overlay image."""
    h, w = label2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls in range(1, num_classes):
        color = _CLASS_COLORS[cls] if cls < len(_CLASS_COLORS) else (1.0, 1.0, 0.0, 0.5)
        rgba[label2d == cls] = color
    return rgba


def _pick_axial_slice(gt_labels: np.ndarray) -> int:
    """Return the slice index (first spatial axis) with the most tumor voxels."""
    tumor_per_slice = (gt_labels > 0).reshape(gt_labels.shape[0], -1).sum(axis=1)
    if tumor_per_slice.max() == 0:
        return gt_labels.shape[0] // 2
    return int(tumor_per_slice.argmax())


def _save_overlay(
    background2d: np.ndarray,
    gt2d: np.ndarray,
    pred2d: np.ndarray,
    num_classes: int,
    path: Path,
) -> None:
    """Save a 3-panel figure: MRI | MRI+ground truth | MRI+prediction."""
    bg = background2d.astype(np.float32)
    lo, hi = np.percentile(bg, 1), np.percentile(bg, 99)
    if hi > lo:
        bg = np.clip((bg - lo) / (hi - lo), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["MRI (FLAIR)", "Ground truth", "Prediction"]
    overlays = [None, gt2d, pred2d]
    for ax, title, overlay in zip(axes, titles, overlays):
        ax.imshow(bg, cmap="gray")
        if overlay is not None:
            ax.imshow(_label_to_rgba(overlay, num_classes))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _nanmean_per_class(metric_tensor: torch.Tensor) -> np.ndarray:
    """Aggregate a (num_classes,) metric, treating inf as nan and ignoring nans."""
    arr = metric_tensor.detach().cpu().numpy().astype(np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr


@torch.no_grad()
def evaluate_segmentation(
    model,
    loader,
    device,
    num_classes: int = 4,
    out_dir: str | Path = "outputs/segmentation_eval",
    num_overlays: int = 8,
) -> dict:
    """
    Run the val/test set, compute per-class Dice + HD95, save metrics.json and
    a handful of overlay PNGs. Returns the metrics dict.
    """
    model.eval().to(device)
    out_dir = Path(out_dir)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    post = get_postprocessing_transform(num_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean_batch"
    )

    saved = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device)  # (N, 1, *spatial)
        logits = model(image)
        pred_onehot = post(logits)  # (N, C, *spatial)
        target_onehot = one_hot(target, num_classes=num_classes)

        dice_metric(pred_onehot, target_onehot)
        hd95_metric(pred_onehot, target_onehot)

        if saved < num_overlays:
            pred_labels = pred_onehot.argmax(dim=1).cpu().numpy()  # (N, *spatial)
            gt_labels = target[:, 0].cpu().numpy()  # (N, *spatial)
            images = image.cpu().numpy()  # (N, C, *spatial)
            flair_idx = min(3, images.shape[1] - 1)

            for n in range(images.shape[0]):
                if saved >= num_overlays:
                    break
                gt = gt_labels[n]
                pred = pred_labels[n]
                if gt.ndim == 3:  # 3D volume -> pick best axial slice
                    s = _pick_axial_slice(gt)
                    bg = images[n, flair_idx, s]
                    gt, pred = gt[s], pred[s]
                else:  # already 2D
                    bg = images[n, flair_idx]
                _save_overlay(bg, gt, pred, num_classes, overlay_dir / f"seg_overlay_{saved:03d}.png")
                saved += 1

    dice_pc = _nanmean_per_class(dice_metric.aggregate())
    hd95_pc = _nanmean_per_class(hd95_metric.aggregate())

    class_names = BRATS_CLASS_NAMES if num_classes == len(BRATS_CLASS_NAMES) else [
        f"class_{i}" for i in range(num_classes)
    ]
    fg = slice(1, num_classes)  # foreground = exclude background for the headline mean
    metrics = {
        "dice_per_class": {n: _to_py(v) for n, v in zip(class_names, dice_pc)},
        "hd95_per_class": {n: _to_py(v) for n, v in zip(class_names, hd95_pc)},
        "mean_dice_foreground": _to_py(np.nanmean(dice_pc[fg])),
        "mean_hd95_foreground": _to_py(np.nanmean(hd95_pc[fg])),
        "mean_dice_all": _to_py(np.nanmean(dice_pc)),
        "num_overlays_saved": saved,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _to_py(value) -> float | None:
    """JSON-safe float: NaN/inf -> None."""
    v = float(value)
    return v if np.isfinite(v) else None
