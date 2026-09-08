"""Segmentation evaluation metrics for BraTS-style multi-class masks."""

import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

BRATS_CLASS_NAMES = ["background", "NCR_NET", "ED", "ET"]


def get_postprocessing_transform(num_classes: int = 4):
    """
    Return a batch-safe callable turning logits (N, C, spatial...) into
    one-hot predictions (N, C, spatial...) via channel-wise argmax.

    MONAI's AsDiscrete(argmax=True) argmaxes over dim 0, which is wrong for a
    batched tensor (it would collapse the batch, not the class channel), so
    channel selection is done explicitly here over dim 1.
    """

    def postprocess(logits: torch.Tensor) -> torch.Tensor:
        pred_labels = logits.argmax(dim=1, keepdim=True)  # (N, 1, spatial...)
        return one_hot(pred_labels, num_classes=num_classes)

    return postprocess


class _DiceMetricWithLabelTarget:
    """
    Wraps MONAI's DiceMetric (which requires one-hot y_pred and y) so callers
    can pass an integer-label target (N, spatial...) directly, without a
    channel dimension.
    """

    def __init__(self, num_classes: int, include_background: bool = True, reduction: str = "mean_batch"):
        self.num_classes = num_classes
        self._metric = DiceMetric(include_background=include_background, reduction=reduction)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == y_pred.ndim - 1:
            y = y.unsqueeze(1)
        y_onehot = one_hot(y, num_classes=self.num_classes)
        return self._metric(y_pred, y_onehot)

    def aggregate(self):
        return self._metric.aggregate()

    def reset(self):
        self._metric.reset()


def get_dice_metric(num_classes: int = 4, include_background: bool = True, reduction: str = "mean_batch"):
    """Dice metric accepting one-hot predictions and an integer-label target."""
    return _DiceMetricWithLabelTarget(num_classes, include_background, reduction)
