"""Loss functions for segmentation and classification."""

import torch.nn as nn
from monai.losses import DiceCELoss


def get_dice_ce_loss(num_classes: int = 4) -> nn.Module:
    """
    Dice + cross-entropy loss for multi-class segmentation.

    Expects raw logits (N, C, ...) and integer labels (N, ...); one-hot
    conversion and softmax are handled internally by MONAI's DiceCELoss.
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
    )


def get_bce_with_logits_loss() -> nn.Module:
    """Binary cross-entropy from logits, for binary tumor-presence classification."""
    return nn.BCEWithLogitsLoss()
