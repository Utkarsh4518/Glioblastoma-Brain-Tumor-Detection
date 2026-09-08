"""Reusable 2D augmentation pipeline (MONAI dict-transforms)."""

from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandZoomd,
)


def get_train_transforms_2d(keys: list[str] = ["image", "mask"], label_key: str = "mask") -> Compose:
    """
    2D training augmentations: flips, rotation, zoom (nearest for the label
    channel, bilinear for the image), mild noise/contrast jitter on the image.
    """
    image_keys = [k for k in keys if k != label_key]
    interp_modes = ["bilinear" if k != label_key else "nearest" for k in keys]

    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
        RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=interp_modes),
        RandGaussianNoised(keys=image_keys, prob=0.2, std=0.05),
        RandAdjustContrastd(keys=image_keys, prob=0.2, gamma=(0.8, 1.2)),
    ])
