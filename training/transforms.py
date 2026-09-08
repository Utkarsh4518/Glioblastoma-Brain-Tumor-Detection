"""
Training transforms built from config.

build_transforms(cfg, split) returns a MONAI Compose for augmentation.
2D (h5) and 3D (nifti) pipelines are dimension-consistent.
"""

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    ResizeWithPadOrCropd,
)

from data.transforms import get_train_transforms_2d


def build_transforms(cfg, split: str = "train"):
    """
    Build training or validation transforms from config.

    Args:
        cfg: Config with data.mode and data.roi_size (for nifti).
        split: "train" (augmentations) or "val" (no augmentations; returns None).

    Returns:
        Compose for train, or None for val (no transform).
    """
    if split != "train":
        return None

    data_cfg = cfg.get("data", {})
    mode = data_cfg.get("mode", "nifti")
    keys = ["image", "mask"]

    if mode == "h5":
        # Dataset already yields channel-first image (C,H,W) and mask (1,H,W).
        return get_train_transforms_2d(keys=keys, label_key="mask")

    if mode == "nifti":
        # 3D-safe: flip on all spatial axes, rotate in axial plane. RandRotate90d
        # swaps H/W on non-cubic patches, so a final pad-or-crop guarantees every
        # sample comes out exactly roi_size (otherwise batches fail to collate).
        roi_size = tuple(data_cfg.get("roi_size", [128, 128, 128]))
        return Compose([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
            ResizeWithPadOrCropd(keys=keys, spatial_size=roi_size),
        ])

    return None
