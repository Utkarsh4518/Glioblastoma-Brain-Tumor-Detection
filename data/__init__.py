# BraTS 2020 data loading and preprocessing

from pathlib import Path

import torch
from torch.utils.data import random_split

from data.h5_brats_dataset import H5BraTSDataset
from data.nifti_brats_dataset import NiftiBraTSDataset


def build_dataset(cfg, split: str = "train"):
    """
    Build segmentation dataset for the given split.

    Args:
        cfg: Hydra/OmegaConf config with data, paths, seed.
        split: "train" or "val".

    Returns:
        Subset of the full dataset for the requested split.

    Config:
        data.mode: "h5" (2D slice HDF5) or "nifti" (3D NIfTI volumes).
        data.path: used when mode is "h5".
        paths.data_root: used when mode is "nifti", or fallback for h5 path.
        data.train_val_split: fraction for train (default 0.85).
        data.roi_size: [D, H, W] for nifti patch size.
        data.pattern, data.image_key, data.mask_key: H5-specific.
    """
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    mode = data_cfg.get("mode", "nifti")
    train_val_split = data_cfg.get("train_val_split", 0.85)
    seed = cfg.get("seed", 42)

    if mode == "h5":
        data_root = Path(
            str(data_cfg.get("path", paths_cfg.get("data_root", "./data/h5_brats"))))
        full = H5BraTSDataset(
            data_root=data_root,
            pattern=data_cfg.get("pattern", "volume_*.h5"),
            image_key=data_cfg.get("image_key") or None,
            mask_key=data_cfg.get("mask_key") or None,
            transform=None,
        )
    elif mode == "nifti":
        data_root = Path(paths_cfg.get("data_root", "./data/brats2020"))
        roi_size = tuple(data_cfg.get("roi_size", [128, 128, 128]))
        full = NiftiBraTSDataset(root=data_root, patch_size=roi_size)
    else:
        raise ValueError(f"Unknown data.mode: {mode!r}")

    n = len(full)
    n_train = int(n * train_val_split)
    n_val = n - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    if split == "train":
        return train_ds
    if split == "val":
        return val_ds
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")
