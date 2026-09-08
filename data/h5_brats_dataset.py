"""
BraTS 2020 2D slice dataset from Kaggle-style HDF5 files.

Expects one .h5 file per slice (default pattern "volume_*.h5"), each with:
  - an image dataset (H, W, 4) or (4, H, W): stacked T1, T1ce, T2, FLAIR
  - a mask dataset, either:
      (H, W, 3) one-hot without a background channel (NCR/NET, ED, ET) --
      the common encoding for public Kaggle BraTS2020 slice mirrors, where
      background is wherever all three channels are 0, or
      (H, W) integer label map already in 0..3.

image_key/mask_key default to "image"/"mask"; override via config if a
given mirror uses different dataset names.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5BraTSDataset(Dataset):
    """2D BraTS slice dataset backed by per-slice HDF5 files."""

    def __init__(
        self,
        data_root: Path | str,
        pattern: str = "volume_*.h5",
        image_key: Optional[str] = None,
        mask_key: Optional[str] = None,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.image_key = image_key or "image"
        self.mask_key = mask_key or "mask"
        self.transform = transform

        self.files = sorted(self.data_root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "r") as f:
            image = np.asarray(f[self.image_key])
            mask = np.asarray(f[self.mask_key])

        if image.ndim == 3 and image.shape[-1] <= 8 and image.shape[0] > 8:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = image.astype(np.float32)

        mean = image.mean(axis=(-2, -1), keepdims=True)
        std = image.std(axis=(-2, -1), keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        image = (image - mean) / std

        if mask.ndim == 3:
            channel_axis = 0 if mask.shape[0] <= 8 else -1
            mask_chw = np.moveaxis(mask, channel_axis, 0)
            label = np.zeros(mask_chw.shape[1:], dtype=np.int64)
            has_fg = mask_chw.sum(axis=0) > 0
            label[has_fg] = np.argmax(mask_chw, axis=0)[has_fg] + 1
        else:
            label = mask.astype(np.int64)

        return image, label

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(self.files[idx])
        image_t = torch.from_numpy(image)
        # Channel-first mask (1, H, W) to match the dataset-wide invariant
        # (see NiftiBraTSDataset): image (C, spatial), mask (1, spatial).
        mask_t = torch.from_numpy(mask).long().unsqueeze(0)
        if self.transform is not None:
            out = self.transform({"image": image_t, "mask": mask_t})
            image_t, mask_t = out["image"], out["mask"]
        return {"image": image_t, "mask": mask_t}
