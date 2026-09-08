"""BraTS classification dataset: per-slice or per-volume tumor presence label."""

from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset

from data.nifti_brats_dataset import NiftiBraTSDataset


class BraTS2020ClassificationDataset(Dataset):
    """
    Binary tumor-presence classification dataset derived from BraTS NIfTI volumes.

    mode="2d": one sample per axial slice with any modality signal present
    (label = 1 if that slice's mask has any tumor voxel).
    mode="3d": one sample per subject (label = 1 if the volume's mask has
    any tumor voxel).

    Volumes are loaded from disk lazily and cached one-at-a-time (last
    accessed subject), since 2D mode indexes many slices per subject.
    """

    def __init__(self, root: Path | str, mode: Literal["2d", "3d"] = "2d"):
        self.mode = mode
        self._volumes = NiftiBraTSDataset(root=root, patch_size=None)
        self._cache_vol_idx: Optional[int] = None
        self._cache_image = None
        self._cache_mask = None

        if mode == "3d":
            self._index: list = list(range(len(self._volumes)))
            return

        self._index: list[tuple[int, int]] = []
        for vol_idx in range(len(self._volumes)):
            image, mask = self._get_volume(vol_idx)
            n_slices = image.shape[1]
            for slice_idx in range(n_slices):
                if image[:, slice_idx].max() > 0:
                    self._index.append((vol_idx, slice_idx))

    def _get_volume(self, vol_idx: int):
        if self._cache_vol_idx != vol_idx:
            pid = self._volumes.subjects[vol_idx]
            self._cache_image, self._cache_mask = self._volumes._load_volume(pid)
            self._cache_vol_idx = vol_idx
        return self._cache_image, self._cache_mask

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.mode == "3d":
            vol_idx = self._index[idx]
            image, mask = self._get_volume(vol_idx)
            label = float(mask.max() > 0)
            return {"image": torch.from_numpy(image), "label": torch.tensor(label)}

        vol_idx, slice_idx = self._index[idx]
        image, mask = self._get_volume(vol_idx)
        image_slice = image[:, slice_idx].copy()
        mask_slice = mask[slice_idx]
        label = float(mask_slice.max() > 0)
        return {"image": torch.from_numpy(image_slice), "label": torch.tensor(label)}
