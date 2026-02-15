"""
BraTS 2020 NIfTI volume dataset.

Loads .nii.gz files per subject, stacks modalities into (C, D, H, W),
remaps segmentation 4 -> 3. Returns torch tensors. No MONAI transforms.

Expected structure:
    root/
        patient_id/
            patient_id_t1.nii.gz
            patient_id_t1ce.nii.gz
            patient_id_t2.nii.gz
            patient_id_flair.nii.gz
            patient_id_seg.nii.gz
"""

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]


def _discover_subjects(root: Path) -> list[str]:
    """Find all patient directories with required NIfTI files."""
    root = Path(root)
    subjects = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        pid = item.name
        required = [
            item / f"{pid}_t1.nii.gz",
            item / f"{pid}_t1ce.nii.gz",
            item / f"{pid}_t2.nii.gz",
            item / f"{pid}_flair.nii.gz",
            item / f"{pid}_seg.nii.gz",
        ]
        if all(f.exists() for f in required):
            subjects.append(pid)
    return sorted(subjects)


def _zscore_normalize(x: np.ndarray, axis: Optional[tuple] = None) -> np.ndarray:
    """Z-score normalization; axis defaults to spatial dims."""
    axis = axis or tuple(range(x.ndim))
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return ((x - mean) / std).astype(np.float32)


def _remap_mask_four_class(seg: np.ndarray) -> np.ndarray:
    """Map BraTS labels: 0,1,2 unchanged; 4 -> 3."""
    out = np.asarray(seg, dtype=np.int64)
    out[seg == 4] = 3
    return out


class NiftiBraTSDataset(Dataset):
    """
    BraTS 2020 segmentation dataset from NIfTI volumes.

    Loads .nii.gz per subject, stacks modalities (C, D, H, W), remaps mask 4->3.
    Returns torch tensors. Optional center-crop to patch_size.
    """

    def __init__(
        self,
        root: Path | str,
        patch_size: Optional[tuple[int, int, int]] = None,
    ):
        """
        Args:
            root: Data root containing patient_id/ subdirs with *_t1.nii.gz, etc.
            patch_size: Optional (D, H, W). If set, center-crop to this size; else full volume.
        """
        self.root = Path(root)
        self.patch_size = patch_size

        self.subjects = _discover_subjects(self.root)
        if not self.subjects:
            raise FileNotFoundError(
                f"No BraTS subjects found under {self.root}. "
                "Expected: patient_id/patient_id_{{t1,t1ce,t2,flair,seg}}.nii.gz"
            )

        self._paths: dict[str, dict[str, Path]] = {}
        for pid in self.subjects:
            d = self.root / pid
            self._paths[pid] = {
                "t1": d / f"{pid}_t1.nii.gz",
                "t1ce": d / f"{pid}_t1ce.nii.gz",
                "t2": d / f"{pid}_t2.nii.gz",
                "flair": d / f"{pid}_flair.nii.gz",
                "seg": d / f"{pid}_seg.nii.gz",
            }

    def __len__(self) -> int:
        return len(self.subjects)

    def _load_volume(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """Load image (C, D, H, W) and mask (D, H, W)."""
        paths = self._paths[pid]
        modalities = []
        for mod in MODALITY_SUFFIXES:
            arr = nib.load(paths[mod]).get_fdata()
            modalities.append(arr)
        image = np.stack(modalities, axis=0)
        image = _zscore_normalize(image, axis=(-3, -2, -1))

        seg = nib.load(paths["seg"]).get_fdata()
        mask = _remap_mask_four_class(seg)
        return image.astype(np.float32), mask

    def _crop_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center-crop to patch_size. image (C,D,H,W), mask (D,H,W)."""
        pd, ph, pw = self.patch_size
        _, d, h, w = image.shape
        cd, ch, cw = d // 2, h // 2, w // 2
        d0 = max(0, cd - pd // 2)
        h0 = max(0, ch - ph // 2)
        w0 = max(0, cw - pw // 2)
        d1 = min(d, d0 + pd)
        h1 = min(h, h0 + ph)
        w1 = min(w, w0 + pw)
        img_patch = image[:, d0:d1, h0:h1, w0:w1]
        msk_patch = mask[d0:d1, h0:h1, w0:w1]
        if img_patch.shape[1:] != self.patch_size:
            img_pad = np.zeros((4,) + self.patch_size, dtype=np.float32)
            msk_pad = np.zeros(self.patch_size, dtype=np.int64)
            sd, sh, sw = img_patch.shape[1:]
            img_pad[:, :sd, :sh, :sw] = img_patch
            msk_pad[:sd, :sh, :sw] = msk_patch
            return img_pad, msk_pad
        return img_patch, msk_patch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = self.subjects[idx]
        image, mask = self._load_volume(pid)
        if self.patch_size is not None:
            image, mask = self._crop_patch(image, mask)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
        }
