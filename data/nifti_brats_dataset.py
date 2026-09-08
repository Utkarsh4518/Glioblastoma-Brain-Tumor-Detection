"""
BraTS 2020 NIfTI volume dataset.

Loads per-subject NIfTI files, stacks modalities into (C, D, H, W), remaps
segmentation label 4 -> 3. Returns torch tensors. No MONAI transforms.

Handles real-world quirks of the BraTS2020 distribution:
  - files may be .nii or .nii.gz,
  - one subject (BraTS20_Training_355) has a non-standard seg filename
    (W39_1998.09.19_Segm.nii), so seg is discovered by fallback,
  - optional filtering by tumor grade (HGG = glioblastoma, LGG) via the
    dataset's name_mapping.csv.

Expected structure:
    root/
        subject_id/
            subject_id_t1.nii[.gz]
            subject_id_t1ce.nii[.gz]
            subject_id_t2.nii[.gz]
            subject_id_flair.nii[.gz]
            subject_id_seg.nii[.gz]   (or a *seg*/*Segm* fallback)
    root/name_mapping.csv             (optional; needed for grade_filter)
"""

import csv
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]
_NIFTI_EXTS = (".nii.gz", ".nii")


def _find_modality(subject_dir: Path, pid: str, mod: str) -> Optional[Path]:
    """Find {pid}_{mod}.nii or .nii.gz."""
    for ext in _NIFTI_EXTS:
        cand = subject_dir / f"{pid}_{mod}{ext}"
        if cand.exists():
            return cand
    return None


def _find_seg(subject_dir: Path, pid: str) -> Optional[Path]:
    """Find the segmentation file, falling back to any *seg*/*Segm* NIfTI."""
    for ext in _NIFTI_EXTS:
        cand = subject_dir / f"{pid}_seg{ext}"
        if cand.exists():
            return cand
    for cand in sorted(subject_dir.iterdir()):
        name = cand.name.lower()
        if "seg" in name and name.endswith(_NIFTI_EXTS):
            return cand
    return None


def _load_grade_map(root: Path) -> dict[str, str]:
    """Map BraTS2020 subject ID -> grade (HGG/LGG) from name_mapping.csv, if present."""
    csv_path = root / "name_mapping.csv"
    if not csv_path.exists():
        return {}
    grades: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("BraTS_2020_subject_ID", "").strip()
            grade = row.get("Grade", "").strip().upper()
            if sid and grade:
                grades[sid] = grade
    return grades


def _zscore_normalize(x: np.ndarray, axis: Optional[tuple] = None) -> np.ndarray:
    """
    Z-score normalization per channel, done in float32 and in place.

    Full BraTS volumes are large (4x240x240x155); mixing in float64 (numpy's
    default accumulation dtype) would create ~286 MB intermediates per volume
    and, with several parallel DataLoader workers, exhaust host RAM. Keeping
    everything float32 and updating in place avoids those extra copies.
    """
    x = np.asarray(x, dtype=np.float32)
    axis = axis or tuple(range(x.ndim))
    mean = np.mean(x, axis=axis, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=axis, keepdims=True, dtype=np.float32)
    std = np.where(std > 1e-8, std, np.float32(1.0)).astype(np.float32)
    x -= mean
    x /= std
    return x


def _remap_mask_four_class(seg: np.ndarray) -> np.ndarray:
    """Map BraTS labels: 0,1,2 unchanged; 4 -> 3."""
    out = np.asarray(seg, dtype=np.int64)
    out[seg == 4] = 3
    return out


class NiftiBraTSDataset(Dataset):
    """
    BraTS 2020 segmentation dataset from NIfTI volumes.

    Loads per subject, stacks modalities (C, D, H, W), remaps mask 4->3.
    Returns torch tensors: image (C, D, H, W), mask (1, D, H, W).
    Optional center-crop to patch_size and optional grade filtering.
    """

    def __init__(
        self,
        root: Path | str,
        patch_size: Optional[tuple[int, int, int]] = None,
        grade_filter: Optional[str] = None,
    ):
        """
        Args:
            root: Data root containing subject_id/ subdirs.
            patch_size: Optional (D, H, W); center-crop if set, else full volume.
            grade_filter: "HGG" (glioblastoma), "LGG", or None (all). Requires
                name_mapping.csv in root.
        """
        self.root = Path(root)
        self.patch_size = patch_size
        self.grade_filter = grade_filter.upper() if grade_filter else None

        grade_map = _load_grade_map(self.root) if self.grade_filter else {}
        if self.grade_filter and not grade_map:
            raise FileNotFoundError(
                f"grade_filter={self.grade_filter!r} requires name_mapping.csv under {self.root}"
            )

        self._paths: dict[str, dict[str, Path]] = {}
        for item in sorted(self.root.iterdir()):
            if not item.is_dir():
                continue
            pid = item.name
            if self.grade_filter and grade_map.get(pid) != self.grade_filter:
                continue
            mod_paths = {mod: _find_modality(item, pid, mod) for mod in MODALITY_SUFFIXES}
            seg_path = _find_seg(item, pid)
            if any(p is None for p in mod_paths.values()) or seg_path is None:
                continue
            mod_paths["seg"] = seg_path
            self._paths[pid] = mod_paths

        self.subjects = sorted(self._paths.keys())
        if not self.subjects:
            filt = f" (grade_filter={self.grade_filter})" if self.grade_filter else ""
            raise FileNotFoundError(
                f"No BraTS subjects found under {self.root}{filt}. "
                "Expected: subject_id/subject_id_{{t1,t1ce,t2,flair,seg}}.nii[.gz]"
            )

    def __len__(self) -> int:
        return len(self.subjects)

    def _load_volume(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """Load image (C, D, H, W) and mask (D, H, W), reading directly as float32."""
        paths = self._paths[pid]
        # get_fdata(dtype=np.float32) reads in float32 instead of the default
        # float64, halving the per-volume memory footprint (important with
        # multiple DataLoader workers).
        modalities = [nib.load(paths[mod]).get_fdata(dtype=np.float32) for mod in MODALITY_SUFFIXES]
        image = np.stack(modalities, axis=0)
        image = _zscore_normalize(image, axis=(-3, -2, -1))

        seg = nib.load(paths["seg"]).get_fdata(dtype=np.float32)
        mask = _remap_mask_four_class(seg)
        return image, mask

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
            img_pad = np.zeros((image.shape[0],) + self.patch_size, dtype=np.float32)
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
            # Channel-first (1, D, H, W): matches MONAI dict-transform and
            # DiceCELoss(to_onehot_y=True) conventions.
            "mask": torch.from_numpy(mask).long().unsqueeze(0),
        }
