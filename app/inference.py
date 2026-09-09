"""
Inference helpers for the Streamlit glioblastoma app.

Reuses the project's preprocessing (float32 z-score) and model builder so the
app matches how the model was trained: input is the four co-registered MRI
modalities in the order [T1, T1Gd, T2, FLAIR], z-score normalized per modality
and center-cropped to a 128^3 patch; output is a 4-class voxel segmentation
(0 background, 1 necrotic/non-enhancing core, 2 edema, 3 enhancing tumour).
"""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

# Make the project package importable regardless of where Streamlit is launched.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.nifti_brats_dataset import _zscore_normalize, MODALITY_SUFFIXES  # noqa: E402
from models.build_model import build_model  # noqa: E402

# Display order / labels for the four input modalities.
MODALITIES = list(MODALITY_SUFFIXES)  # ["t1", "t1ce", "t2", "flair"]
MODALITY_LABELS = {"t1": "T1", "t1ce": "T1Gd (contrast)", "t2": "T2", "flair": "FLAIR"}

CLASS_NAMES = {
    1: "Necrotic / non-enhancing core",
    2: "Peritumoral edema",
    3: "Enhancing tumour",
}
# RGBA overlay colours per foreground class.
CLASS_COLORS = {1: (220, 30, 30), 2: (30, 200, 30), 3: (40, 110, 255)}

ROI = (128, 128, 128)
VOXEL_ML = 0.001  # 1 mm^3 isotropic -> 0.001 mL

# Bundled fp16 weights (~38 MB) so the app is self-contained and deployable
# without external weight hosting.
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "weights" / "brats_hgg_unet_fp16.pt")


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device):
    """Build the 3D U-Net and load weights. Returns (model, epoch)."""
    cfg = {"data": {"mode": "nifti"},
           "model": {"in_channels": 4, "out_channels": 4, "dropout": 0.2}}
    model = build_model(cfg, in_channels=4, out_channels=4)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    # Weights may be stored in fp16 (bundled file); cast back to fp32 for compute.
    state = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    epoch = ck.get("epoch") if isinstance(ck, dict) else None
    return model, epoch


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as a float32 numpy array."""
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def _center_crop(vol: np.ndarray, roi=ROI) -> np.ndarray:
    """Center-crop (and pad if needed) a (C, D, H, W) volume to roi."""
    _, d, h, w = vol.shape
    pd, ph, pw = roi
    d0, h0, w0 = max(0, d // 2 - pd // 2), max(0, h // 2 - ph // 2), max(0, w // 2 - pw // 2)
    crop = vol[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    if crop.shape[1:] != roi:
        out = np.zeros((vol.shape[0],) + roi, dtype=np.float32)
        cd, ch, cw = crop.shape[1:]
        out[:, :cd, :ch, :cw] = crop
        return out
    return crop


def preprocess(modality_volumes: list[np.ndarray]) -> np.ndarray:
    """Stack [t1, t1ce, t2, flair], z-score per modality, center-crop -> (4, D, H, W)."""
    if len(modality_volumes) != 4:
        raise ValueError("Expected 4 modality volumes in order [T1, T1Gd, T2, FLAIR].")
    shapes = {v.shape for v in modality_volumes}
    if len(shapes) != 1:
        raise ValueError(f"Modalities have mismatched shapes: {shapes}. They must be co-registered.")
    image = np.stack(modality_volumes, axis=0).astype(np.float32)
    image = _zscore_normalize(image, axis=(-3, -2, -1))
    return _center_crop(image, ROI)


@torch.no_grad()
def predict(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Run the model, return an integer label volume (D, H, W)."""
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    if device.type == "cuda":
        with torch.autocast("cuda"):
            logits = model(x)
    else:
        logits = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def summarize(pred: np.ndarray, min_tumor_voxels: int = 50) -> dict:
    """Detection + volumetric summary from a label volume."""
    per_class = {c: int((pred == c).sum()) for c in (1, 2, 3)}
    total = sum(per_class.values())
    return {
        "tumor_present": total >= min_tumor_voxels,
        "total_voxels": total,
        "total_ml": round(total * VOXEL_ML, 2),
        "per_class_voxels": per_class,
        "per_class_ml": {c: round(v * VOXEL_ML, 2) for c, v in per_class.items()},
    }


def best_tumor_slice(pred: np.ndarray) -> int:
    """Axial slice index (first axis) with the most tumour voxels."""
    per_slice = (pred > 0).reshape(pred.shape[0], -1).sum(axis=1)
    return int(per_slice.argmax()) if per_slice.max() > 0 else pred.shape[0] // 2


def make_overlay(bg_slice: np.ndarray, label_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return an RGB uint8 image: grayscale MRI with coloured tumour overlay."""
    bg = bg_slice.astype(np.float32)
    lo, hi = np.percentile(bg, 1), np.percentile(bg, 99)
    if hi > lo:
        bg = np.clip((bg - lo) / (hi - lo), 0, 1)
    rgb = np.stack([bg, bg, bg], axis=-1)
    for cls, color in CLASS_COLORS.items():
        mask = label_slice == cls
        if mask.any():
            col = np.array(color, dtype=np.float32) / 255.0
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * col
    return (rgb * 255).astype(np.uint8)
