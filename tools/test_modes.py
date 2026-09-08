"""
Forward-pass validation for 2D (H5) and 3D (NIfTI) modes.

Loads dataset and model per mode (if data exists), runs one forward pass,
prints input/output shape, device, dtype. No training. Exits 0 on success.

Run from repo root:
  python tools/test_modes.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def _make_cfg(mode: str, data_root: str) -> dict:
    """Minimal config for build_dataset and build_model."""
    return {
        "data": {
            "mode": mode,
            "path": data_root,
            "roi_size": [128, 128, 128],
            "train_val_split": 0.85,
            "pattern": "volume_*.h5",
            "image_key": None,
            "mask_key": None,
        },
        "paths": {"data_root": data_root},
        "model": {"in_channels": 4, "out_channels": 4, "dropout": 0.2},
        "seed": 42,
    }


def _run_mode(name: str, cfg: dict, device: torch.device) -> bool:
    """Load dataset (train subset), build model, one forward pass. Returns True if OK."""
    from data import build_dataset
    from models.build_model import build_model

    try:
        dataset = build_dataset(cfg, "train")
    except FileNotFoundError as e:
        print(f"  [{name}] Skip: no data ({e})")
        return False
    if len(dataset) == 0:
        print(f"  [{name}] Skip: empty dataset")
        return False

    sample = dataset[0]
    image = sample["image"]
    in_channels = int(image.shape[0])
    out_channels = 4

    model = build_model(cfg, in_channels=in_channels, out_channels=out_channels)
    model = model.to(device)
    model.eval()

    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)

    print(f"  [{name}] input shape:  {tuple(x.shape)}")
    print(f"  [{name}] output shape: {tuple(out.shape)}")
    print(f"  [{name}] device: {out.device}")
    print(f"  [{name}] dtype:  {out.dtype}")
    return True


def main() -> int:
    device = torch.device("cpu")
    data_root = os.environ.get("DATA_ROOT", "").strip() or str(PROJECT_ROOT / "data" / "brats2020")
    h5_root = os.environ.get("DATA_ROOT", "").strip() or str(PROJECT_ROOT / "data" / "h5_brats")

    print("Forward validation (2D and 3D modes)")
    print("-" * 50)

    ran_2d = False
    ran_3d = False

    cfg_2d = _make_cfg("h5", h5_root)
    if Path(h5_root).exists():
        print("2D (H5):")
        ran_2d = _run_mode("2D (H5)", cfg_2d, device)
    else:
        print("2D (H5):")
        print(f"  [2D (H5)] Skip: path does not exist: {h5_root}")

    print()

    cfg_3d = _make_cfg("nifti", data_root)
    if Path(data_root).exists():
        print("3D (NIfTI):")
        ran_3d = _run_mode("3D (NIfTI)", cfg_3d, device)
    else:
        print("3D (NIfTI):")
        print(f"  [3D (NIfTI)] Skip: path does not exist: {data_root}")

    print("-" * 50)
    if ran_2d or ran_3d:
        print("OK: forward validation completed.")
        return 0
    print("OK: no data found for either mode (skipped). Exiting successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
