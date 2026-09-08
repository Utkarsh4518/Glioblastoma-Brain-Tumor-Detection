"""
Environment sanity check: verifies required packages import and reports
Python/PyTorch/CUDA versions. Exits non-zero if a required package is missing.

Run from repo root:
    python tools/check_environment.py
"""

import sys

REQUIRED = ["torch", "torchvision", "monai", "nibabel", "h5py", "hydra", "sklearn", "numpy"]


def main() -> int:
    print(f"Python: {sys.version}")

    missing = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt (see README for the PyTorch/CUDA step)")
        return 1

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA memory: {props.total_memory / (1024**3):.1f} GB")
    else:
        print("No CUDA device detected; training will run on CPU (slow).")

    import monai

    print(f"MONAI: {monai.__version__}")
    print("OK: environment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
