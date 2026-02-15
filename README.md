# BraTS 2020: Brain Tumor Segmentation, Classification & Survival Prediction

PyTorch pipeline for BraTS 2020–style brain tumor segmentation and classification. Supports 3D NIfTI volumes and 2D slice-based HDF5 data. Built with MONAI and Hydra.

## Features

- 3D and 2D segmentation (UNet; 2D/3D chosen by data mode)
- Binary tumor classification (EfficientNet-B0, ResNet18)
- Survival prediction (radiomics + MLP/XGBoost)
- Configurable device (CPU / CUDA / auto)
- Forward validation script for 2D and 3D modes

## Installation

### Conda

```bash
git clone https://github.com/Utkarsh4518/Glioblastoma-Brain-Tumor-Detection.git
cd Glioblastoma-Brain-Tumor-Detection
conda env create -f environment.yaml
conda activate brats2020
python tools/check_environment.py
```

### venv

```bash
git clone https://github.com/Utkarsh4518/Glioblastoma-Brain-Tumor-Detection.git
cd Glioblastoma-Brain-Tumor-Detection
python3.10 -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
python tools/check_environment.py
```

## Data

Download BraTS 2020–style data and set `DATA_ROOT` to the training data directory.

**3D (NIfTI):** One folder per subject; each folder contains `patient_id_t1.nii.gz`, `patient_id_t1ce.nii.gz`, `patient_id_t2.nii.gz`, `patient_id_flair.nii.gz`, `patient_id_seg.nii.gz`.

**2D (HDF5):** Directory of `volume_*.h5` files (slice-based). Set `DATA_ROOT` to that directory and use `data=h5` when training.

```bash
# Windows
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData

# Linux/macOS
export DATA_ROOT=/path/to/BraTS2020_TrainingData
```

## Training

### 3D Mode

Uses NIfTI volumes and a 3D UNet.

```bash
python train.py task=segmentation
```

### 2D Mode

Uses HDF5 slice data and a 2D UNet.

```bash
python train.py data=h5 task=segmentation
```

### CPU Debug

Run on CPU with minimal epochs (e.g. for debugging or when no GPU is available):

```bash
python train.py training.device=cpu data.num_workers=0 training.max_epochs=1
```

## Validation

Run one forward pass per mode (2D and 3D) to check data and model. Skips a mode if its data path is missing.

```bash
python tools/test_modes.py
```

## Reproducibility

Seed is fixed in config (default 42). Train/val split is 85/15 with a deterministic generator. Use `pip freeze > requirements-frozen.txt` for pinned dependencies.

## License

Code: MIT. BraTS data use must follow the official BraTS data terms.
