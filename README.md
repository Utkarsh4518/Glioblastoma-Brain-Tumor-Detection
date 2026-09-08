# BraTS 2020: Brain Tumor Segmentation, Classification & Survival Prediction

PyTorch pipeline for BraTS 2020–style brain tumor segmentation and classification. Supports 3D NIfTI volumes and 2D slice-based HDF5 data. Built with MONAI and Hydra.

## Features

- 3D and 2D segmentation (UNet; 2D/3D chosen by data mode)
- Binary tumor classification (EfficientNet-B0, ResNet18)
- Configurable device (CPU / CUDA / auto)
- Forward validation script for 2D and 3D modes

## Status

- **3D NIfTI segmentation** is the primary, exercised path: official BraTS2020 layout, 4-class UNet, Dice+CE loss, Dice metric.
- **2D HDF5 segmentation** and the **classification** task are implemented but assume a specific slice/mask key layout (see `data/h5_brats_dataset.py`) that may need adjusting to whichever Kaggle mirror you use.
- No survival-prediction or uncertainty-quantification module exists yet, despite earlier drafts of this README describing one.

## Installation

Requires Python 3.10+ (tested on 3.10 and 3.14).

```bash
git clone https://github.com/Utkarsh4518/Glioblastoma-Brain-Tumor-Detection.git
cd Glioblastoma-Brain-Tumor-Detection
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Install PyTorch first, matching your hardware:
#   CUDA GPU:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   CPU only:  pip install torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python tools/check_environment.py
```

`check_environment.py` reports the detected Python/PyTorch/CUDA versions and fails loudly if a required package is missing.

## Notebook (recommended entry point)

`brats_glioblastoma_pipeline.ipynb` runs the whole study end to end — data loading, a sample
visualization, 3D U-Net training with live curves, evaluation (per-class Dice + HD95), overlay
figures, and inference on an unlabeled validation volume. Set `DATA_ROOT` and `GRADE_FILTER` in the
config cell (use `GRADE_FILTER="HGG"` to train on glioblastoma/high-grade cases only), leave
`QUICK_TEST=True` for a fast end-to-end check, then set it `False` for the full run. All figures are
saved under `outputs/` for the write-up.

## Data

Download BraTS 2020–style data and set `DATA_ROOT` to the training data directory. Files may be
`.nii` or `.nii.gz`. If `name_mapping.csv` is present, `grade_filter` (HGG/LGG) can subset by grade.

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

## Evaluation

Evaluate a trained checkpoint and generate figures for the results write-up.

**Segmentation** — per-class Dice + HD95 and MRI/mask overlay panels:

```bash
python tools/evaluate.py --task segmentation \
  --checkpoint checkpoints/brats2020_default_best.pt --mode nifti
```

**Classification** — accuracy/precision/recall/F1/ROC-AUC, confusion matrix and ROC curve:

```bash
python tools/evaluate.py --task classification \
  --backbone resnet18 --checkpoint checkpoints/brats2020_default_best.pt
```

Outputs (`metrics.json` + PNGs) are written to `outputs/<task>_eval/`. Note Hydra changes the working directory at train time, so pass `paths.checkpoint_dir=<absolute path>` when training if you want checkpoints in a predictable location.

## Validation

Run one forward pass per mode (2D and 3D) to check data and model. Skips a mode if its data path is missing.

```bash
python tools/test_modes.py
```

## Reproducibility

Seed is fixed in config (default 42). Train/val split is 85/15 with a deterministic generator. Use `pip freeze > requirements-frozen.txt` for pinned dependencies.

## License

Code: MIT. BraTS data use must follow the official BraTS data terms.
