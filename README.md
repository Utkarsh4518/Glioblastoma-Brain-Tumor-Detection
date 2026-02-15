# BraTS 2020: Brain Tumor Segmentation, Classification & Survival Prediction

A PyTorch pipeline for the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) challenge, supporting segmentation, binary tumor classification, survival prediction, and uncertainty quantification.

---

## Project Overview

This repository provides:

- **Segmentation**: Multi-class brain tumor sub-region segmentation (3D UNet, SwinUNETR)
- **Classification**: Binary tumor presence detection (EfficientNet-B0, ResNet18)
- **Survival prediction**: Radiomics + deep features with MLP/XGBoost
- **Uncertainty estimation**: Monte Carlo Dropout for voxel-wise epistemic uncertainty
- **Evaluation**: Dice, Hausdorff 95%, sensitivity, specificity; visual overlays (MRI + mask)

Built with PyTorch, MONAI, and Hydra for reproducible experiments.

---

## BraTS 2020 Dataset

The **Brain Tumor Segmentation (BraTS) 2020** challenge provides multimodal MRI scans of glioma patients from The Cancer Genome Atlas (TCGA) and other multi-institutional sources.

### Modalities

| Modality | Description |
|----------|-------------|
| T1      | Native T1-weighted |
| T1ce    | Contrast-enhanced T1 (T1Gd) |
| T2      | T2-weighted |
| FLAIR   | Fluid-attenuated inversion recovery |

### Segmentation Labels

| Label | Region | Raw value |
|-------|--------|-----------|
| 0 | Background | 0 |
| 1 | NCR/NET (Necrotic core) | 1 |
| 2 | ED (Peritumoral edema) | 2 |
| 3 | ET (Enhancing tumor) | 4 |

Data is preprocessed: 1 mm³ isotropic, skull-stripped, co-registered. Register and download from the [official BraTS 2020 site](https://www.med.upenn.edu/cbica/brats2020/registration.html).

---

## HDF5 Slice-Based Mode (Kaggle Version)

The pipeline can run on **preprocessed Kaggle HDF5 slice datasets** (e.g. `volume_*.h5` with `meta_data.csv` and `survival_info.csv`) instead of raw NIfTI volumes. This mode is intended for experimentation and teaching when full 3D data or official BraTS downloads are not available.

- **2D segmentation only**: Slice-based UNet (`unet_2d`); no 3D volumes or 3D metrics.
- **Slice-level classification**: Binary tumor presence per slice with slice-level (or meta-driven) train/val splits.
- **Aggregated survival features**: Per-volume slice-derived features (tumor slice count, mean/max/total tumor area) merged with `survival_info.csv`; no true volumetric features from slices.
- **Reduced VRAM requirement**: 2D models and slice batches allow smaller GPUs than 3D UNet.
- **Not identical to official BraTS evaluation**: Metrics and protocols differ from the official BraTS 2020 challenge.

> **Warning:** This mode uses preprocessed Kaggle HDF5 slices and is **not directly comparable to official BraTS 2020 challenge submissions.** Use it for development and education; for challenge-comparable results, use the official NIfTI dataset and 3D pipeline.

**Quick start (HDF5):** Set `DATA_ROOT` to the HDF5 dataset directory, then run `python train.py data=h5` (segmentation) or use `tools/inspect_h5_dataset.py`, `tools/sanity_check_h5.py`, and `tools/build_h5_survival_features.py` as needed.

---

## Model Architectures

### Segmentation

| Model | Reference | Notes |
|-------|-----------|-------|
| **UNet** | Çiçek et al., MICCAI 2016; Milletari, 3DV 2016 | 3D U-Net with residual units, batch norm, dropout (MONAI) |
| **SwinUNETR** | Hatamizadeh et al., CVPR 2022 | Transformer-based encoder–decoder (MONAI) |

### Classification

| Model | Reference | Notes |
|-------|-----------|-------|
| **EfficientNet-B0** | Tan & Le, ICML 2019 | Pretrained ImageNet, 4→3 channel adapter |
| **ResNet18** | He et al., CVPR 2016 | Pretrained ImageNet, 4→3 channel adapter |

### Survival

| Model | Notes |
|-------|-------|
| **MLP** | Multi-layer perceptron on radiomics + deep features |
| **XGBoost** | Gradient boosting on radiomics features |

---

## Results

Results are populated from evaluation outputs. Run evaluations first, then generate the summary:

```bash
# 1. Run evaluations (requires DATA_ROOT and checkpoints)
python tools/run_full_segmentation.py
python tools/run_full_classification.py
python -m training.train_survival --features_csv outputs/survival_features.csv

# 2. Collect metrics and update README
python tools/generate_results_summary.py --update-readme
```

### Segmentation (UNet)

| Metric | Background | NCR/NET | ED | ET | Mean |
|--------|------------|---------|----|----|------|
| Dice   | —          | —       | —  | —  | —    |
| HD95   | —          | —       | —  | —  | —    |

*Source: `outputs/segmentation_eval/metrics.json`*

**Sample overlay** (appears after running `run_full_segmentation.py`):

![Segmentation overlay](outputs/segmentation_eval/overlays/seg_overlay_000.png)

### Classification (EfficientNet-B0)

| Metric      | Value |
|-------------|-------|
| Accuracy    | —     |
| ROC-AUC     | —     |
| F1          | —     |
| Precision   | —     |
| Recall      | —     |

*Source: `outputs/classification_eval/metrics.json`*

**Confusion matrix** (appears after running `run_full_classification.py`):

![Confusion matrix](outputs/classification_eval/confusion_matrix.png)

### Survival (5-fold CV)

| Model   | MAE (days)       | RMSE (days)      |
|---------|------------------|------------------|
| MLP     | — ± —            | — ± —            |
| XGBoost | — ± —            | — ± —            |

*Source: `outputs/survival_cv_summary.txt`*

### Monte Carlo Dropout (Uncertainty)

**Mean segmentation** and **uncertainty heatmap** (appear after running `run_mc_dropout_inference.py`):

![MC mean segmentation](outputs/mc_dropout/overlays/mc_mean_seg_000.png)  
*Mean prediction overlay*

![MC uncertainty heatmap](outputs/mc_dropout/overlays/mc_uncertainty_000.png)  
*Voxel-wise variance (uncertainty) overlay*

*Source: `outputs/mc_dropout/overlays/`*

---

## Installation

### Requirements

- **Python:** 3.10.x (recommended: 3.10.14)
- **GPU:** NVIDIA GPU with CUDA 11.8+ and ≥8 GB VRAM for 3D segmentation (CPU-only possible but slow)
- **Data:** BraTS 2020 data must be downloaded separately (see below)

> **Warning:** You must download BraTS 2020 data separately. This repository does not include any imaging data. Register at the [official BraTS 2020 site](https://www.med.upenn.edu/cbica/brats2020/registration.html) to obtain access.

### Setup (venv)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Glioblastoma-Brain-Tumor-Detection.git
cd Glioblastoma-Brain-Tumor-Detection

# Create venv with Python 3.10
python3.10 -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
# source .venv/bin/activate

# Install dependencies (flexible versions)
pip install -r requirements.txt

# OR install frozen versions for exact reproducibility
# pip install -r requirements-frozen.txt

# Verify environment
python tools/check_environment.py

# Forward validation (2D and 3D modes; skips if no data)
python tools/test_modes.py
```

### Setup (conda)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Glioblastoma-Brain-Tumor-Detection.git
cd Glioblastoma-Brain-Tumor-Detection

# Create conda environment
conda env create -f environment.yaml

# Activate
conda activate brats2020

# Verify environment
python tools/check_environment.py

# Forward validation (2D and 3D modes; skips if no data)
python tools/test_modes.py
```

### Data

> **Must download BraTS 2020 data separately.** Register at [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/registration.html) to obtain access. This repository does not include imaging data.

Set the path to the BraTS 2020 data directory:

```bash
# Windows
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData

# Linux/macOS
export DATA_ROOT=/path/to/BraTS2020_TrainingData
```

Expected structure:

```
BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_001_flair.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
...
```

---

## Training

### Full pipeline (all steps)

Run the complete pipeline end-to-end: sanity check → segmentation → classification → survival → uncertainty → README update. Aborts on first failure. Logs total runtime.

```bash
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData   # Windows
# export DATA_ROOT=/path/to/BraTS2020_TrainingData  # Linux/macOS

python tools/run_full_pipeline.py
# Optional: with BraTS survival CSV for survival labels
python tools/run_full_pipeline.py --survival_csv path/to/survival_data.csv
```

### Full segmentation (train + eval + overlays)

```bash
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData   # Windows
# export DATA_ROOT=/path/to/BraTS2020_TrainingData  # Linux/macOS

python tools/run_full_segmentation.py
```

Runs 3D UNet training (100 epochs, 85/15 split, seed 42), validation evaluation (Dice, HD95), overlay saves to `outputs/segmentation_eval/`, and updates the Results table above.

### Full classification (train + eval + confusion matrix)

```bash
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData   # Windows
# export DATA_ROOT=/path/to/BraTS2020_TrainingData  # Linux/macOS

python tools/run_full_classification.py
```

Runs EfficientNet-B0 binary classification (50 epochs, patient-level 85/15 split, early stopping patience=10), saves confusion matrix to `outputs/classification_eval/confusion_matrix.png`, and updates the Results table above.

### Segmentation (train only)

```bash
python train.py task=segmentation model.name=unet
python train.py task=segmentation model.name=swin_unetr  # requires more GPU memory
```

### Classification

```bash
python train.py task=classification model.name=efficientnet_b0
python train.py task=classification model.name=resnet18
```

### Survival

```bash
# 1. Extract survival features (volume, radiomics, deep encoder features)
python tools/extract_survival_features.py \
  --data_root $DATA_ROOT \
  --survival_csv path/to/survival_data.csv \
  --checkpoint checkpoints/segmentation_best.pt \
  --output_csv outputs/survival_features.csv

# 2. Train survival models (5-fold CV, MLP + XGBoost)
python -m training.train_survival --features_csv outputs/survival_features.csv
```

- **5-fold cross-validation** at patient level
- **Metrics**: MAE and RMSE (mean ± std across folds)
- **Warning** if MAE < 100 days (likely overfitting or leakage)
- Summary saved to `outputs/survival_cv_summary.txt`

### Config Overrides

```bash
python train.py experiment_name=my_run seed=42 training.max_epochs=50 data.batch_size=4
```

### Device and CPU fallback

Training uses the device set in config: `training.device` can be `auto`, `cpu`, or `cuda`.

- **`auto`** (default): use CUDA if available, otherwise CPU.
- **`cuda`**: use GPU; if no GPU is available, a warning is logged and CPU is used.
- **`cpu`**: force CPU (no CUDA). Use for debugging or machines without a GPU.

**CPU fallback:** To run entirely on CPU (e.g. no GPU or to avoid CUDA), set `training.device=cpu` and use `data.num_workers=0` to avoid multiprocessing issues:

```bash
python train.py training.device=cpu data.num_workers=0 training.max_epochs=1
```

Training will be slower on CPU; reduce `training.max_epochs` or `data.batch_size` for quick checks.

### Running in 2D mode

2D mode uses **slice-based HDF5 data** (e.g. Kaggle `volume_*.h5`). The pipeline builds a 2D UNet and applies 2D augmentations.

1. Set your HDF5 data path (e.g. `DATA_ROOT` or override in config).
2. Run segmentation with `data=h5`:

```bash
# Windows
set DATA_ROOT=C:\path\to\h5_brats
python train.py data=h5 task=segmentation

# Linux/macOS
export DATA_ROOT=/path/to/h5_brats
python train.py data=h5 task=segmentation
```

3. Optional: validate the 2D pipeline with one forward pass (no data required if path is missing):

```bash
python tools/test_modes.py
```

### Running in 3D mode

3D mode uses **NIfTI volumes** (BraTS 2020 directory layout: `patient_id/patient_id_{t1,t1ce,t2,flair,seg}.nii.gz`). The pipeline builds a 3D UNet and applies 3D augmentations.

1. Download BraTS 2020 data and set `DATA_ROOT` to the training data directory.
2. Run segmentation with the default data config (nifti):

```bash
# Windows
set DATA_ROOT=C:\path\to\BraTS2020_TrainingData
python train.py task=segmentation

# Linux/macOS
export DATA_ROOT=/path/to/BraTS2020_TrainingData
python train.py task=segmentation
```

3. Optional: validate the 3D pipeline:

```bash
python tools/test_modes.py
```

---

## Evaluation

Run full evaluation with metrics and visual overlays:

```bash
# Segmentation (Dice, HD95, sensitivity, specificity + PNG overlays)
python evaluation.py --task segmentation --checkpoint checkpoints/brats2020_default_best.pt --output_dir outputs/eval

# Classification (Accuracy, ROC-AUC, confusion matrix)
python evaluation.py --task classification --checkpoint path/to/classifier.pt
```

### Monte Carlo Dropout (Uncertainty)

Monte Carlo Dropout inference for voxel-wise epistemic uncertainty:

```bash
python tools/run_mc_dropout_inference.py \
  --checkpoint checkpoints/segmentation_best.pt \
  --data_root $DATA_ROOT \
  --n_samples 20 \
  --max_samples 5
```

- **Procedure**: Enable dropout during inference; 20 forward passes; mean prediction + voxel-wise variance
- **Outputs**: Mean segmentation overlay, uncertainty heatmap overlay (saved to `outputs/mc_dropout/overlays/`)
- **Verification**: Computes and prints variance statistics; boundary vs interior variance (uncertainty higher at tumor boundaries)

---

## Reproducibility

- **Random seed**: Fixed via `seed` in config (default: 42). Applied to PyTorch, NumPy, and data splits.
- **Data split**: 85% train / 15% validation, deterministic with fixed generator.
- **Training**: AdamW, cosine LR schedule, mixed precision (AMP). Checkpoints saved at `checkpoints/`.
- **Environment**: Pin versions in `requirements.txt`. For exact reproducibility, use:
  ```bash
  pip freeze > requirements-frozen.txt
  ```

---

## Citation

If you use this code or BraTS data, please cite the following:

### BraTS 2020 Challenge

```bibtex
@misc{Bakas2020,
  title={MICCAI Brain Tumor Segmentation (BraTS) 2020 Benchmark: Prediction of Survival and Pseudoprogression},
  author={Bakas, Spyridon and Menze, Bjoern and Davatzikos, Christos and Kalpathy-Cramer, Jayashree and Farahani, Keyvan and Bilello, Michel and others},
  year={2020},
  publisher={Zenodo},
  url={https://zenodo.org/record/3718903}
}
```

### BraTS Dataset (Required)

```bibtex
@article{Menze2015,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andr{\'a}s and Bauer, Stefan and Kalpathy-Cramer, Jayashree and Farahani, Keyvan and Kirby, Justin and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2015},
  doi={10.1109/TMI.2014.2377694}
}

@article{Bakas2017,
  title={Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features},
  author={Bakas, Spyridon and Reyes, Mauricio and Jakab, Andr{\'a}s and Bauer, Stefan and Rempfler, Markus and Crimi, Alessandro and others},
  journal={Scientific Data},
  volume={4},
  year={2017},
  doi={10.1038/sdata.2017.117}
}

@inproceedings{Bakas2018,
  title={Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge},
  author={Bakas, Spyridon and Reyes, Mauricio and Jakab, Andr{\'a}s and Bauer, Stefan and Rempfler, Markus and Crimi, Alessandro and others},
  booktitle={arXiv preprint arXiv:1811.02629},
  year={2018}
}
```

---

## License

Code: MIT. BraTS data usage is subject to the [official BraTS terms](https://www.med.upenn.edu/cbica/brats2020/data.html).
