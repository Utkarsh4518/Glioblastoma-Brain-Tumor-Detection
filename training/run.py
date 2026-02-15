"""
Create data loaders, model, loss and run training from config.

Data mode: data.mode = "nifti" (3D volumes) or "h5" (2D slices from HDF5).
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig

from data import build_dataset
from data.dataset import BraTS2020ClassificationDataset
from data.transforms import get_train_transforms_2d
from models.segmentation import (
    get_model as get_seg_model,
    get_segmentation_channels_from_dataset,
    print_model_summary,
)
from models.classification import get_classifier
from models.loss import get_dice_ce_loss, get_bce_with_logits_loss
from training.training import Trainer
from utils.seed import set_seed

logger = logging.getLogger(__name__)


def _get_device(device_str: str) -> torch.device:
    """Resolve device from config: auto, cpu, or cuda. No CUDA calls when device is cpu."""
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available; using CPU")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrap_dict_transform_2d(compose):
    """Adapt MONAI dict-based Compose to (image, mask) -> (image, mask) for H5BraTSDataset."""
    def transform(image, mask):
        out = compose({"image": image, "mask": mask})
        return out["image"], out["mask"]
    return transform


def run_training(cfg: DictConfig) -> dict:
    """
    Run training from Hydra config.

    Expects cfg with paths, data, model, training, task.
    data.mode: "nifti" (default) or "h5" for slice-based HDF5.
    """
    set_seed(cfg.get("seed", 42))

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    data_mode = data_cfg.get("mode", "nifti")
    # data_root: for h5 use data.path, else paths.data_root
    if data_mode == "h5":
        data_root = Path(str(data_cfg.get("path", paths.get("data_root", "./data/h5_brats"))))
        print("Running in HDF5 2D mode")
    else:
        data_root = Path(paths.get("data_root", "./data/brats2020"))

    output_dir = Path(paths.get("output_dir", "./outputs"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", "./checkpoints"))
    experiment_name = cfg.get("experiment_name", "brats2020")

    roi_size = tuple(data_cfg.get("roi_size", [128, 128, 128]))
    batch_size = data_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 4)
    train_val_split = data_cfg.get("train_val_split", 0.85)

    task = cfg.get("task", "segmentation")

    if task == "segmentation":
        train_ds = build_dataset(cfg, "train")
        val_ds = build_dataset(cfg, "val")

        if data_mode == "h5":
            train_transform_2d = _wrap_dict_transform_2d(
                get_train_transforms_2d(keys=["image", "mask"], label_key="mask")
            )

            class _TrainWithTransform(torch.utils.data.Dataset):
                """Applies 2D augmentations only to train samples."""
                def __init__(self, underlying_dataset, indices, transform_fn):
                    self._dataset = underlying_dataset
                    self._indices = indices
                    self._transform_fn = transform_fn

                def __len__(self):
                    return len(self._indices)

                def __getitem__(self, idx):
                    out = self._dataset[self._indices[idx]]
                    img, msk = self._transform_fn(out["image"], out["mask"])
                    return {"image": img, "mask": msk}

            train_ds = _TrainWithTransform(
                train_ds.dataset,
                train_ds.indices,
                train_transform_2d,
            )

        in_channels, out_channels = get_segmentation_channels_from_dataset(train_ds)
        if data_mode == "h5":
            model = get_seg_model(
                name=model_cfg.get("name", "unet_2d"),
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(128, 128),
                dropout=model_cfg.get("dropout", 0.2),
            )
            print_model_summary(model, in_channels=in_channels, out_channels=out_channels, spatial_dims=2)
        else:
            model = get_seg_model(
                name=model_cfg.get("name", "unet"),
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=roi_size,
                dropout=model_cfg.get("dropout", 0.2),
            )
        loss_fn = get_dice_ce_loss(num_classes=4)
        num_classes = 4
    else:
        dataset = BraTS2020ClassificationDataset(
            root=data_root,
            mode="2d",
        )
        n = len(dataset)
        n_train = int(n * train_val_split)
        n_val = n - n_train
        gen = torch.Generator().manual_seed(cfg.get("seed", 42))
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

        backbone = model_cfg.get("backbone") or model_cfg.get("name")
        if backbone not in ("efficientnet_b0", "resnet18"):
            backbone = "efficientnet_b0"
        model = get_classifier(
            backbone=backbone,
            pretrained=model_cfg.get("pretrained", True),
            aggregation=model_cfg.get("aggregation", "none"),
        )
        loss_fn = get_bce_with_logits_loss()
        num_classes = 2

    device = _get_device(train_cfg.get("device", "auto"))

    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        task=task,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        device=device,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        max_epochs=train_cfg.get("max_epochs", 100),
        val_interval=train_cfg.get("val_interval", 1),
        ckpt_interval=train_cfg.get("ckpt_interval", 10),
        scheduler=train_cfg.get("scheduler", "cosine"),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 15),
        num_classes=num_classes,
    )

    return trainer.train()
