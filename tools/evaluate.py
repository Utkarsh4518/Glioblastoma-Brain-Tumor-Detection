"""
Evaluate a trained checkpoint and generate thesis figures.

Segmentation (per-class Dice + HD95, MRI/mask overlays):
    python tools/evaluate.py --task segmentation --checkpoint checkpoints/brats2020_default_best.pt
    python tools/evaluate.py --task segmentation --mode h5 --checkpoint <ckpt>

Classification (metrics, confusion matrix, ROC curve):
    python tools/evaluate.py --task classification --backbone resnet18 --checkpoint <ckpt>

DATA_ROOT env var (or --data-root) points at the dataset. Without --checkpoint
a randomly initialized model is used (wiring smoke test only).
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, random_split


def _get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model, checkpoint: str | None, device) -> None:
    if not checkpoint:
        print("WARNING: no --checkpoint given; using randomly initialized weights.")
        return
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded checkpoint {checkpoint} (epoch {epoch})")


def _build_cfg(args) -> dict:
    return {
        "data": {
            "mode": args.mode,
            "path": args.data_root,
            "roi_size": args.roi_size,
            "train_val_split": args.train_val_split,
            "pattern": "volume_*.h5",
            "image_key": None,
            "mask_key": None,
        },
        "paths": {"data_root": args.data_root},
        "model": {"in_channels": 4, "out_channels": 4, "dropout": 0.2},
        "seed": args.seed,
    }


def _evaluate_segmentation(args, device) -> None:
    from data import build_dataset
    from models.build_model import build_model
    from models.segmentation import get_segmentation_channels_from_dataset
    from evaluation.segmentation_eval import evaluate_segmentation

    cfg = _build_cfg(args)
    val_ds = build_dataset(cfg, "val")
    in_ch, out_ch = get_segmentation_channels_from_dataset(val_ds)
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch)
    _load_checkpoint(model, args.checkpoint, device)

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate_segmentation(
        model, loader, device, num_classes=out_ch, out_dir=args.out_dir, num_overlays=args.num_overlays
    )
    print("\nSegmentation metrics:")
    print(f"  mean Dice (foreground): {metrics['mean_dice_foreground']}")
    print(f"  mean HD95 (foreground): {metrics['mean_hd95_foreground']}")
    for name, val in metrics["dice_per_class"].items():
        print(f"    Dice[{name}]: {val}")
    print(f"  Figures + metrics.json written to {args.out_dir}")


def _evaluate_classification(args, device) -> None:
    from data.dataset import BraTS2020ClassificationDataset
    from models.classification import get_classifier
    from evaluation.classification_eval import evaluate_classification

    dataset = BraTS2020ClassificationDataset(root=args.data_root, mode="2d")
    n = len(dataset)
    n_train = int(n * args.train_val_split)
    gen = torch.Generator().manual_seed(args.seed)
    _, val_ds = random_split(dataset, [n_train, n - n_train], generator=gen)

    backbone = args.backbone if args.backbone in ("efficientnet_b0", "resnet18") else "efficientnet_b0"
    model = get_classifier(backbone=backbone, pretrained=False)
    _load_checkpoint(model, args.checkpoint, device)

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate_classification(model, loader, device, out_dir=args.out_dir)
    print("\nClassification metrics:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k}: {float(metrics[k]):.4f}")
    print(f"  Figures + metrics.json written to {args.out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained BraTS checkpoint.")
    p.add_argument("--task", choices=["segmentation", "classification"], required=True)
    p.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint (optional).")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "").strip() or None)
    p.add_argument("--mode", choices=["nifti", "h5"], default="nifti", help="Segmentation data mode.")
    p.add_argument("--backbone", default="efficientnet_b0", help="Classification backbone.")
    p.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<task>_eval).")
    p.add_argument("--num-overlays", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--train-val-split", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    if not args.data_root:
        print("ERROR: set DATA_ROOT or pass --data-root.")
        return 1
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.task}_eval"

    device = _get_device(args.device)
    print(f"Device: {device}")

    if args.task == "segmentation":
        _evaluate_segmentation(args, device)
    else:
        _evaluate_classification(args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
