"""
Complete training loop for BraTS segmentation and classification.

Features:
- Mixed precision (torch.cuda.amp)
- Learning rate scheduler (CosineAnnealingLR or ReduceLROnPlateau)
- Model checkpointing and best model saving
- TensorBoard logging
- Seed fixing for reproducibility
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from evaluation.metrics import BRATS_CLASS_NAMES, get_dice_metric, get_postprocessing_transform
from evaluation.classification_metrics import compute_classification_metrics
from utils.seed import set_seed
from utils.early_stopping import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for BraTS segmentation or classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        task: Literal["segmentation", "classification"],
        output_dir: Path,
        checkpoint_dir: Path,
        experiment_name: str = "brats",
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        val_interval: int = 1,
        ckpt_interval: int = 10,
        scheduler: Literal["cosine", "plateau"] = "cosine",
        early_stopping_patience: Optional[int] = 15,
        num_classes: int = 4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.task = task
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.ckpt_interval = ckpt_interval
        self.scheduler_type = scheduler
        self.num_classes = num_classes

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=1e-6,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max" if task == "segmentation" else "max",
                factor=0.5,
                patience=5,
            )

        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        self.best_metric = 0.0
        self.target_key = "mask" if task == "segmentation" else "label"
        self.metric_key = "mean_dice" if task == "segmentation" else "f1"

        self.early_stopping = None
        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                mode="max",
            )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(self.output_dir / "tensorboard" / experiment_name)
        )

        if task == "segmentation":
            self.post_transform = get_postprocessing_transform(num_classes)
            self.dice_metric = get_dice_metric(
                num_classes=num_classes,
                include_background=True,
                reduction="mean_batch",
            )

    def _train_epoch(self, epoch: int) -> float:
        """Single epoch training with mixed precision."""
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch_idx, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device)
            target = batch[self.target_key].to(self.device)

            if self.task == "classification":
                target = target.unsqueeze(1).float()

            self.optimizer.zero_grad()

            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    output = self.model(image)
                    loss = self.loss_fn(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        return avg_loss

    def _validate_segmentation(self) -> tuple[float, dict]:
        """Validation loop for segmentation."""
        self.model.eval()
        self.dice_metric.reset()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                image = batch["image"].to(self.device)
                target = batch["mask"].to(self.device)

                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = self.model(image)
                        loss = self.loss_fn(output, target)
                else:
                    output = self.model(image)
                    loss = self.loss_fn(output, target)

                total_loss += loss.item()
                n += 1

                pred_onehot = self.post_transform(output)
                self.dice_metric(pred_onehot, target)

        avg_loss = total_loss / max(n, 1)
        dice_per_class = self.dice_metric.aggregate()
        dice_np = dice_per_class.cpu().numpy()
        mean_dice = float(dice_np.mean())

        metrics = {
            "val_loss": avg_loss,
            "mean_dice": mean_dice,
            "dice_per_class": dice_np,
        }
        return mean_dice, metrics

    def _validate_classification(self) -> tuple[float, dict]:
        """Validation loop for classification."""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                image = batch["image"].to(self.device)
                target = batch["label"].to(self.device)

                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = self.model(image)
                        loss = self.loss_fn(output.squeeze(-1), target)
                else:
                    output = self.model(image)
                    loss = self.loss_fn(output.squeeze(-1), target)

                total_loss += loss.item()
                n += 1
                all_logits.append(output.cpu())
                all_labels.append(target.cpu())

        avg_loss = total_loss / max(n, 1)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_classification_metrics(logits, labels)
        metrics["val_loss"] = avg_loss

        return metrics["f1"], metrics

    def _validate(self) -> tuple[float, dict]:
        """Run validation and return main metric + full metrics dict."""
        if self.task == "segmentation":
            return self._validate_segmentation()
        return self._validate_classification()

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """Save checkpoint and optionally best model."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / f"{self.experiment_name}_epoch{epoch}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(ckpt, best_path)
            logger.info(f"Saved best model: {best_path}")

    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: dict) -> None:
        """Log to TensorBoard."""
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)

        if self.task == "segmentation":
            self.writer.add_scalar("val/mean_dice", val_metrics["mean_dice"], epoch)
            dice_pc = val_metrics.get("dice_per_class")
            if dice_pc is not None and len(BRATS_CLASS_NAMES) == len(dice_pc):
                for name, val in zip(BRATS_CLASS_NAMES, dice_pc):
                    self.writer.add_scalar(f"val/dice_{name}", float(val), epoch)
        else:
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                self.writer.add_scalar(f"val/{k}", val_metrics[k], epoch)
            cm = val_metrics.get("confusion_matrix")
            if cm is not None:
                tn, fp, fn, tp = cm.ravel()
                self.writer.add_scalar("val/tn", int(tn), epoch)
                self.writer.add_scalar("val/fp", int(fp), epoch)
                self.writer.add_scalar("val/fn", int(fn), epoch)
                self.writer.add_scalar("val/tp", int(tp), epoch)

    def train(self) -> dict:
        """Run full training loop."""
        logger.info(
            f"Starting training: task={self.task}, epochs={self.max_epochs}, "
            f"lr={self.learning_rate}"
        )

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if epoch % self.val_interval == 0:
                metric_value, val_metrics = self._validate()

                self._log_metrics(epoch, train_loss, val_metrics)

                is_best = metric_value > self.best_metric
                if is_best:
                    self.best_metric = metric_value

                dice_str = ""
                if self.task == "segmentation":
                    dice_pc = val_metrics.get("dice_per_class")
                    if dice_pc is not None and len(BRATS_CLASS_NAMES) == len(dice_pc):
                        dice_str = " | " + " | ".join(
                            f"{n}={float(v):.3f}" for n, v in zip(BRATS_CLASS_NAMES, dice_pc)
                        )
                elif self.task == "classification":
                    dice_str = " | ".join(
                        f"{k}={float(val_metrics.get(k, 0)):.3f}"
                        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                    )
                logger.info(
                    f"Epoch {epoch} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} | "
                    f"{self.metric_key}={metric_value:.4f}{dice_str} | lr={current_lr:.2e}"
                )

                if self.scheduler_type == "plateau":
                    self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()

                if epoch % self.ckpt_interval == 0 or is_best:
                    self._save_checkpoint(epoch, val_metrics, is_best)

                if self.early_stopping is not None:
                    if self.early_stopping(metric_value):
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(best {self.metric_key}={self.best_metric:.4f})"
                        )
                        break
            else:
                if self.scheduler_type == "cosine":
                    self.scheduler.step()

        self.writer.close()
        return {"best_metric": self.best_metric}
