"""Helpers for building/inspecting segmentation models."""

import logging

logger = logging.getLogger(__name__)

# Fixed label-space sizes per dataset backend: a single sample's mask.max()
# is not reliable (a slice/volume may lack a rarer class, e.g. no enhancing
# tumor), which would silently shrink out_channels and break the loss/metric.
_KNOWN_NUM_CLASSES = {
    "NiftiBraTSDataset": 4,  # BraTS: background, NCR/NET, ED, ET
    "H5BraTSDataset": 2,  # binary tumor / not-tumor
}


def _unwrap_dataset(dataset):
    """Walk through torch Subset / custom wrapper layers to the base dataset."""
    seen = set()
    while hasattr(dataset, "dataset") or hasattr(dataset, "_dataset"):
        if id(dataset) in seen:
            break
        seen.add(id(dataset))
        dataset = getattr(dataset, "dataset", None) or getattr(dataset, "_dataset")
    return dataset


def get_segmentation_channels_from_dataset(dataset) -> tuple[int, int]:
    """
    Infer (in_channels, out_channels) for a segmentation dataset.

    in_channels comes from one sample's image channel dimension. out_channels
    uses a fixed label-space size for known dataset classes (a single
    sample's mask may not contain every class); falls back to sample-based
    inference for unrecognized dataset types.
    """
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    in_channels = int(image.shape[0])

    base = _unwrap_dataset(dataset)
    class_name = type(base).__name__ if base is not None else None
    if class_name in _KNOWN_NUM_CLASSES:
        out_channels = _KNOWN_NUM_CLASSES[class_name]
    else:
        out_channels = max(int(mask.max().item()) + 1, 2)
    return in_channels, out_channels


def print_model_summary(model, in_channels: int, out_channels: int, spatial_dims: int) -> None:
    """Log parameter count and I/O configuration for a segmentation model."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: spatial_dims={spatial_dims}, in_channels={in_channels}, "
        f"out_channels={out_channels}, params={n_params:,} (trainable={n_trainable:,})"
    )
