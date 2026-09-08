"""Binary tumor-presence classifier built on a torchvision backbone.

BraTS images have 4 modality channels (T1, T1ce, T2, FLAIR); torchvision
backbones are pretrained on 3-channel ImageNet input, so the first conv
layer is replaced with a 4-input version that reuses the pretrained RGB
weights and initializes the 4th channel as their mean.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

_SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet18")


def _expand_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a new Conv2d with `in_channels` inputs, reusing pretrained weights."""
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        old_weight = conv.weight  # (out, 3, kh, kw)
        new_conv.weight[:, :3] = old_weight
        if in_channels > 3:
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean_weight.expand(-1, in_channels - 3, -1, -1)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def get_classifier(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    in_channels: int = 4,
    aggregation: str = "none",
) -> nn.Module:
    """
    Build a binary classifier with a single-logit output head, shape (N, 1).

    Args:
        backbone: "efficientnet_b0" or "resnet18".
        pretrained: Load ImageNet weights before adapting the input conv.
        in_channels: Number of input channels (BraTS: 4).
        aggregation: Only "none" (per-sample classification) is implemented.
    """
    if backbone not in _SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone!r}; expected one of {_SUPPORTED_BACKBONES}")
    if aggregation != "none":
        raise NotImplementedError(
            f"aggregation={aggregation!r} is not implemented; only 'none' (per-sample) is supported"
        )

    if backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        if in_channels != 3:
            old_conv = model.features[0][0]
            model.features[0][0] = _expand_conv_in_channels(old_conv, in_channels)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    # resnet18
    weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
    model = tvm.resnet18(weights=weights)
    if in_channels != 3:
        model.conv1 = _expand_conv_in_channels(model.conv1, in_channels)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
