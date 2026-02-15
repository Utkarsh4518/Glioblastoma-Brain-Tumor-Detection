"""
Build segmentation model from config.

Uses MONAI UNet. spatial_dims derived from data.mode:
  - data.mode == "h5"  -> 2D UNet (spatial_dims=2)
  - data.mode == "nifti" -> 3D UNet (spatial_dims=3)
"""

from monai.networks.nets import UNet


def build_model(cfg, in_channels=None, out_channels=None):
    """
    Build MONAI UNet for segmentation from config.

    Args:
        cfg: Config with data.mode and model.*.
        in_channels: Override; else cfg.model.in_channels (default 4).
        out_channels: Override; else cfg.model.out_channels (default 4).

    Returns:
        UNet module (2D or 3D by data.mode).
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    mode = data_cfg.get("mode", "nifti")
    spatial_dims = 2 if mode == "h5" else 3
    in_ch = in_channels if in_channels is not None else model_cfg.get("in_channels", 4)
    out_ch = out_channels if out_channels is not None else model_cfg.get("out_channels", 4)
    dropout = model_cfg.get("dropout", 0.2)

    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=dropout,
    )
