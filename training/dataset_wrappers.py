"""Dataset wrappers used by both the training entry point and notebooks."""

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    """
    Applies a MONAI dict-transform (Compose expecting/returning
    {"image", "mask"}) to each item of a base segmentation dataset or Subset.

    Kept separate from the datasets themselves so validation data can reuse
    the same base without augmentation.
    """

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        out = self.base[idx]
        out = self.transform({"image": out["image"], "mask": out["mask"]})
        return {"image": out["image"], "mask": out["mask"]}
