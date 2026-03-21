import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL import Image

from src.corruptions import CORRUPTION_FUNCS


class CorruptedCIFAR10(Dataset):
    """
    Returns:
      clean_img, corrupted_img, label
    """

    def __init__(
        self,
        root="./data/raw",
        train=True,
        download=True,
        transform=None,
        corruption_type=None,
        severity=1,
        corruption_prob=1.0,
    ):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform = transform

        self.corruption_type = corruption_type
        self.severity = severity
        self.corruption_prob = corruption_prob

        if corruption_type is not None:
            if corruption_type not in CORRUPTION_FUNCS:
                raise ValueError(
                    f"Unknown corruption_type={corruption_type}. "
                    f"Choose from {list(CORRUPTION_FUNCS.keys())}"
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Ensure PIL image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        clean_img = img

        corrupted_img = img
        if self.corruption_type is not None:
            if torch.rand(1).item() < self.corruption_prob:
                corrupted_img = CORRUPTION_FUNCS[self.corruption_type](
                    img, severity=self.severity
                )

        if self.transform is not None:
            clean_img = self.transform(clean_img)
            corrupted_img = self.transform(corrupted_img)

        return clean_img, corrupted_img, label
