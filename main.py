import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import CorruptedCIFAR10


def show_batch(clean, corrupted, n=8):
    plt.figure(figsize=(12, 4))

    for i in range(n):
        # clean
        plt.subplot(2, n, i + 1)
        plt.imshow(clean[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.title("Clean")

        # corrupted
        plt.subplot(2, n, n + i + 1)
        plt.imshow(corrupted[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.title("Corrupted")

    plt.tight_layout()
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CorruptedCIFAR10(
        root="./data/raw",
        train=True,
        download=True,
        transform=transform,
        corruption_type="gaussian_noise",
        severity=3,
        corruption_prob=1.0,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    clean, corrupted, labels = next(iter(loader))
    show_batch(clean, corrupted, n=8)


if __name__ == "__main__":
    main()
