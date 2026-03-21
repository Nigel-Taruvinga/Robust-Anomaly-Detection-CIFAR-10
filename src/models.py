import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Simple convolutional autoencoder for CIFAR-10 (3x32x32)
    """

    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder: 3x32x32 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 32x16x16
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x8x8
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 128x4x4
            nn.ReLU(True),

            nn.Flatten(),                               # 128*4*4
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        # Decoder: latent -> 3x32x32
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(True),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x8x8
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x16x16
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 3x32x32
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder_conv(x)
        return x
