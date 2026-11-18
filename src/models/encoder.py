"""CNN Encoder that extracts horizontal spatial feature sequences from grayscale images."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CNNEncoder(nn.Module):
    """Convolutional feature extractor preserving left-to-right spatial order.

    Input shape:  [B, 1, H, W]
    Output shape: [B, S, encoder_dim] where S is the horizontal spatial sequence length.
    """

    def __init__(self, in_channels: int = 1, encoder_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim

        self.features = nn.Sequential(
            # Block 1: [B, 1, H, W] -> [B, 32, H/2, W/2]
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: pool height only so horizontal digit order stays fine-grained
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 4: [B, 128, H/8, W/4] -> [B, encoder_dim, H/16, W/8]
            nn.Conv2d(128, encoder_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, images: Tensor) -> Tensor:
        """Encode grayscale images [B, 1, H, W] into spatial sequences [B, S, E]."""
        feat_map = self.features(images)  # [B, E, H', W']
        pooled = self.vertical_pool(feat_map)  # [B, E, 1, W']
        seq = pooled.squeeze(2).permute(0, 2, 1).contiguous()  # [B, W', E]
        return seq
