from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Compact CNN baseline for 28x28 grayscale medical images."""

    def __init__(self, dropout: float = 0.20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EnhancedCNN(nn.Module):
    """Regularized CNN with batch normalization and global pooling."""

    def __init__(self, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(96, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Small residual block used by ResidualCNN."""

    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualCNN(nn.Module):
    """A slightly stronger residual CNN for repeated benchmark runs.

    This is still intentionally lightweight for Colab, but it provides a more
    serious deep-learning comparator than a single plain CNN stack.
    """

    def __init__(self, dropout: float = 0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            ResidualBlock(64, 0.10),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.stem(x)))


def build_model(name: str = "enhanced_cnn", dropout: float | None = None) -> nn.Module:
    """Factory for model variants used in the experiment table."""
    name = name.lower()
    if name == "tiny_cnn":
        return TinyCNN(dropout=0.20 if dropout is None else dropout)
    if name == "enhanced_cnn":
        return EnhancedCNN(dropout=0.30 if dropout is None else dropout)
    if name == "residual_cnn":
        return ResidualCNN(dropout=0.35 if dropout is None else dropout)
    raise ValueError(f"Unknown model name: {name}")
