"""EEGNet-inspired compact CNN for EEG classification.

Based on Lawhern et al., 2018 - designed specifically for BCI tasks.
Extremely compact (~2-4K parameters) for fast inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Compact CNN for EEG classification.

    Architecture:
    1. Temporal convolution (learns frequency filters)
    2. Depthwise spatial convolution (learns spatial filters per temporal feature)
    3. Separable convolution (combines features)
    4. Fully connected classifier

    Input shape: (batch, 1, n_channels, n_samples)
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 6,
        n_samples: int = 500,
        f1: int = 8,
        f2: int = 16,
        d: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Block 1: Temporal + Spatial
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f1 * d, (n_channels, 1),
                               groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.conv3 = nn.Conv2d(f1 * d, f1 * d, (1, 16),
                               padding=(0, 8), groups=f1 * d, bias=False)
        self.conv4 = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self._forward_features(dummy)
            flatten_size = dummy.shape[1]

        self.fc = nn.Linear(flatten_size, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.fc(x)

    def predict_proba_numpy(self, windows: "np.ndarray") -> "np.ndarray":
        """Convenience method for inference from numpy arrays.

        Args:
            windows: Shape (n_windows, n_channels, n_samples) or
                     (n_windows, n_samples, n_channels).

        Returns:
            Probability matrix (n_windows, n_classes).
        """
        import numpy as np

        self.eval()
        with torch.no_grad():
            if windows.ndim == 2:
                windows = windows[np.newaxis]
            if windows.shape[-1] < windows.shape[-2]:
                # (n, samples, channels) -> (n, channels, samples)
                windows = np.transpose(windows, (0, 2, 1))
            x = torch.from_numpy(windows).float().unsqueeze(1)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).numpy()
        return probs
