"""EEG data augmentation for CNN training.

All augmentations operate on EEG tensors and are applied
during training only (not at test time).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGAugmentationDataset(Dataset):
    """PyTorch Dataset with on-the-fly EEG augmentation.

    Augmentations:
    1. Gaussian noise addition
    2. Time shifting (circular shift)
    3. Channel dropout (zero out random channels)
    4. Amplitude scaling

    Args:
        X: Windows, shape (n_windows, n_channels, n_samples) -- already transposed.
        y: Labels, shape (n_windows,).
        augment: Whether to apply augmentations.
        noise_std: Std of Gaussian noise relative to signal std.
        max_shift: Maximum time shift in samples.
        channel_drop_prob: Probability of dropping each channel.
        scale_range: (min_scale, max_scale) for amplitude scaling.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = True,
        noise_std: float = 0.1,
        max_shift: int = 50,
        channel_drop_prob: float = 0.1,
        scale_range: tuple[float, float] = (0.8, 1.2),
    ):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (n, 1, ch, t)
        self.y = torch.from_numpy(y).long()
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.channel_drop_prob = channel_drop_prob
        self.scale_range = scale_range

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()  # (1, n_channels, n_samples)
        y = self.y[idx]

        if self.augment:
            x = self._apply_augmentations(x)

        return x, y

    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.noise_std * x.std()
            x = x + noise

        # 2. Time shift (circular)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)

        # 3. Channel dropout
        if torch.rand(1).item() < 0.3:
            n_channels = x.shape[1]
            drop_mask = (torch.rand(1, n_channels, 1) > self.channel_drop_prob).float()
            x = x * drop_mask

        # 4. Amplitude scaling
        if torch.rand(1).item() < 0.5:
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * scale

        return x
