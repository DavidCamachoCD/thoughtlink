"""PyTorch Dataset wrapper for ThoughtLink EEG data."""

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG windows.

    Wraps numpy arrays for use with PyTorch DataLoader.
    Also provides sklearn-compatible access via get_data().

    Args:
        windows: Shape (n_windows, n_samples, n_channels).
        labels: Integer class labels (n_windows,).
        subjects: Optional subject IDs (n_windows,).
        transform: Optional callable applied to each window.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray | None = None,
        transform: Callable | None = None,
    ):
        self.windows = windows
        self.labels = labels
        self.subjects = subjects
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single sample as PyTorch tensors.

        Returns:
            (window_tensor, label) where window_tensor has shape
            (1, n_channels, n_samples) suitable for EEGNet input.
        """
        window = self.windows[idx]

        if self.transform is not None:
            window = self.transform(window)

        # (n_samples, n_channels) -> (1, n_channels, n_samples)
        if window.shape[-1] < window.shape[-2]:
            window = window.T
        tensor = torch.from_numpy(window).float().unsqueeze(0)

        return tensor, int(self.labels[idx])

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all data as numpy arrays (sklearn-compatible).

        Returns:
            (windows, labels) as numpy arrays.
        """
        return self.windows, self.labels

    def get_subjects(self) -> np.ndarray | None:
        """Get subject IDs if available."""
        return self.subjects
