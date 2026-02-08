"""Tests for EEGDataset PyTorch wrapper."""

import numpy as np
import pytest
import torch

from thoughtlink.data.dataset import EEGDataset


class TestEEGDataset:
    def setup_method(self):
        self.n_windows = 20
        self.n_samples = 500
        self.n_channels = 6
        self.windows = np.random.randn(
            self.n_windows, self.n_samples, self.n_channels
        ).astype(np.float32)
        self.labels = np.random.randint(0, 5, self.n_windows)
        self.subjects = np.array([0] * 10 + [1] * 10)

    def test_len(self):
        ds = EEGDataset(self.windows, self.labels)
        assert len(ds) == self.n_windows

    def test_getitem_returns_tensor(self):
        ds = EEGDataset(self.windows, self.labels)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(label, int)

    def test_getitem_shape(self):
        ds = EEGDataset(self.windows, self.labels)
        tensor, _ = ds[0]
        # Should be (1, n_channels, n_samples) for EEGNet
        assert tensor.shape == (1, self.n_channels, self.n_samples)

    def test_get_data(self):
        ds = EEGDataset(self.windows, self.labels)
        w, l = ds.get_data()
        np.testing.assert_array_equal(w, self.windows)
        np.testing.assert_array_equal(l, self.labels)

    def test_get_subjects(self):
        ds = EEGDataset(self.windows, self.labels, subjects=self.subjects)
        subj = ds.get_subjects()
        np.testing.assert_array_equal(subj, self.subjects)

    def test_get_subjects_none(self):
        ds = EEGDataset(self.windows, self.labels)
        assert ds.get_subjects() is None

    def test_transform(self):
        def normalize(w):
            return (w - w.mean()) / (w.std() + 1e-8)

        ds = EEGDataset(self.windows, self.labels, transform=normalize)
        tensor, _ = ds[0]
        assert tensor.shape == (1, self.n_channels, self.n_samples)
        # Transformed should differ from original
        raw_ds = EEGDataset(self.windows, self.labels)
        raw_tensor, _ = raw_ds[0]
        assert not torch.allclose(tensor, raw_tensor)

    def test_dataloader_compatible(self):
        from torch.utils.data import DataLoader

        ds = EEGDataset(self.windows, self.labels)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 1, self.n_channels, self.n_samples)
        assert batch_y.shape == (4,)
