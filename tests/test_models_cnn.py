"""Tests for EEGNet CNN model."""

import numpy as np
import pytest
import torch

from thoughtlink.models.cnn import EEGNet


class TestEEGNet:
    def test_output_shape(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        x = torch.randn(4, 1, 6, 500)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_sample(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        x = torch.randn(1, 1, 6, 500)
        out = model(x)
        assert out.shape == (1, 5)

    def test_different_n_classes(self):
        for n_classes in [2, 3, 5, 10]:
            model = EEGNet(n_classes=n_classes)
            x = torch.randn(2, 1, 6, 500)
            out = model(x)
            assert out.shape == (2, n_classes)

    def test_predict_proba_numpy(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        windows = np.random.randn(3, 6, 500).astype(np.float32)
        probs = model.predict_proba_numpy(windows)
        assert probs.shape == (3, 5)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_predict_proba_numpy_transposed_input(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        # (n, samples, channels) -> auto-transposed to (n, channels, samples)
        windows = np.random.randn(3, 500, 6).astype(np.float32)
        probs = model.predict_proba_numpy(windows)
        assert probs.shape == (3, 5)

    def test_predict_proba_numpy_single_window(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        window = np.random.randn(6, 500).astype(np.float32)
        probs = model.predict_proba_numpy(window)
        assert probs.shape == (1, 5)

    def test_eval_mode_deterministic(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        model.eval()
        x = torch.randn(2, 1, 6, 500)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        torch.testing.assert_close(out1, out2)

    def test_parameter_count_compact(self):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 10000, f"Too many parameters: {n_params}"
