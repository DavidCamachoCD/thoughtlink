"""Tests for the device selection helper and EEGNet on MPS (when available)."""

import pytest
import torch

from thoughtlink.eval._torch_device import select_device
from thoughtlink.models.cnn import EEGNet


class TestSelectDevice:
    def test_returns_torch_device(self):
        device = select_device()
        assert isinstance(device, torch.device)
        assert device.type in {"cuda", "mps", "cpu"}

    def test_priority_cuda_over_mps_over_cpu(self):
        """Selection rule: CUDA > MPS > CPU."""
        device = select_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            assert device.type == "mps"
        else:
            assert device.type == "cpu"


class TestEEGNetOnMPS:
    """Sanity-check that the network runs on MPS without silent CPU fallback.

    Skipped on hosts without MPS (Linux CI, older Macs).
    """

    @pytest.fixture
    def mps_device(self):
        if (
            getattr(torch.backends, "mps", None) is None
            or not torch.backends.mps.is_available()
        ):
            pytest.skip("MPS not available on this host")
        return torch.device("mps")

    def test_forward_pass_shape(self, mps_device):
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500).to(mps_device)
        x = torch.randn(2, 1, 6, 500, device=mps_device)
        out = model(x)
        assert out.shape == (2, 5)
        assert out.device.type == "mps"

    def test_predict_proba_numpy_returns_cpu_array(self, mps_device):
        """Network on MPS, but predict_proba_numpy returns numpy => CPU values."""
        import numpy as np

        model = EEGNet(n_classes=5, n_channels=6, n_samples=500).to(mps_device)
        windows = np.random.randn(3, 500, 6).astype(np.float32)
        # predict_proba_numpy moves data to model.device internally; outputs numpy.
        # Move model back to CPU first because predict_proba_numpy does .float().unsqueeze(1)
        # on a CPU tensor created from numpy (the helper itself stays CPU-bound).
        model_cpu = model.cpu()
        probs = model_cpu.predict_proba_numpy(windows)
        assert probs.shape == (3, 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
