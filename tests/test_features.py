"""Tests for feature extraction."""

import numpy as np
import pytest

from thoughtlink.features.eeg_features import (
    compute_band_powers,
    compute_hjorth,
    compute_time_domain,
    extract_window_features,
    extract_features_from_windows,
)
from thoughtlink.features.fusion import fuse_features, fuse_feature_matrices


class TestBandPowers:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_band_powers(window)
        # 4 bands * 6 channels = 24
        assert result.shape == (24,)

    def test_finite_values(self):
        window = np.random.randn(500, 6)
        result = compute_band_powers(window)
        assert np.all(np.isfinite(result))


class TestHjorth:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_hjorth(window)
        # 3 params * 6 channels = 18
        assert result.shape == (18,)

    def test_finite_values(self):
        window = np.random.randn(500, 6)
        result = compute_hjorth(window)
        assert np.all(np.isfinite(result))


class TestTimeDomain:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_time_domain(window)
        # 4 stats * 6 channels = 24
        assert result.shape == (24,)


class TestExtractWindowFeatures:
    def test_default_features(self):
        window = np.random.randn(500, 6)
        result = extract_window_features(window)
        # 24 band power + 18 hjorth = 42
        assert result.shape == (42,)

    def test_with_time_domain(self):
        window = np.random.randn(500, 6)
        result = extract_window_features(window, include_time_domain=True)
        # 24 + 18 + 24 = 66
        assert result.shape == (66,)


class TestExtractFeaturesFromWindows:
    def test_batch_extraction(self):
        windows = np.random.randn(10, 500, 6)
        result = extract_features_from_windows(windows)
        assert result.shape[0] == 10
        assert result.shape[1] == 42  # default features


class TestFusion:
    def test_fuse_features(self):
        eeg = np.random.randn(42)
        nirs = np.random.randn(20)
        result = fuse_features(eeg, nirs)
        assert result.shape == (62,)

    def test_fuse_without_nirs(self):
        eeg = np.random.randn(42)
        result = fuse_features(eeg, None)
        assert result.shape == (42,)

    def test_fuse_matrices(self):
        eeg = np.random.randn(100, 42)
        nirs = np.random.randn(100, 20)
        result = fuse_feature_matrices(eeg, nirs)
        assert result.shape == (100, 62)
