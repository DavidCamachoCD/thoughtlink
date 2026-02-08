"""Tests for NIRS feature extraction."""

import numpy as np
import pytest

from thoughtlink.features.nirs_features import (
    compute_nirs_temporal_features,
    fit_nirs_pca,
    transform_nirs_pca,
)


class TestComputeNirsTemporalFeatures:
    def test_output_shape(self):
        nirs_stim = np.random.randn(20, 40)
        result = compute_nirs_temporal_features(nirs_stim)
        assert result.shape == (3 * 40,)

    def test_finite_values(self):
        nirs_stim = np.random.randn(20, 40)
        result = compute_nirs_temporal_features(nirs_stim)
        assert np.all(np.isfinite(result))

    def test_empty_timepoints(self):
        nirs_stim = np.random.randn(0, 40)
        result = compute_nirs_temporal_features(nirs_stim)
        assert result.shape == (120,)
        np.testing.assert_array_equal(result, 0.0)

    def test_single_timepoint(self):
        nirs_stim = np.random.randn(1, 40)
        result = compute_nirs_temporal_features(nirs_stim)
        # Slope portion (last 40 elements) should be zero
        slope = result[80:]
        np.testing.assert_array_equal(slope, 0.0)

    def test_constant_signal_zero_slope(self):
        nirs_stim = np.ones((10, 40)) * 3.0
        result = compute_nirs_temporal_features(nirs_stim)
        slope = result[80:]
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)


class TestNirsPCA:
    def test_fit_and_transform_shape(self):
        features_list = [np.random.randn(120) for _ in range(50)]
        pca, X_reduced = fit_nirs_pca(features_list, n_components=20)
        assert X_reduced.shape == (50, 20)

    def test_transform_shape(self):
        features_list = [np.random.randn(120) for _ in range(50)]
        pca, _ = fit_nirs_pca(features_list, n_components=20)
        single = np.random.randn(120)
        result = transform_nirs_pca(single, pca)
        assert result.shape == (20,)

    def test_n_components_clamped(self):
        features_list = [np.random.randn(120) for _ in range(5)]
        pca, X_reduced = fit_nirs_pca(features_list, n_components=20)
        assert X_reduced.shape[1] <= 5

    def test_finite_values(self):
        features_list = [np.random.randn(120) for _ in range(30)]
        pca, X_reduced = fit_nirs_pca(features_list, n_components=10)
        assert np.all(np.isfinite(X_reduced))
