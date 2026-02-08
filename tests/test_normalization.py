"""Tests for subject-level normalization."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.features.normalization import normalize_features_by_subject


class TestNormalizeBySubject:
    def test_output_shape(self):
        X = np.random.randn(100, 42)
        subjects = ["s1"] * 50 + ["s2"] * 50
        X_norm, stats = normalize_features_by_subject(X, subjects)
        assert X_norm.shape == (100, 42)
        assert len(stats) == 2

    def test_zero_mean_per_subject(self):
        X = np.random.randn(100, 10) + 10.0
        subjects = ["s1"] * 50 + ["s2"] * 50
        X_norm, _ = normalize_features_by_subject(X, subjects)
        for subj in ["s1", "s2"]:
            mask = np.array(subjects) == subj
            np.testing.assert_allclose(
                X_norm[mask].mean(axis=0), 0.0, atol=1e-10
            )

    def test_unit_variance_per_subject(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10) * 5.0
        subjects = ["s1"] * 100 + ["s2"] * 100
        X_norm, _ = normalize_features_by_subject(X, subjects)
        for subj in ["s1", "s2"]:
            mask = np.array(subjects) == subj
            np.testing.assert_allclose(
                X_norm[mask].std(axis=0), 1.0, atol=0.05
            )

    def test_precomputed_stats(self):
        X = np.random.randn(50, 10)
        subjects = ["s1"] * 50
        stats = {"s1": (np.zeros(10), np.ones(10))}
        X_norm, _ = normalize_features_by_subject(X, subjects, stats=stats)
        np.testing.assert_array_equal(X_norm, X)

    def test_single_subject(self):
        X = np.random.randn(30, 5) + 100.0
        subjects = ["s1"] * 30
        X_norm, stats = normalize_features_by_subject(X, subjects)
        assert "s1" in stats
        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)
