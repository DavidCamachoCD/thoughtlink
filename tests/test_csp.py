"""Tests for CSP feature extraction."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.features.csp_features import CSPFeatureExtractor


class TestCSPFeatureExtractor:
    def test_fit_transform_shape(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 500, 6)
        y = rng.randint(0, 5, 100)
        csp = CSPFeatureExtractor(n_components=4, n_classes=5)
        features = csp.fit_transform(X, y)
        # 5 classes * 4 components = 20
        assert features.shape == (100, 20)

    def test_transform_without_fit_raises(self):
        X = np.random.randn(10, 500, 6)
        csp = CSPFeatureExtractor()
        with pytest.raises(RuntimeError):
            csp.transform(X)

    def test_finite_values(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 500, 6)
        y = rng.randint(0, 2, 50)
        csp = CSPFeatureExtractor(n_components=2, n_classes=2)
        features = csp.fit_transform(X, y)
        assert np.all(np.isfinite(features))

    def test_binary_classification(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 500, 6)
        y = rng.randint(0, 2, 60)
        csp = CSPFeatureExtractor(n_components=3, n_classes=2)
        features = csp.fit_transform(X, y)
        # 2 classes * 3 components = 6
        assert features.shape == (60, 6)

    def test_transform_new_data(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 500, 6)
        y_train = rng.randint(0, 3, 80)
        X_test = rng.randn(20, 500, 6)

        csp = CSPFeatureExtractor(n_components=4, n_classes=3)
        csp.fit(X_train, y_train)
        features = csp.transform(X_test)
        assert features.shape[0] == 20
        assert features.shape[1] == 12  # 3 classes * 4 components
