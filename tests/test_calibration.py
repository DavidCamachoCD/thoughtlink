"""Tests for calibrators (sklearn, temperature, hierarchical)."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from thoughtlink.inference.calibration import (
    HierarchicalCalibrator,
    SklearnCalibrator,
    TemperatureScaler,
)
from thoughtlink.eval.diagnostics import expected_calibration_error
from thoughtlink.models.hierarchical import HierarchicalClassifier


def _make_classification(n=400, n_features=10, n_classes=5, seed=0, noise=1.0):
    """Synthetic Gaussian-mixture data. `noise` controls separability:
    larger -> harder problem -> base classifier accuracy in (0.4, 0.8) range."""
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(n_classes, n_features))
    y = rng.randint(0, n_classes, size=n)
    X = centers[y] + rng.normal(size=(n, n_features)) * noise
    return X, y


class TestSklearnCalibrator:
    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError):
            SklearnCalibrator(method="bad")

    def test_predict_proba_shape(self):
        X, y = _make_classification()
        X_train, X_calib, X_test = X[:200], X[200:300], X[300:]
        y_train, y_calib, _ = y[:200], y[200:300], y[300:]
        base = LogisticRegression(max_iter=500).fit(X_train, y_train)

        cal = SklearnCalibrator(method="sigmoid").fit(base, X_calib, y_calib)
        probs = cal.predict_proba(X_test)
        assert probs.shape == (X_test.shape[0], 5)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_isotonic_reduces_ece_on_overconfident_predictor(self):
        # Construct an explicitly overconfident base model: a wrapper that
        # sharpens any sklearn estimator's `predict_proba` by raising it to a
        # large power. This ensures pre-calibration ECE is genuinely high so we
        # can verify isotonic moves it down.
        X, y = _make_classification(n=1500, seed=1, noise=2.5)
        X_train, X_calib, X_test = X[:600], X[600:1000], X[1000:]
        y_train, y_calib, y_test = y[:600], y[600:1000], y[1000:]
        base_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)

        class _Sharpen(ClassifierMixin, BaseEstimator):
            def __init__(self, inner=None, power=8.0):
                self.inner = inner
                self.power = power

            def fit(self, X, y):
                self.classes_ = self.inner.classes_
                return self

            def predict_proba(self, X):
                p = np.power(self.inner.predict_proba(X), self.power)
                return p / p.sum(axis=1, keepdims=True)

            def predict(self, X):
                return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        sharp = _Sharpen(inner=base_lr, power=8.0).fit(X_train, y_train)
        ece_pre = expected_calibration_error(y_test, sharp.predict_proba(X_test))
        # Sanity: the sharpener really did break calibration.
        assert ece_pre > 0.05

        cal = SklearnCalibrator(method="isotonic").fit(sharp, X_calib, y_calib)
        ece_post = expected_calibration_error(y_test, cal.predict_proba(X_test))
        assert ece_post < ece_pre

    def test_predict_returns_class_labels(self):
        X, y = _make_classification()
        base = LogisticRegression(max_iter=500).fit(X[:200], y[:200])
        cal = SklearnCalibrator(method="sigmoid").fit(base, X[200:300], y[200:300])
        preds = cal.predict(X[300:310])
        assert preds.shape == (10,)
        assert set(preds.tolist()) <= set(range(5))


class TestTemperatureScaler:
    def test_overconfident_logits_yield_T_above_one(self):
        rng = np.random.RandomState(7)
        n, k = 500, 5
        # Build sharp logits whose argmax matches the true label only ~60% of the time.
        y = rng.randint(0, k, size=n)
        logits = rng.normal(size=(n, k)) * 0.3
        # Force the true class to be the largest 60% of the time, by a wide margin.
        sharpen = rng.binomial(1, 0.6, size=n).astype(bool)
        logits[sharpen, y[sharpen]] += 5.0  # very confident on these
        # The other 40%: keep a strong-but-wrong class.
        wrong_idx = (y + 1) % k
        logits[~sharpen, wrong_idx[~sharpen]] += 5.0
        scaler = TemperatureScaler().fit(logits, y)
        assert scaler.T > 1.0  # overconfident -> needs softening

    def test_predict_proba_normalises(self):
        rng = np.random.RandomState(8)
        logits = rng.normal(size=(20, 5))
        y = rng.randint(0, 5, size=20)
        scaler = TemperatureScaler().fit(logits, y)
        probs = scaler.predict_proba(logits)
        assert probs.shape == (20, 5)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_argmax_invariance(self):
        # Temperature scaling preserves argmax (Guo et al. 2017).
        rng = np.random.RandomState(9)
        logits = rng.normal(size=(50, 5))
        y = rng.randint(0, 5, size=50)
        scaler = TemperatureScaler().fit(logits, y)
        probs = scaler.predict_proba(logits)
        assert np.array_equal(np.argmax(logits, axis=1), np.argmax(probs, axis=1))


class TestHierarchicalCalibrator:
    def _make_hierarchical_data(self, seed=10):
        rng = np.random.RandomState(seed)
        n = 300
        n_features = 12
        # Gaussian per class -> the hierarchical SVCs can learn it.
        means = rng.normal(size=(5, n_features)) * 2.5
        y = np.repeat(np.arange(5), n // 5)
        rng.shuffle(y)
        X = means[y] + rng.normal(size=(n, n_features))
        return X, y

    def test_predict_proba_shape_and_normalised(self):
        X, y = self._make_hierarchical_data()
        X_train, X_calib, X_test = X[:180], X[180:240], X[240:]
        y_train, y_calib, _ = y[:180], y[180:240], y[240:]
        hier = HierarchicalClassifier().fit(X_train, y_train)
        cal = HierarchicalCalibrator(method="sigmoid").fit(hier, X_calib, y_calib)
        probs = cal.predict_proba(X_test)
        assert probs.shape == (X_test.shape[0], 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert (probs >= 0).all()

    def test_calls_fit_first(self):
        cal = HierarchicalCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict_proba(np.zeros((1, 12)))
