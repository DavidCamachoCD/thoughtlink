"""Tests for split-conformal predictors (APS and naive)."""

import numpy as np
import pytest

from thoughtlink.inference.conformal import (
    APSConformalPredictor,
    NaiveConformalPredictor,
)


def _well_separated_probs(n=400, n_classes=5, sharpness=0.85, seed=0):
    """Return (probs, y) where the model is correct ~`sharpness` of the time
    and assigns most of its mass to a single class, with small noise on others.
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, size=n)
    correct = rng.binomial(1, sharpness, size=n).astype(bool)
    pred = np.where(correct, y, (y + 1) % n_classes)
    eps = rng.uniform(0.02, 0.06, size=(n, n_classes))
    eps[np.arange(n), pred] += 0.7
    probs = eps / eps.sum(axis=1, keepdims=True)
    return probs, y


class TestAPSConformal:
    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            APSConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError):
            APSConformalPredictor(alpha=1.0)

    def test_predict_set_before_fit(self):
        with pytest.raises(RuntimeError):
            APSConformalPredictor().predict_set(np.array([[0.2, 0.2, 0.2, 0.2, 0.2]]))

    def test_empirical_coverage_meets_guarantee(self):
        probs, y = _well_separated_probs(n=600, sharpness=0.85, seed=1)
        calib_probs, test_probs = probs[:300], probs[300:]
        calib_y, test_y = y[:300], y[300:]
        cp = APSConformalPredictor(alpha=0.1).fit(calib_probs, calib_y)
        # Marginal coverage guarantee is 1 - alpha (= 0.9). Allow ~3pt slack
        # for finite-sample noise on n_test = 300.
        coverage = cp.empirical_coverage(test_probs, test_y)
        assert coverage >= 0.87

    def test_set_never_empty(self):
        probs, y = _well_separated_probs(n=200, seed=2)
        cp = APSConformalPredictor(alpha=0.1).fit(probs[:100], y[:100])
        sets = cp.predict_set(probs[100:])
        assert all(len(s) >= 1 for s in sets)

    def test_singleton_for_confident_predictions(self):
        probs, y = _well_separated_probs(n=400, sharpness=0.95, seed=3)
        cp = APSConformalPredictor(alpha=0.1).fit(probs[:200], y[:200])
        # A near-one-hot probability should give a singleton set.
        sharp = np.array([[0.97, 0.01, 0.01, 0.005, 0.005]])
        sets = cp.predict_set(sharp)
        assert len(sets[0]) == 1

    def test_set_grows_under_uncertainty(self):
        probs, y = _well_separated_probs(n=400, sharpness=0.8, seed=4)
        cp = APSConformalPredictor(alpha=0.1).fit(probs[:200], y[:200])
        # Uniform probability vector -> set should include multiple classes.
        uniform = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        sets = cp.predict_set(uniform)
        assert len(sets[0]) > 1


class TestNaiveConformal:
    def test_empirical_coverage_meets_guarantee(self):
        probs, y = _well_separated_probs(n=600, sharpness=0.85, seed=5)
        cp = NaiveConformalPredictor(alpha=0.1).fit(probs[:300], y[:300])
        coverage = cp.empirical_coverage(probs[300:], y[300:])
        assert coverage >= 0.87

    def test_set_never_empty(self):
        probs, y = _well_separated_probs(n=200, seed=6)
        cp = NaiveConformalPredictor(alpha=0.1).fit(probs[:100], y[:100])
        sets = cp.predict_set(probs[100:])
        assert all(len(s) >= 1 for s in sets)
