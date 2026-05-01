"""Tests for likelihood-ratio estimation used by weighted conformal."""

import numpy as np
import pytest

from thoughtlink.inference.domain_weights import (
    diagnose_weights,
    estimate_likelihood_ratio,
)


class TestEstimateLikelihoodRatio:
    def test_no_shift_weights_near_one(self):
        """When calib and test come from the same distribution, weights ~ 1."""
        rng = np.random.RandomState(0)
        X_calib = rng.normal(size=(400, 5))
        X_test = rng.normal(size=(400, 5))
        w = estimate_likelihood_ratio(X_calib, X_test)
        # Median weight should be near 1.0 -- no shift means no preference.
        assert 0.5 < np.median(w) < 2.0
        # No extreme values either.
        assert w.max() < 5.0

    def test_clear_shift_high_test_density_yields_large_weight(self):
        """Calib points near the test cluster should get larger weights."""
        rng = np.random.RandomState(1)
        # Calib: standard normal centered at 0. Test: shifted to mean +3.
        X_calib = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
        X_test = rng.normal(loc=3.0, scale=1.0, size=(400, 4))
        w = estimate_likelihood_ratio(X_calib, X_test)

        # Calib points whose first coordinate is closer to +3 should have
        # a larger weight than those near 0.
        first_coord = X_calib[:, 0]
        upper_half = w[first_coord > np.median(first_coord)]
        lower_half = w[first_coord <= np.median(first_coord)]
        assert upper_half.mean() > lower_half.mean() * 1.5

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            estimate_likelihood_ratio(np.zeros((10, 4)), np.zeros((10, 5)))

    def test_clip_bounds_weights(self):
        rng = np.random.RandomState(2)
        X_calib = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
        X_test = rng.normal(loc=8.0, scale=1.0, size=(200, 4))
        w = estimate_likelihood_ratio(X_calib, X_test, clip=10.0)
        assert w.max() <= 10.0 + 1e-9
        assert w.min() >= 1.0 / 10.0 - 1e-9

    def test_weights_are_positive(self):
        rng = np.random.RandomState(3)
        X_calib = rng.normal(size=(100, 3))
        X_test = rng.normal(loc=1.5, size=(100, 3))
        w = estimate_likelihood_ratio(X_calib, X_test)
        assert (w > 0).all()


class TestDiagnoseWeights:
    def test_uniform_weights_have_full_ess(self):
        w = np.ones(100)
        d = diagnose_weights(w)
        assert d["effective_sample_size"] == pytest.approx(100.0)
        assert d["ess_ratio"] == pytest.approx(1.0)
        assert d["max"] == 1.0
        assert d["min"] == 1.0

    def test_concentrated_weights_have_low_ess(self):
        w = np.zeros(100)
        w[0] = 100.0  # all mass on one point
        w[1:] = 0.01
        d = diagnose_weights(w)
        # Effective sample size should be near 1.
        assert d["ess_ratio"] < 0.05
