"""Tests for calibration diagnostics (ECE, MCE, Brier, reliability_curve)."""

import numpy as np
import pytest

from thoughtlink.inference.diagnostics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)


class TestECE:
    def test_perfectly_calibrated_is_zero(self):
        # Each row is one-hot on the true class -> confidence 1.0, accuracy 1.0.
        n = 100
        y = np.random.RandomState(0).randint(0, 5, size=n)
        probs = np.zeros((n, 5))
        probs[np.arange(n), y] = 1.0
        assert expected_calibration_error(y, probs) == pytest.approx(0.0)

    def test_overconfident_uniform_predictor(self):
        # Model always predicts class 0 with prob 0.9 but accuracy is 1/5.
        n = 500
        y = np.random.RandomState(0).randint(0, 5, size=n)
        probs = np.full((n, 5), 0.025)
        probs[:, 0] = 0.9
        ece = expected_calibration_error(y, probs)
        # Confidence ~0.9, accuracy ~0.2 -> ECE ~0.7.
        assert 0.6 < ece < 0.8

    def test_in_zero_one_range(self):
        rng = np.random.RandomState(1)
        n = 200
        y = rng.randint(0, 5, size=n)
        probs = rng.dirichlet(np.ones(5), size=n)
        ece = expected_calibration_error(y, probs)
        assert 0.0 <= ece <= 1.0


class TestMCE:
    def test_perfectly_calibrated_is_zero(self):
        n = 100
        y = np.random.RandomState(0).randint(0, 5, size=n)
        probs = np.zeros((n, 5))
        probs[np.arange(n), y] = 1.0
        assert maximum_calibration_error(y, probs) == pytest.approx(0.0)

    def test_at_least_as_large_as_ece(self):
        rng = np.random.RandomState(2)
        n = 300
        y = rng.randint(0, 5, size=n)
        probs = rng.dirichlet(np.ones(5), size=n)
        ece = expected_calibration_error(y, probs)
        mce = maximum_calibration_error(y, probs)
        assert mce + 1e-9 >= ece


class TestBrier:
    def test_perfect_one_hot_is_zero(self):
        n = 50
        y = np.array([0, 1, 2, 3, 4] * 10)
        probs = np.zeros((n, 5))
        probs[np.arange(n), y] = 1.0
        assert brier_score(y, probs) == pytest.approx(0.0)

    def test_uniform_predictor_value(self):
        n = 100
        y = np.random.RandomState(0).randint(0, 5, size=n)
        probs = np.full((n, 5), 0.2)
        # Per row: (1 - 0.2)^2 + 4 * 0.2^2 = 0.64 + 0.16 = 0.80.
        assert brier_score(y, probs) == pytest.approx(0.8)

    def test_non_negative(self):
        rng = np.random.RandomState(3)
        n = 200
        y = rng.randint(0, 5, size=n)
        probs = rng.dirichlet(np.ones(5), size=n)
        assert brier_score(y, probs) >= 0.0


class TestReliabilityCurve:
    def test_shape_and_keys(self):
        rng = np.random.RandomState(4)
        n = 200
        y = rng.randint(0, 5, size=n)
        probs = rng.dirichlet(np.ones(5), size=n)
        out = reliability_curve(y, probs, n_bins=10)
        assert set(out.keys()) == {
            "bin_edges", "bin_centers", "bin_counts", "bin_confidence", "bin_accuracy",
        }
        assert out["bin_edges"].shape == (11,)
        assert out["bin_centers"].shape == (10,)
        assert out["bin_counts"].shape == (10,)
        assert out["bin_confidence"].shape == (10,)
        assert out["bin_accuracy"].shape == (10,)

    def test_counts_sum_to_n(self):
        rng = np.random.RandomState(5)
        n = 175
        y = rng.randint(0, 5, size=n)
        probs = rng.dirichlet(np.ones(5), size=n)
        out = reliability_curve(y, probs, n_bins=15)
        assert out["bin_counts"].sum() == n
