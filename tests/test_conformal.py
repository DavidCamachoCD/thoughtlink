"""Tests for split-conformal predictors (APS and naive)."""

import numpy as np
import pytest

from thoughtlink.inference.conformal import (
    APSConformalPredictor,
    NaiveConformalPredictor,
    WeightedAPSConformalPredictor,
    _weighted_quantile,
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


class TestWeightedQuantile:
    def test_uniform_weights_match_unweighted(self):
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        weights = np.ones_like(values)
        # 80% weighted quantile of [0.1, 0.3, 0.5, 0.7, 0.9] -> 0.7 or 0.9.
        q = _weighted_quantile(values, weights, 0.8)
        assert q in {0.7, 0.9}

    def test_concentrated_weight_dominates(self):
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # Almost all mass on the lowest value -> any quantile picks the small one.
        weights = np.array([100.0, 0.01, 0.01, 0.01, 0.01])
        q = _weighted_quantile(values, weights, 0.5)
        assert q == 0.1

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError):
            _weighted_quantile(np.array([0.1, 0.5]), np.array([-1.0, 1.0]), 0.5)


class TestWeightedAPSConformal:
    def test_uniform_weights_match_unweighted_aps(self):
        """With identical weights the weighted predictor must agree with APS."""
        probs, y = _well_separated_probs(n=600, sharpness=0.85, seed=10)
        calib_probs, test_probs = probs[:300], probs[300:]
        calib_y, test_y = y[:300], y[300:]

        cp = APSConformalPredictor(alpha=0.1).fit(calib_probs, calib_y)
        weights = np.ones(len(calib_y))
        wcp = WeightedAPSConformalPredictor(alpha=0.1).fit(
            calib_probs, calib_y, weights
        )

        cov_uw = cp.empirical_coverage(test_probs, test_y)
        cov_w = wcp.empirical_coverage(test_probs, test_y)
        # Both should achieve coverage close to 0.9.
        assert abs(cov_uw - cov_w) < 0.05

    def test_recovers_coverage_under_synthetic_shift(self):
        """Calibration biased toward easy samples; test biased toward hard ones.
        Weighted conformal should recover coverage closer to 1 - alpha than the
        unweighted version under this controlled shift.
        """
        rng = np.random.RandomState(11)
        n = 800
        # Build probas where 'easy' samples have a near-one-hot top class
        # and 'hard' samples are near uniform.
        easy_probs, easy_y = _well_separated_probs(
            n=n, sharpness=0.95, seed=11
        )
        hard_probs, hard_y = _well_separated_probs(
            n=n, sharpness=0.50, seed=12
        )

        # Calibration is biased toward easy samples (low non-conformity scores);
        # test is biased toward hard samples (high non-conformity scores).
        calib_probs = np.concatenate([easy_probs[:300], hard_probs[:50]])
        calib_y = np.concatenate([easy_y[:300], hard_y[:50]])
        test_probs = np.concatenate([easy_probs[300:350], hard_probs[50:300]])
        test_y = np.concatenate([easy_y[300:350], hard_y[50:300]])

        # Weights that emphasise the few hard calibration points (matching the
        # test mix). This is the 'oracle' weighted-conformal setting.
        weights = np.ones(len(calib_y))
        weights[300:] = 6.0  # the hard calib points get higher weight

        cp = APSConformalPredictor(alpha=0.1).fit(calib_probs, calib_y)
        wcp = WeightedAPSConformalPredictor(alpha=0.1).fit(
            calib_probs, calib_y, weights
        )
        cov_uw = cp.empirical_coverage(test_probs, test_y)
        cov_w = wcp.empirical_coverage(test_probs, test_y)
        # Weighted should not be worse than unweighted under this shift.
        assert cov_w + 0.01 >= cov_uw

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            WeightedAPSConformalPredictor(alpha=0.0)

    def test_predict_set_before_fit(self):
        with pytest.raises(RuntimeError):
            WeightedAPSConformalPredictor().predict_set(
                np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
            )


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
