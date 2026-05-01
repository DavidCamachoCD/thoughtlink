"""Tests for BrainPolicy and StepResult."""

import numpy as np
import pytest

from thoughtlink.bridge.brain_policy import BrainPolicy, StepResult
from thoughtlink.data.loader import CLASS_NAMES


class _DummyModel:
    """Mock model returning fixed probabilities."""

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, 5))
        probs[:, 0] = 0.8
        probs[:, 4] = 0.2
        return probs


class TestStepResult:
    def test_fields(self):
        result = StepResult(
            timestamp_s=1.0,
            raw_intent="Right Fist",
            stable_intent="Right Fist",
            action="RIGHT",
            confidence=0.8,
            probs=np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
            latency_ms=5.0,
        )
        assert result.timestamp_s == 1.0
        assert result.action == "RIGHT"
        assert result.confidence == 0.8


class TestBrainPolicy:
    @pytest.fixture
    def config(self):
        return {
            "preprocessing": {
                "eeg": {"sfreq": 500.0},
                "window_duration_s": 1.0,
            },
            "inference": {
                "confidence_threshold": 0.6,
                "hysteresis_margin": 0.1,
                "debounce_count": 1,
                "smoother_window": 1,
                "prediction_hz": 2.0,
            },
            "features": {},
        }

    def test_step_produces_valid_result(self, config):
        policy = BrainPolicy(model=_DummyModel(), config=config)
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        result = policy.step(probs)
        assert isinstance(result, StepResult)
        assert result.raw_intent == "Right Fist"
        assert result.action in {"RIGHT", "LEFT", "FORWARD", "STOP"}

    def test_run_on_array(self, config):
        policy = BrainPolicy(model=_DummyModel(), config=config)
        eeg = np.random.randn(1500, 6)
        results = policy.run_on_array(eeg)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_reset(self, config):
        policy = BrainPolicy(model=_DummyModel(), config=config)
        eeg = np.random.randn(1500, 6)
        policy.run_on_array(eeg)
        policy.reset()
        assert len(policy.decoder.buffer) == 0

    def test_on_step_callback(self, config):
        results_captured = []
        policy = BrainPolicy(
            model=_DummyModel(),
            config=config,
            on_step=lambda r: results_captured.append(r),
        )
        eeg = np.random.randn(1500, 6)
        results = policy.run_on_array(eeg)
        assert len(results_captured) == len(results)

    def test_conformal_uncertain_set_holds_action(self, config):
        """If the conformal predictor returns a set of >1 class, the stability
        pipeline must receive a uniform vector and keep the default action."""

        class _UncertainConformal:
            def predict_set(self, probs):
                # Always return an uncertain set with multiple classes.
                return [{0, 1, 2}]

        policy = BrainPolicy(
            model=_DummyModel(),
            config=config,
            conformal=_UncertainConformal(),
        )
        # Even though _DummyModel reports 0.8 for "Right Fist", the uncertain
        # conformal set should force the stability filter to hold "Relax".
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        result = policy.step(probs)
        assert result.stable_intent == "Relax"
        assert result.action == "STOP"

    def test_calibrator_used_as_decoder_model(self, config):
        """When a calibrator is supplied, the decoder routes through it."""

        class _CalibratorWrap:
            def __init__(self, base):
                self._base = base

            def predict_proba(self, X):
                return self._base.predict_proba(X)

        base = _DummyModel()
        wrapped = _CalibratorWrap(base)
        policy = BrainPolicy(model=base, config=config, calibrator=wrapped)
        assert policy.decoder.model is wrapped
