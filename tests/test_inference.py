"""Tests for inference pipeline: confidence filter, smoother, decoder."""

import numpy as np
import pytest

from thoughtlink.inference.confidence import (
    IntentConfidenceFilter,
    MajorityVotingSmoother,
    StabilityPipeline,
)


CLASSES = ["Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"]


class TestIntentConfidenceFilter:
    def test_high_confidence_accepted(self):
        f = IntentConfidenceFilter(
            confidence_threshold=0.6,
            hysteresis_margin=0.1,
            debounce_count=1,
        )
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        result = f.update(probs, CLASSES)
        assert result == "Right Fist"

    def test_low_confidence_stays_relax(self):
        f = IntentConfidenceFilter(confidence_threshold=0.6, debounce_count=1)
        probs = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
        result = f.update(probs, CLASSES)
        assert result == "Relax"  # default

    def test_debounce_prevents_immediate_switch(self):
        f = IntentConfidenceFilter(
            confidence_threshold=0.5,
            hysteresis_margin=0.0,
            debounce_count=3,
        )
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])

        # First two calls: still debouncing
        assert f.update(probs, CLASSES) == "Relax"
        assert f.update(probs, CLASSES) == "Relax"
        # Third call: debounce satisfied
        assert f.update(probs, CLASSES) == "Right Fist"

    def test_hysteresis_requires_higher_confidence_to_switch(self):
        f = IntentConfidenceFilter(
            confidence_threshold=0.5,
            hysteresis_margin=0.15,
            debounce_count=1,
        )
        # First set current action to Right Fist
        probs_right = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        f.update(probs_right, CLASSES)

        # Try to switch with moderate confidence - threshold to switch is 0.65
        probs_left = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
        result = f.update(probs_left, CLASSES)
        # Should stay Right Fist (0.6 < 0.65 threshold for switching)
        assert result == "Right Fist"

    def test_reset(self):
        f = IntentConfidenceFilter(debounce_count=1)
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        f.update(probs, CLASSES)
        f.reset()
        assert f.current_action == "Relax"


class TestMajorityVotingSmoother:
    def test_majority_vote(self):
        s = MajorityVotingSmoother(window_size=3)
        assert s.smooth("Right Fist") == "Right Fist"
        assert s.smooth("Right Fist") == "Right Fist"
        assert s.smooth("Left Fist") == "Right Fist"  # 2 vs 1

    def test_switch_after_majority(self):
        s = MajorityVotingSmoother(window_size=3)
        s.smooth("Right Fist")
        s.smooth("Left Fist")
        s.smooth("Left Fist")
        assert s.smooth("Left Fist") == "Left Fist"

    def test_reset(self):
        s = MajorityVotingSmoother(window_size=3)
        s.smooth("Right Fist")
        s.reset()
        assert len(s.window) == 0


class TestStabilityPipeline:
    def test_end_to_end(self):
        pipeline = StabilityPipeline(
            confidence_threshold=0.5,
            hysteresis_margin=0.0,
            debounce_count=1,
            smoother_window=1,
        )
        probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        result = pipeline.process(probs, CLASSES)
        assert result == "Right Fist"

    def test_stable_output_under_noise(self):
        pipeline = StabilityPipeline(
            confidence_threshold=0.6,
            hysteresis_margin=0.1,
            debounce_count=3,
            smoother_window=5,
        )
        # Send consistent Right Fist signals
        for _ in range(10):
            probs = np.array([0.75, 0.05, 0.05, 0.05, 0.1])
            pipeline.process(probs, CLASSES)

        # One noisy prediction shouldn't change output
        noisy_probs = np.array([0.3, 0.3, 0.1, 0.1, 0.2])
        result = pipeline.process(noisy_probs, CLASSES)
        assert result == "Right Fist"
