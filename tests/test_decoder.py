"""Tests for RealtimeDecoder."""

import numpy as np
import pytest

from thoughtlink.inference.decoder import RealtimeDecoder


class _DummyModel:
    """Mock model that returns uniform probabilities."""

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.ones((n_samples, 5)) / 5.0


class TestRealtimeDecoder:
    def test_predict_returns_none_when_buffer_empty(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
        )
        probs, latency = decoder.predict()
        assert probs is None
        assert latency == 0.0

    def test_predict_returns_none_when_buffer_too_short(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
            window_size_s=1.0,
            sfreq=500.0,
        )
        decoder.feed_samples(np.random.randn(100, 6))
        probs, latency = decoder.predict()
        assert probs is None

    def test_predict_returns_probs_when_enough_data(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
            window_size_s=1.0,
            sfreq=500.0,
        )
        decoder.feed_samples(np.random.randn(500, 6))
        probs, latency = decoder.predict()
        assert probs is not None
        assert probs.shape == (5,)
        assert latency > 0.0

    def test_predict_latency_positive(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
        )
        decoder.feed_samples(np.random.randn(500, 6))
        _, latency = decoder.predict()
        assert latency > 0.0

    def test_clear_resets_buffer(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
        )
        decoder.feed_samples(np.random.randn(500, 6))
        decoder.clear()
        assert len(decoder.buffer) == 0
        probs, _ = decoder.predict()
        assert probs is None

    def test_buffer_rolls_over(self):
        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
            buffer_duration_s=2.0,
            sfreq=500.0,
        )
        # Feed 3 seconds worth — buffer only holds 2s
        decoder.feed_samples(np.random.randn(1500, 6))
        assert len(decoder.buffer) == 1000  # 2s * 500Hz

    def test_preprocessor_is_called(self):
        called = {"count": 0}

        def mock_preproc(w):
            called["count"] += 1
            return w

        decoder = RealtimeDecoder(
            model=_DummyModel(),
            feature_extractor=lambda w: np.random.randn(42),
            preprocessor=mock_preproc,
        )
        decoder.feed_samples(np.random.randn(500, 6))
        decoder.predict()
        assert called["count"] == 1
