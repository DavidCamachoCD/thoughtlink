"""Tests for EEG and NIRS preprocessing."""

import numpy as np
import pytest

from thoughtlink.preprocessing.eeg import preprocess_eeg, create_raw
from thoughtlink.preprocessing.nirs import preprocess_nirs
from thoughtlink.preprocessing.windowing import (
    extract_eeg_windows,
    extract_nirs_stimulus,
    windows_from_samples,
)


class TestEEGPreprocessing:
    def test_preprocess_shape(self):
        eeg = np.random.randn(7499, 6) * 10
        result = preprocess_eeg(eeg)
        assert result.shape == (7499, 6)

    def test_preprocess_finite(self):
        eeg = np.random.randn(7499, 6) * 10
        result = preprocess_eeg(eeg)
        assert np.all(np.isfinite(result))

    def test_create_raw(self):
        eeg = np.random.randn(7499, 6) * 10
        raw = create_raw(eeg)
        assert raw.info["sfreq"] == 500.0
        assert len(raw.ch_names) == 6


class TestNIRSPreprocessing:
    def test_preprocess_shape(self):
        nirs = np.random.randn(72, 40, 3, 2, 3)
        result = preprocess_nirs(nirs, duration=9.0)
        assert result.ndim == 2
        assert result.shape[0] > 0  # has timepoints

    def test_baseline_correction(self):
        nirs = np.ones((72, 40, 3, 2, 3))
        result = preprocess_nirs(nirs, duration=9.0)
        # After baseline correction of constant signal, should be near zero
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


class TestWindowing:
    def test_eeg_window_count(self):
        eeg = np.random.randn(7499, 6)
        windows = extract_eeg_windows(eeg, duration=9.0)
        # 8s of stimulus, 1s windows, 0.5s stride = 15 windows
        assert windows.shape[0] >= 14
        assert windows.shape[1] == 500
        assert windows.shape[2] == 6

    def test_eeg_window_size(self):
        eeg = np.random.randn(7499, 6)
        windows = extract_eeg_windows(
            eeg, duration=9.0, window_duration_s=1.0,
        )
        assert windows.shape[1] == 500  # 1s * 500Hz

    def test_nirs_stimulus_shape(self):
        nirs = np.random.randn(72, 40, 3, 2, 3)
        result = extract_nirs_stimulus(nirs, duration=9.0)
        assert result.ndim == len(nirs.shape)

    def test_windows_from_samples(self):
        samples = [
            {
                "eeg": np.random.randn(7499, 6),
                "label": "Right Fist",
                "duration": 9.0,
                "subject_id": "s1",
            },
            {
                "eeg": np.random.randn(7499, 6),
                "label": "Left Fist",
                "duration": 9.0,
                "subject_id": "s1",
            },
        ]
        X, y, subjects = windows_from_samples(samples)
        assert X.ndim == 3
        assert len(y) == X.shape[0]
        assert len(subjects) == X.shape[0]
        assert set(y) == {0, 1}  # Right Fist=0, Left Fist=1
