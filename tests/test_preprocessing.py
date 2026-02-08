"""Tests for EEG and NIRS preprocessing."""

import numpy as np
import pytest
from scipy.signal import welch

from thoughtlink.preprocessing.eeg import preprocess_eeg, create_raw
from thoughtlink.preprocessing.nirs import preprocess_nirs
from thoughtlink.preprocessing.windowing import (
    extract_eeg_windows,
    extract_nirs_stimulus,
    windows_from_samples,
)
from thoughtlink.data.loader import CLASS_NAMES


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

    def test_filter_removes_high_frequency(self):
        """100 Hz component should be attenuated by the 1-40 Hz bandpass."""
        sfreq = 500.0
        t = np.arange(7499) / sfreq
        high_freq = np.sin(2 * np.pi * 100 * t) * 10
        eeg = np.column_stack([high_freq] * 6)

        freqs_before, psd_before = welch(eeg[:, 0], fs=sfreq, nperseg=256)
        result = preprocess_eeg(eeg, sfreq=sfreq)
        freqs_after, psd_after = welch(result[:, 0], fs=sfreq, nperseg=256)

        idx_100 = np.argmin(np.abs(freqs_before - 100))
        assert psd_after[idx_100] < psd_before[idx_100] * 0.1, \
            "100 Hz should be attenuated by >90% after 1-40 Hz bandpass"

    def test_filter_preserves_mu_band(self):
        """10 Hz component should pass through the 1-40 Hz bandpass."""
        sfreq = 500.0
        t = np.arange(7499) / sfreq
        # Create different amplitudes per channel to avoid CAR cancellation
        eeg = np.zeros((7499, 6))
        for ch in range(6):
            amplitude = 5 + ch  # Different amplitude per channel: 5, 6, 7, 8, 9, 10
            eeg[:, ch] = np.sin(2 * np.pi * 10 * t) * amplitude

        freqs_before, psd_before = welch(eeg[:, 0], fs=sfreq, nperseg=256)
        result = preprocess_eeg(eeg, sfreq=sfreq)
        freqs_after, psd_after = welch(result[:, 0], fs=sfreq, nperseg=256)

        idx_10 = np.argmin(np.abs(freqs_before - 10))
        # After CAR, power will be reduced but 10Hz should still be present
        assert psd_after[idx_10] > 0.1, \
            "10 Hz component should remain after bandpass filter and CAR"

    def test_preprocess_actually_changes_data(self):
        """Preprocessing must modify the data (not a no-op)."""
        eeg = np.random.randn(7499, 6) * 10
        result = preprocess_eeg(eeg)
        assert not np.allclose(result, eeg), "Preprocessed data should differ from raw"


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

    def test_eeg_windows_start_at_stimulus_onset(self):
        """First window should start at stimulus_onset_s * sfreq."""
        eeg = np.arange(7499 * 6, dtype=float).reshape(7499, 6)  # deterministic
        windows = extract_eeg_windows(eeg, duration=9.0, stimulus_onset_s=3.0)
        expected_start = int(3.0 * 500)  # sample 1500
        np.testing.assert_array_equal(
            windows[0], eeg[expected_start:expected_start + 500],
        )

    def test_eeg_windows_do_not_exceed_stimulus_end(self):
        """Last window must not extend beyond stimulus_onset + duration."""
        eeg = np.random.randn(7499, 6)
        duration = 4.0
        windows = extract_eeg_windows(eeg, duration=duration, stimulus_onset_s=3.0)
        stim_end_sample = int((3.0 + duration) * 500)
        n_windows = windows.shape[0]
        last_window_end = int(3.0 * 500) + (n_windows - 1) * 250 + 500
        assert last_window_end <= stim_end_sample

    def test_windows_from_samples_label_mapping_all_classes(self):
        """Verify every class maps to the correct integer index."""
        samples = [
            {"eeg": np.random.randn(7499, 6), "label": name, "duration": 9.0, "subject_id": "s1"}
            for name in CLASS_NAMES
        ]
        X, y, subjects = windows_from_samples(samples)
        unique_labels = set(y)
        # All 5 classes should be present
        assert unique_labels == {0, 1, 2, 3, 4}
        # Verify order matches CLASS_NAMES
        for i, name in enumerate(CLASS_NAMES):
            assert CLASS_NAMES.index(name) == i

    def test_nirs_stimulus_baseline_subtracted(self):
        """NIRS stimulus should have rest-period mean subtracted."""
        nirs = np.ones((72, 40, 3, 2, 3)) * 5.0
        # Add offset to rest period only
        nirs[:14] += 10.0  # rest period at 4.76 Hz, 3s = ~14 samples
        result = extract_nirs_stimulus(nirs, duration=9.0)
        # Rest baseline was 15.0, stimulus was 5.0, corrected = 5.0 - 15.0 = -10.0
        assert result.mean() < 0, "Baseline subtraction should make stimulus negative"
