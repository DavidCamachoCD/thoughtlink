"""Tests for feature extraction."""

import numpy as np
import pytest

from thoughtlink.features.eeg_features import (
    BANDS,
    compute_band_powers,
    compute_hjorth,
    compute_time_domain,
    compute_wavelet_features,
    _wavelet_entropy,
    extract_window_features,
    extract_features_from_windows,
)
from thoughtlink.features.fusion import fuse_features, fuse_feature_matrices


class TestBandPowers:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_band_powers(window)
        # 4 bands * 6 channels = 24
        assert result.shape == (24,)

    def test_finite_values(self):
        window = np.random.randn(500, 6)
        result = compute_band_powers(window)
        assert np.all(np.isfinite(result))


class TestHjorth:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_hjorth(window)
        # 3 params * 6 channels = 18
        assert result.shape == (18,)

    def test_finite_values(self):
        window = np.random.randn(500, 6)
        result = compute_hjorth(window)
        assert np.all(np.isfinite(result))


class TestTimeDomain:
    def test_output_shape(self):
        window = np.random.randn(500, 6)
        result = compute_time_domain(window)
        # 4 stats * 6 channels = 24
        assert result.shape == (24,)


class TestExtractWindowFeatures:
    def test_default_features(self):
        window = np.random.randn(500, 6)
        result = extract_window_features(window)
        # 24 band power + 18 hjorth = 42
        assert result.shape == (42,)

    def test_with_time_domain(self):
        window = np.random.randn(500, 6)
        result = extract_window_features(window, include_time_domain=True)
        # 24 + 18 + 24 = 66
        assert result.shape == (66,)


class TestExtractFeaturesFromWindows:
    def test_batch_extraction(self):
        windows = np.random.randn(10, 500, 6)
        result = extract_features_from_windows(windows)
        assert result.shape[0] == 10
        assert result.shape[1] == 42  # default features


class TestFusion:
    def test_fuse_features(self):
        eeg = np.random.randn(42)
        nirs = np.random.randn(20)
        result = fuse_features(eeg, nirs)
        assert result.shape == (62,)

    def test_fuse_without_nirs(self):
        eeg = np.random.randn(42)
        result = fuse_features(eeg, None)
        assert result.shape == (42,)

    def test_fuse_matrices(self):
        eeg = np.random.randn(100, 42)
        nirs = np.random.randn(100, 20)
        result = fuse_feature_matrices(eeg, nirs)
        assert result.shape == (100, 62)


class TestBandPowerCorrectness:
    """Verify band power actually measures power in the correct frequency bands."""

    def test_sinusoid_in_mu_band_detected(self):
        """A 10 Hz sinusoid should have highest power in the mu band (8-13 Hz)."""
        sfreq = 500.0
        t = np.arange(500) / sfreq
        mu_signal = np.sin(2 * np.pi * 10 * t)
        window = np.column_stack([mu_signal] * 6)  # (500, 6)

        powers = compute_band_powers(window, sfreq=sfreq)
        # Band order: theta(0:6), mu(6:12), beta(12:18), low_gamma(18:24)
        theta_power = powers[0:6].mean()
        mu_power = powers[6:12].mean()
        beta_power = powers[12:18].mean()
        gamma_power = powers[18:24].mean()

        assert mu_power > theta_power, "Mu should be stronger than theta for 10 Hz"
        assert mu_power > beta_power, "Mu should be stronger than beta for 10 Hz"
        assert mu_power > gamma_power, "Mu should be stronger than gamma for 10 Hz"

    def test_sinusoid_in_beta_band_detected(self):
        """A 20 Hz sinusoid should have highest power in the beta band (13-30 Hz)."""
        sfreq = 500.0
        t = np.arange(500) / sfreq
        beta_signal = np.sin(2 * np.pi * 20 * t)
        window = np.column_stack([beta_signal] * 6)

        powers = compute_band_powers(window, sfreq=sfreq)
        beta_power = powers[12:18].mean()
        theta_power = powers[0:6].mean()
        mu_power = powers[6:12].mean()

        assert beta_power > theta_power
        assert beta_power > mu_power

    def test_zero_signal_returns_finite(self):
        window = np.zeros((500, 6))
        result = compute_band_powers(window)
        assert np.all(np.isfinite(result))

    def test_louder_signal_has_more_power(self):
        """Doubling amplitude should increase log power."""
        window = np.random.randn(500, 6)
        p1 = compute_band_powers(window)
        p2 = compute_band_powers(window * 4.0)
        assert np.all(p2 > p1), "4x amplitude should mean more power in every band"

    def test_band_definitions_match_constants(self):
        """Verify BANDS dict has expected frequency ranges."""
        assert BANDS["theta"] == (4, 8)
        assert BANDS["mu"] == (8, 13)
        assert BANDS["beta"] == (13, 30)
        assert BANDS["low_gamma"] == (30, 40)


class TestHjorthCorrectness:
    """Verify Hjorth parameters have correct mathematical properties."""

    def test_constant_signal_zero_activity(self):
        """Constant signal has zero variance (activity)."""
        window = np.ones((500, 6)) * 5.0
        result = compute_hjorth(window)
        activity = result[0:6]  # log10(var + eps)
        assert np.all(activity < -8), "Constant signal should have near-zero activity"

    def test_higher_amplitude_higher_activity(self):
        """Larger amplitude signal should have higher activity."""
        w1 = np.random.randn(500, 6)
        w2 = w1 * 10.0
        h1 = compute_hjorth(w1)
        h2 = compute_hjorth(w2)
        # Activity is first 6 elements, log-scaled
        assert np.all(h2[0:6] > h1[0:6])


class TestWaveletEntropy:
    def test_uniform_coefficients_high_entropy(self):
        coeffs = np.ones(100)
        entropy = _wavelet_entropy(coeffs)
        assert entropy > 6.0  # log2(100) ~= 6.64

    def test_single_spike_zero_entropy(self):
        coeffs = np.zeros(100)
        coeffs[0] = 1.0
        entropy = _wavelet_entropy(coeffs)
        assert entropy == 0.0

    def test_all_zeros_returns_zero(self):
        coeffs = np.zeros(50)
        entropy = _wavelet_entropy(coeffs)
        assert entropy == 0.0

    def test_non_negative(self):
        coeffs = np.random.randn(200)
        entropy = _wavelet_entropy(coeffs)
        assert entropy >= 0.0

    def test_known_uniform_entropy_value(self):
        """4 equal coefficients: entropy = log2(4) = 2.0 exactly."""
        coeffs = np.array([1.0, 1.0, 1.0, 1.0])
        entropy = _wavelet_entropy(coeffs)
        np.testing.assert_allclose(entropy, 2.0, atol=1e-10)

    def test_known_two_element_entropy(self):
        """2 equal coefficients: entropy = log2(2) = 1.0."""
        coeffs = np.array([3.0, 3.0])
        entropy = _wavelet_entropy(coeffs)
        np.testing.assert_allclose(entropy, 1.0, atol=1e-10)

    def test_negative_coefficients_same_as_positive(self):
        """Entropy uses c^2 so sign shouldn't matter."""
        coeffs_pos = np.array([1.0, 2.0, 3.0])
        coeffs_neg = np.array([-1.0, -2.0, -3.0])
        assert _wavelet_entropy(coeffs_pos) == _wavelet_entropy(coeffs_neg)


class TestWaveletFeatures:
    def test_output_shape_default(self):
        """3 stats * 4 levels (D3,D4,D5,A5) * 6 ch = 72."""
        window = np.random.randn(500, 6)
        result = compute_wavelet_features(window)
        assert result.shape == (72,)

    def test_output_shape_no_approx(self):
        """Without approx: 3 stats * 3 detail levels * 6 ch = 54."""
        window = np.random.randn(500, 6)
        result = compute_wavelet_features(window, include_approx=False)
        assert result.shape == (54,)

    def test_output_shape_single_stat(self):
        """Single stat: 1 * 4 levels * 6 ch = 24."""
        window = np.random.randn(500, 6)
        result = compute_wavelet_features(window, stats=["energy"])
        assert result.shape == (24,)

    def test_output_shape_custom_levels(self):
        """Custom levels [4, 5]: 3 stats * 3 (D4,D5,A5) * 6 ch = 54."""
        window = np.random.randn(500, 6)
        result = compute_wavelet_features(window, levels=[4, 5])
        assert result.shape == (54,)

    def test_output_shape_single_channel(self):
        """Single channel: 3 stats * 4 levels * 1 ch = 12."""
        window = np.random.randn(500, 1)
        result = compute_wavelet_features(window)
        assert result.shape == (12,)

    def test_finite_values(self):
        window = np.random.randn(500, 6)
        result = compute_wavelet_features(window)
        assert np.all(np.isfinite(result))

    def test_zero_input_finite(self):
        window = np.zeros((500, 6))
        result = compute_wavelet_features(window)
        assert np.all(np.isfinite(result))

    def test_different_wavelets(self):
        window = np.random.randn(500, 6)
        for wavelet in ["db4", "db8", "sym5", "coif3"]:
            result = compute_wavelet_features(window, wavelet=wavelet)
            assert result.shape == (72,)
            assert np.all(np.isfinite(result))

    def test_deterministic(self):
        window = np.random.randn(500, 6)
        r1 = compute_wavelet_features(window.copy())
        r2 = compute_wavelet_features(window.copy())
        np.testing.assert_array_equal(r1, r2)

    def test_energy_increases_with_amplitude(self):
        window = np.random.randn(500, 6)
        r1 = compute_wavelet_features(window, stats=["energy"])
        r2 = compute_wavelet_features(window * 2.0, stats=["energy"])
        assert np.all(r2 > r1)

    def test_mu_sinusoid_has_energy_in_d5(self):
        """10 Hz sinusoid should concentrate energy in D5 (7.8-15.6 Hz)."""
        sfreq = 500.0
        t = np.arange(500) / sfreq
        mu_signal = np.sin(2 * np.pi * 10 * t)
        window = np.column_stack([mu_signal] * 6)

        # Extract energy only, 1 channel to simplify
        result = compute_wavelet_features(
            window[:, :1], stats=["energy"], levels=[3, 4, 5], include_approx=True,
        )
        # Order: D3 energy, D4 energy, D5 energy, A5 energy (1 channel)
        d3_energy, d4_energy, d5_energy, a5_energy = result[0], result[1], result[2], result[3]

        assert d5_energy > d3_energy, "10 Hz should be in D5 not D3"
        assert d5_energy > a5_energy, "10 Hz should be in D5 not A5"

    def test_beta_sinusoid_has_energy_in_d4(self):
        """20 Hz sinusoid should concentrate energy in D4 (15.6-31.25 Hz)."""
        sfreq = 500.0
        t = np.arange(500) / sfreq
        beta_signal = np.sin(2 * np.pi * 20 * t)
        window = beta_signal.reshape(-1, 1)

        result = compute_wavelet_features(
            window, stats=["energy"], levels=[3, 4, 5], include_approx=True,
        )
        d3_energy, d4_energy, d5_energy, a5_energy = result

        assert d4_energy > d5_energy, "20 Hz should be in D4 not D5"
        assert d4_energy > a5_energy, "20 Hz should be in D4 not A5"

    def test_different_wavelets_produce_different_outputs(self):
        """Changing wavelet family must change the output values."""
        window = np.random.randn(500, 6)
        r_db4 = compute_wavelet_features(window, wavelet="db4")
        r_db8 = compute_wavelet_features(window, wavelet="db8")
        assert not np.allclose(r_db4, r_db8), "Different wavelets must produce different results"

    def test_config_dict_changes_output(self):
        """Passing different wavelet_config to extract_window_features changes result."""
        window = np.random.randn(500, 6)
        cfg_db4 = {"family": "db4", "levels": [3, 4, 5], "include_approx": True, "stats": ["energy", "entropy", "std"]}
        cfg_db8 = {"family": "db8", "levels": [3, 4, 5], "include_approx": True, "stats": ["energy", "entropy", "std"]}

        r1 = extract_window_features(window, include_wavelet=True, wavelet_config=cfg_db4)
        r2 = extract_window_features(window, include_wavelet=True, wavelet_config=cfg_db8)

        # First 42 features (band power + Hjorth) should be identical
        np.testing.assert_array_equal(r1[:42], r2[:42])
        # Wavelet portion (42:114) should differ
        assert not np.allclose(r1[42:], r2[42:])


class TestExtractWindowFeaturesWithWavelet:
    def test_default_without_wavelet_unchanged(self):
        window = np.random.randn(500, 6)
        result = extract_window_features(window)
        assert result.shape == (42,)

    def test_with_wavelet(self):
        """24 band power + 18 Hjorth + 72 wavelet = 114."""
        window = np.random.randn(500, 6)
        result = extract_window_features(window, include_wavelet=True)
        assert result.shape == (114,)

    def test_all_features_enabled(self):
        """24 + 18 + 24 + 72 = 138."""
        window = np.random.randn(500, 6)
        result = extract_window_features(
            window, include_time_domain=True, include_wavelet=True,
        )
        assert result.shape == (138,)

    def test_batch_with_wavelet(self):
        windows = np.random.randn(10, 500, 6)
        result = extract_features_from_windows(windows, include_wavelet=True)
        assert result.shape == (10, 114)
