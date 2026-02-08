"""EEG feature extraction: band power, CSP, Hjorth parameters, and DWT wavelets."""

import numpy as np
import pywt
from scipy.signal import welch


BANDS = {
    "theta": (4, 8),
    "mu": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 40),
}


def compute_band_powers(
    window: np.ndarray,
    sfreq: float = 500.0,
    bands: dict | None = None,
) -> np.ndarray:
    """Compute log band power features for an EEG window.

    Args:
        window: Shape (n_samples, n_channels), e.g. (500, 6).
        sfreq: Sampling frequency.
        bands: Dict of {band_name: (low_hz, high_hz)}.

    Returns:
        Feature vector of shape (n_bands * n_channels,).
    """
    if bands is None:
        bands = BANDS

    nperseg = min(256, window.shape[0])
    freqs, psd = welch(window, fs=sfreq, nperseg=nperseg, axis=0)
    # psd shape: (n_freq_bins, n_channels)

    features = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.log10(psd[mask].mean(axis=0) + 1e-10)  # (n_channels,)
        features.append(band_power)

    return np.concatenate(features)


def compute_hjorth(window: np.ndarray) -> np.ndarray:
    """Compute Hjorth parameters for an EEG window.

    Computes Activity, Mobility, and Complexity for each channel.

    Args:
        window: Shape (n_samples, n_channels).

    Returns:
        Feature vector of shape (3 * n_channels,).
    """
    # Activity: variance of the signal
    activity = np.var(window, axis=0)

    # First derivative
    d1 = np.diff(window, axis=0)
    d1_var = np.var(d1, axis=0)

    # Second derivative
    d2 = np.diff(d1, axis=0)
    d2_var = np.var(d2, axis=0)

    # Mobility: sqrt(var(d1) / var(signal))
    mobility = np.sqrt(d1_var / (activity + 1e-10))

    # Complexity: mobility(d1) / mobility(signal)
    mobility_d1 = np.sqrt(d2_var / (d1_var + 1e-10))
    complexity = mobility_d1 / (mobility + 1e-10)

    return np.concatenate([
        np.log10(activity + 1e-10),  # log scale for stability
        mobility,
        complexity,
    ])


def compute_time_domain(window: np.ndarray) -> np.ndarray:
    """Compute time-domain statistics per channel.

    Args:
        window: Shape (n_samples, n_channels).

    Returns:
        Feature vector of shape (4 * n_channels,).
    """
    mean_abs = np.mean(np.abs(window), axis=0)
    std = np.std(window, axis=0)

    # Zero-crossing rate
    signs = np.sign(window)
    zcr = np.mean(np.abs(np.diff(signs, axis=0)), axis=0) / 2.0

    # Root mean square
    rms = np.sqrt(np.mean(window ** 2, axis=0))

    return np.concatenate([mean_abs, std, zcr, rms])


def _wavelet_entropy(coeffs: np.ndarray) -> float:
    """Compute Shannon entropy of wavelet coefficients.

    Normalizes squared coefficients into a probability distribution,
    then computes Shannon entropy.

    Args:
        coeffs: 1D array of wavelet coefficients.

    Returns:
        Shannon entropy value (non-negative float).
    """
    c_sq = coeffs ** 2
    total = c_sq.sum()
    if total < 1e-20:
        return 0.0
    p = c_sq / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_wavelet_features(
    window: np.ndarray,
    wavelet: str = "db4",
    levels: list[int] | None = None,
    include_approx: bool = True,
    stats: list[str] | None = None,
) -> np.ndarray:
    """Compute DWT-based features for an EEG window.

    Performs multi-level discrete wavelet decomposition and extracts
    statistical features from the detail and approximation coefficients
    at selected decomposition levels.

    At 500 Hz with level 5:
        D5: 7.8-15.6 Hz (mu), D4: 15.6-31.25 Hz (beta),
        D3: 31.25-62.5 Hz (low gamma), A5: 0-7.8 Hz (theta/delta)

    Args:
        window: Shape (n_samples, n_channels), e.g. (500, 6).
        wavelet: PyWavelets wavelet name (e.g., "db4").
        levels: Which detail levels to use (1-indexed). Default: [3, 4, 5].
        include_approx: Whether to include the final approximation coefficients.
        stats: Statistics to compute per level. Options: "energy", "entropy", "std".

    Returns:
        Feature vector of shape (n_stats * n_selected_levels * n_channels,).
        With defaults: 3 stats * 4 levels (D3, D4, D5, A5) * 6 channels = 72.
    """
    if levels is None:
        levels = [3, 4, 5]
    if stats is None:
        stats = ["energy", "entropy", "std"]

    max_level = max(levels)
    n_channels = window.shape[1]

    all_features = []

    for ch in range(n_channels):
        coeffs = pywt.wavedec(window[:, ch], wavelet, level=max_level)
        # coeffs[0] = cA_max_level (approximation)
        # coeffs[i] = cD_(max_level - i + 1) for i >= 1

        selected_coeffs = []
        for lvl in sorted(levels):
            detail_idx = max_level - lvl + 1
            selected_coeffs.append(coeffs[detail_idx])

        if include_approx:
            selected_coeffs.append(coeffs[0])

        for c in selected_coeffs:
            for stat in stats:
                if stat == "energy":
                    all_features.append(np.log10(np.sum(c ** 2) + 1e-10))
                elif stat == "entropy":
                    all_features.append(_wavelet_entropy(c))
                elif stat == "std":
                    all_features.append(np.std(c))

    return np.array(all_features)


def extract_window_features(
    window: np.ndarray,
    sfreq: float = 500.0,
    include_hjorth: bool = True,
    include_time_domain: bool = False,
    include_wavelet: bool = False,
    wavelet_config: dict | None = None,
) -> np.ndarray:
    """Extract all features from a single EEG window.

    Args:
        window: Shape (n_samples, n_channels).
        sfreq: Sampling frequency.
        include_hjorth: Whether to include Hjorth parameters.
        include_time_domain: Whether to include time-domain stats.
        include_wavelet: Whether to include DWT wavelet features.
        wavelet_config: Optional dict with keys: family, levels, include_approx, stats.

    Returns:
        Feature vector.
    """
    features = [compute_band_powers(window, sfreq)]

    if include_hjorth:
        features.append(compute_hjorth(window))

    if include_time_domain:
        features.append(compute_time_domain(window))

    if include_wavelet:
        wc = wavelet_config or {}
        features.append(compute_wavelet_features(
            window,
            wavelet=wc.get("family", "db4"),
            levels=wc.get("levels", [3, 4, 5]),
            include_approx=wc.get("include_approx", True),
            stats=wc.get("stats", ["energy", "entropy", "std"]),
        ))

    return np.concatenate(features)


def extract_features_from_windows(
    windows: np.ndarray,
    sfreq: float = 500.0,
    include_hjorth: bool = True,
    include_time_domain: bool = False,
    include_wavelet: bool = False,
    wavelet_config: dict | None = None,
) -> np.ndarray:
    """Extract features from all windows.

    Args:
        windows: Shape (n_windows, n_samples, n_channels).
        sfreq: Sampling frequency.
        include_hjorth: Whether to include Hjorth parameters.
        include_time_domain: Whether to include time-domain stats.
        include_wavelet: Whether to include DWT wavelet features.
        wavelet_config: Optional dict with keys: family, levels, include_approx, stats.

    Returns:
        Feature matrix of shape (n_windows, n_features).
    """
    features = np.array([
        extract_window_features(
            w, sfreq, include_hjorth, include_time_domain,
            include_wavelet, wavelet_config,
        )
        for w in windows
    ])
    # Replace NaN/Inf from edge effects in bandpass filtering or zero-variance windows
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features
