"""EEG feature extraction: band power, CSP, and Hjorth parameters."""

import numpy as np
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


def extract_window_features(
    window: np.ndarray,
    sfreq: float = 500.0,
    include_hjorth: bool = True,
    include_time_domain: bool = False,
) -> np.ndarray:
    """Extract all features from a single EEG window.

    Args:
        window: Shape (n_samples, n_channels).
        sfreq: Sampling frequency.
        include_hjorth: Whether to include Hjorth parameters.
        include_time_domain: Whether to include time-domain stats.

    Returns:
        Feature vector.
    """
    features = [compute_band_powers(window, sfreq)]

    if include_hjorth:
        features.append(compute_hjorth(window))

    if include_time_domain:
        features.append(compute_time_domain(window))

    return np.concatenate(features)


def extract_features_from_windows(
    windows: np.ndarray,
    sfreq: float = 500.0,
    include_hjorth: bool = True,
    include_time_domain: bool = False,
) -> np.ndarray:
    """Extract features from all windows.

    Args:
        windows: Shape (n_windows, n_samples, n_channels).
        sfreq: Sampling frequency.

    Returns:
        Feature matrix of shape (n_windows, n_features).
    """
    features = np.array([
        extract_window_features(w, sfreq, include_hjorth, include_time_domain)
        for w in windows
    ])
    # Replace NaN/Inf from edge effects in bandpass filtering or zero-variance windows
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features
