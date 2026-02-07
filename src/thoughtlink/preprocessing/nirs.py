"""TD-NIRS preprocessing pipeline."""

import numpy as np


NIRS_SFREQ = 4.76
N_MODULES = 40


def preprocess_nirs(
    nirs_data: np.ndarray,
    duration: float,
    nirs_sfreq: float = NIRS_SFREQ,
    stimulus_onset_s: float = 3.0,
    max_stimulus_s: float = 8.0,
    sds_indices: tuple = (1, 2),
    moment_indices: tuple = (0, 1),
) -> np.ndarray:
    """Preprocess NIRS data: baseline correct and extract stimulus period.

    Args:
        nirs_data: Shape (72, 40, 3, 2, 3) -
            (timepoints, modules, sds_ranges, wavelengths, moments).
        duration: Stimulus duration in seconds.
        nirs_sfreq: NIRS sampling frequency.
        stimulus_onset_s: Stimulus onset time.
        max_stimulus_s: Max stimulus period to use.
        sds_indices: Which source-detector separations to use
            (0=short, 1=medium, 2=long).
        moment_indices: Which moments to use (0=log_sum, 1=mean_tof, 2=variance).

    Returns:
        Baseline-corrected, selected NIRS features.
        Shape: (n_stim_timepoints, n_features).
    """
    rest_end = int(stimulus_onset_s * nirs_sfreq)
    stim_start = rest_end
    effective_duration = min(duration, max_stimulus_s)
    stim_end = min(
        int((stimulus_onset_s + effective_duration) * nirs_sfreq),
        nirs_data.shape[0],
    )

    # Baseline correction: subtract rest period mean
    baseline = nirs_data[:rest_end].mean(axis=0, keepdims=True)
    corrected = nirs_data[stim_start:stim_end] - baseline

    # Select SDS and moments
    # corrected shape: (n_t, 40, 3, 2, 3)
    selected = corrected[:, :, sds_indices, :, :][:, :, :, :, moment_indices]
    # shape: (n_t, 40, len(sds_indices), 2, len(moment_indices))

    # Flatten spatial/spectral dims
    n_t = selected.shape[0]
    features = selected.reshape(n_t, -1)

    return features
