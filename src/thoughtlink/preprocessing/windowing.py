"""Sliding window extraction for EEG and NIRS signals."""

import numpy as np


def extract_eeg_windows(
    eeg_data: np.ndarray,
    duration: float,
    sfreq: float = 500.0,
    stimulus_onset_s: float = 3.0,
    window_duration_s: float = 1.0,
    window_stride_s: float = 0.5,
    max_stimulus_s: float = 8.0,
) -> np.ndarray:
    """Extract sliding windows from EEG stimulus period.

    Args:
        eeg_data: Raw EEG array, shape (n_samples, n_channels) e.g. (7499, 6).
        duration: Stimulus duration in seconds from the label metadata.
        sfreq: Sampling frequency in Hz.
        stimulus_onset_s: When the stimulus starts (seconds from chunk start).
        window_duration_s: Duration of each window in seconds.
        window_stride_s: Stride between consecutive windows in seconds.
        max_stimulus_s: Maximum stimulus period to use.

    Returns:
        Array of shape (n_windows, window_samples, n_channels).
    """
    window_samples = int(window_duration_s * sfreq)
    stride_samples = int(window_stride_s * sfreq)

    # Define stimulus region
    stim_start = int(stimulus_onset_s * sfreq)
    effective_duration = min(duration, max_stimulus_s)
    stim_end = min(
        int((stimulus_onset_s + effective_duration) * sfreq),
        eeg_data.shape[0],
    )

    # Extract windows
    windows = []
    pos = stim_start
    while pos + window_samples <= stim_end:
        window = eeg_data[pos : pos + window_samples, :]
        windows.append(window)
        pos += stride_samples

    if not windows:
        # Fallback: return at least one window from available data
        available = eeg_data[stim_start:stim_end, :]
        if available.shape[0] >= window_samples:
            windows.append(available[:window_samples, :])
        else:
            # Pad with zeros if needed
            padded = np.zeros((window_samples, eeg_data.shape[1]))
            padded[: available.shape[0], :] = available
            windows.append(padded)

    return np.array(windows)


def extract_nirs_stimulus(
    nirs_data: np.ndarray,
    duration: float,
    nirs_sfreq: float = 4.76,
    stimulus_onset_s: float = 3.0,
    max_stimulus_s: float = 8.0,
) -> np.ndarray:
    """Extract NIRS stimulus period with baseline correction.

    Args:
        nirs_data: Raw NIRS array, shape (72, 40, 3, 2, 3).
        duration: Stimulus duration in seconds.
        nirs_sfreq: NIRS sampling frequency.
        stimulus_onset_s: Stimulus onset time.
        max_stimulus_s: Max stimulus period.

    Returns:
        Baseline-corrected NIRS stimulus data, same last dims as input.
    """
    rest_end = int(stimulus_onset_s * nirs_sfreq)
    stim_start = rest_end
    effective_duration = min(duration, max_stimulus_s)
    stim_end = min(
        int((stimulus_onset_s + effective_duration) * nirs_sfreq),
        nirs_data.shape[0],
    )

    # Baseline: mean over rest period
    baseline = nirs_data[:rest_end].mean(axis=0, keepdims=True)

    # Baseline-corrected stimulus data
    stim_data = nirs_data[stim_start:stim_end] - baseline

    return stim_data


def windows_from_samples(
    samples: list[dict],
    sfreq: float = 500.0,
    window_duration_s: float = 1.0,
    window_stride_s: float = 0.5,
    max_stimulus_s: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract all EEG windows from a list of samples.

    Args:
        samples: List of sample dicts from loader.
        sfreq: EEG sampling frequency.
        window_duration_s: Window duration.
        window_stride_s: Window stride.
        max_stimulus_s: Max stimulus period.

    Returns:
        (X, y, subject_ids) where:
            X: shape (total_windows, window_samples, n_channels)
            y: shape (total_windows,) integer labels
            subject_ids: list of subject_id per window
    """
    from thoughtlink.data.loader import CLASS_NAMES

    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    all_windows = []
    all_labels = []
    all_subjects = []

    for sample in samples:
        windows = extract_eeg_windows(
            eeg_data=sample["eeg"],
            duration=sample["duration"],
            sfreq=sfreq,
            window_duration_s=window_duration_s,
            window_stride_s=window_stride_s,
            max_stimulus_s=max_stimulus_s,
        )

        label_idx = label_to_idx[sample["label"]]
        n_windows = windows.shape[0]

        all_windows.append(windows)
        all_labels.extend([label_idx] * n_windows)
        all_subjects.extend([sample["subject_id"]] * n_windows)

    X = np.concatenate(all_windows, axis=0)
    y = np.array(all_labels)

    return X, y, all_subjects
