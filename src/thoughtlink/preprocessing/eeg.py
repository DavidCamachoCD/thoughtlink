"""EEG preprocessing pipeline using MNE-Python."""

import numpy as np
import mne


CHANNELS = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
SFREQ = 500.0


def create_raw(eeg_data: np.ndarray, sfreq: float = SFREQ) -> mne.io.RawArray:
    """Create MNE RawArray from numpy EEG data.

    Args:
        eeg_data: Shape (n_samples, n_channels), values in microvolts.
        sfreq: Sampling frequency in Hz.

    Returns:
        MNE RawArray with proper channel info.
    """
    info = mne.create_info(
        ch_names=CHANNELS,
        sfreq=sfreq,
        ch_types="eeg",
    )
    # Convert microvolts -> volts (MNE expects SI units)
    raw = mne.io.RawArray(eeg_data.T * 1e-6, info, verbose=False)
    return raw


def preprocess_eeg(
    eeg_data: np.ndarray,
    sfreq: float = SFREQ,
    bandpass_low: float = 1.0,
    bandpass_high: float = 40.0,
) -> np.ndarray:
    """Full EEG preprocessing pipeline.

    Steps:
    1. Create MNE RawArray
    2. Bandpass filter 1-40 Hz (captures mu 8-13Hz and beta 13-30Hz)
    3. Common Average Reference (CAR)

    Args:
        eeg_data: Shape (n_samples, n_channels), values in microvolts.
        sfreq: Sampling frequency.
        bandpass_low: Low cutoff frequency.
        bandpass_high: High cutoff frequency.

    Returns:
        Preprocessed EEG data, shape (n_samples, n_channels), in microvolts.
    """
    raw = create_raw(eeg_data, sfreq)

    # Bandpass filter
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )

    # Common Average Reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # Return data in microvolts (convert back from volts)
    return raw.get_data().T * 1e6


def preprocess_sample(sample: dict) -> dict:
    """Preprocess EEG data in a sample dict (in-place).

    Args:
        sample: Sample dict from loader with 'eeg' key.

    Returns:
        Same dict with 'eeg' replaced by preprocessed data.
    """
    sample["eeg"] = preprocess_eeg(sample["eeg"])
    return sample


def preprocess_all(samples: list[dict]) -> list[dict]:
    """Preprocess EEG for all samples."""
    for i, sample in enumerate(samples):
        preprocess_sample(sample)
        if (i + 1) % 100 == 0:
            print(f"Preprocessed {i + 1}/{len(samples)} samples")
    return samples
