"""Feature fusion combining EEG and NIRS features."""

import numpy as np


def fuse_features(
    eeg_features: np.ndarray,
    nirs_features: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate EEG and NIRS features.

    Args:
        eeg_features: EEG feature vector.
        nirs_features: Optional NIRS feature vector.

    Returns:
        Combined feature vector.
    """
    if nirs_features is None:
        return eeg_features
    return np.concatenate([eeg_features, nirs_features])


def fuse_all_features(
    eeg_features: np.ndarray,
    csp_features: np.ndarray | None = None,
    nirs_features: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate all feature types column-wise.

    Args:
        eeg_features: Shape (n_samples, n_eeg_features).
        csp_features: Shape (n_samples, n_csp_features) or None.
        nirs_features: Shape (n_samples, n_nirs_features) or None.

    Returns:
        Combined matrix.
    """
    parts = [eeg_features]
    if csp_features is not None:
        parts.append(csp_features)
    if nirs_features is not None:
        parts.append(nirs_features)
    return np.hstack(parts)


def fuse_feature_matrices(
    eeg_matrix: np.ndarray,
    nirs_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate EEG and NIRS feature matrices column-wise.

    Since EEG has multiple windows per trial but NIRS has one vector per trial,
    NIRS features are repeated for each EEG window of the same trial.

    Args:
        eeg_matrix: Shape (n_samples, n_eeg_features).
        nirs_matrix: Shape (n_samples, n_nirs_features) or None.

    Returns:
        Combined matrix of shape (n_samples, n_eeg_features + n_nirs_features).
    """
    if nirs_matrix is None:
        return eeg_matrix
    return np.hstack([eeg_matrix, nirs_matrix])
