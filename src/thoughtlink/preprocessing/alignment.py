"""Euclidean alignment for cross-subject domain adaptation.

Reference: He & Wu, 2020 - "Transfer Learning for Brain-Computer Interfaces"
"""

import numpy as np
from scipy.linalg import sqrtm, inv


def compute_subject_covariance(X_windows: np.ndarray) -> np.ndarray:
    """Compute the mean covariance matrix for a subject's windows.

    Args:
        X_windows: Shape (n_windows, n_samples, n_channels).

    Returns:
        Mean covariance matrix, shape (n_channels, n_channels).
    """
    covs = []
    for i in range(X_windows.shape[0]):
        window = X_windows[i]  # (n_samples, n_channels)
        cov = np.cov(window, rowvar=False)  # (n_channels, n_channels)
        covs.append(cov)
    return np.mean(covs, axis=0)


def euclidean_align(
    X_windows: np.ndarray,
    subject_ids: list[str],
) -> np.ndarray:
    """Apply Euclidean alignment per subject.

    For each subject, computes R = mean covariance, then:
        X_aligned = X @ R^(-1/2).T

    This whitens each subject's spatial patterns toward identity,
    reducing inter-subject variability.

    Args:
        X_windows: Shape (n_windows, n_samples, n_channels).
        subject_ids: Subject ID per window.

    Returns:
        Aligned windows, same shape as input.
    """
    X_aligned = X_windows.copy()
    subject_ids_arr = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids_arr)

    for subj in unique_subjects:
        mask = subject_ids_arr == subj
        subj_windows = X_windows[mask]

        R = compute_subject_covariance(subj_windows)

        # R^(-1/2) - the whitening matrix
        try:
            R_inv_sqrt = np.real(inv(sqrtm(R)))
        except np.linalg.LinAlgError:
            # Fallback: skip alignment for this subject if singular
            continue

        for i in np.where(mask)[0]:
            X_aligned[i] = X_windows[i] @ R_inv_sqrt

    return X_aligned
