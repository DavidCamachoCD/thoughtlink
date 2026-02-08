"""Subject-level feature normalization for cross-subject generalization."""

import numpy as np


def normalize_features_by_subject(
    X: np.ndarray,
    subject_ids: list[str],
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Z-score normalize features per subject.

    For each subject, computes mean and std of their features,
    then normalizes to zero-mean unit-variance. This removes
    cross-subject baseline amplitude differences.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        subject_ids: List of subject_id per sample, len == n_samples.
        stats: Optional pre-computed {subject_id: (mean, std)} dict.
            If None, stats are computed from X.

    Returns:
        (X_normalized, stats_dict) where:
            X_normalized: Same shape as X, z-scored per subject.
            stats_dict: {subject_id: (mean_array, std_array)} for reuse.
    """
    X_norm = X.copy()
    subject_ids_array = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids_array)
    computed_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for subj in unique_subjects:
        mask = subject_ids_array == subj
        if stats is not None and subj in stats:
            mean, std = stats[subj]
        else:
            mean = X[mask].mean(axis=0)
            std = X[mask].std(axis=0)
            std[std < 1e-10] = 1.0  # prevent division by zero
        computed_stats[subj] = (mean, std)
        X_norm[mask] = (X[mask] - mean) / std

    return X_norm, computed_stats
