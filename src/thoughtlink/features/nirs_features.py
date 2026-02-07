"""NIRS feature extraction from preprocessed TD-NIRS data."""

import numpy as np
from sklearn.decomposition import PCA


def compute_nirs_temporal_features(nirs_stim: np.ndarray) -> np.ndarray:
    """Compute temporal features from NIRS stimulus period.

    Args:
        nirs_stim: Shape (n_timepoints, n_features) from nirs.preprocess_nirs().

    Returns:
        Feature vector summarizing the temporal NIRS response.
    """
    if nirs_stim.shape[0] == 0:
        return np.zeros(nirs_stim.shape[1] * 3)

    # Mean activation over stimulus period
    mean_act = nirs_stim.mean(axis=0)

    # Peak activation (max absolute)
    peak_act = nirs_stim[np.argmax(np.abs(nirs_stim).sum(axis=1))]

    # Slope: linear fit over time for each feature
    n_t = nirs_stim.shape[0]
    if n_t > 1:
        t = np.arange(n_t)
        t_centered = t - t.mean()
        slope = (t_centered[:, None] * nirs_stim).sum(axis=0) / (t_centered ** 2).sum()
    else:
        slope = np.zeros(nirs_stim.shape[1])

    return np.concatenate([mean_act, peak_act, slope])


def fit_nirs_pca(
    nirs_features_list: list[np.ndarray],
    n_components: int = 20,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on NIRS features and transform.

    Args:
        nirs_features_list: List of feature vectors from compute_nirs_temporal_features.
        n_components: Number of PCA components.

    Returns:
        (fitted PCA, transformed features matrix).
    """
    X = np.array(nirs_features_list)
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced


def transform_nirs_pca(
    nirs_features: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Transform NIRS features using a fitted PCA.

    Args:
        nirs_features: Feature vector from compute_nirs_temporal_features.
        pca: Fitted PCA model.

    Returns:
        Reduced feature vector.
    """
    return pca.transform(nirs_features.reshape(1, -1)).flatten()
