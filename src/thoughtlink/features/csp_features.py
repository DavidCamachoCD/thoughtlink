"""Common Spatial Patterns (CSP) feature extraction using MNE-Python.

CSP maximizes variance ratio between classes, making it ideal
for motor imagery classification. Uses One-vs-Rest for multi-class.
"""

import numpy as np
from mne.decoding import CSP


class CSPFeatureExtractor:
    """One-vs-Rest CSP for multi-class EEG classification.

    Fits one CSP per class (class vs rest), then concatenates
    the log-variance features from all CSP filters.

    Follows sklearn-style fit/transform interface.

    Args:
        n_components: Number of CSP components per binary problem.
        reg: Regularization for covariance estimation.
            "ledoit_wolf" recommended for small sample sizes.
        log: Whether to return log-variance features.
        n_classes: Number of classes (for OvR setup).
    """

    def __init__(
        self,
        n_components: int = 4,
        reg: str | None = "ledoit_wolf",
        log: bool = True,
        n_classes: int = 5,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.n_classes = n_classes
        self.csps_: list[CSP] = []
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatureExtractor":
        """Fit OvR CSP filters.

        Args:
            X: EEG windows, shape (n_windows, n_samples, n_channels).
                Will be transposed to (n_windows, n_channels, n_samples)
                for MNE CSP.
            y: Labels, shape (n_windows,).

        Returns:
            self
        """
        # MNE CSP expects (n_epochs, n_channels, n_samples)
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float64)

        self.csps_ = []
        unique_classes = sorted(np.unique(y))

        for cls in unique_classes:
            y_binary = (y == cls).astype(int)
            # Need at least 2 samples per class for CSP
            if y_binary.sum() < 2 or (1 - y_binary).sum() < 2:
                continue
            csp = CSP(
                n_components=min(self.n_components, X.shape[2]),
                reg=self.reg,
                log=self.log,
            )
            csp.fit(X_t, y_binary)
            self.csps_.append(csp)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract CSP features from windows.

        Args:
            X: EEG windows, shape (n_windows, n_samples, n_channels).

        Returns:
            CSP feature matrix, shape (n_windows, n_fitted_classes * n_components).
        """
        if not self.is_fitted_:
            raise RuntimeError("CSPFeatureExtractor not fitted. Call fit() first.")

        X_t = np.transpose(X, (0, 2, 1)).astype(np.float64)

        features = []
        for csp in self.csps_:
            feat = csp.transform(X_t)
            features.append(feat)

        result = np.hstack(features)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
