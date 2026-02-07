"""Hierarchical 2-stage intent classifier.

Stage 1: Relax vs Active (binary) - high sensitivity gate
Stage 2: 4-class active intent (Right/Left/Both Fists + Tongue)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thoughtlink.data.loader import CLASS_NAMES


# Relax is index 4 in CLASS_NAMES
RELAX_IDX = CLASS_NAMES.index("Relax")
ACTIVE_CLASSES = [i for i in range(len(CLASS_NAMES)) if i != RELAX_IDX]
ACTIVE_CLASS_NAMES = [CLASS_NAMES[i] for i in ACTIVE_CLASSES]


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Two-stage hierarchical classifier for intent decoding.

    Stage 1: Binary classifier (Relax=0 vs Active=1).
    Stage 2: Multi-class classifier over active classes only.

    This reduces false triggers by filtering rest states first.
    """

    def __init__(
        self,
        stage1_model: Pipeline | None = None,
        stage2_model: Pipeline | None = None,
        stage1_threshold: float = 0.5,
    ):
        self.stage1_threshold = stage1_threshold

        self.stage1_model = stage1_model or Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42)),
        ])

        self.stage2_model = stage2_model or Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42)),
        ])

    def _make_binary_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert multi-class labels to binary (0=Relax, 1=Active)."""
        return (y != RELAX_IDX).astype(int)

    def _filter_active(
        self, X: np.ndarray, y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter to only active (non-Relax) samples and remap labels."""
        mask = y != RELAX_IDX
        X_active = X[mask]
        y_active = y[mask]

        # Remap: original class indices -> 0..3
        self._active_label_map = {orig: new for new, orig in enumerate(ACTIVE_CLASSES)}
        self._active_label_inverse = {new: orig for orig, new in self._active_label_map.items()}
        y_remapped = np.array([self._active_label_map[yi] for yi in y_active])

        return X_active, y_remapped

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train both stages.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,) using CLASS_NAMES indexing.
        """
        # Stage 1: Relax vs Active
        y_binary = self._make_binary_labels(y)
        print(f"Stage 1: {(y_binary == 0).sum()} Relax, {(y_binary == 1).sum()} Active")
        self.stage1_model.fit(X, y_binary)

        # Stage 2: Active classes only
        X_active, y_active = self._filter_active(X, y)
        print(f"Stage 2: {len(y_active)} active samples, {len(set(y_active))} classes")
        self.stage2_model.fit(X_active, y_active)

        self.classes_ = np.arange(len(CLASS_NAMES))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using hierarchical pipeline.

        Returns:
            Probability matrix (n_samples, n_classes=5).
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(CLASS_NAMES)))

        # Stage 1: P(active) and P(relax)
        stage1_proba = self.stage1_model.predict_proba(X)
        # stage1_proba[:, 0] = P(relax), stage1_proba[:, 1] = P(active)
        p_relax = stage1_proba[:, 0]
        p_active = stage1_proba[:, 1]

        # Stage 2: P(class | active) for all samples
        stage2_proba = self.stage2_model.predict_proba(X)

        # Combined: P(class) = P(active) * P(class | active)
        probs[:, RELAX_IDX] = p_relax
        for new_idx, orig_idx in self._active_label_inverse.items():
            probs[:, orig_idx] = p_active * stage2_proba[:, new_idx]

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
