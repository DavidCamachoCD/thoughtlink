"""Ensemble classifier combining multiple model predictions.

Supports both soft voting (probability averaging) and hard voting
(majority vote on predicted labels).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble classifier using soft or hard voting.

    Follows sklearn-style fit/predict/predict_proba interface.

    Args:
        models: List of (name, model) tuples. Models must have
            predict_proba() for soft voting.
        voting: "soft" for probability averaging, "hard" for majority vote.
        weights: Optional per-model weights for weighted voting.
    """

    def __init__(
        self,
        models: list[tuple[str, object]] | None = None,
        voting: str = "soft",
        weights: list[float] | None = None,
    ):
        self.models = models or []
        self.voting = voting
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """Train all constituent models."""
        self.classes_ = np.unique(y)
        for name, model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across models."""
        weights = self.weights or [1.0] * len(self.models)
        total_weight = sum(weights)

        probas = []
        for (name, model), w in zip(self.models, weights):
            proba = model.predict_proba(X)
            probas.append(proba * w)

        return sum(probas) / total_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using voting strategy."""
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:  # hard voting
            predictions = np.array([m.predict(X) for _, m in self.models])
            # Majority vote per sample
            from scipy.stats import mode
            result, _ = mode(predictions, axis=0, keepdims=False)
            return result.astype(int)
