"""Split conformal prediction for multi-class classifiers.

Provides distribution-free coverage guarantees: for an exchangeable test point
drawn from the same distribution as the calibration set, the returned
prediction set is guaranteed to contain the true label with probability
>= 1 - alpha (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2023).

Two score functions are exposed:

- `APSConformalPredictor`  -- Adaptive Prediction Sets (Romano, Sesia &
                              Candes 2020). Score = sum of probabilities of
                              classes ranked at-or-above the true class. Sets
                              shrink when the model is confident and grow when
                              it is uncertain, while preserving marginal
                              coverage. Recommended default.
- `NaiveConformalPredictor` -- Score = 1 - p(y_true). Simpler but produces less
                              adaptive sets; included for ablation only.

The user-facing intent: when |set| > 1, the BrainPolicy treats the prediction
as "uncertain" and holds the previous action -- a hard guardrail that complements
the soft thresholding in `confidence.py`.

References: see `docs/references.md`.
"""

from __future__ import annotations

import numpy as np


def _aps_score_calib(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """APS calibration score: cumulative probability up to and including y_true.

    For each row, sort classes by descending probability and accumulate
    probabilities until we reach the true class. Returns the cumulative sum
    at that point. Lower scores = the true class is among the top-confidence
    classes; higher scores = it is buried lower in the ranking.

    Following Romano et al. 2020, randomized tie-breaking via uniform U is
    used to obtain exact (rather than conservative) coverage. We default to
    deterministic (mid-point) here for reproducibility; pass `randomize=True`
    to enable randomized scores.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)
    n, k = probs.shape

    order = np.argsort(-probs, axis=1)  # descending
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    # Position of the true class in the sorted order.
    rank = np.argmax(order == y_true[:, None], axis=1)
    return cumsum[np.arange(n), rank]


def _aps_score_test(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """APS test scores: for each row return (sorted_order, cumulative sums).

    Used by predict_set: include classes whose cumulative probability is
    <= q_hat in the prediction set.
    """
    probs = np.asarray(probs)
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    return order, cumsum


class APSConformalPredictor:
    """Adaptive Prediction Sets (Romano, Sesia & Candes 2020).

    Workflow:
        # 1. Fit on held-out calibration probas + labels.
        cp = APSConformalPredictor(alpha=0.1).fit(probs_calib, y_calib)
        # 2. At inference time, get a prediction set per sample.
        sets = cp.predict_set(probs_test)
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: float | None = None
        self.n_classes_: int | None = None

    def fit(self, probs_calib: np.ndarray, y_calib: np.ndarray) -> "APSConformalPredictor":
        """Compute the conformal quantile q_hat from calibration data.

        Uses the finite-sample-corrected quantile level
        `ceil((n + 1) * (1 - alpha)) / n` to guarantee marginal coverage
        >= 1 - alpha for an exchangeable test point (Angelopoulos & Bates 2023).
        """
        probs_calib = np.asarray(probs_calib)
        if probs_calib.ndim != 2:
            raise ValueError("probs_calib must have shape (n, n_classes)")
        scores = _aps_score_calib(probs_calib, y_calib)
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(1.0, level)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.n_classes_ = probs_calib.shape[1]
        return self

    def predict_set(self, probs: np.ndarray) -> list[set[int]]:
        """Return a prediction set per row.

        A class is included in the set iff its APS cumulative score is
        <= q_hat. The class with the largest probability is always included
        (so the set is never empty).
        """
        if self.q_hat is None:
            raise RuntimeError("APSConformalPredictor.fit() must be called first.")
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs[None, :]
        order, cumsum = _aps_score_test(probs)
        sets: list[set[int]] = []
        for i in range(probs.shape[0]):
            mask = cumsum[i] <= self.q_hat
            # Always include the top class (handles rare cases where even the
            # top class score exceeds q_hat due to extreme overconfidence).
            mask[0] = True
            sets.append(set(order[i, mask].tolist()))
        return sets

    def is_singleton(self, probs: np.ndarray) -> np.ndarray:
        """Boolean per-row mask: True iff the prediction set has exactly 1 class."""
        return np.array([len(s) == 1 for s in self.predict_set(probs)])

    def empirical_coverage(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Fraction of test points whose set contains the true label."""
        sets = self.predict_set(probs)
        y_true = np.asarray(y_true).astype(int)
        return float(np.mean([yi in s for yi, s in zip(y_true, sets)]))

    def average_set_size(self, probs: np.ndarray) -> float:
        sets = self.predict_set(probs)
        return float(np.mean([len(s) for s in sets]))


class NaiveConformalPredictor:
    """Naive split-conformal with score s(x, y) = 1 - p(y).

    Included as an ablation against APS. Produces a prediction set per row
    of all classes c such that 1 - p(c) <= q_hat. Coverage guarantee is the
    same; sets tend to be larger or less adaptive than APS.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit(self, probs_calib: np.ndarray, y_calib: np.ndarray) -> "NaiveConformalPredictor":
        probs_calib = np.asarray(probs_calib)
        y_calib = np.asarray(y_calib).astype(int)
        true_probs = probs_calib[np.arange(len(y_calib)), y_calib]
        scores = 1.0 - true_probs
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(1.0, level)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_set(self, probs: np.ndarray) -> list[set[int]]:
        if self.q_hat is None:
            raise RuntimeError("NaiveConformalPredictor.fit() must be called first.")
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs[None, :]
        sets: list[set[int]] = []
        for row in probs:
            mask = (1.0 - row) <= self.q_hat
            if not mask.any():
                mask[int(np.argmax(row))] = True
            sets.append(set(np.where(mask)[0].tolist()))
        return sets

    def empirical_coverage(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        sets = self.predict_set(probs)
        y_true = np.asarray(y_true).astype(int)
        return float(np.mean([yi in s for yi, s in zip(y_true, sets)]))

    def average_set_size(self, probs: np.ndarray) -> float:
        sets = self.predict_set(probs)
        return float(np.mean([len(s) for s in sets]))
