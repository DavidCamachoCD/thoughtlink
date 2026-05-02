"""Calibration diagnostics for probabilistic classifiers.

References:
- Naeini, Cooper & Hauskrecht (2015). Obtaining Well Calibrated Probabilities
  Using Bayesian Binning. AAAI. -- ECE, MCE.
- Brier (1950). Verification of Forecasts Expressed in Terms of Probability. -- Brier score.

See `docs/references.md` for full citations.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Equal-width binning ECE on top-1 (predicted-class) confidence.

    For each bin, computes |average confidence - average accuracy| and
    returns the sample-weighted mean across bins. Equal to 0 when the
    model is perfectly calibrated.

    Args:
        y_true: True class indices, shape (n,).
        probs: Class probabilities, shape (n, n_classes).
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        ECE in [0, 1].
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Right-inclusive on the last bin so confidence == 1.0 lands in it.
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Worst-case calibration gap across non-empty bins (Naeini et al. 2015)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        gap = abs(accuracies[mask].mean() - confidences[mask].mean())
        if gap > mce:
            mce = gap
    return float(mce)


def brier_score(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int | None = None,
) -> float:
    """Multi-class Brier score: mean squared error between probs and one-hot y.

    Brier (1950). Proper scoring rule, decomposable into reliability +
    resolution + uncertainty (Murphy 1973). Equal to 0 only when every
    prediction is one-hot and correct.

    Args:
        y_true: True class indices, shape (n,).
        probs: Class probabilities, shape (n, n_classes).
        n_classes: Optional class count (defaults to probs.shape[1]).

    Returns:
        Brier score in [0, 2] (0 best).
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    if n_classes is None:
        n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Per-bin data for plotting a reliability diagram.

    Returns a dict (rather than a tuple) so callers don't have to remember
    positional order, and callers using matplotlib don't pull it as a hard
    dependency of this module.

    Args:
        y_true: True class indices, shape (n,).
        probs: Class probabilities, shape (n, n_classes).
        n_bins: Number of equal-width confidence bins.

    Returns:
        Dict with keys:
          - bin_edges: shape (n_bins + 1,)
          - bin_centers: shape (n_bins,)
          - bin_counts: number of samples per bin
          - bin_confidence: average reported confidence per bin (NaN if empty)
          - bin_accuracy: empirical accuracy per bin (NaN if empty)
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts = np.zeros(n_bins, dtype=int)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            continue
        bin_conf[i] = confidences[mask].mean()
        bin_acc[i] = accuracies[mask].mean()

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_counts": counts,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
    }
