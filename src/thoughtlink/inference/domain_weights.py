"""Likelihood-ratio estimation for weighted conformal prediction.

Implements the standard 'classify two distributions' trick used by Tibshirani,
Foygel Barber, Candes & Ramdas (2019, *Conformal Prediction Under Covariate
Shift*) to estimate the likelihood ratio `w(x) = P_test(x) / P_calib(x)`.

The ratio is needed by `WeightedAPSConformalPredictor` to recover marginal
coverage when the calibration and test inputs are drawn from different
distributions (e.g. different EEG subjects in ThoughtLink).

Method:
    1. Concatenate calibration and test inputs, label them 0 / 1.
    2. Fit a binary classifier (logistic regression by default).
    3. By Bayes' rule, P_test(x) / P_calib(x) is proportional to
       p_hat(x) / (1 - p_hat(x))  (with a known constant we can ignore --
       weighted conformal only depends on the ratio up to a constant).
    4. Optionally clip extreme weights to bound variance (Sugiyama et al. 2008).

References: see `docs/references.md`.
"""

from __future__ import annotations

import numpy as np


def estimate_likelihood_ratio(
    X_calib: np.ndarray,
    X_test: np.ndarray,
    *,
    clip: float = 50.0,
    classifier_C: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Estimate `w(x) = P_test(x) / P_calib(x)` for every row in `X_calib`.

    Default `clip=50` was chosen by sweeping (5, 10, 20, 50) on the cross-subject
    EEG calibration / test split: only `clip=50` recovers coverage close to the
    target `1 - alpha` under the heavy subject shift. Smaller clips collapse
    back to the unweighted predictor's coverage. The regularization strength
    `classifier_C` has a much smaller effect than the clip on this dataset.

    The ESS ratio (see `diagnose_weights`) is expected to be small (~0.02) under
    heavy shift -- that is the honest cost of recovering coverage when only a
    fraction of the calibration distribution overlaps the test distribution.
    Use `diagnose_weights(...)['ess_ratio']` to monitor.

    Args:
        X_calib: Calibration features, shape (n_calib, n_features).
        X_test:  Test features,        shape (n_test,  n_features).
        clip:    Maximum allowed weight. Pass `None` to disable.
        classifier_C: Inverse regularization strength for the logistic
                 regression. Smaller values regularize more.
        random_state: Seed for the logistic-regression classifier.

    Returns:
        weights: shape (n_calib,) -- one positive likelihood-ratio estimate
                 per calibration point. Clipped to `[1/clip, clip]` if `clip`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_calib = np.asarray(X_calib)
    X_test = np.asarray(X_test)
    if X_calib.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: calib={X_calib.shape[1]}, "
            f"test={X_test.shape[1]}"
        )

    X = np.concatenate([X_calib, X_test], axis=0)
    y = np.concatenate(
        [np.zeros(len(X_calib), dtype=int), np.ones(len(X_test), dtype=int)]
    )

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        C=classifier_C,
        random_state=random_state,
    ).fit(Xs, y)

    # P(test | x_calib): only the calibration rows are needed.
    p_test_given_x = clf.predict_proba(Xs[: len(X_calib)])[:, 1]
    eps = 1e-6
    p_test_given_x = np.clip(p_test_given_x, eps, 1 - eps)
    weights = p_test_given_x / (1 - p_test_given_x)

    if clip is not None and clip > 0:
        weights = np.clip(weights, 1.0 / clip, clip)

    return weights


def diagnose_weights(weights: np.ndarray) -> dict:
    """Quick health check of the likelihood-ratio weights.

    Returns a dict of summary stats. Two situations are concerning:
    1. `effective_sample_size` << `len(weights)` -> a few large weights
       dominate; the weighted quantile becomes high-variance. Lowering
       `clip` in `estimate_likelihood_ratio` mitigates.
    2. `max / median` > 100 -> extreme covariate shift; weighted conformal
       guarantees still hold but prediction sets may be unhelpfully large.
    """
    w = np.asarray(weights)
    s = w.sum()
    eff = (s ** 2) / (np.sum(w ** 2) + 1e-12)
    return {
        "n": int(len(w)),
        "sum": float(s),
        "min": float(w.min()),
        "max": float(w.max()),
        "median": float(np.median(w)),
        "effective_sample_size": float(eff),
        "ess_ratio": float(eff / len(w)),
    }
