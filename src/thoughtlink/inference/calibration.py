"""Post-hoc confidence calibration for ThoughtLink classifiers.

Provides three calibrators that share a common `predict_proba(X)` surface so
they are drop-in replacements for the underlying model:

- `SklearnCalibrator`     -- isotonic regression (Zadrozny & Elkan 2002) or
                             Platt scaling (Platt 1999), via sklearn's
                             CalibratedClassifierCV with cv='prefit'.
- `TemperatureScaler`     -- single-parameter softmax temperature for neural
                             networks (Guo et al. 2017) -- requires logits.
- `HierarchicalCalibrator` -- calibrates Stage 1 (rest/active) and Stage 2
                             (4-class) of `HierarchicalClassifier` independently.
                             The marginal product P(active) * P(class|active)
                             remains a valid joint probability.

References: see `docs/references.md`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from thoughtlink.data.loader import CLASS_NAMES
from thoughtlink.models.hierarchical import (
    ACTIVE_CLASSES,
    RELAX_IDX,
    HierarchicalClassifier,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SklearnCalibrator:
    """Wraps an already-trained sklearn classifier with isotonic or Platt
    calibration fitted on a held-out calibration set.

    Internally delegates to `sklearn.calibration.CalibratedClassifierCV`
    with `cv='prefit'`. The original model is not refit -- only the
    calibration mapping is learned, which means this is fast and does not
    perturb the decision function.
    """

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")
        self.method = method
        self._calibrated = None

    def fit(self, base_model, X_calib: np.ndarray, y_calib: np.ndarray) -> "SklearnCalibrator":
        """Fit calibration on the held-out calibration set.

        Args:
            base_model: Fitted sklearn-style estimator with `predict_proba`.
            X_calib: Calibration features.
            y_calib: Calibration labels (integer-encoded).
        """
        from sklearn.calibration import CalibratedClassifierCV

        # sklearn >= 1.6 removed `cv='prefit'` in favor of `FrozenEstimator`.
        # Try the modern API first and fall back to the legacy form so the
        # module works across the supported sklearn range.
        try:
            from sklearn.frozen import FrozenEstimator

            self._calibrated = CalibratedClassifierCV(
                FrozenEstimator(base_model), method=self.method, cv=None
            )
        except ImportError:
            self._calibrated = CalibratedClassifierCV(
                base_model, method=self.method, cv="prefit"
            )

        self._calibrated.fit(X_calib, y_calib)
        self.classes_ = self._calibrated.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._calibrated is None:
            raise RuntimeError("SklearnCalibrator.fit() must be called first.")
        return self._calibrated.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TemperatureScaler:
    """Single-parameter softmax temperature scaling (Guo et al. 2017, ICML).

    Learns a scalar T > 0 that minimizes NLL on the calibration logits.
    Predictions are unchanged (argmax invariant to positive scaling) but
    confidences are softened (T > 1) when the model is overconfident, or
    sharpened (T < 1) when underconfident.

    Optimization is a 1-D LBFGS over log(T) (so T stays positive without
    a hard constraint).
    """

    def __init__(self, max_iter: int = 50):
        self.T = 1.0
        self.max_iter = max_iter

    def fit(self, logits_calib: np.ndarray, y_calib: np.ndarray) -> "TemperatureScaler":
        """Fit T by minimizing NLL on (logits_calib, y_calib).

        Args:
            logits_calib: Pre-softmax logits, shape (n, n_classes).
            y_calib: Integer class labels, shape (n,).
        """
        import torch

        logits_t = torch.tensor(np.asarray(logits_calib), dtype=torch.float64)
        y_t = torch.tensor(np.asarray(y_calib), dtype=torch.long)
        log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        nll = torch.nn.NLLLoss()
        optim = torch.optim.LBFGS(
            [log_T], lr=0.1, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optim.zero_grad()
            T = torch.exp(log_T)
            log_probs = torch.log_softmax(logits_t / T, dim=1)
            loss = nll(log_probs, y_t)
            loss.backward()
            return loss

        optim.step(closure)
        self.T = float(torch.exp(log_T).detach().cpu().item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.T

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and softmax.

        Note: input is logits (not probs) -- this is the signature divergence
        from `SklearnCalibrator` because temperature scaling is not well-defined
        on already-softmaxed probabilities.
        """
        return _softmax(self.transform_logits(logits))


class HierarchicalCalibrator:
    """Calibrates each stage of `HierarchicalClassifier` independently.

    The hierarchical model produces P(class) = P(relax) for the rest class,
    or P(active) * P(class | active) for active classes. If both Stage 1
    (binary rest/active) and Stage 2 (4-class active) are individually
    calibrated, the multiplicative composition remains a valid joint
    probability (each marginal is calibrated, so the product is the calibrated
    joint under the standard chain-rule decomposition).

    This class re-implements the prediction logic of HierarchicalClassifier
    using calibrated stage outputs.
    """

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        self.method = method
        self.stage1_cal: SklearnCalibrator | None = None
        self.stage2_cal: SklearnCalibrator | None = None
        self._active_label_inverse: dict[int, int] | None = None

    def fit(
        self,
        hier_model: HierarchicalClassifier,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> "HierarchicalCalibrator":
        """Fit Stage 1 and Stage 2 calibrators on the held-out set.

        Args:
            hier_model: Already-fitted HierarchicalClassifier.
            X_calib: Calibration features.
            y_calib: Calibration labels (in CLASS_NAMES indexing).
        """
        # Stage 1: binary rest/active labels.
        y_binary = (np.asarray(y_calib) != RELAX_IDX).astype(int)
        self.stage1_cal = SklearnCalibrator(self.method).fit(
            hier_model.stage1_model, X_calib, y_binary
        )

        # Stage 2: only active samples, remapped to 0..3.
        y_arr = np.asarray(y_calib)
        active_mask = y_arr != RELAX_IDX
        X_active = X_calib[active_mask]
        y_active = y_arr[active_mask]
        active_label_map = {orig: new for new, orig in enumerate(ACTIVE_CLASSES)}
        y_remapped = np.array([active_label_map[yi] for yi in y_active])

        self.stage2_cal = SklearnCalibrator(self.method).fit(
            hier_model.stage2_model, X_active, y_remapped
        )
        self._active_label_inverse = {
            new: orig for orig, new in active_label_map.items()
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Combine calibrated Stage 1 and Stage 2 outputs.

        Mirrors `HierarchicalClassifier.predict_proba` logic, but uses
        calibrated probabilities for both stages.
        """
        if self.stage1_cal is None or self.stage2_cal is None:
            raise RuntimeError("HierarchicalCalibrator.fit() must be called first.")

        n = X.shape[0]
        probs = np.zeros((n, len(CLASS_NAMES)))

        stage1 = self.stage1_cal.predict_proba(X)
        # By convention: column 0 = rest, column 1 = active.
        p_relax = stage1[:, 0]
        p_active = stage1[:, 1]

        stage2 = self.stage2_cal.predict_proba(X)

        probs[:, RELAX_IDX] = p_relax
        for new_idx, orig_idx in self._active_label_inverse.items():
            probs[:, orig_idx] = p_active * stage2[:, new_idx]
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
