# References — Calibration & Conformal Guardrails

Bibliography for the post-hoc confidence calibration + conformal prediction
layer added in the `research` branch. Each entry lists the paper, venue,
DOI/arXiv, what we take from it, and the file(s) where it is implemented.

---

## Calibration

### [Platt 1999] Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods
- **Venue:** *Advances in Large Margin Classifiers*, MIT Press, pp. 61-74.
- **What we take:** Sigmoid calibration (commonly called "Platt scaling"):
  fit a 1-D logistic on `(score, label)` pairs to map scores to probabilities.
- **Where:** [`src/thoughtlink/inference/calibration.py`](../src/thoughtlink/inference/calibration.py) — `SklearnCalibrator(method="sigmoid")`. Used as fallback
  for small calibration sets.

### [Zadrozny & Elkan 2002] Transforming Classifier Scores into Accurate Multiclass Probability Estimates
- **Venue:** KDD 2002, pp. 694-699.
- **DOI:** [10.1145/775047.775151](https://doi.org/10.1145/775047.775151)
- **What we take:** Isotonic regression for multi-class calibration via
  one-vs-rest decomposition. This is the algorithm underlying
  `sklearn.calibration.CalibratedClassifierCV(method="isotonic")`.
- **Where:** [`src/thoughtlink/inference/calibration.py`](../src/thoughtlink/inference/calibration.py) — `SklearnCalibrator(method="isotonic")` (default).

### [Niculescu-Mizil & Caruana 2005] Predicting Good Probabilities with Supervised Learning
- **Venue:** ICML 2005, pp. 625-632.
- **DOI:** [10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
- **What we take:** Empirical evidence on when sigmoid vs isotonic is preferable
  (sigmoid wins with very small calibration sets; isotonic wins as the calibration
  set grows past ~1k samples). Justifies our choice of isotonic as default — at
  ~80 calibration samples per subject in 5 classes the comparison is close, so
  the config exposes both methods.
- **Where:** Justification comment in [`configs/default.yaml`](../configs/default.yaml) and discussion in `notebooks/06_confidence_calibration.ipynb`.

### [Naeini, Cooper & Hauskrecht 2015] Obtaining Well Calibrated Probabilities Using Bayesian Binning
- **Venue:** AAAI 2015.
- **What we take:** Formal definition of **Expected Calibration Error (ECE)**
  and **Maximum Calibration Error (MCE)** via equal-width binning of the
  top-class confidence.
- **Where:** [`src/thoughtlink/inference/diagnostics.py`](../src/thoughtlink/inference/diagnostics.py) — `expected_calibration_error()`, `maximum_calibration_error()`.

### [Guo, Pleiss, Sun & Weinberger 2017] On Calibration of Modern Neural Networks
- **Venue:** ICML 2017.
- **arXiv:** [1706.04599](https://arxiv.org/abs/1706.04599)
- **What we take:** (1) Empirical demonstration that modern deep networks are
  systematically overconfident. (2) **Temperature scaling**: optimise a single
  scalar `T > 0` on the calibration set's NLL; divide all logits by `T` before
  softmax. Predictions are unchanged (argmax invariant), only confidences move.
- **Where:** [`src/thoughtlink/inference/calibration.py`](../src/thoughtlink/inference/calibration.py) — `TemperatureScaler`. Supported by the new `return_logits=True` path in [`src/thoughtlink/models/cnn.py`](../src/thoughtlink/models/cnn.py) and [`src/thoughtlink/models/temporal.py`](../src/thoughtlink/models/temporal.py).

### [Kull, Silva Filho & Flach 2017] Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers
- **Venue:** AISTATS 2017.
- **What we take:** Reference for why Platt's sigmoid is a poor fit when miscalibration
  is asymmetric (S-shape vs. L/J-shape), motivating isotonic as the safer general default.
  Beta calibration is mentioned but not implemented in the first iteration.
- **Where:** Discussion section of `notebooks/06_confidence_calibration.ipynb`.

---

## Conformal prediction

### [Vovk, Gammerman & Shafer 2005] Algorithmic Learning in a Random World
- **Venue:** Springer.
- **What we take:** Foundational framework for conformal prediction and the
  exchangeability-based proof of marginal coverage `P(y ∈ C(x)) ≥ 1 − α`.
- **Where:** Module docstring of [`src/thoughtlink/inference/conformal.py`](../src/thoughtlink/inference/conformal.py).

### [Romano, Sesia & Candès 2020] Classification with Valid and Adaptive Coverage
- **Venue:** NeurIPS 2020.
- **arXiv:** [2006.02544](https://arxiv.org/abs/2006.02544)
- **What we take:** **Adaptive Prediction Sets (APS)**. Score function: rank
  classes by descending probability and accumulate until the true class is
  reached. The resulting calibration scores produce conformal prediction sets
  that are *adaptive* — small when the model is confident, larger under genuine
  uncertainty — while preserving marginal coverage. We use APS as the default
  score in our conformal layer.
- **Where:** [`src/thoughtlink/inference/conformal.py`](../src/thoughtlink/inference/conformal.py) — `APSConformalPredictor` (`_aps_score_calib`, `_aps_score_test`, `predict_set`).

### [Angelopoulos & Bates 2023] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification
- **Venue:** *Foundations and Trends in Machine Learning*, vol. 16, no. 4.
- **arXiv:** [2107.07511](https://arxiv.org/abs/2107.07511)
- **What we take:** (1) The finite-sample-corrected calibration quantile level
  `⌈(n + 1)(1 − α)⌉ / n`, required for valid coverage with small calibration
  sets. (2) The recommendation to apply post-hoc calibration *before* fitting
  conformal, since well-calibrated probabilities lead to smaller prediction
  sets without changing the coverage guarantee.
- **Where:** Comments in `APSConformalPredictor.fit()` (quantile level) and in
  `scripts/calibrate_models.py` (calibration-then-conformal pipeline order).

---

## Supporting / context

### [Brier 1950] Verification of Forecasts Expressed in Terms of Probability
- **Venue:** *Monthly Weather Review*, 78(1), pp. 1-3.
- **What we take:** Brier score, a strictly proper scoring rule for
  probabilistic predictions. Decomposable into reliability + resolution +
  uncertainty (Murphy 1973). Reported alongside ECE because it penalises
  both miscalibration and lack of resolution.
- **Where:** [`src/thoughtlink/inference/diagnostics.py`](../src/thoughtlink/inference/diagnostics.py) — `brier_score()`.

### scikit-learn — `sklearn.calibration.CalibratedClassifierCV` and `sklearn.isotonic.IsotonicRegression`
- **What we take:** Reference implementations of Platt scaling and isotonic
  regression with `cv='prefit'` support — the calibration model is fitted on a
  held-out set without retraining the underlying estimator.
- **Where:** Underlying implementation used by `SklearnCalibrator` and
  `HierarchicalCalibrator` in [`src/thoughtlink/inference/calibration.py`](../src/thoughtlink/inference/calibration.py).
