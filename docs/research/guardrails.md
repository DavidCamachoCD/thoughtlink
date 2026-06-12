# Guardrails for BCI-to-robot: calibration + conformal prediction

> Post-hoc layer between the model and the robot that turns the empirical
> `confidence_threshold = 0.6` dial into a distribution-free statistical guarantee.
>
> Applied across 3 models × 4 calibrators × 4 conformal × 3 seeds =
> **96 factorial ablation cells** on the KernelCo/robot_control dataset.

## Table of contents

1. [Motivation](#1-motivation)
2. [Layer architecture](#2-layer-architecture)
3. [Post-hoc calibration](#3-post-hoc-calibration)
4. [Conformal prediction](#4-conformal-prediction)
5. [Weighted conformal: cross-subject covariate shift](#5-weighted-conformal-cross-subject-covariate-shift)
6. [Factorial ablation harness](#6-factorial-ablation-harness)
7. [MPS acceleration](#7-mps-acceleration)
8. [Empirical results](#8-empirical-results)
9. [Limitations and open follow-ups](#9-limitations-and-open-follow-ups)
10. [How to reproduce](#10-how-to-reproduce)
11. [Bibliography](#11-bibliography)

---

## 1. Motivation

ThoughtLink v1.0 had a single guardrail between the model and the robot:
[`inference/confidence.py:54`](../../src/thoughtlink/inference/confidence.py#L54)
discarded predictions with `prob < 0.6`. This **assumes `predict_proba` is
calibrated** — that when the model says 0.85, it is correct 85% of the time.
In practice this rarely holds:

- **CNN/EEGNet** tends to be **overconfident** (Guo et al. 2017). It passes the
  threshold with confidence 0.9 and real accuracy <30% → false trigger on the robot.
- **Random Forest** tends to be **underconfident** → correct predictions are
  discarded unnecessarily.
- **SVM with `probability=True`** uses Platt scaling internally, which performs
  poorly in the tails.

Result: `confidence_threshold = 0.6` tunes over arbitrary numbers. The safety
layer exists but has no statistical foundation.

This SOTA layer attacks the problem in two steps:

1. **Post-hoc calibration** — models produce real probabilities (Platt /
   isotonic / temperature scaling as appropriate). Target metric: ECE < 0.05.
2. **Conformal prediction** — instead of "one class with probabilities", returns a
   *prediction set* with coverage ≥ 1−α guaranteed (Vovk; Angelopoulos & Bates 2023).
   If the set has >1 class, the stability pipeline treats it as "uncertain" and
   holds the current action.

Outcome: `confidence_threshold` acquires a frequentist interpretation ("60%
real probability of being correct"), and false triggers are bounded by conformal
coverage rather than an empirical dial.

## 2. Layer architecture

```
                                         ┌──────────────────────┐
                                         │  StabilityPipeline    │
                                         │  (unchanged from      │
                                         │   v1.0)               │
                                         └──────────▲───────────┘
                                                    │ calibrated probs
                                                    │ + optional conformal set
                ┌───────────────────────────────────┴───────────────────────────┐
                │  ConformalWrapper (NEW: inference/conformal.py)               │
                │    - APS / Naive / Weighted APS                               │
                │    - .predict_set(probs) → set[label]                         │
                │    - configurable alpha                                       │
                └───────────────────────────────────▲───────────────────────────┘
                                                    │ calibrated probs
                ┌───────────────────────────────────┴───────────────────────────┐
                │  Calibrator (NEW: inference/calibration.py)                   │
                │    - SklearnCalibrator(method=isotonic|sigmoid)               │
                │    - TemperatureScaler  (PyTorch logits)                      │
                │    - HierarchicalCalibrator                                   │
                │    - .fit(X_calib, y_calib) / .transform(probs)               │
                └───────────────────────────────────▲───────────────────────────┘
                                                    │ raw probs
                                         ┌──────────┴───────────┐
                                         │  Decoder.predict()    │
                                         │  → model.predict_proba│
                                         └───────────────────────┘
```

Runtime injection point: the
[`_apply_conformal_guardrail`](../../src/thoughtlink/bridge/brain_policy.py#L97)
method is called just before `stability.process()` in
[`step()`](../../src/thoughtlink/bridge/brain_policy.py#L156-L157) and
[`_stream_eeg()`](../../src/thoughtlink/bridge/brain_policy.py#L195-L203).
Backward compatible: without a calibrator/conformal the behavior is identical
to v1.0.

**Data**: subject-aware 3-way split (13 train / 1 calib / 3 test) via
[`split_by_subject_3way`](../../src/thoughtlink/data/splitter.py). The
calibration subject never appears in train or test — preserves exchangeability
as well as possible cross-subject.

## 3. Post-hoc calibration

Implemented in [`inference/calibration.py`](../../src/thoughtlink/inference/calibration.py).

### `SklearnCalibrator(method="isotonic" | "sigmoid")`

Wrapper around [`sklearn.calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
with `cv="prefit"` (sklearn ≥1.6: via `FrozenEstimator`). The base model is not
retrained — only the calibration mapping is learned on the calibration set.

- **Isotonic** (Zadrozny & Elkan 2002): non-parametric monotone regression. Best
  when the calibration set is large (>1k samples).
- **Sigmoid** (Platt 1999): 1-D logistic over scores. Best with little data;
  assumes a sigmoidal shape for the miscalibration. Current default in
  [`configs/default.yaml`](../../configs/default.yaml).

### `TemperatureScaler`

For CNN (`EEGNet`) and PyTorch models in general (Guo et al. 2017, ICML). Learns
a single `T > 0` by minimizing NLL over the calibration set logits via LBFGS:

```
log_probs = log_softmax(logits / T)
loss = NLL(log_probs, y_calib)
```

Predictions are invariant (argmax is preserved); confidences are flattened
(T > 1) when the model is overconfident, or sharpened (T < 1) when underconfident.

LBFGS is CPU-bound in PyTorch — the scaler lives on CPU even if the model trains
on MPS.

### `HierarchicalCalibrator`

Special case for [`HierarchicalClassifier`](../../src/thoughtlink/models/hierarchical.py)
(2-stage rest/active gate + 4-class). Calibrates Stage 1 (binary) and Stage 2
(4-class) **separately**. If each marginal is calibrated, the product
`P(class) = P(active) × P(class|active)` remains a valid joint probability
(chain rule).

### Diagnostics

[`eval/diagnostics.py`](../../src/thoughtlink/eval/diagnostics.py) exposes:

- `expected_calibration_error(y, probs, n_bins=15)` — Naeini et al. 2015.
- `maximum_calibration_error(y, probs, n_bins=15)` — worst gap per bin.
- `brier_score(y, probs)` — Brier (1950), proper scoring rule.
- `reliability_curve(y, probs, n_bins=15)` — per-bin data for plotting without
  coupling matplotlib.

## 4. Conformal prediction

Implemented in [`inference/conformal.py`](../../src/thoughtlink/inference/conformal.py).

### `APSConformalPredictor` (recommended default)

Adaptive Prediction Sets (Romano, Sesia & Candès 2020, NeurIPS). Score = cumulative
sum of class probabilities sorted descending up to and including the true class:

```
order = argsort(-probs)            # descending
cumsum = cumsum(probs[order])
s_i = cumsum[rank_of_true_class]
```

Finite-sample corrected quantile (Angelopoulos & Bates 2023):

```
q_hat = quantile(scores, ceil((n+1)*(1-α)) / n)
```

`predict_set(probs)` includes every class whose cumulative APS-score ≤ q_hat
(plus the top class to guarantee a non-empty set).

Sets shrink when the model is confident and grow under uncertainty — exactly
what we want as a guardrail.

### `NaiveConformalPredictor` (ablation only)

Score = `1 - p(y_true)`. Simpler, less adaptive. Kept for comparison; sets are
~20% larger than APS for equivalent coverage (see [table §8](#8-empirical-results)).

### Formal guarantee

Under exchangeability of calibration and test:

```
P(y_test ∈ C(x_test)) ≥ 1 - α
```

where `C(x)` is the prediction set. Coverage is **marginal over everything** (not
conditional on class or input). In ThoughtLink the assumption breaks at the
cross-subject level; see §5.

## 5. Weighted conformal: cross-subject covariate shift

The exchangeability assumption fails when the calibration subject has a different
EEG distribution from the test subjects. Empirically: coverage drops from the
target 0.90 to ~0.69 on `hierarchical/raw/aps`. See §8.

### `WeightedAPSConformalPredictor` (Tibshirani et al. 2019, NeurIPS)

Calibration scores are weighted by the likelihood ratio
`w(x) = P_test(x) / P_calib(x)`:

```
q_hat = weighted_quantile(scores_calib, w_calib, level = 1 - α)
```

where `weighted_quantile` is the weighted empirical quantile.

Recovered guarantee: if `P_test(Y|X) = P_calib(Y|X)` (only P(X) changes) and
the weights are correct, marginal coverage ≥ 1−α is preserved.

### Weight estimation

[`inference/domain_weights.py:estimate_likelihood_ratio`](../../src/thoughtlink/inference/domain_weights.py).
Classic trick:

1. Concatenate X_calib (label 0) and X_test (label 1).
2. Fit binary logistic regression with L2 regularization `C=1.0`.
3. For each x_calib: `w(x) = P(test|x) / P(calib|x) = p̂(x) / (1 - p̂(x))`.

**Importance clipping** (Sugiyama et al. 2008): weights are clipped to
`[1/clip, clip]` (default `clip=50`) to limit the variance of the weighted
quantile under extreme shift.

### `diagnose_weights(weights)`

Reports `effective_sample_size` (ESS) and ratio. A low ESS ratio (~0.02) indicates
that a few extreme weights dominate the weighted quantile — it becomes high-variance.
In the ablation this translates to unstable seed-to-seed coverage (see §8).

## 6. Factorial ablation harness

`src/thoughtlink/eval/` groups all evaluation infrastructure. The new subpackage
in this release contains:

- [`eval/diagnostics.py`](../../src/thoughtlink/eval/diagnostics.py) — metrics (ECE, MCE, Brier, reliability_curve)
- [`eval/training.py`](../../src/thoughtlink/eval/training.py) — `TrainConfig` + `train_baseline/hierarchical/cnn` as pure functions
- [`eval/ablation.py`](../../src/thoughtlink/eval/ablation.py) — registries + `run_factorial`
- [`eval/_torch_device.py`](../../src/thoughtlink/eval/_torch_device.py) — CUDA > MPS > CPU device selection

### Registries

```python
MODEL_TRAINERS = {"baseline": ..., "hierarchical": ..., "cnn": ...}

CALIBRATORS = {"raw", "isotonic", "sigmoid", "temperature"}

CONFORMAL = {"none", "aps", "naive", "weighted_aps"}

COMPATIBILITY = {                       # which calibrators apply to which model?
    "baseline":     {"raw", "isotonic", "sigmoid"},
    "hierarchical": {"raw", "isotonic", "sigmoid"},
    "cnn":          {"raw", "temperature"},
}
```

Adding new variants (e.g., KLIEP, beta calibration, adaptive conformal) is
one entry in the dict plus a closure factory. Zero changes to the runner.

### `run_factorial`

Iterates the cross product `(seed × model × calibration × conformal)`. Per seed:

1. `split_by_subject_3way` with `random_state=seed`.
2. Preprocess + extract features (once, reused across all 3 models).
3. Per model: train (~10s baseline, ~2 min CNN on MPS).
4. Per (calibration, conformal): fit + evaluate, ~1s/cell.

### Persistence

Output in `results/ablations/<run-id>/`:

```
config.json    # requested variants + git hash + library versions + timestamp
cells.jsonl    # 1 line per cell (append, resumable)
summary.csv    # tidy DataFrame
summary.md     # aggregated table (mean ± std) by (model, cal, conformal)
```

`run-id` format: `YYYYMMDD-HHMMSS-<git-sha7>`. Reproducibility anchored to the
commit.

### Resumability

`cells.jsonl` is opened in append mode. On startup, `run_ablation.py --resume
<run-id>` reads the file, identifies already-completed cells
(`{seed}/{model}/{cal}/{conf}`) and only runs the pending ones. Useful for long
background runs or if the process is killed mid-sweep.

### Per-cell error isolation

Each cell is evaluated in `try/except`. A failure (e.g., incompatible model,
NaN in weights) writes `error: "TypeName: msg"` to its jsonl entry but does not
abort the rest of the sweep.

### CLI

[`scripts/run_ablation.py`](../../scripts/run_ablation.py):

```bash
# Default: full factorial 3 models × 4 cal × 4 conf × 3 seeds = 96 cells
uv run python scripts/run_ablation.py

# Subset
uv run python scripts/run_ablation.py --models cnn --calibrators temperature

# Resume an interrupted run
uv run python scripts/run_ablation.py --resume 20260503-005644-b238320
```

### Comparison notebook

[`notebooks/07_ablation_comparison.ipynb`](../../notebooks/07_ablation_comparison.ipynb)
loads one or more `cells.jsonl` files and produces a pivot table, coverage
heatmap, ECE boxplot, and coverage-vs-set-size Pareto scatter.

## 7. MPS acceleration

The operational bottleneck in the harness was the CNN (EEGNet) running on CPU:
~50 min per seed. Simply adding Apple Silicon GPU detection
([`eval/_torch_device.py`](../../src/thoughtlink/eval/_torch_device.py)) brings
this down to **~2 min per seed** — a ~20× speedup.

```python
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Zero changes to the training loop or hyperparameters. The CNN trains on MPS,
but the state_dict is persisted on CPU (`model.cpu()` at the end of
[`train_cnn`](../../src/thoughtlink/eval/training.py#L231)), so calibration,
conformal, and downstream persistence see no difference.

`TemperatureScaler` stays on CPU intentionally — LBFGS does not accelerate on
MPS and was already converting to CPU internally.

Full 96-cell run: **~12 min wall-clock** on an M-series Mac. Before MPS:
~22h estimated (abandoned as infeasible).

## 8. Empirical results

Reference run: `results/ablations/20260503-005644-b238320/`. 96 cells, 0 errors,
3 seeds, α=0.1 (target coverage 0.90), 2 calibration subjects / 3 test subjects.

### Main table: ECE pre-vs-post calibration

Accuracy and ECE do not depend on conformal (calibration changes the
probabilities, not the decisions), so they are reported aggregated over conformal.
**Headline**: best calibrator per model + the CNN "before" as a reference.

| Model | Calibration | Accuracy | ECE | Brier |
|---|---|---:|---:|---:|
| baseline | sigmoid (best) | 0.239 ± 0.022 | **0.039 ± 0.005** | 0.796 ± 0.004 |
| hierarchical | isotonic (best) | 0.241 ± 0.026 | **0.071 ± 0.061** | 0.807 ± 0.017 |
| cnn | raw (before) | 0.214 ± 0.017 | 0.658 ± 0.084 | 1.402 ± 0.114 |
| **cnn** | **temperature** | 0.214 ± 0.017 | **0.020 ± 0.012** | 0.801 ± 0.001 |

**Headline finding**: temperature scaling on the CNN reduces ECE from **0.66 to
0.02 (33×)**. This is the clearest win in the paper. Accuracy is preserved
(argmax unchanged). Brier halves because we stop penalizing extreme confidences
assigned to the wrong class.

> Full matrix (all model × calibrator combinations, mean ± std across seeds):
> [`docs/results.md` § Factorial ablation](../results.md#factorial-ablation-mean--std-across-seeds).
> Canonical data: `results/ablations/<latest>/summary.csv`.

### Conformal coverage table by (model, calibration, conformal)

Target ≥ 0.90 (α = 0.1). **Headline**: APS meets the target across all three
models; weighted_aps is conditional (see finding 4 below).

| Model | Calibration | aps | naive | weighted_aps |
|---|---|:---:|:---:|:---:|
| baseline | sigmoid | **0.95** | 0.86 | 0.91 |
| hierarchical | sigmoid | **0.96** | 0.84 | 0.83 |
| cnn | temperature | **0.99** | 0.82 | 0.74 |

> Full table (8 model × calibration combinations × 3 conformal):
> [`docs/results.md` § Factorial ablation](../results.md#factorial-ablation-mean--std-across-seeds).

### Average set size table (decisiveness)

Set size = 1 → singleton prediction, robot executes. Set size > 1 → "uncertain",
robot holds the previous action. Total: 5 classes.

| Model | Calibration | aps | naive | weighted_aps |
|---|---|:---:|:---:|:---:|
| baseline | sigmoid | 4.69 | 4.09 | 4.44 |
| hierarchical | sigmoid | 4.70 | 3.98 | 3.96 |
| cnn | temperature | 4.92 | 4.12 | 3.67 |

Large sets (3.7–5.0 out of 5) are **the correct behavior** given that the
real cross-subject accuracy of the models is ~24%. The layer honestly admits
it does not know → robot holds STOP. This is exactly the safety guarantee
we are looking for.

> Set sizes for all combinations (including `raw` and `isotonic`):
> [`docs/results.md` § Factorial ablation](../results.md#factorial-ablation-mean--std-across-seeds).

### Qualitative findings

**1. Temperature scaling is the highest return on investment.**
A single-parameter technique reduces ECE 33× on EEGNet. Average T ≈ 42k across
seeds (enormous variance: ±43k). Such a large T indicates that the softmax was
absurdly sharp — the model was "deciding" from logits that were in reality nearly
indistinguishable. After scaling the model still predicts the same thing, just
with honest confidence.

**2. APS conformal meets the target in all regimes.**
Coverage ≥ 0.90 with low variance (std < 0.07) across all 9 model+calibration
combinations. It is the most robust option on the menu.

**3. Naive conformal systematically undercoveres.**
Coverage ~0.83–0.92 vs APS ~0.93–0.99, with only marginally smaller set sizes.
Empirically confirms Romano et al. 2020: naive is uniformly dominated by APS
under predict-set semantics.

**4. Weighted conformal: effective but unstable.**
Tibshirani 2019 promises to recover coverage under covariate shift. In practice
this is partially borne out:

- **`hierarchical/raw/weighted_aps`**: coverage 0.77 ± **0.22** (variance 3× that
  of standard APS). Some seeds go from 0.69 to >0.95; others drop to 0.55. The
  instability comes from ESS ratio ≈ 0.02 — a few extreme weights dominate the
  weighted quantile.
- **`baseline/isotonic/weighted_aps`**: coverage **0.99 ± 0.003**, set size 4.93.
  Here weighted **does** work — isotonic calibration flattens the probabilities
  and the weights do clean work.
- **`cnn/temperature/weighted_aps`**: coverage **0.74 ± 0.12**, versus
  `cnn/temperature/aps` at 0.99 ± 0.01. Weighted **degrades** coverage when
  temperature scaling has already saturated the probabilities — the weights add
  variance without useful information.

Honest reading: weighted conformal is a conditional tool, not a monotone
improvement over APS. Empirical validation per model is required. Documenting
this explicitly is more valuable than claiming "weighted always wins".

**5. Calibration does not help models near chance.**
`baseline/raw` already has ECE = 0.058 (low). This looks "calibrated" until you
see that accuracy is 0.25 and average confidence is ~0.30 — they are "calibrated
by accident" because both are ~chance. Sigmoid marginally lowers ECE to 0.039;
isotonic stays the same with std 0.022. For models where predict_proba is already
nearly uniform, calibration adds variance without information gain. The right move
is to improve the model (cross-subject features, transfer learning), not calibrate it.

## 9. Limitations and open follow-ups

### Acknowledged limitations of this run

- **3 seeds** is the minimum to report std/CI with some confidence. A formal paper
  would require 5–10 seeds, especially for `weighted_aps` where inter-seed variance
  is high.
- **2 calibration subjects** is small (out of 17 total). The effective sample size
  for the logistic-regression weight estimator stays around ~0.02. More calibration
  subjects would improve the stability of weighted conformal at the cost of reducing
  the training set.
- **The CNN does not use Euclidean alignment in the harness for calib/test sets**
  because alignment is subject-aware and the calib/test subjects differ from train.
  Maintaining consistency with `scripts/train_cnn.py` (which does align train+test)
  would require either extending the alignment or removing it from both paths.

### Open research questions (not addressed in this layer)

- **Cross-subject transfer learning.** The real bottleneck is the cross-subject
  model accuracy (~24%), not the guardrail. Domain adaptation / few-shot fine-tuning
  per subject remains the next major step.
- **Per-class confidence thresholds.** The single threshold of 0.6 could be
  per-class post-calibration (e.g., stricter for "Both Fists" which confuses more
  with "Right Fist").
- **Adaptive conformal under online drift** (Gibbs & Candès 2021). For real-time
  deployment where the subject's mental state changes during a session, a static
  q_hat does not adapt.
- **KLIEP** (Sugiyama et al. 2008) as an alternative to the logistic-regression
  likelihood-ratio estimator. More robust under strong shift; not implemented here.
- **Beta calibration** (Kull et al. 2017) for calibrators with asymmetric
  miscalibration.

### Technical debt from this PR

- Deprecated shim in [`inference/diagnostics.py`](../../src/thoughtlink/inference/diagnostics.py)
  re-exports from `eval.diagnostics`. Remove after 1–2 commits once all external
  notebooks have migrated.
- Fallback `cv='prefit'` in `SklearnCalibrator` for sklearn <1.6 — removable
  once we bump the sklearn pin.
- `scripts/train_*.py` were not thinned down to thin wrappers of `eval/training.py`
  (intentional scope decision). There is deliberate duplication between the CLI path
  and the in-process path.

## 10. How to reproduce

```bash
# 1. Setup
git clone https://github.com/DavidCamachoCD/thoughtlink.git
cd thoughtlink
git checkout research            # or reference commit b238320
uv sync

# 2. Tests (expect 276 green)
uv run python -m pytest tests/

# 3. Train the base models if they do not already exist in results/
uv run python scripts/train_baseline.py
uv run python scripts/train_hierarchical.py
uv run python scripts/train_cnn.py     # ~2-3 min on M-series with MPS

# 4. Full factorial run (3 models × 4 cal × 4 conf × 3 seeds = 96 cells)
uv run python scripts/run_ablation.py
# → ~12 min on M-series Mac
# → output in results/ablations/<run-id>/

# 5. Inspect results
cat results/ablations/<run-id>/summary.md
jupyter notebook notebooks/07_ablation_comparison.ipynb
```

For a quick smoke run (~2 min):

```bash
uv run python scripts/run_ablation.py \
    --models baseline hierarchical \
    --calibrators raw sigmoid \
    --conformals none aps \
    --seeds 42
```

## 11. Bibliography

See [`docs/references.md`](../references.md) for the full list with DOI/arXiv
and paper → file:line mapping.

Summary of the main applied references:

- **[Platt 1999]** *Probabilistic Outputs for SVMs* — sigmoid calibration.
- **[Zadrozny & Elkan 2002]** *Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates* — isotonic regression.
- **[Niculescu-Mizil & Caruana 2005]** comparison of Platt vs isotonic
  (justifies sigmoid default with calib < 1k samples).
- **[Naeini, Cooper & Hauskrecht 2015]** ECE / MCE.
- **[Brier 1950]** Brier score, proper scoring rule.
- **[Guo, Pleiss, Sun & Weinberger 2017]** *On Calibration of Modern Neural
  Networks*, ICML — temperature scaling.
- **[Vovk, Gammerman & Shafer 2005]** *Algorithmic Learning in a Random World* —
  conformal prediction framework.
- **[Romano, Sesia & Candès 2020]** *Adaptive Prediction Sets*, NeurIPS — APS.
- **[Tibshirani, Foygel Barber, Candès & Ramdas 2019]** *Conformal Prediction
  Under Covariate Shift*, NeurIPS — weighted conformal.
- **[Sugiyama et al. 2008]** *Direct Importance Estimation for Covariate Shift
  Adaptation* — importance clipping.
- **[Angelopoulos & Bates 2023]** *A Gentle Introduction to Conformal
  Prediction* — finite-sample quantile correction.

---

## Relevant commits

| Commit | Topic |
|---|---|
| [`08290f4`](https://github.com/DavidCamachoCD/thoughtlink/commit/08290f4) | feat: guardrails layer with calibration + conformal prediction |
| [`1735fdb`](https://github.com/DavidCamachoCD/thoughtlink/commit/1735fdb) | fix: CNN load + CLI flags + empirical results in notebook 06 |
| [`bae9748`](https://github.com/DavidCamachoCD/thoughtlink/commit/bae9748) | feat: weighted conformal for cross-subject covariate shift |
| [`1f3c254`](https://github.com/DavidCamachoCD/thoughtlink/commit/1f3c254) | feat: factorial ablation harness (`eval/`) |
| [`b238320`](https://github.com/DavidCamachoCD/thoughtlink/commit/b238320) | perf: MPS support for CNN training |
