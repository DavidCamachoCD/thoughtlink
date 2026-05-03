# Changelog

All notable changes to ThoughtLink will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned according to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-05-03

Capa de guardrails post-hoc + harness de ablación factorial. La narrativa
completa, motivación y resultados están en
[`docs/research/guardrails.md`](docs/research/guardrails.md).

### Added — Guardrails layer (`src/thoughtlink/inference/`)

- `inference/calibration.py`: `SklearnCalibrator` (isotonic / sigmoid),
  `TemperatureScaler` (Guo et al. 2017), `HierarchicalCalibrator` que
  calibra Stage 1 y Stage 2 por separado preservando el chain rule. (08290f4)
- `inference/conformal.py`: `APSConformalPredictor` (Romano et al. 2020) +
  `NaiveConformalPredictor` para ablación. (08290f4)
- `inference/conformal.py`: `WeightedAPSConformalPredictor` (Tibshirani,
  Foygel Barber, Candès & Ramdas 2019) para covariate shift. (bae9748)
- `inference/domain_weights.py`: estimador de likelihood ratio por logistic
  regression + importance clipping (Sugiyama et al. 2008). (bae9748)
- `bridge/brain_policy.py`: parámetros opcionales `calibrator` y `conformal`,
  backward compatible. Si el set conformal tiene >1 clase, mantiene la
  acción anterior. (08290f4)
- `models/cnn.py` + `models/temporal.py`: flag `return_logits` para exponer
  los logits pre-softmax (necesario para temperature scaling). (08290f4)
- `data/splitter.py`: `split_by_subject_3way` (13 train / 1 calib / 3 test).
  (08290f4)

### Added — Evaluation infrastructure (`src/thoughtlink/eval/`)

- `eval/diagnostics.py`: ECE, MCE, Brier, reliability_curve. Movido desde
  `inference/diagnostics.py` (shim de re-export con DeprecationWarning).
  (1f3c254)
- `eval/training.py`: `TrainConfig` + `train_baseline` / `train_hierarchical`
  / `train_cnn` como funciones puras in-process. (1f3c254)
- `eval/ablation.py`: `AblationCell` + registries (`MODEL_TRAINERS`,
  `CALIBRATORS`, `CONFORMAL`, `COMPATIBILITY`) + `run_factorial`. Output
  versionado a `results/ablations/<run-id>/{config.json, cells.jsonl,
  summary.csv, summary.md}`. Append-mode JSONL = resumable via
  `--resume <run-id>`. Per-cell error isolation. (1f3c254)
- `eval/_torch_device.py`: `select_device()` con prioridad CUDA > MPS > CPU.
  (b238320)

### Added — Scripts

- `scripts/calibrate_models.py`: pipeline end-to-end calibración + conformal
  con flags `--method`, `--alpha`, `--n-calib-subjects`, `--weighted`.
  (08290f4 + bae9748)
- `scripts/run_ablation.py`: CLI factorial con todos los axes + `--resume`.
  Default = full factorial (96 cells, ~12 min en M-series). (1f3c254)

### Changed

- `inference/diagnostics.py`: ahora es un shim deprecated de
  `eval.diagnostics`. Se borrará en futuras releases. (1f3c254)
- `eval/training.py` + `scripts/train_cnn.py`: detección de device
  CUDA-or-CPU reemplazada por `select_device()` MPS-aware. **Speedup ~20×**
  en entrenamiento del CNN (~50 min/seed → ~2 min/seed). Cero cambios
  metodológicos: epochs, patience, augmentation idénticos. (b238320)
- `configs/default.yaml`: nuevas secciones `calibration:`, `conformal:`,
  `data.split_3way:`. Default `sklearn_method: "sigmoid"` per
  Niculescu-Mizil & Caruana 2005 con calib < 1k samples. (08290f4 + bae9748)
- `scripts/run_demo.py`: flag `--calibrated` para usar artefactos calibrados
  en runtime. (08290f4)
- `scripts/benchmark_latency.py`: entradas para overhead de calibrador y
  conformal. (08290f4)

### Fixed

- `scripts/calibrate_models.py`: hyperparams del checkpoint EEGNet (`f1=16,
  f2=32, d=2`) coinciden con `scripts/train_cnn.py`. (1735fdb)
- `inference/calibration.py`: usa `FrozenEstimator` con sklearn ≥1.6
  (`cv='prefit'` se mantiene como fallback para versiones más viejas).
  (08290f4)

### Empirical findings (run referencia: `20260503-005644-b238320`)

96 cells, 0 errores, 3 seeds (42, 123, 456), α=0.1, 2 sujetos calibración,
3 sujetos test. Detalles completos en
[`docs/research/guardrails.md`](docs/research/guardrails.md#8-resultados-empíricos)
y `notebooks/07_ablation_comparison.ipynb`.

- **Temperature scaling sobre CNN**: ECE 0.658 → 0.020 (33× reducción).
  Validación clean del paper de Guo et al. 2017 sobre EEGNet.
- **APS conformal cumple cobertura ≥0.90** (target α=0.1) en las 9
  combinaciones de modelo + calibración con std < 0.07.
- **Naive < APS** confirmado: cobertura 0.83-0.92 vs 0.93-0.99 con set sizes
  apenas más pequeños. Naive queda como ablación, no como producción.
- **Weighted conformal: condicional, no monótono**. Cobertura recuperada en
  `baseline/isotonic/weighted_aps` (0.99 ± 0.003) pero degradada en
  `cnn/temperature/weighted_aps` (0.74 ± 0.12). ESS ratio ≈ 0.02 — pesos
  extremos dominan. Documentado como hallazgo honesto.
- **Calibración no ayuda a modelos cerca de chance** (`baseline` con accuracy
  ~0.25): ECE pre 0.058, post-sigmoid 0.039 — mejora marginal porque las
  probas ya estaban "calibradas por accidente" (≈uniformes).

### Documentation

- `docs/research/guardrails.md`: writeup técnico completo (~600 líneas) con
  motivación, arquitectura, métricas, hallazgos empíricos, limitaciones y
  cómo reproducir.
- `docs/references.md`: bibliografía completa con DOI/arXiv y mapeo paper →
  archivo:línea. +Tibshirani 2019, +Sugiyama 2008. (bae9748)
- `notebooks/06_confidence_calibration.ipynb`: workflow end-to-end con
  reliability diagrams. (08290f4)
- `notebooks/07_ablation_comparison.ipynb`: comparador de runs con tabla
  pivote, heatmap de cobertura, boxplot de ECE, scatter Pareto. (1f3c254)

### Tests

- `tests/test_calibration.py`: 9 tests para los 3 calibradores. (08290f4)
- `tests/test_conformal.py`: 15 tests, 3 predictores conformal +
  `_weighted_quantile`. (08290f4 + bae9748)
- `tests/test_diagnostics.py`: 10 tests. (08290f4)
- `tests/test_domain_weights.py`: 7 tests para likelihood-ratio estimator.
  (bae9748)
- `tests/test_ablation.py`: 13 tests para registries + persistencia +
  resumability. (1f3c254)
- `tests/test_torch_device.py`: 4 tests, smoke MPS sobre EEGNet. (b238320)
- Tests añadidos a archivos pre-existentes: `test_brain_policy.py` (+2 para
  guardrail wiring), `test_data_splitter.py` (+4 para `split_by_subject_3way`).
- Total: **270 → 276 tests, 100% pasando**. (6 archivos nuevos con 58 tests +
  6 tests añadidos a archivos existentes; el delta neto refleja también
  refactors menores en otros tests.)

### Performance

- CNN training en M-series Mac: 50 min → 2 min con MPS (~20×). (b238320)
- Run factorial completo: 22h (CPU, abandonado) → 12 min (MPS). (b238320)
- Overhead runtime de la capa de guardrails: <1ms por predicción
  (calibración: matrix-mul O(K²) con K=5; conformal: lookup en cuantil
  pre-computado).

---

## [1.0.0] - 2026-02-08

### Added

#### Models
- `models/temporal.py`: GRU-based temporal model for sequential EEG feature decoding with bidirectional support
- `create_sequences()` helper to group consecutive feature windows into sequences

#### Data
- `data/dataset.py`: PyTorch `EEGDataset` wrapper compatible with DataLoader and sklearn

#### Scripts
- `scripts/export_onnx.py`: Export trained models (CNN + sklearn) to ONNX format with verification via onnxruntime

#### Notebooks
- `notebooks/02_feature_engineering.ipynb`: Feature separability analysis with t-SNE, band power distributions, feature importance, and per-class heatmaps
- `notebooks/04_wavelet_analysis.ipynb`: DWT wavelet feature evaluation (138 dimensions)
- `notebooks/05_wavelet_vs_baseline_comparison.ipynb`: Side-by-side comparison of standard vs wavelet features

#### Tests
- `tests/test_temporal.py`: 12 tests for TemporalEEGNet and create_sequences
- `tests/test_dataset.py`: 8 tests for EEGDataset (DataLoader compatibility, transforms)

#### Dependencies
- Added `skl2onnx>=0.7` for sklearn-to-ONNX conversion

### Changed
- Updated `pyproject.toml` to v1.0.0
- `bridge/mujoco_controller.py`: Graceful handling when `bri` package is not installed (lazy import)
- Updated `notebooks/01_data_exploration.ipynb` and `03_model_comparison.ipynb` with execution outputs

### Fixed
- MuJoCo controller tests no longer crash when `bri` is not installed
- Deduplicated npz files in loader to prevent double-counting
- Handle NaN sensor dropouts in EEG preprocessing

### Notes
- Notebooks 02/04/05 re-executed with the full 17-subject dataset (1395 samples,
  14 train / 3 test subjects) after the loader deduplication and NaN-handling fixes.
- Final test suite: 212 tests across 18 files, all passing.

---

## [0.4.0] - 2026-02-07

### Added

#### Integration
- `bridge/brain_policy.py`: BrainPolicy orchestrator — main loop from brain signals to robot actions with simulated real-time streaming
- `bridge/orchestrator.py`: Multi-robot dispatch with deduplication and failure tracking
- `bridge/mujoco_controller.py`: MuJoCo controller wrapping `bri` for Unitree G1 humanoid
- `scripts/run_demo.py`: End-to-end demo script with CLI args, live terminal output, color-coded actions, and summary statistics
- `scripts/run_mujoco_demo.py`: Brain-to-robot MuJoCo demo

#### Visualization
- `viz/dashboard.py`: Streamlit real-time dashboard with EEG traces, probability bars, action timeline, and step log
- `viz/temporal_stability.py`: Publication-ready plots — action timeline, confidence trace, probability heatmap, and combined 3-panel report
- `viz/latent_viz.py`: t-SNE/UMAP embedding visualization and feature importance analysis

---

## [0.1.0] - 2026-02-07

### Added

#### Data Pipeline
- `data/loader.py`: HuggingFace dataset download and `.npz` file parsing
- `data/splitter.py`: Subject-aware train/test splitting to prevent data leakage
- `configs/default.yaml`: Centralized hyperparameters for the full pipeline

#### Preprocessing
- `preprocessing/eeg.py`: EEG preprocessing with MNE-Python (bandpass 1-40Hz, CAR)
- `preprocessing/nirs.py`: TD-NIRS baseline correction and stimulus extraction
- `preprocessing/windowing.py`: 1s sliding windows with 50% overlap

#### Feature Extraction
- `features/eeg_features.py`: Band power (4 bands x 6 channels), Hjorth parameters, time-domain stats, DWT wavelets
- `features/csp_features.py`: Common Spatial Patterns via MNE
- `features/nirs_features.py`: NIRS temporal features (mean, peak, slope) with PCA reduction
- `features/fusion.py`: EEG + NIRS feature concatenation

#### Models
- `models/baseline.py`: 4 sklearn pipelines (LogReg, SVM Linear, SVM RBF, Random Forest)
- `models/hierarchical.py`: 2-stage classifier (rest-vs-active gate + 4-class decoder)
- `models/cnn.py`: Compact EEGNet CNN in PyTorch (~2-4K params)
- `models/ensemble.py`: Soft/hard voting ensemble

#### Inference
- `inference/decoder.py`: Real-time rolling buffer decoder with windowed prediction
- `inference/confidence.py`: Confidence threshold, hysteresis, debouncing, majority voting
- `inference/smoother.py`: Backward-compatible smoother re-export

#### Bridge
- `bridge/intent_to_action.py`: 5-class intent to robot action mapping

#### Scripts
- `scripts/train_baseline.py`: Train and evaluate all baseline models
- `scripts/train_hierarchical.py`: Train hierarchical 2-stage classifier
- `scripts/train_wavelet.py`: Train with DWT wavelet features
- `scripts/benchmark_latency.py`: Per-component latency benchmarking

#### Tests
- Full test suite: preprocessing, features, inference, bridge, models

#### Documentation
- `README.md`: Project overview, architecture, status, setup instructions
- `ROADMAP.md`: Versioned implementation plan with task assignments
- `CONTRIBUTING.md`: Contribution guidelines

#### Project Setup
- `pyproject.toml`: UV/Hatch build config with all dependencies
- `Dockerfile`: CUDA 12.4 + Python 3.12 + MuJoCo
- `compose.yaml`: Docker Compose V2 with GPU support
- Python >=3.11, <3.14
