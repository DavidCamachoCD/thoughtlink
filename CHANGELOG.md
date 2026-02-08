# Changelog

All notable changes to ThoughtLink will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned according to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
