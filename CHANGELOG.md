# Changelog

All notable changes to ThoughtLink will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned according to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Streamlit real-time dashboard (`viz/dashboard.py`)
- t-SNE/UMAP embedding visualization (`viz/latent_viz.py`)
- MuJoCo robot controller integration (`bridge/brain_policy.py`)
- ONNX model export pipeline
- End-to-end demo script (`scripts/run_demo.py`)
- Multi-robot orchestrator (`bridge/orchestrator.py`)
- Jupyter notebooks (EDA, feature engineering, model comparison)

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
- `features/eeg_features.py`: Band power (4 bands x 6 channels) and Hjorth parameters
- `features/nirs_features.py`: NIRS temporal features (mean, peak, slope) with PCA reduction
- `features/fusion.py`: EEG + NIRS feature concatenation (~62 features per window)

#### Models
- `models/baseline.py`: 4 sklearn pipelines (LogReg, SVM Linear, SVM RBF, Random Forest)
- `models/hierarchical.py`: 2-stage classifier (rest-vs-active gate + 4-class decoder)
- `models/cnn.py`: Compact EEGNet CNN in PyTorch (~2-4K params)

#### Inference
- `inference/decoder.py`: Real-time rolling buffer decoder with windowed prediction
- `inference/confidence.py`: Confidence threshold, hysteresis, debouncing, majority voting
- `inference/smoother.py`: Backward-compatible smoother re-export

#### Bridge
- `bridge/intent_to_action.py`: 5-class intent to robot action mapping

#### Scripts
- `scripts/train_baseline.py`: Train and evaluate all baseline models
- `scripts/train_hierarchical.py`: Train hierarchical 2-stage classifier
- `scripts/benchmark_latency.py`: Per-component latency benchmarking

#### Tests
- `tests/test_preprocessing.py`: EEG, NIRS, and windowing tests
- `tests/test_features.py`: Band power, Hjorth, fusion tests
- `tests/test_inference.py`: Confidence filter, smoother, stability pipeline tests

#### Documentation
- `README.md`: Project overview, architecture, status, setup instructions
- `ROADMAP.md`: Versioned implementation plan with task assignments
- `CONTRIBUTING.md`: Contribution guidelines
- `CHANGELOG.md`: This file

#### Project Setup
- `pyproject.toml`: UV/Hatch build config with all dependencies
- Python >=3.11, <3.14
- Optional MuJoCo dependency for robot simulation
