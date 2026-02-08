# ThoughtLink: From Brain to Robot

Decode non-invasive brain signals (EEG + TD-NIRS) into high-level commands for humanoid robot control via MuJoCo simulation.

Built for **Global AI Hackathon** (Hack-Nation x Kernel x Dimensional), Feb 7-8, 2026 — Challenge #9.

## System Architecture

```
[Brain Signals .npz]
        |
        v
[Preprocessing]
  EEG: bandpass 1-40Hz, CAR, 1s windows
  NIRS: baseline correction, PCA
        |
        v
[Feature Extraction]
  Band Power (mu/beta) + Hjorth + Time Domain + DWT Wavelets
        |
        v
[Hierarchical Classifier]
  Stage 1: Rest vs Active (binary gate)
  Stage 2: Right Fist | Left Fist | Both Fists | Tongue (4-class)
        |
        v
[Stability Layer]
  Confidence threshold + Hysteresis + Debouncing + Majority voting
        |
        v
[Intent -> Action Mapper]
  Right Fist  -> RIGHT
  Left Fist   -> LEFT
  Both Fists  -> FORWARD
  Tongue      -> STOP
  Relax       -> STOP
        |
        v
[Robot Controller -> MuJoCo Simulation]
```

## Dataset

| Property | Value |
|----------|-------|
| Source | [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) |
| Format | `.npz`, 15-second chunks |
| Samples | ~900 files |
| EEG | 6 channels (AFF6, AFp2, AFp1, AFF5, FCz, CPz), 500 Hz |
| TD-NIRS | 40 modules, 4.76 Hz, 5D tensor `(72, 40, 3, 2, 3)` |
| Classes | Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax |
| Timing | Rest 0-3s, stimulus onset at 3s (~9s duration) |

## Project Structure

```
thoughtlink/
├── pyproject.toml              # UV package manager
├── compose.yaml                # Docker Compose V2 (GPU)
├── Dockerfile                  # CUDA 12.4 + Python 3.12 + MuJoCo
├── configs/default.yaml        # Centralized hyperparameters
├── src/thoughtlink/
│   ├── data/                   # HuggingFace loader, subject-aware splitting, PyTorch Dataset
│   ├── preprocessing/          # EEG (MNE), NIRS, sliding windows, augmentation
│   ├── features/               # Band power, Hjorth, time-domain, DWT wavelets, CSP, fusion
│   ├── models/                 # Baselines (sklearn), Hierarchical, EEGNet (PyTorch), GRU temporal
│   ├── inference/              # Real-time decoder, confidence filter, smoother
│   ├── bridge/                 # Intent-to-action mapping, BrainPolicy, MuJoCo controller, orchestrator
│   └── viz/                    # Streamlit dashboard, temporal stability, t-SNE/UMAP latent viz
├── scripts/                    # Training, benchmarking, ONNX export
├── tests/                      # Unit tests (212 passing)
└── notebooks/                  # EDA, feature engineering, model comparison, wavelet analysis
```

## Models

| Model | Type | File |
|-------|------|------|
| Logistic Regression | sklearn | `models/baseline.py` |
| SVM (Linear + RBF) | sklearn | `models/baseline.py` |
| Random Forest | sklearn | `models/baseline.py` |
| Hierarchical 2-Stage | sklearn | `models/hierarchical.py` |
| EEGNet CNN | PyTorch | `models/cnn.py` |
| Temporal GRU | PyTorch | `models/temporal.py` |
| Voting Ensemble | sklearn | `models/ensemble.py` |

## Progress

### v0.1.0 - v0.3.0: Core pipeline

- [x] Project scaffolding with UV, pyproject.toml, configs
- [x] Data pipeline: HuggingFace download, `.npz` parsing, subject-aware split
- [x] EEG preprocessing: bandpass 1-40 Hz, CAR via MNE-Python
- [x] NIRS preprocessing: baseline correction, SDS selection, PCA
- [x] Windowing: 1s sliding windows with 50% overlap (~15x augmentation)
- [x] Feature extraction: band power, Hjorth, time-domain, DWT wavelets, CSP
- [x] Feature fusion: EEG + NIRS concatenation (~66 features/window)
- [x] Baseline models: LogReg, SVM Linear, SVM RBF, Random Forest
- [x] Hierarchical classifier: 2-stage rest-gate + 4-class decoder
- [x] EEGNet CNN: compact PyTorch (~2-4K params, <3ms inference target)
- [x] Stability pipeline: confidence threshold + hysteresis + debouncing + majority voting
- [x] Real-time decoder: rolling buffer with windowed prediction
- [x] Intent-to-action mapping: 5 classes -> robot Action enum
- [x] Training scripts: baseline + hierarchical + wavelet with metrics export
- [x] Latency benchmark script: per-component timing (target <50ms)
- [x] Docker: CUDA 12.4 + Compose V2 with GPU support

### v0.4.0: Integration

- [x] BrainPolicy: main loop from brain signals to robot actions (`bridge/brain_policy.py`)
- [x] End-to-end demo script (`scripts/run_demo.py`) with live terminal output
- [x] Streamlit real-time dashboard (`viz/dashboard.py`)
- [x] Temporal stability visualization (`viz/temporal_stability.py`)
- [x] Multi-robot orchestrator (`bridge/orchestrator.py`) with simulated fleet

### v0.5.0: MuJoCo integration

- [x] MuJoCo controller wrapping `bri` package (`bridge/mujoco_controller.py`)
- [x] Unitree G1 humanoid simulation via [brain-robot-interface](https://github.com/Nabla7/brain-robot-interface)
- [x] MuJoCo demo script: brain signals -> humanoid robot (`scripts/run_mujoco_demo.py`)

### v1.0.0: Demo & polish

- [x] Temporal GRU model for sequential EEG decoding (`models/temporal.py`)
- [x] PyTorch Dataset wrapper (`data/dataset.py`)
- [x] ONNX export script for production inference (`scripts/export_onnx.py`)
- [x] t-SNE/UMAP embedding visualization (`viz/latent_viz.py`)
- [x] Feature engineering notebook with separability analysis
- [x] Model comparison notebook with full metrics
- [x] Wavelet analysis notebooks (DWT features + baseline comparison)
- [x] 212 unit tests passing

## Setup

### Local (macOS / Linux)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/DavidCamachoCD/thoughtlink.git
cd thoughtlink
uv sync

# With MuJoCo simulation support
uv sync --extra sim

# Run tests
uv run python -m pytest tests/ -v
```

### Docker (Ubuntu AMD64 + NVIDIA GPU)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
# Build
docker compose build

# Jupyter notebooks (http://localhost:8888)
docker compose up jupyter

# Train models
docker compose run --rm train-baseline
docker compose run --rm train-hierarchical
docker compose run --rm train-wavelet

# Export models to ONNX
docker compose run --rm export-onnx

# Benchmark latency
docker compose run --rm benchmark

# Tests
docker compose run --rm test

# Interactive shell
docker compose run --rm shell
```

## Usage

```bash
# Train baseline models (LogReg, SVM, RF)
uv run python scripts/train_baseline.py

# Train hierarchical 2-stage model
uv run python scripts/train_hierarchical.py

# Train with wavelet features
uv run python scripts/train_wavelet.py

# Export trained models to ONNX
uv run python scripts/export_onnx.py

# Run end-to-end demo (replays .npz files through full pipeline)
uv run python scripts/run_demo.py
uv run python scripts/run_demo.py --live --delay 0.1   # real-time terminal output

# Run MuJoCo demo (brain signals -> humanoid robot in simulation)
uv run mjpython scripts/run_mujoco_demo.py             # macOS
uv run python scripts/run_mujoco_demo.py               # Linux

# Launch Streamlit dashboard
uv run streamlit run src/thoughtlink/viz/dashboard.py

# Benchmark inference latency
uv run python scripts/benchmark_latency.py

# Run unit tests
uv run python -m pytest tests/ -v
```

## Known Failure Modes & Open Research Questions

### Failure Modes

| Failure Mode | Impact | Current Mitigation |
|---|---|---|
| **Subject variability** | EEG patterns differ between individuals; model trained on subjects A-D may fail on subject E | Subject-aware splitting prevents data leakage, but cross-subject generalization remains limited |
| **Low-channel limitation** | Only 6 EEG channels (vs 64+ in research BCIs); spatial resolution is poor for fine motor imagery | CSP optimizes available channels; focus on FCz/CPz (motor cortex) for strongest signal |
| **Class confusion: Both Fists vs single fist** | Bilateral and unilateral motor imagery share overlapping mu/beta suppression patterns | Hierarchical model isolates rest first, reducing 5-class to 4-class problem |
| **Relax state contamination** | Drowsiness or distraction during "active" periods produces false rest classification | Stage 1 rest-gate has high sensitivity; hysteresis prevents rapid oscillation |
| **Temporal lag** | 1s window + stability pipeline adds ~1-2s latency between intent and action | Acceptable for high-level commands (not fine motor control); configurable trade-off |
| **NIRS slow dynamics** | fNIRS at 4.76 Hz cannot track fast intent changes; hemodynamic response is ~5s delayed | NIRS used only as secondary robustness signal, not primary decision driver |

### Open Research Questions

1. **Cross-subject transfer learning** — Can a model trained on N subjects generalize to a new unseen subject without calibration? Current BCI research suggests domain adaptation or few-shot fine-tuning is needed.

2. **Phase-aware intent modeling** — Our system treats each window independently. Modeling intent phases (initiation, sustain, release) could improve transition detection and reduce false triggers during state changes.

3. **Optimal complexity vs latency** — Our hierarchical SVM achieves a good balance, but when does a CNN or transformer actually improve accuracy enough to justify the added inference cost? Our benchmark infrastructure enables this comparison.

4. **Scalability bottleneck** — Our orchestrator dispatches O(N) synchronously. For 1000+ robots, async dispatch with priority queues and failure-aware routing would be needed.

5. **Confidence calibration** — Are the model's `predict_proba` outputs well-calibrated? Overconfident models could bypass the confidence threshold and cause false triggers. Platt scaling or isotonic regression could help.

## References

- [KernelCo/robot_control dataset](https://huggingface.co/datasets/KernelCo/robot_control)
- [brain-robot-interface repo](https://github.com/Nabla7/brain-robot-interface)
- [MNE-Python CSP Motor Imagery](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [EEG BCI real-time robotic hand control (Nature 2025)](https://www.nature.com/articles/s41467-025-61064-x)
- [BCI with AI copilots (Nature Machine Intelligence)](https://www.nature.com/articles/s42256-025-01090-y)
