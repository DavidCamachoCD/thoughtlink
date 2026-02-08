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
  Band Power (mu/beta) + Hjorth + NIRS fusion
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
│   ├── data/                   # HuggingFace loader + subject-aware splitting
│   ├── preprocessing/          # EEG (MNE), NIRS, sliding windows
│   ├── features/               # Band power, Hjorth, NIRS features, fusion
│   ├── models/                 # Baselines (sklearn), Hierarchical, EEGNet (PyTorch)
│   ├── inference/              # Real-time decoder, confidence filter, smoother
│   ├── bridge/                 # Intent-to-action mapping + BrainPolicy + MuJoCo controller
│   └── viz/                    # Streamlit dashboard + temporal stability plots
├── scripts/                    # Training & benchmarking scripts
├── tests/                      # Unit tests (59 passing)
└── notebooks/                  # EDA & model comparison (planned)
```

## Progress

### v0.1.0 - v0.3.0: Core pipeline (done)

- [x] Project scaffolding with UV, pyproject.toml, configs
- [x] Data pipeline: HuggingFace download, `.npz` parsing, subject-aware split
- [x] EEG preprocessing: bandpass 1-40 Hz, CAR via MNE-Python
- [x] NIRS preprocessing: baseline correction, SDS selection, PCA
- [x] Windowing: 1s sliding windows with 50% overlap (~15x augmentation)
- [x] Feature extraction: band power (4 bands x 6 ch), Hjorth params, NIRS temporal
- [x] Feature fusion: EEG + NIRS concatenation (~62 features/window)
- [x] Baseline models: LogReg, SVM Linear, SVM RBF, Random Forest
- [x] Hierarchical classifier: 2-stage rest-gate + 4-class decoder
- [x] EEGNet CNN: compact PyTorch (~2-4K params, <3ms inference target)
- [x] Stability pipeline: confidence threshold + hysteresis + debouncing + majority voting
- [x] Real-time decoder: rolling buffer with windowed prediction
- [x] Intent-to-action mapping: 5 classes -> robot Action enum
- [x] Training scripts: baseline + hierarchical with metrics export
- [x] Latency benchmark script: per-component timing (target <50ms)
- [x] Unit tests: 30/30 passing (preprocessing, features, inference)
- [x] Docker: CUDA 12.4 + Compose V2 with GPU support
- [x] Security: .gitignore hardened, .env.example, .dockerignore

### v0.4.0: Integration (done)

- [x] BrainPolicy: main loop from brain signals to robot actions (`bridge/brain_policy.py`)
- [x] End-to-end demo script (`scripts/run_demo.py`) with live terminal output
- [x] Streamlit real-time dashboard (`viz/dashboard.py`)
- [x] Temporal stability visualization (`viz/temporal_stability.py`)
- [x] Multi-robot orchestrator (`bridge/orchestrator.py`) with simulated fleet
- [x] Tests for bridge module: 50/50 total passing
- [ ] ONNX model export (stretch goal)

### v0.5.0: MuJoCo integration (done)

- [x] MuJoCo controller wrapping `bri` package (`bridge/mujoco_controller.py`)
- [x] Unitree G1 humanoid simulation via [brain-robot-interface](https://github.com/Nabla7/brain-robot-interface)
- [x] MuJoCo demo script: brain signals -> humanoid robot (`scripts/run_mujoco_demo.py`)
- [x] Controller implements `RobotController` protocol (drop-in for Orchestrator)
- [x] Tests for MuJoCoController: 59/59 total passing

### v1.0.0: Demo & polish (next)

- [ ] t-SNE/UMAP embedding visualization
- [ ] Model comparison notebook
- [ ] Latency vs accuracy tradeoff plots

## Setup

### Local (macOS / Linux)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/DavidCamachoCD/thoughtlink.git
cd thoughtlink
uv sync

# Run tests
uv run pytest tests/ -v
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
