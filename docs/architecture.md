# ThoughtLink — System Architecture

## Overview

ThoughtLink is a **brain-to-robot intent decoding system** that translates non-invasive brain signals (EEG + fNIRS) into discrete, high-level robot commands. It is designed as an **intent interpreter**, not a controller — robots already know how to move; ThoughtLink tells them *what* to do.

The system follows a strictly **linear pipeline** where each stage transforms data and passes it forward:

```
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  .npz files  │───>│ Preprocessing  │───>│    Features      │───>│     Models       │
│  (EEG+NIRS)  │    │ (bandpass,CAR) │    │ (band power,     │    │ (hierarchical    │
│              │    │                │    │  Hjorth, fusion)  │    │  2-stage SVM)    │
└─────────────┘    └────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                                         │
                                                                    probabilities
                                                                         │
                                                                         v
┌─────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Robot(s)   │<───│    Bridge      │<───│   Stability     │<───│    Inference     │
│  (MuJoCo /   │    │ (intent →      │    │ (confidence +   │    │ (rolling buffer  │
│   simulated) │    │  action map)   │    │  hysteresis +   │    │  + windowed      │
│              │    │                │    │  debounce +     │    │  prediction)     │
│              │    │                │    │  voting)        │    │                  │
└─────────────┘    └────────────────┘    └─────────────────┘    └──────────────────┘
```

---

## Data Format

All data comes from the [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) dataset.

Each `.npz` file is a **15-second recording** containing:

| Signal | Shape | Sampling Rate | Description |
|--------|-------|---------------|-------------|
| EEG (`feature_eeg`) | `(7499, 6)` | 500 Hz | 6 channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz |
| NIRS (`feature_moments`) | `(72, 40, 3, 2, 3)` | 4.76 Hz | 40 modules, time-domain moments |
| Label | string | — | One of 5 classes |

**Timing within each chunk:**
```
0s ──── 3s ──────────────── 12s ──── 15s
│ REST  │     STIMULUS      │  REST  │
│       │  (active intent)  │        │
```

**5 Classes → 4 Robot Actions:**

| Brain Intent | Label Index | Robot Action |
|---|---|---|
| Right Fist | 0 | `RIGHT` |
| Left Fist | 1 | `LEFT` |
| Both Fists | 2 | `FORWARD` |
| Tongue Tapping | 3 | `STOP` |
| Relax | 4 | `STOP` |

---

## Module-by-Module Architecture

### 1. Data (`src/thoughtlink/data/`)

**Purpose:** Load and split the dataset.

```
loader.py
  download_dataset()      → downloads from HuggingFace
  load_sample(path)       → dict with eeg, nirs, label, subject_id, duration
  load_all()              → list of all sample dicts

splitter.py
  split_by_subject()      → train/test sets where NO subject appears in both
```

**Key rule:** Subject-aware splitting is mandatory. Mixing subjects across train/test causes data leakage because EEG patterns are highly subject-specific.

### 2. Preprocessing (`src/thoughtlink/preprocessing/`)

**Purpose:** Clean raw signals and extract temporal windows.

```
eeg.py
  preprocess_eeg(eeg_data) → bandpass 1-40Hz + CAR
    1. Create MNE RawArray (µV → V conversion)
    2. FIR bandpass filter [1, 40] Hz
       - Captures: theta (4-8), mu (8-13), beta (13-30), low gamma (30-40)
       - Removes: DC drift (<1Hz), line noise (>40Hz)
    3. Common Average Reference (CAR)
       - Subtracts mean across channels at each timepoint
       - Reduces common-mode noise
    4. Convert back to µV

nirs.py
  preprocess_nirs(nirs_data) → baseline correction + SDS selection + PCA
    1. Select medium + long source-detector separations (indices 1, 2)
    2. Baseline correction (subtract rest-period mean)
    3. PCA reduction to 20 components

windowing.py
  extract_eeg_windows(eeg_data, duration) → (n_windows, 500, 6)
    - 1-second windows (500 samples at 500Hz)
    - 50% overlap (stride = 250 samples)
    - Only from stimulus period (3s to ~11s)
    - ~15 windows per 15s chunk → 15x data augmentation
```

**Data shapes through preprocessing:**
```
Raw EEG:        (7499, 6)       — 15s at 500Hz, 6 channels
Preprocessed:   (7499, 6)       — same shape, cleaner signal
Windows:        (n_windows, 500, 6) — ~15 windows of 1s each
```

### 3. Features (`src/thoughtlink/features/`)

**Purpose:** Extract numerical features from each 1s window.

```
eeg_features.py
  compute_band_powers(window)    → 24 features (4 bands × 6 channels)
    - Welch PSD → log10 mean power in each band
    - Bands: theta [4-8], mu [8-13], beta [13-30], low_gamma [30-40]

  compute_hjorth(window)         → 18 features (3 params × 6 channels)
    - Activity: signal variance (amplitude)
    - Mobility: frequency content (mean frequency)
    - Complexity: bandwidth (frequency change)

  extract_window_features(window) → 42 features (default: band_power + hjorth)

nirs_features.py
  extract_nirs_features(nirs_data) → 20 features (after PCA)
    - Temporal stats: mean, peak, slope per module
    - PCA reduces to 20 components

fusion.py
  fuse_features(eeg_feat, nirs_feat) → ~62 features
    - Simple concatenation: [eeg_features | nirs_features]
```

**Feature vector per window: ~42 (EEG-only) or ~62 (EEG+NIRS)**

### 4. Models (`src/thoughtlink/models/`)

**Purpose:** Classify feature vectors into 5 intent classes.

#### Baselines (`baseline.py`)
Four sklearn pipelines, each with StandardScaler + classifier:
- Logistic Regression (C=1.0)
- SVM Linear (C=10.0)
- SVM RBF (C=10.0, gamma=scale)
- Random Forest (200 trees, max_depth=10)

#### Hierarchical Classifier (`hierarchical.py`) — **Primary model**

This is our main approach, directly addressing the "Hierarchical Intent Models" direction from the challenge.

```
                    Input: feature vector (42d)
                              │
                    ┌─────────▼──────────┐
                    │  STAGE 1: Binary   │
                    │  Relax vs Active   │
                    │  (SVM RBF)         │
                    └──────┬──────┬──────┘
                           │      │
                   P(relax)│      │P(active)
                           │      │
                           ▼      ▼
                    ┌──────────────────────┐
                    │  STAGE 2: 4-class    │
                    │  Right | Left |      │
                    │  Both  | Tongue      │
                    │  (SVM RBF)           │
                    └──────────┬───────────┘
                               │
                    Combined: P(class) = P(active) × P(class|active)
                              P(relax) = P(relax) from Stage 1
```

**Why hierarchical?**
- Reduces false triggers by filtering rest states first (addresses evaluation criterion #4)
- Stage 1 is a simple binary problem → high accuracy
- Stage 2 only runs on active samples → cleaner decision boundary
- Combined probabilities are well-calibrated for the confidence filter

#### EEGNet CNN (`cnn.py`)
- Compact PyTorch model (~2-4K parameters)
- Temporal convolution → spatial convolution → classification
- Alternative to sklearn baselines for comparison

### 5. Inference (`src/thoughtlink/inference/`)

**Purpose:** Real-time prediction with temporal stability.

#### Decoder (`decoder.py`)
```
RealtimeDecoder
  - Rolling buffer (deque, 5s capacity)
  - feed_samples(chunk)    → push new EEG into buffer
  - predict()              → extract latest 1s window → features → model → probs
  - Simulates live BCI: architecturally identical to real headset streaming
```

#### Stability Pipeline (`confidence.py`)

Three sequential mechanisms prevent oscillation and false triggers:

```
Raw probabilities
       │
       ▼
┌──────────────────────┐
│ 1. CONFIDENCE        │  Only act if P(class) > 0.6
│    THRESHOLD         │  Below threshold → keep current action
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. HYSTERESIS        │  Switching to NEW action requires P > 0.7
│    (±0.1 margin)     │  Continuing SAME action requires P > 0.5
│                      │  Prevents oscillation at decision boundary
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 3. DEBOUNCING        │  New action must win 3 consecutive predictions
│    (count=3)         │  before committing. Prevents flicker.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4. MAJORITY VOTING   │  Sliding window of 5 actions → most common wins
│    (window=5)        │  Final smoothing layer
└──────────┬───────────┘
           │
           ▼
    Stable action output
```

### 6. Bridge (`src/thoughtlink/bridge/`)

**Purpose:** Map decoded intent to robot actions and coordinate multi-robot dispatch.

#### Intent Mapping (`intent_to_action.py`)
Simple dictionary: 5 class names → 4 action strings (RIGHT, LEFT, FORWARD, STOP).

#### BrainPolicy (`brain_policy.py`)
Main orchestrator loop that ties everything together:
```python
policy = BrainPolicy(model=trained_model, config=config)
results = policy.run_on_file("data/sample.npz")
# Each result: StepResult(timestamp, raw_intent, stable_intent, action, confidence, probs, latency)
```

Supports:
- File replay (simulated streaming from .npz)
- Array input (pre-preprocessed data)
- Callback `on_step` for real-time UI updates
- Manual `step(probs)` for external decoders

#### Orchestrator (`orchestrator.py`)
One-to-many robot dispatch:
```
BrainPolicy (1 brain) ──> Orchestrator ──> Robot_001
                                      ──> Robot_002
                                      ──> ...
                                      ──> Robot_100
```

Features:
- Deduplication (only dispatch on action change)
- Per-robot failure tracking
- Fleet-wide emergency stop
- O(N) dispatch, tested at 100 robots in <10ms

### 7. Visualization (`src/thoughtlink/viz/`)

**Purpose:** Real-time dashboard and publication-ready plots.

- **dashboard.py** — Streamlit app: EEG traces, probability bars, action timeline, step log
- **temporal_stability.py** — Matplotlib: action timeline, confidence trace, probability heatmap, 3-panel report

---

## Configuration

All hyperparameters live in `configs/default.yaml`. No hardcoded values in source code.

Key parameters:

| Parameter | Value | Rationale |
|---|---|---|
| `bandpass_low` | 1.0 Hz | Remove DC drift |
| `bandpass_high` | 40.0 Hz | Capture up to low gamma, reject line noise |
| `window_duration_s` | 1.0 s | Good frequency resolution for mu/beta |
| `window_stride_s` | 0.5 s | 50% overlap → 2 predictions/second |
| `confidence_threshold` | 0.6 | Minimum confidence to act |
| `hysteresis_margin` | 0.1 | ±0.1 around threshold for switching |
| `debounce_count` | 3 | 3 consecutive agreements to commit |
| `smoother_window` | 5 | Majority vote over last 5 actions |
| `prediction_hz` | 2.0 | 2 predictions per second |

---

## Latency Budget

Target: **< 50ms** end-to-end per prediction tick.

| Component | Expected Latency |
|---|---|
| Feature extraction (1s window) | ~2-5 ms |
| Model inference (SVM) | ~1-3 ms |
| Stability pipeline | ~0.01 ms |
| Orchestrator dispatch (100 robots) | ~0.1 ms |
| **Total** | **~3-8 ms** |

Note: EEG preprocessing (bandpass + CAR) runs once per 15s chunk (~50-100ms), not per prediction tick.

---

## Design Decisions

1. **EEG first, NIRS as enhancement** — EEG at 500Hz captures motor ERD/ERS in real-time. NIRS at 4.76Hz adds robustness but cannot drive fast decisions.

2. **1s windows with 50% overlap** — Good frequency resolution for mu (8-13Hz) and beta (13-30Hz) bands. Overlap provides ~15x data augmentation per chunk.

3. **Hierarchical classification** — Reduces false positives by design. Rest-gate is a simple binary problem with high accuracy.

4. **4-layer stability pipeline** — Each layer addresses a different temporal artifact: threshold (noise), hysteresis (oscillation), debounce (flicker), voting (drift).

5. **Simulated streaming** — Replaying .npz files through a rolling buffer is architecturally identical to live BCI. Only the data source changes.

6. **Orchestrator with deduplication** — Only dispatches on action *change*, not every tick. Reduces robot command bandwidth by ~90%.
