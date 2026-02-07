# ThoughtLink: From Brain to Robot

## Implementation Plan & Roadmap

---

## Context

**Hackathon**: Global AI Hackathon (Hack-Nation x Kernel x Dimensional), Feb 7-8, 2026
**Challenge**: #9 ThoughtLink - Decode non-invasive brain signals into high-level instructions for humanoid robots
**Approach**: Hierarchical Intent Models
**Hardware**: GPU available (PyTorch + ONNX)

---

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
  Band Power (mu/beta) + CSP + Hjorth + NIRS fusion
        |
        v
[Hierarchical Classifier]
  Stage 1: Rest vs Active (binary)
  Stage 2: Right Fist | Left Fist | Both Fists | Tongue (4-class)
        |
        v
[Confidence/Stability Layer]
  Confidence threshold + Hysteresis + Debouncing + Majority voting
        |
        v
[Intent -> Action Mapper]
  Right Fist  -> Action.RIGHT
  Left Fist   -> Action.LEFT
  Both Fists  -> Action.FORWARD
  Tongue Tap  -> Action.STOP
  Relax       -> Action.STOP
        |
        v
[bri Controller -> MuJoCo Simulation]
  + Real-time visualization dashboard
```

---

## Project Structure

```
thoughtlink/
├── pyproject.toml                    # UV package manager
├── configs/default.yaml              # Centralized hyperparameters
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA, class balance, visualization
│   ├── 02_feature_engineering.ipynb  # CSP, PSD, NIRS features
│   └── 03_model_comparison.ipynb    # Model comparison
├── src/thoughtlink/
│   ├── data/
│   │   ├── loader.py                # HuggingFace download + .npz loading
│   │   ├── dataset.py               # sklearn/PyTorch compatible dataset
│   │   └── splitter.py              # Split by subject_id
│   ├── preprocessing/
│   │   ├── eeg.py                   # MNE-Python pipeline
│   │   ├── nirs.py                  # TD-NIRS processing
│   │   └── windowing.py             # Sliding windows
│   ├── features/
│   │   ├── eeg_features.py          # Band power, CSP, Hjorth
│   │   ├── nirs_features.py         # Hemodynamic features + PCA
│   │   └── fusion.py                # EEG+NIRS concatenation
│   ├── models/
│   │   ├── baseline.py              # LogReg, SVM, RF (sklearn)
│   │   ├── hierarchical.py          # 2-stage classifier
│   │   ├── cnn.py                   # Compact EEGNet (PyTorch)
│   │   └── temporal.py              # Sequential GRU (stretch goal)
│   ├── inference/
│   │   ├── decoder.py               # Rolling buffer + windowed prediction
│   │   ├── confidence.py            # Thresholds, hysteresis, debounce
│   │   └── smoother.py              # Majority voting
│   ├── bridge/
│   │   ├── intent_to_action.py      # 5 classes -> Action enum
│   │   ├── brain_policy.py          # Main loop: signal -> robot
│   │   └── orchestrator.py          # Multi-robot dispatch
│   └── viz/
│       ├── dashboard.py             # Streamlit real-time
│       └── latent_viz.py            # t-SNE/UMAP embeddings
├── scripts/
│   ├── train_baseline.py
│   ├── train_hierarchical.py
│   ├── run_demo.py                  # End-to-end demo
│   └── benchmark_latency.py
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_inference.py
```

---

## Dataset (KernelCo/robot_control)

| Property | Value |
|----------|-------|
| Source | `huggingface.co/datasets/KernelCo/robot_control` |
| Format | .npz, 15-second chunks |
| Count | ~900 files |
| EEG | 6 channels (AFF6, AFp2, AFp1, AFF5, FCz, CPz), 500 Hz, (7499, 6) |
| TD-NIRS | 40 modules, 4.76 Hz, (72, 40, 3, 2, 3) |
| Classes | Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax |
| Timing | Rest 0-3s, stimulus from 3s (~9s duration) |
| Split | By `subject_id` (mandatory to prevent data leakage) |

---

## Versioned Roadmap

### v0.1.0 - Foundation (Saturday 14:00 - 17:00) ~3h

**Goal**: Working project with data loaded and preprocessed.

| ID | Task | Owner | Deliverable |
|----|------|-------|-------------|
| F1 | Create repo structure + `pyproject.toml` with UV | David | Successful `uv sync` with all deps |
| F2 | Implement `loader.py`: HuggingFace download + .npz loading | David | `load_all()` returns list of dicts |
| F3 | Implement `splitter.py`: split by subject_id | David | Train/val/test sets without data leakage |
| F4 | Notebook `01_data_exploration.ipynb`: full EDA | Nat | Class distribution, subjects, signal visualization |
| F5 | Implement `eeg.py`: bandpass 1-40Hz, CAR, segmentation | Nat | `preprocess_eeg()` returns clean RawArray |
| F6 | Implement `windowing.py`: 1s windows, 50% overlap | David | `extract_windows()` generates (n_windows, 500, 6) |
| F7 | Clone + verify `brain-robot-interface` with MuJoCo | David | `mjpython examples/minimal_policy.py` works |

**Checkpoint v0.1.0**: Data downloaded, preprocessed, and split. MuJoCo verified.

---

### v0.2.0 - Features & Baselines (Saturday 17:00 - 21:00) ~4h

**Goal**: Complete feature extraction and first trained models with metrics.

| ID | Task | Owner | Deliverable |
|----|------|-------|-------------|
| B1 | Implement `eeg_features.py`: band power (mu/beta/theta/gamma) | David | 24 features per window via Welch PSD |
| B2 | Implement CSP features via `mne.decoding.CSP` | Nat | 4-10 optimized CSP components |
| B3 | Implement Hjorth parameters (activity, mobility, complexity) | Nat | 18 additional features per window |
| B4 | Implement `nirs.py` + `nirs_features.py`: baseline correction + PCA | Nat | 20 NIRS features per trial |
| B5 | Implement `fusion.py`: concatenate EEG + NIRS | David | ~70 feature vector per window |
| B6 | Implement `baseline.py`: LogReg, SVM, RF pipelines | Nat | 4 sklearn models with `predict_proba` |
| B7 | Script `train_baseline.py`: train + evaluate binary (L vs R) | Nat | Accuracy, Kappa, F1, confusion matrix |
| B8 | Evaluate baselines multi-class (5 classes) | Nat | Comparative model table |

**Checkpoint v0.2.0**: At least one binary model with >65% accuracy. End-to-end feature pipeline working.

---

### v0.3.0 - Hierarchical Model + Confidence (Saturday 21:00 - 01:00) ~4h

**Goal**: Hierarchical model trained with complete stability layer.

| ID | Task | Owner | Deliverable |
|----|------|-------|-------------|
| H1 | Implement `hierarchical.py`: Stage 1 (Relax vs Active) + Stage 2 (4 classes) | David | `HierarchicalClassifier` class with `fit()` and `predict_proba()` |
| H2 | Script `train_hierarchical.py`: train both stages | David | Per-stage metrics + combined metrics |
| H3 | Implement `confidence.py`: threshold + hysteresis + debounce | David | `IntentConfidenceFilter.update(probs) -> action` |
| H4 | Implement `smoother.py`: majority voting | David | `MajorityVotingSmoother.smooth(action) -> action` |
| H5 | EEGNet CNN in `cnn.py` (PyTorch) | Nat | Compact model ~2K params, inference <3ms |
| H6 | Train CNN + compare vs baselines | Nat | Accuracy/latency table: CNN vs sklearn |
| H7 | Notebook `02_feature_engineering.ipynb`: visualize features | Nat | Per-class separability plots |
| H8 | Export best model to ONNX | Nat | .onnx file + inference script |

**Checkpoint v0.3.0**: Hierarchical model + confidence filter working. CNN trained if baselines < 70%.

---

### v0.4.0 - Integration (Sunday 01:00 - 04:00) ~3h

**Goal**: Complete pipeline from brain signal to robot in simulation.

| ID | Task | Owner | Deliverable |
|----|------|-------|-------------|
| I1 | Implement `intent_to_action.py`: map 5 classes -> Action enum | David | Validated mapping dictionary |
| I2 | Implement `decoder.py`: rolling buffer + windowed prediction | David | `RealtimeDecoder` with `feed_samples()` and `predict()` |
| I3 | Implement `brain_policy.py`: main loop signal -> robot | David | Robot moves in MuJoCo from .npz data |
| I4 | Script `run_demo.py`: end-to-end demo with simulated stream | David | Working demo replaying test files |
| I5 | Script `benchmark_latency.py`: measure per-component latency | Nat | Latency table: total <50ms |
| I6 | Implement `orchestrator.py`: multi-robot dispatch | Nat | Dispatch to 2-3 simultaneous controllers |
| I7 | Unit tests: preprocessing, features, inference, bridge | Nat | `pytest tests/` passes without errors |

**Checkpoint v0.4.0**: Working end-to-end demo. Robot responds to brain signals in simulation.

---

### v1.0.0 - Demo & Polish (Sunday 04:00 - 08:00) ~4h

**Goal**: Presentation ready with dashboard, visualizations, and documentation.

| ID | Task | Owner | Deliverable |
|----|------|-------|-------------|
| D1 | Streamlit dashboard: EEG traces + probability bars + confidence state | David | `streamlit run dashboard.py` working |
| D2 | Notebook `03_model_comparison.ipynb`: full comparison | Nat | Tables + charts for presentation |
| D3 | t-SNE/UMAP of feature space | Nat | Separability visualization |
| D4 | Plot latency vs accuracy (all models) | Nat | Scatter plot for bonus |
| D5 | Plot temporal stability: action vs time | David | Color-coded action timeline |
| D6 | README with architecture, setup, results | David | Complete documentation |
| D7 | Practice 3-minute demo | Both | Presentation script |
| D8 | Record backup video | Both | MP4 demo video |

**Checkpoint v1.0.0**: Delivery ready. Demo practiced. Video recorded.

---

## Visual Timeline Summary

```
Saturday 14:00 ├─── v0.1.0 Foundation ───────────┤ 17:00
               │    Setup, data, preprocessing    │
               │                                  │
         17:00 ├─── v0.2.0 Features & Baselines ─┤ 21:00
               │    Features, baseline models      │
               │                                  │
         21:00 ├─── v0.3.0 Hierarchical + CNN ───┤ 01:00
               │    Main model, confidence         │
               │                                  │
Sunday   01:00 ├─── v0.4.0 Integration ──────────┤ 04:00
               │    Bridge, demo, tests            │
               │                                  │
         04:00 ├─── v1.0.0 Demo & Polish ────────┤ 08:00
               │    Dashboard, viz, presentation   │
               │                                  │
         08:00 └─── DELIVERY ────────────────────┘
```

---

## Work Division by Version

### David (Generative AI, RAG) — Focus: Infrastructure, Pipeline, Integration

```
v0.1.0  F1 Repo + pyproject.toml
        F2 loader.py (HuggingFace)
        F3 splitter.py (subject split)
        F6 windowing.py
        F7 Verify MuJoCo

v0.2.0  B1 eeg_features.py (band power)
        B5 fusion.py (EEG+NIRS)

v0.3.0  H1 hierarchical.py (2-stage model)
        H2 train_hierarchical.py
        H3 confidence.py (threshold+hysteresis+debounce)
        H4 smoother.py (majority voting)

v0.4.0  I1 intent_to_action.py
        I2 decoder.py (rolling buffer)
        I3 brain_policy.py (main loop)
        I4 run_demo.py

v1.0.0  D1 Streamlit Dashboard
        D5 Temporal stability plot
        D6 README
```

### Nat (Data Science, ML, CV) — Focus: Signals, Models, Metrics

```
v0.1.0  F4 EDA Notebook
        F5 eeg.py (MNE preprocessing)

v0.2.0  B2 CSP features (MNE)
        B3 Hjorth parameters
        B4 nirs.py + nirs_features.py
        B6 baseline.py (sklearn pipelines)
        B7 train_baseline.py (binary)
        B8 Multi-class baselines

v0.3.0  H5 EEGNet CNN (PyTorch)
        H6 Train CNN + compare
        H7 Feature engineering notebook
        H8 ONNX export

v0.4.0  I5 benchmark_latency.py
        I6 orchestrator.py
        I7 Unit tests

v1.0.0  D2 Model comparison notebook
        D3 t-SNE/UMAP
        D4 Latency vs accuracy plot
```

---

## Evaluation Criteria → How We Address Them

| Criterion | Weight | Our Solution |
|-----------|--------|--------------|
| **Intent Decoding Performance** | High | Hierarchical model + CSP + band power. Comparison of 6+ models. |
| **Inference Speed & Latency** | High | Target <50ms. ONNX export. Detailed per-component benchmark. |
| **Temporal Stability** | High | Hysteresis + debouncing + majority voting. Stability plot. |
| **False Trigger Rate** | High | Stage 1 (rest-detect) filters ~100% of false positives during rest. |
| **Scalability** | Medium | Multi-robot orchestrator. O(1) decoder + O(N) dispatch argument. |
| **Demo Clarity** | Medium | Streamlit dashboard + MuJoCo side-by-side. Backup video. |
| **Bonus (tradeoffs)** | Extra | Latency-accuracy scatter. Simple vs complex. Documented failure modes. |

---

## Technical Decisions

1. **EEG first, NIRS as enhancement**: EEG at 500Hz captures motor ERD/ERS. NIRS at 4.76Hz adds robustness but not fast decisions.
2. **1s windows, 50% overlap**: Good frequency resolution + multiplies data x15.
3. **Hierarchical model**: Reduces false positives by design (addresses criterion #4).
4. **ONNX for production**: 10-100x faster inference than native PyTorch.
5. **Simulated stream**: Replaying .npz as stream is architecturally identical to live.

---

## Risk Mitigation

| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| Dataset <1K samples | Windowing x15, regularization | Per-subject individual models |
| Only 6 EEG channels | CSP optimizes available channels | Focus on FCz/CPz (motor cortex) |
| Accuracy <60% | Review data quality | Deliver failure mode analysis as bonus |
| MuJoCo fails on macOS | `mjpython` wrapper | Demo without sim: dashboard + predictions only |
| Not enough time for CNN | Sklearn baseline is sufficient | v0.3.0 CNN is optional if v0.2.0 >70% |
| Fatigue (18h straight) | Independent parallel tasks | Each version is a standalone deliverable |

---

## Sources and References

- [Dataset KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control)
- [Repo brain-robot-interface](https://github.com/Nabla7/brain-robot-interface)
- [MNE-Python CSP Motor Imagery](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [EEG BCI real-time robotic hand control (Nature 2025)](https://www.nature.com/articles/s41467-025-61064-x)
- [BCI with AI copilots (Nature Machine Intelligence)](https://www.nature.com/articles/s42256-025-01090-y)
- [EEG-PyTorch-BCI pipeline](https://github.com/berdakh/eeg-pytorch-bci)
- [AI Agent Hackathon Trends 2026](https://semgrep.dev/blog/2025/what-a-hackathon-reveals-about-ai-agent-trends-to-expect-2026/)
- [EEG Signal Processing Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385593/)
- [Non-Invasive BCI Frontiers 2026](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1795349/abstract)
