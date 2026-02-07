# ThoughtLink — API Quick Reference

Quick reference for all public interfaces. Use this to know what functions and classes are available without reading the full source.

---

## Data Loading

```python
from thoughtlink.data.loader import (
    download_dataset,       # () -> Path                  Download from HuggingFace
    load_sample,            # (path) -> dict              Load single .npz file
    load_all,               # (data_dir?) -> list[dict]   Load all samples
    get_class_distribution, # (samples) -> dict           Count per class
    get_subject_distribution, # (samples) -> dict         Count per subject
    CLASS_NAMES,            # ["Right Fist", "Left Fist", "Both Fists", "Tongue Tapping", "Relax"]
    EEG_SFREQ,             # 500.0
    NIRS_SFREQ,            # 4.76
)

from thoughtlink.data.splitter import (
    split_by_subject,       # (samples, test_size=0.2) -> (train, test)
)
```

**Sample dict keys:** `eeg`, `nirs`, `label`, `subject_id`, `session_id`, `duration`, `file_path`

---

## Preprocessing

```python
from thoughtlink.preprocessing.eeg import (
    preprocess_eeg,     # (eeg_data, sfreq=500) -> ndarray(n_samples, 6)
    preprocess_sample,  # (sample_dict) -> sample_dict (in-place)
    preprocess_all,     # (samples) -> samples (in-place, prints progress)
    create_raw,         # (eeg_data) -> mne.io.RawArray
)

from thoughtlink.preprocessing.nirs import (
    preprocess_nirs,    # (nirs_data, n_pca=20) -> ndarray(n_timepoints, 20)
)

from thoughtlink.preprocessing.windowing import (
    extract_eeg_windows,   # (eeg, duration, ...) -> ndarray(n_windows, 500, 6)
    extract_nirs_stimulus, # (nirs, duration, ...) -> ndarray
    windows_from_samples,  # (samples, ...) -> (X, y, subject_ids)
)
```

---

## Feature Extraction

```python
from thoughtlink.features.eeg_features import (
    compute_band_powers,          # (window) -> ndarray(24,)    4 bands × 6 ch
    compute_hjorth,               # (window) -> ndarray(18,)    3 params × 6 ch
    compute_time_domain,          # (window) -> ndarray(24,)    4 stats × 6 ch
    extract_window_features,      # (window) -> ndarray(42,)    band_power + hjorth
    extract_features_from_windows, # (windows) -> ndarray(n, 42)  batch extraction
)

from thoughtlink.features.nirs_features import (
    extract_nirs_features,  # (nirs_data) -> ndarray(20,)  PCA-reduced
)

from thoughtlink.features.fusion import (
    fuse_features,  # (eeg_feat, nirs_feat?) -> ndarray(~62,)  concatenation
)
```

---

## Models

All models follow the sklearn interface: `fit(X, y)`, `predict(X)`, `predict_proba(X)`.

```python
from thoughtlink.models.baseline import (
    create_baselines,  # () -> dict[str, Pipeline]    4 sklearn pipelines
    evaluate_model,    # (model, X, y, class_names) -> dict with accuracy, kappa, etc.
)

from thoughtlink.models.hierarchical import (
    HierarchicalClassifier,  # sklearn-compatible 2-stage classifier
    RELAX_IDX,               # 4 (index of Relax in CLASS_NAMES)
    ACTIVE_CLASSES,          # [0, 1, 2, 3]
    ACTIVE_CLASS_NAMES,      # ["Right Fist", "Left Fist", "Both Fists", "Tongue Tapping"]
)
# Usage:
#   model = HierarchicalClassifier(stage1_threshold=0.5)
#   model.fit(X_train, y_train)
#   probs = model.predict_proba(X_test)  # shape (n_samples, 5)

from thoughtlink.models.cnn import (
    EEGNet,  # PyTorch nn.Module
)
```

---

## Inference

```python
from thoughtlink.inference.decoder import (
    RealtimeDecoder,
)
# Usage:
#   decoder = RealtimeDecoder(model, feature_extractor, window_size_s=1.0, sfreq=500)
#   decoder.feed_samples(chunk)     # (n_new, 6)
#   probs, latency = decoder.predict()  # ndarray(5,) or None

from thoughtlink.inference.confidence import (
    IntentConfidenceFilter,   # threshold + hysteresis + debounce
    MajorityVotingSmoother,   # sliding window majority vote
    StabilityPipeline,        # combined: filter + smoother
)
# Usage:
#   pipeline = StabilityPipeline(confidence_threshold=0.6, ...)
#   action_name = pipeline.process(probs, CLASS_NAMES)  # -> "Right Fist"
```

---

## Bridge

```python
from thoughtlink.bridge.intent_to_action import (
    intent_to_action_name,    # ("Right Fist") -> "RIGHT"
    INTENT_TO_ACTION_NAME,    # full mapping dict
)

from thoughtlink.bridge.brain_policy import (
    BrainPolicy,   # Main orchestrator
    StepResult,     # Dataclass: timestamp_s, raw_intent, stable_intent, action, confidence, probs, latency_ms
    load_config,    # (path?) -> dict
)
# Usage:
#   policy = BrainPolicy(model, config, on_step=callback)
#   results = policy.run_on_file("data/sample.npz")    # list[StepResult]
#   results = policy.run_on_array(eeg_preprocessed)     # list[StepResult]
#   result  = policy.step(probs)                        # single StepResult
#   policy.reset()

from thoughtlink.bridge.orchestrator import (
    Orchestrator,           # Multi-robot dispatcher
    SimulatedController,    # Mock robot for testing
    DispatchResult,         # Dataclass: action, n_robots, n_success, n_failed, dispatch_ms, ...
    create_simulated_fleet, # (n_robots, fail_rate=0) -> Orchestrator
)
# Usage:
#   orch = create_simulated_fleet(100)
#   result = orch.dispatch(step_result)   # DispatchResult or None (deduplicated)
#   orch.emergency_stop()
#   stats = orch.get_stats()
```

---

## Visualization

```python
from thoughtlink.viz.temporal_stability import (
    plot_action_timeline,      # (results, ground_truth?, ax?) -> Figure
    plot_confidence_trace,     # (results, threshold?, ax?) -> Figure
    plot_probability_heatmap,  # (results, class_names?, ax?) -> Figure
    plot_full_report,          # (results, ground_truth?, save_path?) -> Figure  # 3-panel
)
```

Dashboard (run via CLI):
```bash
uv run streamlit run src/thoughtlink/viz/dashboard.py
```

---

## Configuration

```python
from thoughtlink.bridge.brain_policy import load_config

config = load_config("configs/default.yaml")
# config["preprocessing"]["eeg"]["sfreq"]           -> 500.0
# config["preprocessing"]["window_duration_s"]       -> 1.0
# config["inference"]["confidence_threshold"]        -> 0.6
# config["models"]["hierarchical"]["stage1_threshold"] -> 0.5
# config["bridge"]["intent_to_action"]               -> {"Right Fist": "RIGHT", ...}
```

---

## Common Workflows

### Train and evaluate
```python
from thoughtlink.data.loader import load_all
from thoughtlink.data.splitter import split_by_subject
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.models.hierarchical import HierarchicalClassifier

samples = load_all()
train, test = split_by_subject(samples)
preprocess_all(train); preprocess_all(test)
X_train_w, y_train, _ = windows_from_samples(train)
X_test_w, y_test, _ = windows_from_samples(test)
X_train = extract_features_from_windows(X_train_w)
X_test = extract_features_from_windows(X_test_w)

model = HierarchicalClassifier()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)
```

### Run demo on a file
```python
from thoughtlink.bridge.brain_policy import BrainPolicy, load_config

config = load_config()
policy = BrainPolicy(model=trained_model, config=config)
results = policy.run_on_file("data/raw/sample.npz")
for r in results:
    print(f"{r.timestamp_s:.1f}s  {r.action}  conf={r.confidence:.2f}")
```

### Dispatch to robot fleet
```python
from thoughtlink.bridge.orchestrator import create_simulated_fleet

orch = create_simulated_fleet(n_robots=50)
for step in results:
    dispatch = orch.dispatch(step)
    if dispatch:
        print(f"{dispatch.action} -> {dispatch.n_success}/{dispatch.n_robots} robots")
```
